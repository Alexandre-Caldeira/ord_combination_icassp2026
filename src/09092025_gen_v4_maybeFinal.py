import numpy as np
import h5py
import yaml
import os
import time
import multiprocessing
import psutil
from tqdm import tqdm
from datetime import timedelta
import warnings

# ==============================================================================
# FULLY IMPLEMENTED SIMULATION CLASSES
# ==============================================================================
class DataLoader:
    """
    Generates simulated EEG time-series data using a dedicated random generator.
    """
    def __init__(self, fs, nfft, signal_frequencies):
        self.fs = fs
        self.nfft = nfft
        self.signal_frequencies = signal_frequencies
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def generate_simulation(self, snr_db, n_windows=30, signal_present=True):
        total_points = self.nfft * n_windows
        t = np.arange(total_points) / self.fs
        noise = self.rng.standard_normal(total_points)
        sigma_n = 2 / self.nfft
        snr_n = np.sqrt(sigma_n)
        noise = (noise - np.mean(noise)) / np.std(noise) * snr_n
        
        pure_signal = np.zeros(total_points)
        if signal_present:
            snr_linear = 10**(snr_db / 10.0)
            snr_s = np.sqrt(4 * sigma_n * snr_linear / self.nfft) 
            for freq in self.signal_frequencies:
                pure_signal += snr_s * np.sin(2 * np.pi * freq * t + self.rng.random() * 2 * np.pi)

        combined_signal = (pure_signal + noise)
        # Normalization is removed to be consistent with experimental data that is not normalized at this stage
        # combined_signal = combined_signal / np.std(combined_signal) 
        
        return combined_signal.reshape((self.nfft, n_windows), order='F')

class FeatureCalculator:
    """
    Calculates features for both signal and noise frequencies iteratively,
    matching the logic of the experimental data processing script.
    """
    def __init__(self, fs, nfft, n_windows, signal_freqs, noise_freqs, l_tfl=10, msc_max_windows=40):
        self.fs, self.nfft, self.n_windows = fs, nfft, n_windows
        self.all_freqs = signal_freqs + noise_freqs
        self.l_tfl = l_tfl
        self.msc_max_windows = msc_max_windows

    def generate_features(self, signal_data, noise_ref_data):
        """
        Generates all 7 features iteratively, window by window, for all target frequencies.
        The final output shape is (n_windows, n_freqs, n_features).
        """
        fft_all_windows = np.fft.fft(signal_data, axis=0).T # Transpose to (windows, nfft)
        fft_all_noise_ref = np.fft.fft(noise_ref_data, axis=0).T

        all_final_metrics = []

        for k in range(self.n_windows): # k is the index of the current window
            # MODIFICATION 1: Mean/Std Dev over frequencies of the CURRENT window
            fft_mags_current_window_all_freqs = np.abs(fft_all_windows[k, :self.nfft // 2])
            mean_mag = np.mean(fft_mags_current_window_all_freqs)
            std_mag = np.std(fft_mags_current_window_all_freqs)

            # MODIFICATION 2: Capped/Sliding window logic for cumulative metrics
            num_windows_so_far = k + 1
            if num_windows_so_far <= 1:
                M = 0; start_idx = 0; end_idx = 0
            elif num_windows_so_far <= self.msc_max_windows:
                M = num_windows_so_far; start_idx = 0; end_idx = k + 1
            else:
                M = self.msc_max_windows; end_idx = k + 1; start_idx = end_idx - M

            calc_ffts = fft_all_windows[start_idx:end_idx, :]
            calc_noise_ffts = fft_all_noise_ref[start_idx:end_idx, :]
            
            metrics_for_this_step = []
            for f in self.all_freqs:
                target_bin = int(f * self.nfft / self.fs)
                
                if M > 0:
                    fft_at_bin = calc_ffts[:, target_bin]
                    fft_noise_at_bin = calc_noise_ffts[:, target_bin]
                    
                    num_msc = np.abs(np.sum(fft_at_bin))**2
                    den_msc = M * np.sum(np.abs(fft_at_bin)**2)
                    msc = num_msc / den_msc if den_msc > 1e-9 else 0.0

                    angles = np.angle(fft_at_bin)
                    csm = np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2
                    power_signal_tfg = np.sum(np.abs(fft_at_bin)**2)
                    power_noise_tfg = np.sum(np.abs(fft_noise_at_bin)**2)
                    tfg = power_signal_tfg / (power_signal_tfg + power_noise_tfg) if (power_signal_tfg + power_noise_tfg) > 1e-9 else 0.0
                    mag_freq = np.mean(np.abs(fft_at_bin))
                    phi_freq = np.angle(np.mean(fft_at_bin))
                else:
                    msc, csm, tfg, mag_freq, phi_freq = 0.0, 0.0, 0.0, 0.0, 0.0

                power_spectrum_current_win = np.abs(fft_all_windows[k, :])**2
                power_signal_tfl = power_spectrum_current_win[target_bin]
                side_lobe_indices = np.arange(target_bin - self.l_tfl, target_bin + self.l_tfl + 1)
                side_lobe_indices = side_lobe_indices[(side_lobe_indices >= 0) & (side_lobe_indices < self.nfft // 2) & (side_lobe_indices != target_bin)]
                power_noise_mean = np.mean(power_spectrum_current_win[side_lobe_indices]) if len(side_lobe_indices) > 0 else 0.0
                tfl = power_signal_tfl / power_noise_mean if power_noise_mean > 1e-9 else 0.0

                snr_meas = -10 * np.log10(mean_mag / std_mag) if std_mag > 1e-9 and mean_mag > 1e-9 else 0.0
                
                metrics_for_this_step.append([msc, csm, tfg, tfl, snr_meas, mag_freq, phi_freq])
            
            all_final_metrics.append(metrics_for_this_step)
            
        return np.array(all_final_metrics, dtype='float32')


# ==============================================================================
# HDF5 DATA STORAGE & INCREMENTAL SIMULATION 
# ==============================================================================

def initialize_hdf5_store(filepath, config):
    print(f"Initializing new HDF5 store at: {filepath}")
    p_conf = config['parameters']
    s_conf = config['simulation']
    num_total_freqs = len(p_conf['signal_freqs']) + len(p_conf['noise_freqs'])
    num_features = 7 
    n_windows = p_conf['n_windows']

    with h5py.File(filepath, 'w') as f:
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.attrs[f"{key}_{sub_key}"] = str(sub_value)
            else:
                f.attrs[key] = str(value)

        for snr_db in p_conf['snr_range']:
            grp = f.create_group(f"snr_{snr_db}")
            grp.create_dataset('metrics', 
                               shape=(0, n_windows, num_total_freqs, num_features), 
                               maxshape=(None, n_windows, num_total_freqs, num_features), 
                               dtype='float32', chunks=True)
            
            n_time_samples = p_conf['nfft'] * n_windows
            grp.create_dataset('time_domain', 
                               shape=(0, n_time_samples), 
                               maxshape=(s_conf['time_domain_saves'], n_time_samples), 
                               dtype='float32', chunks=True, compression="gzip")

def sanity_check_config(filepath, config):
    if not os.path.exists(filepath): return True
    with h5py.File(filepath, 'r') as f:
        print("Performing configuration sanity check...")
        for key, value in config['parameters'].items():
            attr_key = f"parameters_{key}"
            if attr_key in f.attrs:
                if isinstance(value, list):
                    if str(f.attrs[attr_key]) != str(value): return False
                elif str(f.attrs[attr_key]) != str(value): return False
    print("Sanity check passed.")
    return True

def run_single_trial(args):
    """Function executed by each worker process."""
    snr_db, config, seed, save_time_domain = args
    p_conf = config['parameters']
    
    dtl = DataLoader(p_conf['fs'], p_conf['nfft'], p_conf['signal_freqs'])
    dtl.set_seed(seed)
    
    feat_calc = FeatureCalculator(
        p_conf['fs'], p_conf['nfft'], p_conf['n_windows'], 
        p_conf['signal_freqs'], p_conf['noise_freqs']
    )

    signal_data = dtl.generate_simulation(snr_db, p_conf['n_windows'], signal_present=True)
    noise_ref_data = dtl.generate_simulation(-np.inf, p_conf['n_windows'], signal_present=False)
    
    # generate_features now returns shape (n_windows, n_freqs, n_features)
    metrics_array = feat_calc.generate_features(signal_data, noise_ref_data)
    
    time_domain_data = signal_data.flatten(order='F').astype('float32') if save_time_domain else None
    
    return {'metrics': metrics_array, 'time_domain': time_domain_data}

def run_incremental_simulation(config):
    s_conf = config['simulation']
    p_conf = config['parameters']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    filepath = os.path.join(project_root, s_conf['hdf5_filepath'])

    total_trials_goal = s_conf['total_trials_goal']
    batch_size = s_conf['batch_size']
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not sanity_check_config(filepath, config):
        exit("Exiting due to configuration mismatch.")
    if not os.path.exists(filepath):
        initialize_hdf5_store(filepath, config)
    
    simulation_plan = {}
    with h5py.File(filepath, 'r') as f:
        for snr in p_conf['snr_range']:
            group_name = f"snr_{snr}"
            completed_trials = f[group_name]['metrics'].shape[0]
            simulation_plan[snr] = completed_trials

    print("\n--- Current Simulation Status ---")
    for snr, trials in simulation_plan.items():
        print(f"  SNR {snr:5.1f} dB: {trials:6d} / {total_trials_goal} trials completed.")
    
    phases = [{"name": f"Completion to {total_trials_goal}", "goal": total_trials_goal}]
    seed_sequence = np.random.SeedSequence(config['master_seed'])
    num_cores = s_conf['cpu_cores'] if s_conf['cpu_cores'] > 0 else psutil.cpu_count(logical=False)
    print(f"\nInitializing multiprocessing pool with {num_cores} cores.")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for phase in phases:
            print(f"\n--- Starting {phase['name']} ---")
            for snr_db in p_conf['snr_range']:
                start_trial = simulation_plan[snr_db]
                if start_trial >= phase['goal']: continue
                
                trials_to_run_in_phase = phase['goal'] - start_trial
                snr_specific_seed_seq = seed_sequence.spawn(len(p_conf['snr_range']))[p_conf['snr_range'].index(snr_db)]
                trial_seeds = snr_specific_seed_seq.spawn(total_trials_goal)

                with tqdm(total=trials_to_run_in_phase, desc=f"SNR {snr_db:5.1f} dB") as pbar:
                    while start_trial < phase['goal']:
                        current_batch_size = min(batch_size, phase['goal'] - start_trial)
                        tasks = []
                        for i in range(current_batch_size):
                            trial_idx = start_trial + i
                            save_td = trial_idx < s_conf['time_domain_saves']
                            tasks.append((snr_db, config, trial_seeds[trial_idx], save_td))

                        results = [res for res in pool.imap_unordered(run_single_trial, tasks)]
                        pbar.update(len(results))

                        with h5py.File(filepath, 'a') as f:
                            grp = f[f"snr_{snr_db}"]
                            metrics_batch = np.array([r['metrics'] for r in results])
                            td_batch = [r['time_domain'] for r in results if r['time_domain'] is not None]
                            
                            dset = grp['metrics']
                            dset.resize(dset.shape[0] + len(metrics_batch), axis=0)
                            dset[-len(metrics_batch):] = metrics_batch

                            if td_batch:
                                dset_td = grp['time_domain']
                                if dset_td.shape[0] < s_conf['time_domain_saves']:
                                    new_size = min(s_conf['time_domain_saves'], dset_td.shape[0] + len(td_batch))
                                    dset_td.resize(new_size, axis=0)
                                    dset_td[-len(td_batch):] = np.array(td_batch)
                        
                        start_trial += current_batch_size
                        simulation_plan[snr_db] = start_trial

    print("\n===== ALL SIMULATION PHASES COMPLETED! =====\n")

if __name__ == '__main__':
    # This block allows running the script from the command line
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'config.yml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        run_incremental_simulation(config)
    except NameError:
        print("\nNOTE: Script is likely being run in an interactive environment (e.g., Jupyter).")
        print("Please define the 'config' dictionary manually or run as a standalone .py file.")