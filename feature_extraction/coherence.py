import os
import numpy as np
import pandas as pd
from scipy.signal import coherence
from joblib import Parallel, delayed
from mne.io import read_raw_edf


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

def compute_coherence_for_pair(data1, data2, sfreq, band, nperseg=1024):
    fmin, fmax = band
    f, Cxy = coherence(data1, data2, fs=sfreq, nperseg=nperseg)
    freq_band = (f >= fmin) & (f < fmax)
    return np.mean(Cxy[freq_band])

def compute_coherence_matrix(data, sfreq, band, n_jobs=-1, nperseg=1024):
    n_channels = data.shape[0]
    pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

    coherence_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_coherence_for_pair)(data[i], data[j], sfreq, band, nperseg)
        for i, j in pairs
    )

    coherence_matrix = np.zeros((n_channels, n_channels))
    for idx, (i, j) in enumerate(pairs):
        coherence_matrix[i, j] = coherence_values[idx]
        coherence_matrix[j, i] = coherence_values[idx]  # Symmetric matrix

    return coherence_matrix, pairs

def process_file(filepath, band, n_jobs=-1, nperseg=1024):
    print(f"Processing file: {filepath}")

    raw = read_raw_edf(filepath, preload=True)
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    num_channels = min(64, data.shape[0])
    data = data[:num_channels]
    channel_names = raw.info['ch_names'][:num_channels]

    coherence_matrix, pairs = compute_coherence_matrix(data, sfreq, band, n_jobs=n_jobs, nperseg=nperseg)

    results = []
    for idx, (i, j) in enumerate(pairs):
        results.append({
            "Filename": os.path.basename(filepath),
            "Channel1": channel_names[i],
            "Channel2": channel_names[j],
            "Coherence": coherence_matrix[i, j]
        })

    return results

def main():
    input_dir = "./eeg_raw_ica_subset"
    output_file = "./feature_extraction/results/all_coherence_results.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    band = BANDS["alpha"]
    n_jobs = -1
    nperseg = 1024

    all_results = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".edf"):
            filepath = os.path.join(input_dir, file)
            results = process_file(filepath, band, n_jobs=n_jobs, nperseg=nperseg)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Saved all coherence results to {output_file}")

if __name__ == "__main__":
    main()
