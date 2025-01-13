import os
import numpy as np
import pandas as pd
from mne.filter import filter_data
from mne.io import read_raw_edf

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

def compute_plv_for_pair(phase1, phase2):
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))

def compute_plv_matrix(data, sfreq, band):
    fmin, fmax = band
    band_data = filter_data(data, sfreq, fmin, fmax, verbose=False)
    phase_data = np.angle(np.exp(1j * np.angle(np.fft.fft(band_data, axis=1))))

    n_channels = data.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            plv_matrix[i, j] = compute_plv_for_pair(phase_data[i], phase_data[j])
            plv_matrix[j, i] = plv_matrix[i, j]

    return plv_matrix, [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

def process_file(filepath, band):
    print(f"Processing file: {filepath}")

    raw = read_raw_edf(filepath, preload=True)
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    num_channels = min(64, data.shape[0])
    data = data[:num_channels]
    channel_names = raw.info['ch_names'][:num_channels]

    plv_matrix, pairs = compute_plv_matrix(data, sfreq, band)

    results = []
    for idx, (i, j) in enumerate(pairs):
        results.append({
            "Filename": os.path.basename(filepath),
            "Channel1": channel_names[i],
            "Channel2": channel_names[j],
            "PLV": plv_matrix[i, j]
        })
    return results

def main():
    input_dir = "./eeg_raw_ica_subset"
    output_file = "./feature_extraction/results/all_plv_results.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    band = BANDS["alpha"]

    all_results = []

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".edf"):
            filepath = os.path.join(input_dir, file)
            results = process_file(filepath, band)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Saved all PLV results to {output_file}")

if __name__ == "__main__":
    main()
