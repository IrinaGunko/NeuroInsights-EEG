import os
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from utils.results_saver import save_to_csv
from mne.time_frequency import psd_array_welch

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
    "gamma_high": (50, 99),
}

def compute_total_power(psds):
    total_power = psds.sum(axis=1)
    return total_power

def compute_psd_welch(raw, fmin=1, fmax=99, n_fft=1024, n_overlap=512, n_per_seg=None, n_jobs=-1):
    data = raw.get_data()
    sfreq = raw.info.get("sfreq", 250)
    psds, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        n_jobs=n_jobs,
        average="mean",
    )
    return psds, freqs

def extract_psd_features(psds, freqs, total_power, raw, filename):
    features = []
    for band, (fmin, fmax) in BANDS.items():
        valid_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        band_power = psds[:, valid_indices].mean(axis=1)
        for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
            normalized_power = band_power[ch_idx] / total_power[ch_idx] if total_power[ch_idx] != 0 else 0
            features.append({
                "Filename": filename,
                "Channel": ch_name,
                "Band": band,
                "PowerPSDWelch": band_power[ch_idx],
                "PowerPSDWelchNormalized": normalized_power,
            })
    return pd.DataFrame(features)

def process_file(file, input_dir, output_file):
    filepath = os.path.join(input_dir, file)
    print(f"Processing file: {file}")

    raw = read_raw_edf(filepath, preload=True)
    if raw is None:
        print(f"Skipping file {file} due to loading error.")
        return

    psds, freqs = compute_psd_welch(raw)
    total_power = compute_total_power(psds)

    psd_features = extract_psd_features(psds, freqs, total_power, raw, file)

    save_to_csv(psd_features, output_file)
    print(f"Saved Welch features for {file} to {output_file}")

def main():
    input_dir = "./eeg_raw_ica"
    output_file = "./feature_extraction/results/psd_features_welch.csv"

    for file in sorted(os.listdir(input_dir)):  # Ensure files are processed in order
        if file.endswith(".edf"):
            process_file(file, input_dir, output_file)

if __name__ == "__main__":
    main()
