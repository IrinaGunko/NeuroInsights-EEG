import os
import numpy as np
import pandas as pd
from utils.results_saver import save_to_csv
from mne.io import read_raw_edf
from mne.time_frequency import tfr_array_morlet

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
    "gamma_high": (50, 99),
}

def extract_tfr_features(tfr_data, freqs, raw, filename):
    features = []
    time_avg_tfr = tfr_data.mean(axis=-1)

    for band, (fmin, fmax) in BANDS.items():
        valid_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if valid_indices.size == 0:
            print(f"Warning: No valid frequencies found for band '{band}' in file {filename}.")
            for ch_name in raw.info["ch_names"]:
                features.append({
                    "Filename": filename,
                    "Channel": ch_name,
                    "Band": band,
                    "PowerTFRMorlet": np.nan,
                })
            continue

        band_power = time_avg_tfr[:, valid_indices].mean(axis=1)
        for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
            features.append({
                "Filename": filename,
                "Channel": ch_name,
                "Band": band,
                "PowerTFRMorlet": band_power[ch_idx],
            })
    return pd.DataFrame(features)

def compute_tfr_morlet(raw, freqs, n_cycles, use_fft=True, decim=1, n_jobs=-1, output="power"):

    data = raw.get_data()[np.newaxis, :, :]
    sfreq = raw.info.get("sfreq", 250)
    tfr_data = tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=use_fft,
        decim=decim,
        output=output,
        n_jobs=n_jobs,
    )

    return tfr_data[0], freqs

def process_file(file, input_dir, output_file):

    filepath = os.path.join(input_dir, file)
    print(f"Processing file: {file}")

    raw = read_raw_edf(filepath, preload=True)
    if raw is None:
        print(f"Skipping file {file} due to loading error.")
        return

    freqs = np.arange(1, 100, 1)
    n_cycles = freqs / 2

    tfr_data, freqs = compute_tfr_morlet(raw, freqs=freqs, n_cycles=n_cycles)

    tfr_features = extract_tfr_features(tfr_data, freqs, raw, file)

    save_to_csv(tfr_features, output_file)
    print(f"Saved Morlet TFR features for {file} to {output_file}")

def main():
    input_dir = "./eeg_raw_ica"
    output_file = "./feature_extraction/results/tfr_features_morlet.csv"

    for file in sorted(os.listdir(input_dir)):  # Ensure files are processed in order
        if file.endswith(".edf"):
            process_file(file, input_dir, output_file)

if __name__ == "__main__":
    main()
