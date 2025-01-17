import os
import pandas as pd
from mne.io import read_raw_edf
from utils.results_saver import save_to_csv
from mne.time_frequency import Spectrum

def compute_spectrum(raw, fmin=1, fmax=99, method="welch", n_jobs=-1):

    spectrum_data = raw.compute_psd(
        fmin=fmin,
        fmax=fmax,
        method=method,
        n_jobs=n_jobs
    )
    return spectrum_data

def extract_spectrum_features(spectrum_data, filename):

    df = spectrum_data.to_data_frame()
    df["Filename"] = filename
    return df

def process_file(file, input_dir, output_file):
    filepath = os.path.join(input_dir, file)
    print(f"Processing file: {file}")

    raw = read_raw_edf(filepath, preload=True)
    if raw is None:
        print(f"Skipping file {file} due to loading error.")
        return

    spectrum_data = compute_spectrum(raw)

    spectrum_features = extract_spectrum_features(spectrum_data, file)

    save_to_csv(spectrum_features, output_file)
    print(f"Saved Spectrum features for {file} to {output_file}")

def main():

    input_dir = "./eeg_raw_ica_subset"
    output_file = "./feature_extraction/results/spectrum_features.csv"


    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".edf"):
            process_file(file, input_dir, output_file)

if __name__ == "__main__":
    main()
