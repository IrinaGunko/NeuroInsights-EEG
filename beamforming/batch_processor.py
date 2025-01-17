import os
import argparse
from pathlib import Path
from beamforming.beamformer import Beamformer
from utils.logger_manager import LoggerManager
from utils.results_saver import save_to_npy, save_to_hdf5
import mne
from joblib import Parallel, delayed
import csv

logger = LoggerManager.get_logger("BatchProcessor")

def save_results(data, filepath, save_as_hdf5=False, metadata=None):
    if save_as_hdf5:
        save_to_hdf5(data, filepath.with_suffix(".hdf5"), metadata)
    else:
        save_to_npy(data, filepath.with_suffix(".npy"))

def process_single_file(edf_file, output_dir, subjects_dir, save_as_hdf5=False):
    try:
        logger.info(f"Processing file: {edf_file.name}")
        beamformer = Beamformer(subjects_dir)
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        fwd = beamformer.create_forward_model(raw)
        logger.info("Dynamically calculating regularization parameter...")
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)
        print(noise_cov.data.mean())
        reg = 1e-5 if noise_cov.data.mean() < 1e-4 else 0.05
        stc = beamformer.apply_beamformer(raw, fwd, cov = noise_cov, reg=reg)
        stc_atlas = beamformer.map_to_atlas(stc, fwd)
        output_path = Path(output_dir) / f"{edf_file.stem}_stc_atlas"
        metadata = {"subject": beamformer.subject, "atlas": beamformer.atlas_name}
        save_results(stc_atlas, output_path, save_as_hdf5=save_as_hdf5, metadata=metadata)
        logger.info(f"Results saved at {output_path}")
    except Exception as e:
        logger.error(f"Error processing file {edf_file.name}: {e}", exc_info=True)

def process_files_from_csv(csv_file, output_dir, subjects_dir, use_parallel=False, n_jobs=4, save_as_hdf5=False):
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            edf_files = [Path(row[0]) for row in list(reader)[:10]]
        if use_parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(process_single_file)(edf_file, output_dir, subjects_dir, save_as_hdf5)
                for edf_file in edf_files
            )
        else:
            for edf_file in edf_files:
                process_single_file(edf_file, output_dir, subjects_dir, save_as_hdf5)
        logger.info("Processing completed.")
    except Exception as e:
        logger.error("Batch processing failed.", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG files using beamforming.")
    parser.add_argument("--batch", action="store_true", help="Process all files in the batch folder.")
    parser.add_argument("--singlefile", action="store_true", help="Process a single .edf file.")
    parser.add_argument("--hdf5", action="store_true", help="Save results in HDF5 format.")
    args = parser.parse_args()
    CSV_FILE = "followup_recordings.csv"
    OUTPUT_DIR = "beamforming_results"
    SINGLE_FILE = "eeg_raw_ica_subset/sub-001_ses-1_task-EyesClosed_acq-post_eeg_fastica_processed.edf"
    SUBJECTS_DIR = "subjects_dir"
    if args.singlefile:
        process_single_file(Path(SINGLE_FILE), OUTPUT_DIR, SUBJECTS_DIR, save_as_hdf5=args.hdf5)
    elif args.batch:
        process_files_from_csv(CSV_FILE, OUTPUT_DIR, SUBJECTS_DIR, save_as_hdf5=args.hdf5)
    else:
        logger.error("Specify either --batch or --singlefile.")
