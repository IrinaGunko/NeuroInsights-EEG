import concurrent.futures
from pathlib import Path
from preprocessing import preprocess_file
from utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger("ParallelPipeline")


def process_single_file(file, output_dir, preprocessing_steps):
    try:
        logger.info(f"Starting pipeline for {file}")
        results = preprocess_file(file, output_dir, preprocessing_steps)
        if results.get("error"):
            logger.error(f"Processing failed for {file}: {results['error']}")
        else:
            intermediate_file = results.get("intermediate_file")
            ica_files = results.get("ica_files", [])
            logger.info(f"Intermediate file saved: {intermediate_file}")
            logger.info(f"ICA files generated: {ica_files}")
        logger.info(f"Pipeline completed for {file}")
    except Exception as e:
        logger.error(f"Error processing file {file}: {e}", exc_info=True)


def run_parallel_pipeline(input_dir, output_dir, preprocessing_steps, max_files=4, max_workers=None):
    file_list = list(Path(input_dir).glob("*.edf"))[:max_files]
    if not file_list:
        logger.error(f"No files found in {input_dir}")
        return

    logger.info(f"Found {len(file_list)} files to process.")
    logger.info(f"Using up to {max_workers if max_workers else 'auto-detected'} workers for parallel processing.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, file, output_dir, preprocessing_steps): file for file in file_list}

        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                future.result()
                logger.info(f"Successfully processed file: {file}")
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")

    logger.info("Parallel pipeline run completed.")


if __name__ == "__main__":
    input_dir = "eeg_raw"
    output_dir = "eeg_raw_ica"
    preprocessing_steps = [
        {"method": "downsample", "params": {"target_sfreq": 250}},
        {"method": "apply_notch_filter", "params": {"freqs": (50, 100)}},
        {"method": "apply_bandpass_filter", "params": {"l_freq": 1, "h_freq": 99}},
        {"method": "apply_ica", "params": {"method": "fastica", "n_components": 40}},
#       {"method": "apply_ica", "params": {"method": "picard", "n_components": 20}},
#       {"method": "apply_ica", "params": {"method": "infomax", "n_components": 20}}
    ]
    max_files = 3265
    max_workers = 6

    run_parallel_pipeline(input_dir, output_dir, preprocessing_steps, max_files=max_files, max_workers=max_workers)
