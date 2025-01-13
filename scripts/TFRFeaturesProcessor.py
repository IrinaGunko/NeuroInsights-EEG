import numpy as np
from mne.time_frequency import tfr_array_morlet, psd_array_welch
from feature_extraction.feature_extractor import (
    extract_psd_features,
    extract_tfr_features,
    compute_band_and_relative_power,
)
from utils.database_manager import DatabaseManager
from utils.logger_manager import LoggerManager
from repositories.ExtractedFeaturesRepository import ExtractedFeaturesRepository
from mne import io


class TFRFeaturesProcessor:
    def __init__(self, db_path="data/neuroinsights.db"):
        self.db_path = db_path
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        self.db_manager = DatabaseManager(db_path)
        self.feature_repo = ExtractedFeaturesRepository(self.db_manager)
        self.feature_repo.initialize_feature_tables()

        self.freqs = np.arange(1, 51, 1)  # 1 Hz resolution
        self.n_cycles = self.freqs / 2    # Proportional to frequencies

    def process_all_files(self):
        query_select = "SELECT session_id, recording_filename FROM sessions;"
        try:
            with self.db_manager as manager:
                rows = manager.connection.execute(query_select).fetchall()

            for session_id, recording_filename in rows:
                self.logger.info(f"Processing file: {recording_filename}")

                raw = self._load_eeg_file(recording_filename)
                if raw is None:
                    continue

                tfr_features = extract_tfr_features(
                    raw, freqs=self.freqs, n_cycles=self.n_cycles, use_fft=True, decim=1, output="power"
                )

                psd_features = extract_psd_features(raw, fmin=1, fmax=99, n_fft=1024, n_overlap=512)

                power_features = compute_band_and_relative_power(raw)

                merged_features = self._merge_features(
                    session_id, recording_filename, tfr_features, psd_features, power_features
                )

                self.feature_repo.add_tfr_features(merged_features)

                self.logger.info(f"Inserted TFR and PSD features for file: {recording_filename}")

            self._log_sample_features()

        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            raise

    @staticmethod
    def _load_eeg_file(filename):
        try:
            raw = io.read_raw_edf(filename, preload=True)
            raw.pick_types(eeg=True)  # Keep only EEG channels
            return raw
        except Exception as e:
            LoggerManager.get_logger("TFRFeaturesProcessor").error(
                f"Failed to load EEG file {filename}: {e}"
            )
            return None

    @staticmethod
    def _merge_features(session_id, recording_filename, tfr, psd, power):
        feature_map = {}

        for tfr_feat in tfr:
            key = (tfr_feat["Channel"], tfr_feat["Band"])
            feature_map[key] = tfr_feat
            feature_map[key]["session_id"] = session_id
            feature_map[key]["recording_filename"] = recording_filename

        for psd_feat in psd:
            key = (psd_feat["Channel"], psd_feat["Band"])
            if key in feature_map:
                feature_map[key].update(psd_feat)

        for power_feat in power:
            key = (power_feat["Channel"], power_feat["Band"])
            if key in feature_map:
                feature_map[key].update(power_feat)

        return list(feature_map.values())

    def _log_sample_features(self):
        query = "SELECT * FROM tfr_features LIMIT 10;"
        try:
            with self.db_manager as manager:
                results = manager.connection.execute(query).fetchall()
                self.logger.info("Sample rows from tfr_features:")
                for row in results:
                    self.logger.info(row)
        except Exception as e:
            self.logger.error(f"Failed to log sample features: {e}")


if __name__ == "__main__":
    processor = TFRFeaturesProcessor()
    processor.process_all_files()