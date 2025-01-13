from feature_extraction.feature_extractor import (
    extract_temporal_frequency_features,
    extract_statistical_features,
    compute_channel_basic_features,
)
from utils.database_manager import DatabaseManager
from utils.logger_manager import LoggerManager
from repositories.ExtractedFeaturesRepository import ExtractedFeaturesRepository
from mne import io


class StatisticalFeaturesProcessor:
    def __init__(self, db_path="data/neuroinsights.db"):
        self.db_path = db_path
        self.logger = LoggerManager.get_logger(self.__class__.__name__)
        self.db_manager = DatabaseManager(db_path)
        self.feature_repo = ExtractedFeaturesRepository(self.db_manager)

        self.feature_repo.initialize_feature_tables()

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

                temporal_features = extract_temporal_frequency_features(raw)
                statistical_features = extract_statistical_features(raw)
                hjorth_features = compute_channel_basic_features(raw)

                merged_features = self._merge_features(
                    session_id, recording_filename, temporal_features, statistical_features, hjorth_features
                )

                self.feature_repo.add_statistical_features(merged_features)

                self.logger.info(f"Inserted features for file: {recording_filename}")

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
            LoggerManager.get_logger("StatisticalFeaturesProcessor").error(
                f"Failed to load EEG file {filename}: {e}"
            )
            return None

    @staticmethod
    def _merge_features(session_id, recording_filename, temporal, statistical, hjorth):
        feature_map = {f["Channel"]: f for f in statistical}

        for temp_feat in temporal:
            if temp_feat["Channel"] in feature_map:
                feature_map[temp_feat["Channel"]].update(temp_feat)

        for hjorth_feat in hjorth:
            if hjorth_feat["Channel"] in feature_map:
                feature_map[hjorth_feat["Channel"]].update(hjorth_feat)

        for feature in feature_map.values():
            feature["session_id"] = session_id
            feature["recording_filename"] = recording_filename

        return list(feature_map.values())

    def _log_sample_features(self):
        query = "SELECT * FROM statistical_features LIMIT 10;"
        try:
            with self.db_manager as manager:
                results = manager.connection.execute(query).fetchall()
                self.logger.info("Sample rows from statistical_features:")
                for row in results:
                    self.logger.info(row)
        except Exception as e:
            self.logger.error(f"Failed to log sample features: {e}")


if __name__ == "__main__":
    processor = StatisticalFeaturesProcessor()
    processor.process_all_files()
