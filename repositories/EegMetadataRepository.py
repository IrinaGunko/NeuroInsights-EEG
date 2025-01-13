from utils.logger_manager import LoggerManager

class EegMetadataRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def add_metadata(self, task_name, institution_name, institution_address, institutional_department,
manufacturer, manufacturer_model_name, cap_manufacturer, cap_model_name,
recording_type, eeg_placement_scheme, eeg_reference, sampling_frequency,
software_filters, eeg_channel_count, eog_channel_count, power_line_frequency, eeg_ground):
        query = '''
        INSERT INTO eeg_metadata (
            task_name, institution_name, institution_address, institutional_department,
            manufacturer, manufacturer_model_name, cap_manufacturer, cap_model_name,
            recording_type, eeg_placement_scheme, eeg_reference, sampling_frequency,
            software_filters, eeg_channel_count, eog_channel_count, power_line_frequency, eeg_ground
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        params = (
            task_name, institution_name, institution_address, institutional_department,
            manufacturer, manufacturer_model_name, cap_manufacturer, cap_model_name,
            recording_type, eeg_placement_scheme, eeg_reference, sampling_frequency,
            software_filters, eeg_channel_count, eog_channel_count, power_line_frequency, eeg_ground
        )
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Added EEG metadata for task: %s", task_name)
        except Exception as e:
            self.logger.error("Failed to add EEG metadata: %s", e)
            raise

    def get_all_metadata(self):
        query = "SELECT * FROM eeg_metadata;"
        try:
            result = self.db_manager.execute_query(query)
            self.logger.info("Retrieved all EEG metadata")
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve EEG metadata: %s", e)
            raise

    def get_metadata_by_id(self, metadata_id):
        query = "SELECT * FROM eeg_metadata WHERE eeg_metadata_id = ?;"
        try:
            result = self.db_manager.execute_query(query, (metadata_id,))
            self.logger.info("Retrieved EEG metadata with ID: %d", metadata_id)
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve EEG metadata by ID: %s", e)
            raise

    def delete_metadata(self, metadata_id):
        query = "DELETE FROM eeg_metadata WHERE eeg_metadata_id = ?;"
        try:
            self.db_manager.execute_query(query, (metadata_id,))
            self.logger.info("Deleted EEG metadata with ID: %d", metadata_id)
        except Exception as e:
            self.logger.error("Failed to delete EEG metadata: %s", e)
            raise

    def update_metadata(self, metadata_id, **kwargs):
        update_fields = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE eeg_metadata SET {update_fields} WHERE eeg_metadata_id = ?;"
        params = list(kwargs.values()) + [metadata_id]
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Updated EEG metadata with ID: %d", metadata_id)
        except Exception as e:
            self.logger.error("Failed to update EEG metadata: %s", e)
            raise
