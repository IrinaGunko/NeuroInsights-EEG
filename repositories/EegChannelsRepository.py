from utils.logger_manager import LoggerManager

class EegChannelsRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def add_channel(self, channel_name, channel_type, units, low_cutoff, high_cutoff):
        query = '''
        INSERT INTO eeg_channels (
            channel_name, channel_type, units, low_cutoff, high_cutoff
        ) VALUES (?, ?, ?, ?, ?);
        '''
        params = (channel_name, channel_type, units, low_cutoff, high_cutoff)
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Added EEG channel: %s", channel_name)
        except Exception as e:
            self.logger.error("Failed to add EEG channel: %s", e)
            raise

    def get_all_channels(self):
        query = "SELECT * FROM eeg_channels;"
        try:
            result = self.db_manager.execute_query(query)
            self.logger.info("Retrieved all EEG channels")
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve EEG channels: %s", e)
            raise

    def delete_channel(self, channel_id):
        query = "DELETE FROM eeg_channels WHERE channel_id = ?;"
        try:
            self.db_manager.execute_query(query, (channel_id,))
            self.logger.info("Deleted EEG channel with ID: %d", channel_id)
        except Exception as e:
            self.logger.error("Failed to delete EEG channel: %s", e)
            raise

    def update_channel(self, channel_id, **kwargs):
        update_fields = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE eeg_channels SET {update_fields} WHERE channel_id = ?;"
        params = list(kwargs.values()) + [channel_id]
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Updated EEG channel with ID: %d", channel_id)
        except Exception as e:
            self.logger.error("Failed to update EEG channel: %s", e)
            raise
