from utils.logger_manager import LoggerManager

class ParticipantsRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def add_participant(self, participant_name, gender, age, handedness, is_followup):
        query = '''
        INSERT INTO participants (participant_name, gender, age, handedness, is_followup)
        VALUES (?, ?, ?, ?, ?);
        '''
        params = (participant_name, gender, age, handedness, is_followup)
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Added participant: %s", participant_name)
        except Exception as e:
            self.logger.error("Failed to add participant: %s", e)
            raise

    def get_all_participants(self):
        query = "SELECT * FROM participants;"
        try:
            result = self.db_manager.execute_query(query)
            self.logger.info("Retrieved all participants")
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve participants: %s", e)
            raise

    def get_participant_by_id(self, participant_id):
        query = "SELECT * FROM participants WHERE participant_id = ?;"
        try:
            result = self.db_manager.execute_query(query, (participant_id,))
            self.logger.info("Retrieved participant with ID: %d", participant_id)
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve participant by ID: %s", e)
            raise

    def delete_participant(self, participant_id):
        query = "DELETE FROM participants WHERE participant_id = ?;"
        try:
            self.db_manager.execute_query(query, (participant_id,))
            self.logger.info("Deleted participant with ID: %d", participant_id)
        except Exception as e:
            self.logger.error("Failed to delete participant: %s", e)
            raise

    def update_participant(self, participant_id, **kwargs):
        update_fields = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE participants SET {update_fields} WHERE participant_id = ?;"
        params = list(kwargs.values()) + [participant_id]
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Updated participant with ID: %d", participant_id)
        except Exception as e:
            self.logger.error("Failed to update participant: %s", e)
            raise
