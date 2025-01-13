from utils.logger_manager import LoggerManager

class SessionsRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def add_session(self, participant_id, session_number, recording_year, recording_duration, 
                    late_trigger_count, is_followup, recording_filename, eyes_state, cognitive_load_status):
        query = '''
        INSERT INTO sessions (
            participant_id, session_number, recording_year, recording_duration, 
            late_trigger_count, is_followup, recording_filename, eyes_state, cognitive_load_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        params = (
            participant_id, session_number, recording_year, recording_duration,
            late_trigger_count, is_followup, recording_filename, eyes_state, cognitive_load_status
        )
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Added session for participant ID: %d", participant_id)
        except Exception as e:
            self.logger.error("Failed to add session: %s", e)
            raise

    def get_all_sessions(self):
        query = "SELECT * FROM sessions;"
        try:
            result = self.db_manager.execute_query(query)
            self.logger.info("Retrieved all sessions")
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve sessions: %s", e)
            raise

    def get_sessions_by_participant_id(self, participant_id):
        query = "SELECT * FROM sessions WHERE participant_id = ?;"
        try:
            result = self.db_manager.execute_query(query, (participant_id,))
            self.logger.info("Retrieved sessions for participant ID: %d", participant_id)
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve sessions by participant ID: %s", e)
            raise

    def delete_session(self, session_id):
        query = "DELETE FROM sessions WHERE session_id = ?;"
        try:
            self.db_manager.execute_query(query, (session_id,))
            self.logger.info("Deleted session with ID: %d", session_id)
        except Exception as e:
            self.logger.error("Failed to delete session: %s", e)
            raise

    def update_session(self, session_id, **kwargs):
        update_fields = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE sessions SET {update_fields} WHERE session_id = ?;"
        params = list(kwargs.values()) + [session_id]
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Updated session with ID: %d", session_id)
        except Exception as e:
            self.logger.error("Failed to update session: %s", e)
            raise
