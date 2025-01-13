from utils.logger_manager import LoggerManager

class LateTriggerEventsRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def add_event(self, session_id, onset, duration, event_type):
        query = '''
        INSERT INTO late_trigger_events (session_id, onset, duration, type)
        VALUES (?, ?, ?, ?);
        '''
        params = (session_id, onset, duration, event_type)
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Added late trigger event for session ID: %d", session_id)
        except Exception as e:
            self.logger.error("Failed to add late trigger event: %s", e)
            raise

    def get_all_events(self):
        query = "SELECT * FROM late_trigger_events;"
        try:
            result = self.db_manager.execute_query(query)
            self.logger.info("Retrieved all late trigger events")
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve late trigger events: %s", e)
            raise

    def get_events_by_session_id(self, session_id):
        query = "SELECT * FROM late_trigger_events WHERE session_id = ?;"
        try:
            result = self.db_manager.execute_query(query, (session_id,))
            self.logger.info("Retrieved late trigger events for session ID: %d", session_id)
            return result
        except Exception as e:
            self.logger.error("Failed to retrieve events by session ID: %s", e)
            raise

    def delete_event(self, event_id):
        query = "DELETE FROM late_trigger_events WHERE event_id = ?;"
        try:
            self.db_manager.execute_query(query, (event_id,))
            self.logger.info("Deleted late trigger event with ID: %d", event_id)
        except Exception as e:
            self.logger.error("Failed to delete late trigger event: %s", e)
            raise

    def update_event(self, event_id, **kwargs):
        update_fields = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE late_trigger_events SET {update_fields} WHERE event_id = ?;"
        params = list(kwargs.values()) + [event_id]
        try:
            self.db_manager.execute_query(query, params)
            self.logger.info("Updated late trigger event with ID: %d", event_id)
        except Exception as e:
            self.logger.error("Failed to update late trigger event: %s", e)
            raise
