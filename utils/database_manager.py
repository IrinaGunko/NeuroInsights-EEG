import duckdb

class DatabaseManager:
    def __init__(self, db_path='neuroinsights.db'):
        self.db_path = db_path

    def __enter__(self):
        self.connection = duckdb.connect(self.db_path, read_only=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            self.connection.close()

    def initialize_schema(self, schema_definitions):
        for schema in schema_definitions:
            self.connection.execute(schema)

    def execute_query(self, query, params=None):
        return self.connection.execute(query, params or []).fetchall()

    def execute_batch(self, query, params_list):
        try:
            for params in params_list:
                self.connection.execute(query, params)
            print(f"Batch operation completed for {len(params_list)} records.")
        except Exception as e:
            print(f"Batch operation failed: {e}")
            raise
