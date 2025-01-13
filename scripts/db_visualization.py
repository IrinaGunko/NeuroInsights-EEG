import streamlit as st
import duckdb
import pandas as pd
from graphviz import Digraph

DB_PATH = "data/neuroinsights.db"

def fetch_data(query):
    conn = duckdb.connect(database=DB_PATH, read_only=True)
    try:
        data = pd.read_sql(query, conn)
    finally:
        conn.close()
    return data

def get_table_names():
    query = "SHOW TABLES;"
    return fetch_data(query)["name"].tolist()

def get_table_schema(table_name):
    query = f"PRAGMA table_info('{table_name}');"
    return fetch_data(query)

def generate_schema_diagram(tables, relationships):
    dot = Digraph(comment="DuckDB Schema")
    for table in tables:
        schema = get_table_schema(table)
        if not schema.empty:
            columns = "\n".join(schema["name"].tolist())
            dot.node(table, f"{table}\n{columns}")

    for rel in relationships:
        dot.edge(rel["from_table"], rel["to_table"], label=rel["label"])

    return dot

relationships = [
    {"from_table": "sessions", "to_table": "participants", "label": "participant_id"},
    {"from_table": "late_trigger_events", "to_table": "sessions", "label": "session_id"},
    {"from_table": "statistical_features", "to_table": "sessions", "label": "session_id"},
    {"from_table": "tfr_features", "to_table": "sessions", "label": "session_id"},
    {"from_table": "connectivity_features", "to_table": "sessions", "label": "session_id"},
]

st.title("NeuroInsights Data Viewer")

st.sidebar.header("Database Explorer")
table_names = get_table_names()

if table_names:
    selected_table = st.sidebar.selectbox("Select a table:", table_names)
else:
    st.error("No tables found in the database.")

if selected_table:
    st.subheader(f"Schema of Table: {selected_table}")
    schema = get_table_schema(selected_table)
    if not schema.empty:
        st.dataframe(schema)
    else:
        st.warning("Schema not found for the selected table.")

    st.subheader(f"Sample Data from {selected_table}")
    query = f"SELECT * FROM {selected_table} LIMIT 5;"
    data = fetch_data(query)
    if not data.empty:
        formatted_data = data.style.format(precision=15, na_rep="NaN")
        st.dataframe(formatted_data)
    else:
        st.warning(f"No data found in table {selected_table}.")

    st.subheader(f"Filter Data in {selected_table}")
    filter_column = st.selectbox(
        "Select a column to filter:",
        schema["name"].tolist() if not schema.empty else []
    )
    filter_value = st.text_input(f"Enter value to search in '{filter_column}':")

    if filter_value:
        filtered_query = f"""
        SELECT * FROM {selected_table}
        WHERE LOWER(CAST({filter_column} AS TEXT)) LIKE LOWER('%{filter_value}%')
        """
        filtered_data = fetch_data(filtered_query)
        if not filtered_data.empty:
            formatted_filtered_data = filtered_data.style.format(precision=15, na_rep="NaN")
            st.dataframe(formatted_filtered_data)
        else:
            st.warning(f"No records found in {selected_table} where {filter_column} contains '{filter_value}'.")

st.sidebar.subheader("Schema Diagram")
if st.sidebar.button("Generate Schema Diagram"):
    diagram = generate_schema_diagram(table_names, relationships)
    st.graphviz_chart(diagram.source)

st.sidebar.info("Relationships are manually defined in the app.")
