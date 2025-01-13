import pandas as pd
import json
import os

def save_to_csv(dataframe, filepath):

    if os.path.exists(filepath):
        dataframe.to_csv(filepath, mode='a', header=False, index=False)
    else:
        dataframe.to_csv(filepath, index=False)
    print(f"Saved data to CSV: {filepath}")

def save_to_json(data, filepath):
    try:
        with open(filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Saved data to JSON: {filepath}")
    except Exception as e:
        print(f"Error saving to JSON {filepath}: {e}")
