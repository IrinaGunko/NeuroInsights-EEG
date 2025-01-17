import pandas as pd
import json
import os
import numpy as np
import h5py

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

def save_to_npy(data, filepath):
    try:
        np.save(filepath, data)
        print(f"Saved data to NPY: {filepath}")
    except Exception as e:
        print(f"Error saving to NPY {filepath}: {e}")

def save_to_hdf5(data, filepath, metadata=None):
    try:
        with h5py.File(filepath, "w") as hdf5_file:
            hdf5_file.create_dataset("data", data=data)
            if metadata:
                for key, value in metadata.items():
                    hdf5_file.attrs[key] = value
        print(f"Saved data to HDF5: {filepath}")
    except Exception as e:
        print(f"Error saving to HDF5 {filepath}: {e}")
