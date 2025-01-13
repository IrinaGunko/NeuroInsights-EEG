import streamlit as st
import os
import mne
import tempfile
import subprocess


def save_uploaded_file_to_tempfile(uploaded_file):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    return temp_file_path


def convert_edf_to_fif(edf_file_path, output_dir):
    try:
        raw = mne.io.read_raw_edf(edf_file_path, preload=True)
        base_name = os.path.basename(edf_file_path).replace(".edf", ".fif")
        fif_file_path = os.path.join(output_dir, base_name)
        raw.save(fif_file_path, overwrite=True)
        return fif_file_path
    except Exception as e:
        raise RuntimeError(f"Error converting file: {e}")


def render():
    st.title("EEG File Management")
    if "temp_file_path" not in st.session_state:
        st.session_state.temp_file_path = None
    if "raw_initial" not in st.session_state:
        st.session_state.raw_initial = None
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None

    action = st.radio(
        "Select an action:",
        options=["Upload EEG File", "Choose Preloaded File", "Convert EDF to FIF"],
    )

    if action == "Upload EEG File":
        uploaded_file = st.file_uploader("Upload EEG file (.edf or .fif):", type=["edf", "fif"])

        if uploaded_file:
            st.success(f"Uploaded file: {uploaded_file.name}")
            try:
                temp_file_path = save_uploaded_file_to_tempfile(uploaded_file)
                st.session_state.temp_file_path = temp_file_path
                st.session_state.current_file_name = uploaded_file.name  # Update file name

                if uploaded_file.name.endswith(".fif"):
                    raw = mne.io.read_raw_fif(temp_file_path, preload=True)
                elif uploaded_file.name.endswith(".edf"):
                    raw = mne.io.read_raw_edf(temp_file_path, preload=True)
                else:
                    st.error("Unsupported file format.")
                    return

                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage)

                st.session_state.raw_initial = raw
                st.success("EEG file loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")

    elif action == "Choose Preloaded File":
        eeg_raw_folder = "eeg_raw"  # Path to your folder with preloaded files
        preloaded_files = [
            f for f in os.listdir(eeg_raw_folder) if f.endswith((".edf", ".fif"))
        ]
        selected_file = st.selectbox("Select a preloaded file:", ["None"] + preloaded_files)

        if selected_file != "None":
            file_path = os.path.join(eeg_raw_folder, selected_file)
            st.session_state.temp_file_path = file_path
            st.session_state.current_file_name = selected_file  # Update file name
            try:
                if file_path.endswith(".fif"):
                    raw = mne.io.read_raw_fif(file_path, preload=True)
                elif file_path.endswith(".edf"):
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                else:
                    st.error("Unsupported file format.")
                    return

                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage)

                st.session_state.raw_initial = raw
                st.success("Preloaded file loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load preloaded file: {e}")

    elif action == "Convert EDF to FIF":
        uploaded_edf = st.file_uploader("Upload an EDF file to convert to FIF:", type=["edf"])

        if uploaded_edf:
            st.success(f"Uploaded file: {uploaded_edf.name}")
            try:
                temp_file_path = save_uploaded_file_to_tempfile(uploaded_edf)
                output_dir = "eeg_raw"
                converted_fif_path = convert_edf_to_fif(temp_file_path, output_dir)

                with open(converted_fif_path, "rb") as f:
                    fif_bytes = f.read()
                st.download_button(
                    label="Download Converted FIF File",
                    data=fif_bytes,
                    file_name=os.path.basename(converted_fif_path),
                    mime="application/octet-stream",
                )
                st.success(f"Converted file saved to: {converted_fif_path}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")
