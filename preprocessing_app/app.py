import streamlit as st
from pathlib import Path
import mne
from utils import preprocessing_steps, visualization
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="EEG Preprocessing App", layout="wide")

# Preloaded file options
PRELOADED_FILES_DIR = "data/preloaded"
PRELOADED_FILES = [f for f in Path(PRELOADED_FILES_DIR).glob("*.edf")]

# Tabs for the app
tabs = st.tabs(["File Selection", "Preprocessing", "Save & Download"])

# Global variables to store results
global_raw = None  # Holds the currently loaded MNE Raw object
preprocessing_results = {}

### File Selection Tab ###
with tabs[0]:
    st.header("Step 1: Select EEG File")
    # File uploader
    uploaded_file = st.file_uploader("Upload your .edf file", type=["edf"])
    # Preloaded files
    preloaded_file = st.selectbox("Or choose a preloaded file:", PRELOADED_FILES)
    # Load the selected file
    selected_file = uploaded_file if uploaded_file else preloaded_file
    if selected_file:
        try:
            global_raw = mne.io.read_raw_edf(selected_file, preload=True)
            st.success(f"Successfully loaded file: {selected_file}")
            st.write(global_raw)
        except Exception as e:
            st.error(f"Error loading file: {e}")

### Preprocessing Tab ###
with tabs[1]:
    st.header("Step 2: Choose Preprocessing Steps")

    # Ensure a file is loaded
    if global_raw is None:
        st.warning("Please select a file in the 'File Selection' tab.")
    else:
        # Preprocessing step selection
        downsample = st.checkbox("Downsampling")
        notch_filter = st.checkbox("Notch Filtering")
        bandpass_filter = st.checkbox("Band-pass Filtering")
        apply_ica = st.checkbox("Artifact Removal with ICA")

        # Dynamic parameter input for selected methods
        if downsample:
            target_freq = st.slider("Target Frequency (Hz)", 100, 500, 250)
        if notch_filter:
            freqs = st.multiselect("Select Notch Frequencies", [50, 100, 150], default=[50])
        if bandpass_filter:
            l_freq = st.slider("Low Frequency (Hz)", 0.1, 50.0, 1.0)
            h_freq = st.slider("High Frequency (Hz)", 50.0, 200.0, 99.0)
        if apply_ica:
            ica_method = st.selectbox("ICA Method", ["fastica", "picard", "infomax"])
            n_components = st.slider("Number of ICA Components", 10, 64, 40)

        # Button to apply preprocessing
        if st.button("Apply Preprocessing"):
            try:
                raw = global_raw.copy()
                preprocessing_results.clear()

                # Apply steps in sequence
                if downsample:
                    raw = preprocessing_steps.downsample(raw, target_sfreq=target_freq)
                    preprocessing_results["downsample"] = raw.copy()
                if notch_filter:
                    raw = preprocessing_steps.apply_notch_filter(raw, freqs=freqs)
                    preprocessing_results["notch_filter"] = raw.copy()
                if bandpass_filter:
                    raw = preprocessing_steps.apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
                    preprocessing_results["bandpass_filter"] = raw.copy()
                if apply_ica:
                    raw = preprocessing_steps.apply_ica(raw, method=ica_method, n_components=n_components)
                    preprocessing_results["ica"] = raw.copy()

                # Update the global raw object
                global_raw = raw
                st.success("Preprocessing applied successfully!")

                # Visualizations
                for step, result_raw in preprocessing_results.items():
                    st.subheader(f"Visualization: {step.capitalize()}")
                    fig_before = visualization.plot_raw_signal(global_raw, step + "_before")
                    st.pyplot(fig_before)
                    fig_after = visualization.plot_raw_signal(result_raw, step + "_after")
                    st.pyplot(fig_after)

            except Exception as e:
                st.error(f"Error during preprocessing: {e}")

### Save & Download Tab ###
with tabs[2]:
    st.header("Step 3: Save and Download")
    if global_raw is None:
        st.warning("No processed data available. Please preprocess a file first.")
    else:
        st.subheader("Save Processed File")
        file_name = st.text_input("Enter file name:", "processed_file.edf")
        if st.button("Save File"):
            try:
                output_path = Path(f"output/{file_name}")
                global_raw.export(output_path, fmt="EDF", overwrite=True)
                st.success(f"File saved successfully at: {output_path}")

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download File",
                        data=f,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"Error saving file: {e}")
