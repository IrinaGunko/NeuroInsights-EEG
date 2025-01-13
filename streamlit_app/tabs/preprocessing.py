import streamlit as st
from pathlib import Path
import matplotlib
from utils.visualization import plot_raw, plot_psd, plot_ica_components, plot_ica_overlay
from preprocessing.preprocessing import PREPROCESSING_METHODS
matplotlib.use("Agg")

def render():
    st.title("Preprocessing")
    st.subheader("Choose Preprocessing Steps")

    if "raw_initial" not in st.session_state or st.session_state.raw_initial is None:
        st.warning("Please upload and load an EEG file in the 'Upload EEG File' tab.")
        return

    raw_initial = st.session_state.raw_initial
    preprocessing_steps = []

    if st.checkbox("Downsample"):
        target_sfreq = st.number_input(
            "Target Sampling Frequency (Hz):", min_value=1, max_value=1000, value=250, step=1,
            help="The target sampling frequency for the EEG data."
        )
        preprocessing_steps.append({"method": "downsample", "params": {"target_sfreq": target_sfreq}})

    if st.checkbox("Apply Notch Filter"):
        freqs_input = st.text_input(
            "Notch Filter Frequencies (comma-separated):", value="50, 60",
            help="Frequencies to apply notch filtering (e.g., 50, 60 Hz)."
        )
        freqs = [float(f.strip()) for f in freqs_input.split(",")]
        preprocessing_steps.append({"method": "apply_notch_filter", "params": {"freqs": freqs}})

    if st.checkbox("Apply Bandpass Filter"):
        l_freq = st.number_input(
            "Low Frequency (Hz):", min_value=0.1, max_value=500.0, value=1.0, step=0.1,
            help="Lower bound of the frequency range for the band-pass filter."
        )
        h_freq = st.number_input(
            "High Frequency (Hz):", min_value=0.1, max_value=500.0, value=99.0, step=0.1,
            help="Upper bound of the frequency range for the band-pass filter."
        )
        preprocessing_steps.append({"method": "apply_bandpass_filter", "params": {"l_freq": l_freq, "h_freq": h_freq}})

    if st.checkbox("Apply Independent Component Analysis (ICA)"):
        ica_method = st.selectbox(
            "ICA Method:", ["fastica", "picard", "infomax"],
            help="Method to use for ICA decomposition."
        )
        n_components = st.number_input(
            "Number of ICA Components:", min_value=1, max_value=100, value=20, step=1,
            help="Number of ICA components to compute."
        )
        preprocessing_steps.append({"method": "apply_ica", "params": {"method": ica_method, "n_components": n_components}})

    if st.button("Run Preprocessing"):
        if not preprocessing_steps:
            st.warning("Please select at least one preprocessing step.")
            return

        processed_raw = raw_initial.copy()

        for step in preprocessing_steps:
            method = step["method"]
            params = step["params"]

            st.markdown(f"### Visualization for {method.replace('_', ' ').capitalize()}")
            try:
                raw_before = processed_raw.copy()
                if method == "apply_ica":
                    raw_ica = processed_raw.copy()
                    processed_raw, fig, fig2 = PREPROCESSING_METHODS[method](raw_ica, **params)
                    st.pyplot(fig)
                    st.pyplot(fig2)
                else:
                    processed_raw = PREPROCESSING_METHODS[method](processed_raw, **params)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before")
                    if method in ["apply_notch_filter", "apply_bandpass_filter"]:
                        st.pyplot(plot_psd(raw_before))
                    else:
                        st.pyplot(plot_raw(raw_before))
                with col2:
                    st.markdown("#### After")
                    if method in ["apply_notch_filter", "apply_bandpass_filter"]:
                        st.pyplot(plot_psd(processed_raw))
                    else:
                        st.pyplot(plot_raw(processed_raw))

                st.success(f"{method.replace('_', ' ').capitalize()} applied successfully.")
            except Exception as e:
                st.error(f"Failed to apply {method.replace('_', ' ').capitalize()}: {e}")

        st.session_state["processed_raw"] = processed_raw
        st.success("All selected preprocessing steps applied successfully.")

    if "processed_raw" in st.session_state:
        st.markdown("### Download Preprocessed File")
        processed_raw = st.session_state["processed_raw"]
        temp_dir = Path("./temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        edf_file = temp_dir / "preprocessed_file.edf"
        fif_file = temp_dir / "preprocessed_file.fif"

        processed_raw.export(str(edf_file), fmt="EDF", overwrite=True)
        processed_raw.save(str(fif_file), overwrite=True)

        with open(edf_file, "rb") as f:
            edf_data = f.read()
            st.download_button(
                label="Download as .edf",
                data=edf_data,
                file_name=edf_file.name,
                mime="application/octet-stream",
            )

        with open(fif_file, "rb") as f:
            fif_data = f.read()
            st.download_button(
                label="Download as .fif",
                data=fif_data,
                file_name=fif_file.name,
                mime="application/octet-stream",
            )
