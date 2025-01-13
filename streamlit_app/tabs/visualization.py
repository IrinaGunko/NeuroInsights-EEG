import streamlit as st
from utils.visualization import (
    plot_raw,
    plot_psd,
    plot_ica_components,
    plot_ica_overlay,
    plot_topomap,
    plot_spectrogram,
    plot_epochs,
    plot_connectivity,
)
import mne

def render():
    st.title("EEG Visualization")

    if "raw_initial" not in st.session_state or st.session_state.raw_initial is None:
        st.warning("Please upload and load an EEG file in the Upload File tab.")
        return

    raw = st.session_state.raw_initial
    st.sidebar.title("Visualization Options")

    if st.sidebar.checkbox("Plot Raw Signals"):
        st.subheader("Raw EEG Signals")
        fig = plot_raw(raw)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Power Spectral Density (PSD)"):
        st.subheader("Power Spectral Density")
        fig = plot_psd(raw)
        st.pyplot(fig)

    if st.sidebar.checkbox("Plot Topographic Map"):
        st.subheader("Topographic Map")
        try:
            raw_copy = raw.copy().pick("eeg")
            data = raw_copy.get_data().mean(axis=1)
            info = raw_copy.info
            fig = plot_topomap(data, info, title="Topographic Map")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while plotting the topographic map: {e}")

    if st.sidebar.checkbox("Plot Spectrogram"):
        st.subheader("Spectrogram (Time-Frequency Plot)")
        try:
            fig = plot_spectrogram(raw)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while plotting the spectrogram: {e}")

    if st.sidebar.checkbox("Plot Epochs"):
        st.subheader("Segmented EEG Epochs")
        try:
            events = mne.make_fixed_length_events(raw, duration=1.0)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=1, preload=True, baseline=None)
            fig = plot_epochs(epochs)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while plotting epochs: {e}")

    if "ica" in st.session_state:
        ica = st.session_state["ica"]
        if st.sidebar.checkbox("Plot ICA Components"):
            st.subheader("ICA Components")
            try:
                fig = plot_ica_components(ica, raw)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while plotting ICA components: {e}")

        if st.sidebar.checkbox("Plot ICA Overlay"):
            st.subheader("ICA Overlay")
            try:
                fig = plot_ica_overlay(ica, raw)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while plotting the ICA overlay: {e}")

    if st.sidebar.checkbox("Plot Connectivity (Example)"):
        st.subheader("Connectivity Plot")
        try:
            connectivity_matrix = mne.connectivity.seed_target_indices(
                raw.info, method="coh"
            )
            info = raw.info
            fig = plot_connectivity(connectivity_matrix, info)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred while plotting connectivity: {e}")
