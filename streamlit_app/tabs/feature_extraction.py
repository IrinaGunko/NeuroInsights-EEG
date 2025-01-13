import numpy as np
import streamlit as st
import pandas as pd
import mne

from feature_extraction.feature_extractor import (
    extract_temporal_frequency_features,
    extract_statistical_features,
    extract_psd_features,
    extract_tfr_features,
    compute_band_and_relative_power,
    compute_channel_basic_features,
)

def render():
    st.title("Feature Extraction")

    pd.options.display.float_format = '{:.12g}'.format

    if "raw_initial" in st.session_state and st.session_state.raw_initial:
        raw = st.session_state.raw_initial

        raw.pick(raw.info["ch_names"][:64])

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)

        st.info("Using uploaded or preloaded EEG file with 10-20 montage applied.")

        st.info("Extracting features from EEG data. This may take a few moments...")

        temporal_freq_features = pd.DataFrame(extract_temporal_frequency_features(raw))
        statistical_features = pd.DataFrame(extract_statistical_features(raw))
        psd_features = pd.DataFrame(extract_psd_features(raw))
        tfr_features = pd.DataFrame(
            extract_tfr_features(raw, freqs=np.array([4, 8, 13, 30, 50]), n_cycles=7)
        )
        band_relative_power_features = pd.DataFrame(compute_band_and_relative_power(raw))
        channel_basic_features = pd.DataFrame(compute_channel_basic_features(raw))

        combined_features = pd.concat(
            [
                temporal_freq_features,
                statistical_features,
                psd_features,
                tfr_features,
                band_relative_power_features,
                channel_basic_features,
            ],
            axis=1,
        )

        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        numeric_features = combined_features.select_dtypes(include=["number"])

        st.session_state.features = numeric_features

        st.subheader("Extracted Features")
        st.dataframe(numeric_features)

        csv = numeric_features.to_csv(index=False)
        st.download_button(
            label="Download Features as CSV",
            data=csv,
            file_name="extracted_features.csv",
            mime="text/csv",
        )

    else:
        st.warning("No EEG file loaded. Please upload or select a file in the Upload tab.")
