import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

MODEL_PATH = "models/lightgbm.pkl"
TRAINED_FEATURES = [
    "power_tfr_morlet",
    "power_psd_welch",
    "band_power",
    "relative_power",
    "amplitude_modulation",
    "event_related_dynamics",
    "signal_variance",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_to_peak_amplitude",
    "zero_crossing_rate",
    "spectral_entropy",
    "shannon_entropy",
    "mean",
    "variance",
    "standard_deviation",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "snr",
    "spike_count",
]

NORMALIZE_COLS = [
    "amplitude_modulation",
    "event_related_dynamics",
    "signal_variance",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_to_peak_amplitude",
    "zero_crossing_rate",
    "power_tfr_morlet",
    "power_psd_welch",
    "band_power",
]

def load_model(path):
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def plot_feature_importance(model, features):
    importance = model.feature_importance()
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance,
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    st.pyplot(plt)

def plot_feature_contributions(feature_vector, feature_names):
    contributions = pd.DataFrame({
        "Feature": feature_names,
        "Value": feature_vector.flatten(),
    }).sort_values(by="Value", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(contributions["Feature"], contributions["Value"], color="orange")
    plt.xlabel("Contribution")
    plt.ylabel("Feature")
    plt.title("Feature Contributions to Prediction")
    st.pyplot(plt)

def render():
    st.title("ML Prediction")
    pd.options.display.float_format = '{:.12g}'.format

    if "features" in st.session_state and st.session_state.features is not None:
        features = st.session_state.features

        st.subheader("Raw Features (Before Processing)")
        st.dataframe(features.style.format("{:.12g}"))

        st.info("Loading trained model...")
        model = load_model(MODEL_PATH)

        if model is not None:
            COLUMN_NAME_MAPPING = {
                "AmplitudeModulation": "amplitude_modulation",
                "EventRelatedDynamics": "event_related_dynamics",
                "ShannonEntropy": "shannon_entropy",
                "Mean": "mean",
                "Variance": "variance",
                "StandardDeviation": "standard_deviation",
                "PeakToPeak": "peak_to_peak",
                "ZeroCrossingRate": "zero_crossing_rate",
                "Kurtosis": "kurtosis",
                "Skewness": "skewness",
                "SNR": "snr",
                "SpikeCount": "spike_count",
                "PowerPSDWelch": "power_psd_welch",
                "PowerPSDWelchNormalized": "power_psd_welch_normalized",
                "PowerTFRMorlet": "power_tfr_morlet",
                "BandPower": "band_power",
                "RelativePower": "relative_power",
                "SpectralEntropy": "spectral_entropy",
                "SignalVariance": "signal_variance",
                "HjorthActivity": "hjorth_activity",
                "HjorthMobility": "hjorth_mobility",
                "HjorthComplexity": "hjorth_complexity",
                "PeakToPeakAmplitude": "peak_to_peak_amplitude",
            }

            st.info("Renaming features to match model expectations...")
            features = features.rename(columns=COLUMN_NAME_MAPPING)

            missing_features = [col for col in TRAINED_FEATURES if col not in features.columns]
            st.warning(f"Missing Features: {missing_features}")

            features = features.reindex(columns=TRAINED_FEATURES, fill_value=0)

            st.subheader("Features After Reindexing")
            st.dataframe(features.style.format("{:.12g}"))

            st.info("Normalizing features...")
            scaler = StandardScaler()
            normalize_cols = [col for col in NORMALIZE_COLS if col in features.columns]
            if normalize_cols:
                features[normalize_cols] = scaler.fit_transform(features[normalize_cols])

            st.subheader("Normalized Features")
            st.dataframe(features.style.format("{:.12g}"))

            feature_vector = features.iloc[0].values.reshape(1, -1)

            st.info("Making predictions...")
            try:
                prediction = model.predict(feature_vector)[0]
                probability = model.predict(feature_vector, raw_score=False)[0]

                st.subheader("Prediction Result")
                if prediction == 1:
                    st.success(f"The EEG data indicates it was recorded **after** cognitive tasks.\n\nConfidence: {probability * 100:.2f}%")
                else:
                    st.warning(f"The EEG data indicates it was recorded **before** cognitive tasks.\n\nConfidence: {(1 - probability) * 100:.2f}%")

                st.subheader("Input Features")
                st.dataframe(features.style.format("{:.12g}"))

                st.subheader("Feature Contributions to Prediction")
                plot_feature_contributions(feature_vector, TRAINED_FEATURES)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Model could not be loaded. Please ensure the model file exists and is valid.")
    else:
        st.warning("No features extracted. Please extract features in the Feature Extraction tab before making predictions.")
