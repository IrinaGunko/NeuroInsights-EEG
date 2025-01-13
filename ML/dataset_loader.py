import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/neuroinsights.db"

class DatasetLoader:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def load_features(self, label_column="cognitive_load_status", test_size=0.2, random_state=42):
        query = f"""
        SELECT
            tf.power_tfr_morlet,
            tf.power_psd_welch,
            tf.band_power,
            tf.relative_power,
            sf.amplitude_modulation,
            sf.event_related_dynamics,
            sf.signal_variance,
            sf.hjorth_activity,
            sf.hjorth_mobility,
            sf.hjorth_complexity,
            sf.peak_to_peak_amplitude,
            sf.zero_crossing_rate,
            sf.spectral_entropy,
            sf.shannon_entropy,
            sf.mean,
            sf.variance,
            sf.standard_deviation,
            sf.peak_to_peak,
            sf.kurtosis,
            sf.skewness,
            sf.snr,
            sf.spike_count,
            s.{label_column}
        FROM
            tfr_features AS tf
        LEFT JOIN
            statistical_features AS sf
        ON
            tf.session_id = sf.session_id
            AND tf.channel = sf.channel
        LEFT JOIN
            sessions AS s
        ON
            tf.session_id = s.session_id;
        """
        with duckdb.connect(self.db_path) as conn:
            data = conn.execute(query).fetch_df()

        print(f"Loaded data shape: {data.shape}")
        if data.empty:
            raise ValueError("The query returned no data. Check the database and query.")

        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in the dataset.")

        label_mapping = {"PRE": 0, "POST": 1}
        labels = data[label_column].map(label_mapping)
        if labels.isnull().any():
            raise ValueError("Label column contains values outside of the expected range ('PRE', 'POST').")

        columns_to_drop = [label_column, "recording_filename", "channel", "band", "session_id"]
        features = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        print("Features (first 5 rows):")
        print(features.head())
        print("Labels (first 5 values):")
        print(labels.head())

        features_to_normalize = [
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

        scaler = StandardScaler()
        normalize_cols = [col for col in features_to_normalize if col in features.columns]
        if normalize_cols:
            features[normalize_cols] = scaler.fit_transform(features[normalize_cols])
        else:
            print("No columns selected for normalization.")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test
