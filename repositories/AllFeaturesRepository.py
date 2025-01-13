from utils.logger_manager import LoggerManager

class AllFeaturesRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def initialize_all_features_table(self):
        schema = '''
        CREATE SEQUENCE IF NOT EXISTS all_features_seq;
        CREATE TABLE IF NOT EXISTS all_features (
            id INTEGER PRIMARY KEY DEFAULT nextval('all_features_seq'),
            session_id INTEGER,
            recording_filename VARCHAR(100),
            channel VARCHAR(10),
            band VARCHAR(20),
            power_tfr_morlet DOUBLE,
            power_psd_welch DOUBLE,
            power_psd_welch_normalized DOUBLE,
            band_power DOUBLE,
            relative_power DOUBLE,
            amplitude_modulation DOUBLE,
            event_related_dynamics DOUBLE,
            spectral_entropy DOUBLE,
            signal_variance DOUBLE,
            hjorth_activity DOUBLE,
            hjorth_mobility DOUBLE,
            hjorth_complexity DOUBLE,
            peak_to_peak_amplitude DOUBLE,
            shannon_entropy DOUBLE,
            mean DOUBLE,
            variance DOUBLE,
            standard_deviation DOUBLE,
            peak_to_peak DOUBLE,
            zero_crossing_rate DOUBLE,
            kurtosis DOUBLE,
            skewness DOUBLE,
            snr DOUBLE,
            spike_count INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        '''
        try:
            with self.db_manager as manager:
                manager.connection.execute(schema)
            self.logger.info("All features table initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize all_features table: {e}")
            raise

    def add_all_features(self, features):
        query = """
        INSERT INTO all_features (
            session_id, recording_filename, channel, band,
            power_tfr_morlet, power_psd_welch, power_psd_welch_normalized,
            band_power, relative_power, amplitude_modulation,
            event_related_dynamics, spectral_entropy, signal_variance,
            hjorth_activity, hjorth_mobility, hjorth_complexity,
            peak_to_peak_amplitude, shannon_entropy, mean,
            variance, standard_deviation, peak_to_peak,
            zero_crossing_rate, kurtosis, skewness, snr, spike_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self.db_manager as manager:
                for feature in features:
                    manager.connection.execute(query, (
                        int(feature["session_id"]),
                        str(feature["recording_filename"]),
                        str(feature["Channel"]),
                        str(feature["Band"]),
                        float(feature.get("PowerTFRMorlet", 0)),
                        float(feature.get("PowerPSDWelch", 0)),
                        float(feature.get("PowerPSDWelchNormalized", 0)),
                        float(feature.get("BandPower", 0)),
                        float(feature.get("RelativePower", 0)),
                        float(feature.get("AmplitudeModulation", 0)),
                        float(feature.get("EventRelatedDynamics", 0)),
                        float(feature.get("SpectralEntropy", 0)),
                        float(feature.get("SignalVariance", 0)),
                        float(feature.get("HjorthActivity", 0)),
                        float(feature.get("HjorthMobility", 0)),
                        float(feature.get("HjorthComplexity", 0)),
                        float(feature.get("PeakToPeakAmplitude", 0)),
                        float(feature.get("ShannonEntropy", 0)),
                        float(feature.get("Mean", 0)),
                        float(feature.get("Variance", 0)),
                        float(feature.get("StandardDeviation", 0)),
                        float(feature.get("PeakToPeak", 0)),
                        float(feature.get("ZeroCrossingRate", 0)),
                        float(feature.get("Kurtosis", 0)),
                        float(feature.get("Skewness", 0)),
                        float(feature.get("SNR", 0)),
                        int(feature.get("SpikeCount", 0)),
                    ))
            self.logger.info(f"Added {len(features)} all_features entries.")
        except Exception as e:
            self.logger.error(f"Failed to add all_features: {e}")
            raise

    def delete_all_features(self):
        query = "DELETE FROM all_features;"
        try:
            with self.db_manager as manager:
                manager.connection.execute(query)
            self.logger.info("All entries in all_features table deleted successfully.")
        except Exception as e:
            self.logger.error(f"Failed to delete all_features: {e}")
            raise

    def fetch_all_features(self, limit=None):
        query = "SELECT * FROM all_features"
        if limit:
            query += f" LIMIT {limit}"
        try:
            with self.db_manager as manager:
                result = manager.connection.execute(query).fetchall()
            self.logger.info(f"Fetched {len(result)} all_features entries.")
            return result
        except Exception as e:
            self.logger.error(f"Failed to fetch all_features: {e}")
            raise
