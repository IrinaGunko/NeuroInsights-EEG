from utils.logger_manager import LoggerManager


class ExtractedFeaturesRepository:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = LoggerManager.get_logger(self.__class__.__name__)

    def initialize_feature_tables(self):
        feature_schemas = [
            '''
            CREATE SEQUENCE IF NOT EXISTS statistical_features_seq;
            CREATE TABLE IF NOT EXISTS statistical_features (
                id INTEGER PRIMARY KEY DEFAULT nextval('statistical_features_seq'),
                session_id INTEGER,
                recording_filename VARCHAR(100),
                channel VARCHAR(10),
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
            ''',
            '''
            CREATE SEQUENCE IF NOT EXISTS tfr_features_seq;
            CREATE TABLE IF NOT EXISTS tfr_features (
                id INTEGER PRIMARY KEY DEFAULT nextval('tfr_features_seq'),
                session_id INTEGER,
                recording_filename VARCHAR(100),
                channel VARCHAR(10),
                band VARCHAR(20),
                power_tfr_morlet DOUBLE,
                power_psd_welch DOUBLE,
                power_psd_welch_normalized DOUBLE,
                band_power DOUBLE,
                relative_power DOUBLE,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            ''',
            '''
            CREATE SEQUENCE IF NOT EXISTS connectivity_features_seq;
            CREATE TABLE IF NOT EXISTS connectivity_features (
                id INTEGER PRIMARY KEY DEFAULT nextval('connectivity_features_seq'),
                session_id INTEGER,
                recording_filename VARCHAR(100),
                channel_1 VARCHAR(10),
                channel_2 VARCHAR(10),
                plv DOUBLE,
                coherence DOUBLE,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            '''
        ]
        try:
            with self.db_manager as manager:
                for schema in feature_schemas:
                    manager.connection.execute(schema)
            self.logger.info("Feature tables initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize feature tables: {e}")
            raise

    def add_statistical_features(self, features):
        query = """
        INSERT INTO statistical_features (
            session_id, recording_filename, channel, amplitude_modulation,
            event_related_dynamics, spectral_entropy, signal_variance,
            hjorth_activity, hjorth_mobility, hjorth_complexity, peak_to_peak_amplitude,
            shannon_entropy, mean, variance, standard_deviation, peak_to_peak,
            zero_crossing_rate, kurtosis, skewness, snr, spike_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self.db_manager as manager:
                for feature in features:
                    manager.connection.execute(query, (
                        int(feature["session_id"]),
                        str(feature["recording_filename"]),
                        str(feature["Channel"]),
                        float(feature["AmplitudeModulation"]),
                        float(feature["EventRelatedDynamics"]),
                        float(feature["SpectralEntropy"]),
                        float(feature["SignalVariance"]),
                        float(feature["HjorthActivity"]),
                        float(feature["HjorthMobility"]),
                        float(feature["HjorthComplexity"]),
                        float(feature["PeakToPeakAmplitude"]),
                        float(feature["ShannonEntropy"]),
                        float(feature["Mean"]),
                        float(feature["Variance"]),
                        float(feature["StandardDeviation"]),
                        float(feature["PeakToPeak"]),
                        float(feature["ZeroCrossingRate"]),
                        float(feature["Kurtosis"]),
                        float(feature["Skewness"]),
                        float(feature["SNR"]),
                        int(feature["SpikeCount"]),
                    ))
            self.logger.info(f"Added {len(features)} statistical features.")
        except Exception as e:
            self.logger.error(f"Failed to add statistical features: {e}")
            raise

    def add_tfr_features(self, features):
        query = """
        INSERT INTO tfr_features (
            session_id, recording_filename, channel, band,
            power_tfr_morlet, power_psd_welch, power_psd_welch_normalized,
            band_power, relative_power
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    ))
            self.logger.info(f"Added {len(features)} TFR features.")
        except Exception as e:
            self.logger.error(f"Failed to add TFR features: {e}")
            raise

