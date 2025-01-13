SCHEMA_DEFINITIONS = [
    '''
    CREATE SEQUENCE IF NOT EXISTS participants_seq;
    CREATE TABLE IF NOT EXISTS participants (
        participant_id INTEGER PRIMARY KEY DEFAULT nextval('participants_seq'),
        participant_name VARCHAR(10),
        gender CHAR(1),
        age TINYINT,
        handedness VARCHAR(10),
        is_followup BOOLEAN
    );
    ''',
    '''
    CREATE SEQUENCE IF NOT EXISTS sessions_seq;
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY DEFAULT nextval('sessions_seq'),
        participant_id INTEGER,
        session_number INTEGER,
        recording_year INTEGER,
        recording_duration INTEGER,
        late_trigger_count INTEGER,
        is_followup BOOLEAN,
        recording_filename VARCHAR(100),
        eyes_state CHAR(1),
        cognitive_load_status VARCHAR(3),
        FOREIGN KEY (participant_id) REFERENCES participants(participant_id)
    );
    ''',
    '''
    CREATE SEQUENCE IF NOT EXISTS late_trigger_events_seq;
    CREATE TABLE IF NOT EXISTS late_trigger_events (
        event_id INTEGER PRIMARY KEY DEFAULT nextval('late_trigger_events_seq'),
        session_id INTEGER,
        onset INTEGER,
        duration INTEGER,
        type VARCHAR(50),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    ''',
    '''
    CREATE SEQUENCE IF NOT EXISTS eeg_metadata_seq;
    CREATE TABLE IF NOT EXISTS eeg_metadata (
        eeg_metadata_id INTEGER PRIMARY KEY DEFAULT nextval('eeg_metadata_seq'),
        task_name VARCHAR(50),
        institution_name VARCHAR(100),
        institution_address VARCHAR(100),
        institutional_department VARCHAR(50),
        manufacturer VARCHAR(50),
        manufacturer_model_name VARCHAR(50),
        cap_manufacturer VARCHAR(50),
        cap_model_name VARCHAR(50),
        recording_type VARCHAR(20),
        eeg_placement_scheme VARCHAR(20),
        eeg_reference VARCHAR(10),
        sampling_frequency INTEGER,
        software_filters VARCHAR(50),
        eeg_channel_count INTEGER,
        eog_channel_count INTEGER,
        power_line_frequency INTEGER,
        eeg_ground VARCHAR(50)
    );
    ''',
    '''
    CREATE SEQUENCE IF NOT EXISTS eeg_channels_seq;
    CREATE TABLE IF NOT EXISTS eeg_channels (
        channel_id INTEGER PRIMARY KEY DEFAULT nextval('eeg_channels_seq'),
        channel_name VARCHAR(10),
        channel_type VARCHAR(10),
        units VARCHAR(10),
        low_cutoff VARCHAR(10),
        high_cutoff INTEGER
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
    ''',
    '''
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
]