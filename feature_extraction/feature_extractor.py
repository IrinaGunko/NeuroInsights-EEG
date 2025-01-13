import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from mne.time_frequency import tfr_array_morlet, psd_array_welch
from mne.filter import filter_data



BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
    "gamma_high": (50, 99),
}

def extract_temporal_frequency_features(raw):
    features = []
    for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
        data = raw.get_data()[ch_idx]
        amplitude_modulation = np.abs(np.diff(data)).mean()
        event_related_dynamics = np.max(data) - np.min(data)
        features.append({
            "Channel": ch_name,
            "AmplitudeModulation": amplitude_modulation,
            "EventRelatedDynamics": event_related_dynamics,
        })
    return features

def extract_statistical_features(raw, entropy_bins=100, spike_threshold_multiplier=5):
    features = []
    data = raw.get_data()
    for ch_idx, ch_name in enumerate(raw.info['ch_names']):
        channel_data = data[ch_idx]
        shannon_entropy = entropy(np.histogram(channel_data, bins=entropy_bins, density=True)[0])
        mean_val = np.mean(channel_data)
        variance = np.var(channel_data)
        std_dev = np.std(channel_data)
        peak_to_peak = np.ptp(channel_data)
        zero_crossing_rate = ((np.diff(np.sign(channel_data)) != 0).sum()) / len(channel_data)
        kurtosis_val = kurtosis(channel_data)
        skewness_val = skew(channel_data)
        signal_power = np.mean(channel_data ** 2)
        noise_power = np.var(channel_data - np.mean(channel_data))
        snr = signal_power / noise_power if noise_power > 0 else np.nan
        spike_threshold = spike_threshold_multiplier * std_dev
        spike_count = np.sum(np.abs(channel_data) > spike_threshold)
        features.append({
            "Channel": ch_name,
            "ShannonEntropy": shannon_entropy,
            "Mean": mean_val,
            "Variance": variance,
            "StandardDeviation": std_dev,
            "PeakToPeak": peak_to_peak,
            "ZeroCrossingRate": zero_crossing_rate,
            "Kurtosis": kurtosis_val,
            "Skewness": skewness_val,
            "SNR": snr,
            "SpikeCount": spike_count,
        })
    return features

def extract_psd_features(raw, fmin=1, fmax=99, n_fft=1024, n_overlap=512, n_per_seg=None, n_jobs=-1):
    features = []
    data = raw.get_data()
    sfreq = raw.info.get("sfreq", 250)
    psds, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        n_jobs=n_jobs,
        average="mean",
    )
    total_power = psds.sum(axis=1)
    for band, (fmin, fmax) in BANDS.items():
        valid_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        band_power = psds[:, valid_indices].mean(axis=1)  # Mean across band
        for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
            normalized_power = band_power[ch_idx] / total_power[ch_idx] if total_power[ch_idx] != 0 else 0
            features.append({
                "Channel": ch_name,
                "Band": band,
                "PowerPSDWelch": band_power[ch_idx],
                "PowerPSDWelchNormalized": normalized_power,
            })
    return features

def extract_plv_features(raw):
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    channel_names = raw.info['ch_names']
    n_channels = len(channel_names)
    features = []
    for band_name, (fmin, fmax) in BANDS.items():
        band_data = filter_data(data, sfreq, fmin, fmax, verbose=False)
        phase_data = np.angle(np.exp(1j * np.angle(np.fft.fft(band_data, axis=1))))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                plv = np.abs(np.mean(np.exp(1j * (phase_data[i] - phase_data[j]))))
                features.append({
                    "Channel1": channel_names[i],
                    "Channel2": channel_names[j],
                    "PLV": plv,
                    "FrequencyBand": band_name
                })
    return features

def extract_tfr_features(raw, freqs, n_cycles, use_fft=True, decim=1, n_jobs=-1, output="power"):
    features = []
    data = raw.get_data()[np.newaxis, :, :]
    sfreq = raw.info.get("sfreq", 250)
    tfr_data_all = tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=use_fft,
        decim=decim,
        output=output,
        n_jobs=n_jobs,
    )
    tfr_data = tfr_data_all[0]
    time_avg_tfr = tfr_data.mean(axis=-1)
    for band, (fmin, fmax) in BANDS.items():
        valid_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if valid_indices.size == 0:
            for ch_name in raw.info["ch_names"]:
                features.append({
                    "Channel": ch_name,
                    "Band": band,
                    "PowerTFRMorlet": np.nan,
                })
            continue
        band_power = time_avg_tfr[:, valid_indices].mean(axis=1)
        for ch_idx, ch_name in enumerate(raw.info["ch_names"]):
            features.append({
                "Channel": ch_name,
                "Band": band,
                "PowerTFRMorlet": band_power[ch_idx],
            })
    return features

def normalize_features(df, columns):
    for col in columns:
        if col in df:
            zscore_col = f"{col}_ZScore"
            df[zscore_col] = (df[col] - df[col].mean()) / df[col].std()
            minmax_col = f"{col}_MinMaxScaling"
            df[minmax_col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def band_power(psds, freqs):
    features = {}
    for band, (fmin, fmax) in BANDS.items():
        band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(band_idx) == 0:
            features[band] = np.zeros(psds.shape[0])
        else:
            features[band] = psds[:, band_idx].mean(axis=1)
    return features

def relative_power(band_power, total_power):
    return {
        band: np.divide(
            power, total_power, where=(total_power != 0), out=np.zeros_like(power)
        )
        for band, power in band_power.items()
    }
def spectral_entropy(psds):
    psds_norm = psds / np.sum(psds, axis=1, keepdims=True)
    psds_norm = np.clip(psds_norm, 1e-12, None)
    return entropy(psds_norm, axis=1)

def compute_band_and_relative_power(raw):
    psds, freqs = raw.compute_psd(fmin=1, fmax=99).get_data(return_freqs=True)
    num_channels = min(64, psds.shape[0])
    channel_names = raw.info["ch_names"][:num_channels]
    total_power = psds.sum(axis=1)[:num_channels]
    bp = band_power(psds, freqs)
    rp = relative_power(bp, total_power)
    features = []
    for ch_idx, ch_name in enumerate(channel_names):
        for band in BANDS:
            features.append({
                "Channel": ch_name,
                "Band": band,
                "BandPower": bp[band][ch_idx],
                "RelativePower": rp[band][ch_idx],
            })
    return features


def compute_channel_basic_features(raw):
    data = raw.get_data()
    psds, _ = raw.compute_psd(fmin=1, fmax=99).get_data(return_freqs=True)
    num_channels = min(64, data.shape[0])
    channel_names = raw.info["ch_names"][:num_channels]

    se = spectral_entropy(psds)[:num_channels]
    sv = np.var(data, axis=1)[:num_channels]
    activity = np.var(data, axis=1)[:num_channels]
    mobility = np.sqrt(np.var(np.diff(data, axis=1), axis=1) / activity)[:num_channels]
    complexity = np.sqrt(
        np.var(np.diff(np.diff(data, axis=1), axis=1), axis=1)
        / np.var(np.diff(data, axis=1), axis=1)
    )[:num_channels]
    ppa = np.ptp(data, axis=1)[:num_channels]

    features = []
    for ch_idx, ch_name in enumerate(channel_names):
        row = {
            "Channel": ch_name,
            "SpectralEntropy": se[ch_idx],
            "SignalVariance": sv[ch_idx],
            "HjorthActivity": activity[ch_idx],
            "HjorthMobility": mobility[ch_idx],
            "HjorthComplexity": complexity[ch_idx],
            "PeakToPeakAmplitude": ppa[ch_idx],
        }
        features.append(row)
    return features