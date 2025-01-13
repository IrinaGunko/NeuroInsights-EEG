import mne
from pathlib import Path
from utils.logger_manager import LoggerManager
from utils.visualization import plot_raw, plot_psd
import matplotlib


logger = LoggerManager.get_logger("Preprocessing")

PREPROCESSING_METHODS = {}

def register_preprocessing_method(name):
    def decorator(func):
        PREPROCESSING_METHODS[name] = func
        return func
    return decorator

def save_file(raw, output_dir, base_name, suffix):
    try:
        file_path = Path(output_dir) / f"{base_name}_{suffix}.edf"
        raw.export(file_path, fmt="EDF", overwrite=True)
        logger.info(f"File saved: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file {base_name}_{suffix}: {e}", exc_info=True)
        return None

def save_plot(fig, output_dir, base_name, suffix):
    try:
        plot_path = Path(output_dir) / f"{base_name}_{suffix}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved: {plot_path}")
        return str(plot_path)
    except Exception as e:
        logger.error(f"Error saving plot {base_name}_{suffix}: {e}", exc_info=True)
        return None

@register_preprocessing_method("downsample")
def downsample(raw, target_sfreq=250):
    raw.resample(sfreq=target_sfreq)
    logger.info(f"Downsampled to {target_sfreq} Hz.")
    return raw

@register_preprocessing_method("apply_notch_filter")
def apply_notch_filter(raw, freqs=(50, 100)):
    raw.notch_filter(freqs=freqs, fir_design="firwin")
    logger.info(f"Notch filter applied for frequencies: {freqs}.")
    return raw

@register_preprocessing_method("apply_bandpass_filter")
def apply_bandpass_filter(raw, l_freq=1, h_freq=99):
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    logger.info(f"Band-pass filter applied (l_freq={l_freq}, h_freq={h_freq}).")
    return raw

@register_preprocessing_method("apply_ica")
def apply_ica(raw, method="fastica", n_components=40, random_state=42, max_iter="auto", montage="standard_1020"):
    raw.set_montage(montage)
    logger.info(f"Montage set to {montage} for ICA.")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter
    )
    ica.fit(raw)
    logger.info(f"ICA applied using {method}.")
    fig = ica.plot_components(show=False)
    fig2 = ica.plot_overlay(raw, show=False)
    ica.apply(raw)

    return raw , fig, fig2

def preprocess_file(input_file, output_dir, preprocessing_steps):
    try:
        logger.info(f"Processing file: {input_file}")
        raw = mne.io.read_raw_edf(input_file, preload=True)
        base_name = Path(input_file).stem
        for step in preprocessing_steps:
            method_name = step.get("method")
            params = step.get("params", {})
            if method_name in PREPROCESSING_METHODS:
                if method_name == "apply_ica":
                    raw_ica = raw.copy()
                    raw_ica = PREPROCESSING_METHODS[method_name](raw_ica, **params)
                    algo = params.get("method", "ica")
                    ica_file = save_file(raw_ica, output_dir, base_name, f"{algo}_processed")
                else:
                    raw = PREPROCESSING_METHODS[method_name](raw, **params)
            else:
                logger.error(f"Unknown preprocessing method: {method_name}")
        return {"ica_files": ica_file}

    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}", exc_info=True)
        return {"error": str(e)}
