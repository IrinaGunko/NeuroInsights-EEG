import matplotlib.pyplot as plt
from pathlib import Path
from utils.logger_manager import LoggerManager
import mne
import numpy as np
from nilearn import plotting
logger = LoggerManager.get_logger("Visualization")

def plot_raw(raw):
    fig = raw.plot(show=False, block=False)
    return fig

def plot_psd(raw, fmin=0.5, fmax=99):
    raw.set_montage("standard_1020")
    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    fig = psd.plot(picks="eeg", show=False)
    return fig

def plot_ica_components(ica, raw):
    fig = ica.plot_components(show=False)
    return fig

def plot_ica_overlay(ica, raw):
    fig = ica.plot_overlay(raw, show=False)
    return fig

def plot_topomap(data, info, times=None, title="Topographic Map"):
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, info, axes=ax, show=False, cmap="RdBu_r", sphere="auto")
    ax.set_title(title)
    return fig

def plot_spectrogram(raw, fmin=0.1, fmax=50.0):
    fig = raw.plot_psd_topomap(fmin=fmin, fmax=fmax, show=False)
    return fig

def plot_epochs(epochs):
    fig = epochs.plot(show=False)
    return fig

def plot_evoked(evoked):
    fig = evoked.plot(show=False)
    return fig

def plot_connectivity(connectivity_matrix, info):
    fig, ax = plt.subplots()
    mne.viz.plot_connectivity_circle(connectivity_matrix, info.ch_names, show=False, ax=ax)
    return fig

def plot_channel_overlay(raw):
    fig = raw.plot(show=False, duration=30, n_channels=len(raw.ch_names), block=False)
    return fig

import matplotlib.pyplot as plt
from pathlib import Path

def plot_static_brain_surface(stc, subjects_dir, save_path):
    brain = stc.plot(subjects_dir=subjects_dir, hemi='both', views='lat', size=(800, 600), background='white')
    brain.save_image(save_path)
    brain.close()



def plot_topographic_map(stc_atlas, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stc_atlas.mean(axis=1))  # Example: Mean time course of atlas regions
    ax.set_title("Topographic Map")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    fig.savefig(save_path)
    plt.close(fig)


def plot_roi_time_series(stc_atlas_path, labels_path, output_dir, file_name="roi_time_series.png"):
    try:
        # Load ROI time series and labels
        stc_atlas = np.load(stc_atlas_path)
        labels = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s',
                                            subjects_dir=labels_path)

        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, time_series in enumerate(stc_atlas):
            ax.plot(time_series, label=labels[idx].name)
        ax.set_title("ROI Time Series (Destrieux Atlas)")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Activity")
        ax.legend(loc="upper right", fontsize="small", bbox_to_anchor=(1.2, 1))
        # Save plot
        plot_path = Path(output_dir) / file_name
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Time series plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error plotting ROI time series: {e}")

def plot_brain_surface(stc_path, subjects_dir, surface='inflated', file_name="brain_surface.png"):
    try:
        # Load source estimate
        stc = mne.read_source_estimate(stc_path)
        # Plot on brain surface
        brain = stc.plot(subject='fsaverage', subjects_dir=subjects_dir, surface=surface)
        screenshot = brain.screenshot()
        # Save plot
        output_path = Path(stc_path).parent / file_name
        plt.imsave(output_path, screenshot)
        print(f"Brain surface plot saved to: {output_path}")
    except Exception as e:
        print(f"Error plotting brain surface: {e}")
