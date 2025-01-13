import matplotlib.pyplot as plt
from pathlib import Path
from utils.logger_manager import LoggerManager
import mne

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
