import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def count_chunks(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return len(data)

def iter_fft_chunks(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        for entry in data:
            yield entry["fft"]

def plot_all_class_fft_heatmaps(data_folder, freq_start=0, freq_stop=24000):
    pkl_files = sorted(Path(data_folder).glob("*.pkl"))
    num_classes = len(pkl_files)
    ncols = 2
    nrows = (num_classes + 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(36, 13), squeeze=False, constrained_layout=True)

    for i, pkl_file in enumerate(pkl_files):
        class_name = pkl_file.stem
        total_chunks = count_chunks(pkl_file)

        fft_shape = None
        for fft in iter_fft_chunks(pkl_file):
            fft_shape = fft.shape
            break
        if fft_shape is None:
            continue

        fft_matrix = np.empty((total_chunks, fft_shape[0]), dtype=np.float32)
        for j, fft in enumerate(iter_fft_chunks(pkl_file)):
            fft_matrix[j] = fft

        # Slice frequency range
        fft_matrix = fft_matrix[:, freq_start:freq_stop]

        # Resample FFT matrix to log scale
        num_bins = fft_matrix.shape[1]
        linear_freqs = np.linspace(freq_start, freq_stop, num_bins)
        log_freqs = np.geomspace(freq_start, freq_stop, num_bins)
        log_indices = np.interp(log_freqs, linear_freqs, np.arange(num_bins))

        log_fft_matrix = np.empty((fft_matrix.shape[0], num_bins), dtype=np.float32)
        for j in range(fft_matrix.shape[0]):
            log_fft_matrix[j] = np.interp(log_indices, np.arange(num_bins), fft_matrix[j])
        log_fft_matrix = log_fft_matrix.T  # transpose for imshow

        vmax = np.percentile(log_fft_matrix, 99)
        ax = axs[i // ncols][i % ncols]
        im = ax.imshow(log_fft_matrix, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax)
        ax.set_title(class_name)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Frequency (Hz)")

        tick_freqs = np.geomspace(freq_start, freq_stop, 6)
        tick_indices = [np.argmin(np.abs(log_freqs - f)) for f in tick_freqs]
        ax.set_yticks(tick_indices)
        ax.set_yticklabels([f"{int(f)}" for f in tick_freqs])

    fig.colorbar(im, ax=axs, label='Magnitude', shrink=0.85)
    plt.show()

# Example usage
plot_all_class_fft_heatmaps("data_structures", freq_start=60, freq_stop=3200)
