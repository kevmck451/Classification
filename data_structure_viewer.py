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

def plot_all_class_fft_heatmaps(data_folder, freq_start=40, freq_stop=12000):
    pkl_files = sorted(Path(data_folder).glob("*.pkl"))
    num_classes = len(pkl_files)
    ncols = 2
    nrows = (num_classes + 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False, constrained_layout=True)

    for i, pkl_file in enumerate(pkl_files):
        class_name = pkl_file.stem
        total_chunks = count_chunks(pkl_file)

        # Get FFT size from first chunk
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

        vmax = np.percentile(fft_matrix, 99)
        ax = axs[i // ncols][i % ncols]
        im = ax.imshow(fft_matrix.T, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax)
        ax.set_title(class_name)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yticks(np.linspace(0, freq_stop - freq_start, 5))
        ax.set_yticklabels(np.linspace(freq_start, freq_stop, 5, dtype=int))

    fig.colorbar(im, ax=axs, label='Magnitude', shrink=0.85)
    plt.show()

# Example usage
plot_all_class_fft_heatmaps("data_structures", freq_start=40, freq_stop=12000)
