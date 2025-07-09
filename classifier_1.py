# open an audio file
# extract the spectra
# determine the shape so that i can make all class tempates / fingerprints match
# compare the fingerprint to the real sample to see if it's close enough to match or needs more refinement




import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def plot_fingerprints(fingerprints, freq_start, freq_stop):
    num_bins = len(next(iter(fingerprints.values())))
    freqs = np.geomspace(freq_start, freq_stop, num_bins)

    plt.figure(figsize=(10, 6))
    for label, fp in fingerprints.items():
        plt.plot(freqs, fp, label=label)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("Class Fingerprints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def iter_fft_chunks(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        for entry in data:
            yield entry["fft"]

def load_log_fft_matrix(pkl_file, freq_start, freq_stop):
    # Load all FFT chunks
    fft_list = list(iter_fft_chunks(pkl_file))
    fft_matrix = np.array(fft_list, dtype=np.float32)
    fft_matrix = fft_matrix[:, freq_start:freq_stop]

    # Convert to log scale
    num_bins = fft_matrix.shape[1]
    linear_freqs = np.linspace(freq_start, freq_stop, num_bins)
    log_freqs = np.geomspace(freq_start, freq_stop, num_bins)
    log_indices = np.interp(log_freqs, linear_freqs, np.arange(num_bins))

    log_fft_matrix = np.empty((fft_matrix.shape[0], num_bins), dtype=np.float32)
    for j in range(fft_matrix.shape[0]):
        log_fft_matrix[j] = np.interp(log_indices, np.arange(num_bins), fft_matrix[j])

    return log_fft_matrix.T  # shape: (freq_bins, chunks)

def build_class_fingerprints(data_folder, freq_start, freq_stop):
    fingerprints = {}
    for pkl_file in sorted(Path(data_folder).glob("*.pkl")):
        class_label = pkl_file.stem
        log_fft_matrix = load_log_fft_matrix(pkl_file, freq_start, freq_stop)
        fingerprint = np.mean(log_fft_matrix, axis=1)  # average across time
        fingerprint /= np.linalg.norm(fingerprint) + 1e-6
        fingerprints[class_label] = fingerprint

    with open("fingerprints.pkl", "wb") as f:
        pickle.dump(fingerprints, f)
    '''
    with open("fingerprints.pkl", "rb") as f:
        fingerprints = pickle.load(f)

    print(fingerprints.keys())
    # Output: dict_keys(['drone multi rotor', 'vehicles diesel', ...])
    '''

    return fingerprints

def match_test_file(test_file, class_fingerprints, freq_start, freq_stop):
    log_fft_matrix = load_log_fft_matrix(test_file, freq_start, freq_stop)
    test_fingerprint = np.mean(log_fft_matrix, axis=1)
    test_fingerprint /= np.linalg.norm(test_fingerprint) + 1e-6

    print(f"\nMatching: {test_file.name}\n")
    for label, fp in class_fingerprints.items():
        dist = cosine(test_fingerprint, fp)
        print(f"{label:20s}  →  Cosine distance: {dist:.4f}")




if __name__ == '__main__':
    # Put all class .pkl files in one folder (e.g., "data_structures/")
    # Pick one file to act as a test input
    data_folder = Path("data_structures")
    test_file = Path("data_structures/drone multi rotor.pkl")
    # test_file = Path("data_structures/drone fixed wing.pkl")
    # test_file = Path("data_structures/generators.pkl")
    # test_file = Path("data_structures/vehicles diesel.pkl")
    # test_file = Path("data_structures/vehicles gas.pkl")
    freq_start = 60
    freq_stop = 3000

    fingerprints = build_class_fingerprints(data_folder, freq_start, freq_stop)
    match_test_file(test_file, fingerprints, freq_start, freq_stop)
    # plot_fingerprints(fingerprints, freq_start, freq_stop)

    with open("fingerprints.pkl", "rb") as f:
        fingerprints = pickle.load(f)

    print(fingerprints.keys())
    # Output: dict_keys(['drone multi rotor', 'vehicles diesel', ...])
