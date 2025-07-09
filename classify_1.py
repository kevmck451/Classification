import pickle
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from pathlib import Path
import matplotlib.pyplot as plt

def compute_log_fft(chunk, freq_start, freq_stop, sr):
    fft_vals = np.abs(np.fft.rfft(chunk, n=sr))
    fft_vals = fft_vals[freq_start:freq_stop]

    num_bins = fft_vals.shape[0]
    linear_freqs = np.linspace(freq_start, freq_stop, num_bins)
    log_freqs = np.geomspace(freq_start, freq_stop, num_bins)
    log_indices = np.interp(log_freqs, linear_freqs, np.arange(num_bins))

    log_fft = np.interp(log_indices, np.arange(num_bins), fft_vals)
    return log_fft

def compute_fingerprint(log_fft):
    return log_fft / (np.linalg.norm(log_fft) + 1e-6)

def compute_probabilities(test_fp, class_fingerprints):
    distances = {
        label: cosine(test_fp, fp)
        for label, fp in class_fingerprints.items()
    }
    similarities = {label: 1 - dist for label, dist in distances.items()}
    total = sum(similarities.values())
    return {label: sim / total for label, sim in similarities.items()}

if __name__ == '__main__':
    # --- CONFIG ---
    fingerprint_path = "fingerprints.pkl"
    test_wav = Path("Test_Data/angel.wav")
    # test_wav = Path("Test_Data/diesel.wav")
    # test_wav = Path("Test_Data/hex.wav")
    # test_wav = Path("Test_Data/large_MR.wav")
    # test_wav = Path("Test_Data/mavic.wav")
    freq_start = 60
    freq_stop = 3000
    sr = 48000
    # --------------

    # Load class fingerprints
    with open(fingerprint_path, "rb") as f:
        class_fingerprints = pickle.load(f)

    # Load audio
    y, _ = librosa.load(test_wav, sr=sr, mono=True)
    chunk_len = sr  # 1 second
    num_chunks = len(y) // chunk_len

    print(f"\nMatching {test_wav.name} in {num_chunks} chunks:\n")

    # for i in range(num_chunks):
    #     chunk = y[i * chunk_len: (i + 1) * chunk_len]
    #     log_fft = compute_log_fft(chunk, freq_start, freq_stop, sr)
    #     test_fp = compute_fingerprint(log_fft)
    #     probs = compute_probabilities(test_fp, class_fingerprints)
    #
    #     print(f"Chunk {i + 1:02d}:")
    #     for label, p in sorted(probs.items(), key=lambda x: -x[1]):
    #         print(f"  {label:20s} → {p:.3f}")
    #     print()

    # --- Prepare class label order and probability matrix ---
    labels = list(class_fingerprints.keys())
    all_probs = []

    for i in range(num_chunks):
        chunk = y[i * chunk_len: (i + 1) * chunk_len]
        log_fft = compute_log_fft(chunk, freq_start, freq_stop, sr)
        test_fp = compute_fingerprint(log_fft)
        probs = compute_probabilities(test_fp, class_fingerprints)
        all_probs.append([probs[label] for label in labels])

    prob_matrix = np.array(all_probs).T  # shape: (num_labels, num_chunks)

    # --- Plot heatmap with jet colormap and probability scale ---
    plt.figure(figsize=(14, 6))
    im = plt.imshow(prob_matrix, aspect='auto', origin='lower',
                    cmap='jet', vmin=0, vmax=1)

    plt.colorbar(im, label='Probability [0–1]')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xticks(ticks=np.linspace(0, num_chunks - 1, min(20, num_chunks), dtype=int))
    plt.xlabel("Chunk Index (1-second steps)")
    plt.ylabel("Class Label")
    plt.title(f"Per-chunk Class Probabilities for {test_wav.name}")
    plt.tight_layout()
    plt.show()