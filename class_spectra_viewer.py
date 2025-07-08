import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample

def compute_average_spectrum(ch, rate, nperseg):
    ch = ch - np.mean(ch)  # Remove DC offset
    if len(ch) < nperseg:
        ch = np.pad(ch, (0, nperseg - len(ch)), mode='constant')
    f, t, Zxx = stft(ch, fs=rate, nperseg=nperseg, noverlap=0,
                    window='hann', boundary=None)
    mag = np.abs(Zxx)
    avg_mag = np.mean(mag, axis=1)
    db = 20 * np.log10(np.maximum(avg_mag, 1e-12))
    return f, db

def load_spectra(folder, target_rate=48000, nperseg=48000):
    spectra = []
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith('.wav'):
            continue
        path = os.path.join(folder, file)
        rate, data = wavfile.read(path)

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        else:
            data = data.T

        if rate != target_rate:
            print(f"Resampling {file} from {rate} → {target_rate}")
            new_len = int(data.shape[1] * target_rate / rate)
            data = np.array([resample(ch, new_len) for ch in data])

        for ch in data:
            f, db = compute_average_spectrum(ch, target_rate, nperseg)
            spectra.append(db)

    return f, np.stack(spectra)

if __name__ == '__main__':
    # folder = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 FOSSN/PHASE 2/Software/Classification/Data/vehicles diesel'
    folder = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 FOSSN/PHASE 2/Software/Classification/Data/vehicles gas'
    nperseg = 48000  # MAX frequency resolution (1 Hz bins)
    freqs, all_spectra = load_spectra(folder, nperseg=nperseg)

    # Normalize: subtract max of each spectrum
    normalized = all_spectra - np.max(all_spectra, axis=1, keepdims=True)
    clipped = np.clip(normalized, -100, 0)

    # Heatmap
    plt.figure()
    plt.imshow(clipped.T, aspect='auto', origin='lower',
               extent=[0, clipped.shape[0], freqs[0], freqs[-1]],
               cmap='viridis')
    plt.colorbar(label='Relative Magnitude [dB]')
    plt.xlabel('File / Channel Index')
    plt.ylabel('Frequency [Hz]')
    plt.title('Peak-Normalized Spectra (Heatmap)')
    plt.tight_layout()
    plt.show()
