

import numpy as np
import librosa
import pickle
from pathlib import Path

def process_single_file(wav_file, output_path):
    try:
        data, sr = librosa.load(str(wav_file), sr=48000, mono=False)

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        result = []
        for ch_idx, channel in enumerate(data):
            if len(channel) < 48000:
                continue

            num_chunks = len(channel) // 48000
            for i in range(num_chunks):
                chunk = channel[i * 48000:(i + 1) * 48000]
                fft_vals = np.abs(np.fft.rfft(chunk, n=48000))
                result.append({
                    "file": wav_file.name,
                    "channel": ch_idx,
                    "chunk": i + 1,
                    "fft": fft_vals
                })

        # Save this file’s result immediately
        if result:
            with open(output_path, "ab") as f:
                pickle.dump(result, f)

        return len(result)

    except Exception as e:
        print(f"Error processing {wav_file}: {repr(e)}")
        return 0

def process_audio_folder(folder_path, output_path):
    folder_path = Path(folder_path)
    wav_files = list(folder_path.rglob("*.wav"))

    # Truncate the output file if it exists
    open(output_path, 'wb').close()

    total = 0
    for wav_file in wav_files:
        total += process_single_file(wav_file, output_path)

    print(f"Saved {total} chunks to {output_path}")

if __name__ == '__main__':
    base = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 FOSSN/PHASE 2/Software/Classification/Data'
    classes = ['vehicles gas', 'vehicles diesel', 'normal', 'generators', 'drone multi rotor', 'drone fixed wing']

    for class_name in classes:
        process_audio_folder(f'{base}/{class_name}', f'data_structures/{class_name}.pkl')

