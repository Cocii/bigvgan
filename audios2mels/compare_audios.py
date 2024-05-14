import os
import numpy as np
from scipy.io import wavfile
import librosa

file1 = '/workspace/BigVGAN/BigVGAN/audios2mels/audios/0047_vocoder_finetuned.wav'
file2 = '/workspace/BigVGAN/BigVGAN/audios2mels/audios/0047_vocoder.wav'

sample_rate1, audio1 = wavfile.read(file1)
sample_rate2, audio2 = wavfile.read(file2)

if sample_rate1 != sample_rate2:
    print("The two audio files have different sample rates.")
    exit()


if len(audio1) != len(audio2):
    print("The two audio files have different lengths.")
    exit()


diff = np.abs(audio1 - audio2)
max_diff = np.max(diff)

if max_diff < 1e-6:
    print("The two audio files are identical.")
else:
    print("The two audio files are not identical. Maximum difference:", max_diff)


# Calculate mel spectrograms
mel_spec1 = librosa.feature.melspectrogram(y=audio1.astype(np.float32), sr=sample_rate1)
mel_spec2 = librosa.feature.melspectrogram(y=audio2.astype(np.float32), sr=sample_rate2)

# Convert to decibel scale
mel_spec1_db = librosa.power_to_db(mel_spec1, ref=np.max)
mel_spec2_db = librosa.power_to_db(mel_spec2, ref=np.max)

# Calculate the mean squared error between the two mel spectrograms
mse = np.mean((mel_spec1_db - mel_spec2_db) ** 2)

# Calculate the similarity ratio
max_possible_mse = np.mean((np.max(mel_spec1_db) - np.min(mel_spec2_db)) ** 2)
similarity_ratio = 1 - (mse / max_possible_mse)

print("Mean Squared Error between mel spectrograms:", mse)
print("Similarity Ratio between mel spectrograms:", similarity_ratio)