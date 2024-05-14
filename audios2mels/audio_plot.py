import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_size_change = 10
font_path = '/workspace/BigVGAN/BigVGAN/audios2mels/tnr.ttf'
font_prop = FontProperties(fname=font_path)
font_prop.set_size(font_size_change)

audio_folder = '/workspace/BigVGAN/BigVGAN/audios2mels/audios'
signal_folder = '/workspace/BigVGAN/BigVGAN/audios2mels/signals'

os.makedirs(signal_folder, exist_ok=True)

audio_files = [filename for filename in os.listdir(audio_folder) if filename.endswith('.wav')]
assert len(audio_files) >= 3

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

for i, filename in enumerate(audio_files[:3]):
    audio_path = os.path.join(audio_folder, filename)
    y, sr = librosa.load(audio_path)

    # Plot the time-domain waveform
    axs[i].set_title(f'Time-domain waveform of {filename[:-4]}', fontproperties=font_prop)
    librosa.display.waveshow(y, sr=sr, ax=axs[i])
    axs[i].set_xlabel('Time [s]', fontproperties=font_prop)
    axs[i].set_ylabel('Amplitude', fontproperties=font_prop)

    # Modify the font of the x-axis, y-axis, and tick labels
    axs[i].tick_params(axis='both', which='major', labelsize=font_size_change)
    for tick in axs[i].get_xticklabels():
        tick.set_fontproperties(font_prop)
    for tick in axs[i].get_yticklabels():
        tick.set_fontproperties(font_prop)

fig.tight_layout()

signal_filename = 'waveform_comparison.pdf'
signal_path = os.path.join(signal_folder, signal_filename)
fig.savefig(signal_path, format='pdf')
plt.close(fig)

print(f'Processed {signal_filename}')