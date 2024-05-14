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
mel_folder = '/workspace/BigVGAN/BigVGAN/audios2mels/mels'

os.makedirs(mel_folder, exist_ok=True)
audio_files = [filename for filename in os.listdir(audio_folder) if filename.endswith('.wav')]

assert len(audio_files) >= 3

fig, axs = plt.subplots(3, 1, figsize=(5, 6))

for i, filename in enumerate(audio_files[:3]):
    audio_path = os.path.join(audio_folder, filename)
    y, sr = librosa.load(audio_path)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save the mel spectrogram as .npy file
    mel_npy_filename = os.path.splitext(filename)[0] + '.npy'
    mel_npy_path = os.path.join(mel_folder, mel_npy_filename)
    np.save(mel_npy_path, mel_spec_db)
    print(f'Saved {mel_npy_filename}')

    # modify the font of the x-axis, y-axis, and tick labels:
    axs[i].tick_params(axis='both', which='major', labelsize=font_size_change)
    
    for tick in axs[i].get_xticklabels():
        tick.set_fontproperties(font_prop)
    for tick in axs[i].get_yticklabels():
        tick.set_fontproperties(font_prop)

    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axs[i], cmap='viridis')

    # define color bar
    # cbar = fig.colorbar(axs[i].collections[0], ax=axs[i], format="%+2.0f dB")
    # cbar.ax.tick_params(labelsize=font_size_change)
    # for tick in cbar.ax.get_yticklabels():
    #     tick.set_fontproperties(font_prop)

    axs[i].set_ylabel('Frequency [Hz]', fontproperties=font_prop)
    
axs[0].set_title(f'Mel-frequency spectrogram of Ground-Truth', fontproperties=font_prop)
axs[1].set_title('Mel-frequency spectrogram of $\mathcal{V}$', fontproperties=font_prop)
axs[2].set_title('Mel-frequency spectrogram of $\mathcal{V}^{RT}$', fontproperties=font_prop)

axs[0].set_xlabel('', fontproperties=font_prop)
axs[1].set_xlabel('', fontproperties=font_prop)
axs[2].set_xlabel('Time [s]', fontproperties=font_prop)

fig.tight_layout()
mel_filename = 'stacked_spectrograms.pdf'
mel_path = os.path.join(mel_folder, mel_filename)
fig.savefig(mel_path, format='pdf')

plt.close(fig)

print(f'Processed {mel_filename}')