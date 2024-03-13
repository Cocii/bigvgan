python train.py \
--config /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/config.json \
--input_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_fake/audios \
--input_mels_dir /workspace/BigVGAN/BigVGAN/LibriTTS_fake/mels \
--input_training_file /workspace/BigVGAN/BigVGAN/LibriTTS_fake/train_remove_short.txt \
--input_validation_file /workspace/BigVGAN/BigVGAN/LibriTTS_fake/validation.txt \
--list_input_unseen_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_fake/audios \
--list_input_unseen_validation_file /workspace/BigVGAN/BigVGAN/LibriTTS_fake/unseen.txt \
--checkpoint_path /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band \
--fine_tuning True 