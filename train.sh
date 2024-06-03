# # rawnet loss bigvgan finetune
# python train.py \
# --input_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/real_audios \
# --input_mels_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/mels \
# --input_training_file /data/dataset/vocoder/LibriTTS_pair/real_audio_list.txt \
# --input_validation_file /workspace/BigVGAN/BigVGAN/LibriTTS_pair/validation.txt \
# --list_input_unseen_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/real_audios \
# --list_input_unseen_validation_file /workspace/BigVGAN/BigVGAN/LibriTTS_pair/unseen.txt \
# --config /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/config.json \
# --checkpoint_path /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/RealFakeAduio_rawnet_finetune/ \
# --fine_tuning True


## tssd loss bigvgan finetune
python train.py \
--input_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/real_audios \
--input_mels_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/mels \
--input_training_file /data/dataset/vocoder/LibriTTS_pair/bigvgan_train_validate/train.txt \
--input_validation_file /data/dataset/vocoder/LibriTTS_pair/bigvgan_train_validate/validation.txt \
--list_input_unseen_wavs_dir /workspace/BigVGAN/BigVGAN/LibriTTS_pair/real_audios \
--list_input_unseen_validation_file /workspace/BigVGAN/BigVGAN/LibriTTS_pair/unseen.txt \
--config /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/config.json \
--checkpoint_path /workspace/BigVGAN/BigVGAN/bigvgan_22khz_80band/train_generator_rawnet_tssd_assist \
--fine_tuning True \
--validation_interval 10000 \
--checkpoint_interval 50000 \
--debug True