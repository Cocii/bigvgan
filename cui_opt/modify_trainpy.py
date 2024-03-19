import os
with open("/workspace/AdaSpeech/preprocessed_data/libri_spkr/train_remove_short.txt", 'r') as f:
    lines = f.readlines()

# 160947_000064_000000|2588|{Y EH1 S sp DH EH1 R SH IY1 W AA1 Z sp F AE1 S T AH0 S L IY1 P sp AA1 N DH AH0 D AO1 G Z B AE1 K}|yes, there she was, fast asleep on the dog's back.|0
with open("/workspace/BigVGAN/BigVGAN/LibriTTS_fake/train_remove_short.txt", 'w') as f:
    for line in lines:
        parts = line.split('|')
        f.write("/workspace/BigVGAN/BigVGAN/LibriTTS_fake/audios/"+parts[1]+"_"+parts[0]+"|"+parts[3]+"\n")
