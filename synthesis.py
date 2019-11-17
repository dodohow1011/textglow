import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os

import audio
from glow import WaveGlow
import hparams as hp
from text import text_to_sequence
from dataset import FastSpeechDataset, collate_fn, DataLoader
from scipy.io.wavfile import write
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_WAV_VALUE = 32768.0

if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampling_rate = 22050

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    model = WaveGlow().cuda()
    checkpoint = torch.load('256emb12flow/TTSglow_92000')
    model.load_state_dict(checkpoint['model'].state_dict())
    model =  model.remove_weightnorm(model)

    dataset = FastSpeechDataset()
    testing_loader = DataLoader(dataset, 
                                batch_size = 1,
                                shuffle=False,
                                collate_fn=collate_fn,
                                drop_last=True,
                                num_workers=4)
    model = model.eval()

    for i, data_of_batch in enumerate(testing_loader):
        audio_tgt = data_of_batch["audios"]
        src_seq = data_of_batch["texts"]
        src_pos = data_of_batch["pos"]
        mel_tgt = data_of_batch["mels"]
        alignment_target = data_of_batch["alignment"]

        src_seq = torch.from_numpy(src_seq).long().to(device)
        src_pos = torch.from_numpy(src_pos).long().to(device)
        mel_tgt = torch.from_numpy(mel_tgt).float().to(device)
        alignment_target = torch.from_numpy(
            alignment_target).float().to(device)
        mel_max_len = mel_tgt.size(1)
        
        with torch.no_grad():
            audio = model.inference(src_seq, src_pos, mel_max_len, alignment_target, sigma=1.0, alpha=1.0)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        print (torch.mean(audio))
        audio = audio.cpu().numpy()

        audio_tgt = torch.cat(audio_tgt)
        audio_tgt = audio_tgt * MAX_WAV_VALUE
        print (torch.mean(audio_tgt))
        #print (audio_tgt)
        audio_tgt = audio_tgt.squeeze()
        audio_tgt = audio_tgt.cpu().numpy()

        audio = audio.astype('int16')
        audio_tgt = audio_tgt.astype('int16')
        audio_path = os.path.join(
            "results", "test_{}_synthesis.wav".format(i))           
        '''audio_tgt_path = os.path.join(
            "results", "{}_tgt.wav".format(i))'''

        write(audio_path, sampling_rate, audio)
        # write(audio_tgt_path, sampling_rate, audio_tgt)
 
        if i >= 10:
            break
