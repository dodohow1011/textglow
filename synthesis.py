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
from data_utils import FastSpeechDataset, collate_fn, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, num, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes.imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img_1000", "model_test_{}.jpg".format(num)))

def plot_tgt_data(data, num, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes.imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img_1000", "tgt_{}.jpg".format(num)))

def get_waveglow():
    waveglow_path = os.path.join('waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow

def synthesis_griffin_lim(mel, mel_tgt, num):
    with torch.no_grad():
        mel = mel.cpu().numpy().T
        mel_tgt = mel_tgt.cpu().numpy().T
        wav = audio.inv_mel_spectrogram(mel)
        wav_tgt = audio.inv_mel_spectrogram(mel_tgt)
        print("Wav Have Been Synthesized.")
        if not os.path.exists("results_1000"):
            os.mkdir("results_1000")
        audio.save_wav(wav_tgt, os.path.join("results_1000", "{}_tgt.wav".format(num)))
        audio.save_wav(wav, os.path.join("results_1000", "{}.wav".format(num)))

def synthesis_waveglow(mel, waveglow, num, alpha=1.0):
    wav = waveglow.infer(mel, sigma=0.666)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results_1000"):
        os.mkdir("results_1000")
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(
        "results_1000", str(num) + ".wav"))


if __name__ == "__main__":
    # Test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    model = WaveGlow().cuda()
    checkpoint = torch.load('test_1000_512channels_8b/TTSglow_180000')
    model.load_state_dict(checkpoint['model'].state_dict())

    dataset = FastSpeechDataset()
    testing_loader = DataLoader(dataset, 
                                batch_size = 1,
                                shuffle=False,
                                collate_fn=collate_fn,
                                drop_last=True,
                                num_workers=4)
    model = model.train()

    for i, data_of_batch in enumerate(testing_loader):
        if i < 990:
            continue
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
        print(alignment_target)

        mel = model.inference(src_seq, src_pos, mel_max_len, alignment_target, sigma=0.66, alpha=1.0)
        mel = mel.squeeze()
        mel_tgt = mel_tgt.squeeze()
        mel_path = os.path.join(
            "results", "{}_synthesis.pt".format(i))           
        mel_tgt_path = os.path.join(
            "results", "{}_tgt.pt".format(i))
 
        #torch.save(mel, mel_path)
        #torch.save(mel_tgt, mel_tgt_path)
        plot_data([mel.cpu().numpy().T], i)
        plot_tgt_data([mel_tgt.cpu().numpy().T], i)
        synthesis_griffin_lim(mel, mel_tgt, i)
        if i >= 1000:
            break



        ''' glow = get_waveglow()
        synthesis_waveglow(mel, glow, i, alpha=1.0)
        print("Synthesized by Waveglow.")'''
        
