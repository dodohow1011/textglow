import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os, sys

import hparams as hp
import Audio
from text import text_to_sequence
from utils import process_text, pad_1D, pad_2D
from scipy.io.wavfile import read

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    audio = []
    text = []
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    for f in files:
        f = f.split('|')
        text.append(f[2].rstrip())
        audio.append(f[0])
    
    return audio, text

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

class FastSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self):
        # self.text = process_text(os.path.join("data", "train.txt"))
        self.audios, self.texts = files_to_list('metadata.csv')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        mel_gt_name = os.path.join(
            hp.mel_ground_truth, "ljspeech-mel-%05d.npy" % (idx+1))
        mel_gt_target = np.load(mel_gt_name)
        D = np.load(os.path.join(hp.alignment_path, str(idx)+".npy"))

        character = self.texts[idx][0:len(self.texts[idx])]
        character = np.array(text_to_sequence(
            character, hp.text_cleaners))

        filename = os.path.join('../FastSpeech/data/LJSpeech-1.1/wavs', self.audios[idx] + '.wav')
        audio, sampling_rate = load_wav_to_torch(filename)
        audio = audio / MAX_WAV_VALUE

        sample = {"audio": audio,
                  "texts": character,
                  "mel_target": mel_gt_target,
                  "D": D}

        return sample

def collate_fn(batch):
    texts = [d['texts'] for d in batch]
    mels = [d['mel_target'] for d in batch]
    audio = [d['audio'] for d in batch]

    if not hp.pre_target:

        texts, pos_padded = pad_text(texts)
        mels = pad_mel(mels)

        return {"audios": audios, "texts": texts, "pos": pos_padded, "mels": mels}
    
    else:
        alignment_target = [d["D"] for d in batch]

        texts, pos_padded = pad_text(texts)
        alignment_target = pad_alignment(alignment_target)
        mels = pad_mel(mels)

        return {"audios": audio, "texts": texts, "pos": pos_padded, "mels": mels, "alignment": alignment_target}


def pad_text(inputs):

    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=pad)
        pos_padded = np.pad(np.array([(i+1) for i in range(np.shape(x)[0])]),
                            (0, length - x.shape[0]), mode='constant', constant_values=pad)

        return x_padded, pos_padded

    max_len = max((len(x) for x in inputs))

    text_padded = np.stack([pad_data(x, max_len)[0] for x in inputs])
    pos_padded = np.stack([pad_data(x, max_len)[1] for x in inputs])

    return text_padded, pos_padded


def pad_alignment(alignment):

    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=pad)

        return x_padded

    max_len = max((len(x) for x in alignment))

    alignment_padded = np.stack([pad_data(x, max_len) for x in alignment])

    return alignment_padded



def pad_mel(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        '''x = np.pad(x, (0, max_len - np.shape(x)[0]), 
                   mode='constant', constant_values=0)'''
        pad = np.random.normal(loc=0, scale=1, size = (max_len, s))
        pad[:np.shape(x)[0], :] = x

        return pad

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output

'''def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = np.array(src_pos)

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = np.array(mel_pos)

    texts = pad_1D(texts)
    Ds = pad_1D(Ds)
    mel_targets = pad_2D(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "D": Ds,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output'''


if __name__ == "__main__":
    # Test
    dataset = FastSpeechDataset()
    training_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            # print(mel_target.size())
            # print(D.sum())
            print(cnt)
            if mel_target.size(1) == D.sum().item():
                cnt += 1

