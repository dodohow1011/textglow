import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class waveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(waveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, dur_loss, learning_rate, iteration):
            self.add_scalar("duration_predictor.loss", dur_loss, iteration)
            self.add_scalar("training.loss", reduced_loss, iteration)
            # self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)

    def log_alignment(self, model, mel_predict, iteration):

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        # idx = random.randint(0, enc_slf_attn.size(0) - 1)
        idx = 0
        # mel_tgt = mel_tgt.transpose(1, 2)
        mel_predict = mel_predict.transpose(1, 2)
        '''self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_tgt[idx].data.cpu().numpy().T),
            iteration)'''
        self.add_image(
            "mel_predict",
            plot_spectrogram_to_numpy(mel_predict[idx].data.cpu().numpy().T),
            iteration)
        '''self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(target[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(out_mel[idx].data.cpu().numpy()),
            iteration)'''
