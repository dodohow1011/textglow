# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from Transformer.Models import Encoder, Decoder
from Transformer.Layers import Linear, PostNet
from modules import LengthRegulator
import hparams as hp
import sys
import random


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act
    return acts

class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output, alignment_tgt, mel_tgt):
        z, log_s_list, log_det_W_list, duration_predictor_output = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        alignment_tgt.requires_grad = False
        alignment_tgt = alignment_tgt + 1
        alignment_tgt = torch.log(alignment_tgt)
        duration_predictor_loss = nn.MSELoss()(duration_predictor_output, alignment_tgt.squeeze())
        n = z.size(0)*z.size(1)*z.size(2)
        print ("{:3f}, {:3f}, {:3f}".format(duration_predictor_loss, log_s_total/n, log_det_W_total/n))
        return loss/(z.size(0)*z.size(1)*z.size(2)), duration_predictor_loss

class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W

class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)


            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:,spect_offset:spect_offset+2*self.n_channels,:],
                n_channels_tensor)

            # acts = torch.tanh(self.in_layers[i](audio)+spect[:,spect_offset:spect_offset+self.n_channels,:])

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts

        return self.end(output)

class WaveGlow(nn.Module):
    def __init__(self):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(256,
                                                 256,
                                                 1024, stride=256)
        self.n_flows = hp.n_flows
        self.n_early_every = hp.n_early_every
        self.n_early_size = hp.n_early_size

        # model parameters
        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()
        self.WN = torch.nn.ModuleList()
        self.convinv = nn.ModuleList()

        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        
        self.n_layers = hp.n_layers

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = 8
        for k in range(self.n_flows):
            '''if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels = n_remaining_channels - self.n_early_size'''
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_remaining_channels//2, hp.word_vec_dim*8, hp.n_layers, hp.n_channels, hp.kernel_size))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, src_seq, src_pos, mel, audio, mel_max_len=None, length_target=None, alpha=1.0):
        # mel: B x D x T
        # words: B x T
        if audio.size(0) >= 16000:
            max_audio_start = audio.size(0) - 16000
            audio_start = random.randint(0, max_audio_start)
            print (audio_start)
            audio = audio[audio_start:audio_start+16000]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
        audio = audio.unsqueeze(0)
        
        output_mel = []
        output_audio = []
        log_s_list = []
        log_det_W_list = []


        enc_output, _ = self.encoder(src_seq, src_pos)
        length_regulator_output, duration_predictor_output = self.length_regulator(
            enc_output,
            alpha,
            length_target,
            mel_max_len)

        mel = mel.transpose(1, 2)
        lr_output = length_regulator_output.transpose(1, 2)
        lr_output = self.upsample(lr_output)
        assert(lr_output.size(2) > audio.size(1))
        if lr_output.size(2) > audio.size(1):
            lr_output = lr_output[:, :, :audio.size(1)]

        lr_output = lr_output.unfold(2, 8, 8).permute(0, 2, 1, 3)
        lr_output = lr_output.contiguous().view(lr_output.size(0), lr_output.size(1), -1).permute(0, 2, 1)
        audio = audio.unfold(1, 8, 8).permute(0, 2, 1)
        #decoder_output = self.decoder(length_regulator_output, decoder_pos)
        #decoder_output = decoder_output.transpose(1, 2)

        for k in range(self.n_flows):

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]


            audio_output = self.WN[k]((audio_0, lr_output))
            log_s = audio_output[:, n_half:, :]
            t = audio_output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + t
            log_s_list.append(log_s)
            
            audio = torch.cat([audio_0, audio_1], 1)
        
        output_audio.append(audio)

        return torch.cat(output_audio, 1), log_s_list, log_det_W_list, duration_predictor_output

    def inference(self, src_seq, src_pos, mel_max_length=None, length_target=None, sigma=1.0, alpha=1.0):

        enc_output, enc_mask = self.encoder(src_seq, src_pos)
        length_regulator_output, decoder_pos, _ = self.length_regulator(
            enc_output,
            enc_mask,
            length_target,
            alpha,
            mel_max_length)
        
        #decoder_output = self.decoder(length_regulator_output, decoder_pos)
        #decoder_output = decoder_output.transpose(1, 2)
        lr_output = length_regulator_output.transpose(1,2)

        lr_output = self.upsample(lr_output)
        time_cutoff = self.upsample.kernel[0] - self.upsample.stride[0]
        lr_output = lr_output[:, :, :-time_cutoff]
        lr_output = lr_output.unfold(2, 8, 8).permute(0, 2, 1, 3)
        lr_output = lr_output.contiguous().view(lr_output.size(0), lr_output.size(1), -1).permute(0, 2, 1)

    

        if lr_output.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(lr_output.size(0),
                                          self.n_remaining_channels,
                                          lr_output.size(2)).normal_()
        else:
            audio = torch.cuda.FloatTensor(lr_output.size(0),
                                           self.n_remaining_channels,
                                           output.size(2)).normal_()

        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN[k]((audio_0, lr_output))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)

            audio = self.convinv[k](audio, reverse=True)

            '''if k % self.n_early_every == 0 and k > 0:
                if lr_output.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio),1)'''

        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = nn.ModuleList()
    for old_conv in conv_list:
        old_conv = nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
