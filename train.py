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
import argparse
import json
import os
import sys
import torch

from multiprocessing import cpu_count

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from glow import WaveGlow, WaveGlowLoss
from dataset import FastSpeechDataset, collate_fn, DataLoader
# from alignment import get_alignment, get_tacotron2
import hparams as hp
from logger import waveglowLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow().cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def parse_batch(batch):
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return text_padded, input_lengths, mel_padded, max_len, output_lengths

def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger = waveglowLogger(os.path.join(output_directory, log_directory))
    
    return logger

def train(num_gpus, rank, group_name, output_directory, log_directory, checkpoint_path):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(hp.sigma)
    model = WaveGlow().cuda()


    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    learning_rate = hp.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if hp.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path:
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    # Get dataset
    dataset = FastSpeechDataset()

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=0)

    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if hp.with_tensorboard and rank == 0:
        logger = prepare_directories_and_logger(output_directory, log_directory)

    model = model.train()
    epoch_offset = max(0, int(iteration / len(training_loader)))
    beta = hp.batch_size
    print ("Total Epochs: {}".format(hp.epochs))
    print ("Batch Size: {}".format(hp.batch_size))
    
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hp.epochs):
        print("Epoch: {}".format(epoch))
        for i, data_of_batch in enumerate(training_loader):
            model.zero_grad()
            
            if not hp.pre_target:
                # Prepare Data
                src_seq = data_of_batch["texts"]
                src_pos = data_of_batch["pos"]
                mel_tgt = data_of_batch["mels"]

                src_seq = torch.from_numpy(src_seq).long().to(device)
                src_pos = torch.from_numpy(src_pos).long().to(device)
                mel_tgt = torch.from_numpy(mel_tgt).float().to(device)
                alignment_target = get_alignment(
                    src_seq, tacotron2).float().to(device)
                # For Data Parallel
                mel_max_len = mel_tgt.size(1)
            
            else:
                # Prepare Data
                audio = data_of_batch["audios"]
                src_seq = data_of_batch["texts"]
                src_pos = data_of_batch["pos"]
                mel_tgt = data_of_batch["mels"]
                alignment_target = data_of_batch["alignment"]

                audio = torch.cat(audio).to(device)
                src_seq = torch.from_numpy(src_seq).long().to(device)
                src_pos = torch.from_numpy(src_pos).long().to(device)
                mel_tgt = torch.from_numpy(mel_tgt).float().to(device)
                alignment_target = torch.from_numpy(
                    alignment_target).float().to(device)
                # For Data Parallel
                mel_max_len = mel_tgt.size(1)

            outputs = model(src_seq, src_pos, mel_tgt, audio, mel_max_len, alignment_target)
            _, _ , _, duration_predictor = outputs
            max_like, dur_loss = criterion(outputs, alignment_target)
            # mel_loss = criterion(outputs, alignment_target, mel_tgt)
            loss = max_like + dur_loss

            loss.backward()
            reduced_loss = loss.item()
            print ('{}:\t{:.9f}'.format(iteration, reduced_loss))

            #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)a
            '''if iteration % 8 == 0:
                optimizer.step()
                model.zero_grad()'''
            optimizer.step()

            if hp.with_tensorboard and rank == 0:
                logger.log_training(reduced_loss, dur_loss, learning_rate, iteration)

            if (iteration % hp.save_step == 0):
                if rank == 0:
                    # logger.log_alignment(model, mel_predict, iteration)
                    checkpoint_path = "{}/TTSglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    '''parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')'''
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()

    num_gpus = 1
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
    train(num_gpus, args.rank, args.group_name, args.output_directory, args.log_directory, args.checkpoint_path)
