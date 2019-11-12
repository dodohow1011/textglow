# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 22050
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Model
word_vec_dim = 256
encoder_conv1d_filter_size = 1024
encoder_n_layer = 6
encoder_head = 2
max_sep_len = 2048
encoder_output_size = 256
decoder_n_layer = 6
decoder_head = 2
decoder_conv1d_filter_size = 1024
decoder_output_size = 256
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1
n_early_every = 4
n_early_size = 2
n_flows = 12
n_layers = 8
n_channels = 256
kernel_size = 3

# FFTBlock
Head = 2
N = 6
d_model = 256

# Train
pre_target = True
n_warm_up_step = 4000
learning_rate = 1e-5
batch_size = 1
epochs = 10000
dataset_path = "dataset"
logger_path = "logger"
mel_ground_truth = "./mels"
alignment_path = "./alignments"
waveglow_path = "./model_waveglow"
checkpoint_path = "./model_new"
grad_clip_thresh = 1.0
decay_step = [200000, 500000, 1000000]
save_step = 2000
log_step = 10
clear_Time = 20

# Experiment
fp16_run = False
distributed_run = False
dist_backend = "nccl"
dist_url = "tcp:''localhost:54321"
cudnn_enabled = True
cudnn_benchmark = False
with_tensorboard = True
ignore_layers = ['embedding.weight']
seed = 1234
sigma = 1.0 
