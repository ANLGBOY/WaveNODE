import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of WaveNODE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='DATASETS/ljspeech/', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='WaveNODE', help='Model name')
    parser.add_argument('--load_step', type=int, default=0, help='Load step')
    parser.add_argument('--epochs', '-e', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=6500, help='Step size of optimizer scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Decay ratio of learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval during training')
    parser.add_argument('--synth_interval', type=int, default=250, help='Sampling interval during training')
    parser.add_argument('--num_sample', type=int, default=1, help='Number of samples to synthesize during training')

    parser.add_argument('--d_i', type=int, default=3, help='Base of dilation in WaveNet')
    parser.add_argument('--n_layer_wvn', type=int, default=4, help='Number of layers in WaveNet')
    parser.add_argument('--n_channel_wvn', type=int, default=128, help='Number of channels in WaveNet')
    parser.add_argument('--n_block', type=int, default=4, help='Number of flow blocks')
    parser.add_argument('--scale_init', type=int, default=4, help='Initial scale factor for input')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor for each flow stack')
    parser.add_argument('--split_period', type=int, default=2, help='Split period of latent variables (multi-scale architecture)')
    parser.add_argument('--T', type=float, default=1.0, help='Integration Interval for ODE')
    parser.add_argument('--tol', type=float, default=1e-5, help='Training tolerance for ODE')
    parser.add_argument('--norm', type=str, default='actnorm', choices=['actnorm', 'mbnorm', 'none'], help='Type of normalization')

    # for snthesize.py
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature')
    parser.add_argument('--num_synth', type=int, default=10, help='Number of samples to synthesize')
    parser.add_argument('--tol_synth', type=float, default=1e-5, help='Test tolerance for ODE')

    args = parser.parse_args()

    return args