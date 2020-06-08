import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from args import parse_args
from data import LJspeechDataset, collate_fn_synthesize
from hps import Hyperparameters
from model import WaveNODE
from utils import count_nfe, get_logger, mkdir
import librosa
import os
import time
import json

torch.backends.cudnn.benchmark = False


def load_dataset(args):
    test_dataset = LJspeechDataset(args.data_path, False, 0.1)
    synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                            num_workers=args.num_workers, pin_memory=True)

    return synth_loader


def build_model(hps):
    model = WaveNODE(hps)
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return model


def synthesize(model, temp, tol_synth, log_speed):
    start = time.time()
    n_samples = 0
    NFE_tot = 0
    cnt = 0
    for _, (x, c) in enumerate(synth_loader):
        x, c = x.to(device), c.to(device)
        q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
        z = q_0.sample()
        a = torch.FloatTensor(z.shape).uniform_(0.2, 1.0).to(device)
        z = z * a
        with torch.no_grad():
            y_gen = model.reverse(z, c).squeeze()

        n_samples += len(y_gen)
        NFE_tot += count_nfe(model)
        cnt += 1
        print('{} samples/sec \t NFE: {}'.format(n_samples/(time.time()-start), NFE_tot / cnt))
    
    state = {}
    state['tol_synth'] = tol_synth
    state['samples/sec'] = n_samples/(time.time()-start)
    state['NFE'] = NFE_tot / cnt
    log_speed.write('%s\n' % json.dumps(state))
    log_speed.flush()


def load_checkpoint(step, model):
    checkpoint_path = os.path.join(load_path, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)

    # generalized load procedure for both single-gpu and DataParallel models
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    g_epoch = checkpoint["global_epoch"]
    g_step = checkpoint["global_step"]

    return model, g_epoch, g_step


if __name__ == "__main__":
    global global_step
    global start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    log_path, load_path = mkdir(args, test=True)
    log_speed = get_logger(log_path, args.model_name, test_speed=True)
    synth_loader = load_dataset(args)
    hps = Hyperparameters(args)
    model = build_model(hps)
    model, global_epoch, global_step = load_checkpoint(args.load_step, model)
    model = WaveNODE.remove_weightnorm(model)
    model.to(device)
    model.eval()

    if args.tol_synth != args.tol:
        from model import NODEBlock
        print('change tolerance to {}'.format(args.tol_synth))
        for block in model.blocks:
            if isinstance(block, NODEBlock):
                block.chains[2].test_atol = args.tol_synth
                block.chains[2].test_rtol = args.tol_synth

    with torch.no_grad():
        synthesize(model, args.temp, args.tol_synth, log_speed)

    log_speed.close()

    print('Done!')
