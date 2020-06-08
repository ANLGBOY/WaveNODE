import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from args import parse_args
from data import LJspeechDataset, collate_fn
from hps import Hyperparameters
from model import WaveNODE
from utils import get_logger, mkdir
import librosa
import os
import json

torch.backends.cudnn.benchmark = True


def load_dataset(args):
    test_dataset = LJspeechDataset(args.data_path, False, 0.1)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True)

    return test_loader


def build_model(hps):
    model = WaveNODE(hps)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of parameters:', n_params)
    
    return model


def evaluate(epoch, itr, model, log_eval):
    global global_step

    model.eval()
    running_loss = [0., 0., 0.]
    epoch_loss = 0.
    for _, (x, c) in enumerate(test_loader):
        x, c = x.to(device), c.to(device)
        log_p, logdet = model(x, c)
        log_p, logdet = torch.mean(log_p), torch.mean(logdet)
        loss = -(log_p + logdet)
        running_loss[0] += loss.item() / len(test_loader)
        running_loss[1] += log_p.item() / len(test_loader)
        running_loss[2] += logdet.item() / len(test_loader)
        epoch_loss += loss.item()
        print('NLL:', loss.item())

    state = {}
    state['Global Step'] = global_step
    state['Epoch'] = epoch
    state['eval itr'] = itr
    state['NLL, Log p(z), Log Det'] = running_loss
    log_eval.write('%s\n' % json.dumps(state))
    log_eval.flush()
    epoch_loss /= len(test_loader)
    print('Evaluation Loss : {:.4f}'.format(epoch_loss))

    return epoch_loss


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    log_path, load_path = mkdir(args, test=True)
    log_eval = get_logger(log_path, args.model_name, test_cll=True)
    test_loader = load_dataset(args)
    hps = Hyperparameters(args)
    model = build_model(hps)
    model, global_epoch, global_step = load_checkpoint(args.load_step, model)
    model = WaveNODE.remove_weightnorm(model)
    model.to(device)
    model.eval()

    N = 32
    test_epoch_avg = 0.
    for itr in range(N):
        with torch.no_grad():
            test_epoch_avg += evaluate(global_epoch, itr, model, log_eval) / N
    
    state = {}
    state['AVG Evaluation Loss'] = test_epoch_avg
    log_eval.write('%s\n' % json.dumps(state))
    log_eval.flush()

    print('AVG Evaluation Loss : {:.4f}'.format(test_epoch_avg))

    log_eval.close()