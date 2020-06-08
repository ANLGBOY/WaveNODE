import torch
from layers import CNF
import os


def actnorm_init(train_loader, model, device):
    x_seed, c_seed = next(iter(train_loader))
    x_seed, c_seed = x_seed.to(device), c_seed.to(device)
    with torch.no_grad():
        _, _ = model(x_seed, c_seed)

    print('ActNorm is initilized!')

    del x_seed, c_seed, _


def count_nfe(model):
    class AccNumEvals(object):
        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)

    return accumulator.num_evals


def get_logger(log_path, model_name, test_cll=False, test_speed=False):
    log_eval = open(os.path.join(log_path, '{}.txt'.format('eval')), 'a')

    if test_cll:
        return log_eval
    if test_speed:
        log_speed = open(os.path.join(log_path, '{}.txt'.format('speed')), 'a')
        return log_speed

    log = open(os.path.join(log_path, '{}.txt'.format(model_name)), 'a')
    log_train = open(os.path.join(log_path, '{}.txt'.format('train')), 'a')

    return log, log_train, log_eval


def mkdir(args, synthesize=False, test=False):
    set_desc = 'bc-' + str(args.batch_size) + '-d_i-' + str(args.d_i) + '-l-' + str(args.n_layer_wvn) + '-c-' \
    + str(args.n_channel_wvn) + '-b-' + str(args.n_block) + '-s-' + str(args.scale) +'-s_i-'+ str(args.scale_init) \
    + '-split-' + str(args.split_period) + '-T-' + str(args.T) + '-tol-' + str(args.tol) + '-norm-' + str(args.norm)

    if synthesize:
        if args.tol_synth != args.tol:
            sample_path = 'synthesize/' + args.model_name + '/' + set_desc +'/temp_' + str(args.temp) + '-tol-' + str(args.tol_synth)
        else:
            sample_path = 'synthesize/' + args.model_name + '/' + set_desc +'/temp_' + str(args.temp)
        save_path = 'params/' + args.model_name + '/' + set_desc
        if not os.path.isdir(sample_path):
            os.makedirs(sample_path)

        return sample_path, save_path

    if test:
        log_path = 'test_log/' + args.model_name + '/' + set_desc
        load_path = 'params/' + args.model_name + '/' + set_desc
        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        return log_path, load_path

    sample_path = 'samples/' + args.model_name + '/' + set_desc
    save_path = 'params/' + args.model_name + '/' + set_desc
    load_path = 'params/' + args.model_name + '/' + set_desc
    log_path = 'log/' + args.model_name + '/' + set_desc

    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    return sample_path, save_path, load_path, log_path
