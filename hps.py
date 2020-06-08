class Hyperparameters():
    def __init__(self, args):
        self.d_i = args.d_i
        self.n_layer_wvn = args.n_layer_wvn
        self.n_channel_wvn = args.n_channel_wvn
        self.n_block = args.n_block
        self.scale = args.scale
        self.scale_init = args.scale_init
        self.split_period = args.split_period
        self.T = args. T
        self.tol = args.tol
        self.norm = args.norm
        self.pretrained = True if args.load_step > 0 else False
