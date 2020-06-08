import torch
from torch import nn
from layers import ActNorm, MovingBatchNorm1d, CNF, WaveNetPrior
from odefunc import ODEnet, ODEfunc
from math import log, pi
from torch.distributions.normal import Normal


class SqueezeLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, c, log_det):
        B, C, T = x.size()
        S = self.scale
        squeezed_x = x.contiguous().view(B, C, T // S, S).permute(0, 1, 3, 2)
        squeezed_x = squeezed_x.contiguous().view(B, C * S, T // S)
        squeezed_c = c.contiguous().view(B, -1, T // S, S).permute(0, 1, 3, 2)
        squeezed_c = squeezed_c.contiguous().view(B, -1, T // S)

        return squeezed_x, squeezed_c, log_det

    def reverse(self, z, c):
        B, C, T = z.size()
        S = self.scale
        unsqueezed_z = z.contiguous().view(B, C // S, S, T).permute(0, 1, 3, 2)
        unsqueezed_z = unsqueezed_z.contiguous().view(B, C // S, T * S)
        unsqueezed_c = c.contiguous().view(B, -1, S, T).permute(0, 1, 3, 2)
        unsqueezed_c = unsqueezed_c.contiguous().view(B, -1, T * S)

        return unsqueezed_z, unsqueezed_c


class ActnormLayer(nn.Module):
    def __init__(self, in_channel, pretrained):
        super().__init__()
        self.actnorm = ActNorm(in_channel, pretrained=pretrained)

    def forward(self, x, c, log_det):
        z, log_det_new = self.actnorm(x)
        log_det += log_det_new

        return z, c, log_det

    def reverse(self, z, c=None):
        x = self.actnorm.reverse(z)

        return x, c


class MBNLayer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.mbnorm = MovingBatchNorm1d(in_channel)

    def forward(self, x, c, log_det):
        z, log_det = self.mbnorm(x, log_det)

        return z, c, log_det

    def reverse(self, z, c=None):
        x = self.mbnorm(z, reverse=True)

        return x, c


class DELayer(nn.Module):
    def __init__(self, in_channel, cin_channel, d_i):
        super().__init__()
        self.prior = WaveNetPrior(in_channel // 2, cin_channel // 2, d_i, 2, 256)

    def forward(self, x, c, log_det):
        z1, z2 = x.chunk(2, 1)
        c1, _ = c.chunk(2, 1)
        mean, log_sd = self.prior(z1, c1).chunk(2, 1)
        log_p = self.gaussian_log_p(z2, mean, log_sd).sum()

        return z1, c1, log_det, log_p

    def reverse(self, z, c, eps):
        c1, _ = c.chunk(2, 1)
        mean, log_sd = self.prior(z, c1).chunk(2, 1)
        q_0 = Normal(eps.new_zeros(eps.size()), eps.new_ones(eps.size()))
        eps = q_0.sample()
        z_new = self.gaussian_sample(eps, mean, log_sd)
        x = torch.cat([z, z_new], 1)

        return x, c

    def gaussian_log_p(self, x, mean, log_sd):
        return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

    def gaussian_sample(self, eps, mean, log_sd):
        return mean + torch.exp(log_sd) * eps


class NODEBlock(nn.Module):
    def __init__(self, chains):
        super().__init__()
        self.chains = nn.ModuleList(chains)

    def forward(self, x, c, log_det):
        for chain in self.chains:
            x, c, log_det = chain(x, c, log_det)
        z = x

        return z, c, log_det, 0

    def reverse(self, z, c):
        for chain in self.chains[::-1]:
            z, c = chain.reverse(z, c)
        x = z

        return x, c


class WaveNODE(nn.Module):
    def __init__(self, hps):
        super().__init__()

        in_channel = 1  # number of channels in audio
        cin_channel = 80  # number of channels in mel-spectrogram (freq. axis)

        self.blocks = nn.ModuleList()
        self.n_block = hps.n_block
        self.scale = hps.scale
        self.scale_init = hps.scale_init
        self.split_period = hps.split_period

        self.upsample = torch.nn.ConvTranspose1d(cin_channel, cin_channel, 1024, stride=256)
        self.squeeze_init = SqueezeLayer(hps.scale_init)

        in_channel = in_channel * hps.scale_init
        cin_channel = cin_channel * hps.scale_init

        for i in range(self.n_block):
            self.blocks.append(self.build_node_block(in_channel, cin_channel, hps.d_i, hps.n_layer_wvn, hps.n_channel_wvn,
                                                     hps.T, hps.tol, hps.scale, hps.norm, hps.pretrained))
            in_channel = in_channel * self.scale
            cin_channel = cin_channel * self.scale

            if (i+1) % self.split_period == 0 and i != self.n_block - 1:
                # For ease of implementation, we construct DELayer separately from NODEBlock.
                self.blocks.append(DELayer(in_channel, cin_channel, hps.d_i))
                in_channel = in_channel // 2
                cin_channel = cin_channel // 2

    def build_node_block(self, in_channel, cin_channel, d_i, n_layer_wvn, n_channel_wvn, T, tol, scale, norm, pretrained):
        def build_cnf(in_channel, cin_channel):
            diffeq = ODEnet(in_channel, cin_channel, d_i, n_layer_wvn, n_channel_wvn)
            odefunc = ODEfunc(diffeq=diffeq)
            cnf = CNF(
                odefunc=odefunc,
                train_T=False,
                T=T,
                tol=tol
            )
            return cnf

        chains = [SqueezeLayer(scale)]

        after_squeeze_size_i = in_channel * scale
        after_squeeze_size_c = cin_channel * scale

        if norm == 'actnorm':
            chains += [ActnormLayer(after_squeeze_size_i, pretrained=pretrained)]
        elif norm == 'mbnorm':
            chains += [MBNLayer(after_squeeze_size_i)]
        else:
            print('Caution: No normalization!')

        chains += [build_cnf(after_squeeze_size_i, after_squeeze_size_c)]
        node_block = NODEBlock(chains)

        return node_block

    def forward(self, x, mel):
        B, C, T = x.size()

        c = self.upsample(mel)
        if c.size(2) > x.size(2):
            c = c[:, :, :x.size(2)]

        out = x
        log_p_sum = 0
        log_det = torch.zeros([B, 1]).type_as(out)

        out, c, log_det = self.squeeze_init(out, c, log_det)
        for block in self.blocks:
            out, c, log_det, logp_new = block(out, c, log_det)
            log_p_sum += logp_new

        z = out

        log_p_sum += 0.5 * (- log(2.0 * pi) - z.pow(2)).sum()

        log_det = log_det.sum() / (B * C * T)
        log_p = log_p_sum / (B * C * T)

        return log_p, log_det

    def reverse(self, z, mel):
        c = self.upsample(mel)
        if c.size(2) > z.size(2):
            c = c[:, :, :z.size(2)]

        z_list = []
        c_list = []

        S = self.scale_init
        B, _, T = z.size()
        squeezed_z = z.view(B, -1, T // S, S).permute(0, 1, 3, 2)
        z = squeezed_z.contiguous().view(B, -1, T // S)
        squeezed_c = c.view(B, -1, T // S, S).permute(0, 1, 3, 2)
        c = squeezed_c.contiguous().view(B, -1, T // S)

        S = self.scale
        for i in range(self.n_block):
            B, _, T = z.size()
            squeezed_z = z.view(B, -1, T // S, S).permute(0, 1, 3, 2)
            z = squeezed_z.contiguous().view(B, -1, T // S)
            squeezed_c = c.view(B, -1, T // S, S).permute(0, 1, 3, 2)
            c = squeezed_c.contiguous().view(B, -1, T // S)

            if (i+1) % self.split_period == 0 and i != self.n_block-1:
                z, z_factor = z.chunk(2, 1)
                z_list.append(z_factor)
                c, c_factor = c.chunk(2, 1)
                c_list.append(c_factor)

        z_list_idx = len(z_list) - 1
        c_list_idx = len(c_list) - 1
        out = z
        for block in self.blocks[::-1]:
            if isinstance(block, DELayer):
                c = torch.cat((c, c_list[c_list_idx]), dim=1)
                out, c = block.reverse(out, c, z_list[z_list_idx])
                z_list_idx -= 1
                c_list_idx -= 1
            else:
                out, c = block.reverse(out, c)

        out, c = self.squeeze_init.reverse(out, c)
        x = out

        return x

    @staticmethod
    def remove_weightnorm(model):
        wavenode = model
        for block in wavenode.blocks:
            if isinstance(block, NODEBlock):
                chains = block.chains
                for chain in chains:
                    if isinstance(chain, CNF):
                        WaveNet = chain.odefunc.diffeq.layer
                        WaveNet.start = torch.nn.utils.remove_weight_norm(WaveNet.start)
                        WaveNet.in_layers = remove(WaveNet.in_layers)
                        WaveNet.cond_layer = torch.nn.utils.remove_weight_norm(WaveNet.cond_layer)
                        WaveNet.res_skip_layers = remove(WaveNet.res_skip_layers)

            if isinstance(block, DELayer):
                WaveNetPrior = block.prior
                WaveNetPrior.start = torch.nn.utils.remove_weight_norm(WaveNetPrior.start)
                WaveNetPrior.in_layers = remove(WaveNetPrior.in_layers)
                WaveNetPrior.cond_layer = torch.nn.utils.remove_weight_norm(WaveNetPrior.cond_layer)
                WaveNetPrior.res_skip_layers = remove(WaveNetPrior.res_skip_layers)
                    
        return wavenode


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
        
    return new_conv_list
