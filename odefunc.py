import torch
import torch.nn as nn


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


@torch.jit.script
def fused_add_tanh_sigmoid_multiply_with_t(input_a, input_b, t, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b + t.view(t.shape[0], t.shape[1], 1)
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act

    return acts


# WaveNet is imported from https://github.com/NVIDIA/waveglow
class WaveNet(nn.Module):
    def __init__(self, n_in_channels, n_mel_channels, d_i, n_layers, n_channels, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = d_i ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            time_layer = nn.Linear(1, 2*n_channels)
            time_layer = torch.nn.utils.weight_norm(time_layer, name='weight')
            self.time_layers.append(time_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, t):
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply_with_t(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :],
                self.time_layers[i](t),
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts

        return self.end(output)


class ODEnet(nn.Module):
    def __init__(self, x_channel, c_channel, d_i, n_layers, n_channels):
        super(ODEnet, self).__init__()
        self.layer = WaveNet(x_channel, c_channel, d_i, n_layers, n_channels)

    def forward(self, y, c, t):
        dx = self.layer(y, c, t)

        return dx


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.divergence_fn = divergence_approx

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        self._num_evals += 1

        y = states[0]
        c = states[2]

        if self._e is None:
            self._e = sample_rademacher_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
            dy = self.diffeq(y, c, t)  # dy: dy/dt at time t
            divergence = self.divergence_fn(dy, y, e=self._e).view(y.shape[0], 1)

        return tuple([dy, -divergence] + [torch.zeros_like(c).requires_grad_(True)])
