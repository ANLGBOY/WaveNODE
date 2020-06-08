import torch
from torch import nn
from torchdiffeq import odeint_adjoint


def logabs(x): return torch.log(torch.abs(x))


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class ActNorm(nn.Module):
    def __init__(self, in_channel, log_det=True, pretrained=False):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))
        self.initialized = pretrained
        self.log_det = log_det

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).permute(1, 0, 2)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        _, _, T = x.size()

        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        log_abs = logabs(self.scale)
        log_det = torch.sum(log_abs) * T

        # Instead of < y = s * x + b > in Eq.(25), we use < y = s * (x + b) > for ease of implementation.
        return self.scale * (x + self.loc), log_det

    def reverse(self, output):
        return (output / self.scale) - self.loc


class MovingBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        return [1, -1, 1]

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)
            batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx + self._log_detgrad(x, used_var).view(x.size(0), -1).sum(1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        return x

    def _log_detgrad(self, x, used_var):
        log_detgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            log_detgrad += weight
        return log_detgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(
                name=self.__class__.__name__, **self.__dict__)
        )


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, tol=1e-5, train_T=False, solver='dopri5'):
        super().__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))

        self.odefunc = odefunc
        self.solver = solver
        self.atol = tol
        self.rtol = tol
        self.test_solver = solver
        self.test_atol = tol
        self.test_rtol = tol
        self.solver_options = {}

    def forward(self, x, c, log_det):
        _logpx = torch.zeros(x.shape[0], 1).to(x)
        states = (x, _logpx, c)

        if self.train_T:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(x)
        else:
            integration_times = torch.tensor([0.0, self.T]).to(x)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        state_t = odeint_adjoint(
            self.odefunc,
            states,
            integration_times.to(x),
            atol=[self.atol, self.atol, self.atol],
            rtol=[self.rtol, self.rtol, self.atol],
            method=self.solver,
            options=self.solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t, c = state_t[:3]
        log_det = log_det - logpz_t

        return z_t, c, log_det

    def reverse(self, z, c):
        _logpz = torch.zeros(z.shape[0], 1).to(z)
        states = (z, _logpz, c)

        if self.train_T:
            integration_times = torch.stack([torch.tensor(0.0).to(z), self.sqrt_end_time * self.sqrt_end_time]).to(z)
        else:
            integration_times = torch.tensor([0., self.T], requires_grad=False).to(z)

        integration_times = self._flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        state_t = odeint_adjoint(
            self.odefunc,
            states,
            integration_times.to(z),
            atol=self.test_atol,
            rtol=self.test_rtol,
            method=self.test_solver,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        x_t, _, c = state_t[:3]

        return x_t, c

    def num_evals(self):
        return self.odefunc._num_evals.item()

    def _flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]


class WaveNetPrior(nn.Module):
    def __init__(self, n_in_channels, n_mel_channels, d_i, n_layers, n_channels, kernel_size=3):
        super().__init__()
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
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
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

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect):
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset+2*self.n_channels, :],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts

        return self.end(output)
