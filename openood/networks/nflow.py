import normflows as nf
import torch
from normflows.flows import Flow


class ClampedMLP(nf.nets.MLP):
    def __init__(self, clamp: float, method: str = 'HARD', **kwargs):
        super().__init__(**kwargs)
        self.clamp = clamp
        clamp_activations = {
            'HARD': lambda u: torch.clamp(u, min=-clamp, max=clamp),
            'TANH': lambda u: clamp * torch.tanh(u / clamp),
            'SIGMOID': lambda u: clamp * 2. * (torch.sigmoid(u / clamp) - 0.5),
            'ATAN': lambda u: clamp * 0.636 * torch.atan(u / clamp)
        }
        if not method or not clamp:
            self.clamp_fn = lambda u: u
        elif method in clamp_activations:
            self.clamp_fn = clamp_activations[method]
        else:
            raise ValueError(f'Unknown clamp method: {method}')

    def forward(self, x):
        x = super().forward(x)
        x = self.clamp_fn(x)
        return x


class L2Norm(Flow):
    def __init__(self, eps=1.0e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        norms = torch.norm(z, p=2, dim=1, keepdim=True)
        z_ = z / (norms + self.eps)
        log_det = torch.sum(torch.log(norms + self.eps))
        return z_, log_det


def get_normalizing_flow(network_config):
    normalize_input = network_config.normalize_input
    latent_size = network_config.latent_size
    hidden_size = network_config.hidden_size
    clamp_value = network_config.clamp_value
    clamp_method = network_config.clamp_method
    clamp_t = network_config.clamp_t
    n_flows = network_config.n_flows

    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(n_flows):
        s = ClampedMLP(clamp=clamp_value,
                       method=clamp_method,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        t = nf.nets.MLP(layers=[latent_size, hidden_size, latent_size],
                        init_zeros=True) if not clamp_t else \
            ClampedMLP(clamp=clamp_value,
                       method=clamp_method,
                       layers=[latent_size, hidden_size, latent_size],
                       init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    if normalize_input:
        flows += [L2Norm()]

    q0 = nf.distributions.DiagGaussian(latent_size)
    # Note that in inverse method which is applied to the features
    # extracted from the backbone, the order of the flows is reversed.
    # ActNorm z-score normalizes (zero mean and unit variance) the input,
    # using two learnable parameters "mean" and "std" which are initialized
    # by the statistics of the first batch.
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm
