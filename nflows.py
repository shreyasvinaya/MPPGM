import torch
import numpy as np
import torch.nn as nn
import tqdm
from tqdm import tqdm
from typing import Sequence, Tuple
# from deepchem.models.torch_models.normalizing_flows_pytorch import NormalizingFlow as NormFlow
"""below code to be added to deepchem"""
"""Normalizing flows for transforming probability distributions using PyTorch.
"""


class NormFlow(nn.Module):
    """Normalizing flows are widley used to perform generative models.
  This algorithm gives advantages over variational autoencoders (VAE) because
  of ease in sampling by applying invertible transformations
  (Frey, Gadepally, & Ramsundar, 2022).

  Example
  --------
  >>> import deepchem as dc
  >>> from deepchem.models.torch_models.layers import Affine
  >>> from deepchem.models.torch_models.normalizing_flows_pytorch import NormalizingFlow
  >>> import torch
  >>> from torch.distributions import MultivariateNormal
  >>> # initialize the transformation layer's parameters
  >>> dim = 2
  >>> samples = 96
  >>> transforms = [Affine(dim)]
  >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
  >>> # initialize normalizing flow model
  >>> model = NormalizingFlow(transforms, distribution, dim)
  >>> # evaluate the log_prob when applying the transformation layers
  >>> input = distribution.sample(torch.Size((samples, dim)))
  >>> len(model.log_prob(input))
  96
  >>> # evaluates the the sampling method and its log_prob
  >>> len(model.sample(samples))
  2

  """

    def __init__(self, transform: Sequence, base_distribution, dim: int,
                 **kwargs) -> None:
        """This class considers a transformation, or a composition of transformations
    functions (layers), between a base distribution and a target distribution.

    Parameters
    ----------
    transform: Sequence
      Bijective transformation/transformations which are considered the layers
      of a Normalizing Flow model.
    base_distribution: torch.Tensor
      Probability distribution to initialize the algorithm. The Multivariate Normal
      distribution is mainly used for this parameter.
    dim: int
      Value of the Nth dimension of the dataset.

    """
        super(NormFlow, self).__init__()
        self.dim = dim
        self.transforms = nn.ModuleList(transform)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.base_distribution = base_distribution

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """This method computes the probability of the inputs when
    transformation/transformations are applied.

    Parameters
    ----------
    inputs: torch.Tensor
      Tensor used to evaluate the log_prob computation of the learned
      distribution.
      shape: (samples, dim)

    Returns
    -------
    log_prob: torch.Tensor
      This tensor contains the value of the log probability computed.
      shape: (samples)

    """
        inputs = inputs.to(self.device)
        log_prob = torch.zeros(inputs.shape[0]).to(self.device)
        for biject in reversed(self.transforms):
            inputs, inverse_log_det_jacobian = biject.inverse(inputs)
            log_prob += inverse_log_det_jacobian

        log_prob += self.base_distribution.log_prob(inputs)

        return -torch.mean(log_prob)

    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a sampling from the transformed distribution.
    Besides the outputs (sampling), this method returns the logarithm of
    probability to obtain the outputs at the base distribution.

    Parameters
    ----------
    n_samples: int
      Number of samples to select from the transformed distribution

    Returns
    -------
    sample: tuple
      This tuple contains a two torch.Tensor objects. The first represents
      a sampling of the learned distribution when transformations had been
      applied. The secong torc.Tensor is the computation of log probabilities
      of the transformed distribution.
      shape: ((samples, dim), (samples))

    """
        outputs = self.base_distribution.sample((n_samples, ))
        log_prob = self.base_distribution.log_prob(outputs)

        for biject in self.transforms:
            outputs, log_det_jacobian = biject.forward(outputs)
            log_prob += log_det_jacobian

        return outputs, log_prob


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1,
                                        as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class ClampExp(nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """

    def __init__(self):
        """
        Constructor
        :param lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)


class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """

    def __init__(self, scale=1.0):
        """
        Constructor
        :param scale: Scale to apply to features
        """
        super().__init__()
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu)

    def forward(self, input):
        return input * self.scale


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self,
                 layers,
                 leaky=0.0,
                 score_scale=None,
                 output_fn=None,
                 output_scale=None,
                 init_zeros=False,
                 dropout=None,
                 device: torch.device = torch.device('cpu')):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param score_scale: Factor to apply to the scores, i.e. output before
        output_fn.
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", "relu", "tanh", or "clampexp"
        :param output_scale: Rescale outputs if output_fn is specified, i.e.
        scale * output_fn(out / scale)
        :param init_zeros: Flag, if true, weights and biases of last layer
        are initialized with zeros (helpful for deep models, see arXiv 1807.03039)
        :param dropout: Float, if specified, dropout is done before last layer;
        if None, no dropout is done
        """
        super().__init__()
        self.device = device
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
            elif output_fn == "clampexp":
                net.append(ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)


class MaskedAffineFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    Masked affine flow f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """
        Constructor
        :param b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        :param t: translation mapping, i.e. neural network, where first input dimension is batch dim,
        if None no translation is applied
        :param s: scale mapping, i.e. neural network, where first input dimension is batch dim,
        if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = lambda x: torch.zeros_like(x)
        else:
            self.add_module("s", s)

        if t is None:
            self.t = lambda x: torch.zeros_like(x)
        else:
            self.add_module("t", t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale,
                            dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum(
            (1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(
                z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) *
                           torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)
