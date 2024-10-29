import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class GaussianMixture2DDataset:
    def __init__(
        self,
        num_samples,
        n_comp=2,
        loc=torch.zeros(2, 2),
        scale=torch.ones(2, 2),
        pi=(torch.ones(2) / 2),
    ):
        samples_per_component = (pi * num_samples).long()
        self.samples = torch.cat(
            [
                torch.randn(samples_per_component[i], 2) * scale[i] + loc[i]
                for i in range(n_comp)
            ],
            0,
        )

    def __getitem__(self, idx):
        return self.samples[idx]


class _Bijection(nn.Module):
    def __init__(self):
        super(_Bijection, self).__init__()

    def forward(self, x):
        pass

    def inverse(self, z):
        pass


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


class BijectiveLinear(_Bijection):
    def __init__(self, dim):
        super(BijectiveLinear, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):  # x has shape NxD
        z = x @ self.weight + self.bias
        # det(A) == det(A.T) so there is no reason to add .T
        log_abs_det_weight = (self.weight.det().abs() + 1e-9).log()

        log_abs_det = log_abs_det_weight.expand(x.size(0))
        return z, log_abs_det  # shapes NxD, N

    def inverse(self, z):
        x = (z - self.bias) @ torch.linalg.inv(self.weight)
        return x  # NxD

    def regularization(self):
        return ((self.weight.T @ self.weight) - torch.eye(self.dim)).abs().sum()


class NormalizingFlow(nn.Module):
    """
    Base class for normalizing flow.
    """

    def __init__(self, transforms, input_dim):
        super(NormalizingFlow, self).__init__()
        self.transforms = transforms  # has to be of type nn.Sequential.

        self.register_buffer("loc", torch.zeros(input_dim))
        self.register_buffer("log_scale", torch.zeros(input_dim))
        self.base_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))

    def forward(self, x):
        """Transforms the input sample to the latent representation z.

        Args:
            x (torch.Tensor): input sample

        Returns:
            torch.Tensor: latent representation of the input sample
        """
        z = x
        for t in self.transforms:
            z, _ = t.forward(z)
        return z

    def inverse(self, z):
        """Transforms the latent representation z back to the input space.

        Args:
            z (torch.Tensor): latent representation

        Returns:
            torch.Tensor: representation in the input space
        """
        x = z
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def log_prob(self, x):
        """Calculates the log-likelihood of the given sample x (see equation (1)).

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: log-likelihood of x
        """
        if len(x.shape) < 2:
            x = x.unsqueeze(0)  # add batch dim

        N = x.shape[0]
        z = x
        log_abs_det = torch.zeros(N)
        for t in self.transforms:
            z, log_abs_deti = t.forward(z)
            log_abs_det += log_abs_deti

        log_pz = sum_except_batch(self.base_dist.log_prob(z))
        log_px = log_pz + log_abs_det
        return log_px

    def sample(self, num_samples, T=1):
        """Generates new samples from the normalizing flow.

        Args:
            num_samples (int): number of samples to generate
            T (float, optional): sampleing temperature. Defaults to 1.

        Returns:
            torch.Tensor: generated samples
        """
        z = self.base_dist.sample(torch.Size([num_samples])) * T
        x = z
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x


class SimpleNF(NormalizingFlow):
    def __init__(self, input_dim, num_steps=2):
        transforms = nn.Sequential()
        for _ in range(num_steps):
            transforms.append(BijectiveLinear(input_dim))

        super(SimpleNF, self).__init__(transforms=transforms, input_dim=input_dim)

    def gather_regularization(self):
        return sum([m.regularization() for m in self.transforms])


class SimpleTransform(nn.Module):
    def __init__(self, dim_in, dim_out, inflate_coef=1):
        super(SimpleTransform, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        internal_dim = int(dim_in * inflate_coef)

        """
        The final linear layer has a dimension of dim * 2 because we have dim parameters for log_s and dim parameters for t.
        Concatenate [log_scale, shift], it would be the same if we had two independent nets.
        It's similar to transfer learning where everything except the last layer is the same for both parameters,
        and only weights and biases for the last layer are different.
        """
        self.model = nn.Sequential(
            nn.Linear(dim_in, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, dim_out * 2),
        )

        """
        Since we want the identity matrix at the initialization of the NF, we actually want these conditions:
            1) t = 0
            2) exp(log_s) = 1 => log_s = 0
        So we can actually just set the last layer w *= 0, b *= 0.
        For stability, it might also be a good idea to set previous layers to smaller values.
        Not sure if this is a good idea.
        """
        with torch.no_grad():
            """
            for layer in model:
                if isinstance(layer, nn.Linear):
                    layer.weight *= 0.01 
                    layer.bias *= 0
            """
            last_linear_layer = self.model[-1]
            last_linear_layer.weight *= 0
            last_linear_layer.bias *= 0

    def forward(self, x):
        out = self.model(x)
        log_s, t = torch.chunk(out, dim=1, chunks=2)
        return log_s, t


class AffineCouplingLayer(_Bijection):
    def __init__(self, net):
        super(AffineCouplingLayer, self).__init__()
        self.net = net

    def forward(self, x):  # NxD
        m = self.net.dim_in
        x1, x2 = x[:, :m], x[:, m:]
        log_s, t = self.net(x1)
        s = torch.exp(log_s)
        z1 = x1
        z2 = s * x2 + t

        log_det = torch.sum(log_s, dim=1)
        z = torch.cat([z1, z2], dim=1)
        return z, log_det  # NxD , N

    def inverse(self, y):  # NxD
        m = self.net.dim_in
        y1, y2 = y[:, :m], y[:, m:]
        x1 = y1
        log_s, t = self.net(x1)
        s = torch.exp(log_s)
        x2 = (y2 - t) / (s + 1e-7)  # numerical stability

        x = torch.cat([x1, x2], dim=1)
        return x  # NxD


class SwitchSides(_Bijection):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y = torch.cat([x2, x1], dim=1)
        return y, 0.0

    def inverse(self, z):
        """
        torch.chunk will add one element to the left side if there’s an odd number of elements.
        To ensure that inverse(forward(x)) == x, we then need to manually add an extra element
        to the right side during the inverse operation.
        torch.chunk automatically gives an extra element to the left side in forward,
        but for inverse we must manually give an extra element to the right side.
        """
        dim = z.shape[1]
        split_idx = dim // 2
        z1, z2 = z[:, :split_idx], z[:, split_idx:]
        x = torch.cat([z2, z1], dim=1)
        return x


class SimpleRealNVP(NormalizingFlow):
    def __init__(self, input_dim, num_steps=2):
        transforms = nn.Sequential()
        for i in range(num_steps):
            dim_in = input_dim // 2
            transforms.append(
                AffineCouplingLayer(SimpleTransform(dim_in, input_dim - dim_in))
            )
            if i != num_steps - 1:
                transforms.append(SwitchSides())
        super(SimpleRealNVP, self).__init__(transforms, input_dim)
