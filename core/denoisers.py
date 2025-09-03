from typing import Literal
import torch as th


class DeepDenoiser:
    """
    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., 2022), or EDM in short.
    Implementing the preconditioning logic around the neural network.

    Parameters:
    ----------
    - device: device on which to run the denoiser.
    - net: the raw neural network (F in EDM).
    - target: either 'x', 'eps', or 'v'. 'v' is the one used in EDM, and is equivalent
        to velocity prediction (Salimans and Ho, 2021).
    - weights: optional path to pretained weights.
    """

    def __init__(self, net: th.nn.Module, target: Literal["x", "eps", "v"]):
        self.net = net
        self.target = target

    def input_scaling(self, sigma: th.Tensor) -> th.Tensor:
        "c_in in EDM paper."
        # corresponds to VP scaling
        return 1 / th.sqrt(sigma**2 + 1)

    def output_scaling(self, sigma: th.Tensor) -> th.Tensor:
        "c_out in EDM paper."
        if self.target == "x":
            return th.ones_like(sigma)
        if self.target == "eps":
            return -sigma
        if self.target == "v":
            return sigma / th.sqrt(sigma**2 + 1)
        else:
            raise NotImplemented

    def skip_scaling(self, sigma: th.Tensor) -> th.Tensor:
        "c_skip in EDM paper."
        if self.target == "x":
            return th.zeros_like(sigma)
        if self.target == "eps":
            return th.ones_like(sigma)
        if self.target == "v":
            return 1 / (sigma**2 + 1)
        else:
            raise NotImplemented

    def noise_mapping(self, sigma: th.Tensor) -> th.Tensor:
        "c_noise in EDM paper."
        # chosen empirically
        return th.log(sigma)

    def __call__(self, y: th.Tensor, sigma: th.Tensor):
        B = y.shape[0]
        sigma = y.new_ones((B,)) * sigma
        dims = y.dim() - 1
        scaled_y = self.input_scaling(sigma).reshape(B, *[1] * dims) * y
        cond = self.noise_mapping(sigma)
        # TODO train the unet with 64 bits
        out = self.net(scaled_y.to(th.float32), cond.to(th.float32)).to(th.float64)
        out = (
            self.skip_scaling(sigma).reshape(B, *[1] * dims) * y
            + self.output_scaling(sigma).reshape(B, *[1] * dims) * out
        )
        return out
