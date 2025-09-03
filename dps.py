import torch as th
from tqdm import tqdm
from typing import Callable
import factor as ff
import linop

import torch.nn.functional as F
import math
import optim


def denoising_gibbs_sampler_cov(
    y: th.Tensor, var: float, phi: ff.Factor, burn_in: int = 1000, n_samples: int = 2000
):
    B, _, L = y.shape
    samples = y.new_empty((B, n_samples, L))
    # TODO here we are computing a running mean for the covariance computation
    # just to throw it away and compute the mean from the samples later. this indicates
    # an improper separation...
    mean = y.new_zeros((B, 1, L))
    covariance = y.new_zeros((B, L, L))

    i = 0

    def accumulate(x: th.Tensor):
        nonlocal i, mean, covariance

        if i >= burn_in:
            samples[:, i - burn_in : i - burn_in + 1] = x.detach().clone()
            delta = x - mean
            mean += delta / (i - burn_in + 1)
            delta2 = x - mean
            outer = delta.transpose(1, 2) * delta2
            covariance += outer

        i += 1

    data_factor = ff.Gauss(y, th.full_like(y, var))
    phi.sample_posterior(
        y, linop.Id(), data_factor, n_iter=burn_in + n_samples, callback=accumulate
    )

    return samples, covariance[:, None] / (i - burn_in - 1)


# TODO come up with sensible name and sensible separation.
# still dont like that the factors themselves implement the gibbs samplers
# TODO can we speed up through warm starts?
def denoising_gibbs_sampler(
    y: th.Tensor,
    var: float,
    phi: ff.Factor,
    burn_in: int = 1000,
    n_samples: int = 2000,
):
    B, _, L = y.shape
    samples = y.new_empty((B, n_samples, L))

    i = 0

    def accumulate(x: th.Tensor):
        nonlocal i
        if i >= burn_in:
            samples[:, i - burn_in : i - burn_in + 1] = x.clone()

        i += 1

    data_factor = ff.Gauss(y, th.full_like(y, var))
    phi.sample_posterior(
        y, linop.Id(), data_factor, n_iter=burn_in + n_samples, callback=accumulate
    )

    return samples


def diffpir(
    x_init: th.Tensor,
    denoiser,
    betas: th.Tensor,
    prox_step,
    zeta: float = 0.1,
    rho_: float = 0.005,
) -> th.Tensor:
    alphas = 1.0 - betas
    alphas_bar = th.cumprod(alphas, dim=0)
    sigmas = th.sqrt(1 - alphas_bar)

    x = th.randn_like(x_init)
    x0 = x.clone()

    for t in tqdm(range(betas.shape[0] - 1, 0, -1), desc="DiffPIR"):
        rho = rho_ / (sigmas[t] / th.sqrt(alphas_bar[t])) ** 2
        x0 = denoiser(x / th.sqrt(alphas_bar[t]), sigmas[t] / th.sqrt(alphas_bar[t]))
        x0_hat = prox_step(x0, rho)
        eps_hat = 1 / th.sqrt(1 - alphas_bar[t]) * (x - th.sqrt(alphas_bar[t]) * x0_hat)
        z = th.randn_like(x)
        x = th.sqrt(alphas_bar[t - 1]) * x0_hat + th.sqrt(1 - alphas_bar[t - 1]) * (
            math.sqrt(1 - zeta) * eps_hat + math.sqrt(zeta) * z * (t > 1)
        )

    return x


def cdps(
    x_init: th.Tensor,
    denoiser,
    betas: th.Tensor,
    A,
    y,
    mode="autograd",
    zeta_prime=0.07,
):
    x = th.randn_like(x_init)
    x0 = x.clone()
    alphas = 1.0 - betas
    alphas_bar = th.cumprod(alphas, dim=0)
    sigmas = th.sqrt(1 - alphas_bar)
    ones = x.new_ones((x.shape[0],))

    for t in tqdm(range(betas.shape[0] - 1, 0, -1), desc="C-DPS"):
        if mode == "autograd":
            with th.enable_grad():
                x.requires_grad_(True)
                x0 = denoiser(
                    x / th.sqrt(alphas_bar[t]), sigmas[t] / th.sqrt(alphas_bar[t])
                )
                l2_error = 0.5 * th.sum((y - A @ x0) ** 2, dim=2, keepdim=True)
                grad_x = th.autograd.grad(
                    outputs=l2_error.view(x.shape[0]), inputs=x, grad_outputs=ones
                )[0].detach()
        else:
            samples, cov = denoiser(
                x / th.sqrt(alphas_bar[t]), sigmas[t] / th.sqrt(alphas_bar[t])
            )
            x0 = samples.mean(1, keepdim=True)
            l2_error = 0.5 * th.sum((y - A @ x0) ** 2, dim=2, keepdim=True)

            jacobian = cov / (sigmas[t] ** 2 / th.sqrt(alphas_bar[t]))
            grad = A.T @ (A @ x0 - y)
            grad_x = th.bmm(jacobian.squeeze(), grad.squeeze()[:, :, None]).squeeze()[
                :, None
            ]

        zeta = zeta_prime / th.sqrt(l2_error)
        z = th.randn_like(x) * (t > 1)
        x_temp = (
            th.sqrt(alphas[t]) * (1 - alphas_bar[t - 1]) / (1 - alphas_bar[t]) * x
            + th.sqrt(alphas_bar[t - 1]) * betas[t] / (1 - alphas_bar[t]) * x0
            + th.sqrt(betas[t]) * z
        )

        x = x_temp - zeta * grad_x

    return x


def dpnp(x_init: th.Tensor, denoiser, A, y, noise_var, betas, K=40, eta_initial: float=1., mode="gibbs"):
    alphas = 1.0 - betas
    alphas_bar = th.cumprod(alphas, dim=0)
    sigmas = th.sqrt(1 - alphas_bar)

    K_initial = K // 5
    eta_final = 0.15

    x = th.randn_like(x_init) * math.sqrt(eta_initial / 4)
    dims = (1, 2)

    def prox_sampler(x_i, eta, tol=1e-8, max_iter=100000):
        prec = A.T @ A / noise_var + linop.Id() * (1 / eta**2)
        rhs_mean = (A.T @ (y / noise_var)) + x_i / eta**2
        mean = optim.cg(prec, rhs_mean, x_i, dims=dims, max_iter=max_iter)

        u = th.randn_like(x_i)
        sigma = noise_var**0.5
        n = th.randn_like(y)
        xi = A.T @ (n / sigma) + (1.0 / eta) * u
        z = optim.cg(prec, xi, th.zeros_like(x_i), dims=dims, tol=tol, max_iter=max_iter)
        sample = mean + z
        return sample

    def etas(k):
        return eta_initial if k < K_initial else (eta_final / eta_initial) ** (
            (k - K_initial) / (K - K_initial)
        ) * eta_initial

    if mode == "gibbs":
        denoising_sampler = denoiser
    else:

        def dds_ddpm(x_init: th.Tensor, eta: float) -> th.Tensor:
            x = x_init
            # Tprime is defined as the largest t such that alphabar_t > 1 / (eta**2+1)
            # We implement this by finding the first where its less and subtracting one
            # argmax gets the first entry and doesnt work on bool
            less = alphas_bar < 1 / (eta**2 + 1)
            if th.any(less):
                Tprime = (alphas_bar < 1 / (eta**2 + 1)).int().argmax() - 1
            else:
                Tprime = len(alphas_bar) - 1
            exploded_sigmas = sigmas / th.sqrt(alphas_bar)
            for j in tqdm(range(Tprime, 0, -1), desc="DDS-DDPM", leave=False):
                sigma, sigma_next = exploded_sigmas[j], exploded_sigmas[j - 1]
                ratio = (sigma_next / sigma) ** 2

                sigma_noise = th.sqrt(sigma**2 - sigma_next**2)
                x0 = denoiser(x, sigma)
                z = th.randn_like(x)
                x = ratio * x + (1 - ratio) * x0 + sigma_noise * z * (j > 1)

            return x

        denoising_sampler = dds_ddpm

    for k in tqdm(range(K), desc="DPnP"):
        eta = etas(k)
        x_pcs = prox_sampler(x, eta)
        x = denoising_sampler(x_pcs, eta)

    return x


def ddpm(
    x_init: th.Tensor,
    denoiser,
    betas: th.Tensor,
    callback: Callable[[th.Tensor], None] = lambda _: None,
):

    x = th.randn_like(x_init)
    x0 = x.clone()

    alphas = 1.0 - betas
    alphas_bar = th.cumprod(alphas, dim=0)
    sigmas = th.sqrt(1 - alphas_bar)

    for t in tqdm(range(betas.shape[0] - 1, 0, -1)):
        x0 = denoiser(x / th.sqrt(alphas_bar[t]), sigmas[t] / th.sqrt(alphas_bar[t]))
        z = th.randn_like(x)
        eps = th.sqrt(alphas_bar[t]) * x0 - x
        x = (x + (1 - alphas[t]) * eps / (sigmas[t] ** 2)) / th.sqrt(
            alphas[t]
        ) + th.sqrt(1 - alphas[t]) * z * (t > 1)
        callback(x)

    return x
