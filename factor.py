import torch as th
import math
from tqdm import tqdm
from typing import Callable
from linop import LinOp
from pathlib import Path
from abc import ABC, abstractmethod

import logsumexpv2 as lse
import torch.distributions as td

from core import solver
import linop


class Factor(ABC):
    @abstractmethod
    def sample(self, size: th.Size) -> th.Tensor: ...

    @abstractmethod
    def path(self) -> Path: ...

    @abstractmethod
    def sample_posterior(
        self, x0, A, factor, n_iter: int = 2000, callback=lambda _: None
    ) -> th.Tensor: ...


class GLMFactor(Factor):
    @abstractmethod
    def mu_map(self, z: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def var_map(self, z: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def latent_sampler(self, u: th.Tensor) -> th.Tensor: ...

    # TODO this is hardcoded for our purposes (D reg operator),
    # but is in principle much more general.
    def sample_posterior(
        self, x0, A, factor, n_iter: int = 2000, callback=lambda _: None
    ):
        K = linop.Stack((A, linop.Grad1D()))
        D = linop.finite_difference_matrix(x0.shape[2]).to(x0.device, x0.dtype)
        A_mat = (
            (A @ th.eye(x0.shape[2], device=x0.device, dtype=x0.dtype)[:, None])
            .squeeze()
            .T
        )
        K_mat = th.cat((A_mat, D), dim=0)
        K_mat = K_mat.unsqueeze(0)
        phi = MultiFactor((factor, self))
        return glm_gibbs(K, K_mat, phi, x0, num_iters=n_iter, callback=callback)


class BernoulliLaplace(Factor):
    def __init__(self, p: float, b: float):
        super().__init__()
        self.b = b
        self.p = p
        self.bernoulli = td.Bernoulli(p)
        self.laplace = td.Laplace(0, b)

    def sample(self, size: th.Size):
        bernoulli_samples = self.bernoulli.sample(size)
        laplace_samples = self.laplace.sample(size)
        return bernoulli_samples * laplace_samples

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        log_lap = self.laplace.log_prob(x)
        is_zero = th.abs(x) < 5e-5
        log_prob = th.empty_like(x)
        log_prob[~is_zero] = (
            th.log(th.tensor([self.p], device=x.device)) + log_lap[~is_zero]
        )
        log_prob[is_zero] = th.log(
            1 - th.tensor([self.p], device=x.device, dtype=x.dtype)
        )
        return log_prob

    def path(self) -> Path:
        return Path("bernoulli-laplace") / f"p={self.p}_b={self.b}"

    # TODO also this is hardcoded for our purposes (inside of mmse_bernoulli_laplace
    # we assume a gaussian likelihood and a D reg operator), but is in principle much
    # more general (I believe).
    def sample_posterior(
        self, x0, A, factor, n_iter: int = 1000, callback=lambda _: None
    ) -> th.Tensor:
        # The factor is parametrized by the probablility of the bernoulli outputting
        # a "1", but the `bernoulli_mass` here is the probability of it outputting
        # a "0"!
        return mmse_bernoulli_laplace(
            x0.squeeze(),
            (
                (
                    A @ th.eye(x0.shape[2], device=x0.device, dtype=x0.dtype)[:, None]
                ).squeeze()
            ).T,
            factor,
            num_iters=n_iter,
            bernoulli_mass=1 - self.p,
            laplace_scale=self.b,
            callback=callback,
        )[:, None]


class StudentT(GLMFactor):
    def __init__(self, df: float):
        self.df = df
        self.dist = td.StudentT(df)

    def mu_map(self, z: th.Tensor) -> th.Tensor:
        return th.zeros_like(z)

    def var_map(self, z: th.Tensor) -> th.Tensor:
        return 1 / z

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        return self.dist.log_prob(x)

    def latent_sampler(self, u: th.Tensor) -> th.Tensor:
        return td.Gamma((self.df + 1.0) / 2.0, (self.df + (u**2)) / 2).sample()

    def sample(self, size: th.Size) -> th.Tensor:
        return self.dist.sample(size)

    def path(self) -> Path:
        return Path("student") / f"{self.df}"


class Laplace(GLMFactor):
    def __init__(self, b: float):
        self.b = b
        self.dist = td.Laplace(0, b)

    def mu_map(self, z: th.Tensor) -> th.Tensor:
        return th.zeros_like(z)

    def var_map(self, z: th.Tensor) -> th.Tensor:
        return z

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        return self.dist.log_prob(x)

    def latent_sampler(self, u: th.Tensor) -> th.Tensor:
        # Compute the GIG parameters for the latent samplers
        a_gig = 1.0 / (self.b**2) * th.ones_like(u)
        b_gig = u**2
        p_gig = 0.5 * th.ones_like(u)

        # NOTE: We get a division by zero error in the gig sampler if one of the b
        # parameters is zero (this is equivalent to one of the u components being zero)
        # TODO here its possible to dispatch to the gamma sampler, i think we discussed
        # this once?
        b_gig = th.maximum(b_gig, 1e-7 * th.ones_like(b_gig))
        return lse.gig_sampler(a_gig, b_gig, p_gig)

    def sample(self, size: th.Size) -> th.Tensor:
        return self.dist.sample(size)

    def path(self) -> Path:
        return Path("laplace") / f"{self.b}"


class Gauss(GLMFactor):
    def __init__(self, mu: float, var: float):
        self.mu = mu
        self.var = var

    def mu_map(self, z: th.Tensor) -> th.Tensor:
        return self.mu * z

    def var_map(self, z: th.Tensor) -> th.Tensor:
        return self.var * z

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        return -((x - self.mu) ** 2) / (2 * self.var)

    def latent_sampler(self, u: th.Tensor) -> th.Tensor:
        return th.ones_like(u)

    def sample(self, size: th.Size) -> th.Tensor:
        return self.mu + th.randn(size) * math.sqrt(self.var)

    def path(self) -> Path:
        return Path("gauss") / f"{self.var}"


class GMM(GLMFactor):
    def __init__(self, mus, sigmas, weights) -> None:
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights

    # TODO these are just copied over from previously, need to check if correct and
    # simplify anyways
    def mu_map(self, z: th.Tensor) -> th.Tensor:
        if z.dim() == 2:
            z = z[:, None, :, None]
            dims = z.shape
            return th.gather(
                self.mus[None, :, None, :].expand(dims[0], -1, dims[2], -1),
                dim=-1,
                index=z,
            )[:, 0, :, 0]

        # TODO: Check if this still works for the bigger cases
        z = z.unsqueeze(-1)
        dims = z.shape
        return th.gather(
            self.mus[None, :, None, None, :].expand(dims[0], -1, dims[2], dims[3], -1),
            dim=-1,
            index=z,
        ).squeeze()

    def var_map(self, z: th.Tensor) -> th.Tensor:
        if z.dim() == 2:
            z = z[:, None, :, None]
            dims = z.shape
            return th.gather(
                self.sigmas[None, :, None, :].expand(dims[0], -1, dims[2], -1),
                dim=-1,
                index=z,
            )[:, 0, :, 0]

        # TODO: Check if this still works for the bigger cases
        z = z.unsqueeze(-1)
        dims = z.shape
        return th.gather(
            self.sigmas[None, :, None, None, :].expand(
                dims[0], -1, dims[2], dims[3], -1
            ),
            dim=-1,
            index=z,
        ).squeeze()

    def latent_sampler(self, u: th.Tensor) -> th.Tensor:
        if u.dim() == 2:
            return lse.categorical_sampler(
                u[:, None, None, :], self.weights, self.mus, self.sigmas
            )[:, 0, 0, :]

        return lse.categorical_sampler(u, self.weights, self.mus, self.sigmas)


# TODO this implement the same interface as a factor, maybe we can make this nicer
class MultiFactor:
    def __init__(self, factors: tuple[Factor, ...]):
        self.factors = factors

    def latent_sampler(self, zs: tuple[th.Tensor, ...]):
        return tuple(f.latent_sampler(z) for f, z in zip(self.factors, zs))

    def var_map(self, zs: tuple[th.Tensor, ...]):
        assert len(zs) == len(self.factors)
        return tuple(f.var_map(z) for f, z in zip(self.factors, zs))

    def mu_map(self, zs: tuple[th.Tensor, ...]):
        assert len(zs) == len(self.factors)
        return tuple(f.mu_map(z) for f, z in zip(self.factors, zs))


# TODO For the time being im passing both the operator and the matrix....
# this is very ugly, but the implementation of the latent samplers and everytthing is
# just so much nicer with the operator...
def glm_gibbs(
    K: LinOp,
    Kmat: th.Tensor,
    phi: GLMFactor | MultiFactor,
    x_0: th.Tensor,
    gaussian_sampler: Callable[
        [LinOp, th.Tensor, th.Tensor, th.Tensor], th.Tensor
    ] = solver.cholesky_map,
    num_iters=100,
    callback=lambda *_: None,
) -> th.Tensor:

    x = x_0.clone().detach()

    for _ in tqdm(range(num_iters), desc="Gibbs", leave=False):
        z = phi.latent_sampler(K @ x)
        mu = phi.mu_map(z)
        var = phi.var_map(z)
        x = gaussian_sampler(Kmat, mu, var, x)

        callback(x)

    return x


def mmse_bernoulli_laplace(
    x0: th.Tensor,
    H: th.Tensor,
    factor,
    num_iters: int,
    bernoulli_mass: float,
    laplace_scale: float,
    callback: Callable[[th.Tensor], None] = lambda _: None,
) -> th.Tensor:
    y = factor.mu.squeeze()
    noise_var = factor.var.mean().item()
    D = linop.finite_difference_matrix(x0.shape[1]).to(y.device, y.dtype)
    Dinv = th.linalg.inv(D)
    A = H @ Dinv
    w = th.zeros_like(x0)
    v = th.bernoulli(th.full_like(x0, 0.5)).to(th.bool)
    u = th.ones_like(x0.squeeze())

    exp = th.distributions.Exponential(
        rate=th.tensor(1 / (2 * laplace_scale**2), device=x0.device, dtype=x0.dtype)
    )

    # Arguments to the GIG sampler that remain constant throughout the iterations
    a_gig = th.full_like(u, 1 / laplace_scale**2)
    p_gig = th.full_like(u, 0.5)

    # Clamp min is needed here for bernoulli-laplace where some jumps are really 0.
    # However, those jumps have no effect since the bernoulli will fill the values
    def sample_w(v, u):
        return th.where(
            v,
            lse.gig_sampler(a_gig, th.clamp_min(u**2, 1e-7), p_gig),
            exp.sample(v.shape),
        )

    var = th.tensor(noise_var, device=y.device, dtype=y.dtype)
    p = th.tensor(bernoulli_mass, device=y.device, dtype=y.dtype)
    for _ in tqdm(range(num_iters), desc="Gibbs", leave=False):
        w = sample_w(v, u)
        v = sample_v(v, w, y, A, var, p)
        u = sample_u(v, w, y, A, var)

        callback((u @ Dinv.T)[:, None])

    return u @ Dinv.T


def build_Binv(
    v: th.Tensor, w: th.Tensor, A: th.Tensor, noise_var: th.Tensor
) -> th.Tensor:
    B, N, _ = v.shape[0], *A.shape
    VW = v * w
    Aw = A[None] * VW[:, None]
    eye = noise_var * th.eye(N, dtype=A.dtype, device=A.device)
    Bmat = eye.expand(B, N, N) + Aw @ A.T
    return th.cholesky_inverse(th.linalg.cholesky(Bmat))


@th.compile(fullgraph=True)
def _gibbs_flip(
    vk_: th.Tensor,
    wk: th.Tensor,
    y: th.Tensor,
    ak: th.Tensor,
    Binv: th.Tensor,
    lam: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    vk = vk_.clone()
    akB = Binv @ ak
    tau = (ak * akB).sum(1)
    beta = (y * akB).sum(1)

    # 0 → 1 proposal
    logodds_add = (
        th.log1p(-lam)
        - th.log(lam)
        - 0.5 * th.log1p(wk * tau)
        + 0.5 * wk * beta.pow(2) / (1 + wk * tau)
    )
    add_hit = (th.rand_like(tau) < th.sigmoid(logodds_add)) & (vk == 0)
    f = -wk / (1 + wk * tau)
    Binv_new = Binv + (f * add_hit).view(-1, 1, 1) * (
        akB.unsqueeze(2) @ akB.unsqueeze(1)
    )
    vk[add_hit] = 1

    # 1 → 0 proposal
    logodds_rem = (
        th.log(lam)
        - th.log1p(-lam)
        - 0.5 * th.log1p(-wk * tau)
        - 0.5 * wk * beta.pow(2) / (1 - wk * tau)
    )
    rem_hit = (th.rand_like(tau) < th.sigmoid(logodds_rem)) & (vk == 1)
    f = wk / (1 - wk * tau)
    Binv_new += (f * rem_hit).view(-1, 1, 1) * (akB.unsqueeze(2) @ akB.unsqueeze(1))
    vk[rem_hit] = 0

    return vk, Binv_new


def sample_v(
    v: th.Tensor,
    w: th.Tensor,
    y: th.Tensor,
    A: th.Tensor,
    var: th.Tensor,
    p: th.Tensor,
) -> th.Tensor:
    Binv = build_Binv(v, w, A, var)

    for i, (vi, wi, ai) in enumerate(zip(v.T, w.T, A.T)):
        vk, Binv = _gibbs_flip(vi, wi, y, ai, Binv, p)
        v[:, i].copy_(vk)

    return v.bool()


def sample_u(
    mask: th.Tensor, w: th.Tensor, y: th.Tensor, A: th.Tensor, noise_var: th.Tensor
) -> th.Tensor:
    _, K = mask.shape
    dtype, device = A.dtype, A.device

    mask_f = mask.to(dtype)

    A_masked = A.unsqueeze(0) * mask_f.unsqueeze(1)
    P = th.matmul(A_masked.transpose(1, 2), A_masked) / noise_var
    P += th.diag_embed(mask_f / (w + 1e-12))

    eps = 1e-6 if dtype == th.float32 else 1e-12
    P += th.eye(K, dtype=dtype, device=device) * eps

    R = th.linalg.cholesky(P).transpose(-1, -2)
    Aty = th.matmul(A_masked.transpose(1, 2), y.unsqueeze(2)).squeeze(2)
    Rm = th.linalg.solve(R.transpose(-1, -2), Aty / noise_var)
    z = th.randn_like(Rm)
    u = th.linalg.solve(R, Rm + z)
    return u * mask_f
