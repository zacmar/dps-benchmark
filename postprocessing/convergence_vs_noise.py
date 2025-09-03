import os
from pathlib import Path
import torch as th
import linop
import factor as ff


device = th.device("cuda")
dtype = th.float64


def generate_signals(n_samples):
    base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
    betas = th.linspace(1e-4, 0.02, 1000).to(device).to(dtype)
    alphas = 1.0 - betas
    alphas_bar = th.cumprod(alphas, dim=0)
    sigmas = th.sqrt(1 - alphas_bar) / th.sqrt(alphas_bar)
    sigma_samples = [sigmas[0], sigmas[200], sigmas[400], sigmas[600], sigmas[800]]
    print(sigma_samples)

    factors: list[ff.Factor] = [
        ff.Gauss(0, 0.25),
        ff.Laplace(1.0),
        ff.StudentT(1.0),
        ff.BernoulliLaplace(0.1, 1.0),
    ]
    A = linop.Id()

    for sigma in sigma_samples:
        for phi in factors:
            factor_path = base_path / phi.path()
            s = th.load(
                factor_path / "signals" / "test" / "s.pth", weights_only=True
            ).to(device, dtype)[0:1]
            _, _, L = s.shape
            s = s.repeat(1000, 1, 1)

            n_test = 1000

            samples = th.empty((n_test, n_samples, L), device=s.device, dtype=dtype)

            i = 0

            def accumulate(x: th.Tensor):
                nonlocal i
                samples[:, i : i + 1] = x.clone()
                i += 1

            # Make sure noise is the same
            y = s + sigma * th.randn_like(s[0])[None]
            # MMSE estimators
            phi.sample_posterior(
                y,
                A,
                ff.Gauss(y, th.full_like(y, sigma**2)),
                n_iter=n_samples,
                callback=accumulate,
            )

            mmse_path = factor_path / "convergence-vs-noise" / f"sigma={sigma:.3f}"
            mmse_path.mkdir(parents=True, exist_ok=True)
            # save samples
            th.save(samples, mmse_path / "posterior_samples.pth")


if __name__ == "__main__":
    n_samples = 10000

    with th.no_grad():
        generate_signals(n_samples)
