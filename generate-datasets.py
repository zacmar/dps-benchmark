import os
from pathlib import Path
import torch as th
import numpy as np
import linop
import factor as ff


device = th.device("cuda")
dtype = th.float64


def generate_signals(
    L: int,
    n_train: int,
    n_val: int,
    n_test: int,
    burn_in: int,
    n_samples: int,
    snr: float,
):
    base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
    factors: list[ff.Factor] = [
        ff.Gauss(0, 0.25),
        ff.Laplace(1.0),
        ff.BernoulliLaplace(0.1, 1.0),
        ff.StudentT(1.0),
        ff.StudentT(2.0),
        ff.StudentT(3.0),
    ]
    gauss_kernel = linop.gaussian_kernel_1d(13, 2)[None].to(device, dtype)
    indices = th.load("indices.pth").to(device)
    # Original code for constructing indices.pth:
    # remove 30% of the indices
    # indices = th.rand((64, )) > 0.4
    # th.save(indices, "indices.pth")

    operators: dict[str, linop.LinOp] = {
        "identity": linop.Id(),
        "convolution": linop.Conv1d(gauss_kernel),
        "sample": linop.Sample(indices),
    }

    # Buffers that are reused for all factors and operators
    samples = th.empty((n_test, n_samples, L), device=device, dtype=dtype)
    burn_in_samples = th.empty((n_test, burn_in, L), device=device, dtype=dtype)
    curve_mmmse = th.empty((1, n_samples), device=device, dtype=dtype)
    mean_window = 1
    curve_burn_in_mean = th.empty(
        (1, burn_in - mean_window), device=device, dtype=dtype
    )

    for phi in factors:
        print(f"Generating signals for {phi}")
        factor_path = base_path / phi.path()

        # Generate reference signals
        for name, n in zip(["train", "validation", "test"], [n_train, n_val, n_test]):
            u = phi.sample(th.Size((n, 1, L)))
            s = u.cumsum(2)
            path = factor_path / "signals" / name
            path.mkdir(parents=True, exist_ok=True)
            th.save(s, path / "s.pth")

        # Generate measurements
        for op_name, A in operators.items():
            s = th.load(
                factor_path / "signals" / "test" / "s.pth", weights_only=True
            ).to(device, dtype)

            noiseless_measurements = A @ s
            signal_power = th.median(th.mean(noiseless_measurements**2, dim=2)).item()
            var = signal_power / 10 ** (snr / 10)
            measurements = noiseless_measurements + np.sqrt(var) * th.randn_like(
                noiseless_measurements
            )

            m_path = factor_path / "measurements" / op_name
            m_path.mkdir(parents=True, exist_ok=True)
            th.save(measurements, m_path / "m.pth")
            th.save(th.full((1,), var), m_path / "var.pth")

            i = 0

            def accumulate(x: th.Tensor):
                nonlocal i
                if i > mean_window and i < burn_in:
                    burn_in_samples[:, i : i + 1] = x.clone()
                    current_mean = burn_in_samples[:, i - mean_window : i, :].mean(1)
                    previous_mean = burn_in_samples[
                        :, i - mean_window - 1 : i - 1, :
                    ].mean(1)
                    mean_diff = (current_mean - previous_mean).pow(2).mean().item()
                    curve_burn_in_mean[:, i - mean_window] = mean_diff

                if i >= burn_in:

                    samples[:, i - burn_in : i - burn_in + 1] = x.clone()
                    # curve_mmmse[:, i - burn_in] = (
                    #     (samples[:, : i - burn_in + 1, :].mean(1, keepdim=True) - s)
                    #     ** 2
                    # ).mean()

                i += 1

            # MMSE estimators
            phi.sample_posterior(
                A.T @ measurements,
                A,
                ff.Gauss(measurements, th.full_like(measurements, var)),
                n_iter=burn_in + n_samples,
                callback=accumulate,
            )

            mmse_path = m_path / "mmse"
            mmse_path.mkdir(parents=True, exist_ok=True)
            # save samples
            th.save(samples, mmse_path / "posterior_samples.pth")
            th.save(samples.mean(1, keepdim=True), mmse_path / "m.pth")
            th.save(
                samples.var(1, unbiased=False, keepdim=True),
                mmse_path / "var_gibbs.pth",
            )
            th.save(curve_mmmse, mmse_path / "curve_mmse.pth")
            th.save(curve_burn_in_mean, mmse_path / "curve_burn_in_mean.pth")


if __name__ == "__main__":
    L = 64
    n_train = 1_000_000
    n_val = 1_000
    n_test = 1_000

    burn_in = 10_000
    n_samples = 20_000

    snr = 25
    th.manual_seed(0)

    with th.no_grad():
        generate_signals(L, n_train, n_val, n_test, burn_in, n_samples, snr)
