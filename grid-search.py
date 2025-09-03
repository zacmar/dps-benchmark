import argparse

import torch as th
import linop
import factor as ff
from pathlib import Path
import os
import optim
import dps
import pickle
from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D


device = th.device("cuda")
dtype = th.float64


gauss_kernel = linop.gaussian_kernel_1d(13, 2)[None].to(device, dtype)
indices = th.load("indices.pth").to(device)
operators = {
    "identity": linop.Id(),
    "convolution": linop.Conv1d(gauss_kernel),
    "sample": linop.Sample(indices),
}
# signal length
L = 64
proxs = {
    "identity": lambda x, y, r: optim.prox_l2(x, y, r),
    "convolution": lambda x, y, r: optim.prox_step_conv(x, gauss_kernel[0], L, y, r),
    "sample": lambda x, y, r: optim.prox_inpainting(x, y, r, indices),
}

base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
lamdas = th.logspace(-4, 4, 200, device="cuda", dtype=dtype)
D = linop.Grad1D()
L_D = 2


def _make_matrix(A):
    return (A @ th.eye(L, device=device, dtype=dtype)[:, None]).squeeze().T


Dm = _make_matrix(D)

denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(device), "eps")
betas = th.linspace(1e-4, 0.02, 1000).to(device).to(dtype)
n_samples = 10

# DPS
zeta_grid_cdps = th.linspace(0.001, 1, 40, dtype=dtype, device=device)

# DiffPIR
zeta_grid_diffpir = th.tensor([0.1, 0.3, 0.5, 0.6, 0.7], device=device, dtype=dtype)
rho_grid = th.tensor([0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5], device=device, dtype=dtype)
diffpir_grid = th.stack(th.meshgrid(zeta_grid_diffpir, rho_grid, indexing="ij"), -1)

# DPnP
eta_grid = th.logspace(-1, 4, 40)

tau = 0.99
sigma = 1 / L_D**2
mses_l1, mses_log, mses_l2, mses_cdps, mses_diffpir, mses_dpnp = [
    [] for _ in range(6)
]

def search(phi: ff.Factor, op_name: str):
    n_signals = 1000 if "student" in str(phi.path()) else 50
    factor_path = base_path / phi.path()

    denoiser.net.load_state_dict(th.load(factor_path / "model.pth"))

    # Load validation signals and simulate measurements
    signals = th.load(factor_path / "signals" / "validation" / "s.pth").to(
        device, dtype
    )[:n_signals]
    noise_var = th.load(factor_path / "measurements" / op_name / "var.pth").to(
        device, dtype
    )
    A = operators[op_name]
    # A matrix needed for L2 estimator (matrix inversion with linalg package)
    Am = _make_matrix(A)
    y = A @ signals + th.sqrt(noise_var) * th.randn_like(A @ signals)

    # For diagnostics, not used when we are confident everything is fine
    def print_energy(x, l, _):
        print(((A @ x - y) ** 2).sum() / 2 + l * (D @ x).abs().sum())

    def prox_infty(p: th.Tensor, l) -> th.Tensor:
        return p.clip(min=-l, max=l)

    # Warm start of l1 estimator, save saddle point for much fewer iterations later.
    xl1, ps = optim.pdhg1(
        A.T @ y,
        D @ A.T @ y,
        D,
        tau,
        lambda x: proxs[op_name](x, y, 1 / tau),
        sigma,
        lambda x: prox_infty(x, lamdas[0]),
        max_iter=5000,
    )
    for lamda in lamdas:

        def nabla(x):
            return A.T @ (A @ x - y) + lamda * D.T @ ((2 * D @ x) / (1 + (D @ x) ** 2))

        Lip_nabla = 1 + 2 * L_D**2 * lamda

        # For diagnostics, not used when we are confident everything is fine
        def print_log_energy(x):
            print(((A @ x - y) ** 2).sum() / 2 + lamda * th.log(1 + (D @ x) ** 2).sum())

        # L1
        xl1, ps = optim.pdhg1(
            xl1,
            ps.clamp(-lamda, lamda),
            D,
            tau,
            lambda x: proxs[op_name](x, y, 1 / tau),
            sigma,
            lambda x: prox_infty(x, lamda),
            max_iter=200,
        )
        mses_l1.append(((xl1 - signals) ** 2).mean())

        # Log
        xlog = optim.fista(xl1, nabla, lambda x: x, 1 / Lip_nabla, 200)
        mses_log.append(((xlog - signals) ** 2).mean())

        # L2
        xl2 = th.linalg.solve(Am.T @ Am + lamda * Dm.T @ Dm, (A.T @ y).squeeze().T).T[
            :, None
        ]
        mses_l2.append(((xl2 - signals) ** 2).mean())

    # Samples for the DPS algos
    ys = y.repeat(n_samples, 1, 1)

    # CDPS
    for zeta_prime in zeta_grid_cdps:

        def sampler(x0, denoiser, betas):
            return dps.cdps(x0, denoiser, betas, A, ys, "autograd", zeta_prime)

        samples = sampler(A.T @ ys, denoiser, betas)
        xdps = samples.view(n_samples, *(A.T @ y).shape).mean(0)
        mses_cdps.append(((xdps - signals) ** 2).mean())

    # DiffPIR
    prox_step = lambda x, r: proxs[op_name](x, ys, r)
    for zeta in zeta_grid_diffpir:
        for rho in rho_grid:

            def sampler(x0, denoiser, betas):
                return dps.diffpir(x0, denoiser, betas, prox_step, zeta, rho)

            samples = sampler(A.T @ ys, denoiser, betas)
            xdiffpir = samples.view(n_samples, *(A.T @ y).shape).mean(0)
            mses_diffpir.append(((xdiffpir - signals) ** 2).mean())

    # DPnP
    for eta in eta_grid:

        def sampler(x0, denoiser, betas):
            return dps.dpnp(x0, denoiser, A, ys, noise_var, betas, eta_initial=eta.item(), mode="learned")

        samples = sampler(A.T @ ys, denoiser, betas)
        xdpnp = samples.view(n_samples, *(A.T @ y).shape).mean(0)
        mses_dpnp.append(((xdpnp - signals) ** 2).mean())

    for metric, grid, name in zip(
        [mses_l1, mses_log, mses_l2, mses_cdps, mses_diffpir, mses_dpnp],
        [lamdas, lamdas, lamdas, zeta_grid_cdps, diffpir_grid, eta_grid],
        ["l1", "log", "l2", "cdps", "diffpir", "dpnp"],
    ):
        out_path = factor_path / "measurements" / op_name / "grid-search" / name
        out_path.mkdir(parents=True, exist_ok=True)
        th.save(th.tensor(metric), out_path / "mses.pth")
        th.save(grid, out_path / "grid.pth")

    # We immediately compute the test estimators here for the model based methods
    y = th.load(base_path / phi.path() / "measurements" / op_name / "m.pth")

    # L1
    lamda = lamdas[th.argmin(th.tensor(mses_l1))].item()
    xl1, ps = optim.pdhg1(
        A.T @ y,
        D @ A.T @ y,
        D,
        tau,
        lambda x: proxs[op_name](x, y, 1 / tau),
        sigma,
        lambda x: prox_infty(x, lamda),
        max_iter=5000,
    )

    # Log
    lamda = lamdas[th.argmin(th.tensor(mses_log))].item()

    def nabla(x):
        return A.T @ (A @ x - y) + lamda * D.T @ ((2 * D @ x) / (1 + (D @ x) ** 2))

    xlog = optim.fista(xl1, nabla, lambda x: x, 1 / Lip_nabla, 1000)

    # L2
    lamda = lamdas[th.argmin(th.tensor(mses_l2))].item()
    Am = _make_matrix(A)
    xl2 = th.linalg.solve(Am.T @ Am + lamda * Dm.T @ Dm, (A.T @ y).squeeze().T).T[
        :, None
    ]

    for xest, name in zip([xl1, xlog, xl2], ["l1", "log", "l2"]):
        out_path = factor_path / "measurements" / op_name / "model-based" / name
        out_path.mkdir(parents=True, exist_ok=True)
        th.save(xest, out_path / "xhat.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Posterior sampling from various inverse problems whose signals are derived from jump distributions specified in the factors.\nFor help regarding the parameters of the factors, use `{operator} {factor} -h`."
    )
    parser.add_argument(
        "operator",
        help="Forward operator.",
        choices=["identity", "convolution", "sample"],
    )
    sub = parser.add_subparsers(
        dest="factor", required=True, help="Name of the jump distribution."
    )

    p_gauss = sub.add_parser("gauss", help="Gauss factor.")
    p_gauss.add_argument("var", type=float, help="Variance.", default=0.25)
    p_gauss.set_defaults(make_factor=lambda arg: ff.Gauss(0, arg.var))

    p_laplace = sub.add_parser("laplace", help="Laplace factor.")
    p_laplace.add_argument("b", type=float, help="Scale.", default=1.0)
    p_laplace.set_defaults(make_factor=lambda arg: ff.Laplace(arg.b))

    p_student = sub.add_parser("student", help="Student's T distribution factor.")
    p_student.add_argument("df", type=float, help="Degrees of freedom.", default=1.0)
    p_student.set_defaults(make_factor=lambda arg: ff.StudentT(arg.df))

    p_bl = sub.add_parser("bernoulli-laplace", help="Bernoulli-Laplace factor.")
    p_bl.add_argument(
        "p",
        type=float,
        help='Probability of Bernoulli taking on the value "1". This is (1 - lambda) in the paper!',
        default=0.1,
    )
    p_bl.add_argument(
        "b", type=float, help="Scale of the Laplace distribution.", default=1.0
    )
    p_bl.set_defaults(make_factor=lambda arg: ff.BernoulliLaplace(arg.p, arg.b))
    args = parser.parse_args()
    phi = args.make_factor(args)

    th.manual_seed(0)

    with th.no_grad():
        search(phi, args.operator)
