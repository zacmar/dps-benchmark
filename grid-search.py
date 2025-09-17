import argparse

import torch as th
import linop
import factor as ff
from pathlib import Path
import os
import optim
import dps
from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D
import common
from typing import Callable

common.init()


base_path = common.base_path
lamdas = th.logspace(-5, 5, 300, device=common.device, dtype=common.dtype)

denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(common.device), "eps")
betas = th.linspace(1e-4, 0.02, 1000).to(common.device).to(common.dtype)
n_samples = 10

# DiffPIR
zeta_grid_diffpir = th.tensor(
    [0.3, 0.7], device=common.device, dtype=common.dtype
)
rho_grid = th.logspace(-4, 1, 20, device=common.device, dtype=common.dtype)

grids = {
    'cdps': th.logspace(-3, 1, 40, dtype=common.dtype, device=common.device),
    'diffpir': th.stack(th.meshgrid(zeta_grid_diffpir, rho_grid, indexing="ij"), -1).view(-1, 2),
    'dpnp': th.logspace(-1, 4, 40),
}

tau = 0.99
D = linop.Grad1D()
L_D = 2
sigma = 1 / L_D**2

def evaluate_sampler(A, ys, sampler):
    samples = sampler(A.T @ ys, denoiser, betas)
    means = samples.view(n_samples, *(A.T @ y).shape).mean(0)
    return ((means - signals) ** 2).mean()

def make_eval_cdps(op_name, y, zeta_prime):
    ys = y.repeat(n_samples, 1, 1)
    def sampler(x0, denoiser, betas):
        return dps.cdps(x0, denoiser, betas, A, ys, "learned", zeta_prime)
    return evaluate_sampler(A, ys, sampler)

def make_eval_diffpir(op_name, y, zeta_rho):
    ys = y.repeat(n_samples, 1, 1)
    prox_step = lambda x, r: common.proxs[op_name](x, ys, r)
    def sampler(x0, denoiser, betas):
        return dps.diffpir(x0, denoiser, betas, prox_step, *zeta_rho)
    return evaluate_sampler(A, ys, sampler)

def make_eval_dpnp(op_name, y, eta_initial):
    ys = y.repeat(n_samples, 1, 1)
    A = common.operators[op_name]
    def sampler(x0, denoiser, betas):
        return dps.dpnp(x0, denoiser, A, ys, noise_var, betas, eta_initial=eta_initial, mode="learned")
    return evaluate_sampler(A, ys, sampler)


samplers = {
    'cdps': make_eval_cdps,
    'diffpir': make_eval_diffpir,
    'dpnp': make_eval_dpnp,
}


def search(estimation_map: Callable[th.tensor, [th.Tensor]], grid: th.Tensor):
    fn_values = th.empty_like(grid)
    for i, param in enumerate(grid):
        fn_values[i] = estimation_map(param)

    return fn_values

def search_model_based(y, op_name: str):
    A = common.operators[op_name]
        # For diagnostics, not used when we are confident everything is fine
    def print_energy(x, l, _):
        print(((A @ x - y) ** 2).sum() / 2 + l * (D @ x).abs().sum())

    def prox_infty(p: th.Tensor, l) -> th.Tensor:
        return p.clip(min=-l, max=l)
    mses_l1, mses_log, mses_l2 = [], [], []
    # Warm start of l1 estimator, save saddle point for much fewer iterations later.
    xl1, ps = optim.pdhg1(
        A.T @ y,
        D @ A.T @ y,
        D,
        tau,
        lambda x: common.proxs[op_name](x, y, 1 / tau),
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
            lambda x: common.proxs[op_name](x, y, 1 / tau),
            sigma,
            lambda x: prox_infty(x, lamda),
            max_iter=2000,
        )
        mses_l1.append(((xl1 - signals) ** 2).mean())

        # Log
        xlog = optim.fista(xl1, nabla, lambda x: x, 1 / Lip_nabla, 200)
        mses_log.append(((xlog - signals) ** 2).mean())

        # L2
        rhs = A.T @ y
        xl2 = optim.cg(
            A.T @ A + lamda * D.T @ D,
            rhs,
            x0=rhs * 0,
            tol=1e-8,
            max_iter=2000,
            dims=(1, 2),
        )
        mses_l2.append(((xl2 - signals) ** 2).mean())

    for metric, name in zip([mses_l1, mses_log, mses_l2], ["l1", "log", "l2"]):
        out_path = factor_path / "measurements" / op_name / "grid-search" / name
        out_path.mkdir(parents=True, exist_ok=True)
        th.save(th.tensor(metric), out_path / "mses.pth")
        th.save(lamdas, out_path / "grid.pth")

    # We immediately compute the test estimators here for the model based methods
    y = th.load(base_path / phi.path() / "measurements" / op_name / "m.pth")

    # L1
    lamda = lamdas[th.argmin(th.tensor(mses_l1))].item()
    xl1, ps = optim.pdhg1(
        A.T @ y,
        D @ A.T @ y,
        D,
        tau,
        lambda x: common.proxs[op_name](x, y, 1 / tau),
        sigma,
        lambda x: prox_infty(x, lamda),
        max_iter=10000,
    )

    # Log
    lamda = lamdas[th.argmin(th.tensor(mses_log))].item()

    def nabla(x):
        return A.T @ (A @ x - y) + lamda * D.T @ ((2 * D @ x) / (1 + (D @ x) ** 2))

    xlog = optim.fista(xl1, nabla, lambda x: x, 1 / Lip_nabla, 1000)

    # L2
    lamda = lamdas[th.argmin(th.tensor(mses_l2))].item()
    rhs = A.T @ y
    xl2 = optim.cg(
        A.T @ A + lamda * D.T @ D, rhs, x0=rhs * 0, tol=1e-8, max_iter=2000, dims=(1, 2)
    )

    for xest, name in zip([xl1, xlog, xl2], ["l1", "log", "l2"]):
        out_path = factor_path / "measurements" / op_name / "model-based" / name
        out_path.mkdir(parents=True, exist_ok=True)
        th.save(xest, out_path / "xhat.pth")


if __name__ == "__main__":
    parser = common.make_parser(
        blocks=("operator", "factor", "algorithm"),
        description="Grid search for hyperparameters parameters of model-based methods and DPS algos for signals with jump distributions defined by {factor} and operator {operator}.\nFor help regarding the parameters of the factors, use `{operator} {factor} -h`.",
        algorithm_choices=common.ALGO_CHOICES + ["model-based"]
    )
    args = parser.parse_args()
    phi = common.build_factor(args)
    op_name = args.operator
    th.manual_seed(0)

    factor_path = base_path / phi.path()
    denoiser.net.load_state_dict(th.load(factor_path / "model.pth"))

    # Load validation signals and simulate measurements
    signals = th.load(factor_path / "signals" / "validation" / "s.pth").to(
        common.device, common.dtype
    )
    noise_var = th.load(factor_path / "measurements" / op_name / "var.pth").to(
        common.device, common.dtype
    )
    A = common.operators[op_name]
    y = A @ signals + th.sqrt(noise_var) * th.randn_like(A @ signals)
    if args.algorithm == 'model-based':
        with th.no_grad():
            search_model_based(y, op_name)
    else:
        with th.no_grad():
            mses = search(lambda p: samplers[args.algorithm](op_name, y, p), grids[args.algorithm])
            factor_path = common.base_path / phi.path()
            out_path = factor_path / "measurements" / op_name / "grid-search" / args.algorithm
            out_path.mkdir(parents=True, exist_ok=True)
            th.save(mses, out_path / "mses.pth")
            th.save(grids[args.algorithm], out_path / "grid.pth")