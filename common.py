from pathlib import Path
import os
import torch as th
import linop, optim
import argparse
import factor as ff


device = th.device("cuda")
dtype = th.float64
L = 64
base_path = Path(os.environ.get("EXPERIMENTS_ROOT", ".")) / "optimality-framework"
indices_path = "indices.pth"

indices = None
gauss_kernel = None
indices_rfft = None
operators = {}
proxs = {}
D = None


def _rfft_mask_from_indices(indices_bool: th.Tensor, L: int) -> th.Tensor:
    m = L // 2 + 1
    mask_real = th.cat(
        (th.ones((5,), device=indices_bool.device, dtype=th.bool), indices_bool[5:33])
    )
    mask = th.cat((mask_real, mask_real)).clone()
    mask[m + 0] = False  # imag(DC)
    if L % 2 == 0:
        mask[m + (m - 1)] = False  # imag(Nyquist)
    return mask


def _build():
    global indices, gauss_kernel, indices_rfft, operators, proxs, D
    indices = th.load(indices_path).to(device)
    gauss_kernel = linop.gaussian_kernel_1d(13, 2)[None].to(device, dtype)
    indices_rfft = _rfft_mask_from_indices(indices, L)
    D = linop.Grad1D()

    operators = {
        "identity": linop.Id(),
        "convolution": linop.Conv1d(gauss_kernel),
        "sample": linop.Sample(indices),
        "fourier": linop.Sample(indices_rfft) @ linop.Rfft(n=L),
    }
    proxs = {
        "identity": lambda x, y, r: optim.prox_l2(x, y, r),
        "convolution": lambda x, y, r: optim.prox_step_conv(
            x, gauss_kernel[0], L, y, r
        ),
        "sample": lambda x, y, r: optim.prox_inpainting(x, y, r, indices),
        "fourier": lambda x, y, r: optim.prox_fourier(x, y, r, indices_rfft),
    }


def init(
    *,
    L_: int | None = None,
    device_: th.device | None = None,
    dtype_: th.dtype | None = None,
    base_: Path | None = None,
    indices_path_: str | None = None,
):
    """Optionally override defaults, then (re)build globals."""
    global L, device, dtype, BASE, indices_path
    if L_ is not None:
        L = int(L_)
    if device_ is not None:
        device = device_
    if dtype_ is not None:
        dtype = dtype_
    if base_ is not None:
        BASE = Path(base_)
    if indices_path_ is not None:
        indices_path = indices_path_
    _build()


def make_matrix(A):
    I = th.eye(L, device=device, dtype=dtype)[:, None]
    return (A @ I).squeeze().T


OPERATOR_CHOICES = ["identity", "convolution", "sample", "fourier"]
ALGO_CHOICES = ["diffpir", "cdps", "dpnp"]
DENOISER_CHOICES = ["learned", "gibbs"]


def add_operator_block(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("operator", choices=OPERATOR_CHOICES, help="Forward operator.")


def add_factor_block(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="factor", required=True, help="Jump distribution.")

    p = sub.add_parser("gauss", help="Gauss factor.")
    p.add_argument("var", type=float, default=0.25, help="Variance.")
    p.set_defaults(_factor_ctor=lambda a: ff.Gauss(0, a.var))

    p = sub.add_parser("laplace", help="Laplace factor.")
    p.add_argument("b", type=float, default=1.0, help="Scale.")
    p.set_defaults(_factor_ctor=lambda a: ff.Laplace(a.b))

    p = sub.add_parser("student", help="Student's t factor.")
    p.add_argument("df", type=float, default=1.0, help="Degrees of freedom.")
    p.set_defaults(_factor_ctor=lambda a: ff.StudentT(a.df))

    p = sub.add_parser("bernoulli-laplace", help="Bernoulli–Laplace factor.")
    p.add_argument(
        "p", type=float, default=0.1, help="P(B=1). (This is 1 - lambda in the paper.)"
    )
    p.add_argument("b", type=float, default=1.0, help="Laplace scale.")
    p.set_defaults(_factor_ctor=lambda a: ff.BernoulliLaplace(a.p, a.b))


def build_factor(args):
    if not hasattr(args, "_factor_ctor"):
        raise ValueError("No factor block parsed; cannot build factor.")
    return args._factor_ctor(args)


def add_algorithm_block(parser: argparse.ArgumentParser, choices=None) -> None:
    opts = choices or ALGO_CHOICES
    parser.add_argument(
        "algorithm", choices=opts, help="Posterior sampling algorithm."
    )


def add_denoiser_block(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "denoiser", choices=DENOISER_CHOICES, help="Which denoiser to use."
    )


def make_parser(
    *, blocks=("operator", "factor", "algorithm", "denoiser"), description="", algorithm_choices=None,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    for b in blocks:
        if b == "operator":
            add_operator_block(p)
        elif b == "factor":
            add_factor_block(p)
        elif b == "algorithm":
            add_algorithm_block(p, choices=algorithm_choices)
        elif b == "denoiser":
            add_denoiser_block(p)
        else:
            raise ValueError(f"Unknown block '{b}'")
    return p
