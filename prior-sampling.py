import argparse
import torch as th
from pathlib import Path
import factor as ff
import os
import dps
from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D


# Handle device splitting outside with CUDA_VISIBLE_DEVICES
device = th.device("cuda")
dtype = th.float64


def main(factor: ff.Factor):
    th.manual_seed(0)
    base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
    path = base_path / factor.path()
    B, L = 1000, 64
    T = 1000
    betas = th.linspace(1e-4, 0.02, T).to(device).to(dtype)

    # def denoiser(x, sigma):
    #     return su.denoising_gibbs_sampler(
    #         x, sigma**2, factor, burn_in=10, n_samples=20
    #     ).mean(1, keepdim=True)

    denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(device), "eps")
    denoiser.net.load_state_dict(th.load(path / "model.pth"))

    x0 = th.randn((B, 1, L), device=device, dtype=dtype)
    process = th.empty((B, T, L), device=device, dtype=dtype)
    process[:, 0] = x0.squeeze()
    i = 1

    def callback(x):
        nonlocal i
        process[:, i] = x.squeeze()
        i += 1

    _ = dps.ddpm(x0, denoiser, betas, callback=callback)

    # save_dir = path / "diffusion-generated-signals" / "gibbs"
    save_dir = path / "diffusion-generated-signals" / "learned"
    save_dir.mkdir(parents=True, exist_ok=True)
    th.save(process, save_dir / f"diffusion_process_x.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Posterior sampling from various inverse problems whose signals are derived from jump distributions specified in the factors.\nFor help regarding the parameters of the factors, use `{operator} {factor} -h`."
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

    factor = args.make_factor(args)
    with th.no_grad():
        main(factor)
