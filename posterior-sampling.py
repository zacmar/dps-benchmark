import argparse

from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D
import torch as th
from pathlib import Path
import factor as ff
import os
import linop
import dps
import pickle
import optim


# Handle device splitting outside with CUDA_VISIBLE_DEVICES
device = th.device("cuda")
dtype = th.float64
th.manual_seed(0)

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

parser.add_argument(
    "algorithm",
    help="Posterior sampling algorithm.",
    choices=["diffpir", "cdps", "dpnp"],
)

parser.add_argument(
    "denoiser",
    help="Which denoiser to use.",
    choices=["learned", "gibbs"],
)
parser.add_argument(
    "--tuned",
    action="store_true",
    help="Use the hyperparameters found for the oracle denoising sampler. Only valid with algorithm=dpnp and denoiser=gibbs.",
)
args = parser.parse_args()
if args.tuned and not (args.algorithm == "dpnp" and args.denoiser == "gibbs"):
    parser.error("--tuned is only allowed when algorithm is 'dpnp' and denoiser is 'gibbs'")
factor = args.make_factor(args)
base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
factor_path = base_path / factor.path()
if args.denoiser == "learned":
    denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(device), "eps")
    denoiser.net.load_state_dict(th.load(factor_path / "model.pth"))
else:
    n_samples_gibbs = 300
    burn_in_period = 100
    if args.algorithm == "cdps":

        def denoiser(x, sigma):
            return dps.denoising_gibbs_sampler_cov(
                x, sigma**2, factor, burn_in=burn_in_period, n_samples=n_samples_gibbs
            )

    else:
        if args.algorithm == "dpnp":
            n_samples_gibbs = 1

        def denoiser(x, sigma):
            return dps.denoising_gibbs_sampler(
                x, sigma**2, factor, burn_in=burn_in_period, n_samples=n_samples_gibbs
            ).mean(1, keepdim=True)


# TODO this should be somewhere global
gauss_kernel = linop.gaussian_kernel_1d(13, 2)[None].to(device, dtype)
indices = th.load("indices.pth").to(device)

operators = {
    "identity": linop.Id(),
    "convolution": linop.Conv1d(gauss_kernel),
    "sample": linop.Sample(indices),
}

A = operators[args.operator]

path = base_path / factor.path() / "measurements" / args.operator
ys, var = (
    th.load(path / f"{name}.pth", weights_only=True).to(device).to(dtype)
    for name in ["m", "var"]
)

if args.algorithm == "cdps":
    if args.denoiser == "learned":
        mode = "autograd"
    else:
        mode = "gibbs"

    # Load parameters that were found by grid search
    grid, mses = [th.load(path / "grid-search" / "cdps" / f"{name}.pth") for name in ["grid", "mses"]]
    zeta_prime = grid[th.argmin(mses)].item()

    def sampler(x0, denoiser, betas):
        return dps.cdps(x0, denoiser, betas, A, y, mode=mode, zeta_prime=zeta_prime)

elif args.algorithm == "diffpir":
    # Load parameters that were found by grid search
    grid, mses = [th.load(path / "grid-search" / "diffpir" / f"{name}.pth") for name in ["grid", "mses"]]
    zeta, rho = grid[
        th.unravel_index(th.argmin(mses), grid.shape[:-1])
    ]

    def prox_step(x, rho):
        if args.operator == "identity":
            return optim.prox_l2(x, ys, rho)
        elif args.operator == "convolution":
            return optim.prox_step_conv(x, gauss_kernel[0], 64, ys, rho)
        else:
            return optim.prox_inpainting(x, ys, rho, indices)

    def sampler(x0, denoiser, betas):
        return dps.diffpir(x0, denoiser, betas, prox_step, zeta, rho)

else:
    if args.denoiser == "learned":
        mode = "learned"
    else:
        mode = "gibbs"
    if args.tuned:
        grid_path = path / "grid-search" / "dpnp" / "gibbs"
    else:
        grid_path = path / "grid-search" / "dpnp"
    grid, mses = [th.load(grid_path / f"{name}.pth") for name in ["grid", "mses"]]
    eta = grid[th.argmin(mses)]

    def sampler(x0, denoiser, betas):
        return dps.dpnp(x0, denoiser, A, ys, var, betas, eta_initial=eta, mode=mode)


n_test = ys.shape[0]
n_samples = 50
L = (A.T @ ys).shape[2]
ys = ys.repeat(n_samples, 1, 1)
samples = th.empty_like(A.T @ ys)
betas = th.linspace(1e-4, 0.02, 1000).to(device).to(dtype)

batch_size = ys.shape[0]

# Do the sampling
with th.no_grad():
    for i, y in enumerate(th.split(ys, batch_size)):
        post_samples = sampler(A.T @ ys, denoiser, betas)
        samples[i * batch_size : (i + 1) * batch_size] = post_samples

diffusion_path = path / "dps" / args.algorithm / args.denoiser
if args.tuned:
    diffusion_path = diffusion_path / "tuned"
diffusion_path.mkdir(parents=True, exist_ok=True)
th.save(samples.view(n_samples, n_test, 1, L).clone(), diffusion_path / "samples.pth")
