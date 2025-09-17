from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D
import torch as th
import dps
import common

common.init()
th.manual_seed(0)

parser = common.make_parser(
    blocks=("operator", "factor", "algorithm", "denoiser"),
    description="Posterior sampling from various inverse problems whose signals are derived from jump distributions specified in the factors.\nFor help regarding the parameters of the factors, use `{operator} {factor} -h`.",
)
args = parser.parse_args()
factor = common.build_factor(args)
base_path = common.base_path
factor_path = base_path / factor.path()
if args.denoiser == "learned":
    denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(common.device), "eps")
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


A = common.operators[args.operator]
path = base_path / factor.path() / "measurements" / args.operator
ys, var = (
    th.load(path / f"{name}.pth", weights_only=True).to(common.device, common.dtype)
    for name in ["m", "var"]
)
n_test = ys.shape[0]
n_samples = 50
L = (A.T @ ys).shape[2]
ys = ys.repeat(n_samples, 1, 1)
samples = th.empty_like(A.T @ ys)
betas = th.linspace(1e-4, 0.02, 1000).to(common.device, common.dtype)

batch_size = ys.shape[0]


if args.algorithm == "cdps":
    grid, mses = [
        th.load(path / "grid-search" / "cdps" / f"{name}.pth")
        for name in ["grid", "mses"]
    ]
    zeta_prime = grid[th.argmin(mses)].item()

    def sampler(x0, denoiser, betas):
        return dps.cdps(
            x0, denoiser, betas, A, y, mode=args.denoiser, zeta_prime=zeta_prime
        )

elif args.algorithm == "diffpir":
    grid, mses = [
        th.load(path / "grid-search" / "diffpir" / f"{name}.pth")
        for name in ["grid", "mses"]
    ]
    mses = mses[:, 0]
    zeta, rho = grid[th.argmin(mses)]

    def sampler(x0, denoiser, betas):
        return dps.diffpir(
            x0,
            denoiser,
            betas,
            lambda x, r: common.proxs[args.operator](x, y, r),
            zeta,
            rho,
        )

else:
    grid, mses = [
        th.load(path / "grid-search" / "dpnp" / f"{name}.pth")
        for name in ["grid", "mses"]
    ]
    eta = grid[th.argmin(mses)]

    def sampler(x0, denoiser, betas):
        return dps.dpnp(
            x0, denoiser, A, y, var, betas, eta_initial=eta, mode=args.denoiser
        )


# Do the sampling
with th.no_grad():
    for i, y in enumerate(th.split(ys, batch_size)):
        post_samples = sampler(A.T @ y, denoiser, betas)
        samples[i * batch_size : (i + 1) * batch_size] = post_samples

diffusion_path = path / "dps" / args.algorithm / args.denoiser
diffusion_path.mkdir(parents=True, exist_ok=True)
th.save(samples.view(n_samples, n_test, 1, L).clone(), diffusion_path / "samples.pth")
