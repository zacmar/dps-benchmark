import torch as th
import common
import dps
import factor as ff
from pathlib import Path
import numpy as np
from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D

common.init()

th.manual_seed(0)

def save_csv(path: Path, x):
    x_ = np.concatenate((np.arange(x.shape[2])[None], x.cpu().numpy()[:, 0]), 0)
    header = "x"
    # For latex 1-based indexing is usually easier
    for i in range(1, x.shape[0] + 1):
        header += f",y{i}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, x_.T, delimiter=",", header=header, comments="")


phi = ff.BernoulliLaplace(0.1, 1.0)
betas = th.linspace(1e-4, 0.02, 1000).to(common.device, common.dtype)
burn_in_period = 100
n_samples_gibbs = 300
op_name = "convolution"
A = common.operators[op_name]
out_path = Path("postprocessing") / "figure-data" / "conditional"

path = common.base_path / phi.path() / "measurements" / op_name
ys, var = (
    th.load(path / f"{name}.pth", weights_only=True).to(common.device, common.dtype)
    for name in ["m", "var"]
)
ys = ys[:1]
save_csv(out_path / "measurements.csv", ys)

signal = th.load(common.base_path / phi.path() / "signals" / "test" / "s.pth")[:1]
print(signal.shape)
save_csv(out_path / "signal.csv", signal)

n_test = ys.shape[0]
n_samples = 50
L = (A.T @ ys).shape[2]
ys = ys.repeat(n_samples, 1, 1)
samples = th.empty_like(A.T @ ys)
betas = th.linspace(1e-4, 0.02, 1000).to(common.device, common.dtype)

batch_size = ys.shape[0]

# Setting like in posterior sampling
grid, mses = [
    th.load(path / "grid-search" / "diffpir" / f"{name}.pth")
    for name in ["grid", "mses"]
]
mses = mses[:, 0]
zeta, rho = grid[th.argmin(mses)]

def denoiser(x, sigma):
    return dps.denoising_gibbs_sampler(
        x, sigma**2, phi, burn_in=burn_in_period, n_samples=n_samples_gibbs
    ).mean(1, keepdim=True)

t = 1000

def save_intermediate(x):
    global t
    t -= 1
    save_csv(out_path / f"{t:03d}.csv", x)

factor_path = common.base_path / phi.path()

# For testing
denoiser_learned = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(common.device), "eps")
denoiser_learned.net.load_state_dict(th.load(factor_path / "model.pth"))


def sampler(x0, denoiser, betas):
    return dps.diffpir(
        x0,
        denoiser,
        betas,
        lambda x, r: common.proxs[op_name](x, ys, r),
        zeta,
        rho,
        callback=save_intermediate
    )

with th.no_grad():
    post_samples = sampler(A.T @ ys, denoiser, betas)
save_csv(out_path / "mmse_diffpir.csv", post_samples.mean(0, keepdim=True))
save_csv(out_path / "std_diffpir.csv", post_samples.std(0, keepdim=True))

# reference gibbs reconstruction
mmse_gibbs = th.load(common.base_path / phi.path() / "measurements" / op_name / "mmse" / "m.pth")
save_csv(out_path / "mmse_gibbs.csv", mmse_gibbs[:1])
var_gibbs = th.load(common.base_path / phi.path() / "measurements" / op_name / "mmse" / "var_gibbs.pth")
save_csv(out_path / "std_gibbs.csv", th.sqrt(var_gibbs[:1]))