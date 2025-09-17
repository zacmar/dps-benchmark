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
factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.BernoulliLaplace(0.1, 1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
]
betas = th.linspace(1e-4, 0.02, 1000).to(common.device, common.dtype)
burn_in_period = 100
n_samples_gibbs = 300

def save_csv(path: Path, x):
    x_ = np.concatenate((np.arange(x.shape[2])[None], x.cpu().numpy().squeeze()), 0)
    header = "x"
    # For latex 1-based indexing is usually easier
    for i in range(1, x.shape[0] + 1):
        header += f",y{i}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, x_.T, delimiter=",", header=header, comments="")

# Samples from all distributions with gibbs denoiser
n_samples = 10
for phi in factors:
    x0 = th.randn((n_samples, 1, common.L), device=common.device, dtype=common.dtype)
    def denoiser(x, sigma):
        return dps.denoising_gibbs_sampler(
            x, sigma**2, phi, burn_in=burn_in_period, n_samples=n_samples_gibbs
        ).mean(1, keepdim=True)
    t = 1000
    out_dir = Path("./postprocessing") / "figure-data" / phi.path() / "unconditional"
    def save_intemediate(x: th.Tensor):
        global t
        t -= 1
        save_csv(out_dir / f"{t:03d}.csv", x)

    samples = dps.ddpm(x0, denoiser, betas, callback=save_intemediate)

# Samples from St(1) for the histogram comparison
phi = ff.StudentT(1.0)
factor_path = common.base_path / phi.path()
denoiser_learned = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(common.device), "eps")
denoiser_learned.net.load_state_dict(th.load(factor_path / "model.pth"))
def denoiser_gibbs(x, sigma):
    return dps.denoising_gibbs_sampler(
        x, sigma**2, phi, burn_in=burn_in_period, n_samples=n_samples_gibbs
    ).mean(1, keepdim=True)

x0 = th.randn((n_samples, 1, common.L), device=common.device, dtype=common.dtype)
with th.no_grad():
    for denoiser, name in zip([denoiser_learned, denoiser_gibbs], ["learned", "gibbs"]):
        th.manual_seed(0)
        out_dir = Path("./postprocessing") / "figure-data" / phi.path() / "unconditional" / "learned-vs-gibbs"
        samples = dps.ddpm(x0.clone(), denoiser, betas)
        save_csv(out_dir / f"{name}.csv", samples)