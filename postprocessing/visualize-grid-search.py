import torch as th
from pathlib import Path
import os
import factor as ff
import numpy as np


def save_csv(path: Path, grid: th.Tensor, mses: th.Tensor):
    mses = th.log(mses)
    mses -= mses.min()
    x = th.stack((grid, mses), 1).cpu().numpy()
    header = "val,mse"
    path.mkdir(exist_ok=True, parents=True)
    np.savetxt(path / "data.csv", x, delimiter=",", header=header, comments="")


root = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"

factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.BernoulliLaplace(0.1, 1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
]
algorithms = ["cdps", "diffpir", "dpnp", "l1", "log", "l2"]
operators = ["identity", "convolution", "sample", "fourier"]

for phi in factors:
    for operator in operators:
        for algorithm in algorithms:
            load_path = (
                root
                / phi.path()
                / "measurements"
                / operator
                / "grid-search"
                / algorithm
            )
            if algorithm == "diffpir":
                grid = th.load(load_path / "grid.pth").cpu().view(2,20,2)
                mses = th.load(load_path / "mses.pth").cpu()[:, 0]
                argmin = th.unravel_index(mses.argmin(), grid.shape[:2])
                mses = mses.view(grid.shape[:2])[argmin[0]]
                grid = grid[argmin[0], :, 1]
            else:
                grid = th.load(load_path / "grid.pth").cpu()
                mses = th.load(load_path / "mses.pth").cpu()
            save_path = (
                Path("./postprocessing/figure-data")
                / phi.path()
                / "measurements"
                / operator
                / "grid-search"
                / algorithm
            )
            save_csv(save_path, grid, mses)
