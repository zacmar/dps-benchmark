import torch as th
from pathlib import Path
import numpy as np
import factor as ff
import os


def save_csv(path: Path, x):
    x_ = np.concatenate((np.arange(x.shape[2])[None], x.cpu().numpy().squeeze()), 0)
    header = "x"
    # For latex 1-based indexing is usually easier
    for i in range(1, x.shape[0] + 1):
        header += f",y{i}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, x_.T, delimiter=",", header=header, comments="")


def replace_path(path):
    return Path("./postprocessing/figure-data") / path.relative_to(base_path)


operators = ["convolution", "identity", "sample"]
factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
    ff.BernoulliLaplace(0.1, 1.0),
]
algorithms = ["cdps", "diffpir", "dpnp"]
model_based = ["l2", "l1", "log"]
signals_indices = [0, 1, 2]

base_path = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
for phi in factors:
    factor_path = base_path / phi.path() / "measurements"
    for operator in operators:
        operator_path = factor_path / operator
        # Gold-standard Gibbs posterior
        for name in ["m", "var_gibbs"]:
            ref = th.load(operator_path / "mmse" / f"{name}.pth")[signals_indices]
            save_csv(replace_path(operator_path) / "mmse" / f"{name}.csv", ref)
        for algorithm in algorithms:
            for instance in ["learned", "gibbs"]:
                instance_path = operator_path / "dps" / algorithm / instance
                ref = th.load(instance_path / "samples.pth")[:, signals_indices, :, :]
                for name, statistic in zip(["mean", "variance"], [ref.mean(0), ref.var(0)]):
                    save_csv(replace_path(instance_path) / f"{name}.csv", statistic)
        for method in model_based:
            instance_path = operator_path / "model-based" / method
            ref = th.load(instance_path / "xhat.pth")[signals_indices]
            save_csv(replace_path(instance_path) / f"{method}.csv", ref)