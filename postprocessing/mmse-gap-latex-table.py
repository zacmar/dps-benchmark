import torch as th
from pathlib import Path
import linop
import factor as ff
import numpy as np
import common
from scipy.stats import wilcoxon
common.init()

ROOT = common.base_path
DEVICE = common.device

factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.BernoulliLaplace(0.1, 1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
]

alpha=0.05

operators = {
    "identity": ("Denoising", common.operators["identity"]),
    "convolution": ("Deconvolution", common.operators["convolution"]),
    "sample": ("Imputation", common.operators["sample"]),
    "fourier": ("Fourier", common.operators["fourier"]),
}

factor_header = {
    "gauss/0.25": "Gaussian",
    "laplace/1.0": "Laplace",
    "bernoulli-laplace/p=0.1_b=1.0": "Bernoulli--Laplace",
    "student/1.0": "Student-$t$(1)",
    "student/2.0": "Student-$t$(2)",
    "student/3.0": "Student-$t$(3)",
}

# Algorithm row order and display names (DPS algorithms only)
alg_rows = [
    ("cdps", "C-DPS"),
    ("diffpir", "DiffPIR"),
    ("dpnp", "DPnP"),
]

# Model-based estimators
model_methods = [
    ("l1", "L1"),
    ("l2", "L2"),
]

def fmt_sx(mean: float, std: float) -> str:
    return f"{mean:.2f} \\pm {std:.2f}"

results = {}
comp_results = {}

for opname, (problem_name, A) in operators.items():
    results.setdefault(problem_name, {})
    comp_results.setdefault(problem_name, {})

    for alg_key, alg_disp in alg_rows:
        for phi in factors:
            base_path = ROOT / phi.path()

            signals = th.load(base_path / "signals" / "test" / "s.pth").to(DEVICE)
            mmse = th.load(base_path / f"measurements/{opname}/mmse/m.pth").to(DEVICE)

            # learned
            diff_learned = th.load(
                base_path / "measurements" / opname / "dps" / alg_key / "learned" / "samples.pth",
                map_location=DEVICE,
            ).mean(0).to(DEVICE)

            mse_ref = th.mean((mmse - signals) ** 2, dim=(1, 2))
            mse_learned = th.mean((diff_learned - signals) ** 2, dim=(1, 2))
            dB_learned = 10 * th.log10(mse_learned / mse_ref)

            col_name = factor_header[str(phi.path())]
            results[problem_name].setdefault(alg_disp, {})
            results[problem_name][alg_disp][col_name] = fmt_sx(
                dB_learned.mean().item(), dB_learned.std().item()
            )
            p = base_path / "measurements" / opname / "dps" / alg_key / "gibbs" / "samples.pth"
            # Just to be able to run the script while some computations may still be
            # ongoing
            if p.exists():
                diff_gibbs = th.load(p, map_location=DEVICE).mean(0).to(DEVICE)
            else:
                diff_gibbs = None
            comp_results[problem_name].setdefault(alg_disp, {})
            if diff_gibbs is not None:
                mse_gibbs = th.mean((diff_gibbs - signals) ** 2, dim=(1, 2))
                dB_gibbs = 10 * th.log10(mse_gibbs / mse_ref)
                dB_diff = dB_gibbs - dB_learned
                mean_diff = dB_diff.mean().item()
                std_diff  = dB_diff.std().item()

                d = dB_diff.detach().cpu().numpy()
                med = float(np.median(d))
                if med > 0:
                    _, p = wilcoxon(d, alternative="greater")
                elif med < 0:
                    _, p = wilcoxon(d, alternative="less")
                else:
                    p = 1.0

                cell = fmt_sx(mean_diff, std_diff)
                if p < alpha:
                    cell += r"\textsuperscript{*}"

                comp_results[problem_name][alg_disp][col_name] = cell
            else:
                # leave empty if gibbs file missing
                comp_results[problem_name][alg_disp][col_name] = ""

    for phi in factors:
        base_path = ROOT / phi.path()
        signals = th.load(base_path / "signals" / "test" / "s.pth").to(DEVICE)
        mmse = th.load(base_path / f"measurements/{opname}/mmse/m.pth").to(DEVICE)
        col_name = factor_header[str(phi.path())]

        for meth_key, meth_disp in model_methods:
            xhat = th.load(
                base_path / f"measurements/{opname}/model-based/{meth_key}/xhat.pth",
                map_location=DEVICE,
            ).to(DEVICE)
            mse_ref = th.mean((mmse - signals) ** 2, dim=(1, 2))
            mse_model = th.mean((xhat - signals) ** 2, dim=(1, 2))
            out = 10 * th.log10(mse_model / mse_ref)
            cell = fmt_sx(out.mean().item(), out.std().item())

            results[problem_name].setdefault(meth_disp, {})
            results[problem_name][meth_disp][col_name] = cell


def print_table(results):
    ordered_cols = [factor_header[str(phi.path())] for phi in factors]
    colspec = "ll*{6}{l}"

    print(r"\begin{tabular}{" + colspec + "}")
    print(r"    \toprule")
    header_cols = " & ".join(ordered_cols)
    print(rf"    Problem & Algorithm & {header_cols} \\")
    print(r"    \midrule")

    problem_order = ["Denoising", "Deconvolution", "Imputation", "Fourier"]
    row_labels = ["C-DPS", "DiffPIR", "DPnP", "L1", "L2"]

    for pid, problem_name in enumerate(problem_order):
        block = results.get(problem_name, {})
        for ridx, alg_disp in enumerate(row_labels):
            row_cells = [block.get(alg_disp, {}).get(col, "") for col in ordered_cols]
            if ridx == 0:
                print(f"    \\multirow{{{len(row_labels)}}}{{*}}{{{problem_name}}} ", end="")
            else:
                print("    ", end="")
            print(f"& {alg_disp} & " + " & ".join(row_cells) + r" \\")
        if pid < len(problem_order) - 1:
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"\end{tabular}")


def print_comparison_table(comp_results):
    ordered_cols = [factor_header[str(phi.path())] for phi in factors]
    colspec = "ll*{6}{l}"

    print(r"\begin{tabular}{" + colspec + "}")
    print(r"    \toprule")
    header_cols = " & ".join(ordered_cols)
    print(rf"    Problem & Algorithm & {header_cols} \\")
    print(r"    \midrule")

    problem_order = ["Denoising", "Deconvolution", "Imputation", "Fourier"]
    row_labels = ["C-DPS", "DiffPIR", "DPnP"]

    for pid, problem_name in enumerate(problem_order):
        block = comp_results.get(problem_name, {})
        for ridx, alg_disp in enumerate(row_labels):
            row_cells = [block.get(alg_disp, {}).get(col, "") for col in ordered_cols]
            if ridx == 0:
                print(f"    \\multirow{{{len(row_labels)}}}{{*}}{{{problem_name}}} ", end="")
            else:
                print("    ", end="")
            print(f"& {alg_disp} & " + " & ".join(row_cells) + r" \\")
        if pid < len(problem_order) - 1:
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"\end{tabular}")


print_table(results)
print()
print_comparison_table(comp_results)
