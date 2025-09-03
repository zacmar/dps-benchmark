import torch as th
from pathlib import Path
import os
import linop
import factor as ff
import numpy as np
from scipy.stats import wilcoxon

BENCH = "mean"
ROOT = Path(os.environ["EXPERIMENTS_ROOT"]) / "optimality-framework"
DEVICE = th.device("cuda")

gauss_kernel = linop.gaussian_kernel_1d(13, 2)[None]
indices = th.load("indices.pth")

# ----------------- Factors / Operators -----------------
factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.BernoulliLaplace(0.1, 1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
]

# display rows (order matches your table)
# third flag: True => merged (model-based baselines, use \multicolumn{2}{Y}{...})
disp_rows = [
    ("cdps",   "C-DPS",  False),
    ("diffpir","DiffPIR",False),
    ("dpnp",   "DPnP",   False),
    ("dpnp*",  "DPnP*",  False),   # NEW tuned row (only Gibbs column is filled)
    ("l1",     "L1",     True),
    ("log",    "Log",    True),
    ("l2",     "L2",     True),
]

operators = {
    "identity":    ("Denoising",     linop.Id()),
    "convolution": ("Deconvolution", linop.Conv1d(gauss_kernel)),
    "sample":      ("Imputation",    linop.Sample(indices)),
}

factor_header = {
    "gauss/0.25": "Gaussian",
    "laplace/1.0": "Laplace",
    "bernoulli-laplace/p=0.1_b=1.0": "Bernoulli--Laplace",
    "student/1.0": "Student-$t$(1)",
    "student/2.0": "Student-$t$(2)",
    "student/3.0": "Student-$t$(3)",
}

# alg rows that have (Learned, Gibbs) cells
alg_rows = [
    ("cdps",   "C-DPS"),
    ("diffpir","DiffPIR"),
    ("dpnp",   "DPnP"),
]

# scaling hooks (kept at 1)
norm_factors = {k: 1 for k in factor_header.keys()}

model_methods = [("l1","L1"), ("log","Log"), ("l2","L2")]

def fmt_sx(mean: float, std: float) -> str:
    # siunitx separate-uncertainty string
    return f"{mean:.2f} \\pm {std:.2f}"

alpha = 0.05
results = {}

# =================== Fill results ===================
for opname, (problem_name, A) in operators.items():
    results.setdefault(problem_name, {})

    # diffusion-style methods (Learned/Gibbs)
    for alg_key, alg_disp in alg_rows:
        for phi in factors:
            base_path = ROOT / phi.path()
            mmse = th.load(base_path / f"measurements/{opname}/mmse/m.pth", map_location="cpu").to(DEVICE)

            # mean over samples as in your code
            diff_learned = th.load(
                base_path / "measurements" / opname / "dps" / alg_key / "learned" / "samples.pth",
                map_location="cpu",
            ).to(DEVICE).mean(0)
            diff_gibbs = th.load(
                base_path / "measurements" / opname / "dps" / alg_key / "gibbs" / "samples.pth",
                map_location="cpu",
            ).to(DEVICE).mean(0)

            gap_learned_vec = th.mean((diff_learned - mmse) ** 2, dim=(1, 2)).cpu().numpy()
            gap_gibbs_vec   = th.mean((diff_gibbs   - mmse) ** 2, dim=(1, 2)).cpu().numpy()

            # one-sided Wilcoxon on raw MSEs to decide the star
            _, p_less    = wilcoxon(gap_learned_vec, gap_gibbs_vec, alternative="less")
            _, p_greater = wilcoxon(gap_learned_vec, gap_gibbs_vec, alternative="greater")

            # display on -log scale like your table example
            gap_learned_vec = -np.log(gap_learned_vec)
            gap_gibbs_vec   = -np.log(gap_gibbs_vec)

            gap_learned_mean, gap_learned_std = float(np.mean(gap_learned_vec)), float(np.std(gap_learned_vec))
            gap_gibbs_mean,   gap_gibbs_std   = float(np.mean(gap_gibbs_vec)),   float(np.std(gap_gibbs_vec))

            col_name = factor_header[str(phi.path())]
            results[problem_name].setdefault(alg_disp, {})

            learned_cell = fmt_sx(gap_learned_mean, gap_learned_std)
            gibbs_cell   = fmt_sx(gap_gibbs_mean,   gap_gibbs_std)

            # put a single star on whichever is significantly better
            if p_less < alpha and p_greater < alpha:
                print(f"WARNING: {alg_disp} {opname} {phi.path()} has both p-values < {alpha}")
            elif p_less < alpha:
                learned_cell += r"\textsuperscript{*}"
            elif p_greater < alpha:
                gibbs_cell += r"\textsuperscript{*}"

            results[problem_name][alg_disp][col_name] = (learned_cell, gibbs_cell)

    # DPnP* tuned row (fill Gibbs column from .../gibbs/tuned/samples.pth; leave Learned empty)
    for phi in factors:
        base_path = ROOT / phi.path()
        mmse = th.load(base_path / f"measurements/{opname}/mmse/m.pth", map_location="cpu").to(DEVICE)
        tuned_path = base_path / "measurements" / opname / "dps" / "dpnp" / "gibbs" / "tuned" / "samples.pth"

        col_name = factor_header[str(phi.path())]
        results[problem_name].setdefault("DPnP*", {})
        if tuned_path.exists():
            tuned = th.load(tuned_path, map_location="cpu").to(DEVICE).mean(0)
            gap_vec = th.mean((tuned - mmse) ** 2, dim=(1, 2)).cpu().numpy()
            gap_vec = -np.log(gap_vec)
            gap_mean, gap_std = float(np.mean(gap_vec)), float(np.std(gap_vec))
            results[problem_name]["DPnP*"][col_name] = ("", fmt_sx(gap_mean, gap_std))
        else:
            results[problem_name]["DPnP*"][col_name] = ("", "")

    # Model-based baselines (merged cells)
    for phi in factors:
        base_path = ROOT / phi.path()
        mmse = th.load(base_path / f"measurements/{opname}/mmse/m.pth", map_location="cpu").to(DEVICE)
        col_name = factor_header[str(phi.path())]
        for meth_key, meth_disp in model_methods:
            try:
                xhat = th.load(
                    base_path / f"measurements/{opname}/model-based/{meth_key}/xhat.pth",
                    map_location="cpu",
                ).to(DEVICE)
            except FileNotFoundError:
                results[problem_name].setdefault(meth_disp, {})
                results[problem_name][meth_disp][col_name] = ""
                continue
            gap_vec = th.mean((xhat - mmse) ** 2, dim=(1, 2)).cpu().numpy()
            gap_vec = -np.log(gap_vec)
            gap_mean, gap_std = float(np.mean(gap_vec)), float(np.std(gap_vec))
            results[problem_name].setdefault(meth_disp, {})
            results[problem_name][meth_disp][col_name] = fmt_sx(gap_mean, gap_std)

# =================== Print LaTeX table ===================
def print_table(results):
    ordered_cols = [factor_header[str(phi.path())] for phi in factors]  # 6 distributions

    # Use your exact S / Y column types (copy from your LaTeX)
    colspec = (
        r"ll%"
        r"*{4}{S[round-mode=places,round-precision=2,table-format=1.2(2)\textsuperscript{*}]}"
        r"*{2}{S[round-mode=places,round-precision=2,table-format=1.2(3)\textsuperscript{*}]}"
        r"*{2}{S[round-mode=places,round-precision=2,table-format=-2.2(3)\textsuperscript{*}]}"
        r"*{2}{S[round-mode=places,round-precision=2,table-format=2.2(3)\textsuperscript{*}]}"
        r"*{2}{S[round-mode=places,round-precision=2,table-format=2.2(2)\textsuperscript{*}]}"
    )

    # cmidrule ranges for 6 pairs: (3-4),(5-6),(7-8),(9-10),(11-12),(13-14)
    cmis = r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}"

    # distribution header line
    dist_names = " & ".join([rf"\multicolumn{{2}}{{c}}{{{name}}}" for name in ordered_cols])

    print(r"\begin{tabular}{" + colspec + "}")
    print(r"    \toprule")
    print(r"     & & " + dist_names + r" \\")
    print("     " + cmis)
    print(r"     & & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} \\")
    print(r"    \midrule")

    problem_order = ["Denoising", "Deconvolution", "Imputation"]
    n_rows_block = len(disp_rows)

    for pid, problem_name in enumerate(problem_order):
        block = results.get(problem_name, {})
        for ridx, (row_key, row_disp, is_merged) in enumerate(disp_rows):
            row_cells = []
            for col in ordered_cols:
                if is_merged:
                    cell = block.get(row_disp, {}).get(col, "")
                    row_cells.append(rf"\multicolumn{{2}}{{Y}}{{{cell}}}")
                else:
                    learned, gibbs = block.get(row_disp, {}).get(col, ("", ""))
                    row_cells.extend([learned, gibbs])

            if ridx == 0:
                # rotate the problem label like in your screenshot
                print(f"    \\multirow{{{n_rows_block}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{problem_name}}}}} ", end="")
            else:
                print("    ", end="")
            print(f"& {row_disp} & " + " & ".join(row_cells) + r" \\")
        if pid < len(problem_order) - 1:
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"\end{tabular}")

print_table(results)