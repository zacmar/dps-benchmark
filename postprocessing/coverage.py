import torch as th
from pathlib import Path
import os
import linop
import factor as ff
import common
common.init()


alpha = 0.9
factors: list[ff.Factor] = [
    ff.Gauss(0, 0.25),
    ff.Laplace(1.0),
    ff.BernoulliLaplace(0.1, 1.0),
    ff.StudentT(1.0),
    ff.StudentT(2.0),
    ff.StudentT(3.0),
]

operators = {
    "identity": ("Denoising", common.operators["identity"]),
    "convolution": ("Deconvolution", common.operators["convolution"]),
    "sample": ("Imputation", common.operators["sample"]),
    "fourier": ("Fourier", common.operators["fourier"])
}

# Short headers
factor_header = {
    "gauss/0.25": "Gaussian",
    "laplace/1.0": "Laplace",
    "bernoulli-laplace/p=0.1_b=1.0": "Bernoulli--Laplace",
    "student/1.0": "Student-$t$(1)",
    "student/2.0": "Student-$t$(2)",
    "student/3.0": "Student-$t$(3)",
}

alg_rows = [
    ("cdps", "C-DPS"),
    ("diffpir", "DiffPIR"),
    ("dpnp", "DPnP"),
]


def weighted_hpd_threshold(logpi: th.Tensor, w: th.Tensor, alpha: float) -> th.Tensor:
    """
    logpi: (b, n) unnormalized log posterior at samples
    w    : (b, n) nonnegative weights (need not be normalized; we'll normalize per batch)
    alpha: float in (0,1)

    returns: (b,) tensor of thresholds lambda_alpha
    """
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-38)
    idx = th.argsort(logpi, dim=1, descending=True)
    b_idx = th.arange(logpi.size(0), device=logpi.device).unsqueeze(1)
    logpi_sorted = logpi[b_idx, idx]
    w_sorted = w[b_idx, idx]
    csum = th.cumsum(w_sorted, dim=1)
    mask = csum >= alpha
    any_true = mask.any(dim=1)
    k_first = th.argmax(mask.to(th.int64), dim=1)
    n = logpi.size(1)
    k = th.where(any_true, k_first, th.full_like(k_first, n - 1))
    lam = logpi_sorted[b_idx.squeeze(1), k]
    return lam


def hpd_covered(
    logpi_true: th.Tensor, logpi_samples: th.Tensor, weights: th.Tensor, alpha: float
) -> th.Tensor:
    """
    logpi_true   : (b,)   log posterior at truths
    logpi_samples: (b, n) log posterior at samples
    weights      : (b, n) sample weights (need not be normalized)
    alpha        : float

    returns: (b,) bool tensor, True iff logpi_true >= lambda_alpha
    """
    lam = weighted_hpd_threshold(logpi_samples, weights, alpha)
    return logpi_true >= lam


D = linop.Grad1D()

with th.no_grad():
    results = {}
    for opname, (problem_name, A) in operators.items():
        results.setdefault(problem_name, {})
        for phi in factors:
            print(opname, phi.path())
            base_path = common.base_path / phi.path()
            s_true = th.load(base_path / "signals/test/s.pth", map_location=common.device).to(common.dtype)
            y = th.load(base_path / f"measurements/{opname}/m.pth")
            var = th.load(base_path / f"measurements/{opname}/var.pth", map_location=common.device)

            def logp(x: th.Tensor):
                return -(((A @ x).view(s_true.shape[0], -1, y.shape[2]) - y) ** 2).sum(
                    (2,)
                ) / 2 / var + phi.log_prob(D @ x).view(
                    s_true.shape[0], -1, s_true.shape[2]
                ).sum(
                    (2)
                )

            ref_samples = (
                th.load(base_path / f"measurements/{opname}/mmse/posterior_samples.pth")[:, :8000]
                .contiguous()
            )
            logpis_signal = logp(s_true).squeeze()
            logpis_sample = logp(ref_samples.view(-1, 1, ref_samples.shape[-1])).view(
                *ref_samples.shape[:2]
            )
            weights = th.ones_like(logpis_sample)
            coverage = (
                hpd_covered(logpis_signal, logpis_sample, weights, alpha)
                .float()
                .mean()
                .item()
            )
            col_name = factor_header[str(phi.path())]
            results[problem_name].setdefault("Reference", {})
            results[problem_name]["Reference"][col_name] = (
                f"{coverage:.2f}",
                f"{coverage:.2f}",
            )

            # Algorithm coverages
            for alg_key, alg_disp in alg_rows:
                for bench_kind in ("learned", "gibbs"):
                    alg_base = (
                        base_path
                        / "measurements"
                        / opname
                        / "dps"
                        / alg_key
                        / bench_kind
                    )
                    # These are arranged like (n_samples, n_signals, 1, length) which is different
                    # to how we saved the ref signals that are like (n_signals, n_samples, length)
                    # so we have to permute
                    p = alg_base / "samples.pth"
                    if p.exists():
                        alg_samples = (
                            th.load(p)
                            .permute(1, 0, 2, 3)
                            .squeeze()
                            .contiguous()
                        )
                        logpis_sample = logp(
                            alg_samples.view(-1, 1, ref_samples.shape[-1])
                        ).view(*alg_samples.shape[:2])
                        weights = th.softmax(logpis_sample, dim=1)
                        coverage = (
                            hpd_covered(logpis_signal, logpis_sample, weights, alpha)
                            .float()
                            .mean()
                            .item()
                        )
                        learned_str, gibbs_str = (
                            results[problem_name]
                            .setdefault(alg_disp, {})
                            .get(col_name, ("", ""))
                        )
                        if bench_kind == "learned":
                            learned_str = f"{coverage:.2f}"
                        else:
                            gibbs_str = f"{coverage:.2f}"
                        results[problem_name][alg_disp][col_name] = (learned_str, gibbs_str)

            del (
                s_true,
                y,
                var,
                ref_samples,
                alg_samples,
                logpis_signal,
                logpis_sample,
                weights,
            )
            th.cuda.empty_cache()


# ---------- Print the LaTeX table ----------
def print_table(results):
    ordered_cols = [factor_header[str(phi.path())] for phi in factors]
    colspec = "ll*{8}{S[round-mode=places,round-precision=2,table-format=1.2]}"

    print(r"\begin{tabular}{" + colspec + "}")
    print(r"    \toprule")
    print(
        r"    \multirow{3}{*}{Problem} & \multirow{3}{*}{Algorithm} & \multicolumn{8}{c}{Jump Distribution} \\"
    )
    print(r"    \cmidrule(lr){3-10}")
    print(
        r"     & & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Laplace} & \multicolumn{2}{c}{Bernoulli--Laplace} & \multicolumn{2}{c}{Student-$t$} \\"
    )
    print(
        r"     \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
    )
    print(
        r"     & & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} & {Learned} & {Gibbs} \\"
    )
    print(r"    \midrule")

    problem_order = ["Denoising", "Deconvolution", "Imputation", "Fourier"]
    for pid, problem_name in enumerate(problem_order):
        block = results.get(problem_name, {})

        # Reference row
        row_cells = []
        for col in ordered_cols:
            learned, gibbs = block.get("Reference", {}).get(col, ("", ""))
            row_cells.extend([learned, gibbs])
        print(
            f"    \\multirow{{4}}{{*}}{{{problem_name}}} & Reference & "
            + " & ".join(row_cells)
            + r" \\"
        )

        # Algorithm rows
        for alg_key, alg_disp in alg_rows:
            row_cells = []
            for col in ordered_cols:
                learned, gibbs = block.get(alg_disp, {}).get(col, ("", ""))
                row_cells.extend([learned, gibbs])
            print(f"    & {alg_disp} & " + " & ".join(row_cells) + r" \\")
        if pid < len(problem_order) - 1:
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"\end{tabular}")


print_table(results)
