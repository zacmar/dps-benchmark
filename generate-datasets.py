import torch as th
import numpy as np
import factor as ff
import common

common.init()


def generate_signals(
    L: int,
    phi: ff.Factor,
    op_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    burn_in: int,
    n_samples: int,
    snr: float,
):
    samples = th.empty((n_test, n_samples, L), device=common.device, dtype=common.dtype)
    factor_path = common.base_path / phi.path()

    # Generate reference signals
    for name, n in zip(["train", "validation", "test"], [n_train, n_val, n_test]):
        u = phi.sample(th.Size((n, 1, L)))
        s = u.cumsum(2)
        path = factor_path / "signals" / name
        path.mkdir(parents=True, exist_ok=True)
        th.save(s, path / "s.pth")

    # Generate measurements
    A = common.operators[op_name]
    s = th.load(factor_path / "signals" / "test" / "s.pth", weights_only=True).to(
        common.device, common.dtype
    )

    noiseless_measurements = A @ s
    signal_power = th.median(th.mean(noiseless_measurements.abs() ** 2, dim=2)).item()
    var = signal_power / 10 ** (snr / 10)
    measurements = noiseless_measurements + np.sqrt(var) * th.randn_like(
        noiseless_measurements
    )

    m_path = factor_path / "measurements" / op_name
    m_path.mkdir(parents=True, exist_ok=True)
    th.save(measurements, m_path / "m.pth")
    th.save(th.full((1,), var), m_path / "var.pth")

    i = 0

    def accumulate(x: th.Tensor):
        nonlocal i
        if i >= burn_in:
            samples[:, i - burn_in : i - burn_in + 1] = x.clone()

        i += 1

    # MMSE estimators
    phi.sample_posterior(
        A.T @ measurements,
        A,
        ff.Gauss(measurements, th.full_like(measurements, var)),
        n_iter=burn_in + n_samples,
        callback=accumulate,
    )

    mmse_path = m_path / "mmse"
    mmse_path.mkdir(parents=True, exist_ok=True)
    # save samples
    th.save(samples, mmse_path / "posterior_samples.pth")
    th.save(samples.mean(1, keepdim=True), mmse_path / "m.pth")
    th.save(
        samples.var(1, unbiased=False, keepdim=True),
        mmse_path / "var_gibbs.pth",
    )


if __name__ == "__main__":
    th.manual_seed(0)

    parser = common.make_parser(
        blocks=("operator", "factor"),
        description="Dataset generation: {Train, test, validation} signals with jump distributions defined by {factor} and its parameters and and measurements obtained with {operator}.\nFor help regarding the parameters of the factors, use `{operator} {factor} -h`.",
    )
    args = parser.parse_args()
    phi = common.build_factor(args)
    n_train = 1_000_000
    n_val = 1_000
    n_test = 1_000

    burn_in = 10_000
    n_samples = 20_000

    snr = 25

    with th.no_grad():
        generate_signals(
            common.L,
            phi,
            args.operator,
            n_train,
            n_val,
            n_test,
            burn_in,
            n_samples,
            snr,
        )
