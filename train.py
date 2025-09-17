import torch as th
import math
import yaml
from datetime import datetime

from core.denoisers import DeepDenoiser
from core.networks import ConditionedUnet1D
from torch.utils.tensorboard import SummaryWriter
import common
from tqdm import tqdm
common.init()

def sample_sigma(
    x: th.Tensor, sigma_distribution: str, rng: th.Generator | None = None
) -> th.Tensor:
    if sigma_distribution == "log-gaussian":
        log_sigma_mean = -1.2
        log_sigma_sd = 1.2
        sigma = th.exp(
            log_sigma_sd * th.randn(x.shape[0], generator=rng, device=x.device)
            + log_sigma_mean
        )
    elif (
        sigma_distribution == "log-uniform"
    ):  # the base does not matter here : just need to be consistent with log-exp
        log_sigma_min = math.log(0.01)
        log_sigma_max = math.log(100)
        sigma = th.exp(
            log_sigma_min
            + (log_sigma_max - log_sigma_min)
            * th.rand(x.shape[0], generator=rng, device=x.device)
        )
    else:
        raise ValueError(f"{sigma_distribution} is not a valid distribution.")
    return sigma


def diffuse_to_sigma(
    x: th.Tensor, sigma: th.Tensor, rng: th.Generator | None = None
) -> th.Tensor:
    if x.shape[0] == sigma.shape[0]:
        B = sigma.shape[0]
        dims = x.dim() - 1
    else:
        raise ValueError(f"sigma and x do not share the same batch size.")
    z = th.randn(x.shape, device=x.device, generator=rng)
    y = x + z * sigma.reshape(B, *[1] * dims)
    return y


def compute_weighted_mse(
    x: th.Tensor, estimated_x: th.Tensor, sigma: th.Tensor, loss_weighting: str
) -> th.Tensor:
    se_per_signal = th.mean((x - estimated_x) ** 2, dim=list(range(1, x.dim())))
    if loss_weighting == "uniform":
        weight = 1
    elif loss_weighting == "SNR":
        weight = 1 / sigma**2
    elif loss_weighting == "minSNR-5":
        weight = th.clip(1 / sigma**2, max=5)
    elif loss_weighting == "maxSNR-1":
        weight = th.clip(1 / sigma**2, min=1)
    elif loss_weighting == "SNR+1":
        weight = 1 + (1 / sigma**2)
    else:
        raise ValueError(f"{loss_weighting} is not a valid loss weighting strategy.")
    loss = th.mean(se_per_signal * weight)
    return loss


def batch_loss(
    denoiser: DeepDenoiser,
    x: th.Tensor,
    loss_weighting: str = "SNR+1",
    sigma_distribution: str = "log-gaussian",
    seed: int | None = None,
) -> th.Tensor:
    """
    1. Sample a batch of noise standard deviations (sigma) from the given distribution (sigma_distribution).
    2. Diffuse the batch of clean signals (x), i.e. add noise of std. dev. sigma to obtain a batch of noisy signals (y).
    3. Denoise y to obtain an estimate of x (estimated_x).
    4. Compute a sample of the loss, i.e. a weighted MSE between x and estimated_x, with weights given by the weighting scheme (loss_weighting).

    Parameters
    ----------
    - denoiser : a callable that takes y and sigma as input, and returns a denoised signal.
    - x : batch of (ground truth) signals.
    - loss_weighting : determines how the per-example loss is weighted based on sigma ('uniform', 'SNR', 'SNR+1', 'maxSNR-1', 'minSNR-5'). See the notes below for more details.
    - sigma_distribution : determines the distribution from which sigma is sampled ('log-uniform', 'log-gaussian').
    - seed : optional seed used to sample sigma, as well as the Gaussian noise itself.

    Returns
    -------
    - The loss sample obtained for this batch.

    Notes
    -----
    In the EDM paper (Karras, 2022), they advise using a weighting scheme that leads to uniform effective weights on the network targets (x, score, v, etc...).
    - 'uniform' leads to uniform effective weights if the target of the network is x.
    - 'SNR' leads to uniform effective weights if the target of the network is the score (or the noise). Used in DDPM and NCSN.
    - 'SNR+1' leads to uniform effective weights if the target of the network is v. Used in EDM and in the v-prediction paper (Salimans and Ho, 2022).
    - 'maxSNR-1' is an alternative to SNR+1 (Salimans and Ho, 2022).
    - 'minSNR-5' is from "Efficient Diffusion Training via Min-SNR Weighting Strategy" (Hang et al., 2023)
    """
    if seed is None:
        rng = None
    else:
        rng = th.Generator(x.device).manual_seed(seed)

    sigma = sample_sigma(x, sigma_distribution, rng)
    y = diffuse_to_sigma(x, sigma, rng)
    estimated_x = denoiser(y, sigma)
    loss = compute_weighted_mse(x, estimated_x, sigma, loss_weighting)
    return loss


def main(factor_path):

    writer = SummaryWriter(
        log_dir=factor_path
        / "runs"
        / "logger"
        / datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    with open("./configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    denoiser = DeepDenoiser(ConditionedUnet1D(32, (1, 2, 4)).to(common.device), "eps")

    training_data, validation_data = [
        th.load(factor_path / "signals" / split / "s.pth", weights_only=True).to(common.device)
        for split in ["train", "validation"]
    ]

    optim = th.optim.Adam(denoiser.net.parameters(), lr=train_cfg["lr"])
    scheduler = th.optim.lr_scheduler.ExponentialLR(optim, train_cfg["lr_decay"])

    n_parameter_updates = 100_000
    batch_size = train_cfg["batch_size"]
    save_interval = 1000
    eval_every = 100

    for i in tqdm(range(n_parameter_updates), desc="Learning"):
        denoiser.net.train()
        batch_indices = th.randperm(
            training_data.shape[0], dtype=th.int32, device=common.device
        )[:batch_size]
        x = training_data[batch_indices]
        optim.zero_grad()
        loss = batch_loss(
            denoiser, x, train_cfg["loss_weighting"], train_cfg["sigma_distribution"]
        )
        loss.backward()
        optim.step()
        if i % eval_every == 0:
            with th.no_grad():
                denoiser.net.eval()
                batch_indices = th.randperm(
                    validation_data.shape[0], dtype=th.int32, device=common.device
                )[:batch_size]
                x_eval = validation_data[batch_indices]
                valid_loss = batch_loss(
                    denoiser,
                    x_eval,
                    train_cfg["loss_weighting"],
                    train_cfg["sigma_distribution"],
                )
                writer.add_scalar("loss/train", loss.item(), i)
                writer.add_scalar("loss/validation", valid_loss.item(), i)

        if i % save_interval == 0:
            th.save(denoiser.net.state_dict(), factor_path / "model.pth")

        scheduler.step()


if __name__ == "__main__":
    th.manual_seed(0)
    parser = common.make_parser(
        blocks=("factor",),
        description="Train denoisers for signals with jump distributions specified in the factors.\nFor help regarding the parameters of the factors, use `{factor} -h`.",
    )
    args = parser.parse_args()
    # Just to get the path to the training data; we don't really use the factor
    factor = common.build_factor(args)
    base_path = common.base_path
    main(base_path / factor.path())
