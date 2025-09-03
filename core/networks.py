import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F


class ConditionedUnet1D(torch.nn.Module):
    # inspired from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py#L15
    def __init__(
        self,
        # number of channels after first layer (determines the size of the network)
        channels: int,
        multipliers: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()

        dims = [channels * m for m in multipliers]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Embedding for the conditioning scalar
        emb_dim = channels * 2
        self.emb_mlp = nn.Sequential(
            FourierEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers
        self.init_block = ConditionedBlock(1, channels, emb_dim)

        self.downs = ModuleList([])
        for dim_in, dim_out in in_out:
            self.downs.append(
                ModuleList(
                    [
                        nn.Conv1d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
                        ConditionedBlock(dim_out, dim_out, emb_dim),
                    ]
                )
            )

        self.ups = ModuleList([])
        for dim_out, dim_in in reversed(in_out):
            self.ups.append(
                ModuleList(
                    [
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1),
                        ),
                        ConditionedBlock(dim_out + dim_out, dim_out, emb_dim),
                    ]
                )
            )

        self.final_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x, c):
        """
        x is the BxK noisy tensor; c is the B conditionning tensor (timestep, noise, ...)
        """
        h = []
        emb = self.emb_mlp(c)
        x = self.init_block(x, emb)

        for downsample, block in self.downs:
            h.append(x)
            x = downsample(x)
            x = block(x, emb)

        for upsample, block in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, emb)

        return self.final_conv(x)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class FourierEmb(Module):
    def __init__(self, dim, learned=True):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(half_dim), requires_grad=True if learned else False
        )

    def forward(self, x):
        x = x[:, None] * self.weights[None, :] * 2 * torch.pi
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x


class ConditionedBlock(Module):
    def __init__(self, dim, dim_out, emb_dim):
        """
        2x Conv-RMSNorm-ReLU with FiLM conditioning after the first normalization.
        With residual skip.
        """
        super().__init__()
        self.act = nn.ReLU()

        self.mlp = nn.Sequential(self.act, nn.Linear(emb_dim, dim_out * 2))

        self.conv1 = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm1 = RMSNorm(dim_out)

        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, padding=1)
        self.norm2 = RMSNorm(dim_out)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb):
        res = self.res_conv(x)

        emb = self.mlp(emb)
        emb = emb.unsqueeze(2)  # BxCx1
        scale, shift = emb.chunk(2, dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = x * (scale + 1) + shift
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x + res


if __name__ == "__main__":
    from torchinfo import summary

    net = ConditionedUnet1D(32, (1, 2, 4))
    x = torch.zeros((1, 100))
    c = torch.zeros((1,))
    net(x, c)
    summary(net, input_size=((1, 100), (1,)), depth=6)
