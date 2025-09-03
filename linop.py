import numpy as np
import torch as th
from abc import ABC, abstractmethod


# TODO i think these shapes are useless
class LinOp(ABC):
    def __init__(self):
        self.in_shape: tuple
        self.out_shape: tuple

    @abstractmethod
    def apply(self, x: th.Tensor) -> th.Tensor: ...

    @abstractmethod
    def applyT(self, y: th.Tensor) -> th.Tensor: ...

    def __add__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __radd__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __sub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __rsub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __mul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __rmul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __matmul__(self, other):
        if isinstance(other, LinOp):
            return Composition(self, other)
        return self.apply(other)

    @property
    def T(self):
        return Transpose(self)


## Utils classes
class Composition(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp2.in_shape == (-1,) else LinOp2.in_shape
        self.out_shape = (
            LinOp2.out_shape if LinOp1.out_shape == (-1,) else LinOp1.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyT(self, y):
        return self.LinOp2.applyT(self.LinOp1.applyT(y))


class Sum(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp2.applyT(y) + self.LinOp1.applyT(y)


class Diff(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp1.applyT(y) - self.LinOp2.applyT(y)


class ScalarMul(LinOp):
    def __init__(self, LinOp: LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_shape = LinOp.in_shape
        self.out_shape = LinOp.out_shape

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyT(self, y):
        return self.LinOp.applyT(y) * self.scalar


class Transpose(LinOp):
    def __init__(self, LinOpT: LinOp):
        self.LinOpT = LinOpT
        self.in_shape = LinOpT.out_shape
        self.out_shape = LinOpT.in_shape

    def apply(self, x):
        return self.LinOpT.applyT(x)

    def applyT(self, y):
        return self.LinOpT.apply(y)


class Matrix(LinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (matrix.shape[1],)
        self.out_shape = (matrix.shape[0],)

    def apply(self, x):
        return self.H @ x

    def applyT(self, y):
        return self.H.T.conj() @ y


class Mul(LinOp):
    """coefs is for element-wise multiplication"""

    def __init__(self, coefs):
        self.coefs = coefs
        self.in_shape = coefs.shape
        self.out_shape = coefs.shape

    def apply(self, x):
        return self.coefs * x

    def applyT(self, y):
        return self.coefs.conj() * y


class Fft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft(x, norm="ortho")

    def applyT(self, y):
        return th.fft.ifft(y, norm="ortho")


class Ifft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft(x, norm="ortho")

    def applyT(self, y):
        return th.fft.fft(y, norm="ortho")


class Fftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fftshift(x, dim=(-2, -1))

    def applyT(self, y):
        return th.fft.ifftshift(y, dim=(-2, -1))


class Ifftshift(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifftshift(x, dim=(-2, -1))

    def applyT(self, y):
        return th.fft.fftshift(y, dim=(-2, -1))


class Id(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x

    def applyT(self, y):
        return y


class Flip(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.flip(x)

    def applyT(self, y):
        return th.flip(y)


class Roll(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.to(th.int64)

    def apply(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        ind0 = th.arange(n, dtype=th.int64)[:, None, None, None].expand(n, c, h, w)
        ind1 = th.arange(c, dtype=th.int64)[None, :, None, None].expand(n, c, h, w)
        ind2 = th.arange(h, dtype=th.int64)[None, None, :, None].expand(n, c, h, w)
        ind3 = th.arange(w, dtype=th.int64)[None, None, None, :].expand(n, c, h, w)

        return expanded[
            ind0,
            ind1,
            (ind2 + self.shifts[:, 0, None, None, None]) % h,
            (ind3 + self.shifts[:, 1, None, None, None]) % w,
        ]

    def applyT(self, y):
        n, _, h, w = y.shape
        c = self.shifts.shape[0]
        expanded = y.expand(-1, c, -1, -1)
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        ind0 = th.arange(n)[:, None, None, None].expand(n, c, h, w)
        ind1 = th.arange(c)[None, :, None, None].expand(n, c, h, w)
        ind2 = th.arange(h)[None, None, :, None].expand(n, c, h, w)
        ind3 = th.arange(w)[None, None, None, :].expand(n, c, h, w)

        return expanded[
            ind0,
            ind1,
            (ind2 - self.shifts[:, 0, None, None, None]) % h,
            (ind3 - self.shifts[:, 1, None, None, None]) % w,
        ]


class Real(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.real

    def applyT(self, y):
        return y


class Imag(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.imag

    def applyT(self, y):
        return y


# https://chatgpt.com/share/671f9a7c-8ccc-8011-a1f1-57c74c2e1db1
class RealPartExpand(LinOp):
    def __init__(self):
        # TODO this is incorrect
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return x.real

    def applyT(self, y):
        return y + 0j


# 2D classes
class Matrix2(LinOp):
    def __init__(self, matrix):
        self.H = matrix
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return self.H @ x

    def applyT(self, y):
        return self.H.T.conj() @ y


class Fft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft2(x, norm="ortho")

    def applyT(self, y):
        return th.fft.ifft2(y, norm="ortho")


class Ifft2(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.ifft2(x, norm="ortho")

    def applyT(self, y):
        return th.fft.fft2(y, norm="ortho")


class Roll2(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.to(th.int64)

    def apply(self, x):
        n, _, h, w = x.shape
        c = self.shifts.shape[0]
        expanded = x.expand(-1, c, -1, -1)
        # TODO
        # https://discuss.pytorch.org/t/tensor-shifts-in-torch-roll/170655/2
        # This is still not really optimal, lots of stuff done for nothing
        # I think we can make this more efficient with a double-gather
        self.ind0 = th.arange(n, dtype=th.int64, device=x.device)[
            :, None, None, None
        ].expand(n, c, h, w)
        self.ind1 = th.arange(c, dtype=th.int64, device=x.device)[
            None, :, None, None
        ].expand(n, c, h, w)
        self.ind2 = th.arange(h, dtype=th.int64, device=x.device)[
            None, None, :, None
        ].expand(n, c, h, w)
        self.ind3 = th.arange(w, dtype=th.int64, device=x.device)[
            None, None, None, :
        ].expand(n, c, h, w)

        return expanded[
            self.ind0,
            self.ind1,
            (self.ind2 + self.shifts[None, :, 0, None, None]) % h,
            (self.ind3 + self.shifts[None, :, 1, None, None]) % w,
        ]

    def applyT(self, y):
        return y[
            self.ind0,
            self.ind1,
            (self.ind2 - self.shifts[None, :, 0, None, None]) % y.shape[2],
            (self.ind3 - self.shifts[None, :, 1, None, None]) % y.shape[3],
        ].sum(1, keepdim=True)


class PhaseShift(LinOp):
    def __init__(self, shifts, shape):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        M, N = shape
        x = th.arange(M)[:, None].to(shifts.device) / M
        y = th.arange(N)[None, :].to(shifts.device) / N
        self.phase_shift = th.exp(
            -2j
            * np.pi
            * (x[None] * shifts[:, 0, None, None] + y[None] * shifts[:, 1, None, None])
        )[None]

    def apply(self, x):
        return x * self.phase_shift

    def applyT(self, y):
        return (y * self.phase_shift.conj()).sum(1, keepdim=True)


class ShiftInterp(LinOp):
    def __init__(self, shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = shifts.fliplr()
        self.dtype = th.float32

    def apply(self, x):
        x_ = x.expand(self.shifts.shape[0], 1, *x.shape[2:])
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = self.shifts / x.shape[2] * 2.0
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *x.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(x.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(x_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(x_.imag, grid, align_corners=False)
        ).permute(1, 0, 2, 3)

    def applyT(self, y):
        y_ = y.permute(1, 0, 2, 3)
        thetas = th.eye(2, 3, dtype=th.float32)[None].repeat(self.shifts.shape[0], 1, 1)
        thetas[:, :, 2] = -self.shifts / y.shape[2] * 2
        grid = (
            th.nn.functional.affine_grid(
                thetas, (self.shifts.shape[0], 1, *y.shape[2:]), align_corners=False
            )
            .to(self.dtype)
            .to(y.device)
        )
        grid = ((grid + 1) % 2) - 1
        return (
            th.nn.functional.grid_sample(y_.real, grid, align_corners=False)
            + 1j * th.nn.functional.grid_sample(y_.imag, grid, align_corners=False)
        ).sum(0, keepdim=True)


# TODO tests
class Crop2(LinOp):
    def __init__(self, in_shape, crop_shape):
        self.in_shape = in_shape
        self.out_shape = crop_shape

        self.istart = (self.in_shape[0] - self.out_shape[0]) // 2
        self.iend = self.istart + self.out_shape[0]

        self.jstart = (self.in_shape[1] - self.out_shape[1]) // 2
        self.jend = self.jstart + self.out_shape[1]

        ipad2 = self.in_shape[0] - self.iend
        jpad2 = self.in_shape[1] - self.jend

        self.pads = tuple(int(pad) for pad in (self.jstart, jpad2, self.istart, ipad2))

    def apply(self, x):
        return x[:, :, self.istart : self.iend, self.jstart : self.jend]

    def applyT(self, y):
        return th.nn.functional.pad(y, self.pads, mode="constant")


class Roll2_PadZero(LinOp):
    def __init__(self, v_shifts, h_shifts):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.v_shifts = int(v_shifts)
        self.h_shifts = int(h_shifts)

    def apply(self, x):
        x = th.roll(x, self.h_shifts, dims=1)
        if self.h_shifts < 0:
            x[:, self.h_shifts :] = 0
        elif self.h_shifts > 0:
            x[:, 0 : self.h_shifts] = 0

        x = th.roll(x, self.v_shifts, dims=0)
        if self.v_shifts < 0:
            x[self.v_shifts :, :] = 0
        elif self.v_shifts > 0:
            x[0 : self.v_shifts, :] = 0
        return x

    def applyT(self, y):
        y = th.roll(y, -self.h_shifts, dims=1)
        if -self.h_shifts < 0:
            y[:, -self.h_shifts :] = 0
        elif -self.h_shifts > 0:
            y[:, 0 : -self.h_shifts] = 0

        y = th.roll(y, -self.v_shifts, dims=0)
        if -self.v_shifts < 0:
            y[-self.v_shifts :, :] = 0
        elif -self.v_shifts > 0:
            y[0 : -self.v_shifts, :] = 0
        return y


class Stack(LinOp):
    def __init__(self, linops: tuple[LinOp, ...]):
        self.linops = linops
        self.in_shape = (0, 0)
        self.out_shape = (0, 0)

    def apply(self, x: th.Tensor) -> tuple[th.Tensor, ...]:
        return tuple(linop.apply(x) for linop in self.linops)

    # TODO this is not generic yet... probably the best implementation is just
    # with using lists..
    def applyT(self, y: tuple[th.Tensor, ...]) -> th.Tensor:
        assert len(y) == len(self.linops)
        res = self.linops[0].applyT(y[0])
        for idx, linop in enumerate(self.linops[1:], start=1):
            res += linop.applyT(y[idx])
        return res


# functions
def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = th.roll(data, dx, dims=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = th.roll(shifted_data, dy, dims=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


class SumReduce(LinOp):
    def __init__(self, dim, size):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.dim = dim
        self.size = size

    def apply(self, x):
        return x.sum(dim=self.dim, keepdim=True)

    # TODO this is not generic yet wrt dimension
    def applyT(self, y):
        return y.expand(y.shape[0], self.size, *y.shape[2:])


class Grad1D(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x: th.Tensor) -> th.Tensor:
        res = x.clone()
        res[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return res
        # return x[:, :, 1:] - x[:, :, :-1]

    def applyT(self, y: th.Tensor) -> th.Tensor:
        res = y.clone()
        res[:, :, :-1] = y[:, :, :-1] - y[:, :, 1:]
        return res
        # div = th.zeros((y.shape[0], y.shape[1], y.shape[2] + 1), device=y.device)
        # div[:, :, :-1] -= y
        # div[:, :, 1:] += y
        # return div


# TODO this is slightly different to Grad1D as here we map to the same number of pixels
# with neumann boundaries. we need to check how this influences stuff
class Grad2D(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        grad = th.zeros((x.shape[0], 2, *x.shape[2:]), device=x.device, dtype=x.dtype)
        grad[:, 0:1, :, :-1] += x[:, :, :, 1:] - x[:, :, :, :-1]
        grad[:, 1:2, :-1, :] += x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad

    def applyT(self, y):
        div = th.zeros((y.shape[0], 1, *y.shape[2:]), device=y.device, dtype=y.dtype)
        div[:, :, :, 1:] += y[:, 0, :, :-1]
        div[:, :, :, :-1] -= y[:, 0, :, :-1]
        div[:, :, 1:, :] += y[:, 1, :-1, :]
        div[:, :, :-1, :] -= y[:, 1, :-1, :]
        return div


class Conv1d(LinOp):
    def __init__(self, k: th.Tensor):
        self.out_shape = (1,)
        self.in_shape = (1,)
        self.op = th.nn.Conv1d(
            in_channels=1,
            out_channels=k.shape[0],
            kernel_size=k.shape[1],
            bias=False,
            dtype=k.dtype,
            device=k.device,
            padding="same",
            padding_mode="circular",
        )

        for i, kernel in enumerate(k):
            self.op.weight.data[i, 0] = kernel

        self.opT = th.nn.Conv1d(
            in_channels=k.shape[0],
            out_channels=1,
            kernel_size=k.shape[1],
            bias=False,
            dtype=k.dtype,
            device=k.device,
            padding="same",
            padding_mode="circular",
        )

        for i, kernel in enumerate(k):
            self.opT.weight.data[0, i] = th.flip(kernel, (0,))

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.op(x)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.opT(y)


class Conv2d(LinOp):
    def __init__(self, k: th.Tensor):
        self.op = th.nn.Conv2d(
            in_channels=1,
            out_channels=k.shape[0],
            kernel_size=k.shape[1],
            bias=False,
            dtype=k.dtype,
            device=k.device,
            padding=0,
        )

        for i, kernel in enumerate(k):
            self.op.weight.data[i, 0] = kernel

        self.opT = th.nn.Conv2d(
            in_channels=k.shape[0],
            out_channels=1,
            kernel_size=k.shape[1],
            bias=False,
            dtype=k.dtype,
            device=k.device,
            padding=k.shape[1],
            padding_mode="zeros",
        )

        for i, kernel in enumerate(k):
            self.opT.weight.data[0, i] = th.rot90(kernel, 2)

    def apply(self, x: th.Tensor) -> th.Tensor:
        return self.op(x)

    def applyT(self, y: th.Tensor) -> th.Tensor:
        return self.opT(y)


# TODO
class ConvGaussianOperator(LinOp):
    def __init__(self, sigma, trunc=3, K=100):
        self.sigma = sigma  # for the __str__ method
        # Compute sizes
        kernel_radius = round(trunc * sigma)
        kernel_size = 2 * kernel_radius + 1
        M = K - kernel_size + 1
        # Gaussian kernel
        x = th.arange(-kernel_radius, kernel_radius + 1, dtype=th.float32)
        gaussian_kernel = th.exp(-0.5 * (x / sigma).pow(2))
        gaussian_kernel /= gaussian_kernel.sum()
        # Convolution matrix
        H = th.zeros(M, K)
        for i in range(M):
            H[i, i : i + kernel_size] = gaussian_kernel

        super().__init__(H)

    def __str__(self):
        return f"conv-gaussian-{self.sigma}"


# TODO think of good parametrization
class FourierSampling(LinOp):
    def __init__(self):
        pass


class FourierSampOperator(LinOp):
    def __init__(self, n_rows, K=100, seed=42):
        if n_rows < 4:
            raise ValueError("n_rows must be >= 4")  # there are 4 fixed low frequencies
        if n_rows > K // 2 - 3:
            # indices_hf contains K//2 - 10 indices, and we sample n_rows - 7 of them without replacement.
            raise ValueError("n_rows must be <= K//2 - 3")
        # for the __str__ method
        self.n_rows = n_rows

        # We want to sample n_rows of the DFT matrix (M' in the paper)
        rng = None if seed is None else th.Generator("cpu").manual_seed(seed)
        # Fixed low-frequency indices (incl. DC component)
        sampled_indices = th.tensor([0, 1, 2, 3])
        # Sample 3 rows from low frequencies
        if n_rows - len(sampled_indices) >= 3:
            indices_lf = th.arange(4, 10)
            sampled_indices_lf = th.sort(
                indices_lf[th.randperm(len(indices_lf), generator=rng)[:3]]
            ).values  # Randomly sample 3 indices
            sampled_indices = th.cat([sampled_indices, sampled_indices_lf])
        # Sample remaining rows from high frequencies
        if n_rows - len(sampled_indices) > 0:
            indices_hf = th.arange(10, K // 2)
            sampled_indices_hf = th.sort(
                indices_hf[
                    th.randperm(len(indices_hf), generator=rng)[
                        : n_rows - len(sampled_indices)
                    ]
                ]
            ).values  # Randomly sample the remaining indices
            sampled_indices = th.cat([sampled_indices, sampled_indices_hf])

        # Compute DFT matrix, and split into real/imaginary components
        F = th.fft.fft(th.eye(K))
        FR = F.real
        FI = F.imag
        # Construct Fourier sampling matrix
        # Factor 2 because we split the real and imaginary parts. One of the rows is the DC component, which has no imaginary part
        M = 2 * n_rows - 1
        H = th.zeros(M, K)
        H[0] = FR[sampled_indices[0]]  # DC component
        for i, sampled_idx in zip(range(1, M, 2), sampled_indices[1:]):
            H[i, :] = FR[sampled_idx, :]
            H[i + 1, :] = FI[sampled_idx, :]

        super().__init__(H)
        self.sampled_indices = sampled_indices

    # TODO : reimplement pad_output

    def __str__(self):
        return f"fourier-samp-{self.n_rows}"


class Sample(LinOp):
    def __init__(self, indices: th.Tensor):
        self.out_shape = (1,)
        self.in_shape = (1,)
        self.indices = indices

    def apply(self, x: th.Tensor) -> th.Tensor:
        rv = x[:, :, self.indices]
        return rv

    def applyT(self, y: th.Tensor) -> th.Tensor:
        rv = y.new_zeros((*y.shape[:2], self.indices.shape[0]))
        rv[:, :, self.indices] = y.clone()
        return rv


def finite_difference_matrix(L: int = 64):
    return th.eye(L) - th.diag_embed(th.ones(L - 1), -1)


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> th.Tensor:
    x = th.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, steps=kernel_size)
    kernel = th.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()
