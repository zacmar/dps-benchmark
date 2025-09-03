import torch as th
from typing import Callable
import linop
import math
import torch.nn.functional as F


def prox_step_conv(
    z: th.Tensor, kernel: th.Tensor, L: int, ys: th.Tensor, rho: float
) -> th.Tensor:
    Fy = th.fft.rfft(ys)
    kL = kernel.shape[0]
    kernel_padded = th.roll(
        F.pad(kernel, (0, L - kL)), shifts=(-kL // 2 + 1,), dims=(0,)
    )
    K_fft = th.fft.rfft(kernel_padded).unsqueeze(0)
    K_fftc = th.conj(K_fft) * Fy
    denom = rho + (K_fft.real**2 + K_fft.imag**2)
    numerator = rho * th.fft.rfft(z) + K_fftc
    return th.fft.irfft(numerator / denom, n=L)


def prox_l2(z: th.Tensor, y: th.Tensor, rho: float) -> th.Tensor:
    return (rho * z + y) / (rho + 1)


def prox_inpainting(
    z: th.Tensor, y: th.Tensor, rho: float, indices: th.Tensor
) -> th.Tensor:
    op = linop.Sample(indices)

    rv = z.clone()
    rv[:, :, indices] = prox_l2(z[:, :, indices], (op.T @ y)[:, :, indices], rho)
    return rv


def precond_cg_(M_inv, A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    to_do = th.arange(b.shape[0], device=b.device)
    x = x0.clone()
    r = b - A(x, to_do)
    z = M_inv(r, to_do).clone()
    p = z.clone()

    def ip(a, b):
        return th.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while len(to_do):
        Ap = A(p, to_do).clone()
        alpha = ip(r, z) / ip(p, Ap)
        x[to_do] += alpha * p
        r_prev = th.clone(r)
        r -= alpha * Ap
        cond = (r**2).sum(dim=dims).sqrt() > tol
        r, to_do = r[cond], to_do[cond]
        z_prev = th.clone(z)
        z = M_inv(r, to_do).clone()
        beta = ip(r, z) / ip(r_prev, z_prev)[cond]
        p = z + beta * p[cond]

    return x


# Batch preconditioned conjugate-gradient method for solving the system Ax = b with preconditioner M.
# This is equivalent to solving the system B^{-T} A B^{-1} y = B^{-T} where M = B^T B and B^{-1} y = x.
#
# Note that the preconditioned iterations are carried out on x and the true residual (and thus the tol argument refers
# to the norm of the residual of the original system Ax = b).
#
# See the following slides for more details:
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/cg.pdf
#
def precond_cg(M_inv, A, b, x0, tol=1e-4, dims=(1, 2, 3), max_iter=100):

    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    # Compute the residual vector
    # NOTE: Deep copy to not modify the input tensor
    x = x0.clone()
    r = b - A(x)

    # NOTE: We need a deep copy to protect from trivial identity operators
    z = M_inv(r).clone()

    # Compute the next search direction
    # NOTE: We need a deep copy, otherwise p and z point to the same tensor
    p = z.clone()

    def ip(a, b):
        return th.sum(a * b, dim=dims, keepdim=True)

    it = 0
    # Perform the iterations
    while th.max((r**2).sum(dim=dims).sqrt()) > tol:
        # Pick the step-size
        # NOTE: We need a deep copy to protect from trivial identity operators
        Ap = A(p).clone()
        alpha = ip(r, z) / ip(p, Ap)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        alpha[~th.isfinite(alpha)] = 0.0

        # Update the solution
        x += alpha * p

        # Update the residual vector
        # NOTE: We need a deep copy, otherwise r_prev and r point to the same tensor
        r_prev = th.clone(r)
        r -= alpha * Ap

        # Update z
        # NOTE: We need a deep copy, otherwise z_prev and z point to the same tensor
        z_prev = th.clone(z)

        # NOTE: We need a deep copy to protect from trivial identity operators
        z = M_inv(r).clone()

        # Update the search direction
        beta = ip(r, z) / ip(r_prev, z_prev)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        beta[~th.isfinite(beta)] = 0.0

        p = z + beta * p
        it += 1
        if it == max_iter:
            break

    return x


# Batch conjugate-gradient method for solving the system Ax = b, where A is a real,
# symmetric, and positive-definite matrix
def cg_(A, b, x0, tol=1e-4, dims=(1, 2, 3)):
    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    to_do = th.arange(b.shape[0], device=b.device)
    x = x0.clone()
    r = b - A(x, to_do)
    p = r.clone()

    def ip(a, b):
        return th.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    while len(to_do):
        print(len(to_do))
        Ap = A(p, to_do).clone()
        alpha = ip(r, r) / ip(p, Ap)
        x[to_do] += alpha * p
        r_prev = th.clone(r)
        r -= alpha * Ap
        cond = (r**2).sum(dim=dims).sqrt() > tol
        r, to_do = r[cond], to_do[cond]
        beta = ip(r, r) / ip(r_prev, r_prev)[cond]
        p = r + beta * p[cond]

    return x


def cg(A, b, x0, tol=1e-4, dims=(1, 2, 3), max_iter=100):

    # Basic sanity checks
    assert tol > 0.0
    assert len(x0.shape) > 1

    # Compute the residual vector
    # NOTE: Deep copy to not modify the input tensor
    # TODO change to "linop" notation again
    x = x0.clone()
    r = b - A @ x
    # Compute the next search direction
    # NOTE: We need a deep copy, otherwise r and p point to the same tensor and make the iterations wrong
    p = r.clone()

    def ip(a, b):
        return th.sum(a * b, dim=dims, keepdim=True)

    # Perform the iterations
    it = 0
    while th.max((r**2).sum(dim=dims).sqrt()) > tol:
        # Pick the step-size
        # NOTE: We need a deep copy to protect from trivial identity operators
        Ap = A @ p
        alpha = ip(r, r) / ip(p, Ap)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        alpha[~th.isfinite(alpha)] = 0.0

        # Update the solution
        x += alpha * p

        # Update the residual vector
        # NOTE: We need a deep copy, otherwise r_prev and r point to the same tensor and beta gets fixed to one
        r_prev = th.clone(r)
        r -= alpha * Ap

        # Update the search direction
        beta = ip(r, r) / ip(r_prev, r_prev)

        # NOTE: This is required in case one of the systems in batch mode is already solved, which causes that
        # particular solution to become nan in the next iteration.
        beta[~th.isfinite(beta)] = 0.0

        p = r + beta * p
        it += 1
        if it == max_iter:
            break
        # print(((r**2).sum(dim=dims).sqrt()).mean())
    return x


def pdhg1(
    x0: th.Tensor,
    y0: th.Tensor,
    K: linop.LinOp,
    tau: float,
    prox_tG: Callable[[th.Tensor], th.Tensor],
    sigma: float,
    prox_sF: Callable[[th.Tensor], th.Tensor],
    theta: float = 1.0,
    max_iter: int = 5000,
    callback: Callable[[th.Tensor, th.Tensor], None] = lambda *_: None,
) -> th.Tensor:
    x = x0.clone()
    x_bar = x0.clone()
    y = y0.clone()

    for _ in range(max_iter):
        x_prev = x.clone()
        y = prox_sF(y + sigma * K @ x_bar)
        x = prox_tG(x - tau * K.T @ y)
        x_bar = x + theta * (x - x_prev)
        callback(x, y)

    return x, y


def fista(
    x0: th.Tensor,
    nabla_f: Callable[[th.Tensor], th.Tensor],
    prox_g: Callable[[th.Tensor], th.Tensor],
    tau: float,
    max_iter: int = 10_000,
    callback: Callable[[th.Tensor], bool | None] = lambda _: False,
) -> th.Tensor:
    x = x0.clone()
    y = x0.clone()
    x_prev = x0.clone()
    t = 1

    for _ in range(max_iter):
        x = prox_g(y - tau * nabla_f(y))
        t_new = (1 + math.sqrt(1 + 4 * t**2)) / 2
        y = x + (t - 1) / t_new * (x - x_prev)
        t = t_new
        x_prev = x.clone()
        if callback(x):
            break

    return x
