import torch as th
from linop import LinOp
import optim


def perturb_and_map_tup(
    K: LinOp,
    mu: th.Tensor,
    var: th.Tensor,
    x_0: th.Tensor,
    dims: tuple[int, ...] = (1, 2),
):
    y = tuple(m + th.sqrt(v) * th.randn_like(m) for m, v in zip(mu, var))
    b = K.T @ (tuple(yy / v for yy, v in zip(y, var)))

    # TODO use early exit again
    class A(LinOp):
        def apply(self, x):
            return K.T @ tuple(kx / v for kx, v in zip(K @ x, var))

        def applyT(self, y: th.Tensor) -> th.Tensor:
            return y

    return optim.cg(A(), b, x_0, dims=dims, max_iter=10000)


def perturb_and_map(
    K: LinOp,
    mu: th.Tensor,
    var: th.Tensor,
    x_0: th.Tensor,
    tie_break=True,
    adj_K_sqr=None,
):

    y = mu + th.sqrt(var) * th.randn_like(mu)

    def A(x):
        res = K.T @ ((K @ x) / var)

        # TODO: Implement this part smarter
        if tie_break:
            res += th.mean(x, dim=(1, 2), keepdim=True)

        return res

    b = K.T @ (y / var)

    # Generate the samples
    # TODO: Implement this part smarter
    dims = (1,) if mu.dim() == 2 else (1, 2, 3)
    dims = (1, 2)

    def Ai(x, i):
        res = K.T @ ((K @ x) / var[i])
        res += th.mean(x, dim=(1, 2), keepdim=True)

        return res

    if adj_K_sqr is None:
        x_sample = optim.cg_(Ai, b, x_0, dims=dims)
    else:
        M = adj_K_sqr(1.0 / var)

        def M_invi(x, i):
            return x / M[i]

        x_sample = optim.precond_cg_(M_invi, Ai, b, x_0, dims=dims)
    return x_sample


def cholesky_map(
    K: LinOp,
    mu: th.Tensor,
    var: th.Tensor,
    x_0: th.Tensor,
    dims: tuple[int, ...] = (1, 2),
):
    K_mat = K

    mu1 = th.permute(mu[0], (0, 2, 1))
    mu2 = th.permute(mu[1], (0, 2, 1))
    mu_tot = th.cat((mu1, mu2), dim=1)

    var0 = th.permute(var[0], (0, 2, 1))
    var1 = th.permute(var[1], (0, 2, 1))
    var_tot = th.cat((var0, var1), dim=1)

    sigma = 1 / var_tot
    sigma_k = sigma * K_mat
    Kt_sigma_K = th.transpose(K_mat, 1, 2) @ sigma_k

    L = th.linalg.cholesky(Kt_sigma_K)
    z = th.randn_like(L[:, :, 0:1])
    y = th.linalg.solve_triangular(th.transpose(L, 1, 2), z, upper=True)

    Kt_sigma = th.transpose(sigma_k, 1, 2)
    Kt_sigma_mu0 = Kt_sigma @ mu_tot

    mu = th.linalg.solve(Kt_sigma_K, Kt_sigma_mu0)
    res = mu + y
    res = th.permute(res, (0, 2, 1))

    return res
