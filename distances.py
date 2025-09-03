import torch as th


def wasserstein_1_distance(x, emp_cdf1, emp_cdf2):
    # Basic sanity checks
    assert x.dim() == emp_cdf1.dim() == emp_cdf2.dim() == 1
    assert x.numel() == emp_cdf1.numel() == emp_cdf2.numel()

    # Compute the Wasserstein-1 distance via numerical integration
    y = th.abs(emp_cdf1 - emp_cdf2)[:-1]
    dx = th.diff(x)
    return th.sum(y * dx)


# Utility function that computes the Wasserstein-1 distance between two one-dimensional sample tensors
# NOTE: Assumes that X and Y are already sorted
def wasserstein_1_samples(X, Y):
    # Basic sanity checks
    assert X.dim() == Y.dim()
    assert X.numel() == Y.numel()

    # Compute the Wasserstein-1 distance via numerical integration
    return th.mean(th.abs(X - Y), dim=-1)


def zoh(points, vals, x, num_chunks=1):
    # Basic sanity checks
    assert points.dim() == vals.dim() == x.dim() == 1
    assert points.numel() == vals.numel()
    assert num_chunks >= 1

    # Calculate chunk size based on the number of chunks
    chunk_size = (
        x.numel() + num_chunks - 1
    ) // num_chunks  # Ensures the last chunk may be smaller

    # Initialize the output tensor
    vals_out = th.empty_like(x)

    # Process the chunks
    for i in range(0, x.numel(), chunk_size):
        end_i = min(i + chunk_size, x.numel())
        x_chunk = x[i:end_i]

        # Handle inner points
        left, right = points[:-1], points[1:]
        idx = (x_chunk.view(-1, 1) >= left.view(1, -1)) & (
            x_chunk.view(-1, 1) < right.view(1, -1)
        )
        idx = th.argmax(idx.to(th.int8), dim=1)
        vals_chunk = th.gather(vals, dim=0, index=idx)

        # Handle left boundary
        vals_chunk[x_chunk < points[0]] = 0

        # Handle right boundary
        vals_chunk[x_chunk >= points[-1]] = vals[-1]

        # Store the result in the output tensor
        vals_out[i:end_i] = vals_chunk

    return vals_out
