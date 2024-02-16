import torch


def rankfeat(x):
    """Apply RankFeat to the input tensor.

    The input tensor of shape (B, C, H, W) is reshaped to (B, C, H*W) and then SVD is
    computed. The component of the first singular value is removed and the tensor is
    reshaped back to the original shape.

    Args:
        x (torch.Tensor): torch tensor of shape (B, C, H, W) corresponding to
            activations.

    Returns:
        torch.Tensor: torch tensor of shape (B, C, H, W) corresponding to activations
            after RankFeat.
    """
    # Reshape tensor by flattening H and W dimensions
    B, C, H, W = x.size()
    feat1 = x.view(B, C, H * W)

    # Compute SVD, one per element in the batch
    u, s, v = torch.linalg.svd(feat1, full_matrices=False)

    # RankFeat: remove the matrix corresponding to the first singular value u1*s1*v1^T
    feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])

    # Reshape back to original shape
    feat1 = feat1.view(B, C, H, W)
    return feat1


def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v, dim=2, keepdim=True) + eps)


# Power Iteration as SVD substitute for accleration
def power_iteration(A, iter=20):
    u = (
        torch.FloatTensor(1, A.size(1))
        .normal_(0, 1)
        .view(1, 1, A.size(1))
        .repeat(A.size(0), 1, 1)
        .to(A)
    )
    v = (
        torch.FloatTensor(A.size(2), 1)
        .normal_(0, 1)
        .view(1, A.size(2), 1)
        .repeat(A.size(0), 1, 1)
        .to(A)
    )
    for _ in range(iter):
        v = _l2normalize(u.bmm(A)).transpose(1, 2)
        u = _l2normalize(A.bmm(v).transpose(1, 2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1, 2).bmm(v.transpose(1, 2))
    return sub


def rankfeat_power_iteration(x):
    """Apply RankFeat to the input tensor.

    The input tensor of shape (B, C, H, W) is reshaped to (B, C, H*W) and then power
    iteration is computed. The component of the first singular value is removed and the
    tensor is reshaped back to the original shape.

    Args:
        x (torch.Tensor): torch tensor of shape (B, C, H, W) corresponding to
            activations.

    Returns:
        torch.Tensor: torch tensor of shape (B, C, H, W) corresponding to activations
            after RankFeat.
    """
    # Reshape tensor by flattening H and W dimensions
    B, C, H, W = x.size()
    feat1 = x.view(B, C, H * W)

    # Compute power iteration to remove the first singular value component
    feat1 = feat1 - power_iteration(feat1, iter=20)

    # Reshape back to original shape
    feat1 = feat1.view(B, C, H, W)
    return feat1


# Test rankfeat
# Build a tensor of shape (2, 10, 10, 1) and apply rankfeat to it
# The tensor is built by stacking two 10x10 diagonal matrices. The first tensor has
# singular values 1-10 and the second 11-20.
a1 = torch.diagflat(torch.randperm(10, dtype=torch.float32) + 1).view(10, 10, 1)
a2 = torch.diagflat(torch.randperm(10, dtype=torch.float32) + 11).view(10, 10, 1)
a = torch.stack([a1, a2], dim=0)

print("Original tensor:")
print(a.shape)
print(a[0].squeeze())
print(a[1].squeeze())

# RankFeat SVD
b = rankfeat(a)
print("\nRankFeat tensor SVD:")
print(b[0].squeeze())
print(b[1].squeeze())

# RankFeat power iteration
c = rankfeat_power_iteration(a)
print("\nRankFeat tensor Power Iteration (rounded):")
print(c[0].squeeze().round().to(torch.int32))
print(c[1].squeeze().round().to(torch.int32))
