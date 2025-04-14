
import torch
from torch import Tensor

__all__ = ["encode_with_indexes", "decode_with_indexes"]

def encode_with_indexes(
    x: Tensor,
    indexes: Tensor,
    cdfs: Tensor,
    cdfs_sizes: Tensor,
    offsets: Tensor
) -> Tensor:
    """
    Encodes a tensor using indexes and CDFs.
    
    Parameters
    ----------
    x : Tensor
        The input tensor to encode.
    indexes : Tensor
        The indexes to use for encoding.
    cdfs : Tensor
        The CDFs to use for encoding.
    cdfs_sizes : Tensor
        The sizes of the CDFs.
    offsets : Tensor
        The offsets to use for encoding.

    Returns
    -------
    Tensor
        The encoded tensor.
    """
    return torch.ops.cpp_exts.encode_with_indexes(
        x, indexes, cdfs, cdfs_sizes, offsets
    )


def decode_with_indexes(
    x: Tensor,
    indexes: Tensor,
    cdfs: Tensor,
    cdfs_sizes: Tensor,
    offsets: Tensor
) -> Tensor:
    """
    Decodes a tensor using indexes and CDFs.
    
    Parameters
    ----------
    x : Tensor
        The input tensor to decode.
    indexes : Tensor
        The indexes to use for decoding.
    cdfs : Tensor
        The CDFs to use for decoding.
    cdfs_sizes : Tensor
        The sizes of the CDFs.
    offsets : Tensor
        The offsets to use for decoding.

    Returns
    -------
    Tensor
        The decoded tensor.
    """
    return torch.ops.cpp_exts.decode_with_indexes(
        x, indexes, cdfs, cdfs_sizes, offsets
    )


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("cpp_exts::RansEncoder::encode_with_indexes")
def _(x: Tensor,
      indexes: Tensor,
      cdfs: Tensor,
      cdfs_sizes: Tensor,
      offsets: Tensor
) -> Tensor:
    # Check dtypes.
    torch._check(x.dtype in (torch.int, torch.int32), "x must be int32")
    torch._check(indexes.dtype in (torch.int, torch.int32), "indexes must be an integer tensor.")
    torch._check(cdfs.dtype in (torch.int, torch.int32), "cdfs must be an integer tensor.")
    torch._check(cdfs_sizes.dtype in (torch.int, torch.int32), "cdfs_sizes must be an integer tensor.")
    torch._check(offsets.dtype in (torch.int, torch.int32), "offsets must be an integer tensor.")

    # Check dimensions.
    torch._check(x.dim() == 1, "x should be a 1D tensor.")
    torch._check(indexes.dim() == 1, "indexes should be a 1D tensor.")
    torch._check(indexes.size(0) == x.size(0), "indexes should have the same number of elements as x.")
    torch._check(cdfs.dim() == 2, "cdfs must be a 2D tensor.")
    torch._check(cdfs_sizes.dim() == 1 or offsets.dim() == 1, "cdfs_sizes and offsets must be 1D tensors.")
    torch._check(cdfs.size(0) == cdfs_sizes.size(0) or cdfs.size(0) == offsets.size(0), "The first dimension of cdfs must equal the length of cdfs_sizes and offsets.")

    return torch.empty(0, dtype=torch.uint8, device=x.device)


@torch.library.register_fake("cpp_exts::RansDecoder::decode_with_indexes")
def _(encoded: torch.Tensor,
      indexes: torch.Tensor,
      cdfs: torch.Tensor,
      cdfs_sizes: torch.Tensor,
      offsets: torch.Tensor
) -> torch.Tensor:
    # Check dtypes.
    torch._check(encoded.dtype == torch.uint8, "encoded must be a uint8 tensor.")
    torch._check(indexes.dtype in (torch.int, torch.int32), "indexes must be an integer tensor.")
    torch._check(cdfs.dtype in (torch.int, torch.int32), "cdfs must be an integer tensor.")
    torch._check(cdfs_sizes.dtype in (torch.int, torch.int32), "cdfs_sizes must be an integer tensor.")
    torch._check(offsets.dtype in (torch.int, torch.int32), "offsets must be an integer tensor.")

    # Check dimensions.
    torch._check(encoded.dim() == 1, "encoded should be a 1D tensor.")
    torch._check(indexes.dim() == 1, "indexes should be a 1D tensor.")
    torch._check(cdfs.dim() == 2, "cdfs must be a 2D tensor.")
    torch._check(cdfs_sizes.dim() == 1, "cdfs_sizes must be a 1D tensor.")
    torch._check(offsets.dim() == 1, "offsets must be a 1D tensor.")

    # Check shape consistency.
    torch._check(indexes.numel() > 0, "indexes should have at least one element.")
    torch._check(cdfs.size(0) == cdfs_sizes.size(0) and cdfs.size(0) == offsets.size(0),
                 "The first dimension of cdfs must equal the lengths of cdfs_sizes and offsets.")

    return torch.empty(indexes.shape, dtype=torch.int, device=encoded.device)
