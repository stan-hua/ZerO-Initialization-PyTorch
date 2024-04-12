"""
zero_init.py

Description: Full PyTorch implementation of ZerO initialization.
"""

# Standard libraries
import math

# Non-standard libraries
import torch
from scipy.linalg import hadamard


def ZerO_Init_Linear(weight_matrix):
    """
    Perform ZerO initialization on a weight matrix.

    Note
    ----
    Can be used directly with Linear layers.

    Parameters
    ----------
    weight_matrix : torch.Tensor
        Weight matrix of shape (m, n), where n is the input dimension and m is
        the output dimension

    Returns
    -------
    torch.Tensor
        ZerO initialized weight matrix
    """
    # Weight matrix shape = (m, n)
    #   n ::= input dimension
    #   m ::= output dimension
    m, n = weight_matrix.shape
    device = weight_matrix.device

    if m <= n:
        init_matrix = torch.eye(m, n, device=device)
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = (
            torch.eye(m, p, device=device) @
            (torch.tensor(hadamard(p), device=device).float() / (2**(clog_m/2))) @
            torch.eye(p, n, device=device)
        )
    return init_matrix


def ZerO_Init_Conv2d(weight_matrix):
    """
    Perform ZerO initialization on a Conv2d weight matrix.

    Parameters
    ----------
    weight_matrix : torch.Tensor
        Weight matrix of shape (cout, cin, k1, k2). Assumes filters are square
        (i.e., k1=k2)

    Returns
    -------
    torch.Tensor
        ZerO initialized Conv2d weight matrix
    """
    # Conv. weight matrix shape = (cout, cin, k, k)
    #   cout ::= # of output channels (or # of filters)
    #   cin  ::= # of input channels
    #   k1   ::= filter size (first dimension)
    #   k2   ::= filter size (second dimension)

    cout, cin, k1, k2 = weight_matrix.shape
    if k1 != k2:
        raise NotImplementedError(
            "ZerO initialization on Conv2d layer is NOT currently implemented "
            "for non-square kernels!"
        )

    # Initialize the weight tensor
    device = weight_matrix.device
    weight_tensor = torch.zeros((cout, cin, k1, k1), device=device)

    # Apply ZerO initialization on center filter
    center = k1 // 2
    weight_tensor[:, :, center, center] = ZerO_Init_Linear(torch.zeros((cout, cin)))
    return weight_tensor


def ZerO_Init(layer, verbose=False):
    """
    Perform ZerO initialization (in-place) on a Linear or Conv2d layer.

    Parameters
    ----------
    layer : torch.nn.Module
        An arbitrary layer in a model
    """
    # CASE 1: Linear layer
    if isinstance(layer, torch.nn.Linear):
        layer.weight.data = ZerO_Init_Linear(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
    # CASE 2: Conv2d layer
    elif isinstance(layer, torch.nn.Conv2d):
        layer.weight.data = ZerO_Init_Conv2d(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
    # CASE 3: Not implemented
    elif verbose:
        print(f"ZerO initialization is NOT implemented for the layer `{type(layer)}`!")
