
import sys
import torch
import numpy as np

def print_inline_every(iter, freq, term, msg):
    if iter % freq == 0 or iter == term - 1:
        if iter > 0: sys.stdout.write("\033[F\033[K")
        print(msg)

def tensor_to_image(x: torch.FloatTensor) -> np.ndarray:
    '''
    Convert a float image tensor to a byte image.
    
    Parameters
    ----------
    x : FloatTensor
        The input tensor of shape *(1, C, H, W)*.
        
    Returns
    -------
    ndarray[uint8]
        The byte image of shape *(H, W, C)*.
    '''
    x = x.squeeze(0)#.permute(1, 2, 0)   # 1 x C x H x W  -> H x W x C
    x = x.detach().cpu()                # Copy to CPU
    x = x.numpy().astype('uint8')       # Convert to byte array
    return x
    