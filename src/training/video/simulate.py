import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from .utils import tensor_to_image

def simulate_errors(data: torch.Tensor, p=None, r=None, k=1, h=0) -> torch.Tensor:
    '''
    Simulate errors using the Gilbert-Elliot burst error model.

    Creates a list representing an error signal. The error signal is expected
    to be added with modulo 2 addition to a binary signal, so 0 means no error
    occurred in that element and a 1 means an error did occur.

    Parameters
    ----------
    data : Tensor
        The input data to simulate errors on.
    p : float
        Probability of transitioning from the Good state to the Bad state.
    r : float
        Probability of transitioning from the Bad state to the Good state.
    k : float
        Probability of no error occurring when in the Good state.
    h : float
        Probability of no error occurring when in the Bad state.

    Returns
    -------
    errors : Tensor
        The input data with errors added.

    Notes
    -----
    - This implementation is derived from the original by NTIA [1].
    - For 2.4 GHz network, the following parameters are reasonable [2]:
        - Good Link: p=0.13, r=0.84
        - Mid Link: p=0.29, r=0.78
        - Bad Link: p=0.92, r=0.08

    [1] Pieper J; Voran S, Relationships between Gilbert-Elliot Burst Error Model Parameters and Error Statistics, NTIA Technical Memo TM-23-565.  
    [2] A. Bildea, O. Alphand, F. Rousseau and A. Duda, "Link quality estimation with the Gilbert-Elliot model for wireless sensor networks," 2015 IEEE 26th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications (PIMRC), Hong Kong, China, 2015, pp. 2049-2054, doi: 10.1109/PIMRC.2015.7343635.
    '''
    if p is None and r is None:
        rand = np.random.random_sample()
        if rand < 0.4:
            return data
        elif rand < 0.7:
            p, r = 0.13, 0.84
        elif rand < 0.9:
            p, r = 0.29, 0.78
        else:
            p, r = 0.92, 0.08

    shape = data.shape
    device = data.device

    data = data.reshape(shape[0], -1)              # Flatten the data
    data = data.detach().cpu().byte().numpy()   # Convert to byte array

    b = len(data)                               # Number of batches
    n = [len(data[i])*8 for i in range(b)]      # Number of bits in each batch
    _r, _k, _h = 1-r, 1-k, 1-h                  # Pre-invert probabilities
    _p = p/(p+r)                                # Stationary probability of being in the Good state

    # Generate states
    rand = [np.random.random_sample((n_i,)) for n_i in n]   # Generate random numbers for states
    s = [np.zeros((n_i,), dtype=bool) for n_i in n]         # Initialize state arrays
    for i in range(b):
        s[i][0] = rand[i][0] < _p           # Set first state 
        for j in np.arange(1, n[i]):
            # Conditionally choose state based on previous state
            s[i][j] = rand[i][j] < (_r if s[i][j-1] else p)

    # Evaluate error
    if k == 1 and h == 0:
        e = s   # Error always occurs in bad state
    else:
        # Generate random numbers for errors
        rand = [np.random.random_sample((n_i)) for n_i in n]
        # Evaluate error based on states and random numbers
        e = [(_s & (_rand < _h)) | (~_s & (_rand < _k)) for _s, _rand in zip(s, rand)]

    # Pack error masks into bytes
    masks = np.stack([np.packbits(e_i) for e_i in e], axis=0)  # Convert to bytes

    data = np.bitwise_xor(data, masks)                  # Apply error signal to data
    data = torch.from_numpy(data).reshape(shape[0], -1) # Convert back to tensor
    data = data.float().to(device).reshape(shape)       # Reshape to original data shape

    return data


def simulate_impairments(data: torch.Tensor, snr_db: float = 10, interference_prob: float = 0.1, flip_prob: float = 0.01) -> torch.Tensor:
    """
    Apply realistic impairments (AWGN, fading) to a tensor.
    
    Parameters
    ----------
    data : Tensor
        Floating-point data in the range [-128, 127].
    snr_db : float
        Signal-to-noise ratio in dB for AWGN.
    interference_prob : float
        Probability of random bit flips.
    flip_prob : float
        Probability of burst errors.
    device : str
        Device to run computations on ("cuda" or "cpu").
    
    Returns
    -------
    Tensor
        The input data with impairments applied.
    """
    rand = np.random.rand(2)

    shape = data.shape
    device = data.device

    data = data.cpu().byte().reshape(data.shape[0], -1)  # Flatten the data
    
    # unpack bits
    bits = torch.zeros(data.shape[0], data.shape[1] * 8, dtype=torch.float32, device=data.device)
    for i in range(data.shape[0]):
        bits[i] = torch.from_numpy(np.unpackbits(data[i].numpy())).to(data.device, dtype=torch.float32)

    # Simulate additive white Gaussian noise
    if rand[0] < interference_prob:
        noise_std = torch.sqrt(torch.tensor(1 / (2 * 10 ** (snr_db / 10)), device=data.device))
        bits = bits + torch.randn_like(bits) * noise_std
    
    # Simulate Rayleigh fading
    if rand[1] < interference_prob:
        bits = bits * torch.abs(torch.randn_like(bits, device=data.device))

    # Discretize the data to binary
    bits = torch.round(torch.sigmoid(bits))
    bits = bits.reshape(shape[0], -1).byte()

    # Pack bits into bytes
    data = torch.zeros(shape[0], shape[1]*shape[2]*shape[3], dtype=torch.float32, device=data.device)
    for i in range(shape[0]):
        data[i] = torch.from_numpy(np.packbits(bits[i].cpu().numpy())).to(data.device, dtype=torch.float32)

    # Reshape to original data shape
    return data.reshape(*shape).to(device)


def add_uniform_noise(data: torch.Tensor, noise_level: float = 64.0) -> torch.Tensor:
    """
    Add uniform noise to floating point data.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor in range [-128, 127].
    noise_level : float
        The level of uniform noise to add.

    Returns
    -------
    torch.Tensor
        The noisy tensor.
    """
    return torch.clamp(data + torch.empty_like(data).uniform_(-noise_level-1, noise_level), -128.0, 127.0)


def add_gaussian_noise(data: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """
    Add Gaussian noise to floating point data.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor.
    scale : float
        The scale of the Gaussian noise to add.

    Returns
    -------
    torch.Tensor
        The noisy tensor.
    """
    return torch.clamp(data + torch.empty_like(data).normal_(-scale, scale), -128.0, 127.0)


def plot_images(images: dict[str, np.ndarray], plot_dir: str):
    """
    Plot images in a grid. Each row corresponds to one input image, and
    each column corresponds to different operations applied to the input.
    
    Parameters
    ----------
    images : dict[str, np.ndarray]
        Dictionary containing images to plot.  
        Keys: column headers (e.g., 'Input', 'Op1', ...)  
        Values: arrays of shape (n_inputs, H, W, 3) with dtype np.uint8.
    plot_dir : str
        Directory to save the plot.
    """
    


def simulate_transmission(loader: DataLoader, model, plot_dir: str, device: str = "cuda"):
    '''
    Simulate the transmission of data through a noisy channel and visualize the results.

    Parameters
    ----------
    loader : DataLoader
        DataLoader object containing the input data of shape *(1, C, H, W)*.
    encode : method
        Callback function to encode the target data.
    decode : method
        Callback function to decode the 'transmitted' data.
    plot_dir : str
        Directory to save the plots.
    '''
    images = {}

    # Execute the methods
    for i, (data, _) in enumerate(loader):
        data = data.to(device)

        # Convert tensor to image
        input = tensor_to_image(data.unsqueeze(0))

        # Simulate clean transmission
        clean = tensor_to_image(model(data, None)[0].unsqueeze(0))

        # Simulate burst errors
        burst = tensor_to_image(model(data, simulate_errors, p=0.13, r=0.84)[0].unsqueeze(0))

        # Simulate channel impairments
        impair = tensor_to_image(model(data, simulate_impairments)[0].unsqueeze(0))

        # Add uniform noise
        uniform = tensor_to_image(model(data, add_uniform_noise)[0].unsqueeze(0))

        # Add Gaussian noise
        gaussian = tensor_to_image(model(data, add_gaussian_noise)[0].unsqueeze(0))

        # Store images in dictionary
        if i == 0:
            images['Input'] = input
            images['Clean'] = clean
            images['Errors'] = burst
            images['Impaired'] = impair
            images['Uniform'] = uniform
            images['Gaussian'] = gaussian
        else:
            images['Input'] = np.concatenate((images['Input'], input), axis=0)
            images['Clean'] = np.concatenate((images['Clean'], clean), axis=0)
            images['Errors'] = np.concatenate((images['Errors'], burst), axis=0)
            images['Impaired'] = np.concatenate((images['Impaired'], impair), axis=0)
            images['Uniform'] = np.concatenate((images['Uniform'], uniform), axis=0)
            images['Gaussian'] = np.concatenate((images['Gaussian'], gaussian), axis=0)

    # Plot images
    col_headers = list(images.keys())
    n_cols = len(col_headers)
    n_inputs, height, width, _ = images[col_headers[0]].shape

    fig_width = min(n_cols * 2.5 * max(1, width / height), 50)
    fig_height = min(n_inputs * 2.5 * max(1, height / width), 50)

    fig, axes = plt.subplots(
        n_inputs, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
    )

    for col_idx, header in enumerate(col_headers):
        axes[0, col_idx].set_title(header, fontsize=32)
        img_stack = images[header]
        for row_idx in range(n_inputs):
            ax = axes[row_idx, col_idx]
            img_data = img_stack[row_idx]
            ax.imshow(img_data)
            ax.axis('off')

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

    # Save the plot
    plt.savefig(f"{plot_dir}/Simulate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close(fig)
    