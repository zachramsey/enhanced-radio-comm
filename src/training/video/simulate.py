
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from .utils import tensor_to_image

def simulate_errors(data, p=None, r=None, k=1, h=0):
    '''
    Simulate errors using the Gilbert-Elliot burst error model.

    Creates a list representing an error signal. The error signal is expected
    to be added with modulo 2 addition to a binary signal, so 0 means no error
    occurred in that element and a 1 means an error did occur.

    Parameters
    ----------
    data : list[bytes]
        A list of input byte streams.
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
    errors : list[bytes]

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

    b = len(data)                           # Number of batches
    n = [len(data[i])*8 for i in range(b)]  # Number of bits in each batch
    _r, _k, _h = 1-r, 1-k, 1-h              # Pre-invert probabilities
    _p = p/(p+r)                            # Stationary probability of being in the Good state

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
    masks = [np.packbits(e_i) for e_i in e]
    # Get data from the byte streams
    datas = [np.frombuffer(d_i, dtype=np.uint8) for d_i in data]
    # XOR the error signal with the data
    outs = [np.bitwise_xor(d_i, m_i).tobytes() for d_i, m_i in zip(datas, masks)]
    
    return outs


def simulate_impairments(data: bytes, snr_db: float = 10, interference_prob: float = 0.1, flip_prob: float = 0.01, device: str = "cuda") -> bytes:
    """
    Apply realistic impairments (AWGN, fading, interference, burst errors) to a byte stream using PyTorch for GPU acceleration.
    
    Parameters
    ----------
    data : bytes
        The input byte stream.
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
    bytes
        The impaired byte stream.
    """
    # Move data to tensor on GPU
    bits = torch.from_numpy(np.unpackbits(np.frombuffer(data, dtype=np.uint8))).to(device, dtype=torch.float32)
    rand = np.random.rand(2)

    # Simulate additive white Gaussian noise
    if rand[0] < interference_prob:
        noise_std = torch.sqrt(torch.tensor(1 / (2 * 10 ** (snr_db / 10)), device=device))
        bits = bits + torch.randn_like(bits) * noise_std
    
    # Simulate Rayleigh fading
    if rand[1] < interference_prob:
        bits = bits * torch.abs(torch.randn_like(bits, device=device))

    bits = torch.sigmoid(bits).round().to(torch.uint8)
    
    # Move back to CPU, convert to bytes
    bits = np.packbits(bits.detach().cpu().numpy()).tobytes()
    return bits


def simulate_transmission(loader: DataLoader, encode, decode, plot_dir: str):
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
    fig, axs = plt.subplots(len(loader), 4)

    # Execute the methods
    for i, (data, _) in enumerate(loader):
        # Encode the input tensor
        z_string, y_string = encode(data)

        # Simulate transmission errors
        z_string_good = simulate_errors([z_string], p=0.13, r=0.84)
        y_string_good = simulate_errors([y_string], p=0.13, r=0.84)

        z_string_mid = simulate_errors([z_string], p=0.29, r=0.78)
        y_string_mid = simulate_errors([y_string], p=0.29, r=0.78)

        z_string_bad = simulate_errors([z_string], p=0.92, r=0.08)
        y_string_bad = simulate_errors([y_string], p=0.92, r=0.08)

        # Decode the received data
        dec_good = decode(z_string_good, y_string_good)
        dec_mid = decode(z_string_mid, y_string_mid)
        dec_bad = decode(z_string_bad, y_string_bad)

        # Convert tensors to numpy arrays for visualization
        img_in = tensor_to_image(data)
        img_good = tensor_to_image(dec_good)
        img_mid = tensor_to_image(dec_mid)
        img_bad = tensor_to_image(dec_bad)

        # Plot the images
        axs[i, 0].imshow(img_in)
        axs[i, 1].imshow(img_good)
        axs[i, 2].imshow(img_mid)
        axs[i, 3].imshow(img_bad)

        # Hide the axes
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 2].axis('off')
        axs[i, 3].axis('off')

        # Set titles over columns of the first row
        if i == 0:
            axs[i, 0].set_title("Input")
            axs[i, 1].set_title("Good Signal")
            axs[i, 2].set_title("Fair Signal")
            axs[i, 3].set_title("Bad Signal")

    fig.tight_layout()
    plt.savefig(f"{plot_dir}/Simulate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
