import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .utils import tensor_to_image

# def simulate_errors(data: torch.Tensor, p=None, r=None, k=1, h=0) -> torch.Tensor:
#     '''
#     Simulate errors using the Gilbert-Elliot burst error model.

#     Creates a list representing an error signal. The error signal is expected
#     to be added with modulo 2 addition to a binary signal, so 0 means no error
#     occurred in that element and a 1 means an error did occur.

#     Parameters
#     ----------
#     data : Tensor
#         The input data to simulate errors on.
#     p : float
#         Probability of transitioning from the Good state to the Bad state.
#     r : float
#         Probability of transitioning from the Bad state to the Good state.
#     k : float
#         Probability of no error occurring when in the Good state.
#     h : float
#         Probability of no error occurring when in the Bad state.

#     Returns
#     -------
#     errors : Tensor
#         The input data with errors added.

#     Notes
#     -----
#     - This implementation is derived from the original by NTIA [1].
#     - For 2.4 GHz network, the following parameters are reasonable [2]:
#         - Good Link: p=0.13, r=0.84
#         - Mid Link: p=0.29, r=0.78
#         - Bad Link: p=0.92, r=0.08

#     [1] Pieper J; Voran S, Relationships between Gilbert-Elliot Burst Error Model Parameters and Error Statistics, NTIA Technical Memo TM-23-565.  
#     [2] A. Bildea, O. Alphand, F. Rousseau and A. Duda, "Link quality estimation with the Gilbert-Elliot model for wireless sensor networks," 2015 IEEE 26th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications (PIMRC), Hong Kong, China, 2015, pp. 2049-2054, doi: 10.1109/PIMRC.2015.7343635.
#     '''
#     if p is None and r is None:
#         rand = np.random.random_sample()
#         if rand < 0.4:
#             return data
#         elif rand < 0.7:
#             p, r = 0.13, 0.84
#         elif rand < 0.9:
#             p, r = 0.29, 0.78
#         else:
#             p, r = 0.92, 0.08

#     shape = data.shape
#     device = data.device

#     data = data.reshape(shape[0], -1)   # Flatten the data
#     data = data.int().view(torch.uint8) # Reinterpret as uint8
#     data = data.detach().cpu().numpy()  # Convert to numpy array

#     b = data.shape[0]                   # Number of batches
#     n = data.shape[1] * 8               # Number of bits in each batch
#     _r, _k, _h = 1-r, 1-k, 1-h          # Pre-invert probabilities
#     _p = p/(p+r)                        # Stationary probability of being in the Good state

#     # Generate states
#     s = [np.zeros((n_i,), dtype=bool) for n_i in n]         # Initialize state arrays
#     for i in range(b):
#         rand = np.random.random_sample((n,))
#         s[i][0] = rand[0] < _p           # Set first state 
#         for j in np.arange(1, n[i]):
#             # Conditionally choose state based on previous state
#             s[i][j] = rand[j] < (_r if s[i][j-1] else p)

#     # Evaluate error
#     if k == 1 and h == 0:
#         e = s   # Error always occurs in bad state
#     else:
#         # Generate random numbers for errors
#         rand = [np.random.random_sample((n_i)) for n_i in n]
#         # Evaluate error based on states and random numbers
#         e = [(_s & (_rand < _h)) | (~_s & (_rand < _k)) for _s, _rand in zip(s, rand)]

#     # Pack error masks into bytes
#     masks = np.stack([np.packbits(e_i) for e_i in e], axis=0)  # Convert to bytes

#     data = np.bitwise_xor(data, masks)                  # Apply error signal to data
#     data = torch.from_numpy(data).reshape(shape[0], -1) # Convert back to tensor
#     data = data.view

#     return data


def burst_errors(
        tensor: torch.Tensor,
        p_gb: float,
        p_bg: float,
        ber_g: float,
        ber_b: float,
        s_init: bool = True
    ) -> torch.Tensor:
    """
    Simulates burst errors on a tensor using the Gilbert-Elliott model.

    Operates on the bit representation of the float32 tensor elements.

    Parameters
    ----------
    tensor : (torch.Tensor)
        The input tensor (assumed float32).
    p_gb : (float)
        Probability of transitioning from Good state to Bad state per bit.
    p_bg : (float)
        Probability of transitioning from Bad state to Good state per bit.
    ber_g : (float)
        Bit Error Rate (BER) when in the Good state.
    ber_b : (float)
        Bit Error Rate (BER) when in the Bad state.
    s_init : (bool)
        Initial state (True=Bad, False=Good).

    Returns
    -------
    torch.Tensor
        The tensor with simulated Gilbert-Elliott burst errors.
    """
    original_shape = tensor.shape
    device = tensor.device

    # Reinterpret tensor as 32-bit integers and flatten
    int_representation = tensor.contiguous().view(torch.int32)
    flat_int_tensor = int_representation.flatten()
    num_elements = flat_int_tensor.numel()
    total_bits = num_elements * 32

    # Simulate state transitions and bit errors
    states = torch.empty(total_bits, dtype=torch.bool, device=device) # True=Bad, False=Good
    bers = torch.empty(total_bits, dtype=torch.float32, device=device)

    s_curr = s_init # Current state (True=Bad, False=Good)

    # Generate random numbers for state transitions and BER checks upfront
    state_transitions_rand = torch.rand(total_bits, device=device)
    ber_rand = torch.rand(total_bits, device=device)

    # Loop through each bit position
    for i in range(total_bits):
        states[i] = s_curr
        if s_curr:
            bers[i] = ber_b
            # Check for transition B -> G
            if state_transitions_rand[i] < p_bg:
                s_curr = False # Transition to Good
        else: # Current state is Good
            bers[i] = ber_g
            # Check for transition G -> B
            if state_transitions_rand[i] < p_gb:
                s_curr = True # Transition to Bad

    # --- Generate Bit Error Mask ---
    # Determine which bits flip based on the BER for their state
    error_occurs_flat = ber_rand < bers # Boolean mask for each bit

    # Reshape the flat error mask to match (num_elements, 32)
    error_occurs_bits = error_occurs_flat.view(num_elements, 32)

    # --- Efficiently Create Integer Error Mask ---
    # Create powers of 2 for each bit position [1, 2, 4, ..., 2^(N-1)]
    powers_of_2 = (2**torch.arange(32, device=device, dtype=torch.int32)).unsqueeze(0) # Shape (1, 32)

    # Convert boolean mask to int (0 or 1) and multiply by powers of 2
    # Then sum along the bit dimension to get the final integer mask per element
    # error_occurs_bits.int() -> shape (num_elements, 32)
    # powers_of_2             -> shape (1, 32)
    # Broadcasting applies powers_of_2 to each element mask
    int_error_mask = torch.sum(error_occurs_bits.int() * powers_of_2, dim=1) # Shape (num_elements)

    # --- Apply Errors and Reshape ---
    # Apply errors using bitwise XOR
    noisy_flat_int = flat_int_tensor ^ int_error_mask

    # Reshape back to the original tensor's shape
    noisy_int_reshaped = noisy_flat_int.view(original_shape)

    # Reinterpret the modified integer bits back into the original float dtype
    noisy_tensor = noisy_int_reshaped.view(tensor.dtype)

    return noisy_tensor


def bit_errors(data: torch.Tensor, ber: float = 0.01) -> torch.Tensor:
    """
    Simulate bit errors on a tensor using a simple bit-flipping model.
    Operates on the bit representation of the float32 tensor elements.

    Parameters
    ----------
    data : torch.Tensor
        The input tensor (assumed float32).
    ber : float
        Bit Error Rate (BER) for the simulation.

    Returns
    -------
    torch.Tensor
        The tensor with simulated bit errors.
    """
    device = data.device

    data_int = data.contiguous().view(torch.int32)
    error_mask = torch.zeros_like(data_int)
    for i in range(32):
        bit_prob_mask = torch.rand_like(data) < ber
        bit_val = torch.tensor(1, dtype=torch.int32) << i
        curr_mask = torch.where(
            bit_prob_mask,
            bit_val,
            torch.tensor(0, dtype=torch.int32)
        )
        error_mask |= curr_mask
    
    data_int = data_int ^ error_mask
    return data_int.view(data.dtype)


# @torch.compiler.disable
def simulate_transmission(step: int, loader: DataLoader, model: str, encoder, decoder, exec_encoder, exec_decoder, plot_dir, device:str = 'cpu'):
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

    # Execute the methods
    images = {}
    for i, (data, _) in enumerate(loader):
        data = data.to(device)

        # Convert tensor to image
        input = np.expand_dims(tensor_to_image(data), axis=0)

        # Simulate clean transmission
        clean = np.expand_dims(tensor_to_image(model(data, None)[0]), axis=0)

        # Add uniform noise
        uniform_noise = lambda x: x + torch.empty_like(x).uniform_(-20, 20)
        uniform = np.expand_dims(tensor_to_image(model(data, uniform_noise)[0]), axis=0)

        # Add gaussian noise
        gaussian_noise = lambda x: x + torch.empty_like(x).normal_(0, 20)
        gaussian = np.expand_dims(tensor_to_image(model(data, gaussian_noise)[0]), axis=0)

        # Simulate transmission with torch models
        z_string, y_string = encoder((data.squeeze(0).permute(1, 2, 0) * 255.0).round())
        torch_sim = np.expand_dims(decoder(z_string, y_string).detach().cpu().numpy().astype('uint8'), axis=0)

        # Simulate transmission with executorch models
        if exec_encoder is not None and exec_decoder is not None:
            z_string, y_string = exec_encoder((data.squeeze(0).permute(1, 2, 0) * 255.0).round().cpu().byte().contiguous())
            exec_sim = np.expand_dims(exec_decoder(z_string, y_string).numpy().astype('uint8'), axis=0)

        # # Simulate moderate burst errors
        # mod_burst = np.expand_dims(tensor_to_image(model(data, burst_errors, 10e-6, 10e-3, 10e-7, 10e-3)[0]), axis=0)

        # # Simulate difficult burst errors
        # dif_burst = np.expand_dims(tensor_to_image(model(data, burst_errors, 10e-4, 10e-4, 10e-5, 0.25)[0]), axis=0)   

        # Store images in dictionary
        if i == 0:
            images['Input'] = input
            images['No Noise'] = clean
            images['Uniform'] = uniform
            images['Gaussian'] = gaussian
            images['Simulated'] = torch_sim
            if exec_encoder is not None and exec_decoder is not None:
                images['Executorch'] = exec_sim
            # images['Mod Burst'] = mod_burst
            # images['Dif Burst'] = dif_burst            
        else:
            images['Input'] = np.concatenate((images['Input'], input), axis=0)
            images['No Noise'] = np.concatenate((images['No Noise'], clean), axis=0)
            images['Uniform'] = np.concatenate((images['Uniform'], uniform), axis=0)
            images['Gaussian'] = np.concatenate((images['Gaussian'], gaussian), axis=0)
            images['Simulated'] = np.concatenate((images['Simulated'], torch_sim), axis=0)
            if exec_encoder is not None and exec_decoder is not None:
                images['Executorch'] = np.concatenate((images['Executorch'], exec_sim), axis=0)
            # images['Mod Burst'] = np.concatenate((images['Mod Burst'], mod_burst), axis=0)
            # images['Dif Burst'] = np.concatenate((images['Dif Burst'], dif_burst), axis=0)

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
    plt.savefig(f"{plot_dir}/Simulate_{step}.png")
    plt.close(fig)
    