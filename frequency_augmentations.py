import torch
import random
def phase_shift_fourier(fourier_image, x=1.0, y=1.0):
    """
    Perform a random phase shift on a Fourier-image, constrained by maximum values.
    
    Parameters:
    fourier_image (torch.tensor): Complex-valued centered Fourier representation of an image
    x (float): Maximum amount to shift in x direction (0-2 range recommended due to 2pi periodicity)
    y (float): Maximum amount to shift in y direction (0-2 range recommended due to 2pi periodicity)
    
    Returns:
    torch.tensor: The phase-shifted Fourier-image
    """
    fourier_image = torch.fft.fftshift(fourier_image)
    rows, cols = fourier_image.shape
    
    # Generate random shift amounts within the specified ranges
    random_x = torch.rand(1).item() * x
    random_y = torch.rand(1).item() * y
    shift_x = random_y * torch.pi * rows
    shift_y = random_x * torch.pi * cols
    
    freq_x = torch.fft.fftfreq(cols)
    freq_y = torch.fft.fftfreq(rows)
    fx, fy = torch.meshgrid(freq_x, freq_y, indexing='ij')
    
    # Calculate and apply phase shift
    phase_shift = torch.exp(-1j * (shift_x * fx + shift_y * fy))
    shifted_fourier_image = torch.fft.fftshift(fourier_image * phase_shift)
    
    return shifted_fourier_image


def concentric_square_bandwidth_filter(fft, min_size=4, max_size=None, band_width=2):
    """
    Apply a bandwidth filter to remove a random concentric square band from the frequency spectrum.
    Both magnitude and phase are set to zero in the filtered band.
    
    Args:
    fft (torch.Tensor): Complex-valued frequency representation of an image (centered)
    min_size (int): Minimum size of the inner square of the band to remove
    max_size (int): Maximum size of the inner square of the band to remove (default: image size - 2*band_width)
    band_width (int): Width of the band to remove
    
    Returns:
    torch.Tensor: Filtered frequency spectrum
    """
    h, w = fft.shape
    assert h == w, "Input must be square"
    center = h // 2
    
    if max_size is None:
        max_size = h - 2*band_width
    #a
    # Ensure max_size is not larger than image size minus twice the band width
    max_size = min(max_size, h - 2*band_width)
    
    # Randomly choose the size of the inner square of the band
    inner_size = random.randint(min_size, max_size)
    outer_size = inner_size + 2*band_width
    
    # Ensure sizes are odd to guarantee perfect centering
    inner_size = inner_size if inner_size % 2 != 0 else inner_size + 1
    outer_size = outer_size if outer_size % 2 != 0 else outer_size + 1
    
    # Calculate the start and end indices for the inner and outer squares
    inner_start = center - inner_size // 2
    inner_end = center + inner_size // 2 + 1
    outer_start = center - outer_size // 2
    outer_end = center + outer_size // 2 + 1
    
    mask = torch.ones_like(fft, dtype=torch.bool)
    mask[outer_start:outer_end, outer_start:outer_end] = False
    mask[inner_start:inner_end, inner_start:inner_end] = True

    # Apply the mask to the frequency spectrum
    filtered_fft = torch.where(mask, fft, torch.complex(torch.zeros_like(fft.real), torch.zeros_like(fft.imag)))

    return filtered_fft


def mask_frequency_spectrum(spectrum, max_mask_percentage):
    # Ensure the max_mask_percentage is between 0 and 100
    max_mask_percentage = torch.clamp(torch.tensor(max_mask_percentage), 0, 100)
    
    # Generate a random mask percentage
    mask_percentage = torch.rand(1) * max_mask_percentage
    
    # Calculate the number of elements to mask
    num_elements = spectrum.numel()
    num_masked = int(num_elements * mask_percentage.item() / 100)
    
    # Create a flat copy of the spectrum
    flat_spectrum = spectrum.view(-1)
    
    # Randomly select indices to mask
    mask_indices = torch.randperm(num_elements)[:num_masked]
    
    # Create a mask tensor
    mask = torch.ones_like(flat_spectrum, dtype=torch.bool)
    mask[mask_indices] = False
    
    # Apply the mask
    masked_spectrum = flat_spectrum.clone()
    masked_spectrum[~mask] = 0
    
    # Reshape the spectrum back to its original shape
    masked_spectrum = masked_spectrum.view(spectrum.shape)
    
    return masked_spectrum