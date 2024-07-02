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


def mask_frequency_spectrum(spectrum, max_mask_percentage=75):
    """
    This function sets a random percentage between 0 and max_mask_percentage from the frequency spectrum to 0. It is 
    inspired by the masked autoencoder are scalable vision transformers, where it was shown that we can remove 75% of data in the
    image domain and still receive reasonable features.
    Input: The frequency Spectrum (centered or uncentered)
    Output:The masked frequency spectrum.
    """
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



def spiral_matrix(matrix):
    """
    A helper function for the rearrange_matrix function. It moves through a given matrix in kind of a spiral, and returns a 1D matrix 
    of the matrix values went through in a spiral.
    Example:
    Input:
        [[1,2,3],
         [4,5,6],
         [7,8,9]]
    Output: [5,4,7,8,9,6,3,2,1,4]
    """
    # if not matrix:
    #     return []
    
    result = []
    n = len(matrix)
    top, bottom, left, right = 0, n - 1, 0, n - 1
    
    while len(result) < n * n:
        # Traverse right
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        if top <= bottom:
            # Traverse left
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1
        
        if left <= right:
            # Traverse up
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return torch.tensor(result[::-1])

def rearrange_matrix(original_matrix, patch_size=7):
    """
    A function to rearrange the frequency spectrum into low and high frequency patches. 
    It starts by taking the first patch_size*patch_size inner (low) frequencies and building a patch of size patch_size*patch_size out of it.
    Then it takes the next most inner frequencies and builds the next patch. The first patch will be the top left patch of 
    the new matrix, the next will placed right next to first one etc. until 1 row is filled.
    This function is supposed to bring the frequency spectrum in a form that can be better processed by a Vision transformer.

    Input: The centered fourier Spectrum of a matrix
    Output: The rearranged frequency spectrum 

    Parameters:
    originial_matrix: The centered frequency spectrum
    patch_size: One side of the patches. The dim of the original_matrix must be divisible by patch_size
    """
    # Step 1: Transform the matrix using the spiral method
    flattened = spiral_matrix(original_matrix)
    
    # Step 2: Calculate dimensions
    n = len(original_matrix)
    total_elements = n * n
    patches_per_row = n // patch_size
    
    # Step 3: Create patches
    patches = []
    for i in range(0, total_elements, patch_size * patch_size):
        patch = torch.tensor(flattened[i:i + patch_size * patch_size])
        patch = patch.reshape(patch_size, patch_size)
        patches.append(patch)
    
    # Step 4: Rearrange patches
    result = torch.zeros((n, n), dtype=torch.complex64)
    for i in range(patches_per_row):
        for j in range(patches_per_row):
            patch_index = i * patches_per_row + j
            row_start = i * patch_size
            col_start = j * patch_size
            result[row_start:row_start + patch_size, col_start:col_start + patch_size] = patches[patch_index]
    
    return result