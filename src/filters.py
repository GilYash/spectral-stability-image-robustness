import torch


def radial_lowpass_mask(
    height: int,
    width: int,
    cutoff: float,
    device=None,
    dtype=torch.float32,
):
    """
    Create a smooth radial Gaussian low-pass mask in the Fourier domain.

    Parameters
    ----------
    height, width : int
        Spatial dimensions of the image.

    cutoff : float
        Controls how much frequency content is preserved.
        Expected range: (0, 1].

        Larger cutoff  -> weaker filtering.
        Smaller cutoff -> stronger filtering.

    device
        Torch device.

    dtype
        Output tensor dtype.

    Returns
    -------
    mask : torch.Tensor
        Shape: (height, width)
        Values approximately in [0, 1].
    """

    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must be in the range (0, 1].")

    # Normalized frequency coordinates.
    # After fftshift, frequency zero is in the image center.
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    radius_squared = xx**2 + yy**2

    # Smooth Gaussian low-pass.
    mask = torch.exp(
        -radius_squared / (2.0 * cutoff**2)
    )

    return mask


def apply_lowpass(
    image: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """
    Apply a smooth radial low-pass filter to an image.

    Parameters
    ----------
    image : torch.Tensor
        Image in [0, 1].

        Accepted shapes:
            (C, H, W)
            (B, C, H, W)

    cutoff : float
        Low-pass strength.
        Larger -> weaker filtering.
        Smaller -> stronger filtering.

    Returns
    -------
    filtered : torch.Tensor
        Same shape as input.
    """

    if image.ndim not in (3, 4):
        raise ValueError(
            "image must have shape (C, H, W) or (B, C, H, W)"
        )

    height, width = image.shape[-2:]

    mask = radial_lowpass_mask(
        height=height,
        width=width,
        cutoff=cutoff,
        device=image.device,
        dtype=image.dtype,
    )

    # FFT over spatial dimensions only.
    spectrum = torch.fft.fft2(
        image,
        dim=(-2, -1)
    )

    # Move zero frequency to the center.
    spectrum = torch.fft.fftshift(
        spectrum,
        dim=(-2, -1)
    )

    # Apply the same mask independently to every channel / batch item.
    filtered_spectrum = spectrum * mask

    # Move frequencies back before inverse FFT.
    filtered_spectrum = torch.fft.ifftshift(
        filtered_spectrum,
        dim=(-2, -1)
    )

    filtered = torch.fft.ifft2(
        filtered_spectrum,
        dim=(-2, -1)
    ).real

    # Small numerical overshoots are possible after inverse FFT.
    filtered = filtered.clamp(0.0, 1.0)

    return filtered


def make_spectral_views(
    image: torch.Tensor,
    cutoffs,
) -> torch.Tensor:
    """
    Construct the ordered spectral stability views.

    View 0 is always the untouched original image.

    Parameters
    ----------
    image : torch.Tensor
        Shape:
            (C, H, W)
        or
            (B, C, H, W)

        Expected pixel range: [0, 1].

    cutoffs : sequence of float
        Low-pass cutoff values.

        Example:
            [0.8, 0.6, 0.4, 0.25]

    Returns
    -------
    views : torch.Tensor

        For a single image:
            (K+1, C, H, W)

        For a batch:
            (B, K+1, C, H, W)

        The first view is exactly the original image.
    """

    if image.ndim not in (3, 4):
        raise ValueError(
            "image must have shape (C, H, W) or (B, C, H, W)"
        )

    # Important:
    # M_0 == 1, so the original image must be preserved exactly.
    views = [image]

    for cutoff in cutoffs:
        filtered = apply_lowpass(
            image=image,
            cutoff=float(cutoff),
        )
        views.append(filtered)

    if image.ndim == 3:
        return torch.stack(views, dim=0)

    # Batch case:
    # list elements have shape (B, C, H, W)
    # -> output (B, K+1, C, H, W)
    return torch.stack(views, dim=1)