"""Acquisition-like appearance changes applied to a simulated slab (slab profile, bias, contrast, noise)."""
import numpy as np


def slab_profile(n_slices, slice_mm, ramp_mm):
    """Excitation profile across slices: signal falls off linearly over `ramp_mm` at both slab edges."""
    z = (np.arange(n_slices) - (n_slices - 1) / 2) * slice_mm
    if ramp_mm <= 0:
        return np.ones(n_slices)
    return np.clip((n_slices * slice_mm / 2 - np.abs(z)) / ramp_mm, 0, 1)


def bias_field(shape, amplitude, rng, n_terms=4):
    """Smooth multiplicative receive-bias field: exp of a few random low-frequency cosines (~1 cycle per FOV)."""
    grid = np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, n) for n in shape], indexing='ij'), axis=-1)
    log_bias = np.zeros(shape)
    for _ in range(n_terms):
        freq = rng.uniform(0.3, 1.2, size=3) * rng.choice([-1, 1], size=3)
        log_bias += np.cos(2 * np.pi * grid @ freq + rng.uniform(0, 2 * np.pi))
    return np.exp(amplitude * log_bias / np.sqrt(n_terms))


def rician_noise(image, snr, rng, mask=None):
    """Magnitude (Rician) noise with sigma = mean signal in `mask` / snr."""
    ref = image[mask].mean() if mask is not None and mask.any() else image[image > 0].mean()
    sigma = ref / snr
    return np.sqrt((image + rng.normal(0, sigma, image.shape)) ** 2 + rng.normal(0, sigma, image.shape) ** 2)


def apply_appearance(image, slice_mm, rng, ramp_mm=0.0, bias_amplitude=0.0, gamma=1.0, snr=None, mask=None):
    """Slab profile along the slice axis (last axis), bias field, contrast gamma and Rician noise."""
    out = image * slab_profile(image.shape[2], slice_mm, ramp_mm)[None, None, :]
    if bias_amplitude > 0:
        out = out * bias_field(image.shape, bias_amplitude, rng)
    scale = np.percentile(out[out > 0], 99) if (out > 0).any() else 1.0
    out = scale * np.clip(out / scale, 0, None) ** gamma
    if snr is not None:
        out = rician_noise(out, snr, rng, mask)
    return out.astype(np.float32)
