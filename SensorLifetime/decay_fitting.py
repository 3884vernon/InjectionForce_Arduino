import numpy as np
import scipy.optimize as opt

def decay_model(t, I0, tau):
    """Exponential decay model."""
    return I0 * np.exp(-t / tau)

def fit_decay(times, intensities):
    """Fit the exponential decay model to the intensity data."""
    initial_guess = [intensities[0], 50.0]  # Adjust tau guess if needed
    params, _ = opt.curve_fit(decay_model, times, intensities, p0=initial_guess)
    I0, tau = params
    return tau  # Return the lifetime (tau)
