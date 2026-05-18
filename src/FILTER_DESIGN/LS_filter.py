import numpy as np


def least_squares_bandpass(x, lowcut, highcut, fs=20050, order=4):
    """Designs an order-4 linear-phase FIR bandpass filter using the Least Squares

    method from scratch, and applies it to an input signal x.

    Parameters:
    -----------
    x       : np.ndarray, 1D input array (e.g., audio samples)
    lowcut  : float, lower edge of the passband in Hz
    highcut : float, upper edge of the passband in Hz
    fs      : float, sampling rate in Hz (default: 20050)
    order   : int, filter order (must be 4 for this specific specification)

    Returns:
    --------
    y       : np.ndarray, the filtered output signal
    h       : np.ndarray, the 5 computed filter coefficients
    """
    # 1. Input Validation
    if lowcut <= 0 or highcut >= fs / 2:
        raise ValueError(f"Cutoffs must satisfy: 0 < lowcut < highcut < {fs/2} Hz")
    if lowcut >= highcut:
        raise ValueError("lowcut must be strictly less than highcut")
    if order != 4:
        raise ValueError("This explicit structural setup is locked to order=4.")

    num_taps = order + 1  # Order 4 means 5 taps
    num_loops = (order // 2) + 1  # 3 unique coefficients due to symmetry

    # 2. Normalize cutoff frequencies to radians (0 to pi)
    omega_low = 2 * np.pi * lowcut / fs
    omega_high = 2 * np.pi * highcut / fs

    # 3. Create a dense frequency grid representing the spectrum
    num_points = 1000
    omega_grid = np.linspace(0, np.pi, num_points)

    # 4. Define the ideal desired brick-wall response D
    # 1 inside the passband, 0 everywhere else
    D = np.where((omega_grid >= omega_low) & (omega_grid <= omega_high), 1.0, 0.0)

    # 5. Construct the Design Matrix A
    # Each column represents a cosine basis function: cos(0*w), cos(1*w), cos(2*w)
    A = np.zeros((num_points, num_loops))
    for n in range(num_loops):
        A[:, n] = np.cos(n * omega_grid)

    # 6. Solve the Normal Equations: (A^T * A) * a = A^T * D
    # This solves the optimization problem: minimize ||A*a - D||^2
    ATA = A.T @ A
    ATD = A.T @ D
    a_coeffs = np.linalg.solve(ATA, ATD)

    # 7. Reconstruct the 5 symmetric FIR filter taps h[n] from the solved a_coeffs
    h = np.zeros(num_taps)
    center = order // 2  # index 2

    h[center] = a_coeffs[0]  # Center tap
    h[center - 1] = a_coeffs[1] / 2.0  # Tap 1
    h[center + 1] = a_coeffs[1] / 2.0  # Tap 3
    h[center - 2] = a_coeffs[2] / 2.0  # Tap 0
    h[center + 2] = a_coeffs[2] / 2.0  # Tap 4

    # 8. Filter Application (Direct-Form FIR Convolution Difference Equation)
    # y[n] = h[0]*x[n] + h[1]*x[n-1] + h[2]*x[n-2] + h[3]*x[n-3] + h[4]*x[n-4]
    # Implemented vectorially using pure NumPy array slicing shifts
    y = np.zeros_like(x, dtype=float)
    for k in range(num_taps):
        if k == 0:
            y += h[k] * x
        else:
            y[k:] += h[k] * x[:-k]

    return y, h
