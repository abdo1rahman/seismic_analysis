"""
DIGITAL BANDPASS FILTER DESIGN
4th-Order Butterworth BPF using Impulse Invariance Method
Original Specs: 1000-4000 Hz passband, Fs = 20050 Hz

NOW WITH REUSABLE FUNCTION - change lowcut and highcut as needed!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def fft_bandpass_filter(
    x: np.ndarray | list,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
    plot_response: bool = True,
):
    """
    FFT-based Bandpass Filter using Butterworth prototype + Impulse Invariance

    Parameters:
    -----------
    x : array_like
        Input signal
    fs : float
        Sampling rate in Hz
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    order : int
        Filter order (must be even, default 4)
    plot_response : bool
        Whether to plot frequency response

    Returns:
    --------
    y : ndarray
        Filtered output signal
    b, a : ndarray
        Filter coefficients (numerator, denominator)
    """

    # Validate inputs
    if lowcut <= 0 or highcut >= fs / 2:
        raise ValueError(
            f"Cutoffs must satisfy: 0 < lowcut < highcut < fs/2 = {fs/2} Hz"
        )
    if lowcut >= highcut:
        raise ValueError("lowcut must be less than highcut")
    if order % 2 != 0:
        raise ValueError("Order must be even for bandpass filter")

    # ============================================
    # STEP 1: Analog Prototype Design
    # ============================================
    omega1 = 2 * np.pi * lowcut
    omega2 = 2 * np.pi * highcut
    omega0 = np.sqrt(omega1 * omega2)  # Geometric center frequency
    BW = omega2 - omega1  # Bandwidth

    # Lowpass prototype order
    N_lp = order // 2

    # Butterworth lowpass poles
    k = np.arange(1, N_lp + 1)
    poles_lp = np.exp(1j * np.pi * (2 * k + N_lp - 1) / (2 * N_lp))
    # ============================================
    # STEP 2: Lowpass -> Bandpass Transform
    # ============================================
    # Transformation: s -> (s^2 + Omega0^2) / (s * BW)
    poles_bp = []
    for p in poles_lp:
        a, b, c = 1, -p * BW, omega0**2
        discriminant = b**2 - 4 * a * c
        s1 = (-b + np.sqrt(discriminant)) / (2 * a)
        s2 = (-b - np.sqrt(discriminant)) / (2 * a)
        poles_bp.extend([s1, s2])
    poles_bp = np.array(poles_bp)

    # Build analog transfer function
    A_analog = np.poly(poles_bp)
    s_test = 1j * omega0
    A_test = np.polyval(A_analog, s_test)
    K_analog = np.abs(A_test) / (omega0**2)
    B_analog = np.array([K_analog] + [0] * (N_lp + 1))

    # ============================================
    # STEP 3: Partial Fraction Expansion
    # ============================================
    r, p, k_term = signal.residue(B_analog, A_analog)

    # ============================================
    # STEP 4: Impulse Invariance Transformation
    # ============================================
    T = 1 / fs
    z_poles = np.exp(p * T)
    z_residues = T * r

    # Combine into H(z) = B(z)/A(z)
    B_digital = np.array([0.0], dtype=complex)
    A_digital = np.array([1.0], dtype=complex)

    for i in range(len(z_poles)):
        term_num = np.array([z_residues[i]], dtype=complex)
        term_den = np.array([1, -z_poles[i]], dtype=complex)

        # Pad to same length before adding
        B_den = np.polymul(B_digital, term_den)
        A_num = np.polymul(A_digital, term_num)

        max_len = max(len(B_den), len(A_num))
        B_den = np.pad(B_den, (0, max_len - len(B_den)), mode="constant")
        A_num = np.pad(A_num, (0, max_len - len(A_num)), mode="constant")

        new_B = B_den + A_num
        new_A = np.polymul(A_digital, term_den)

        B_digital = new_B
        A_digital = new_A

    # Normalize so A[0] = 1
    B_digital = np.real(B_digital / A_digital[0])
    A_digital = np.real(A_digital / A_digital[0])

    # ============================================
    # STEP 5: Apply Filter to Input Signal
    # ============================================
    y = signal.lfilter(B_digital, A_digital, x)

    # ============================================
    # STEP 6: Plot Frequency Response (optional)
    # ============================================
    if plot_response:
        w, h = signal.freqz(B_digital, A_digital, worN=8192)
        freqs_hz = w * fs / (2 * np.pi)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"FFT Bandpass Filter: {lowcut}-{highcut} Hz @ Fs={fs} Hz (Order={order})",
            fontsize=13,
            fontweight="bold",
        )

        # Magnitude Response (dB)
        ax1 = axes[0, 0]
        ax1.plot(freqs_hz, 20 * np.log10(np.abs(h) + 1e-10), "b-", lw=1.5)
        ax1.axvline(lowcut, color="r", ls="--", alpha=0.7, label=f"low={lowcut}Hz")
        ax1.axvline(highcut, color="r", ls="--", alpha=0.7, label=f"high={highcut}Hz")
        ax1.axhline(-3, color="orange", ls="-.", alpha=0.6, label="-3dB")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("|H(f)| (dB)")
        ax1.set_title("Magnitude Response")
        ax1.set_xlim([0, fs / 2])
        ax1.set_ylim([-60, 5])
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Phase Response
        ax2 = axes[0, 1]
        ax2.plot(freqs_hz, np.unwrap(np.angle(h)) * 180 / np.pi, "g-", lw=1.5)
        ax2.axvline(lowcut, color="r", ls="--", alpha=0.7)
        ax2.axvline(highcut, color="r", ls="--", alpha=0.7)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title("Phase Response")
        ax2.set_xlim([0, fs / 2])
        ax2.grid(True, alpha=0.3)

        # Pole-Zero Plot
        ax3 = axes[1, 0]
        zeros = np.roots(B_digital)
        poles = np.roots(A_digital)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit Circle")
        ax3.plot(
            np.real(zeros),
            np.imag(zeros),
            "bo",
            ms=10,
            mfc="none",
            mew=2,
            label="Zeros",
        )
        ax3.plot(np.real(poles), np.imag(poles), "rx", ms=10, mew=2, label="Poles")
        ax3.set_xlabel("Real")
        ax3.set_ylabel("Imaginary")
        ax3.set_title("Pole-Zero Plot")
        ax3.set_aspect("equal")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Impulse Response
        ax4 = axes[1, 1]
        impulse = np.zeros(80)
        impulse[0] = 1
        h_imp = signal.lfilter(B_digital, A_digital, impulse)
        ax4.stem(
            np.arange(len(h_imp)), h_imp, linefmt="c-", markerfmt="co", basefmt=" "
        )
        ax4.set_xlabel("n (samples)")
        ax4.set_ylabel("h[n]")
        ax4.set_title("Impulse Response")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"bpf_{lowcut}_{highcut}Hz.png", dpi=150, bbox_inches="tight")
        plt.show()

    return y, B_digital, A_digital


# ============================================
# MAIN: Original Assignment + Extra Tests
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("DIGITAL BANDPASS FILTER - IMPULSE INVARIANCE METHOD")
    print("=" * 60)

    # Create test signal with multiple frequency components
    fs = 20050
    t = np.arange(0, 0.2, 1 / fs)

    # Mix: 500Hz (blocked), 1500Hz (pass), 3000Hz (pass), 6000Hz (blocked)
    x = (
        0.5 * np.sin(2 * np.pi * 500 * t)
        + 1.0 * np.sin(2 * np.pi * 1500 * t)
        + 1.0 * np.sin(2 * np.pi * 3000 * t)
        + 0.5 * np.sin(2 * np.pi * 6000 * t)
    )

    print(f"\nTest signal contains: 500Hz, 1500Hz, 3000Hz, 6000Hz")
    print(f"Sampling rate: {fs} Hz")

    # --- ORIGINAL ASSIGNMENT SPECS ---
    print(f"\n{'='*60}")
    print("ORIGINAL ASSIGNMENT: lowcut=1000, highcut=4000")
    print(f"{'='*60}")
    y1, b1, a1 = fft_bandpass_filter(x, fs, lowcut=1000, highcut=4000, order=4)
    print(f"Coefficients: b={np.round(b1, 4)}, a={np.round(a1, 4)}")

    # --- TEST 2: Narrower band ---
    print(f"\n{'='*60}")
    print("TEST 2: lowcut=1500, highcut=2500")
    print(f"{'='*60}")
    y2, b2, a2 = fft_bandpass_filter(x, fs, lowcut=1500, highcut=2500, order=4)
    print(f"Coefficients: b={np.round(b2, 4)}, a={np.round(a2, 4)}")

    # --- TEST 3: Higher band ---
    print(f"\n{'='*60}")
    print("TEST 3: lowcut=3000, highcut=7000")
    print(f"{'='*60}")
    y3, b3, a3 = fft_bandpass_filter(x, fs, lowcut=3000, highcut=7000, order=4)
    print(f"Coefficients: b={np.round(b3, 4)}, a={np.round(a3, 4)}")

    print(f"\n{'='*60}")
    print("DONE! Change lowcut/highcut in the main block to test other values.")
    print("=" * 60)
