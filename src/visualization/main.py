import matplotlib.pyplot as plt
import numpy as np


def plot_separated_components(
    suptitle: str,
    time_vector: np.ndarray,
    l_component: np.ndarray,
    q_component: np.ndarray,
    t_component: np.ndarray,
    freq: np.ndarray,
    l_fft: np.ndarray,
    q_fft: np.ndarray,
    t_fft: np.ndarray,
):
    plt.figure(figsize=(12, 8))
    plt.suptitle(suptitle)
    plt.subplot(3, 2, 1)
    plt.plot(time_vector, l_component, color="blue")
    plt.title("L Component (P-wave) - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(freq, l_fft, color="blue")
    plt.title("L Component (P-wave) - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 6)  # Limit x-axis to focus on relevant frequencies
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.plot(time_vector, q_component, color="green")
    plt.title("Q Component (S-wave) - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.plot(freq, q_fft, color="green")
    plt.title("Q Component (S-wave) - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 6)  # Limit x-axis to focus on relevant frequencies
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.plot(time_vector, t_component, color="red")
    plt.title("T Component (Surface Waves) - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.plot(freq, t_fft, color="red")
    plt.title("T Component (Surface Waves) - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 6)  # Limit x-axis to focus on relevant frequencies
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_event(
    title: str,
    times: np.ndarray,
    data: np.ndarray,
    freq: np.ndarray,
    data_fft: np.ndarray,
    data_filtered: np.ndarray,
    data_filtered_fft: np.ndarray,
):
    # plotting with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Seismic Event: {title}")

    plt.subplot(2, 2, 1)
    plt.plot(times, data, color="blue")
    plt.title("Seismic Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(freq, np.abs(data_fft)[: len(data_fft) // 2], color="red")
    plt.title("FFT of Seismic Waveform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(times, data_filtered, color="green")
    plt.title("Filtered Seismic Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(
        freq,
        np.abs(data_filtered_fft)[: len(data_filtered_fft) // 2],
        color="red",
    )
    plt.title("FFT of Filtered Seismic Waveform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()
