from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

# Cut-off frequencies to allow P-waves and S-waves to pass through
cut_low = 6.0  # Hz
cut_high = 8.0  # Hz

client = Client("EARTHSCOPE")  # Using EARTHSCOPE FDSN client
# North Korea 2017 Test Time
starttime = UTCDateTime("2017-09-03T03:30:01")
# Station in South Korea (Incheon)
try:
    stream = client.get_waveforms(
        network="KG,KS,KN,IC",  # South Korean networks
        station="*",  # All stations
        location="*",
        channel="BHZ",
        starttime=starttime,
        endtime=starttime + 300,
    )
except Exception as e:
    print(f"Error fetching data.\nSeismic servers might be down: {e}")
    exit(1)


stream_raw = stream.copy()
stream_raw.filter("highpass", freq=0.01)  # Removing the frequency bias
stream.filter("bandpass", freqmin=cut_low, freqmax=cut_high)

# Convert to NumPy
data = stream_raw[0].data
times = stream_raw[0].times()
data_fft = np.fft.fft(data)
freq_raw = np.fft.fftfreq(len(data), d=stream_raw[0].stats.delta)[: len(data) // 2]

data_filtered = stream[0].data
data_filtered_fft = np.fft.fft(data_filtered)
times_filtered = stream[0].times()
freq_filtered = np.fft.fftfreq(len(data_filtered), d=stream[0].stats.delta)[: len(data_filtered) // 2]


# plotting with Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(times, data, color="blue")
plt.title("Seismic Waveform from North Korea 2017 Test")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(
    freq_raw, np.abs(data_fft)[: len(data_fft) // 2], color="red"
)
plt.title("FFT of Seismic Waveform")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(times_filtered, data_filtered, color="green")
plt.title("Filtered Seismic Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(
    freq_filtered,
    np.abs(data_filtered_fft)[: len(data_filtered_fft) // 2],
    color="red",
)
plt.title("FFT of Filtered Seismic Waveform")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
