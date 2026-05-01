from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

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
    print(f"Error fetching data: {e}")
    exit(1)

# Convert to NumPy
data = stream[0].data
times = stream[0].times()

data_fft = np.fft.fft(data)
# frequencies = np.fft.fftfreq(len(data), d=stream[0].stats.delta)

# plotting with Matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(times, data, color="blue")
plt.title("Seismic Waveform from North Korea 2017 Test")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(
    times[: len(data_fft) // 2], np.abs(data_fft)[: len(data_fft) // 2], color="red"
)
plt.title("FFT of Seismic Waveform")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.show()
