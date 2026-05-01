from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

north_korea_test_time = "2017-09-03T03:30:05"
aomori_earthquake_time = "2025-12-08T14:17:00"


def plot_event(
    title: str, starttim: str, time_range, network: str, station: str, channel: str
):
    # Cut-off frequencies to allow P-waves and S-waves to pass through
    cut_low = 0.5  # Hz
    cut_high = 6  # Hz

    client = Client("EARTHSCOPE")  # Using EARTHSCOPE FDSN client
    starttime = UTCDateTime(starttim)
    try:
        stream = client.get_waveforms(
            network=network,
            station=station,
            location="*",
            channel=channel,
            starttime=starttime,
            endtime=starttime + time_range,
        )
    except Exception as e:
        print(f"Error fetching data.\nSeismic servers might be down: {e}")
        exit(1)

    stream_raw = stream.copy()
    stream.filter("bandpass", freqmin=cut_low, freqmax=cut_high)

    # Convert to NumPy
    data = stream_raw[0].data
    times = stream_raw[0].times()
    data_fft = np.fft.fft(data)

    data_filtered = stream[0].data
    data_filtered_fft = np.fft.fft(data_filtered)

    freq = np.fft.fftfreq(len(data), d=stream_raw[0].stats.delta)[: len(data) // 2]

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


def compare_events(
    title1: str,
    starttime1: str,
    title2: str,
    network1: str,
    station1: str,
    channel1: str,
    starttime2: str,
    network2: str,
    station2: str,
    channel2: str,
    time_range: int,
):
    # Cut-off frequencies to allow P-waves and S-waves to pass through
    cut_low = 0.5  # Hz
    cut_high = 6  # Hz

    client = Client("EARTHSCOPE")  # Using EARTHSCOPE FDSN client
    starttime1 = UTCDateTime(starttime1)
    starttime2 = UTCDateTime(starttime2)

    try:
        stream1 = client.get_waveforms(
            network=network1,
            station=station1,
            location="*",
            channel=channel1,
            starttime=starttime1,
            endtime=starttime1 + time_range,
        )
        stream2 = client.get_waveforms(
            network=network2,
            station=station2,
            location="*",
            channel=channel2,
            starttime=starttime2,
            endtime=starttime2 + time_range,
        )
    except Exception as e:
        print(f"Error fetching data.\nSeismic servers might be down: {e}")
        exit(1)

    stream1.filter("bandpass", freqmin=cut_low, freqmax=cut_high)
    stream2.filter("bandpass", freqmin=cut_low, freqmax=cut_high)

    data1 = stream1[0].data
    times1 = stream1[0].times()
    freq1 = np.fft.fftfreq(len(data1), d=stream1[0].stats.delta)[: len(data1) // 2]

    data2 = stream2[0].data
    times2 = stream2[0].times()
    freq2 = np.fft.fftfreq(len(data2), d=stream2[0].stats.delta)[: len(data2) // 2]

    # plotting with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Comparison of Seismic Events")

    plt.subplot(2, 2, 1)
    plt.plot(times1, data1, color="blue")
    plt.title(title1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(freq1, np.abs(np.fft.fft(data1))[: len(data1) // 2], color="red")
    plt.title(f"FFT of{title1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(times2, data2, color="green")
    plt.title(title2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(freq2, np.abs(np.fft.fft(data2))[: len(data2) // 2], color="red")
    plt.title(f"FFT of{title2}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()


# plot_event("North Korea 2017 Test", north_korea_test_time, 300, "IU", "INCN", "BHZ")
# plot_event("Aomori, Japan 2025", aomori_earthquake_time, 300, "IU", "INCN", "BHZ")

compare_events(
    title1="North Korea 2017 Nuclear Test",
    starttime1=north_korea_test_time,
    network1="IU",
    station1="INCN",
    channel1="BHZ",
    title2="Aomori, Japan 2025 Earthquake",
    starttime2=aomori_earthquake_time,
    network2="JP",
    station2="JOW",
    channel2="BHZ",
    time_range=300,
)
