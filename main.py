from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.signal.rotate import rotate_zne_lqt
from scipy.signal import butter, filtfilt

north_korea_test_time = "2017-09-03T03:30:05"
aomori_earthquake_time = "2025-12-08T14:17:00"


def fetch_seismic_data(
    network: str,
    station: str,
    channel: str,
    starttime: str,
    time_range: int,
    location: str = "*",
):
    client = Client("EARTHSCOPE")  # Using EARTHSCOPE FDSN client
    starttime = UTCDateTime(starttime)
    try:
        stream = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=starttime + time_range,
        )
        return stream
    except Exception as e:
        print(f"Error fetching data.\nSeismic servers might be down: {e}")
        exit(1)


def plot_event(
    title: str, starttim: str, time_range, network: str, station: str, channel: str
):
    # Cut-off frequencies to allow P-waves and S-waves to pass through
    cut_low = 0.5  # Hz
    cut_high = 6  # Hz

    stream = fetch_seismic_data(network, station, channel, starttim, time_range)

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

    stream1 = fetch_seismic_data(network1, station1, channel1, starttime1, time_range)
    stream2 = fetch_seismic_data(network2, station2, channel2, starttime2, time_range)

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


def get_separated_seismic_waves(
    client_name,
    network,
    station,
    location,
    event_lat,
    event_lon,
    event_depth,
    event_time,
):
    """
    Separates P and S waves into L, Q, T components for a specific seismic event.
    Returns: l, q, t (numpy arrays) and the time vector.
    """
    client = Client(client_name)

    # 1. Coordinate and Distance Math
    try:
        inv = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel="BH*",
            level="response",
        )
    except Exception as e:
        print(f"Error fetching station metadata: {e}")
        exit(1)

    st_coords = inv.get_coordinates(f"{network}.{station}.{location}.BHZ")

    dist_m, az, baz = gps2dist_azimuth(
        event_lat, event_lon, st_coords["latitude"], st_coords["longitude"]
    )
    dist_deg = locations2degrees(
        event_lat, event_lon, st_coords["latitude"], st_coords["longitude"]
    )

    # 2. Calculate Theoretical Incidence Angle
    model = TauPyModel(model="iasp91")
    arrivals = model.get_travel_times(
        source_depth_in_km=event_depth, distance_in_degree=dist_deg, phase_list=["P"]
    )
    inc_angle = arrivals[0].incident_angle

    # 3. Fetch and Prep Waveforms
    st = client.get_waveforms(
        network, station, location, "BH*", event_time - 30, event_time + 500
    )
    st.remove_response(inventory=inv, output="VEL")
    st.detrend("linear")
    st.taper(0.05)
    st.filter("bandpass", freqmin=0.05, freqmax=2.0)

    # Ensure standard Z, N, E ordering
    st.sort(keys=["component"])
    # Usually: BHE (0), BHN (1), BHZ (2)
    e = st[0].data
    n = st[1].data
    z = st[2].data

    # 4. Perform LQT Rotation

    # LQT rotation
    l, q, t = rotate_zne_lqt(z, n, e, baz, inc_angle)

    # Bandpass filter parameters
    fs = st[0].stats.sampling_rate
    lowcut = 0.05
    highcut = 6.0
    order = 4
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")
    l = filtfilt(b, a, l)
    q = filtfilt(b, a, q)
    t = filtfilt(b, a, t)

    # Create time axis for plotting/analysis
    times = st[0].times()

    return l, q, t, times


def plot_separated_components(
    suptitle,
    time_vector,
    l_component,
    q_component,
    t_component,
    freq,
    l_fft,
    q_fft,
    t_fft,
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


aomori_earthquake = {
    "client_name": "EARTHSCOPE",
    "network": "IU",
    "station": "MAJO",
    "location": "00",
    "event_lat": 40.998,
    "event_lon": 142.998,
    "event_depth": 40.7,
    "event_time": UTCDateTime("2025-12-08T14:15:09"),
}

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **aomori_earthquake
)

# Calculate frequency axis for FFT plots
delta = time_vector[1] - time_vector[0]
n = len(time_vector)
freq = np.fft.fftfreq(n, d=delta)[: n // 2]

l_fft = np.abs(np.fft.fft(l_component))[: n // 2]
q_fft = np.abs(np.fft.fft(q_component))[: n // 2]
t_fft = np.abs(np.fft.fft(t_component))[: n // 2]

# Plotting the separated components and their FFTs
plot_separated_components(
    suptitle="Aomori, Japan 2025 Earthquake",
    time_vector=time_vector,
    l_component=l_component,
    q_component=q_component,
    t_component=t_component,
    freq=freq,
    l_fft=l_fft,
    q_fft=q_fft,
    t_fft=t_fft,
)

# north_korea_test = {
#     "client_name": "EARTHSCOPE",
#     "network": "IU",
#     "station": "INCN",
#     "location": "00",
#     "event_lat": 41.343,
#     "event_lon": 129.036,
#     "event_depth": 0.5,
#     "event_time": UTCDateTime("2017-09-03T03:30:05"),
# }

# l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
#     **north_korea_test
# )

# # Calculate frequency axis for FFT plots
# delta = time_vector[1] - time_vector[0]
# n = len(time_vector)
# freq = np.fft.fftfreq(n, d=delta)[: n // 2]

# l_fft = np.abs(np.fft.fft(l_component))[: n // 2]
# q_fft = np.abs(np.fft.fft(q_component))[: n // 2]
# t_fft = np.abs(np.fft.fft(t_component))[: n // 2]

# plot_separated_components(
#     suptitle="North Korea 2017 Nuclear Test",
#     time_vector=time_vector,
#     l_component=l_component,
#     q_component=q_component,
#     t_component=t_component,
#     freq=freq,
#     l_fft=l_fft,
#     q_fft=q_fft,
#     t_fft=t_fft,
# )


# plot_event("North Korea 2017 Test", north_korea_test_time, 300, "IU", "INCN", "BHZ")
# plot_event("Aomori, Japan 2025", aomori_earthquake_time, 300, "IU", "INCN", "BHZ")

# compare_events(
#     title1="North Korea 2017 Nuclear Test",
#     starttime1=north_korea_test_time,
#     network1="IU",
#     station1="INCN",
#     channel1="BHZ",
#     title2="Aomori, Japan 2025 Earthquake",
#     starttime2=aomori_earthquake_time,
#     network2="JP",
#     station2="JOW",
#     channel2="BHZ",
#     time_range=300,
# )
