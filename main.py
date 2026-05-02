from src.processing.main import (
    get_separated_seismic_waves,
    calculate_fft_frequency,
)
from src.visualization.main import plot_separated_components

# from obspy.clients.fdsn import Client
# from obspy import UTCDateTime

# client = Client("EARTHSCOPE")
# t = UTCDateTime("1998-05-11T10:13:42")

# # This will print the available channels for this station and time
# try:
#     inv = client.get_stations(
#         network="II", station="AAK", starttime=t, endtime=t + 600, level="channel"
#     )
#     print(inv)
# except Exception as e:
#     print(f"Discovery failed: {e}")

# aomori_earthquake = {
#     "client_name": "EARTHSCOPE",
#     "network": "IU",
#     "station": "MAJO",
#     "location": "00",
#     "event_lat": 40.998,
#     "event_lon": 142.998,
#     "event_depth": 40.7,
#     "event_time": "2025-12-08T14:15:09",
# }

# l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
#     **aomori_earthquake
# )

# freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
#     l_component, q_component, t_component, time_vector
# )

# # Plotting the separated components and their FFTs
# plot_separated_components(
#     suptitle="Aomori, Japan 2025 Earthquake",
#     time_vector=time_vector,
#     l_component=l_component,
#     q_component=q_component,
#     t_component=t_component,
#     freq=freq,
#     l_fft=l_fft,
#     q_fft=q_fft,
#     t_fft=t_fft,
# )


# India Pokhran Nuclear Test
pokhran_test = {
    "client_name": "EARTHSCOPE",
    "network": "II",
    "station": "ABKT",
    "location": "00",
    "event_lat": 27.078,
    "event_lon": 71.722,
    "event_depth": 0.2,
    "event_time": "1998-05-11T10:13:42",
    "channel": "BHZ,BHN,BHE",
}

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **pokhran_test
)

freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
    l_component, q_component, t_component, time_vector
)

plot_separated_components(
    suptitle="India Pokhran Nuclear Test (1998)",
    time_vector=time_vector,
    l_component=l_component,
    q_component=q_component,
    t_component=t_component,
    freq=freq,
    l_fft=l_fft,
    q_fft=q_fft,
    t_fft=t_fft,
)
