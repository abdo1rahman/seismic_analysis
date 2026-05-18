from src.processing.main import (
    get_separated_seismic_waves,
    calculate_fft_frequency,
    fetch_seismic_data,
)
from src.visualization.main import plot_separated_components, plot_event

north_korea_event = {
    "client_name": "EARTHSCOPE",
    "event_time": "2017-09-03T03:30:01",
    "network": "IC",
    "station": "MDJ",
    "location": "00",
    "event_lat": 41.343,
    "event_lon": 129.036,
    "event_depth": 0.0,
}

north_korea_stream = fetch_seismic_data("IC", "MDJ", "2017-09-03T03:30:01", 300, "00")

plot_event("North Korea Nuclear Test", north_korea_stream)

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **north_korea_event
)

freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
    l_component, q_component, t_component, time_vector
)
plot_separated_components(
    suptitle="North Korea Nuclear Test (2017)",
    time_vector=time_vector,
    l_component=l_component,
    q_component=q_component,
    t_component=t_component,
    freq=freq,
    l_fft=l_fft,
    q_fft=q_fft,
    t_fft=t_fft,
)

aomori_japan_quake = {
    "client_name": "EARTHSCOPE",
    "event_time": "2025-12-08T14:15:00",
    "network": "IU",
    "station": "MAJO",
    "location": "00",
    "event_lat": 40.4,
    "event_lon": 142.5,
    "event_depth": 10.0,
}

aomori_quake_stream = fetch_seismic_data("IU", "MAJO", "2025-12-08T14:15:00", 300, "00")

plot_event("Aomori, Japan 2025 Earthquake", aomori_quake_stream)

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **aomori_japan_quake
)

freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
    l_component, q_component, t_component, time_vector
)
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
