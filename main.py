from src.processing.main import (
    get_separated_seismic_waves,
    calculate_fft_frequency,
)
from src.visualization.main import plot_separated_components

banda_sea_quake = {
    "client_name": "GFZ",
    "event_time": "2024-09-11T16:10:00",
    "network": "IU",
    "station": "SAUI",
    "location": "00",
    "event_lat": -6.5,
    "event_lon": 130.5,
    "event_depth": 155.0,
}

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **banda_sea_quake
)

freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
    l_component, q_component, t_component, time_vector
)
plot_separated_components(
    suptitle="Banda Sea Earthquake (2024)",
    time_vector=time_vector,
    l_component=l_component,
    q_component=q_component,
    t_component=t_component,
    freq=freq,
    l_fft=l_fft,
    q_fft=q_fft,
    t_fft=t_fft,
)
