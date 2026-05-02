from src.processing.main import (
    get_separated_seismic_waves,
    calculate_fft_frequency,
)
from src.visualization.main import plot_separated_components

aomori_earthquake = {
    "client_name": "EARTHSCOPE",
    "network": "IU",
    "station": "MAJO",
    "location": "00",
    "event_lat": 40.998,
    "event_lon": 142.998,
    "event_depth": 40.7,
    "event_time": "2025-12-08T14:15:09",
}

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(
    **aomori_earthquake
)

freq, l_fft, q_fft, t_fft = calculate_fft_frequency(
    l_component, q_component, t_component, time_vector
)

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


# India Pokhran Nuclear Test
pokhran_test = {
    "client_name": "EARTHSCOPE",
    "network": "IU",
    "station": "MAJO",
    "location": "00",
    "event_lat": 26.142,
    "event_lon": 71.478,
    "event_depth": 1.0,
    "event_time": "1998-05-25T09:30:00",
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
