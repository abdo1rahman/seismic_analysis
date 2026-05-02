# Seismic Signal Analysis Project

## Overview

This project focuses on analyzing seismic signals to distinguish between earthquakes and nuclear tests using advanced digital signal processing techniques. By leveraging the power of ObsPy for seismic data retrieval, NumPy for numerical computations, and Matplotlib for visualization, the project provides tools to process, analyze, and visualize seismic waveforms.

The core functionality includes fetching seismic data from global networks, separating seismic waves into longitudinal (L), transverse (Q), and tangential (T) components, performing Fast Fourier Transform (FFT) for frequency domain analysis, and generating comparative plots to highlight differences between natural earthquakes and man-made nuclear explosions.

## Features

- **Data Retrieval**: Fetch seismic waveforms from FDSN (Federation of Digital Seismograph Networks) using ObsPy clients.
- **Wave Separation**: Decompose seismic signals into L, Q, and T components using rotation techniques.
- **Frequency Analysis**: Compute FFT to analyze frequency content of seismic waves.
- **Visualization**: Generate detailed plots comparing time-domain and frequency-domain representations of seismic events.
- **Event Comparison**: Built-in examples comparing the Aomori Earthquake (2025) and the Pokhran Nuclear Test (1998).
- **Modular Design**: Organized into processing and visualization modules for easy extension and maintenance.

## Installation

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/abdo1rahman/seismic_analysis.git
   cd dsp-project
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or, if using the pyproject.toml:

   ```bash
   pip install -e .
   ```

## Usage

### Running the Main Script

The main entry point is `main.py`, which demonstrates the analysis for two seismic events:

```bash
python main.py
```

This will:

- Fetch and process data for the Aomori Earthquake (2025)
- Fetch and process data for the Pokhran Nuclear Test (1998)
- Generate plots showing separated components and their FFTs

### Using Individual Modules

#### Processing Module

The `processing` module contains functions for data retrieval and signal processing:

```python
from processing import get_separated_seismic_waves, calculate_fft_frequency

# Example usage
event_params = {
    "client_name": "EARTHSCOPE",
    "network": "IU",
    "station": "MAJO",
    "location": "00",
    "event_lat": 40.998,
    "event_lon": 142.998,
    "event_depth": 40.7,
    "event_time": "2025-12-08T14:15:09",
}

l_component, q_component, t_component, time_vector = get_separated_seismic_waves(**event_params)
freq, l_fft, q_fft, t_fft = calculate_fft_frequency(l_component, q_component, t_component, time_vector)
```

#### Visualization Module

The `visualization` module provides plotting functions:

```python
from visualization import plot_separated_components

plot_separated_components(
    suptitle="Seismic Event Analysis",
    time_vector=time_vector,
    l_component=l_component,
    q_component=q_component,
    t_component=t_component,
    freq=freq,
    l_fft=l_fft,
    q_fft=q_fft,
    t_fft=t_fft,
)
```

## Project Structure

```
dsp-project/
├── main.py                 # Main script demonstrating the analysis
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt        # List of Python dependencies
├── processing/
│   ├── __init__.py
│   └── main.py             # Core processing functions (data fetching, wave separation, FFT)
├── visualization/
│   ├── __init__.py
│   └── main.py             # Plotting and visualization functions
└── README.md               # This file
```

## Dependencies

- **ObsPy**: For seismic data handling and processing
- **NumPy**: For numerical computations and array operations
- **Matplotlib**: For creating plots and visualizations
- **SciPy**: For signal processing (FFT, filtering)
- Other supporting libraries: requests, pillow, etc.

See `requirements.txt` for the complete list with versions.

## Methodology

### Seismic Wave Separation

The project uses the LQT (Longitudinal, Transverse, Tangential) coordinate system to separate seismic waves:

- **L-component**: Compressional waves (P-waves)
- **Q-component**: Shear waves (S-waves) in the radial direction
- **T-component**: Shear waves in the tangential direction (surface waves)

This separation helps in distinguishing between different types of seismic sources.

### Frequency Domain Analysis

FFT is applied to each component to analyze the frequency content, which can reveal characteristic signatures of earthquakes vs. nuclear tests.

## Examples

The project includes two example analyses:

1. **Aomori Earthquake (2025)**: A natural seismic event
2. **Pokhran Nuclear Test (1998)**: A man-made nuclear explosion

Running `main.py` will generate plots for both events, allowing visual comparison of their seismic signatures.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Data provided by the Incorporated Research Institutions for Seismology (IRIS) through the EARTHSCOPE program
- Built using ObsPy, an open-source Python framework for seismology

## Contact

For questions or issues, please open an issue on the GitHub repository.
