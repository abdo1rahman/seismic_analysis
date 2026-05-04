from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Use EarthScope or AusPass as primary alternatives
# EarthScope URL: http://service.earthscope.org
# AusPass: http://auspass.edu.au
try:
    client = Client("http://service.earthscope.org")
except:
    client = Client("AUSPASS")

event_time = UTCDateTime("2026-04-01T17:15:00")

# Define a bulk request: (Network, Station, Location, Channel, Start, End)
# Using 'GE' (GEOFON) or 'IU' (Global) is safest for Indonesia
bulk = [("IU", "TNTI", "00", "BHZ", event_time - 60, event_time + 600)]

print(f"Attempting bulk download from {client.base_url}...")

try:
    st = client.get_waveforms_bulk(bulk, timeout=60)
    print("Success! Data retrieved.")
    st.plot()
except Exception as e:
    print(f"Failed: {e}")
