from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client("EARTHSCOPE")
t = UTCDateTime("1998-05-11T10:13:42")

# Test these specific historical combinations
candidates = [
    {"nw": "II", "st": "ABKT"},  # Alibek, Turkmenistan
    {"nw": "II", "st": "NIL"},  # Nilore, Pakistan
    {"nw": "G", "st": "WUS"},  # Wushi, China
    {"nw": "IU", "st": "AAK"},  # Ala-Archa, Kyrgyzstan
]

for c in candidates:
    try:
        inv = client.get_stations(
            network=c["nw"],
            station=c["st"],
            starttime=t,
            endtime=t + 600,
            level="channel",
        )
        print(f"SUCCESS: Found data for {c['nw']}.{c['st']}")
        print(inv)
        break
    except Exception:
        print(f"FAILED: {c['nw']}.{c['st']} still 204")
