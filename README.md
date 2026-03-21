### Python receiver (`server.py`)

A consolidated receiver script replaces the original basic example. It supports two modes selected at the command line:

**Standard mode** — receives all sensor types, prints stats to the console, and optionally logs each sensor type to its own CSV file per device.

**Mobility mode** (`--mobility`) — receives GPS and accelerometer only. Writes a single combined CSV per device where each row is one accelerometer packet paired with the most recently received GPS fix. GPS columns are left blank if no fix is available, so rows are written at the configured send interval regardless of GPS availability. GPS fixes with all sub-accuracies equal to zero (a signature of Android's stale cached position) or with horizontal accuracy worse than 100 m are discarded.

```
# Standard mode
python3 server.py                           # UDP, console only
python3 server.py --tcp                     # TCP, console only
python3 server.py --tcp --csv run1          # TCP + per-sensor CSV files
python3 server.py --csv run1                # UDP + per-sensor CSV files

# Mobility mode
python3 server.py --mobility                        # UDP, console only
python3 server.py --mobility --tcp                  # TCP, console only
python3 server.py --mobility --tcp --csv my_run     # TCP + combined CSV per device

# Other options
python3 server.py --port 8080               # override default port (default: 47892)
python3 server.py --debug                   # enable DEBUG logging (shows discarded GPS fixes etc.)
```

When CSV logging is active, sensor data is suppressed from the console so operational messages (connections, disconnections, errors) remain readable. All operational messages are always written to `server.log` with timestamps regardless of mode.

CSV files are named using the device key (UUID or IP) to avoid collisions:

| Mode | File pattern |
|---|---|
| Standard | `<prefix>_<device-key>_<sensor>.csv` |
| Mobility | `<prefix>_<device-key>.csv` |

### Analytics dashboard (`analytics.html`)

A standalone HTML file for visualising mobility-mode CSV output in the browser — no server or installation required. Load one or more CSV files (drag-and-drop or file picker) and the dashboard renders:

- GPS track on an interactive map (Leaflet)
- Speed, bearing, and altitude charts with Savitzky–Golay smoothing
- IMU dynamic load (accelerometer std-dev magnitude) chart
- GPS accuracy chart (horizontal and vertical)
- Multi-device compare mode — overlays all loaded device tracks on the map simultaneously
- Live mode — polls a file at a configurable interval for real-time monitoring
- Configurable GPS accuracy filter — fixes worse than the chosen threshold are excluded from all charts and the map
- Dark/light theme toggle
- Download of the processed analytics CSV
