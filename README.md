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
