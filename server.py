"""
server.py  —  SensaGram receiver

Standard mode  : receives all sensor types, logs each to its own CSV file
                 (one file per sensor type) and/or prints to console.

Mobility mode  : --mobility
                 Only processes GPS + accelerometer. Writes a single combined
                 CSV where each row is one accelerometer event paired with the
                 most recently received GPS fix (blank if none yet).

Logging behaviour:
  - Operational messages (startup, connections, disconnections, errors) are
    ALWAYS written to server.log AND printed to the console.
  - Sensor data is printed to the console ONLY when no CSV is active.
    When CSV logging is on, sensor data goes to the CSV file only — the
    console stays clean so you can watch operational messages without noise.

Usage examples:
    python3 server.py                                 # UDP, console only
    python3 server.py --tcp                           # TCP, console only
    python3 server.py --tcp --csv                     # TCP + per-sensor CSVs, quiet console
    python3 server.py --csv run1                      # UDP + run1_*.csv, quiet console
    python3 server.py --port 9000 --tcp --csv run1

    python3 server.py --mobility                      # UDP, console only
    python3 server.py --mobility --tcp                # TCP, console only
    python3 server.py --mobility --tcp --csv          # TCP + sensagram.csv, quiet console
    python3 server.py --mobility --tcp --csv my_run   # TCP + my_run.csv, quiet console
"""

from udpserver import UDPServer
import json, csv, argparse, socket, threading, logging, sys
from datetime import datetime


# =========================================================================== #
# Logging setup
# =========================================================================== #
# Configured at the bottom of the file once args are parsed, so we know
# whether CSV is active.  Functions below reference the module-level `logger`
# and `csv_active` flag, both set in the entry-point section.

logger     = logging.getLogger("sensagram")
csv_active = False   # set True when any CSV output is configured


def log(msg):
    """Operational message — always written to server.log AND the console."""
    logger.info(msg)


def data_print(msg):
    """Sensor data line — printed to console only when no CSV is active."""
    if not csv_active:
        print(msg)


# =========================================================================== #
# Shared formatting helpers
# =========================================================================== #

def _clean_prefix(prefix):
    """Strip a trailing .csv extension from a user-supplied prefix/filename."""
    return prefix[:-4] if prefix.endswith(".csv") else prefix


def fmt3(values):
    if values and len(values) >= 3:
        return f"x={values[0]:.5f}  y={values[1]:.5f}  z={values[2]:.5f}"
    return str(values)


def fmt1(values):
    if values and len(values) >= 1:
        return f"{values[0]:.5f}"
    return str(values)


def print_stats(label, jsonData, formatter=fmt3):
    """Print aggregated sensor stats — suppressed when CSV is active."""
    values = jsonData.get("values")
    mins   = jsonData.get("min")
    maxs   = jsonData.get("max")
    avg    = jsonData.get("avg")
    stddev = jsonData.get("stdDev")
    data_print(f"  {label}")
    if values is not None: data_print(f"    current : {formatter(values)}")
    if mins   is not None: data_print(f"    min     : {formatter(mins)}")
    if maxs   is not None: data_print(f"    max     : {formatter(maxs)}")
    if avg    is not None: data_print(f"    avg     : {formatter(avg)}")
    if stddev is not None: data_print(f"    std-dev : {formatter(stddev)}")


# =========================================================================== #
# STANDARD MODE — per-sensor CSV logging
# =========================================================================== #

class StandardMode:
    """
    Receives all sensor types, optionally logs each to its own CSV file,
    and prints to the console when no CSV is active.
    """

    def __init__(self, csv_prefix=None):
        self._prefix  = _clean_prefix(csv_prefix) if csv_prefix else None
        self._writers = {}
        self._files   = {}

    # ------------------------------------------------------------------ #
    # CSV helpers
    # ------------------------------------------------------------------ #

    def _get_writer(self, sensor_key, fieldnames):
        if sensor_key not in self._writers:
            filename = f"{self._prefix}_{sensor_key}.csv"
            f = open(filename, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            self._files[sensor_key]   = f
            self._writers[sensor_key] = writer
            log(f"[CSV] {sensor_key} → {filename}")
        return self._writers[sensor_key]

    def _log_xyz(self, sensor_key, received_at, timestamp, jsonData):
        fields = ["received_at", "timestamp",
                  "x", "y", "z",
                  "min_x", "min_y", "min_z",
                  "max_x", "max_y", "max_z",
                  "avg_x", "avg_y", "avg_z",
                  "stddev_x", "stddev_y", "stddev_z"]
        writer = self._get_writer(sensor_key, fields)

        def u(key):
            v = jsonData.get(key)
            return v if v is not None else [None, None, None]

        v, mn, mx, av, sd = (u(k) for k in ("values", "min", "max", "avg", "stdDev"))
        writer.writerow({
            "received_at": received_at, "timestamp": timestamp,
            "x": v[0],  "y": v[1],  "z": v[2],
            "min_x": mn[0], "min_y": mn[1], "min_z": mn[2],
            "max_x": mx[0], "max_y": mx[1], "max_z": mx[2],
            "avg_x": av[0], "avg_y": av[1], "avg_z": av[2],
            "stddev_x": sd[0], "stddev_y": sd[1], "stddev_z": sd[2],
        })
        self._files[sensor_key].flush()

    def _log_scalar(self, sensor_key, received_at, timestamp, jsonData):
        fields = ["received_at", "timestamp", "value", "min", "max", "avg", "stddev"]
        writer = self._get_writer(sensor_key, fields)

        def first(key):
            v = jsonData.get(key)
            return v[0] if v else None

        writer.writerow({
            "received_at": received_at, "timestamp": timestamp,
            "value":  first("values"), "min": first("min"),
            "max":    first("max"),    "avg": first("avg"),
            "stddev": first("stdDev"),
        })
        self._files[sensor_key].flush()

    def _log_gps(self, received_at, jsonData):
        fields = ["received_at", "gps_time", "latitude", "longitude", "altitude",
                  "speed", "bearing", "accuracy",
                  "speed_accuracy", "bearing_accuracy", "vertical_accuracy"]
        writer = self._get_writer("gps", fields)
        writer.writerow({
            "received_at":       received_at,
            "gps_time":          jsonData.get("time"),
            "latitude":          jsonData.get("latitude"),
            "longitude":         jsonData.get("longitude"),
            "altitude":          jsonData.get("altitude"),
            "speed":             jsonData.get("speed"),
            "bearing":           jsonData.get("bearing"),
            "accuracy":          jsonData.get("accuracy"),
            "speed_accuracy":    jsonData.get("speedAccuracyMetersPerSecond"),
            "bearing_accuracy":  jsonData.get("bearingAccuracyDegrees"),
            "vertical_accuracy": jsonData.get("verticalAccuracyMeters"),
        })
        self._files["gps"].flush()

    # ------------------------------------------------------------------ #
    # Packet handler
    # ------------------------------------------------------------------ #

    def handle(self, data):
        jsonData    = json.loads(data)
        sensorType  = jsonData["type"]
        timestamp   = jsonData.get("timestamp")
        received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        do_csv      = self._prefix is not None

        if sensorType == "android.sensor.accelerometer":
            data_print(f"[ACCELEROMETER]  ts={timestamp}")
            print_stats("acceleration (m/s²)", jsonData)
            if do_csv: self._log_xyz("accelerometer", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gyroscope":
            data_print(f"[GYROSCOPE]  ts={timestamp}")
            print_stats("angular velocity (rad/s)", jsonData)
            if do_csv: self._log_xyz("gyroscope", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gravity":
            data_print(f"[GRAVITY]  ts={timestamp}")
            print_stats("gravity (m/s²)", jsonData)
            if do_csv: self._log_xyz("gravity", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.linear_acceleration":
            data_print(f"[LINEAR ACCELERATION]  ts={timestamp}")
            print_stats("linear accel (m/s²)", jsonData)
            if do_csv: self._log_xyz("linear_acceleration", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.rotation_vector":
            data_print(f"[ROTATION VECTOR]  ts={timestamp}")
            print_stats("rotation vector", jsonData)
            if do_csv: self._log_xyz("rotation_vector", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.game_rotation_vector":
            data_print(f"[GAME ROTATION VECTOR]  ts={timestamp}")
            print_stats("game rotation vector", jsonData)
            if do_csv: self._log_xyz("game_rotation_vector", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.magnetic_field":
            data_print(f"[MAGNETIC FIELD]  ts={timestamp}")
            print_stats("magnetic field (µT)", jsonData)
            if do_csv: self._log_xyz("magnetic_field", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.magnetic_field_uncalibrated":
            data_print(f"[MAGNETIC FIELD UNCALIBRATED]  ts={timestamp}")
            print_stats("mag uncal (µT)", jsonData)
            if do_csv: self._log_xyz("magnetic_field_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gyroscope_uncalibrated":
            data_print(f"[GYROSCOPE UNCALIBRATED]  ts={timestamp}")
            print_stats("gyro uncal (rad/s)", jsonData)
            if do_csv: self._log_xyz("gyroscope_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.accelerometer_uncalibrated":
            data_print(f"[ACCELEROMETER UNCALIBRATED]  ts={timestamp}")
            print_stats("accel uncal (m/s²)", jsonData)
            if do_csv: self._log_xyz("accelerometer_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.orientation":
            data_print(f"[ORIENTATION]  ts={timestamp}")
            print_stats("orientation (°)", jsonData)
            if do_csv: self._log_xyz("orientation", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.proximity":
            data_print(f"[PROXIMITY]  ts={timestamp}")
            print_stats("proximity (cm)", jsonData, formatter=fmt1)
            if do_csv: self._log_scalar("proximity", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.step_counter":
            values = jsonData.get("values", [0])
            data_print(f"[STEP COUNTER]  ts={timestamp}  steps={values[0]:.0f}")
            if do_csv: self._log_scalar("step_counter", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.step_detector":
            data_print(f"[STEP DETECTED]  ts={timestamp}")

        elif sensorType == "android.sensor.light":
            data_print(f"[LIGHT]  ts={timestamp}")
            print_stats("illuminance (lux)", jsonData, formatter=fmt1)
            if do_csv: self._log_scalar("light", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.pressure":
            data_print(f"[PRESSURE]  ts={timestamp}")
            print_stats("pressure (hPa)", jsonData, formatter=fmt1)
            if do_csv: self._log_scalar("pressure", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.ambient_temperature":
            data_print(f"[TEMPERATURE]  ts={timestamp}")
            print_stats("temperature (°C)", jsonData, formatter=fmt1)
            if do_csv: self._log_scalar("ambient_temperature", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.relative_humidity":
            data_print(f"[HUMIDITY]  ts={timestamp}")
            print_stats("relative humidity (%)", jsonData, formatter=fmt1)
            if do_csv: self._log_scalar("relative_humidity", received_at, timestamp, jsonData)

        elif sensorType == "android.gps":
            lat      = jsonData["latitude"]
            lon      = jsonData["longitude"]
            alt      = jsonData["altitude"]
            speed    = jsonData["speed"]
            bearing  = jsonData["bearing"]
            accuracy = jsonData["accuracy"]
            gps_time = jsonData["time"]
            data_print(f"[GPS]")
            data_print(f"  lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f} m")
            data_print(f"  speed={speed:.2f} m/s  bearing={bearing:.1f}°  accuracy={accuracy:.1f} m")
            data_print(f"  time={gps_time}")
            if "speedAccuracyMetersPerSecond" in jsonData:
                data_print(
                    f"  speed_accuracy={jsonData['speedAccuracyMetersPerSecond']:.3f} m/s"
                    f"  bearing_accuracy={jsonData.get('bearingAccuracyDegrees', '?'):.2f}°"
                    f"  vertical_accuracy={jsonData.get('verticalAccuracyMeters', '?'):.2f} m"
                )
            if do_csv: self._log_gps(received_at, jsonData)

        else:
            values = jsonData.get("values")
            mins   = jsonData.get("min")
            maxs   = jsonData.get("max")
            avg    = jsonData.get("avg")
            stddev = jsonData.get("stdDev")
            data_print(f"[{sensorType}]  ts={timestamp}")
            data_print(f"  values={values}")
            if mins   is not None: data_print(f"  min   ={mins}")
            if maxs   is not None: data_print(f"  max   ={maxs}")
            if avg    is not None: data_print(f"  avg   ={avg}")
            if stddev is not None: data_print(f"  stdev ={stddev}")
            if do_csv: self._log_xyz(sensorType.split(".")[-1], received_at, timestamp, jsonData)

        data_print("")   # blank line separator between packets


# =========================================================================== #
# MOBILITY MODE — GPS + accelerometer → single combined CSV
# =========================================================================== #

MOBILITY_CSV_FIELDS = [
    "received_at", "gps_time",
    "latitude", "longitude", "altitude",
    "speed", "bearing", "accuracy",
    "speed_accuracy", "bearing_accuracy", "vertical_accuracy",
    "accel_timestamp",
    "accel_x",     "accel_y",     "accel_z",
    "accel_min_x", "accel_min_y", "accel_min_z",
    "accel_max_x", "accel_max_y", "accel_max_z",
    "accel_avg_x", "accel_avg_y", "accel_avg_z",
    "accel_std_x", "accel_std_y", "accel_std_z",
]


class MobilityMode:
    """
    Receives GPS and accelerometer only.

    CSV row trigger: every ACCELEROMETER packet.
    Each row pairs the accelerometer stats with the most recently received GPS
    fix (GPS columns left blank if no fix has arrived yet).  Rows are written
    at the app's send interval regardless of GPS availability.

    GPS packets update an internal cache only; they do not write a CSV row.
    All other sensor types are silently ignored.
    """

    def __init__(self, csv_filename=None):
        self._latest_gps = None
        self._csv_writer = None
        self._csv_file   = None

        if csv_filename:
            path = csv_filename if csv_filename.endswith(".csv") else csv_filename + ".csv"
            self._csv_file   = open(path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=MOBILITY_CSV_FIELDS)
            self._csv_writer.writeheader()
            self._csv_file.flush()
            log(f"[CSV] Mobility log → {path}")

    def _val(self, d, key, idx=None):
        if d is None:
            return ""
        v = d.get(key)
        if v is None:
            return ""
        if idx is not None:
            return v[idx] if len(v) > idx else ""
        return v

    def _write_row(self, accel, gps, received_at):
        v = self._val
        self._csv_writer.writerow({
            "received_at":       received_at,
            "gps_time":          v(gps,   "time"),
            "latitude":          v(gps,   "latitude"),
            "longitude":         v(gps,   "longitude"),
            "altitude":          v(gps,   "altitude"),
            "speed":             v(gps,   "speed"),
            "bearing":           v(gps,   "bearing"),
            "accuracy":          v(gps,   "accuracy"),
            "speed_accuracy":    v(gps,   "speedAccuracyMetersPerSecond"),
            "bearing_accuracy":  v(gps,   "bearingAccuracyDegrees"),
            "vertical_accuracy": v(gps,   "verticalAccuracyMeters"),
            "accel_timestamp":   v(accel, "timestamp"),
            "accel_x":           v(accel, "values",  0),
            "accel_y":           v(accel, "values",  1),
            "accel_z":           v(accel, "values",  2),
            "accel_min_x":       v(accel, "min",     0),
            "accel_min_y":       v(accel, "min",     1),
            "accel_min_z":       v(accel, "min",     2),
            "accel_max_x":       v(accel, "max",     0),
            "accel_max_y":       v(accel, "max",     1),
            "accel_max_z":       v(accel, "max",     2),
            "accel_avg_x":       v(accel, "avg",     0),
            "accel_avg_y":       v(accel, "avg",     1),
            "accel_avg_z":       v(accel, "avg",     2),
            "accel_std_x":       v(accel, "stdDev",  0),
            "accel_std_y":       v(accel, "stdDev",  1),
            "accel_std_z":       v(accel, "stdDev",  2),
        })
        self._csv_file.flush()

    def handle(self, data):
        jsonData   = json.loads(data)
        sensorType = jsonData["type"]

        if sensorType == "android.sensor.accelerometer":
            received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ts     = jsonData.get("timestamp")
            values = jsonData.get("values", [None, None, None])
            data_print(f"[ACCELEROMETER]  ts={ts}")
            data_print(f"  current : x={values[0]}  y={values[1]}  z={values[2]}")
            for stat in ("min", "max", "avg", "stdDev"):
                sv = jsonData.get(stat)
                if sv is not None:
                    data_print(f"  {stat.ljust(6)}: x={sv[0]}  y={sv[1]}  z={sv[2]}")
            if self._latest_gps is None:
                data_print("  gps    : (no fix yet — GPS columns will be blank in CSV)")
            else:
                g = self._latest_gps
                data_print(
                    f"  gps    : lat={g.get('latitude'):.6f}  "
                    f"lon={g.get('longitude'):.6f}  "
                    f"alt={g.get('altitude'):.1f} m"
                )
            data_print("")
            if self._csv_writer:
                self._write_row(jsonData, self._latest_gps, received_at)

        elif sensorType == "android.gps":
            self._latest_gps = jsonData
            lat      = jsonData.get("latitude")
            lon      = jsonData.get("longitude")
            alt      = jsonData.get("altitude")
            speed    = jsonData.get("speed")
            bearing  = jsonData.get("bearing")
            accuracy = jsonData.get("accuracy")
            data_print(f"[GPS]  updated={datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            data_print(f"  lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f} m")
            data_print(f"  speed={speed:.2f} m/s  bearing={bearing:.1f}°  accuracy={accuracy:.1f} m")
            data_print("")

        # all other types silently ignored


# =========================================================================== #
# TCP / UDP transport
# =========================================================================== #

def handle_tcp_client(conn, addr, handler):
    log(f"[TCP] Connected: {addr}")
    buf = ""
    try:
        with conn:
            while True:
                chunk = conn.recv(4096).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            handler(line)
                        except Exception as e:
                            log(f"[TCP] Parse error: {e}")
    except Exception as e:
        log(f"[TCP] Connection error: {e}")
    log(f"[TCP] Disconnected: {addr}")


def run_tcp_server(port, handler):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    log(f"[TCP] Listening on 0.0.0.0:{port}  (waiting for connection…)")
    while True:
        conn, addr = srv.accept()
        threading.Thread(
            target=handle_tcp_client, args=(conn, addr, handler), daemon=True
        ).start()


def run_udp_server(port, handler):
    log(f"[UDP] Listening on 0.0.0.0:{port}")
    server = UDPServer(address=("0.0.0.0", port))
    server.setDataCallBack(handler)
    server.start()


# =========================================================================== #
# Entry point
# =========================================================================== #

parser = argparse.ArgumentParser(
    description="SensaGram receiver — standard and mobility modes",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  Standard mode (all sensors, one CSV per sensor type):
    python3 server.py
    python3 server.py --tcp
    python3 server.py --tcp --csv
    python3 server.py --csv run1

  Mobility mode (GPS + accelerometer only, single combined CSV):
    python3 server.py --mobility
    python3 server.py --mobility --tcp
    python3 server.py --mobility --tcp --csv
    python3 server.py --mobility --tcp --csv my_run
"""
)
parser.add_argument("--port", type=int, default=47892,
                    help="Port to listen on (default: 47892)")
parser.add_argument("--tcp", action="store_true",
                    help="Use TCP instead of UDP")
parser.add_argument("--mobility", action="store_true",
                    help="Mobility mode: GPS + accelerometer only, single combined CSV")
parser.add_argument("--csv", metavar="NAME", nargs="?", const="",
                    help=(
                        "Enable CSV logging. "
                        "Standard mode: NAME is a filename prefix — each sensor gets its own "
                        "file, e.g. run1_accelerometer.csv (default prefix: 'data'). "
                        "Mobility mode: NAME is the output filename (default: 'sensagram.csv')."
                    ))
args = parser.parse_args()

# ------------------------------------------------------------------ #
# Logging: FileHandler always on; StreamHandler only when no CSV
# ------------------------------------------------------------------ #

csv_active = args.csv is not None   # set module-level flag used by data_print()

log_fmt  = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger.setLevel(logging.INFO)

# server.log — always written, regardless of CSV or console state
file_handler = logging.FileHandler("server.log", encoding="utf-8")
file_handler.setFormatter(log_fmt)
logger.addHandler(file_handler)

# Console — always shows operational messages (log()); sensor data (data_print())
# is suppressed by csv_active when CSV is on, so the console stays useful either way.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_fmt)
logger.addHandler(console_handler)

# ------------------------------------------------------------------ #
# Build the handler object for the chosen mode
# ------------------------------------------------------------------ #

if args.mobility:
    if args.csv is None:   csv_arg = None
    elif args.csv == "":   csv_arg = "sensagram"
    else:                  csv_arg = args.csv

    mode = MobilityMode(csv_filename=csv_arg)
    log("[MODE] Mobility — GPS + accelerometer only")
else:
    if args.csv is None:   csv_prefix = None
    elif args.csv == "":   csv_prefix = "data"
    else:                  csv_prefix = args.csv

    mode = StandardMode(csv_prefix=csv_prefix)
    log("[MODE] Standard — all sensor types")
    if csv_prefix:
        log(f"[CSV] Logging enabled  prefix='{_clean_prefix(csv_prefix)}'")

if csv_active:
    log("[CSV] Console sensor output suppressed — data going to CSV file(s)")

log(f"[LOG] Operational messages → server.log")

# ------------------------------------------------------------------ #
# Start transport
# ------------------------------------------------------------------ #

if args.tcp:
    run_tcp_server(args.port, mode.handle)
else:
    run_udp_server(args.port, mode.handle)
