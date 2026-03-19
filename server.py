"""
server.py  —  SensaGram receiver  (multi-device edition)

Supports up to MAX_DEVICES simultaneous SensaGram devices.  Each device is
identified by its source IP address — no changes to the Android app are
required.  Every device gets its own CSV file and its own GPS/sensor state so
there is no cross-contamination between devices.

Standard mode  : receives all sensor types, logs each sensor to its own CSV
                 file per device  (e.g.  run1_192_168_1_5_accelerometer.csv).

Mobility mode  : --mobility
                 GPS + accelerometer only.  One combined CSV per device
                 (e.g.  sensagram_192_168_1_5.csv).

Transport notes
  TCP  : each TCP connection is already a separate thread with a known addr.
  UDP  : udpserver.py now passes (data, addr) to the callback so the IP can
         be extracted from addr[0].

Usage examples
    python3 server.py                                 # UDP, console only
    python3 server.py --tcp                           # TCP, console only
    python3 server.py --tcp --csv                     # TCP + per-device CSVs
    python3 server.py --csv run1                      # UDP + run1_<ip>_*.csv
    python3 server.py --port 9000 --tcp --csv run1

    python3 server.py --mobility
    python3 server.py --mobility --tcp
    python3 server.py --mobility --tcp --csv
    python3 server.py --mobility --tcp --csv my_run   # my_run_<ip>.csv
"""

from udpserver import UDPServer
import json, csv, argparse, socket, threading, logging, sys
from datetime import datetime


# =========================================================================== #
# Configuration
# =========================================================================== #

MAX_DEVICES = 10   # hard cap on simultaneous devices


# =========================================================================== #
# Logging setup
# =========================================================================== #

logger     = logging.getLogger("sensagram")
csv_active = False


def log(msg):
    logger.info(msg)


def data_print(msg, addr=None):
    """Print sensor data to console only when no CSV is active."""
    if not csv_active:
        prefix = f"[{addr}] " if addr else ""
        print(prefix + msg)


# =========================================================================== #
# Device registry — thread-safe, capped at MAX_DEVICES
# =========================================================================== #

class DeviceRegistry:
    """
    Tracks which device keys have been seen.  Once MAX_DEVICES unique keys
    are registered any new key is silently dropped with a log warning.

    The key is a UUID stamped by the app (preferred) or source IP (fallback).
    Call get_or_register(key) before creating per-device state.
    Returns the key on success, None if the cap has been reached.
    """

    def __init__(self):
        self._known: set[str] = set()
        self._lock  = threading.Lock()

    def get_or_register(self, key: str) -> "str | None":
        with self._lock:
            if key in self._known:
                return key
            if len(self._known) >= MAX_DEVICES:
                log(f"[REGISTRY] Device limit ({MAX_DEVICES}) reached — "
                    f"ignoring {key}")
                return None
            self._known.add(key)
            log(f"[REGISTRY] New device: {key}  "
                f"({len(self._known)}/{MAX_DEVICES} active)")
            return key

    def active(self) -> list[str]:
        with self._lock:
            return sorted(self._known)


def _device_key(jsonData: dict, addr) -> str:
    """
    Return a stable identifier for the sending device.

    Preference order:
      1. ``device_id`` field in the JSON payload — a UUID generated once by the
         Android app and persisted in DataStore.  Survives IP changes, LTE
         reconnects, NAT rebinds, and app restarts.
      2. Source IP address — fallback for older app versions that do not yet
         send ``device_id``.  Unreliable on mobile networks but better than
         nothing.
    """
    did = str(jsonData.get("device_id", "")).strip()
    if did:
        return did
    return addr[0] if isinstance(addr, tuple) else str(addr)


def _key_to_tag(key: str) -> str:
    """
    Convert a device key to a filesystem-safe string for use in filenames.
    UUID : '550e8400-e29b-41d4-a716-446655440000'
        -> '550e8400_e29b_41d4_a716_446655440000'
    IP   : '192.168.1.5' -> '192_168_1_5'
    """
    return key.replace("-", "_").replace(".", "_").replace(":", "_")


# =========================================================================== #
# Shared formatting helpers  (unchanged from original)
# =========================================================================== #

def _clean_prefix(prefix):
    return prefix[:-4] if prefix.endswith(".csv") else prefix


def fmt3(values):
    if values and len(values) >= 3:
        return f"x={values[0]:.5f}  y={values[1]:.5f}  z={values[2]:.5f}"
    return str(values)


def fmt1(values):
    if values and len(values) >= 1:
        return f"{values[0]:.5f}"
    return str(values)


def print_stats(label, jsonData, formatter=fmt3, addr=None):
    values = jsonData.get("values")
    mins   = jsonData.get("min")
    maxs   = jsonData.get("max")
    avg    = jsonData.get("avg")
    stddev = jsonData.get("stdDev")
    data_print(f"  {label}", addr)
    if values is not None: data_print(f"    current : {formatter(values)}", addr)
    if mins   is not None: data_print(f"    min     : {formatter(mins)}", addr)
    if maxs   is not None: data_print(f"    max     : {formatter(maxs)}", addr)
    if avg    is not None: data_print(f"    avg     : {formatter(avg)}", addr)
    if stddev is not None: data_print(f"    std-dev : {formatter(stddev)}", addr)


# =========================================================================== #
# STANDARD MODE — per-sensor, per-device CSV logging
# =========================================================================== #

class StandardMode:
    """
    Receives all sensor types.  Each (device, sensor_type) pair gets its own
    CSV file.  File naming: {prefix}_{ip_tag}_{sensor_key}.csv

    handle(data, addr) is the entry point.  addr may be a (ip, port) tuple
    (UDP) or a plain ip string (TCP).
    """

    def __init__(self, csv_prefix=None):
        self._prefix   = _clean_prefix(csv_prefix) if csv_prefix else None
        self._registry = DeviceRegistry()
        # keyed by (ip, sensor_key) → csv.DictWriter
        self._writers: dict[tuple, csv.DictWriter] = {}
        self._files:   dict[tuple, object]         = {}
        self._lock     = threading.Lock()

    # ------------------------------------------------------------------ #
    # CSV helpers
    # ------------------------------------------------------------------ #

    def _get_writer(self, device_key, sensor_key, fieldnames):
        key = (device_key, sensor_key)
        with self._lock:
            if key not in self._writers:
                tag      = _key_to_tag(device_key)
                filename = f"{self._prefix}_{tag}_{sensor_key}.csv"
                f = open(filename, "w", newline="")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                self._files[key]   = f
                self._writers[key] = writer
                log(f"[CSV] {device_key} / {sensor_key} → {filename}")
            return self._writers[key], self._files[key]

    def _log_xyz(self, device_key, sensor_key, received_at, timestamp, jsonData):
        fields = [
            "received_at", "timestamp",
            "x", "y", "z",
            "min_x", "min_y", "min_z",
            "max_x", "max_y", "max_z",
            "avg_x", "avg_y", "avg_z",
            "stddev_x", "stddev_y", "stddev_z",
        ]
        writer, f = self._get_writer(device_key, sensor_key, fields)

        def u(k):
            v = jsonData.get(k)
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
        f.flush()

    def _log_scalar(self, device_key, sensor_key, received_at, timestamp, jsonData):
        fields = ["received_at", "timestamp", "value", "min", "max", "avg", "stddev"]
        writer, f = self._get_writer(device_key, sensor_key, fields)

        def first(k):
            v = jsonData.get(k)
            return v[0] if v else None

        writer.writerow({
            "received_at": received_at, "timestamp": timestamp,
            "value":  first("values"), "min": first("min"),
            "max":    first("max"),    "avg": first("avg"),
            "stddev": first("stdDev"),
        })
        f.flush()

    def _log_gps(self, device_key, received_at, jsonData):
        fields = [
            "received_at", "gps_time", "latitude", "longitude", "altitude",
            "speed", "bearing", "accuracy",
            "speed_accuracy", "bearing_accuracy", "vertical_accuracy",
        ]
        writer, f = self._get_writer(device_key, "gps", fields)
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
        f.flush()

    # ------------------------------------------------------------------ #
    # Packet handler
    # ------------------------------------------------------------------ #

    def handle(self, data, addr):
        """
        Resolve the device key from the JSON payload first (stable UUID stamped
        by the app), falling back to source IP for older app versions.
        """
        jsonData    = json.loads(data)
        device_key  = _device_key(jsonData, addr)

        if self._registry.get_or_register(device_key) is None:
            return   # device limit reached

        sensorType  = jsonData["type"]
        timestamp   = jsonData.get("timestamp")
        received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        do_csv      = self._prefix is not None

        if sensorType == "android.sensor.accelerometer":
            data_print(f"[ACCELEROMETER]  ts={timestamp}", device_key)
            print_stats("acceleration (m/s²)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "accelerometer", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gyroscope":
            data_print(f"[GYROSCOPE]  ts={timestamp}", device_key)
            print_stats("angular velocity (rad/s)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "gyroscope", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gravity":
            data_print(f"[GRAVITY]  ts={timestamp}", device_key)
            print_stats("gravity (m/s²)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "gravity", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.linear_acceleration":
            data_print(f"[LINEAR ACCELERATION]  ts={timestamp}", device_key)
            print_stats("linear accel (m/s²)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "linear_acceleration", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.rotation_vector":
            data_print(f"[ROTATION VECTOR]  ts={timestamp}", device_key)
            print_stats("rotation vector", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "rotation_vector", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.game_rotation_vector":
            data_print(f"[GAME ROTATION VECTOR]  ts={timestamp}", device_key)
            print_stats("game rotation vector", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "game_rotation_vector", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.magnetic_field":
            data_print(f"[MAGNETIC FIELD]  ts={timestamp}", device_key)
            print_stats("magnetic field (µT)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "magnetic_field", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.magnetic_field_uncalibrated":
            data_print(f"[MAGNETIC FIELD UNCALIBRATED]  ts={timestamp}", device_key)
            print_stats("mag uncal (µT)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "magnetic_field_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.gyroscope_uncalibrated":
            data_print(f"[GYROSCOPE UNCALIBRATED]  ts={timestamp}", device_key)
            print_stats("gyro uncal (rad/s)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "gyroscope_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.accelerometer_uncalibrated":
            data_print(f"[ACCELEROMETER UNCALIBRATED]  ts={timestamp}", device_key)
            print_stats("accel uncal (m/s²)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "accelerometer_uncalibrated", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.orientation":
            data_print(f"[ORIENTATION]  ts={timestamp}", device_key)
            print_stats("orientation (°)", jsonData, addr=device_key)
            if do_csv: self._log_xyz(device_key, "orientation", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.proximity":
            data_print(f"[PROXIMITY]  ts={timestamp}", device_key)
            print_stats("proximity (cm)", jsonData, formatter=fmt1, addr=device_key)
            if do_csv: self._log_scalar(device_key, "proximity", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.step_counter":
            values = jsonData.get("values", [0])
            data_print(f"[STEP COUNTER]  ts={timestamp}  steps={values[0]:.0f}", device_key)
            if do_csv: self._log_scalar(device_key, "step_counter", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.step_detector":
            data_print(f"[STEP DETECTED]  ts={timestamp}", device_key)

        elif sensorType == "android.sensor.light":
            data_print(f"[LIGHT]  ts={timestamp}", device_key)
            print_stats("illuminance (lux)", jsonData, formatter=fmt1, addr=device_key)
            if do_csv: self._log_scalar(device_key, "light", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.pressure":
            data_print(f"[PRESSURE]  ts={timestamp}", device_key)
            print_stats("pressure (hPa)", jsonData, formatter=fmt1, addr=device_key)
            if do_csv: self._log_scalar(device_key, "pressure", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.ambient_temperature":
            data_print(f"[TEMPERATURE]  ts={timestamp}", device_key)
            print_stats("temperature (°C)", jsonData, formatter=fmt1, addr=device_key)
            if do_csv: self._log_scalar(device_key, "ambient_temperature", received_at, timestamp, jsonData)

        elif sensorType == "android.sensor.relative_humidity":
            data_print(f"[HUMIDITY]  ts={timestamp}", device_key)
            print_stats("relative humidity (%)", jsonData, formatter=fmt1, addr=device_key)
            if do_csv: self._log_scalar(device_key, "relative_humidity", received_at, timestamp, jsonData)

        elif sensorType == "android.gps":
            lat      = jsonData["latitude"]
            lon      = jsonData["longitude"]
            alt      = jsonData["altitude"]
            speed    = jsonData["speed"]
            bearing  = jsonData["bearing"]
            accuracy = jsonData["accuracy"]
            gps_time = jsonData["time"]
            data_print(f"[GPS]", device_key)
            data_print(f"  lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f} m", device_key)
            data_print(f"  speed={speed:.2f} m/s  bearing={bearing:.1f}°  accuracy={accuracy:.1f} m", device_key)
            data_print(f"  time={gps_time}", device_key)
            if do_csv: self._log_gps(device_key, received_at, jsonData)

        else:
            values = jsonData.get("values")
            data_print(f"[{sensorType}]  ts={timestamp}", device_key)
            data_print(f"  values={values}", device_key)
            if do_csv:
                self._log_xyz(device_key, sensorType.split(".")[-1], received_at, timestamp, jsonData)

        data_print("", device_key)


# =========================================================================== #
# MOBILITY MODE — GPS + accelerometer → one combined CSV per device
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


class _DeviceMobilityState:
    """Holds per-device state for MobilityMode."""
    __slots__ = ("latest_gps", "csv_writer", "csv_file")

    def __init__(self):
        self.latest_gps  = None
        self.csv_writer  = None
        self.csv_file    = None


class MobilityMode:
    """
    GPS + accelerometer only.  One CSV per device.

    CSV row trigger  : every ACCELEROMETER packet.
    GPS packets update the per-device cache; they do not write a row.
    All other sensor types are silently ignored.

    File naming (when csv_prefix is set):
        {prefix}_{ip_tag}.csv
        e.g.  sensagram_192_168_1_5.csv
    """

    def __init__(self, csv_prefix=None):
        self._prefix   = csv_prefix   # None = no CSV (console only)
        self._registry = DeviceRegistry()
        self._states:  dict[str, _DeviceMobilityState] = {}
        self._lock     = threading.Lock()

    # ------------------------------------------------------------------ #
    # Per-device state management
    # ------------------------------------------------------------------ #

    def _get_state(self, device_key: str) -> "_DeviceMobilityState | None":
        """
        Return the existing state for device_key, or create a new one.
        Returns None if the device cap has been reached.
        """
        with self._lock:
            if device_key in self._states:
                return self._states[device_key]

            # First time we see this device — try to register it.
            if self._registry.get_or_register(device_key) is None:
                return None   # cap reached

            state = _DeviceMobilityState()

            if self._prefix:
                tag  = _key_to_tag(device_key)
                path = f"{self._prefix}_{tag}.csv"
                f    = open(path, "w", newline="")
                writer = csv.DictWriter(f, fieldnames=MOBILITY_CSV_FIELDS)
                writer.writeheader()
                f.flush()
                state.csv_writer = writer
                state.csv_file   = f
                log(f"[CSV] Device {device_key} → {path}")

            self._states[device_key] = state
            return state

    # ------------------------------------------------------------------ #
    # CSV write helper
    # ------------------------------------------------------------------ #

    def _val(self, d, key, idx=None):
        if d is None:
            return ""
        v = d.get(key)
        if v is None:
            return ""
        if idx is not None:
            return v[idx] if len(v) > idx else ""
        return v

    def _write_row(self, state: _DeviceMobilityState, accel, received_at):
        gps = state.latest_gps
        v   = self._val
        state.csv_writer.writerow({
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
        state.csv_file.flush()

    # ------------------------------------------------------------------ #
    # Packet handler
    # ------------------------------------------------------------------ #

    def handle(self, data, addr):
        """
        Resolve the device key from the JSON payload (stable UUID from app),
        falling back to source IP for older app versions.
        """
        jsonData   = json.loads(data)
        device_key = _device_key(jsonData, addr)
        state      = self._get_state(device_key)
        if state is None:
            return   # device limit reached

        sensorType = jsonData["type"]

        if sensorType == "android.sensor.accelerometer":
            received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            ts     = jsonData.get("timestamp")
            values = jsonData.get("values", [None, None, None])
            data_print(f"[ACCELEROMETER]  ts={ts}", device_key)
            data_print(f"  current : x={values[0]}  y={values[1]}  z={values[2]}", device_key)
            for stat in ("min", "max", "avg", "stdDev"):
                sv = jsonData.get(stat)
                if sv is not None:
                    data_print(f"  {stat.ljust(6)}: x={sv[0]}  y={sv[1]}  z={sv[2]}", device_key)
            if state.latest_gps is None:
                data_print("  gps    : (no fix yet)", device_key)
            else:
                g = state.latest_gps
                data_print(
                    f"  gps    : lat={g.get('latitude'):.6f}  "
                    f"lon={g.get('longitude'):.6f}  "
                    f"alt={g.get('altitude'):.1f} m",
                    device_key
                )
            data_print("", device_key)
            if state.csv_writer:
                self._write_row(state, jsonData, received_at)

        elif sensorType == "android.gps":
            state.latest_gps = jsonData
            lat      = jsonData.get("latitude")
            lon      = jsonData.get("longitude")
            alt      = jsonData.get("altitude")
            speed    = jsonData.get("speed")
            bearing  = jsonData.get("bearing")
            accuracy = jsonData.get("accuracy")
            data_print(
                f"[GPS]  lat={lat:.6f}  lon={lon:.6f}  "
                f"alt={alt:.1f} m  spd={speed:.2f} m/s  "
                f"brg={bearing:.1f}°  acc={accuracy:.1f} m",
                device_key
            )
            data_print("", device_key)

        # all other types silently ignored in mobility mode


# =========================================================================== #
# TCP / UDP transport  (unchanged structure; handle() signature updated)
# =========================================================================== #

TCP_RECV_TIMEOUT = 60
TCP_KA_IDLE  = 15
TCP_KA_INTVL = 5
TCP_KA_CNT   = 3

_handler_lock = threading.Lock()


def _apply_keepalive(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "TCP_KEEPIDLE"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  TCP_KA_IDLE)
    if hasattr(socket, "TCP_KEEPINTVL"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KA_INTVL)
    if hasattr(socket, "TCP_KEEPCNT"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,   TCP_KA_CNT)


def handle_tcp_client(conn, addr, handler):
    """
    Receive newline-delimited JSON from one TCP client.

    Key change vs original: handler is now called as
        handler(line, addr[0])
    so MobilityMode/StandardMode can route the packet to the correct
    per-device state using the client's IP address.
    """
    MAX_BUF_BYTES = 256 * 1024

    log(f"[TCP] Connected: {addr}")
    _apply_keepalive(conn)
    conn.settimeout(TCP_RECV_TIMEOUT)

    buf = ""
    try:
        while True:
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                log(f"[TCP] Idle timeout ({TCP_RECV_TIMEOUT} s): {addr} — closing")
                break
            except (ConnectionResetError, ConnectionAbortedError) as e:
                log(f"[TCP] Connection reset by peer {addr}: {e}")
                break

            if not chunk:
                break

            buf += chunk.decode("utf-8", errors="replace")

            if len(buf) > MAX_BUF_BYTES:
                log(f"[TCP] Buffer overflow from {addr} ({len(buf)} bytes) — discarding")
                buf = ""
                continue

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    with _handler_lock:
                        try:
                            # ← pass addr[0] (the IP string) as the second arg
                            handler(line, addr[0])
                        except Exception as e:
                            log(f"[TCP] Parse/handler error from {addr}: {e}")

    except OSError as e:
        log(f"[TCP] Socket error from {addr}: {e}")
    finally:
        try:
            conn.close()
        except OSError:
            pass
        log(f"[TCP] Disconnected: {addr}")


def run_tcp_server(port, handler):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _apply_keepalive(srv)
    srv.bind(("0.0.0.0", port))
    srv.listen(10)   # backlog raised to match MAX_DEVICES
    log(f"[TCP] Listening on 0.0.0.0:{port}  (up to {MAX_DEVICES} devices)")
    log(f"[TCP] Keepalive: idle={TCP_KA_IDLE}s  interval={TCP_KA_INTVL}s  probes={TCP_KA_CNT}")
    log(f"[TCP] Recv timeout: {TCP_RECV_TIMEOUT}s")
    while True:
        try:
            conn, addr = srv.accept()
        except OSError as e:
            log(f"[TCP] accept() error: {e}")
            continue
        threading.Thread(
            target=handle_tcp_client, args=(conn, addr, handler), daemon=True
        ).start()


def run_udp_server(port, handler):
    """
    UDP server.  udpserver.py now calls handler(data, addr) where addr is
    (ip_string, port_int) — the IP is what MobilityMode/StandardMode uses.
    """
    log(f"[UDP] Listening on 0.0.0.0:{port}  (up to {MAX_DEVICES} devices)")
    server = UDPServer(address=("0.0.0.0", port))
    server.setDataCallBack(handler)
    server.start()
    # Block the main thread so the daemon listener keeps running.
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


# =========================================================================== #
# Entry point
# =========================================================================== #

parser = argparse.ArgumentParser(
    description="SensaGram receiver — multi-device edition",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""
Up to {MAX_DEVICES} devices are supported simultaneously.
Each device is identified by its source IP address.
CSV files are named  <prefix>_<ip>_<sensor>.csv  (standard)
                 or  <prefix>_<ip>.csv            (mobility).

Examples:
  Standard mode:
    python3 server.py
    python3 server.py --tcp --csv run1

  Mobility mode:
    python3 server.py --mobility --tcp --csv my_run
"""
)
parser.add_argument("--port",     type=int, default=47892)
parser.add_argument("--tcp",      action="store_true")
parser.add_argument("--mobility", action="store_true")
parser.add_argument("--csv",      metavar="NAME", nargs="?", const="",
                    help="Enable CSV logging (prefix or filename stem).")
args = parser.parse_args()

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

csv_active = args.csv is not None

log_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("server.log", encoding="utf-8")
file_handler.setFormatter(log_fmt)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_fmt)
logger.addHandler(console_handler)

# ------------------------------------------------------------------ #
# Mode selection
# ------------------------------------------------------------------ #

if args.mobility:
    if args.csv is None:    csv_arg = None
    elif args.csv == "":    csv_arg = "sensagram"
    else:                   csv_arg = args.csv

    mode = MobilityMode(csv_prefix=csv_arg)
    log("[MODE] Mobility — GPS + accelerometer only")
    if csv_arg:
        log(f"[CSV]  Output pattern: {csv_arg}_<ip>.csv")
else:
    if args.csv is None:    csv_prefix = None
    elif args.csv == "":    csv_prefix = "data"
    else:                   csv_prefix = args.csv

    mode = StandardMode(csv_prefix=csv_prefix)
    log("[MODE] Standard — all sensor types")
    if csv_prefix:
        log(f"[CSV]  Output pattern: {_clean_prefix(csv_prefix)}_<ip>_<sensor>.csv")

if csv_active:
    log("[CSV]  Console sensor output suppressed — data going to CSV files")

log(f"[LOG]  Operational messages → server.log")
log(f"[INFO] Max simultaneous devices: {MAX_DEVICES}")

# ------------------------------------------------------------------ #
# Transport
# ------------------------------------------------------------------ #

if args.tcp:
    run_tcp_server(args.port, mode.handle)
else:
    run_udp_server(args.port, mode.handle)
