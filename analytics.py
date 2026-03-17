#!/usr/bin/env python3
"""
mobility_analytics.py  —  GPS + IMU fusion for motorbike dynamics
==================================================================

Produces per-timestep estimates of:
  • Linear acceleration (gravity removed, along-track / lateral / vertical)
  • Velocity            (GPS raw components, Kalman-fused, 3-D)
  • Displacement        (GPS raw, IMU dead-reckoning, Kalman-fused)
  • Altitude change     (GPS raw, Kalman-smoothed)

Works correctly for any phone placement — dashboard mount, tank bag,
jacket pocket, or trouser pocket — without knowing the mounting angle
in advance.

─────────────────────────────────────────────────────────────────────
Why gravity removal is hard and how this code solves it
─────────────────────────────────────────────────────────────────────

The accelerometer always measures  a_body = R @ (a_linear + g_world),
where R is the unknown rotation from world ENU to body frame.
Removing gravity requires knowing R, which changes whenever the phone
is moved or rotated.

OBSERVATION: gravity is a near-DC signal in the body frame.
It only changes when the phone's physical orientation changes (seconds
to minutes).  Vehicle dynamics (braking, acceleration, cornering) are
higher-frequency and approximately zero-mean over any multi-second
window.  Therefore a sufficiently long exponential moving average (EMA)
of accel_avg converges to the gravity vector regardless of orientation,
because the zero-mean vehicle dynamics average out while the true DC
gravity component accumulates in the filtered output.

This is the same principle used by Android's internal TYPE_LINEAR_
ACCELERATION sensor, which applies a complementary filter.  Here we
additionally gate the update rate using GPS-derived acceleration so
the EMA does not track sustained acceleration (e.g. a long uphill).

─────────────────────────────────────────────────────────────────────
GPS-gated three-speed EMA gravity tracker
─────────────────────────────────────────────────────────────────────

Three update modes are selected per-row based on GPS speed and GPS-
derived along-track acceleration magnitude |a_GPS| = |d(speed)/dt|:

  Mode 0 — FROZEN   |a_GPS| > GRAV_FREEZE_ACCEL  (default 0.5 m/s²)
    α = 1.0  →  gravity estimate held fixed
    Rationale: sustained linear acceleration would bias the EMA if
    allowed to update.  GPS-detected hard braking or acceleration
    freezes the filter until the event passes.

  Mode 1 — STATIC   GPS speed < GRAV_STATIC_SPEED  (default 0.5 m/s)
    α = exp(−dt / τ_static)   τ_static default 5 s
    Rationale: the bike is stopped; accel_avg is pure gravity.
    Snap the estimate quickly to the current reading.

  Mode 2 — MOVING   all other rows
    α = exp(−dt / τ_moving)   τ_moving default 60 s
    Rationale: vehicle dynamics are small but non-zero; a long time
    constant ensures they average out before significantly biasing
    the gravity estimate.  Orientation changes in a pocket are slow
    (human body movements) so 60 s is fast enough to track them.

EMA update each row:
    gravity_k = α · gravity_{k-1} + (1−α) · accel_avg_k

Seeding: the first GPS-stationary row (or first row if none exist)
initialises the estimate.  After the first traffic-light stop the
estimate is accurate to the within-window noise floor.

Per-row gravity vectors are stored in gravity_est_x/y/z columns so
the tracking trajectory can be inspected.  grav_update_mode records
which mode (0/1/2) fired at each row.

─────────────────────────────────────────────────────────────────────
GPS-derived acceleration columns (orientation-free ground truth)
─────────────────────────────────────────────────────────────────────

For battery sizing the GPS alone can supply all three acceleration
components without any knowledge of phone orientation.  These are
stored alongside the IMU-derived values for cross-validation:

  gps_accel_forward  = d(GPS_speed)/dt           (m/s², along-track)
  gps_accel_vertical = d(d(altitude)/dt)/dt      (m/s², vertical)
  gps_accel_lateral  = GPS_speed × d(bearing)/dt (m/s², centripetal)

GPS derivatives are noisy at 5 s intervals; a central-difference
Savitzky-Golay-style smoothing is applied via np.gradient.  For clean
power demand calculations, prefer these GPS-derived values over IMU
for the along-track and vertical axes.

─────────────────────────────────────────────────────────────────────
Mounting-angle auto-calibration
─────────────────────────────────────────────────────────────────────

After gravity removal the residual body-frame acceleration has an
unknown rotation relative to the bike's fore-aft axis.  The mounting
offset φ is estimated from:

    d(GPS_speed)/dt  ≈  lin_x·cos φ + lin_y·sin φ

Least-squares over motion rows gives [cos φ, sin φ] → φ.
Requires ≥ 10 rows with GPS speed > 1.5 m/s.

For pocket use: the auto-calibration still attempts a best-fit but
may be unreliable if the phone orientation varied significantly during
the ride.  The lin_accel_mag (rotation-invariant) and GPS-derived
columns remain reliable regardless.

─────────────────────────────────────────────────────────────────────
Reliability summary by phone placement
─────────────────────────────────────────────────────────────────────

  Column                Dashboard  Tank bag  Pocket
  ──────────────────────────────────────────────────
  GPS displacement         ✓         ✓         ✓
  GPS altitude             ✓         ✓         ✓
  Kalman fused distance    ✓         ✓         ✓
  gps_speed_ms (scalar)    ✓         ✓         ✓
  gps_vel_E/N/vert         ✓         ✓         ✓
  gps_speed_3d_ms          ✓         ✓         ✓
  fused_speed_ms           ✓         ✓         ✓
  fused_speed_3d_ms        ✓         ✓         ✓
  gps_accel_forward        ✓         ✓         ✓
  gps_accel_vertical       ✓         ✓         ✓
  gps_accel_lateral        ✓         ✓         ✓
  lin_accel_mag (IMU)      ✓         ✓         ✓  ← rotation-invariant
  gravity_est (tracker)    ✓         ✓         ~  ← valid after 1st stop
  accel_forward (IMU)      ✓         ✓         ~  ← depends on auto-cal
  accel_vertical (IMU)     ✓         ✓         ✗  ← body frame unknown
  accel_lateral  (IMU)     ✓         ✓         ✗  ← body frame unknown

─────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────
  # Default — adaptive gravity, auto mounting-angle:
      python3 mobility_analytics.py run.csv

  # Tune gravity time constants (seconds):
      python3 mobility_analytics.py run.csv --tau-static 3 --tau-moving 90

  # Explicit mounting angle override:
      python3 mobility_analytics.py run.csv --offset 30

  # Tune Kalman noise:
      python3 mobility_analytics.py run.csv --gps-r-pos 8 --kf-q 0.03

─────────────────────────────────────────────────────────────────────
Output
─────────────────────────────────────────────────────────────────────
  analytics_<stem>.csv   — per-row results table
  Summary printed to stdout
"""

import sys, os, argparse
import numpy as np
import pandas as pd

# ── Physical constants ────────────────────────────────────────────────────────
G_STD   = 9.80665   # m/s²  standard gravity
R_EARTH = 6_371_000 # m     mean Earth radius

# ── Adaptive gravity tracker defaults ────────────────────────────────────────
GRAV_FREEZE_ACCEL   = 0.5    # |d(GPS_speed)/dt| above which mode = FROZEN (m/s²)
GRAV_STATIC_SPEED   = 0.5    # GPS speed below which mode = STATIC (m/s)
GRAV_TAU_STATIC_DEF = 5.0    # STATIC EMA time constant (s) — fast
GRAV_TAU_MOVING_DEF = 60.0   # MOVING EMA time constant (s) — slow

# ── Mounting-angle auto-calibration ──────────────────────────────────────────
AUTO_CAL_MIN_SPEED   = 1.5   # GPS speed (m/s) required for a calibration row
AUTO_CAL_MIN_SAMPLES = 10    # minimum motion rows needed

# ── GPS bearing reliability ───────────────────────────────────────────────────
MIN_SPEED_FOR_BEARING = 0.5  # m/s


# =========================================================================== #
# Kalman filter — 1-D  (position, velocity)
# =========================================================================== #

class KF1D:
    """
    1-D Kalman filter for (position, velocity) with IMU-driven prediction
    and independent GPS position / GPS velocity measurement updates.

    Parameters
    ----------
    q       : process noise power spectral density
    r_pos   : GPS position measurement noise std-dev (metres)
    r_vel   : GPS velocity measurement noise std-dev (m/s)
    x0      : initial state [position_m, velocity_ms]
    P0_diag : initial covariance diagonal [pos_var, vel_var]
    """
    def __init__(self, q=0.05, r_pos=12.0, r_vel=0.3,
                 x0=(0.0, 0.0), P0_diag=(100.0, 1.0)):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(P0_diag).astype(float)
        self.q     = q
        self.R_pos = r_pos ** 2
        self.R_vel = r_vel ** 2

    def predict(self, dt: float, a: float):
        """Propagate state by dt seconds using IMU acceleration a (m/s²)."""
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([0.5 * dt**2, dt])
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2],
                                [dt**3 / 2, dt**2    ]])
        self.x = F @ self.x + B * a
        self.P = F @ self.P @ F.T + Q

    def _update(self, h: np.ndarray, z: float, r_var: float):
        S     = float(h @ self.P @ h) + r_var
        K     = (self.P @ h) / S
        innov = z - float(np.dot(h, self.x))
        self.x = self.x + K * innov
        self.P = (np.eye(2) - np.outer(K, h)) @ self.P

    def update_pos(self, pos_m: float):
        self._update(np.array([1.0, 0.0]), pos_m, self.R_pos)

    def update_vel(self, vel_ms: float):
        self._update(np.array([0.0, 1.0]), vel_ms, self.R_vel)


# =========================================================================== #
# Coordinate helpers
# =========================================================================== #

def latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0):
    """Flat-Earth ENU in metres relative to a reference origin. Valid to ~50 km."""
    cos_lat0 = np.cos(np.radians(lat0))
    east  = R_EARTH * cos_lat0 * np.radians(lon - lon0)
    north = R_EARTH * np.radians(lat - lat0)
    up    = alt - alt0
    return east, north, up


# =========================================================================== #
# GPS-gated adaptive gravity tracker
# =========================================================================== #

def track_gravity_adaptive(accel_avg:  np.ndarray,
                            gps_speed: np.ndarray,
                            dt:        np.ndarray,
                            tau_static: float = GRAV_TAU_STATIC_DEF,
                            tau_moving: float = GRAV_TAU_MOVING_DEF,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the time-varying gravity vector in the phone body frame using a
    GPS-gated three-speed exponential moving average.

    Parameters
    ----------
    accel_avg  : (n, 3) per-window average accelerometer readings (m/s²)
    gps_speed  : (n,)   GPS Doppler speed in m/s (NaN for pre-fix rows)
    dt         : (n,)   timestep durations (s)
    tau_static : EMA time constant when stationary (s)
    tau_moving : EMA time constant when rolling (s)

    Returns
    -------
    grav_track   : (n, 3) gravity vector estimate at every row (m/s²)
    update_mode  : (n,)   integer per-row mode: 0=FROZEN, 1=STATIC, 2=MOVING
    """
    n = len(accel_avg)

    # GPS-derived along-track acceleration magnitude (central differences).
    # Used to detect hard acceleration / braking events.
    # Fill NaN speeds with adjacent values for the derivative, but keep a
    # NaN mask so we know which rows truly have no GPS.
    speed_filled = pd.Series(gps_speed).ffill().bfill().fillna(0.0).values
    time_s       = np.maximum(np.cumsum(dt), 1e-6)
    a_gps_at     = np.gradient(speed_filled, time_s)   # m/s²

    # ── Seed: use first GPS-stationary row, else first row ───────────────────
    gps_valid   = np.isfinite(gps_speed)
    static_rows = gps_valid & (gps_speed < GRAV_STATIC_SPEED)
    if static_rows.any():
        seed_idx = int(np.where(static_rows)[0][0])
    else:
        # No confirmed stationary row (pocket before first stop, or short ride).
        # Use the row with |accel| numerically closest to G_STD.
        seed_idx = int(np.argmin(np.abs(np.linalg.norm(accel_avg, axis=1) - G_STD)))

    gravity_est = accel_avg[seed_idx].copy().astype(float)

    grav_track  = np.zeros((n, 3), dtype=float)
    update_mode = np.zeros(n, dtype=int)

    for i in range(n):
        speed_i = gps_speed[i]
        a_at_i  = abs(a_gps_at[i])

        # Determine mode and EMA alpha for this row
        if gps_valid[i] and a_at_i > GRAV_FREEZE_ACCEL:
            mode  = 0          # FROZEN — hard accel/braking
            alpha = 1.0
        elif gps_valid[i] and speed_i < GRAV_STATIC_SPEED:
            mode  = 1          # STATIC — confirmed stopped
            alpha = float(np.exp(-dt[i] / tau_static))
        else:
            mode  = 2          # MOVING or no GPS — slow drift tracking
            alpha = float(np.exp(-dt[i] / tau_moving))

        # EMA: new_gravity = α·old + (1−α)·current_accel
        gravity_est = alpha * gravity_est + (1.0 - alpha) * accel_avg[i]

        grav_track[i]  = gravity_est
        update_mode[i] = mode

    return grav_track, update_mode


# =========================================================================== #
# Mounting-angle auto-calibration
# =========================================================================== #

def estimate_mount_offset(lin_x: np.ndarray,
                           lin_y: np.ndarray,
                           gps_speed: np.ndarray,
                           dt: np.ndarray,
                           ) -> tuple[float, bool, int]:
    """
    Estimate the phone mounting offset φ (degrees) from GPS speed.

    Physical model
    ──────────────
    The bike's forward direction makes angle φ with the phone's body x-axis.
    During straight-line motion:

        d(GPS_speed)/dt  ≈  lin_x·cos φ + lin_y·sin φ

    Stacked across motion rows:   A·u = b
        A  = [lin_x, lin_y]  per motion row
        b  = d(speed)/dt     (central differences)
        u  = [cos φ, sin φ]  (least-squares, then normalised)
        φ  = atan2(u[1], u[0])

    Assumptions
    ───────────
    1. Phone screen faces roughly upward (gravity ≈ body −z).
       In a pocket on its side this is violated; lin_accel_mag is then
       more reliable than the projected components.
    2. Calibration rows contain forward acceleration or braking, not
       pure cornering.

    Returns
    -------
    offset_deg : estimated mounting offset (degrees)
    success    : True if calibration used real motion data
    n_samples  : number of motion rows used
    """
    time_s    = np.maximum(np.cumsum(dt), 1e-6)
    a_gps_at  = np.gradient(
        pd.Series(gps_speed).ffill().bfill().fillna(0.0).values, time_s)

    mask      = np.isfinite(gps_speed) & (gps_speed > AUTO_CAL_MIN_SPEED)
    n_samples = int(mask.sum())

    if n_samples < AUTO_CAL_MIN_SAMPLES:
        return 0.0, False, n_samples

    A      = np.column_stack([lin_x[mask], lin_y[mask]])
    b_clip = np.clip(a_gps_at[mask], -5.0, 5.0)   # clip GPS glitches (±0.5 g)

    u, _, _, _ = np.linalg.lstsq(A, b_clip, rcond=None)
    norm_u     = np.linalg.norm(u)
    if norm_u < 1e-9:
        return 0.0, False, n_samples

    u  /= norm_u
    phi = np.degrees(np.arctan2(u[1], u[0]))
    return phi, True, n_samples


# =========================================================================== #
# Main analysis function
# =========================================================================== #

def analyse(df: pd.DataFrame,
            mount_offset_deg: float | None = None,
            tau_static:       float        = GRAV_TAU_STATIC_DEF,
            tau_moving:       float        = GRAV_TAU_MOVING_DEF,
            gps_r_pos:        float        = 12.0,
            gps_r_vel:        float        = 0.30,
            kf_q:             float        = 0.05) -> pd.DataFrame:
    """
    Run the full GPS + IMU fusion pipeline.

    Parameters
    ----------
    df               : parsed CSV DataFrame (from --mobility server mode)
    mount_offset_deg : explicit mounting offset override (degrees).
                       None → auto-calibrate.  0.0 → force zero, no cal.
    tau_static       : gravity EMA time constant while stationary (s)
    tau_moving       : gravity EMA time constant while rolling (s)
    gps_r_pos        : GPS horizontal position noise std-dev (m)
    gps_r_vel        : GPS velocity noise std-dev (m/s)
    kf_q             : Kalman filter process noise
    """
    n = len(df)

    # ── Timestamps / Δt ───────────────────────────────────────────────────────
    t         = pd.to_datetime(df["received_at"])
    elapsed_s = (t - t.iloc[0]).dt.total_seconds().values
    dt        = np.diff(elapsed_s, prepend=elapsed_s[0])
    dt[0]     = dt[1] if n > 1 else 5.0
    dt        = np.maximum(dt, 0.001)

    # ── IMU arrays (NaN → forward-fill) ──────────────────────────────────────
    def _col(name, fallback=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(fallback).values
        return np.full(n, fallback)

    accel_avg = np.column_stack([_col("accel_avg_x"), _col("accel_avg_y"),
                                  _col("accel_avg_z")])
    accel_avg = pd.DataFrame(accel_avg).ffill().bfill().values

    # ── GPS — tolerant parsing (blank cells → NaN) ────────────────────────────
    lat       = pd.to_numeric(df["latitude"],  errors="coerce").values
    lon       = pd.to_numeric(df["longitude"], errors="coerce").values
    alt       = pd.to_numeric(df["altitude"],  errors="coerce").values
    gps_speed = pd.to_numeric(df["speed"],     errors="coerce").values

    valid_gps = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    first_fix = np.where(valid_gps)[0]
    if len(first_fix) == 0:
        raise ValueError("No valid GPS fixes found in the data.")
    i0 = first_fix[0]
    lat0, lon0, alt0 = lat[i0], lon[i0], alt[i0]

    # ENU for valid rows; forward-fill gaps (Kalman still gets GPS updates only
    # on valid_gps rows — the fill is used for initialisation only)
    east_m = np.full(n, np.nan); north_m = np.full(n, np.nan); up_m = np.full(n, np.nan)
    e_v, n_v, u_v = latlon_to_enu_m(lat[valid_gps], lon[valid_gps],
                                      alt[valid_gps], lat0, lon0, alt0)
    east_m[valid_gps]  = e_v
    north_m[valid_gps] = n_v
    up_m[valid_gps]    = u_v

    east_ff  = pd.Series(east_m ).ffill().bfill().values
    north_ff = pd.Series(north_m).ffill().bfill().values
    up_ff    = pd.Series(up_m   ).ffill().bfill().values
    alt_ff   = pd.Series(alt    ).ffill().bfill().values

    d_east  = np.diff(east_ff,  prepend=east_ff[0])
    d_north = np.diff(north_ff, prepend=north_ff[0])
    d_up    = np.diff(up_ff,    prepend=up_ff[0])

    gps_disp_step_h  = np.sqrt(d_east**2 + d_north**2)
    gps_disp_step_3d = np.sqrt(d_east**2 + d_north**2 + d_up**2)
    gps_disp_cumul   = np.cumsum(gps_disp_step_h)
    delta_alt_gps    = d_up

    # GPS bearing (reported when speed reliable; else position-derived)
    pos_bearing = np.degrees(np.arctan2(d_east, d_north)) % 360
    if "bearing" in df.columns:
        rep_bearing = pd.to_numeric(df["bearing"], errors="coerce").values
        speed_ok    = np.isfinite(gps_speed) & (gps_speed > MIN_SPEED_FOR_BEARING)
        bearing_deg = np.where(speed_ok & np.isfinite(rep_bearing),
                               rep_bearing, pos_bearing)
    else:
        bearing_deg = pos_bearing
    bearing_rad = np.radians(bearing_deg)

    v_acc_arr  = _col("vertical_accuracy", fallback=5.0)
    gps_r_vert = float(np.nanmean(v_acc_arr))

    # ── GPS-derived acceleration (orientation-free ground truth) ─────────────
    # Central differences on GPS speed and altitude.  Smooth the bearing
    # before differentiating to reduce heading noise in lateral acceleration.
    time_s       = np.maximum(np.cumsum(dt), 1e-6)
    speed_filled = pd.Series(gps_speed).ffill().bfill().fillna(0.0).values

    gps_a_fwd  = np.gradient(speed_filled, time_s)           # along-track (m/s²)

    alt_rate   = np.gradient(alt_ff, time_s)                  # climb rate (m/s)
    gps_a_vert = np.gradient(alt_rate, time_s)                # vertical accel (m/s²)

    # Lateral (centripetal): a_lat = v × dθ/dt  where θ is bearing in radians.
    # Unwrap bearing to avoid 360→0 jumps before differentiating.
    bear_unwrap   = np.unwrap(np.radians(bearing_deg))
    bear_rate     = np.gradient(bear_unwrap, time_s)          # rad/s
    gps_a_lat     = speed_filled * bear_rate                  # centripetal (m/s²)

    gps_a_mag = np.sqrt(gps_a_fwd**2 + gps_a_vert**2 + gps_a_lat**2)

    # ── GPS velocity components ───────────────────────────────────────────────
    # Horizontal components from Doppler speed decomposed by GPS bearing.
    # Vertical component is the GPS altitude rate (climb/descent in m/s).
    # All three are orientation-free and valid regardless of phone placement.
    gps_vel_e    = speed_filled * np.sin(bearing_rad)   # East   (m/s, + = East)
    gps_vel_n    = speed_filled * np.cos(bearing_rad)   # North  (m/s, + = North)
    gps_vel_vert = alt_rate                             # Up     (m/s, + = climbing)
    gps_speed_3d = np.sqrt(speed_filled**2 + alt_rate**2)  # total 3-D speed (m/s)

    # ── Adaptive gravity tracking ─────────────────────────────────────────────
    grav_track, update_mode = track_gravity_adaptive(
        accel_avg, gps_speed, dt, tau_static, tau_moving)

    # Per-row gravity-removed linear acceleration in body frame
    lin_accel = accel_avg - grav_track        # shape (n, 3)
    lin_x     = lin_accel[:, 0]
    lin_y     = lin_accel[:, 1]
    lin_z     = lin_accel[:, 2]
    a_lin_mag = np.linalg.norm(lin_accel, axis=1)

    # Gravity magnitude at each row (should stay close to G_STD)
    g_mag_track = np.linalg.norm(grav_track, axis=1)

    # ── Mounting offset: auto-calibrate or explicit override ──────────────────
    offset_auto  = False
    offset_n_cal = 0

    if mount_offset_deg is not None:
        phi_deg   = float(mount_offset_deg)
        algo_used = f"along-track Kalman (offset={phi_deg:+.1f}° explicit)"
    else:
        phi_deg, offset_auto, offset_n_cal = estimate_mount_offset(
            lin_x, lin_y, gps_speed, dt)
        if offset_auto:
            algo_used = (f"along-track Kalman "
                         f"(offset={phi_deg:+.1f}° auto-cal, n={offset_n_cal})")
        else:
            phi_deg   = 0.0
            algo_used = (f"along-track Kalman "
                         f"(offset=0° — insufficient motion for auto-cal, "
                         f"n_motion={offset_n_cal} < {AUTO_CAL_MIN_SAMPLES})")

    # ── Project IMU onto bike axes ────────────────────────────────────────────
    phi_rad = np.radians(phi_deg)
    c, s    = np.cos(phi_rad), np.sin(phi_rad)
    a_fwd   =  c * lin_x + s * lin_y   # along bike, + = forward
    a_lat   = -s * lin_x + c * lin_y   # lateral,    + = left
    a_up    =  lin_z                   # vertical,   + = upward (valid if screen-up)

    # Approximate ENU via GPS bearing
    a_east  = a_fwd * np.sin(bearing_rad) + a_lat * np.cos(bearing_rad)
    a_north = a_fwd * np.cos(bearing_rad) - a_lat * np.sin(bearing_rad)

    # ── Two 1-D Kalman filters ────────────────────────────────────────────────
    kf_at = KF1D(q=kf_q,       r_pos=gps_r_pos,  r_vel=gps_r_vel)
    kf_vt = KF1D(q=kf_q * 0.4, r_pos=gps_r_vert, r_vel=0.5)

    fused_at = np.zeros(n); fused_spd_at = np.zeros(n)
    fused_u  = np.zeros(n); fused_vel_u  = np.zeros(n)

    for i in range(n):
        kf_at.predict(dt[i], a_fwd[i])
        kf_vt.predict(dt[i], a_up[i])
        if valid_gps[i]:
            kf_at.update_pos(gps_disp_cumul[i])
            kf_vt.update_pos(up_ff[i])
        if np.isfinite(gps_speed[i]):
            kf_at.update_vel(float(gps_speed[i]))
        fused_at[i]  = kf_at.x[0]; fused_spd_at[i] = kf_at.x[1]
        fused_u[i]   = kf_vt.x[0]; fused_vel_u[i]  = kf_vt.x[1]

    fused_disp_step  = np.diff(fused_at, prepend=fused_at[0])
    fused_disp_cumul = fused_at
    fused_speed      = fused_spd_at
    fused_dalt_step  = np.diff(fused_u,  prepend=fused_u[0])
    fused_alt_abs    = fused_u + alt0
    fused_e          = fused_at * np.sin(bearing_rad)
    fused_n          = fused_at * np.cos(bearing_rad)
    fused_vel_e      = fused_spd_at * np.sin(bearing_rad)
    fused_vel_n      = fused_spd_at * np.cos(bearing_rad)
    # 3-D fused speed: combines the along-track Kalman speed (horizontal)
    # and the vertical Kalman speed.  This is the total speed of the bike
    # through space including climb/descent.
    fused_speed_3d   = np.sqrt(fused_spd_at**2 + fused_vel_u**2)

    # ── IMU-only dead-reckoning (reference — drifts) ──────────────────────────
    imu_speed = np.zeros(n); imu_dist = np.zeros(n)
    v_imu = 0.0; d_imu = 0.0
    for i in range(n):
        v_imu += a_fwd[i] * dt[i]
        d_imu += v_imu * dt[i]
        imu_speed[i] = v_imu
        imu_dist[i]  = d_imu

    # ── Build results DataFrame ───────────────────────────────────────────────
    results = pd.DataFrame({
        # ── Time ────────────────────────────────────────────────────────────
        "received_at"            : df["received_at"].values,
        "elapsed_s"              : elapsed_s,
        "dt_s"                   : dt,
        "gps_fix_valid"          : valid_gps.astype(int),

        # ── Gravity tracking ────────────────────────────────────────────────
        # Per-row estimated gravity vector in the phone body frame (m/s²).
        # gravity_est_z ≈ +9.81 when the phone is screen-up (dashboard).
        # Magnitude should remain close to 9.807 throughout if tracking is good.
        "gravity_est_x"          : grav_track[:, 0],
        "gravity_est_y"          : grav_track[:, 1],
        "gravity_est_z"          : grav_track[:, 2],
        "gravity_est_mag"        : g_mag_track,
        # Mode: 0 = FROZEN (hard accel), 1 = STATIC (stopped), 2 = MOVING
        "grav_update_mode"       : update_mode,

        # ── 1a. IMU linear acceleration — body frame (gravity removed) ───────
        "accel_avg_x_raw"        : accel_avg[:, 0],
        "accel_avg_y_raw"        : accel_avg[:, 1],
        "accel_avg_z_raw"        : accel_avg[:, 2],
        # Projected onto bike axes (valid when phone is roughly screen-up and
        # auto-calibration succeeded; use lin_accel_mag for pocket placement)
        "accel_forward"          : a_fwd,       # + = accelerating
        "accel_lateral"          : a_lat,       # + = left
        "accel_vertical"         : a_up,        # + = upward
        "lin_accel_mag"          : a_lin_mag,   # rotation-invariant magnitude
        # Approximate ENU via GPS bearing
        "lin_accel_E"            : a_east,
        "lin_accel_N"            : a_north,

        # ── 1b. GPS-derived acceleration (orientation-free) ──────────────────
        # These are the most reliable values for battery sizing regardless of
        # phone placement.  They are noisy at 5 s sample intervals but have no
        # orientation ambiguity.
        "gps_accel_forward"      : gps_a_fwd,   # d(speed)/dt  (m/s²)
        "gps_accel_vertical"     : gps_a_vert,  # d²(alt)/dt²  (m/s²)
        "gps_accel_lateral"      : gps_a_lat,   # v × dθ/dt    (m/s², centripetal)
        "gps_accel_mag"          : gps_a_mag,   # total magnitude

        # ── 2. Velocity (m/s) ───────────────────────────────────────────────
        # GPS raw velocities — orientation-free, valid for any phone placement.
        "gps_speed_ms"           : gps_speed,           # horizontal Doppler (scalar)
        "gps_vel_east_ms"        : gps_vel_e,            # East component
        "gps_vel_north_ms"       : gps_vel_n,            # North component
        "gps_vel_vertical_ms"    : gps_vel_vert,         # Up (+climb) / Down (-descent)
        "gps_speed_3d_ms"        : gps_speed_3d,         # total 3-D speed
        # Kalman-fused velocities — GPS-corrected, smoothed by IMU prediction.
        # fused_speed_ms   = along-track horizontal speed (from along-track KF)
        # vertical_speed_ms = vertical speed             (from vertical KF)
        # fused_speed_3d_ms = combined 3-D speed
        "fused_speed_ms"         : fused_speed,
        "fused_vel_east_ms"      : fused_vel_e,
        "fused_vel_north_ms"     : fused_vel_n,
        "vertical_speed_ms"      : fused_vel_u,
        "fused_speed_3d_ms"      : fused_speed_3d,
        # IMU dead-reckoning speed (forward projection only — drifts without GPS)
        "imu_speed_ms"           : imu_speed,

        # ── 3. Displacement (m) ─────────────────────────────────────────────
        "gps_disp_step_m"        : gps_disp_step_h,
        "gps_disp_3d_step_m"     : gps_disp_step_3d,
        "gps_disp_cumul_m"       : gps_disp_cumul,
        "imu_disp_cumul_m"       : imu_dist,
        "fused_disp_step_m"      : fused_disp_step,
        "fused_disp_cumul_m"     : fused_disp_cumul,
        "fused_east_m"           : fused_e,
        "fused_north_m"          : fused_n,

        # ── 4. Altitude (m) ─────────────────────────────────────────────────
        "altitude_m"             : alt_ff,
        "delta_alt_gps_step_m"   : delta_alt_gps,
        "delta_alt_gps_cumul_m"  : np.cumsum(delta_alt_gps),
        "fused_alt_m"            : fused_alt_abs,
        "fused_dalt_step_m"      : fused_dalt_step,
        "fused_dalt_cumul_m"     : np.cumsum(fused_dalt_step),

        # ── GPS metadata ────────────────────────────────────────────────────
        "latitude"               : lat,
        "longitude"              : lon,
        "gps_bearing_deg"        : bearing_deg,
        "gps_h_accuracy_m"       : _col("accuracy",         fallback=np.nan),
        "gps_v_accuracy_m"       : v_acc_arr,
    })

    # Mode counts for summary
    n_frozen = int((update_mode == 0).sum())
    n_static = int((update_mode == 1).sum())
    n_moving = int((update_mode == 2).sum())

    results.attrs.update({
        "tau_static"       : tau_static,
        "tau_moving"       : tau_moving,
        "n_frozen"         : n_frozen,
        "n_static"         : n_static,
        "n_moving"         : n_moving,
        "grav_mag_mean"    : float(np.mean(g_mag_track)),
        "grav_mag_std"     : float(np.std(g_mag_track)),
        "grav_drift_std"   : float(np.std(grav_track, axis=0).mean()),
        "offset_used_deg"  : phi_deg,
        "offset_auto"      : offset_auto,
        "offset_n_cal"     : offset_n_cal,
        "algo"             : algo_used,
        "n_rows_no_gps"    : int((~valid_gps).sum()),
    })

    return results


# =========================================================================== #
# CLI entry point
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="GPS + IMU Kalman fusion for motorbike dynamics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gravity tracker time constants:
  --tau-static  (default 5 s):  how quickly to snap gravity when stopped.
                Shorter = faster convergence at stops.
  --tau-moving  (default 60 s): how slowly to track orientation drift while riding.
                Longer = safer against being biased by sustained acceleration.
                Shorter = faster to recover from large orientation changes (pocket).

Mounting offset auto-calibration:
  Omit --offset to auto-calibrate from GPS bearing (needs ≥10 rows at >1.5 m/s).
  Use --offset 0 to assume phone x-axis = bike forward, no calibration.
  Use --offset N to fix the angle to N degrees.
""")
    parser.add_argument("csv", nargs="?", default="sensagram.csv",
                        help="Input CSV from --mobility server mode.")
    parser.add_argument("--offset", type=float, default=None,
                        help="Phone mounting offset (degrees). "
                             "Omit to auto-calibrate.")
    parser.add_argument("--tau-static", type=float, default=GRAV_TAU_STATIC_DEF,
                        help=f"Gravity EMA time constant while stopped (s). "
                             f"Default: {GRAV_TAU_STATIC_DEF}")
    parser.add_argument("--tau-moving", type=float, default=GRAV_TAU_MOVING_DEF,
                        help=f"Gravity EMA time constant while moving (s). "
                             f"Default: {GRAV_TAU_MOVING_DEF}")
    parser.add_argument("--gps-r-pos", type=float, default=None,
                        help="GPS position noise std-dev (m). "
                             "Default: mean horizontal accuracy from data.")
    parser.add_argument("--gps-r-vel", type=float, default=0.3,
                        help="GPS velocity noise std-dev (m/s). Default: 0.3")
    parser.add_argument("--kf-q", type=float, default=0.05,
                        help="Kalman process noise. Default: 0.05")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df    = pd.read_csv(args.csv)
    r_pos = args.gps_r_pos or float(
        pd.to_numeric(df.get("accuracy", pd.Series([12.0])),
                      errors="coerce").mean())

    res = analyse(df,
                  mount_offset_deg = args.offset,
                  tau_static        = args.tau_static,
                  tau_moving        = args.tau_moving,
                  gps_r_pos         = r_pos,
                  gps_r_vel         = args.gps_r_vel,
                  kf_q              = args.kf_q)

    stem     = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = f"analytics_{stem}.csv"
    res.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[OK] Results written → {out_path}\n")

    dur = res["elapsed_s"].iloc[-1]
    n   = len(res)

    print("══ Algorithm ═══════════════════════════════════════════════")
    print(f"  {res.attrs['algo']}")
    if res.attrs["offset_auto"]:
        print(f"  Mounting offset    : {res.attrs['offset_used_deg']:+.2f}°  "
              f"(auto-cal from {res.attrs['offset_n_cal']} motion rows)")
    else:
        print(f"  Mounting offset    : {res.attrs['offset_used_deg']:+.2f}°  (explicit/default)")

    print("\n══ Adaptive gravity tracker ════════════════════════════════")
    print(f"  τ_static / τ_moving : {res.attrs['tau_static']:.0f} s / {res.attrs['tau_moving']:.0f} s")
    nf = res.attrs["n_frozen"]; ns = res.attrs["n_static"]; nm = res.attrs["n_moving"]
    print(f"  Mode counts         : FROZEN={nf}  STATIC={ns}  MOVING={nm}  (of {n} rows)")
    print(f"  |gravity| mean±std  : {res.attrs['grav_mag_mean']:.4f} ± "
          f"{res.attrs['grav_mag_std']:.4f} m/s²  "
          f"(ideal: {G_STD:.4f})")
    drift = res.attrs["grav_drift_std"]
    print(f"  Gravity vector drift: ±{drift:.4f} m/s²  "
          f"{'(stable — fixed mount)' if drift < 0.05 else '(variable — pocket/bag use detected)'}")

    print("\n══ GPS coverage ════════════════════════════════════════════")
    print(f"  Rows without GPS    : {res.attrs['n_rows_no_gps']} / {n}")
    print(f"  Mean h-accuracy     : {res['gps_h_accuracy_m'].mean():.2f} m")
    print(f"  Mean v-accuracy     : {res['gps_v_accuracy_m'].mean():.2f} m")

    print("\n══ Session overview ════════════════════════════════════════")
    print(f"  Rows                : {n}")
    print(f"  Duration            : {dur:.1f} s  ({dur / 60:.2f} min)")
    print(f"  Mean Δt             : {res['dt_s'].mean():.2f} s")

    print("\n══ 1a. IMU acceleration (m/s²) — body frame ════════════════")
    for col, label in [("accel_forward",  "Forward (along-track)"),
                        ("accel_lateral",  "Lateral"),
                        ("accel_vertical", "Vertical (screen-up only)"),
                        ("lin_accel_mag",  "Magnitude (always valid)")]:
        s = res[col]
        print(f"  {label:<30}  mean={s.mean():+.4f}  "
              f"std={s.std():.4f}  min={s.min():+.4f}  max={s.max():+.4f}")

    print("\n══ 1b. GPS-derived acceleration (m/s²) — orientation-free ═")
    for col, label in [("gps_accel_forward",  "Forward  d(speed)/dt"),
                        ("gps_accel_vertical", "Vertical d²(alt)/dt²"),
                        ("gps_accel_lateral",  "Lateral  v·dθ/dt"),
                        ("gps_accel_mag",      "Magnitude")]:
        s = res[col]
        print(f"  {label:<30}  mean={s.mean():+.4f}  "
              f"std={s.std():.4f}  min={s.min():+.4f}  max={s.max():+.4f}")

    print("\n══ 2. Velocity (m/s) ═══════════════════════════════════════")
    print(f"  {'Column':<30}  {'mean':>8}  {'max':>8}  {'min':>8}")
    print(f"  {'':-<30}  {'':-<8}  {'':-<8}  {'':-<8}")
    for col, label in [
        ("gps_speed_ms",       "GPS speed (horizontal)"),
        ("gps_speed_3d_ms",    "GPS speed (3-D)"),
        ("gps_vel_east_ms",    "GPS velocity East"),
        ("gps_vel_north_ms",   "GPS velocity North"),
        ("gps_vel_vertical_ms","GPS velocity Up (+climb)"),
        ("fused_speed_ms",     "Fused speed (horizontal)"),
        ("fused_speed_3d_ms",  "Fused speed (3-D)"),
        ("vertical_speed_ms",  "Fused vertical speed"),
    ]:
        s = res[col]
        print(f"  {label:<30}  {s.mean():>+8.3f}  {s.max():>8.3f}  {s.min():>8.3f}")

    print("\n══ 3. Displacement (m) ═════════════════════════════════════")
    print(f"  GPS horizontal      : {res['gps_disp_cumul_m'].iloc[-1]:.3f} m")
    print(f"  GPS 3-D             : {res['gps_disp_3d_step_m'].sum():.3f} m")
    print(f"  IMU dead-reckoning  : {res['imu_disp_cumul_m'].iloc[-1]:.3f} m"
          f"  (reference — drifts)")
    print(f"  Kalman fused        : {res['fused_disp_cumul_m'].iloc[-1]:.3f} m")

    print("\n══ 4. Altitude (m) ═════════════════════════════════════════")
    print(f"  Start altitude      : {res['altitude_m'].iloc[0]:.2f} m")
    print(f"  End altitude        : {res['altitude_m'].iloc[-1]:.2f} m")
    print(f"  GPS Δalt (cumul)    : {res['delta_alt_gps_cumul_m'].iloc[-1]:+.3f} m")
    print(f"  Fused Δalt (cumul)  : {res['fused_dalt_cumul_m'].iloc[-1]:+.3f} m")
    print(f"  GPS alt range       : "
          f"{res['altitude_m'].min():.2f}–{res['altitude_m'].max():.2f} m")


if __name__ == "__main__":
    main()
