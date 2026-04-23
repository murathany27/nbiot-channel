# @title

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec


# ==========================================
# First we set up the satellite and ground station parameters, and calculate the pass times for the next 7 days.
# ==========================================
ts = load.timescale()
t_now = ts.now()

# User selection (set at the top of the script)
# pass_number: 0 = first upcoming pass, 1 = second, 2 = third, ...
selected_pass_number = 1
# event_name options: "rise", "culmination", "set"
selected_event_name = "rise"
# Duration used when a natural end event is not available
post_event_window_minutes = 10
# Ground station coverage radius (km) for map visualization
coverage_radius_km = 700
# Number of direction arrows to draw along the selected track
direction_arrow_count = 6

# TLE values
tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

# satellite and ground station setup
satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
station_lat = 39.9208
station_lon = 32.8541
ground_station = wgs84.latlon(station_lat, station_lon)

# Satellite pass times for the next 7 days
t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
pass_times, pass_events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=10.0)

# Events: 0 = rise, 1 = culmination, 2 = set
rise_indices = [i for i, event in enumerate(pass_events) if event == 0]

print("\n[INFO] Upcoming passes (next 7 days):")
if len(rise_indices) == 0:
    print("[INFO] No passes returned by find_events().")
else:
    for pass_idx in range(min(10, len(rise_indices))):
        rise_idx = rise_indices[pass_idx]
        rise_time = pass_times[rise_idx]
        set_idx = rise_idx + 2

        if set_idx < len(pass_times) and pass_events[set_idx] == 2:
            set_time = pass_times[set_idx]
            duration_min = (set_time.utc_datetime() - rise_time.utc_datetime()).total_seconds() / 60.0
            print(
                f"  - pass_idx={pass_idx} | duration={duration_min:.2f} min | "
                f"rise={rise_time.utc_iso()} | set={set_time.utc_iso()}"
            )
        else:
            print(
                f"  - pass_idx={pass_idx} | duration=unknown | "
                f"rise={rise_time.utc_iso()} | set=N/A"
            )

if len(rise_indices) > 0:
    # Clamp selection to available passes to avoid index errors.
    selected_pass_number = max(0, min(selected_pass_number, len(rise_indices) - 1))
    base_idx = rise_indices[selected_pass_number]

    event_map = {"rise": 0, "culmination": 1, "set": 2}
    selected_event_name = selected_event_name.strip().lower()
    selected_event_code = event_map.get(selected_event_name)

    if selected_event_code is None:
        print(f"[WARN] Unknown event '{selected_event_name}'. Falling back to 'rise'.")
        selected_event_name = "rise"
        selected_event_code = 0

    selected_event_idx = base_idx + selected_event_code

    # Validate selected event index for the chosen pass.
    if selected_event_idx < len(pass_times) and pass_events[selected_event_idx] == selected_event_code:
        t_start = pass_times[selected_event_idx]
    else:
        print(f"[WARN] Event '{selected_event_name}' is not available for selected pass. Falling back to 'rise'.")
        selected_event_name = "rise"
        selected_event_code = 0
        selected_event_idx = base_idx
        t_start = pass_times[selected_event_idx]

    set_idx = base_idx + 2
    has_set_event = set_idx < len(pass_times) and pass_events[set_idx] == 2

    # For rise/culmination, prefer ending at set. For set, use a fixed window.
    if selected_event_code in (0, 1) and has_set_event and pass_times[set_idx].tt > t_start.tt:
        t_end = pass_times[set_idx]
    else:
        t_end = ts.from_datetime(t_start.utc_datetime() + timedelta(minutes=post_event_window_minutes))
else:
    print("[INFO] No satellite pass found. Using a default 10-minute interval.")
    t_start = t_now
    t_end = ts.from_datetime(t_now.utc_datetime() + timedelta(minutes=10))

if len(rise_indices) > 0:
    print(
        f"[INFO] Selected pass_idx={selected_pass_number} of {len(rise_indices)} upcoming passes, "
        f"event='{selected_event_name}'."
    )
    print(f"[INFO] Analysis window start (UTC): {t_start.utc_iso()}")
    print(f"[INFO] Analysis window end   (UTC): {t_end.utc_iso()}")
    print(
        f"[INFO] Analysis duration: "
        f"{(t_end.utc_datetime() - t_start.utc_datetime()).total_seconds() / 60:.2f} minutes"
    )

# ==========================================
# Calculation constants and parameters
# ==========================================
carrier_freq_mhz = 2000
carrier_freq_ghz = carrier_freq_mhz / 1000.0
carrier_freq_hz = carrier_freq_mhz * 1e6
c_km_s = 299792.458

# Link Buget parameters
tx_power_dbm = 30.0
tx_gain_dbi = 20.0
rx_gain_dbi = 0.0
rx_sensitivity_dbm = -125.0

# Results storage
time_minutes_list = []
elevation_list = []
distance_list = []
doppler_list = []
fspl_list = []
total_path_loss_list = []
k_factor_list = []
link_budget_list = []
sat_lat_list = []
sat_lon_list = []

total_seconds = int((t_end.utc_datetime() - t_start.utc_datetime()).total_seconds())
print(f"[INFO] Simulation step count (1-second resolution): {total_seconds}")

for s in range(total_seconds):
    current_t = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=s))

    # 1. Geometry: Elevation, Azimuth, Distance
    diff = satellite - ground_station
    alt, az, dist = diff.at(current_t).altaz()
    el_deg = alt.degrees
    d_km = dist.km

    # if elevation is below horizon, skip calculations (outage)
    if el_deg < 0:
        continue

    # Save results for plotting
    time_minutes_list.append(s / 60.0)
    elevation_list.append(el_deg)
    distance_list.append(d_km)

    # Sub-satellite point for map plotting (ground track)
    subpoint = wgs84.subpoint(satellite.at(current_t))
    sat_lat_list.append(subpoint.latitude.degrees)
    sat_lon_list.append(subpoint.longitude.degrees)

    # 2. Doppler Shift Calculation
    t_next = ts.from_datetime(current_t.utc_datetime() + timedelta(seconds=1))
    d_next = diff.at(t_next).distance().km
    v_rel = d_km - d_next # If approaching, distance decreases and relative velocity becomes positive
    doppler_hz = (v_rel / c_km_s) * carrier_freq_hz
    doppler_list.append(doppler_hz)

    # 3. Path Loss Calculation
    fspl = 0
    atm_loss = 0
    
    fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
    atm_loss = 0.5 / np.sin(np.radians(max(el_deg, 1.0)))

    total_loss = fspl + atm_loss
    fspl_list.append(fspl)
    total_path_loss_list.append(total_loss)

    # 4. Rician K Value Estimation (Simple model; actual K depends on atmospheric and other factors)
    k_db = min(15.0, 2.0 + ((el_deg - 10) / 80) * 13.0)
    k_factor_list.append(k_db)

    # 5. Link Budget Calculation
    # Received Power (dBm) = TxPower(dBm) + TxGain(dBi) + RxGain(dBi) - TotalPathLoss(dB)
    rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
    link_budget_list.append(rx_power)

print("\n[INFO] Computation finished.")
if len(time_minutes_list) == 0:
    print("[WARN] No above-horizon samples in selected interval; plots may be empty.")
else:
    print(f"[INFO] Valid sample count (elevation >= 0): {len(time_minutes_list)}")
    print(
        f"[INFO] Elevation range (deg): "
        f"min={np.min(elevation_list):.2f}, max={np.max(elevation_list):.2f}"
    )
    print(
        f"[INFO] Slant range (km): "
        f"min={np.min(distance_list):.2f}, max={np.max(distance_list):.2f}"
    )
    print(
        f"[INFO] Doppler (Hz): "
        f"min={np.min(doppler_list):.2f}, max={np.max(doppler_list):.2f}"
    )
    print(
        f"[INFO] Total path loss (dB): "
        f"min={np.min(total_path_loss_list):.2f}, max={np.max(total_path_loss_list):.2f}"
    )
    print(
        f"[INFO] Received power (dBm): "
        f"min={np.min(link_budget_list):.2f}, max={np.max(link_budget_list):.2f}, "
        f"mean={np.mean(link_budget_list):.2f}"
    )

    outage_count = int(np.sum(np.array(link_budget_list) < rx_sensitivity_dbm))
    print(
        f"[INFO] Outage samples (< {rx_sensitivity_dbm} dBm): "
        f"{outage_count}/{len(link_budget_list)}"
    )


# ==========================================
# PLOTTING
# ==========================================
time_arr = np.array(time_minutes_list)
el_arr = np.array(elevation_list)
dist_arr = np.array(distance_list)
dop_arr = np.array(doppler_list)
fspl_arr = np.array(fspl_list)
loss_arr = np.array(total_path_loss_list)
k_arr = np.array(k_factor_list)
lb_arr = np.array(link_budget_list)
sat_lat_arr = np.array(sat_lat_list)
sat_lon_arr = np.array(sat_lon_list)


def geodesic_circle(lat_deg, lon_deg, radius_km, num_points=180):
    """Approximate a small-circle boundary around a point on Earth."""
    earth_radius_km = 6371.0
    angular_distance = radius_km / earth_radius_km
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)

    bearings = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle_lats = []
    circle_lons = []

    for bearing in bearings:
        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(angular_distance)
            + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing)
        )
        lon2 = lon1 + np.arctan2(
            np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
            np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
        )
        lon2 = (lon2 + np.pi) % (2.0 * np.pi) - np.pi

        circle_lats.append(np.degrees(lat2))
        circle_lons.append(np.degrees(lon2))

    return np.array(circle_lats), np.array(circle_lons)

fig = plt.figure(figsize=(18, 18))
fig.canvas.manager.set_window_title('NTN Time-Series and Scenario Analysis')
gs = GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.28)

# -- 1. Graph: Path Loss vs Time --
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_arr, loss_arr, label='Total Path Loss (FSPL + Atmospheric)', color='red', linewidth=2)
ax1.plot(time_arr, fspl_arr, linestyle='--', label='FSPL Only', color='orange')
ax1.set_title('1. Path Loss Over Time')
ax1.set_xlabel('Time (Minutes)')
ax1.set_ylabel('Loss (dB)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# -- 2. Graph: Elevation and Distance vs Time --
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Time (Minutes)')
ax2.set_ylabel('Elevation (Degrees)', color='tab:blue')
ax2.plot(time_arr, el_arr, color='tab:blue', linewidth=2, label='Elevation')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.grid(True, linestyle='--', alpha=0.7)

ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Distance / Slant Range (km)', color='tab:green')
ax2_twin.plot(time_arr, dist_arr, linestyle='-.', color='tab:green', linewidth=2, label='Distance')
ax2_twin.tick_params(axis='y', labelcolor='tab:green')
plt.title('2. Elevation and Distance Over Time')

# -- 3. Graph: Doppler vs Time --
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_arr, dop_arr, color='purple', linewidth=2)
ax3.set_title('3. Doppler Shift Over Time')
ax3.set_xlabel('Time (Minutes)')
ax3.set_ylabel('Doppler Frequency (Hz)')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, linestyle='--', alpha=0.7)

# -- 4. Graph: K Factor vs Time --
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_arr, k_arr, color='brown', linewidth=2)
ax4.set_title('4. Rician K-Factor Over Time')
ax4.set_xlabel('Time (Minutes)')
ax4.set_ylabel('K Value (dB)')
ax4.grid(True, linestyle='--', alpha=0.7)

# -- 5. Graph: Link Budget (Received Power) vs Time --
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(time_arr, lb_arr, color='teal', linewidth=2, label='Instant Received Signal Power')
ax5.axhline(rx_sensitivity_dbm, color='red', linestyle='--', linewidth=2, label=f'RX Sensitivity ({rx_sensitivity_dbm} dBm)')

ax5.fill_between(time_arr, -150, rx_sensitivity_dbm, color='red', alpha=0.1, label='No Link (Outage)')
ax5.fill_between(time_arr, rx_sensitivity_dbm, np.max(lb_arr)+5, color='green', alpha=0.1, label='Link Available (Active)')

ax5.set_title('5. Link Budget and Connectivity Feasibility')
ax5.set_xlabel('Time (Minutes)')
ax5.set_ylabel('Signal Power (dBm)')
ax5.set_ylim(-150, np.max(lb_arr)+5)
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.legend(loc='lower right')

# -- 6. Graph: Satellite Ground Track Map --
ax6 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
ax6.set_title('6. Satellite Ground Track and Ground Station')
ax6.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', zorder=0)
ax6.add_feature(cfeature.LAND, facecolor='#f3efe0', zorder=0)
ax6.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
ax6.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7, zorder=1)
grid = ax6.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
grid.top_labels = False
grid.right_labels = False

if len(sat_lat_arr) > 0:
    ax6.plot(
        sat_lon_arr,
        sat_lat_arr,
        color='navy',
        linewidth=2,
        label='Satellite Ground Track',
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax6.scatter(
        sat_lon_arr[0],
        sat_lat_arr[0],
        color='green',
        s=60,
        zorder=4,
        label='Track Start',
        transform=ccrs.PlateCarree(),
    )
    ax6.scatter(
        sat_lon_arr[-1],
        sat_lat_arr[-1],
        color='black',
        s=60,
        zorder=4,
        label='Track End',
        transform=ccrs.PlateCarree(),
    )
else:
    print('[WARN] No satellite ground-track samples available for map plotting.')

# Auto-zoom to selected pass region and include ground station.
if len(sat_lat_arr) > 0:
    lat_min = min(np.min(sat_lat_arr), station_lat)
    lat_max = max(np.max(sat_lat_arr), station_lat)
    lon_min = min(station_lon-30, station_lon)
    lon_max = max(station_lon+30, station_lon)

    lat_margin = max(2.0, 0.15 * max(1e-6, lat_max - lat_min))
    lon_margin = max(2.0, 0.15 * max(1e-6, lon_max - lon_min))

    extent = [
        max(-180.0, lon_min - lon_margin),
        min(180.0, lon_max + lon_margin),
        max(-90.0, lat_min - lat_margin),
        min(90.0, lat_max + lat_margin),
    ]
    ax6.set_extent(extent, crs=ccrs.PlateCarree())
else:
    ax6.set_global()


# Add direction arrows along the track.
if len(sat_lat_arr) > 2:
    arrow_count = max(1, min(direction_arrow_count, len(sat_lat_arr) - 1))
    arrow_indices = np.linspace(0, len(sat_lat_arr) - 2, arrow_count, dtype=int)
    for idx in arrow_indices:
        ax6.annotate(
            '',
            xy=(sat_lon_arr[idx + 1], sat_lat_arr[idx + 1]),
            xytext=(sat_lon_arr[idx], sat_lat_arr[idx]),
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.2),
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

ax6.scatter(
    station_lon,
    station_lat,
    color='red',
    marker='^',
    s=90,
    zorder=5,
    label='Ground Station',
    transform=ccrs.PlateCarree(),
)
ax6.annotate(
    'Ground Station',
    (station_lon, station_lat),
    textcoords='offset points',
    xytext=(8, 8),
    fontsize=9,
    transform=ccrs.PlateCarree(),
)
#ax6.legend(loc='upper right')

plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
plt.show()