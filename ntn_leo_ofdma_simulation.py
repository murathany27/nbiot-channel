import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec

ITUR_AVAILABLE = False

def generate_bits(length):
    return np.random.binomial(n=1, p=0.5, size=length)

def ensure_even_bits(bits):
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    return bits

def symbol_gen(bin_data):
    dibit = np.array([bin_data[i:i+2] for i in range(0, len(bin_data), 2)])
    return dibit

def map_to_constellation(symbols, A):
    constellation_map = {
        (0, 0): (-A, -A),
        (0, 1): (-A,  A),
        (1, 1): ( A,  A),
        (1, 0): ( A, -A)
    }
    return np.array([constellation_map[tuple(dibit)] for dibit in symbols])

def qpsk_modulate(bits, A):
    bits = ensure_even_bits(bits)
    dibits = symbol_gen(bits)
    constellation_points = map_to_constellation(dibits, A)
    I = constellation_points[:, 0]
    Q = constellation_points[:, 1]
    return I + 1j * Q

def add_cp(ofdm_t, cp_length):
    cp = ofdm_t[-cp_length:]
    return np.hstack([cp, ofdm_t])

def remove_cp(signal, cp_length, sub):
    return signal[cp_length:cp_length + sub]

def nearest_neighbor_qpsk(symbol_rx, A):
    constellation_map = {
        (-A, -A): [0, 0],
        (-A,  A): [0, 1],
        ( A, -A): [1, 0],
        ( A,  A): [1, 1]
    }
    constellation_points_rx = np.array(list(constellation_map.keys()))
    bit_values = np.array(list(constellation_map.values()))
    demod_bits = []

    for sym in symbol_rx:
        i_rec = np.real(sym)
        q_rec = np.imag(sym)
        diff = constellation_points_rx - np.array([i_rec, q_rec])
        squared_diff = diff ** 2
        sum_of_squares = np.sum(squared_diff, axis=1)
        distances = np.sqrt(sum_of_squares)
        nearest = np.argmin(distances)
        demod_bits.extend(bit_values[nearest])

    return np.array(demod_bits)

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def awgn(sig, snr_db, cp_len, fft_size, active_subcarriers):
    correction = 10 * np.log10(active_subcarriers / fft_size)
    actual_snr = snr_db + correction
    snr_linear = 10 ** (actual_snr / 10)
    signal_power = np.mean(np.abs(sig) ** 2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    noise = sigma * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
    return sig + noise

def compute_channel_series():
    ts = load.timescale()
    t_now = ts.now()

    tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
    tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

    satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
    station_lat = 39.9208
    station_lon = 32.8541
    ground_station = wgs84.latlon(station_lat, station_lon)

    t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
    pass_times, pass_events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=10.0)

    rise_indices = [i for i, event in enumerate(pass_events) if event == 0]

    if len(rise_indices) > 0:
        idx = rise_indices[0]
        t_start = pass_times[idx]
        if idx + 2 < len(pass_times) and pass_events[idx + 2] == 2:
            t_end = pass_times[idx + 2]
        else:
            t_end = ts.from_datetime(t_start.utc_datetime() + timedelta(minutes=15))
    else:
        t_start = t_now
        t_end = ts.from_datetime(t_now.utc_datetime() + timedelta(minutes=10))

    carrier_freq_mhz = 2000
    carrier_freq_ghz = carrier_freq_mhz / 1000.0
    carrier_freq_hz = carrier_freq_mhz * 1e6
    c_km_s = 299792.458

    tx_power_dbm = 30.0
    tx_gain_dbi = 20.0
    rx_gain_dbi = 0.0
    rx_sensitivity_dbm = -125.0

    time_minutes_list = []
    elevation_list = []
    distance_list = []
    doppler_list = []
    fspl_list = []
    total_path_loss_list = []
    k_factor_list = []
    link_budget_list = []

    total_seconds = int((t_end.utc_datetime() - t_start.utc_datetime()).total_seconds()) + 1

    for s in range(total_seconds):
        current_t = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=s))
        diff = satellite - ground_station
        alt, az, dist = diff.at(current_t).altaz()
        el_deg = alt.degrees
        d_km = dist.km

        if el_deg < 0:
            continue

        time_minutes_list.append(s / 60.0)
        elevation_list.append(el_deg)
        distance_list.append(d_km)

        t_next = ts.from_datetime(current_t.utc_datetime() + timedelta(seconds=1))
        d_next = diff.at(t_next).distance().km
        v_rel = d_km - d_next
        doppler_hz = (v_rel / c_km_s) * carrier_freq_hz
        print(f"Time: {s/60:.2f} min | Elevation: {el_deg:.2f} deg | Distance: {d_km:.2f} km | Doppler: {doppler_hz:.2f} Hz")
        doppler_list.append(doppler_hz)

        fspl = 0
        atm_loss = 0
        if ITUR_AVAILABLE:
            try:
                val = itu525.free_space_loss(d_km, carrier_freq_ghz)
                fspl = float(val.value) if hasattr(val, 'value') else float(val)
            except Exception:
                fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)

            try:
                val_atm = itu676(carrier_freq_ghz, el_deg, 7.5, 1013.25, 15.0)
                atm_loss = float(val_atm.value) if hasattr(val_atm, 'value') else float(val_atm)
            except Exception:
                atm_loss = 0.5 / np.sin(np.radians(max(el_deg, 1.0)))
        else:
            fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
            atm_loss = 0.5 / np.sin(np.radians(max(el_deg, 1.0)))

        total_loss = fspl + atm_loss
        fspl_list.append(fspl)
        total_path_loss_list.append(total_loss)

        k_db = min(15.0, 2.0 + ((el_deg - 10) / 80) * 13.0)
        k_factor_list.append(k_db)

        rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
        link_budget_list.append(rx_power)

    channel_results = {
        'time_arr': np.array(time_minutes_list),
        'el_arr': np.array(elevation_list),
        'dist_arr': np.array(distance_list),
        'dop_arr': np.array(doppler_list),
        'fspl_arr': np.array(fspl_list),
        'loss_arr': np.array(total_path_loss_list),
        'k_arr': np.array(k_factor_list),
        'lb_arr': np.array(link_budget_list),
        'rx_sensitivity_dbm': rx_sensitivity_dbm,
    }

    return channel_results

def resample_channel_to_symbols(channel_results, num_ofdm_symbols):
    valid_len = len(channel_results['time_arr'])
    idx = np.linspace(0, valid_len - 1, num_ofdm_symbols).astype(int)

    channel_symbol = {
        'time_arr': channel_results['time_arr'][idx],
        'el_arr': channel_results['el_arr'][idx],
        'dist_arr': channel_results['dist_arr'][idx],
        'dop_arr': channel_results['dop_arr'][idx],
        'fspl_arr': channel_results['fspl_arr'][idx],
        'loss_arr': channel_results['loss_arr'][idx],
        'k_arr': channel_results['k_arr'][idx],
        'lb_arr': channel_results['lb_arr'][idx],
        'rx_sensitivity_dbm': channel_results['rx_sensitivity_dbm'],
    }
    return channel_symbol

def apply_block_channel(ofdm_t_cp, sub, cp_length, doppler_hz, loss_db, k_factor_db, fft_size):
    block_len = sub + cp_length
    atten_ref_db = np.min(loss_db)
    return ofdm_t_cp

sub = 64
cp_length = 16
system_active_subcarriers = 32
nb_iot_subcarriers = 12
num_ofdm_symbols = 1000
snr_db_range = np.arange(0, 15, 1)

E_bit = 0.5
A = np.sqrt(2 * E_bit) / np.sqrt(2)
pilot_value = A + 1j * A

all_carriers = np.arange(sub)
center = sub // 2
half_system = system_active_subcarriers // 2
system_active_shifted = np.r_[np.arange(center - half_system, center), np.arange(center + 1, center + 1 + half_system)]

half_nb = nb_iot_subcarriers // 2
nb_iot_shifted = np.r_[np.arange(center - half_nb, center), np.arange(center + 1, center + 1 + half_nb)]

nb_idx = np.arange(nb_iot_subcarriers)

def get_nrs_positions_in_symbol(symbol_in_subframe, n_cell_id):
    v_shift = n_cell_id % 6
    if symbol_in_subframe % 2 == 0:
        base = 0
    else:
        base = 1
    pos = np.array([(base + v_shift) % 6, ((base + v_shift) % 6) + 6])
    return pos

def get_data_positions_in_symbol(symbol_in_subframe, n_cell_id):
    nrs_local = get_nrs_positions_in_symbol(symbol_in_subframe, n_cell_id)
    all_local = np.arange(nb_iot_subcarriers)
    data_local = np.setdiff1d(all_local, nrs_local)
    half_data = len(data_local) // 2
    ue1_local_sym = data_local[:half_data]
    ue2_local_sym = data_local[half_data:]
    return nrs_local, ue1_local_sym, ue2_local_sym

print("FFT size:", sub)
print("CP length:", cp_length)
print("Active subcarriers:", system_active_subcarriers)
print("NB-IoT narrowband subcarriers:", nb_iot_subcarriers)
print("DC bin:", center)
print("Active shifted bins:", system_active_shifted)
print("NB-IoT shifted bins:", nb_iot_shifted)

resource_map = np.zeros((num_ofdm_symbols, nb_iot_subcarriers), dtype=int)

for s in range(num_ofdm_symbols):
    symbol_in_subframe = s % 14
    nrs_local, ue1_local_sym, ue2_local_sym = get_data_positions_in_symbol(symbol_in_subframe, n_cell_id=0)
    resource_map[s, ue1_local_sym] = 2
    resource_map[s, ue2_local_sym] = 3
    resource_map[s, nrs_local] = 1

plt.figure(figsize=(10, 4))
plt.imshow(resource_map.T, aspect='auto', origin='lower')
plt.yticks(np.arange(nb_iot_subcarriers))
plt.xlabel("OFDM Symbol Index")
plt.ylabel("NB-IoT Local Subcarrier Index")
plt.title("NB-IoT Downlink Resource Allocation Grid\n0=Empty, 1=Pilot, 2=UE1, 3=UE2")
plt.colorbar()
plt.tight_layout()
plt.show()

bits_per_symbol_ue1 = 2 * 5
bits_per_symbol_ue2 = 2 * 5

bits_tx_ue1 = generate_bits(num_ofdm_symbols * bits_per_symbol_ue1)
bits_tx_ue2 = generate_bits(num_ofdm_symbols * bits_per_symbol_ue2)

symbols_per_user_per_symbol = 5
symbols_tx_ue1 = qpsk_modulate(bits_tx_ue1, A).reshape(num_ofdm_symbols, symbols_per_user_per_symbol)
symbols_tx_ue2 = qpsk_modulate(bits_tx_ue2, A).reshape(num_ofdm_symbols, symbols_per_user_per_symbol)

plt.figure(figsize=(6, 6))
plt.plot(np.real(symbols_tx_ue1.flatten()), np.imag(symbols_tx_ue1.flatten()), 'bo', alpha=0.5, label='UE1 TX')
plt.plot(np.real(symbols_tx_ue2.flatten()), np.imag(symbols_tx_ue2.flatten()), 'ro', alpha=0.5, label='UE2 TX')
plt.xlabel("Inphase")
plt.ylabel("Quadrature")
plt.title("TX QPSK Constellations")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

tx_blocks = []
full_grid_shifted = np.zeros((num_ofdm_symbols, sub), dtype=complex)
nb_grid = np.zeros((num_ofdm_symbols, nb_iot_subcarriers), dtype=complex)

for s in range(num_ofdm_symbols):
    symbol_shifted = np.zeros(sub, dtype=complex)

    symbol_in_subframe = s % 14
    nrs_local, ue1_local_sym, ue2_local_sym = get_data_positions_in_symbol(symbol_in_subframe, n_cell_id=0)

    nrs_shifted = nb_iot_shifted[nrs_local]
    ue1_shifted_sym = nb_iot_shifted[ue1_local_sym]
    ue2_shifted_sym = nb_iot_shifted[ue2_local_sym]

    symbol_shifted[nrs_shifted] = pilot_value
    symbol_shifted[ue1_shifted_sym] = symbols_tx_ue1[s]
    symbol_shifted[ue2_shifted_sym] = symbols_tx_ue2[s]

    full_grid_shifted[s] = symbol_shifted
    nb_grid[s, nrs_local] = pilot_value
    nb_grid[s, ue1_local_sym] = symbols_tx_ue1[s]
    nb_grid[s, ue2_local_sym] = symbols_tx_ue2[s]

    ofdm_input = np.fft.ifftshift(symbol_shifted)
    ofdm_t = np.fft.ifft(ofdm_input) * np.sqrt(sub)
    ofdm_t_cp = add_cp(ofdm_t, cp_length)
    tx_blocks.append(ofdm_t_cp)

tx_signal = np.concatenate(tx_blocks)

plt.figure(figsize=(10, 4))
plt.stem(np.arange(sub), np.abs(full_grid_shifted[0]), basefmt=" ")
plt.xlabel("Shifted Frequency Bin")
plt.ylabel("Magnitude")
plt.title("Full 64-Bin Carrier Map for One OFDM Symbol")
plt.grid(True)
plt.tight_layout()
plt.show()

window = np.hanning(len(tx_signal))
spectrum = np.fft.fftshift(np.fft.fft(tx_signal * window, 4096))

plt.figure(figsize=(10, 4))
plt.plot(20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum)) + 1e-12))
plt.xlabel("Frequency")
plt.ylabel("Magnitude (dB)")
plt.title("OFDMA Spectrum with Guard Bands and NB-IoT Narrowband")
plt.grid(True)
plt.tight_layout()
plt.show()

channel_results = compute_channel_series()
channel_symbol = resample_channel_to_symbols(channel_results, num_ofdm_symbols)

fig = plt.figure(figsize=(16, 12))
fig.canvas.manager.set_window_title('NTN Time Series and Scenario Analysis')
gs = GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(channel_results['time_arr'], channel_results['loss_arr'], label='Total Path Loss', color='red', linewidth=2)
ax1.plot(channel_results['time_arr'], channel_results['fspl_arr'], linestyle='--', label='FSPL Only', color='orange')
ax1.set_title('Path Loss vs Time')
ax1.set_xlabel('Time (Minutes)')
ax1.set_ylabel('Loss (dB)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Time (Minutes)')
ax2.set_ylabel('Elevation (deg)', color='tab:blue')
ax2.plot(channel_results['time_arr'], channel_results['el_arr'], color='tab:blue', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Distance (km)', color='tab:green')
ax2_twin.plot(channel_results['time_arr'], channel_results['dist_arr'], linestyle='-.', color='tab:green', linewidth=2)
ax2_twin.tick_params(axis='y', labelcolor='tab:green')
ax2.set_title('Elevation and Slant Range vs Time')

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(channel_results['time_arr'], channel_results['dop_arr'], color='purple', linewidth=2)
ax3.set_title('Doppler vs Time')
ax3.set_xlabel('Time (Minutes)')
ax3.set_ylabel('Doppler (Hz)')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, linestyle='--', alpha=0.7)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(channel_results['time_arr'], channel_results['k_arr'], color='brown', linewidth=2)
ax4.set_title('Rician K-Factor vs Time')
ax4.set_xlabel('Time (Minutes)')
ax4.set_ylabel('K (dB)')
ax4.grid(True, linestyle='--', alpha=0.7)

ax5 = fig.add_subplot(gs[2, :])
ax5.plot(channel_results['time_arr'], channel_results['lb_arr'], color='teal', linewidth=2, label='Received Power')
ax5.axhline(channel_results['rx_sensitivity_dbm'], color='red', linestyle='--', linewidth=2, label=f"RX Sensitivity ({channel_results['rx_sensitivity_dbm']} dBm)")
ax5.fill_between(channel_results['time_arr'], -150, channel_results['rx_sensitivity_dbm'], color='red', alpha=0.1, label='Outage')
ax5.fill_between(channel_results['time_arr'], channel_results['rx_sensitivity_dbm'], np.max(channel_results['lb_arr']) + 5, color='green', alpha=0.1, label='Link Active')
ax5.set_title('Link Budget and Feasibility')
ax5.set_xlabel('Time (Minutes)')
ax5.set_ylabel('Signal Power (dBm)')
ax5.set_ylim(-150, np.max(channel_results['lb_arr']) + 5)
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.legend(loc='lower right')

plt.tight_layout()
plt.show()

simulated_ber_ue1 = []
simulated_ber_ue2 = []
constellation_example_done = False

fs = sub * 15000
block_len = sub + cp_length

loss_ref_db = np.min(channel_symbol['loss_arr'])
attenuation_db = channel_symbol['loss_arr'] - loss_ref_db
attenuation_lin = 10 ** (-attenuation_db / 20)

for snr_db in snr_db_range:
    rx_blocks_after_channel = []
    h_channel_series = []

    for s in range(num_ofdm_symbols):
        tx_block = tx_blocks[s]
        doppler_hz = channel_symbol['dop_arr'][s]
        k_factor_db = channel_symbol['k_arr'][s]

        t_block = np.arange(len(tx_block)) / fs

        k_lin = 10 ** (k_factor_db / 10)
        los = np.sqrt(k_lin / (k_lin + 1))
        nlos = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2 * (k_lin + 1))
        h_channel = los + nlos

        ch_block = tx_block * np.exp(1j * 2 * np.pi * doppler_hz * t_block)
        ch_block = ch_block * h_channel
        # DÜZELTME: BER testi için path loss iptal edildi
        # ch_block = ch_block * attenuation_lin[s] 

        rx_blocks_after_channel.append(ch_block)
        h_channel_series.append(h_channel)

    ch_signal = np.concatenate(rx_blocks_after_channel)
    rx_signal = awgn(ch_signal, snr_db, cp_length, sub, nb_iot_subcarriers)

    window_rx_before = np.hanning(len(rx_signal))
    spectrum_rx_before = np.fft.fftshift(np.fft.fft(rx_signal * window_rx_before, 4096))

    symbols_per_user_per_symbol = 5
    rx_symbols_ue1 = np.zeros((num_ofdm_symbols, symbols_per_user_per_symbol), dtype=complex)
    rx_symbols_ue2 = np.zeros((num_ofdm_symbols, symbols_per_user_per_symbol), dtype=complex)

    raw_const_before_eq_ue1 = []
    raw_const_before_eq_ue2 = []
    eq_const_ue1 = []
    eq_const_ue2 = []
    rx_blocks_after_comp = []

    idx = 0
    
    for s in range(num_ofdm_symbols):
        rx_block = rx_signal[idx:idx + block_len]

        # CP tabanli kaba CFO kestirimi bu senaryodaki buyuk Doppler icin aliasing uretiyor.
        # Kanal serisinden sembol-bazli Doppler kullanarak faz donmesini geri aliyoruz.
        doppler_comp_hz = channel_symbol['dop_arr'][s]
        t_rx = np.arange(len(rx_block)) / fs
        rx_block_comp = rx_block * np.exp(-1j * 2 * np.pi * doppler_comp_hz * t_rx)
        rx_blocks_after_comp.append(rx_block_comp)

        rx_no_cp = remove_cp(rx_block_comp, cp_length, sub)
        rx_fft = np.fft.fft(rx_no_cp) / np.sqrt(sub)
        rx_shifted = np.fft.fftshift(rx_fft)

        symbol_in_subframe = s % 14
        nrs_local, ue1_local_sym, ue2_local_sym = get_data_positions_in_symbol(symbol_in_subframe, n_cell_id=0)

        nrs_shifted = nb_iot_shifted[nrs_local]
        ue1_shifted_sym = nb_iot_shifted[ue1_local_sym]
        ue2_shifted_sym = nb_iot_shifted[ue2_local_sym]

        rx_pilots = rx_shifted[nrs_shifted]
        h_pilots = rx_pilots / pilot_value

        h_est_real = np.interp(nb_idx, nrs_local, np.real(h_pilots))
        h_est_imag = np.interp(nb_idx, nrs_local, np.imag(h_pilots))
        h_est_nb = h_est_real + 1j * h_est_imag

        h_est_ue1 = h_est_nb[ue1_local_sym]
        h_est_ue2 = h_est_nb[ue2_local_sym]

        raw_ue1 = rx_shifted[ue1_shifted_sym]
        raw_ue2 = rx_shifted[ue2_shifted_sym]

        eq_ue1 = raw_ue1 / h_est_ue1
        eq_ue2 = raw_ue2 / h_est_ue2

        rx_symbols_ue1[s] = eq_ue1
        rx_symbols_ue2[s] = eq_ue2

        raw_const_before_eq_ue1.extend(raw_ue1)
        raw_const_before_eq_ue2.extend(raw_ue2)
        eq_const_ue1.extend(eq_ue1)
        eq_const_ue2.extend(eq_ue2)

        idx += block_len

    rx_signal_after_comp = np.concatenate(rx_blocks_after_comp)
    window_rx_after = np.hanning(len(rx_signal_after_comp))
    spectrum_rx_after = np.fft.fftshift(np.fft.fft(rx_signal_after_comp * window_rx_after, 4096))

    rx_symbols_flat_ue1 = rx_symbols_ue1.flatten()
    rx_symbols_flat_ue2 = rx_symbols_ue2.flatten()

    bits_rx_ue1 = nearest_neighbor_qpsk(rx_symbols_flat_ue1, A)[:len(bits_tx_ue1)]
    bits_rx_ue2 = nearest_neighbor_qpsk(rx_symbols_flat_ue2, A)[:len(bits_tx_ue2)]

    ber_ue1 = np.mean(bits_rx_ue1 != bits_tx_ue1)
    ber_ue2 = np.mean(bits_rx_ue2 != bits_tx_ue2)

    simulated_ber_ue1.append(ber_ue1)
    simulated_ber_ue2.append(ber_ue2)

    print(f"SNR = {snr_db:2d} dB | UE1 BER = {ber_ue1:.8f} | UE2 BER = {ber_ue2:.8f}")

    if snr_db == 14 and not constellation_example_done:
        plt.figure(figsize=(6, 6))
        plt.plot(np.real(raw_const_before_eq_ue1), np.imag(raw_const_before_eq_ue1), 'bo', alpha=0.35, label='UE1 Before EQ')
        plt.plot(np.real(raw_const_before_eq_ue2), np.imag(raw_const_before_eq_ue2), 'ro', alpha=0.35, label='UE2 Before EQ')
        plt.xlabel("Inphase")
        plt.ylabel("Quadrature")
        plt.title("Received Constellation Before Equalization")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.plot(np.real(eq_const_ue1), np.imag(eq_const_ue1), 'bo', alpha=0.35, label='UE1 After EQ')
        plt.plot(np.real(eq_const_ue2), np.imag(eq_const_ue2), 'ro', alpha=0.35, label='UE2 After EQ')
        plt.xlabel("Inphase")
        plt.ylabel("Quadrature")
        plt.title("Estimated / Equalized Constellation")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(nb_idx, np.abs(h_est_nb), 'o-', label='|H_est| over NB-IoT band')
        plt.plot(nrs_local, np.abs(h_pilots), 'rs', label='Pilot estimates')
        plt.xlabel("NB-IoT Local Subcarrier Index")
        plt.ylabel("Magnitude")
        plt.title("Interpolated Channel Estimate at 6 dB")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(20 * np.log10(np.abs(spectrum_rx_before) / np.max(np.abs(spectrum_rx_before)) + 1e-12), label="Before Compensation")
        plt.plot(20 * np.log10(np.abs(spectrum_rx_after) / np.max(np.abs(spectrum_rx_after)) + 1e-12), label="After Compensation")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude (dB)")
        plt.title("RX Spectrum Before / After Doppler Compensation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        constellation_example_done = True

snr_linear = 10 ** (snr_db_range / 10)
# DÜZELTME: QPSK teorik BER formülü düzeltildi
theoretical_ber = q_function(np.sqrt(snr_linear)) 

plt.figure(figsize=(8, 5))
plt.semilogy(snr_db_range, theoretical_ber, 'k--', label='Theoretical QPSK BER')
plt.semilogy(snr_db_range, simulated_ber_ue1, 'bo-', label='UE1 Simulated BER')
plt.semilogy(snr_db_range, simulated_ber_ue2, 'ro-', label='UE2 Simulated BER')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("NB-IoT-Like Downlink OFDMA BER vs SNR")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()