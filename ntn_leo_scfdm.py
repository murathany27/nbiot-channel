import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from skyfield.api import EarthSatellite, load, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec


def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))


def qpsk_modulate(bits):
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): 1 - 1j,
        (1, 1): -1 - 1j,
        (1, 0): -1 + 1j,
    }
    symbols = np.array([mapping[tuple(pair)] for pair in bits.reshape(-1, 2)], dtype=complex)
    return symbols / np.sqrt(2)


def qpsk_demodulate(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:, 0] = symbols.real < 0
    bits[:, 1] = symbols.imag < 0
    return bits.reshape(-1)


def awgn(sig, snr_db):
    snr_linear = 10 ** (snr_db / 10)
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
    ground_station = wgs84.latlon(39.9208, 32.8541)

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
        alt, _, dist = diff.at(current_t).altaz()
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
        doppler_list.append(doppler_hz)

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

    noise_floor_dbm = -174.0 + 10 * np.log10(180000)
    channel_results['snr_arr'] = channel_results['lb_arr'] - noise_floor_dbm

    return channel_results


def resample_channel_to_symbols(channel_results, symbol_count):
    valid_len = len(channel_results['time_arr'])
    idx = np.linspace(0, valid_len - 1, symbol_count).astype(int)

    channel_symbol = {
        'time_arr': channel_results['time_arr'][idx],
        'el_arr': channel_results['el_arr'][idx],
        'dist_arr': channel_results['dist_arr'][idx],
        'dop_arr': channel_results['dop_arr'][idx],
        'fspl_arr': channel_results['fspl_arr'][idx],
        'loss_arr': channel_results['loss_arr'][idx],
        'k_arr': channel_results['k_arr'][idx],
        'lb_arr': channel_results['lb_arr'][idx],
        'snr_arr': channel_results['snr_arr'][idx],
        'rx_sensitivity_dbm': channel_results['rx_sensitivity_dbm'],
    }
    return channel_symbol


def run_simulation():
    np.random.seed(7)

    delta_f = 15000
    M = 12
    N = 128
    CP = 9
    fs = N * delta_f
    block_len = N + CP

    num_slots = 100
    symbols_per_slot = 7
    pilot_symbol_in_slot = 3
    total_symbols = num_slots * symbols_per_slot
    total_data_symbols = num_slots * (symbols_per_slot - 1)

    snr_db_range = np.arange(0, 15, 1)

    pilot_symbol = np.ones(M, dtype=complex) * (1 + 1j) / np.sqrt(2)

    resource_map = np.full((total_symbols, M), 2, dtype=int)
    for s in range(total_symbols):
        if (s % symbols_per_slot) == pilot_symbol_in_slot:
            resource_map[s, :] = 1

    plt.figure(figsize=(10, 4))
    plt.imshow(resource_map.T, aspect='auto', origin='lower')
    plt.xlabel('SC-FDMA Symbol Index')
    plt.ylabel('Allocated Subcarrier Index')
    plt.title('SC-FDMA Resource Grid (1=Pilot, 2=Data)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    bits_tx = np.random.randint(0, 2, total_data_symbols * 2 * M, dtype=np.uint8)
    qpsk_data_symbols = qpsk_modulate(bits_tx).reshape(total_data_symbols, M)

    plt.figure(figsize=(6, 6))
    plt.plot(np.real(qpsk_data_symbols.flatten()), np.imag(qpsk_data_symbols.flatten()), 'bo', alpha=0.4)
    plt.xlabel('Inphase')
    plt.ylabel('Quadrature')
    plt.title('SC-FDMA TX QPSK Constellation')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    tx_blocks = []
    tx_shifted_grid = np.zeros((total_symbols, N), dtype=complex)

    data_counter = 0
    start_idx = (N // 2) - (M // 2)

    for s in range(total_symbols):
        symbol_in_slot = s % symbols_per_slot
        d = pilot_symbol if symbol_in_slot == pilot_symbol_in_slot else qpsk_data_symbols[data_counter]
        if symbol_in_slot != pilot_symbol_in_slot:
            data_counter += 1

        D = np.fft.fft(d, n=M)
        X_shift = np.zeros(N, dtype=complex)
        X_shift[start_idx : start_idx + M] = D
        tx_shifted_grid[s] = X_shift

        x = np.fft.ifft(np.fft.ifftshift(X_shift), n=N)
        tx_block = np.concatenate([x[-CP:], x])
        tx_blocks.append(tx_block)

    tx_signal = np.concatenate(tx_blocks)

    plt.figure(figsize=(10, 4))
    plt.stem(np.arange(N), np.abs(tx_shifted_grid[0]), basefmt=' ')
    plt.xlabel('Shifted Frequency Bin')
    plt.ylabel('Magnitude')
    plt.title('SC-FDMA 128-Bin Carrier Map for One Symbol')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    window = np.hanning(len(tx_signal))
    spectrum = np.fft.fftshift(np.fft.fft(tx_signal * window, 4096))

    plt.figure(figsize=(10, 4))
    plt.plot(20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum)) + 1e-12))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.title('SC-FDMA Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    channel_results = compute_channel_series()
    channel_symbol = resample_channel_to_symbols(channel_results, total_symbols)

    fig = plt.figure(figsize=(16, 12))
    fig.canvas.manager.set_window_title('SC-FDMA NTN Time Series and Scenario Analysis')
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
    ax5.fill_between(
        channel_results['time_arr'],
        channel_results['rx_sensitivity_dbm'],
        np.max(channel_results['lb_arr']) + 5,
        color='green',
        alpha=0.1,
        label='Link Active',
    )
    ax5.set_title('Link Budget and Feasibility')
    ax5.set_xlabel('Time (Minutes)')
    ax5.set_ylabel('Signal Power (dBm)')
    ax5.set_ylim(-150, np.max(channel_results['lb_arr']) + 5)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    simulated_ber = []
    constellation_example_done = False

    for snr_offset_db in snr_db_range:
        rx_blocks_after_channel = []

        for s in range(total_symbols):
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

            block_snr_db = channel_symbol['snr_arr'][s] + snr_offset_db
            noisy_block = awgn(ch_block, block_snr_db)
            rx_blocks_after_channel.append(noisy_block)

        rx_signal = np.concatenate(rx_blocks_after_channel)

        window_rx_before = np.hanning(len(rx_signal))
        spectrum_rx_before = np.fft.fftshift(np.fft.fft(rx_signal * window_rx_before, 4096))

        rx_data_symbols = np.zeros((total_data_symbols, M), dtype=complex)
        raw_const_before_eq = []
        eq_const = []
        rx_blocks_after_comp = []

        idx = 0
        data_counter = 0

        for slot in range(num_slots):
            slot_blocks = []
            for sym in range(symbols_per_slot):
                global_sym = slot * symbols_per_slot + sym
                rx_block = rx_signal[idx : idx + block_len]

                doppler_comp_hz = channel_symbol['dop_arr'][global_sym]
                t_rx = np.arange(len(rx_block)) / fs
                rx_block_comp = rx_block * np.exp(-1j * 2 * np.pi * doppler_comp_hz * t_rx)
                rx_blocks_after_comp.append(rx_block_comp)

                block_no_cp = rx_block_comp[CP:]
                X_hat = np.fft.fftshift(np.fft.fft(block_no_cp, n=N))
                D_hat = X_hat[start_idx : start_idx + M]
                slot_blocks.append(np.fft.ifft(D_hat, n=M))

                idx += block_len

            H_est = slot_blocks[pilot_symbol_in_slot] / pilot_symbol

            for sym in range(symbols_per_slot):
                if sym != pilot_symbol_in_slot:
                    raw = slot_blocks[sym]
                    eq = raw / H_est
                    raw_const_before_eq.extend(raw)
                    eq_const.extend(eq)
                    rx_data_symbols[data_counter] = eq
                    data_counter += 1

        rx_signal_after_comp = np.concatenate(rx_blocks_after_comp)
        window_rx_after = np.hanning(len(rx_signal_after_comp))
        spectrum_rx_after = np.fft.fftshift(np.fft.fft(rx_signal_after_comp * window_rx_after, 4096))

        rx_symbols_flat = rx_data_symbols.reshape(-1)
        bits_rx = qpsk_demodulate(rx_symbols_flat)
        ber = np.mean(bits_rx != bits_tx)
        simulated_ber.append(ber)

        print(f'SNR offset = {snr_offset_db:2d} dB | BER = {ber:.8f}')

        if snr_offset_db == 6 and not constellation_example_done:
            plt.figure(figsize=(6, 6))
            plt.plot(np.real(raw_const_before_eq), np.imag(raw_const_before_eq), 'bo', alpha=0.35, label='Before EQ')
            plt.xlabel('Inphase')
            plt.ylabel('Quadrature')
            plt.title('SC-FDMA Constellation Before Equalization')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6, 6))
            plt.plot(np.real(eq_const), np.imag(eq_const), 'go', alpha=0.35, label='After EQ')
            plt.xlabel('Inphase')
            plt.ylabel('Quadrature')
            plt.title('SC-FDMA Constellation After Equalization')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(
                20 * np.log10(np.abs(spectrum_rx_before) / np.max(np.abs(spectrum_rx_before)) + 1e-12),
                label='Before Compensation',
            )
            plt.plot(
                20 * np.log10(np.abs(spectrum_rx_after) / np.max(np.abs(spectrum_rx_after)) + 1e-12),
                label='After Compensation',
            )
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude (dB)')
            plt.title('SC-FDMA RX Spectrum Before / After Doppler Compensation')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

            constellation_example_done = True

    snr_linear = 10 ** (snr_db_range / 10)
    theoretical_ber = q_function(np.sqrt(snr_linear))

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_db_range, theoretical_ber, 'k--', label='Theoretical QPSK BER')
    plt.semilogy(snr_db_range, simulated_ber, 'bo-', label='SC-FDMA Simulated BER')
    plt.xlabel('SNR Offset (dB)')
    plt.ylabel('BER')
    plt.title('SC-FDMA BER vs SNR')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simulation()
