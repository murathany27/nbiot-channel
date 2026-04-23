import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
import random


def apply_leo_channel(tx_signal, fs, doppler_hz, delay_samples, snr_db, k_factor_db):
    rx = np.pad(tx_signal, (delay_samples, 200), mode='constant')
    t = np.arange(len(rx)) / fs
    rx = rx * np.exp(1j * 2 * np.pi * doppler_hz * t)

    k_lin = 10 ** (k_factor_db / 10)
    los = np.sqrt(k_lin / (k_lin + 1))
    nlos = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2 * (k_lin + 1))
    h_channel = los + nlos
    rx = rx * h_channel

    noise = (np.random.normal(0, 1, len(rx)) + 1j * np.random.normal(0, 1, len(rx))) / np.sqrt(2)
    sig_pwr = np.var(rx)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    rx = rx + noise * np.sqrt(noise_pwr)

    return rx, h_channel


def qpsk_mod(bits):
    mapping = {(0, 0): 1 + 1j, (0, 1): 1 - 1j, (1, 0): -1 + 1j, (1, 1): -1 - 1j}
    return np.array([mapping[tuple(b)] for b in bits.reshape(-1, 2)]) / np.sqrt(2)


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:, 0], bits[:, 1] = symbols.real < 0, symbols.imag < 0
    return bits.reshape(-1)


def run_simulation():
    # 0. SETUP
    ts = load.timescale()
    t_now = ts.now()
    satellite = EarthSatellite(
        '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997',
        '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000',
        'CONNECTA',
        ts,
    )
    ground_station = wgs84.latlon(39.9208, 32.8541)  # Ankara

    t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
    t_pass, events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=1.0)
    peak_indices = [i for i, event in enumerate(events) if event == 0]

    if peak_indices:
        selected_index = random.choice(peak_indices)
        t_peak = t_pass[selected_index]
    else:
        t_peak = t_now

    # 10-minute pass: peak - 5 min to peak + 5 min
    # Step = 10 seconds => 61 steps
    times_array = []
    for s in range(-300, 301, 10):
        times_array.append(ts.from_datetime(t_peak.utc_datetime() + timedelta(seconds=s)))

    # Metrics to save
    results = {
        'time_mins': [],
        'ber': [],
        'papr': [],
        'snr': [],
        'doppler': [],
        'elev': [],
        'k_factor': [],
        'all_rx_syms': [],  # equalized
        'pre_eq_syms': [],  # raw (pre-eq)
    }

    # Common PHY params
    delta_f = 15000
    M = 12
    N = 128
    CP = 9
    fs = N * delta_f
    delay_in_samples = 10
    num_slots = 100
    symbols_per_slot = 7
    pilot_idx = 3
    num_data_per_slot = symbols_per_slot - 1
    total_data_symbols = num_slots * num_data_per_slot
    pilot_symbol = np.ones(M, dtype=complex) * (1 + 1j) / np.sqrt(2)
    block_len = N + CP

    print(f"[INFO] Starting 10-minute simulation... ({len(times_array)} steps)")

    for idx_t, t_sim in enumerate(times_array):
        # 1. CHANNEL PARAMETERS
        elev_deg, _, dist_km = (satellite - ground_station).at(t_sim).altaz()
        elev_deg, dist_km = elev_deg.degrees, dist_km.km

        # If the satellite is below the horizon (elevation < 0), skip or assign worst BER
        if elev_deg < 0:
            results['time_mins'].append((idx_t * 10 - 300) / 60.0)
            results['ber'].append(0.5)
            results['papr'].append(0.0)
            results['snr'].append(-20)
            results['doppler'].append(0)
            results['elev'].append(elev_deg)
            results['k_factor'].append(0)
            print(
                f"[{idx_t + 1}/{len(times_array)}] t={results['time_mins'][-1]:.1f}m | "
                f"El={elev_deg:.1f}° | Below horizon (skipped)"
            )
            continue

        t_next = ts.from_datetime(t_sim.utc_datetime() + timedelta(seconds=1))
        _, _, dist_next = (satellite - ground_station).at(t_next).altaz()
        v_rel = dist_km - dist_next.km

        carrier_freq_mhz = 2000
        real_doppler = (v_rel / 299792.458) * (carrier_freq_mhz * 1e6)

        fspl = 32.44 + 20 * np.log10(dist_km) + 20 * np.log10(carrier_freq_mhz)
        atm_loss = 0.5 / np.sin(np.radians(max(elev_deg, 1.0)))
        noise_floor = -174.0 + 10 * np.log10(180000)

        sat_rx_gain_dbi = 30.0
        ue_tx_gain_dbi = 0.0
        tx_power_dbm = 23.0

        snr_db = (tx_power_dbm + ue_tx_gain_dbi + sat_rx_gain_dbi) - (fspl + atm_loss) - noise_floor
        k_factor_db = min(15.0, 2.0 + ((elev_deg - 10) / 80) * 13.0)

        # 2. TX SIGNAL GENERATION
        bits_tx = np.random.randint(0, 2, total_data_symbols * 2 * M, dtype=np.uint8)
        qpsk_data_syms = qpsk_mod(bits_tx).reshape(total_data_symbols, M)

        tx_blocks = []
        data_counter = 0
        for slot in range(num_slots):
            for sym in range(symbols_per_slot):
                d = pilot_symbol if sym == pilot_idx else qpsk_data_syms[data_counter]
                if sym != pilot_idx:
                    data_counter += 1

                # OFDMA: No DFT precoding
                X_shift = np.zeros(N, dtype=complex)
                start_idx = (N // 2) - (M // 2)
                X_shift[start_idx : start_idx + M] = d

                x = np.fft.ifft(np.fft.ifftshift(X_shift), n=N)
                tx_blocks.append(np.concatenate([x[-CP:], x]))

        tx_signal_base = np.concatenate(tx_blocks)

        # PAPR calculation
        papr_db = 10 * np.log10(np.max(np.abs(tx_signal_base) ** 2) / np.mean(np.abs(tx_signal_base) ** 2))

        est_doppler = real_doppler - 150.0
        t_tx = np.arange(len(tx_signal_base)) / fs
        tx_signal = tx_signal_base * np.exp(-1j * 2 * np.pi * est_doppler * t_tx)

        # 3. LEO CHANNEL
        rx_signal, h_channel = apply_leo_channel(tx_signal, fs, real_doppler, delay_in_samples, snr_db, k_factor_db)

        # 4. RX RECEIVER
        rx_data_syms = np.zeros((total_data_symbols, M), dtype=complex)
        idx = delay_in_samples

        correlation_sum = 0
        temp_idx = delay_in_samples
        for _ in range(num_slots * symbols_per_slot):
            cp_part = rx_signal[temp_idx : temp_idx + CP]
            data_part = rx_signal[temp_idx + N : temp_idx + N + CP]
            correlation_sum += np.sum(cp_part * np.conj(data_part))
            temp_idx += block_len

        estimated_doppler_coarse = -np.angle(correlation_sum) * fs / (2 * np.pi * N)
        t_rx = np.arange(len(rx_signal)) / fs
        rx_signal_coarse_fixed = rx_signal * np.exp(-1j * 2 * np.pi * estimated_doppler_coarse * t_rx)

        data_counter = 0
        pre_eq_list = []
        for slot in range(num_slots):
            slot_blocks = []
            for sym in range(symbols_per_slot):
                block = rx_signal_coarse_fixed[idx : idx + block_len][CP:]
                X_hat = np.fft.fftshift(np.fft.fft(block, n=N))
                start_idx = (N // 2) - (M // 2)
                D_hat = X_hat[start_idx : start_idx + M]
                # OFDMA: No IDFT
                slot_blocks.append(D_hat)
                idx += block_len

            H_est = slot_blocks[pilot_idx] / pilot_symbol

            for sym in range(symbols_per_slot):
                if sym != pilot_idx:
                    pre_eq_list.append(slot_blocks[sym])
                    rx_data_syms[data_counter] = slot_blocks[sym] / H_est
                    data_counter += 1

        rx_syms_flat = rx_data_syms.reshape(-1)
        pre_eq_flat = np.array(pre_eq_list).reshape(-1)

        # Save only 10% of symbols to prevent constellation memory explosion if many iterations
        subset_size = max(1, len(rx_syms_flat) // 10)
        results['all_rx_syms'].extend(rx_syms_flat[:subset_size])
        results['pre_eq_syms'].extend(pre_eq_flat[:subset_size])

        ber = np.mean(qpsk_demod(rx_syms_flat) != bits_tx)

        # Save results
        results['time_mins'].append((idx_t * 10 - 300) / 60.0)
        results['ber'].append(ber)
        results['papr'].append(papr_db)
        results['snr'].append(snr_db)
        results['doppler'].append(real_doppler)
        results['elev'].append(elev_deg)
        results['k_factor'].append(k_factor_db)

        # Save a sample of the signal for spectrum analysis (from the middle of the pass if possible)
        if idx_t == len(times_array) // 2:
            results['tx_signal_sample'] = tx_signal_base

        print(
            f"[{idx_t + 1}/{len(times_array)}] t={results['time_mins'][-1]:.1f}m | "
            f"El={elev_deg:.1f}° | SNR={snr_db:.1f}dB | PAPR={papr_db:.1f}dB | BER={ber:.2e}"
        )

    # ==========================================
    # 5. DASHBOARD VISUALIZATION
    # ==========================================
    fig, axs = plt.subplots(3, 3, figsize=(18, 14))
    fig.canvas.manager.set_window_title('NTN LEO 10-Minute Analysis')
    fig.suptitle(
        f"NTN LEO Satellite 10-Minute Simulation Analysis | Peak Time: {t_peak.utc_strftime('%H:%M UTC')}",
        fontsize=16,
        fontweight='bold',
    )

    t_arr = np.array(results['time_mins'])
    snr_arr = np.array(results['snr'])
    ber_arr = np.array(results['ber'])

    # --- 1. Elevation & SNR ---
    ax1 = axs[0, 0]
    ax1.plot(t_arr, results['elev'], 'b-', linewidth=2, label='Elevation (°)')
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Elevation (°)', color='b')
    ax1.grid(True, alpha=0.3)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_arr, results['snr'], 'g-', linewidth=2, label='SNR (dB)')
    ax1_twin.set_ylabel('SNR (dB)', color='g')
    ax1.set_title('Elevation and SNR Variation')

    # --- 2. Doppler ---
    ax2 = axs[0, 1]
    ax2.plot(t_arr, results['doppler'], 'purple', linewidth=2)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title('Doppler Curve')
    ax2.set_xlabel('Time (Minutes)')
    ax2.set_ylabel('Doppler Shift (Hz)')
    ax2.grid(True, alpha=0.5)

    # --- 3. PAPR ---
    ax3 = axs[0, 2]
    ax3.plot(t_arr, results['papr'], color='orange', linewidth=2)
    ax3.set_title('OFDMA PAPR Variation')
    ax3.set_xlabel('Time (Minutes)')
    ax3.set_ylabel('PAPR (dB)')
    ax3.grid(True, alpha=0.5)

    # --- 4. BER (Bit Error Rate) vs Time ---
    ax4 = axs[1, 0]
    ber_safe = ber_arr + 1e-6  # protect against 0 for log scale
    ax4.semilogy(t_arr, ber_safe, 'r-', linewidth=2)
    ax4.set_title('BER vs Time')
    ax4.set_xlabel('Time (Minutes)')
    ax4.set_ylabel('BER (Log Scale)')
    ax4.grid(True, which='both', ls='--', alpha=0.5)

    # --- 5. BER vs SNR ---
    ax5 = axs[1, 1]
    # Filter out skipped steps (below horizon) if necessary
    valid_mask = np.array(results['elev']) >= 0
    if np.any(valid_mask):
        ax5.semilogy(snr_arr[valid_mask], ber_safe[valid_mask], 'ro', markersize=4)
        ax5.set_title('BER vs SNR')
        ax5.set_xlabel('SNR (dB)')
        ax5.set_ylabel('BER')
        ax5.grid(True, which='both', ls='--', alpha=0.5)
    else:
        ax5.set_title('BER vs SNR (No Data)')

    # --- 6. Rician K-Factor ---
    ax6 = axs[1, 2]
    ax6.plot(t_arr, results['k_factor'], 'm-', linewidth=2)
    ax6.set_title('Rician K-Factor')
    ax6.set_xlabel('Time (Minutes)')
    ax6.set_ylabel('K Value (dB)')
    ax6.grid(True, alpha=0.5)

    # --- 7. Spectrum (PSD) ---
    ax7 = axs[2, 0]
    if 'tx_signal_sample' in results:
        sig = results['tx_signal_sample']
        # Compute PSD
        n_fft = 2048
        psd = np.abs(np.fft.fftshift(np.fft.fft(sig[:n_fft], n=n_fft))) ** 2
        psd_db = 10 * np.log10(psd / np.max(psd) + 1e-12)
        freqs = np.linspace(-fs / 2, fs / 2, n_fft)
        ax7.plot(freqs / 1e3, psd_db, color='cyan')
        ax7.set_title('Transmitted Signal Spectrum (PSD)')
        ax7.set_xlabel('Frequency (kHz)')
        ax7.set_ylabel('Magnitude (dB)')
        ax7.set_ylim(-60, 5)
        ax7.grid(True, alpha=0.3)
    else:
        ax7.set_title('Spectrum (No Sample)')

    # --- 8. Constellation ---
    ax8 = axs[2, 1]
    pre_syms = np.array(results['pre_eq_syms'])
    eq_syms = np.array(results['all_rx_syms'])

    ax8.scatter(
        pre_syms.real,
        pre_syms.imag,
        color='orange',
        alpha=0.3,
        s=5,
        label='Pre-Equalization',
    )
    ax8.scatter(
        eq_syms.real,
        eq_syms.imag,
        color='green',
        alpha=0.3,
        s=5,
        label='Post-Equalization',
    )
    ax8.set_title('Receiver Constellation')
    ax8.set_xlim(-2, 2)
    ax8.set_ylim(-2, 2)
    ax8.grid(True, linestyle='--')
    ax8.legend(loc='upper right', fontsize='x-small')

    # --- 9. Summary / Info ---
    ax9 = axs[2, 2]
    ax9.axis('off')
    info_text = (
        f"Modulation: QPSK\n"
        f"Subcarriers (M): {M}\n"
        f"FFT Size (N): {N}\n"
        f"CP Length: {CP}\n"
        f"Subcarrier Spacing: {delta_f/1e3:.1f} kHz\n"
        f"Sampling Rate: {fs/1e6:.2f} MHz\n\n"
        f"Avg PAPR: {np.mean(results['papr']):.1f} dB\n"
        f"Peak Elev: {np.max(results['elev']):.1f}°\n"
        f"Peak SNR: {np.max(results['snr']):.1f} dB"
    )
    ax9.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == '__main__':
    run_simulation()