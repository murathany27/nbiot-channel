import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
from skyfield.api import EarthSatellite, load, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec

# GRAFİK MODU: True ise sadece Constellation ve BER grafiklerini gösterir.
# False ise tüm teknik analiz grafiklerini (Spektrum, Kanal vb.) açar.
ONLY_CORE_GRAPHS = True
MODULATION_SCHEME = 'BPSK'  # Seçenekler: 'BPSK', 'QPSK', '16QAM'

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

    # Kodun başına eklenecekler
def simple_interleaver(data):
    # Veriyi kareye yakın bir matrise dizip transpozesini alır (Burst hataları dağıtır)
    n = len(data)
    cols = 8 # Sabit sütun sayısı
    rows = n // cols
    if rows == 0: return data
    reshaped = data[:rows*cols].reshape(rows, cols)
    return reshaped.T.flatten()

def simple_deinterleaver(data):
    n = len(data)
    rows = 8 # Yazarken sütun olan okurken satır olur
    cols = n // rows
    if cols == 0: return data
    reshaped = data[:rows*cols].reshape(rows, cols)
    return reshaped.T.flatten()

# FEC: Basit (7,4) Hamming veya Evrişimsel kod yerine hız için 
# bu örnekte mantığı TX/RX içinde kuralım.


def get_bits_per_symbol(scheme):
    if scheme == 'BPSK': return 1
    elif scheme == 'QPSK': return 2
    elif scheme == '16QAM': return 4
    else: raise ValueError("Desteklenmeyen modülasyon!")

def modulate(bits, scheme):
    if scheme == 'BPSK':
        # 0 -> -1, 1 -> +1 (Gücü zaten 1.0)
        return (2.0 * bits - 1.0).astype(complex)
        
    elif scheme == 'QPSK':
        mapping = {(0, 0): 1 + 1j, (0, 1): 1 - 1j, (1, 1): -1 - 1j, (1, 0): -1 + 1j}
        symbols = np.array([mapping[tuple(pair)] for pair in bits.reshape(-1, 2)], dtype=complex)
        return symbols / np.sqrt(2) # Ortalama gücü 1'e sabitlemek için sqrt(2)'ye bölüyoruz
        
    elif scheme == '16QAM':
        # Gray Coding Mapping
        mapping_1d = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
        bits_reshaped = bits.reshape(-1, 4)
        syms = np.zeros(len(bits_reshaped), dtype=complex)
        for i, b in enumerate(bits_reshaped):
            re = mapping_1d[tuple(b[0:2])]
            im = mapping_1d[tuple(b[2:4])]
            syms[i] = re + 1j * im
        return syms / np.sqrt(10) # 16-QAM ortalama gücü 10'dur, 1'e sabitlemek için sqrt(10)'a bölüyoruz

def demodulate(symbols, scheme):
    if scheme == 'BPSK':
        return (symbols.real > 0).astype(np.uint8)
        
    elif scheme == 'QPSK':
        bits = np.zeros((len(symbols), 2), dtype=np.uint8)
        bits[:, 0] = symbols.real < 0
        bits[:, 1] = symbols.imag < 0
        return bits.reshape(-1)
        
    elif scheme == '16QAM':
        symbols = symbols * np.sqrt(10) # Normalizasyonu geri al
        bits = np.zeros((len(symbols), 4), dtype=np.uint8)
        # Reel Kısım (I) Demodülasyonu
        bits[:, 0] = symbols.real > 0
        bits[:, 1] = np.abs(symbols.real) < 2
        # Sanal Kısım (Q) Demodülasyonu
        bits[:, 2] = symbols.imag > 0
        bits[:, 3] = np.abs(symbols.imag) < 2
        return bits.reshape(-1)


def awgn(sig, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(sig) ** 2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    noise = sigma * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
    return sig + noise

def apply_awgn_scfdma(tx_signal, snr_db, N, M):
    snr_linear = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(tx_signal) ** 2)
    # N adet alt taşıyıcının sadece M adedini kullandığımız için gürültü gücünü N/M oranında kalibre ediyoruz
    noise_power = (sig_power / snr_linear) * (N / M)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*tx_signal.shape) + 1j * np.random.randn(*tx_signal.shape))
    return tx_signal + noise


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

    num_slots = 2000
    symbols_per_slot = 7
    pilot_symbol_in_slot = 3
    total_symbols = num_slots * symbols_per_slot
    total_data_symbols = num_slots * (symbols_per_slot - 1)

    snr_db_range = np.arange(0, 15, 1)

    pilot_symbol = np.ones(M, dtype=complex) * (1 + 1j) / np.sqrt(2)

    plt.ion() # İnteraktif modu aç
    fig, (ax_const, ax_ber) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- YENİ: Constellation Heatmap Ayarları ---
    heatmap_buffer = [] # Sembolleri biriktireceğimiz hafıza havuzu
    buffer_size = 2000  # Son 2000 sembolü (yaklaşık 25-30 slot) hafızada tutalım
    
    # 50x50 çözünürlüğünde boş bir ısı haritası başlatıyoruz (cmap='magma', 'jet' veya 'hot' seçebilirsin)
    heatmap_img = ax_const.imshow(np.zeros((50, 50)), extent=[-2, 2, -2, 2], origin='lower', cmap='magma', vmin=0, vmax=10, aspect='auto')
    ax_const.set_xlim(-2, 2); ax_const.set_ylim(-2, 2)
    ax_const.set_title("Canlı Constellation Density (Heatmap)")
    ax_const.grid(True, color='white', alpha=0.2) # Izgarayı hafifletiyoruz ki renkler öne çıksın
    # ---------------------------------------------

    # Anlık BER Plot ayarları (Eski kodunla aynı)
    ber_list = []
    # ...

    # Anlık BER Plot ayarları
    ber_list = []
    ber_plot, = ax_ber.semilogy([], [], 'b-')
    ax_ber.set_xlim(0, num_slots); ax_ber.set_ylim(1e-6, 1)
    ax_ber.set_title("Canlı BER Takibi")
    ax_ber.set_xlabel("Slot Sayısı")
    ax_ber.grid(True)
    
    plt.tight_layout()

    resource_map = np.full((total_symbols, M), 2, dtype=int)
    for s in range(total_symbols):
        if (s % symbols_per_slot) == pilot_symbol_in_slot:
            resource_map[s, :] = 1

    if not ONLY_CORE_GRAPHS:
        plt.figure(figsize=(10, 4))
        plt.imshow(resource_map.T, aspect='auto', origin='lower')
        plt.xlabel('SC-FDMA Symbol Index')
        plt.ylabel('Allocated Subcarrier Index')
        plt.title('SC-FDMA Resource Grid (1=Pilot, 2=Data)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # İstatistikler için sayaçlar
    total_bits_processed = 0
    total_errors = 0
    snr_db = 20  # Sabit bir SNR değeri veya döngü kullanabilirsin
    
    # Başlangıç indeksleri (Mapping için)
    start_idx = (N // 2) - (M // 2)
    
    # Pilot sembolü (Boosting olmadan standart hali)
    pilot_symbol = np.ones(M, dtype=complex) * (1 + 1j) / np.sqrt(2)

    channel_results = compute_channel_series()
    channel_symbol = resample_channel_to_symbols(channel_results, total_symbols)

    if not ONLY_CORE_GRAPHS:
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

    snr_db_range = np.arange(0, 31, 2)
    simulated_ber_perfect = []
    simulated_ber_practical = []
    example_doppler_hz = 1500.0 

   # SNR Döngüsü Başlangıcı
    for snr_db in snr_db_range:
        # Her SNR adımı için istatistikleri sıfırla
        total_errors_prac = 0
        total_errors_perf = 0
        total_bits_step = 0
        ber_history = []
        slot_history = []
        
        print(f"\n📡 Real-time Loopback Başlatıldı (SNR: {snr_db} dB)")

        for slot in range(num_slots):
            # --- A. VERİ ÜRETİMİ (TX) ---
            bps = get_bits_per_symbol(MODULATION_SCHEME)
            
            # 1. Ham veri üretimi (Rate 1/3 için 3'e bölüyoruz)
            raw_bits = np.random.randint(0, 2, ((symbols_per_slot - 1) * bps * M) // 3, dtype=np.uint8)
            
            # 2. FEC: Rate 1/3 Repetition Code
            fec_bits = np.repeat(raw_bits, 3) 
            
            # 3. Interleaving
            interleaved_bits = simple_interleaver(fec_bits)
            
            # 4. Modülasyon
            current_data_syms = modulate(interleaved_bits, MODULATION_SCHEME).reshape(symbols_per_slot - 1, M)
            
            slot_tx_blocks = []
            data_ptr = 0
            
            for sym_idx in range(symbols_per_slot):
                d = pilot_symbol if sym_idx == pilot_symbol_in_slot else current_data_syms[data_ptr]
                if sym_idx != pilot_symbol_in_slot: data_ptr += 1
                
                # SC-FDMA: DFT Precoding -> Mapping -> IFFT -> CP
                D = np.fft.fft(d, n=M)
                X = np.zeros(N, dtype=complex)
                X[start_idx : start_idx + M] = D
                
                x_time = np.fft.ifft(np.fft.ifftshift(X), n=N)
                slot_tx_blocks.append(np.concatenate([x_time[-CP:], x_time]))

            # --- B. KANAL ETKİSİ (CHANNEL) ---
            h_channel = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
            
            # 1500 Hz çok yüksek, bunu LEO uydu kompanzasyonu yapılmış gibi düşünerek düşür:
            current_doppler = 50 # Hz (Örnek değer)
            
            slot_rx_blocks = []
            snr_linear = 10 ** (snr_db / 10)

            for tx_block in slot_tx_blocks:
                t = np.arange(len(tx_block)) / fs
                ch_out = tx_block * np.exp(1j * 2 * np.pi * current_doppler * t) * h_channel
                
                # Sinyalin gerçek gücünü ölç (Modülasyon ne olursa olsun otomatik ayarlar)
                sig_power = np.mean(np.abs(tx_block)**2)
                noise_power = (sig_power / snr_linear) * (N / M)
                noise_sigma = np.sqrt(noise_power / 2)
                
                noise = noise_sigma * (np.random.randn(len(ch_out)) + 1j * np.random.randn(len(ch_out)))
                slot_rx_blocks.append(ch_out + noise)

            # --- C. ALICI İŞLEMLERİ (RX) ---
            rx_raw_syms = []
            for rx_block in slot_rx_blocks:
                # 1. Doppler Kompanzasyonu
                t = np.arange(len(rx_block)) / fs
                comp = rx_block * np.exp(-1j * 2 * np.pi * current_doppler * t)
                
                # 2. CP Kaldır -> FFT -> De-mapping -> M-IFFT (SC-FDMA Alıcı Yapısı)
                X_hat = np.fft.fftshift(np.fft.fft(comp[CP:], n=N))
                D_hat = X_hat[start_idx : start_idx + M]
                rx_raw_syms.append(np.fft.ifft(D_hat, n=M))
                
            # 3. MMSE Kanal Kestirimi (Pilot Sembolü üzerinden)
            H_ls = rx_raw_syms[pilot_symbol_in_slot] / pilot_symbol
            mmse_w = 1 / (1 + (1 / snr_linear))
            H_mmse = H_ls * mmse_w
            
            # 4. Veri Sembollerini Denkleştirme ve Demodülasyon
            demod_bits_buffer = []
            perf_bits_buffer = []
            slot_eq_syms = [] 

            for sym_idx in range(symbols_per_slot):
                if sym_idx != pilot_symbol_in_slot:
                    # Pratik Alıcı (Equalization)
                    eq_sym_prac = rx_raw_syms[sym_idx] / H_mmse
                    slot_eq_syms.extend(eq_sym_prac) # Heatmap için sakla
                    
                    # Demodülasyon (Modülasyon şemasına göre bitleri çıkar)
                    demod_bits_buffer.extend(demodulate(eq_sym_prac, MODULATION_SCHEME))
                    
                    # Kusursuz Alıcı (Kıyaslama için)
                    eq_sym_perf = rx_raw_syms[sym_idx] / h_channel
                    perf_bits_buffer.extend(demodulate(eq_sym_perf, MODULATION_SCHEME))

            # 5. De-interleaving
            prac_deinterleaved = simple_deinterleaver(np.array(demod_bits_buffer))
            perf_deinterleaved = simple_deinterleaver(np.array(perf_bits_buffer))

            # 6. FEC Decoding (Rate 1/3 Çoğunluk Oylaması)
            # 3 bitten en az 2'si '1' ise 1, yoksa '0' kabul et
            decoded_bits_prac = (prac_deinterleaved.reshape(-1, 3).sum(axis=1) >= 2).astype(np.uint8)
            decoded_bits_perf = (perf_deinterleaved.reshape(-1, 3).sum(axis=1) >= 2).astype(np.uint8)

            # Sonuçları numpy array formatına getir (Analiz aşaması için)
            decoded_bits_prac = np.array(decoded_bits_prac)
            decoded_bits_perf = np.array(decoded_bits_perf)


# --- D. ANALİZ VE CANLI GÜNCELLEME ---
            heatmap_buffer.extend(slot_eq_syms)
            if len(heatmap_buffer) > buffer_size:
                heatmap_buffer = heatmap_buffer[-buffer_size:]

            decoded_bits_prac = np.array(decoded_bits_prac)
            decoded_bits_perf = np.array(decoded_bits_perf)
            
            # BURASI DEĞİŞTİ: current_bits yerine raw_bits kullanıyoruz.
            # Boyut uyumsuzluğunu önlemek için [:len(raw_bits)] ile sınırlandırıyoruz.
            errors_prac = np.sum(raw_bits != decoded_bits_prac[:len(raw_bits)])
            errors_perf = np.sum(raw_bits != decoded_bits_perf[:len(raw_bits)])
            
            total_errors_prac += errors_prac
            total_errors_perf += errors_perf
            total_bits_step += len(raw_bits) # current_bits yerine raw_bits
            
            
            # Her 20 slotta bir grafiği güncelle
            if slot % 20 == 0:
                current_ber = total_errors_prac / total_bits_step
                print(f"Slot {slot:4d} | Güncel BER: {current_ber:.2e}")
                
                # Isı Haritası (Heatmap)
                if len(heatmap_buffer) > 0:
                    H, xedges, yedges = np.histogram2d(
                        np.real(heatmap_buffer), 
                        np.imag(heatmap_buffer), 
                        bins=50, 
                        range=[[-2, 2], [-2, 2]]
                    )
                    heatmap_img.set_data(H.T) 
                    max_density = np.max(H)
                    heatmap_img.set_clim(0, max_density if max_density > 0 else 1)
                
                # BER grafiği
                ber_history.append(current_ber)
                slot_history.append(slot)
                ber_plot.set_data(slot_history, ber_history)
                
                ax_ber.set_title(f"{MODULATION_SCHEME} | SNR: {snr_db}dB | BER: {current_ber:.2e}")
                plt.draw()
                plt.pause(0.001)

        # Her SNR adımı bittiğinde ortalama BER'i ana listeye kaydet
        simulated_ber_practical.append(total_errors_prac / total_bits_step)
        simulated_ber_perfect.append(total_errors_perf / total_bits_step) 

    plt.ioff()

    # 3. BER vs SNR Grafiği (Teorik vs Pratik vs Perfect)
    snr_linear = 10 ** (snr_db_range / 10)
    
    # Seçilen modülasyona göre referans SNR (Es/N0) hesaplamaları
    if MODULATION_SCHEME == 'BPSK':
        ebno_linear = snr_linear
        theoretical_awgn_ber = q_function(np.sqrt(2 * ebno_linear))
        theoretical_rayleigh_ber = 0.5 * (1 - np.sqrt(ebno_linear / (1 + ebno_linear)))
    else: # QPSK ve 16-QAM referansı için QPSK tabanını kullanıyoruz
        ebno_linear = snr_linear / 2.0  
        theoretical_awgn_ber = q_function(np.sqrt(2 * ebno_linear))
        theoretical_rayleigh_ber = 0.5 * (1 - np.sqrt(ebno_linear / (1 + ebno_linear)))

    # Sıfıra düşen hataları grafikte çökmemesi için filtreliyoruz
    plot_ber_perf = [b if b > 0 else np.nan for b in simulated_ber_perfect]
    plot_ber_prac = [b if b > 0 else np.nan for b in simulated_ber_practical]

    plt.figure(figsize=(10, 7))
    
    # Siyah ve Yeşil referans çizgilerini dinamik isimlendirdik
    ref_label = 'BPSK' if MODULATION_SCHEME == 'BPSK' else 'QPSK'
    plt.semilogy(snr_db_range, theoretical_awgn_ber, 'g--', linewidth=2, label=f'Theoretical AWGN ({ref_label} Ref)')
    plt.semilogy(snr_db_range, theoretical_rayleigh_ber, 'k--', linewidth=2, label=f'Theoretical Rayleigh ({ref_label} Ref)')
    
    # Simülasyon Sonuçları
    plt.semilogy(snr_db_range, plot_ber_perf, 'bo-', linewidth=2, markersize=6, label=f'Simulated (Perfect CSI - {MODULATION_SCHEME})')
    plt.semilogy(snr_db_range, plot_ber_prac, 'rs-', linewidth=2, markersize=6, label=f'Simulated (Practical CSI - {MODULATION_SCHEME})')
    
    plt.xlabel('SNR ($E_s/N_0$) (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title(f'SC-FDMA BER vs SNR ({MODULATION_SCHEME} - Rayleigh Fading Channel)')
    
    plt.ylim(1e-6, 1)
    plt.xlim(-1, 31)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # KANAL KESTİRİM KAYBI (SNR PENALTY) HESAPLAMA
    # Hedef BER = 10^-3 noktasında iki eğri arasındaki dB farkını buluyoruz
    # ==========================================================
    target_ber = 1e-3
    
    # Sadece 0'dan büyük (geçerli) BER değerlerini logaritmik interpolasyon için ayırıyoruz
    perf_valid = [(snr_db_range[i], np.log10(b)) for i, b in enumerate(simulated_ber_perfect) if b > 0]
    prac_valid = [(snr_db_range[i], np.log10(b)) for i, b in enumerate(simulated_ber_practical) if b > 0]
    
    if len(perf_valid) > 1 and len(prac_valid) > 1:
        # Listeleri ayır ve interpolasyon fonksiyonu x'in artan olmasını istediği için ters çevir ([::-1])
        perf_snrs, perf_bers = zip(*perf_valid)
        prac_snrs, prac_bers = zip(*prac_valid)
        
        # log10(BER) değerine karşılık gelen SNR değerini bul (np.interp ters interpolasyon)
        snr_at_target_perf = np.interp(np.log10(target_ber), perf_bers[::-1], perf_snrs[::-1])
        snr_at_target_prac = np.interp(np.log10(target_ber), prac_bers[::-1], prac_snrs[::-1])
        
        db_penalty = snr_at_target_prac - snr_at_target_perf
        
        print("\n" + "="*40)
        print("📡 SİSTEM PERFORMANS ANALİZİ")
        print("="*40)
        print(f"Hedef BER Noktası       : 10^{int(np.log10(target_ber))}")
        print(f"Perfect CSI Gerekli SNR : {snr_at_target_perf:.2f} dB")
        print(f"Practical CSI (LS) SNR  : {snr_at_target_prac:.2f} dB")
        print(f"Kanal Kestirim Kaybı    : {db_penalty:.2f} dB")
        print("="*40 + "\n")
    else:
        print("[Uyarı] Hedef BER'e ulaşılamadığı için SNR farkı hesaplanamadı.")


if __name__ == '__main__':
    run_simulation()
