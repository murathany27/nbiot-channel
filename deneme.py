import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
import random





# ==========================================
# 0. GERÇEK UYDU KANALI FONKSİYONU VE HESAPLAMALARI
# ==========================================
def get_real_satellite_params():
    ts = load.timescale()
    t_now = ts.now()
    satellite = EarthSatellite('1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997', 
                               '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000', 'CONNECTA', ts)
    ground_station = wgs84.latlon(39.9208, 32.8541) # Ankara
    
    t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
    t_pass, events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=10.0)
    
    tepe_indeksleri = [i for i, event in enumerate(events) if event == 1]
    
    if tepe_indeksleri:
        secilen_indeks = random.choice(tepe_indeksleri)
        t_sim = t_pass[secilen_indeks]
    else:
        t_sim = t_now

    elev_deg, _, dist_km = (satellite - ground_station).at(t_sim).altaz()
    elev_deg, dist_km = elev_deg.degrees, dist_km.km
    
    t_next = ts.from_datetime(t_sim.utc_datetime() + timedelta(seconds=1))
    _, _, dist_next = (satellite - ground_station).at(t_next).altaz()
    v_rel = dist_km - dist_next.km
    
    carrier_freq_mhz = 2000
    real_doppler = (v_rel / 299792.458) * (carrier_freq_mhz * 1e6)
    
    fspl = 32.44 + 20*np.log10(dist_km) + 20*np.log10(carrier_freq_mhz)
    atm_loss = 0.5 / np.sin(np.radians(elev_deg))
    noise_floor = -174.0 + 10*np.log10(180000) 
    
    sat_rx_gain_dbi = 30.0 
    ue_tx_gain_dbi = 0.0 
    tx_power_dbm = 23.0
    
    snr_db = (tx_power_dbm + ue_tx_gain_dbi + sat_rx_gain_dbi) - (fspl + atm_loss) - noise_floor
    k_factor_db = min(15.0, 2.0 + ((elev_deg - 10) / 80) * 13.0)
    
    return real_doppler, snr_db, k_factor_db, elev_deg, t_sim, satellite, ground_station, dist_km, fspl, atm_loss

def apply_leo_channel(tx_signal, fs, doppler_hz, delay_samples, snr_db, k_factor_db):
    rx = np.pad(tx_signal, (delay_samples, 200), mode='constant')
    t = np.arange(len(rx)) / fs
    rx = rx * np.exp(1j * 2 * np.pi * doppler_hz * t)
    
    k_lin = 10 ** (k_factor_db / 10)
    los = np.sqrt(k_lin / (k_lin + 1))
    nlos = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2 * (k_lin + 1)) 
    h_channel = (los + nlos)
    rx = rx * h_channel
    
    noise = (np.random.normal(0, 1, len(rx)) + 1j * np.random.normal(0, 1, len(rx))) / np.sqrt(2)
    sig_pwr = np.var(rx)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    rx = rx + noise * np.sqrt(noise_pwr)
    
    return rx, h_channel

# Uydu Parametrelerini Çek
real_doppler, SNR_dB, k_factor_db, elev, t_sim, satellite, ground_station, dist_km, fspl, atm_loss = get_real_satellite_params()







# ==========================================
# 1. PARAMETRELER VE VERİ ÜRETİMİ (3GPP NPUSCH)
# ==========================================
delta_f = 15000
M = 1; N = 128; CP = 9
fs = N * delta_f
delay_in_samples = 10
est_doppler = real_doppler - 150.0  

num_slots = 10              
symbols_per_slot = 7        
pilot_idx = 3               
num_data_per_slot = symbols_per_slot - 1
total_data_symbols = num_slots * num_data_per_slot

bits_tx = np.random.randint(0, 2, total_data_symbols * 2 * M, dtype=np.uint8)

def qpsk_mod(bits):
    mapping = {(0,0): 1+1j, (0,1): 1-1j, (1,0): -1+1j, (1,1): -1-1j}
    return np.array([mapping[tuple(b)] for b in bits.reshape(-1, 2)]) / np.sqrt(2)

qpsk_data_syms = qpsk_mod(bits_tx).reshape(total_data_symbols, M)
pilot_symbol = np.array([1+1j], dtype=complex) / np.sqrt(2)









# ==========================================
# 2. SC-FDMA VERİCİ (TRANSMITTER)
# ==========================================
tx_blocks = []
data_counter = 0

for slot in range(num_slots):
    for sym in range(symbols_per_slot):
        d = pilot_symbol if sym == pilot_idx else qpsk_data_syms[data_counter]
        if sym != pilot_idx: data_counter += 1

        D = np.fft.fft(d, n=M)
        X_shift = np.zeros(N, dtype=complex)
        X_shift[(N // 2) + 1] = D[0] 

        x = np.fft.ifft(np.fft.ifftshift(X_shift), n=N)
        tx_blocks.append(np.concatenate([x[-CP:], x]))

tx_signal_base = np.concatenate(tx_blocks)
t_tx = np.arange(len(tx_signal_base)) / fs
tx_signal = tx_signal_base * np.exp(-1j * 2 * np.pi * est_doppler * t_tx) 







# ==========================================
# 3. KANAL (GERÇEK LEO FİZİĞİ)
# ==========================================
rx_signal, h_channel = apply_leo_channel(tx_signal, fs, real_doppler, delay_in_samples, SNR_dB, k_factor_db)








# ==========================================
# 4. GERÇEKÇİ SC-FDMA ALICI (RECEIVER)
# ==========================================
rx_data_syms = np.zeros((total_data_symbols, M), dtype=complex)
raw_rx_symbols = [] 
block_len = N + CP
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
coarse_fixed_symbols = [] 

for slot in range(num_slots):
    slot_blocks = []
    for sym in range(symbols_per_slot):
        block = rx_signal_coarse_fixed[idx : idx + block_len][CP:]
        X_hat = np.fft.fftshift(np.fft.fft(block, n=N))
        D_hat = np.zeros(M, dtype=complex); D_hat[0] = X_hat[(N // 2) + 1]
        coarse_fixed_symbols.append(np.fft.ifft(D_hat, n=M)[0])
        slot_blocks.append(np.fft.ifft(D_hat, n=M))
        
        block_raw = rx_signal[idx : idx + block_len][CP:]
        X_hat_raw = np.fft.fftshift(np.fft.fft(block_raw, n=N))
        D_hat_raw = np.zeros(M, dtype=complex); D_hat_raw[0] = X_hat_raw[(N // 2) + 1]
        raw_rx_symbols.append(np.fft.ifft(D_hat_raw, n=M)[0])
        idx += block_len

    H_est = slot_blocks[pilot_idx] / pilot_symbol

    for sym in range(symbols_per_slot):
        if sym != pilot_idx:
            rx_data_syms[data_counter] = slot_blocks[sym] / H_est
            data_counter += 1

rx_syms_flat = rx_data_syms.reshape(-1)
def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:,0], bits[:,1] = symbols.real < 0, symbols.imag < 0
    return bits.reshape(-1)
BER = np.mean(qpsk_demod(rx_syms_flat) != bits_tx)









# ==========================================
# 5. MEGA DASHBOARD GÖRSELLEŞTİRME (3x3 Grid)
# ==========================================
ts_scale = load.timescale()
times_array = [ts_scale.from_datetime(t_sim.utc_datetime() + timedelta(minutes=m)) for m in np.linspace(-5, 5, 100)]
dopplers_theory = []
elevs_theory = []
dists_theory = []

for t_val in times_array:
    e, _, d = (satellite - ground_station).at(t_val).altaz()
    t_n = ts_scale.from_datetime(t_val.utc_datetime() + timedelta(seconds=1))
    _, _, d_n = (satellite - ground_station).at(t_n).altaz()
    v = d.km - d_n.km
    dopplers_theory.append((v / 299792.458) * (2000 * 1e6))
    elevs_theory.append(e.degrees)
    dists_theory.append(d.km)

fig = plt.figure(figsize=(22, 14))
fig.suptitle(f"NTN LEO Uydu ve SC-FDMA Sinyal Analiz Paneli | Geçiş: {t_sim.utc_strftime('%Y-%m-%d %H:%M UTC')}", fontsize=18, fontweight='bold')

# --- 1. Uydu Haritası ---
ax1 = fig.add_subplot(3, 3, 1, projection=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND, facecolor='#e6e6e6')
ax1.add_feature(cfeature.COASTLINE)
ax1.set_extent([20, 50, 30, 50], crs=ccrs.PlateCarree())
subp = wgs84.subpoint(satellite.at(t_sim))
ax1.plot(32.8541, 39.9208, 'r^', markersize=10, transform=ccrs.PlateCarree(), label="Ankara (UE)")
ax1.plot(subp.longitude.degrees, subp.latitude.degrees, 'bo', markersize=8, transform=ccrs.PlateCarree(), label="Uydu")
ax1.plot([32.8541, subp.longitude.degrees], [39.9208, subp.latitude.degrees], 'g--', transform=ccrs.Geodetic())
ax1.set_title("O Anki Yörünge Geometrisi")
ax1.set_xlabel("Boylam (°)") # EKSEN
ax1.set_ylabel("Enlem (°)") # EKSEN
ax1.legend(loc='lower left')

# --- 2. Elevasyon ve Mesafe Değişimi ---
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(np.linspace(-5, 5, 100), elevs_theory, 'b-', label='Elevasyon (°)')
ax2.axvline(0, color='red', linestyle='--', label='Şu An')
ax2.set_title("Geçiş Boyunca Geometrik Değişim")
ax2.set_xlabel('Zaman (Dakika)') # EKSEN
ax2.set_ylabel('Elevasyon Açısı (°)', color='b') # EKSEN
ax2_twin = ax2.twinx()
ax2_twin.plot(np.linspace(-5, 5, 100), dists_theory, 'g-', label='Mesafe (km)')
ax2_twin.set_ylabel('Mesafe (km)', color='g') # EKSEN
ax2.grid(True, alpha=0.3)

# --- 3. Doppler S-Eğrisi ---
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(np.linspace(-5, 5, 100), dopplers_theory, 'purple', linewidth=2)
ax3.plot(0, real_doppler, 'ro', markersize=8, label=f'Anlık: {real_doppler:.0f} Hz')
ax3.axhline(0, color='black', linewidth=1)
ax3.set_title("Doppler S-Eğrisi (10 Dakikalık Geçiş)")
ax3.set_xlabel("Zaman (Dakika)") # EKSEN
ax3.set_ylabel("Doppler Kayması (Hz)") # EKSEN
ax3.grid(True, alpha=0.5)
ax3.legend()

# --- 4. Link Budget (Bağlantı Bütçesi) ---
ax4 = fig.add_subplot(3, 3, 4)
items = ['Tx Gücü', 'Anten', 'FSPL', 'Atmosferik', 'Gürültü Zemin']
values = [23.0, 30.0, -fspl, -atm_loss, -(-121.4)] 
ax4.bar(items, values, color=['green', 'green', 'red', 'red', 'gray'])
ax4.axhline(0, color='black')
ax4.set_title(f"Link Bütçesi | Nihai SNR: {SNR_dB:.1f} dB")
ax4.set_xlabel("Bütçe Parametreleri") # EKSEN
ax4.set_ylabel("Güç Seviyesi (dB / dBm)") # EKSEN

# --- 5. Rician Fading Karakteristiği ---
ax5 = fig.add_subplot(3, 3, 5)
rx_envelope = 10 * np.log10(np.abs(rx_signal[:2000]**2) + 1e-12)
ax5.plot(rx_envelope, color='teal', linewidth=0.8)
ax5.axhline(np.mean(rx_envelope), color='red', linestyle='--')
ax5.set_title(f"Zaman Uzayı Sönümleme (Rician K = {k_factor_db:.1f} dB)")
ax5.set_xlabel("Örneklem (İndeks)") # EKSEN
ax5.set_ylabel("Anlık Sinyal Gücü (dB)") # EKSEN

# --- 6. Tx Sinyal Spektrumu ---
ax6 = fig.add_subplot(3, 3, 6)
freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_signal), d=1/fs)) / 1000
window = np.hanning(len(tx_signal))
S = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_signal * window))) + 1e-12)
S = S - np.max(S)
ax6.plot(freqs, S, color='#1f77b4')
ax6.set_xlim([-100, 100]); ax6.set_ylim([-60, 5])
ax6.set_title("NPUSCH SC-FDMA Spektrumu (15 kHz)")
ax6.set_xlabel("Frekans (kHz)") # EKSEN
ax6.set_ylabel("Güç (dB)") # EKSEN
ax6.grid(True)

# --- 7. Ham Rx Takımyıldızı (Raw Constellation) ---
ax7 = fig.add_subplot(3, 3, 7)
ax7.scatter(np.real(raw_rx_symbols), np.imag(raw_rx_symbols), color='gray', alpha=0.6, s=15)
ax7.set_title(f"1. Ham Sinyal (RFO: {real_doppler - est_doppler:.0f} Hz)")
ax7.set_xlabel("In-Phase (Reel - I)") # EKSEN
ax7.set_ylabel("Quadrature (Sanal - Q)") # EKSEN
ax7.set_xlim(-2, 2); ax7.set_ylim(-2, 2)
ax7.grid(True, linestyle='--'); ax7.axhline(0, color='k', lw=0.5); ax7.axvline(0, color='k', lw=0.5)

# --- 8. Kaba Düzeltilmiş (Coarse Fixed) ---
ax8 = fig.add_subplot(3, 3, 8)
ax8.scatter(np.real(coarse_fixed_symbols), np.imag(coarse_fixed_symbols), color='orange', alpha=0.6, s=15)
ax8.set_title(f"2. CP Kaba Tahmini ({estimated_doppler_coarse:.0f} Hz)")
ax8.set_xlabel("In-Phase (Reel - I)") # EKSEN
ax8.set_ylabel("Quadrature (Sanal - Q)") # EKSEN
ax8.set_xlim(-2, 2); ax8.set_ylim(-2, 2)
ax8.grid(True, linestyle='--'); ax8.axhline(0, color='k', lw=0.5); ax8.axvline(0, color='k', lw=0.5)

# --- 9. Nihai Pilot Equalized ---
ax9 = fig.add_subplot(3, 3, 9)
ax9.scatter(rx_syms_flat.real, rx_syms_flat.imag, color='green', alpha=0.8, s=20)
ax9.set_title(f"3. DMRS Equalized (Final)\nBER: {BER:.2e}")
ax9.set_xlabel("In-Phase (Reel - I)") # EKSEN
ax9.set_ylabel("Quadrature (Sanal - Q)") # EKSEN
ax9.set_xlim(-2, 2); ax9.set_ylim(-2, 2)
ax9.grid(True, linestyle='--'); ax9.axhline(0, color='k', lw=0.5); ax9.axvline(0, color='k', lw=0.5)

plt.tight_layout(pad=2.5)
plt.show()