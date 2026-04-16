import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. NB-IoT İLETİM PARAMETRELERİ (Uplink)
# ==========================================
num_symbols = 1000  # Gönderilecek QPSK sembol sayısı
tx_power_dbm = 23.0 # Standart NB-IoT cihaz iletim gücü (23 dBm = 200 mW)
bandwidth_hz = 180000 # 180 kHz (1 PRB - Physical Resource Block)
subcarrier_spacing = 15000 # 15 kHz SCS

# Sembol süresi hesabı (Doppler'in faza etkisini bulmak için)
symbol_duration_s = 1 / subcarrier_spacing
time_array = np.arange(num_symbols) * symbol_duration_s

# Termal Gürültü Gücü (Karasal ve Uzay için standart taban gürültüsü)
# Formül: -174 dBm/Hz + 10*log10(Bandwidth)
noise_power_dbm = -174.0 + 10 * np.log10(bandwidth_hz)

# ==========================================
# 2. ÖNCEKİ ADIMLARDAN GELEN KANAL VERİLERİ (Örnek Senaryo)
# ==========================================
# (Bunları arayüzde dinamik olarak bağlayacağız, şimdilik test için sabit veriyoruz)
total_path_loss_db = 155.0 # FSPL + Atmosferik + Lognormal Gölgeleme
doppler_shift_hz = -250.0  # LEO uydusu bağıl hızından kaynaklı
rician_k_factor_db = 5.0   # Rician LoS gücü

# ==========================================
# 3. SİNYAL ÜRETİMİ VE KANALIN UYGULANMASI
# ==========================================
# A. Rastgele QPSK Sembolleri Üretme (I ve Q eksenlerinde +1 veya -1)
# 1+1j, 1-1j, -1+1j, -1-1j değerlerini üretip normalize ediyoruz
qpsk_symbols = (np.random.choice([-1, 1], num_symbols) + 1j * np.random.choice([-1, 1], num_symbols)) / np.sqrt(2)

# B. Doppler Faz Kaymasının (Phase Rotation) Uygulanması
# Formül: S_doppler = S * e^(j * 2 * pi * f_doppler * t)
phase_rotation = np.exp(1j * 2 * np.pi * doppler_shift_hz * time_array)
symbols_doppler = qpsk_symbols * phase_rotation

# C. Rician Fading (Çok Yollu Sönümleme) Uygulanması
k_linear = 10 ** (rician_k_factor_db / 10)
los_component = np.sqrt(k_linear / (k_linear + 1))
nlos_component = (np.random.normal(0, 1, num_symbols) + 1j * np.random.normal(0, 1, num_symbols)) * np.sqrt(1 / (2 * (k_linear + 1)))
h_rician = los_component + nlos_component
symbols_faded = symbols_doppler * h_rician

# D. Sinyal/Gürültü Oranı (SNR) ve AWGN (Beyaz Gürültü) Eklenmesi
rx_power_dbm = tx_power_dbm - total_path_loss_db
snr_db = rx_power_dbm - noise_power_dbm
snr_linear = 10 ** (snr_db / 10)

# Gürültüyü SNR'a göre ölçeklendirip sinyale ekliyoruz
noise = (np.random.normal(0, 1, num_symbols) + 1j * np.random.normal(0, 1, num_symbols)) / np.sqrt(2)
noise_scaled = noise / np.sqrt(snr_linear)
rx_symbols_final = symbols_faded + noise_scaled

# ==========================================
# 4. GÖRSELLEŞTİRME (CONSTELLATION DIAGRAMS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'NTN NB-IoT Uplink Sinyal Analizi | SNR: {snr_db:.1f} dB', fontsize=16)

# 1. Orijinal Temiz QPSK
ax1.scatter(qpsk_symbols.real, qpsk_symbols.imag, color='blue', alpha=0.6)
ax1.set_title('1. Temiz QPSK Sembolleri (Tx)')
ax1.set_xlabel('In-Phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.grid(True, linestyle='--')
ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
ax1.axhline(0, color='black', linewidth=0.5); ax1.axvline(0, color='black', linewidth=0.5)

# 2. Sadece Fading ve Gürültü (AWGN) Eklenmiş Hali
symbols_noisy_only = (qpsk_symbols * h_rician) + noise_scaled
ax2.scatter(symbols_noisy_only.real, symbols_noisy_only.imag, color='orange', alpha=0.6)
ax2.set_title('2. Fading + AWGN (Doppler Yok)')
ax2.set_xlabel('In-Phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.grid(True, linestyle='--')
ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
ax2.axhline(0, color='black', linewidth=0.5); ax2.axvline(0, color='black', linewidth=0.5)

# 3. Tüm Etkiler (Fading + AWGN + Doppler)
ax3.scatter(rx_symbols_final.real, rx_symbols_final.imag, color='red', alpha=0.6)
ax3.set_title(f'3. Tam NTN Kanalı (Doppler: {doppler_shift_hz} Hz)')
ax3.set_xlabel('In-Phase (I)')
ax3.set_ylabel('Quadrature (Q)')
ax3.grid(True, linestyle='--')
ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
ax3.axhline(0, color='black', linewidth=0.5); ax3.axvline(0, color='black', linewidth=0.5)

plt.tight_layout(pad=2.0)
plt.show()

print(f"Alınan Sinyal Gücü (Rx): {rx_power_dbm:.2f} dBm")
print(f"Gürültü Tabanı (Noise Floor): {noise_power_dbm:.2f} dBm")
print(f"Hesaplanan SNR: {snr_db:.2f} dB")