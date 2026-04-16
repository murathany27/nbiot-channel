import numpy as np
import matplotlib.pyplot as plt
from leo_channel import apply_leo_channel
# ==========================================
# 1. PARAMETRELER (GERÇEK 3GPP NPUSCH FORMAT 1)
# ==========================================
delta_f = 15000
M = 1
N = 128
CP = 9
SNR_dB = 10

delay_in_samples = 10
doppler_shift = 50000        # Gerçek fiziksel Doppler (50 kHz)
est_doppler = 46000 # Cihazın ön-ödünleme için kullandığı tahmin

# GERÇEK ZAMANLAMA PARAMETRELERİ
num_slots = 10              # 10 Slot (Paket) göndereceğiz
symbols_per_slot = 7        # 3GPP standardı: Her slot 7 semboldür
pilot_idx = 3               # 4. sembol (indeks 3) DMRS (Pilot) sembolüdür
num_data_per_slot = symbols_per_slot - 1
total_data_symbols = num_slots * num_data_per_slot

fs = N * delta_f

# ==========================================
# 2. VERİ VE PİLOT ÜRETİMİ
# ==========================================
# Sadece veri sembolleri için bit üretiyoruz
bits_tx = np.random.randint(0, 2, total_data_symbols * 2 * M, dtype=np.uint8)

def qpsk_mod(bits):
    mapping = {(0,0): 1+1j, (0,1): 1-1j, (1,0): -1+1j, (1,1): -1-1j}
    return np.array([mapping[tuple(b)] for b in bits.reshape(-1, 2)]) / np.sqrt(2)

qpsk_data_syms = qpsk_mod(bits_tx).reshape(total_data_symbols, M)

# PİLOT (DMRS) SEMBOLÜ: Her iki tarafın da bildiği referans (Örn: QPSK 1+1j)
pilot_symbol = np.array([1+1j], dtype=complex) / np.sqrt(2)

# ==========================================
# 3. SC-FDMA VERİCİ (TRANSMITTER)
# ==========================================
tx_blocks = []
data_counter = 0

for slot in range(num_slots):
    for sym in range(symbols_per_slot):

        # Otoyola ne koyacağız? Veri mi, Pilot mu?
        if sym == pilot_idx:
            d = pilot_symbol  # 4. sembol Pilot!
        else:
            d = qpsk_data_syms[data_counter] # Diğerleri Veri!
            data_counter += 1

        D = np.fft.fft(d, n=M)

        X_shift = np.zeros(N, dtype=complex)
        X_shift[(N // 2) + 1] = D[0] # Single-Tone mapping

        x = np.fft.ifft(np.fft.ifftshift(X_shift), n=N)
        tx_blocks.append(np.concatenate([x[-CP:], x]))

tx_signal_base = np.concatenate(tx_blocks)
t_tx = np.arange(len(tx_signal_base)) / fs
tx_signal = tx_signal_base * np.exp(-1j * 2 * np.pi * est_doppler * t_tx) # HATA BURADA

# ==========================================
# 4. KANAL (KOPYA YOK!)
# ==========================================


print(f"[KANAL] Sinyal uzaya çıktı. Eklenen Doppler: {doppler_shift} Hz")
rx_signal = apply_leo_channel(tx_signal, fs, doppler_shift, delay_in_samples, SNR_dB)

# ==========================================
# 5. GERÇEKÇİ SC-FDMA ALICI (RECEIVER)
# ==========================================
rx_data_syms = np.zeros((total_data_symbols, M), dtype=complex)
block_len = N + CP
idx = delay_in_samples # (NPSS ile zamanı bulduğumuzu varsayıyoruz)

# ADIM 5.1: KABA DOPPLER TAHMİNİ (CP İLE)
cp_part = rx_signal[idx : idx + CP]
data_part = rx_signal[idx + N : idx + N + CP]
estimated_doppler_coarse = -np.angle(np.sum(cp_part * np.conj(data_part))) * fs / (2 * np.pi * N)
print(f"[ALICI] CP ile Kaba Doppler Tahmini: {estimated_doppler_coarse:.2f} Hz")

# Kaba düzeltmeyi tüm sinyale uygula (Geriye kalıntı hata / RFO kalacak)
t_rx = np.arange(len(rx_signal)) / fs
rx_signal_coarse_fixed = rx_signal * np.exp(-1j * 2 * np.pi * estimated_doppler_coarse * t_rx)

# ADIM 5.2: PİLOT TABANLI KANAL KESTİRİMİ VE İNCE AYAR (EQUALIZATION)
data_counter = 0

for slot in range(num_slots):
    # Bu slot için blokları tutacağımız geçici liste
    slot_blocks = []
    for sym in range(symbols_per_slot):
        block = rx_signal_coarse_fixed[idx : idx + block_len][CP:]
        X_hat_shift = np.fft.fftshift(np.fft.fft(block, n=N))

        D_hat = np.zeros(M, dtype=complex)
        D_hat[0] = X_hat_shift[(N // 2) + 1]
        slot_blocks.append(np.fft.ifft(D_hat, n=M))

        idx += block_len

    # --- KANAL KESTİRİMİ (CHANNEL ESTIMATION) ---
    # Alınan bozuk pilotu, orijinal pilote bölerek kanalın o anki "fotoğrafını" (H_est) çekiyoruz
    rx_pilot = slot_blocks[pilot_idx]
    H_est = rx_pilot / pilot_symbol

    # --- DENKLEŞTİRME (EQUALIZATION) ---
    # O slottaki 6 veri sembolünü, bulduğumuz bu H_est ile düzelterek (bölerek) okuyoruz
    for sym in range(symbols_per_slot):
        if sym != pilot_idx:
            # İşte gerçek mühendislik! Kalan faz ve genlik hatası PİLOT ile silindi.
            equalized_sym = slot_blocks[sym] / H_est
            rx_data_syms[data_counter] = equalized_sym
            data_counter += 1

# ==========================================
# 6. SONUÇLAR
# ==========================================
rx_syms_flat = rx_data_syms.reshape(-1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_signal), d=1/fs)) / 1000
window = np.hanning(len(tx_signal))
S = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_signal * window))) / np.max(np.abs(np.fft.fftshift(np.fft.fft(tx_signal * window)))) + 1e-12)


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:,0], bits[:,1] = symbols.real < 0, symbols.imag < 0
    return bits.reshape(-1)

BER = np.mean(qpsk_demod(rx_syms_flat) != bits_tx)
print("-" * 60)
print(f"GERÇEKÇİ SİMÜLASYON SONUCU: SC-FDMA BER = {BER:.6e}")
print("-" * 60)

axes[0].plot(freqs, S, color='#1f77b4')
axes[0].set_title("3GPP NPUSCH DMRS Transmit Spectrum")
axes[0].grid(True); axes[0].set_xlim([-200, 200]); axes[0].set_ylim([-80, 5])

axes[1].scatter(rx_syms_flat.real, rx_syms_flat.imag, s=15, color='#ff7f0e')
axes[1].set_title("Equalized Constellation (Pilot-Based)")
axes[1].grid(True); axes[1].axis("equal")
plt.tight_layout()
plt.show()