import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import load, EarthSatellite, wgs84

# ==========================================
# UZAY VE UYDU AYARLARI
# ==========================================
ts = load.timescale()
tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'
satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
ground_station = wgs84.latlon(39.9208, 32.8541) # Ankara

# NB-IoT Fiziksel Katman Parametreleri
carrier_freq_mhz = 2000
tx_power_dbm = 23.0       # İletim gücü (23 dBm)
bandwidth_hz = 180000     # 180 kHz (1 PRB)
subcarrier_spacing = 15000 # 15 kHz
noise_power_dbm = -174.0 + 10 * np.log10(bandwidth_hz) # Termal gürültü

# ==========================================
# MATEMATİKSEL FONKSİYONLAR
# ==========================================
def calculate_geometry(time):
    diff_vector = satellite - ground_station
    alt, az, distance = diff_vector.at(time).altaz()
    return alt.degrees, distance.km

def find_next_pass(start_time):
    t_future = ts.from_datetime(start_time.utc_datetime() + timedelta(days=3))
    pass_times, pass_events = satellite.find_events(ground_station, start_time, t_future, altitude_degrees=10.0)
    for i, event in enumerate(pass_events):
        if event == 1: # Tepe noktasını bul (En iyi sinyal)
            return pass_times[i]
    return start_time

# ==========================================
# ANA ARAYÜZ (GUI) VE SİMÜLASYON SINIFI
# ==========================================
class NTNSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NTN NB-IoT Kanal ve Sinyal Simülatörü")
        self.root.geometry("1600x1000") # 9 grafik için pencereyi biraz daha büyüttük
        
        # --- ÜST PANEL (Kontroller) ---
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Simülasyon Zamanı (UTC):").pack(side=tk.LEFT, padx=5)
        
        self.time_entry = ttk.Entry(control_frame, width=25)
        self.time_entry.insert(0, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        self.time_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Şu An (Now)", command=self.set_time_now).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Sonraki Geçişi Bul", command=self.set_time_next_pass).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="SİMÜLASYONU ÇALIŞTIR", command=self.run_simulation).pack(side=tk.LEFT, padx=20)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Durum: Bekleniyor...")
        ttk.Label(control_frame, textvariable=self.status_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT, padx=20)

        # --- ALT PANEL (Grafikler) ---
        self.fig = plt.figure(figsize=(18, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.run_simulation()

    def set_time_now(self):
        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        self.run_simulation()

    def set_time_next_pass(self):
        current_time_str = self.time_entry.get().strip()[:19]
        try:
            dt = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')
            t_current = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            t_next = find_next_pass(t_current)
            self.time_entry.delete(0, tk.END)
            self.time_entry.insert(0, t_next.utc_strftime('%Y-%m-%d %H:%M:%S'))
            self.run_simulation()
        except ValueError:
            self.status_var.set("Hata: Geçersiz zaman formatı!")

    def run_simulation(self):
        time_str = self.time_entry.get().strip()[:19]
        try:
            dt_sim = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            sim_time = ts.utc(dt_sim.year, dt_sim.month, dt_sim.day, dt_sim.hour, dt_sim.minute, dt_sim.second)
        except ValueError:
            self.status_var.set("Hata: Zaman formatı YYYY-MM-DD HH:MM:SS olmalı.")
            return

        elev_deg, dist_km = calculate_geometry(sim_time)
        self.fig.clf()
        
        if elev_deg <= 0:
            self.status_var.set(f"Durum: UYDU UFKUN ALTINDA! (Elev: {elev_deg:.1f}°) - Sinyal Yok.")
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Uydu şu an ufkun altında.\nLütfen 'Sonraki Geçişi Bul' butonuna tıklayın.", 
                    fontsize=16, ha='center', va='center', color='red')
            ax.axis('off')
            self.canvas.draw()
            return

        # ==========================================
        # FİZİKSEL KANAL HESAPLAMALARI
        # ==========================================
        fspl_db = 32.44 + 20 * np.log10(dist_km) + 20 * np.log10(carrier_freq_mhz)
        atm_loss_db = 0.5 / np.sin(np.radians(elev_deg))
        total_ls_loss = fspl_db + atm_loss_db

        t_next = ts.from_datetime(sim_time.utc_datetime() + timedelta(seconds=1))
        _, dist_next = calculate_geometry(t_next)
        v_rel = dist_km - dist_next 
        doppler_hz = (v_rel / 299792.458) * (carrier_freq_mhz * 1e6)

        sigma_shad = max(2.0, 8.0 - ((elev_deg - 10) / 80) * 6.0)
        k_factor_db = min(15.0, 2.0 + ((elev_deg - 10) / 80) * 13.0)
        k_linear = 10 ** (k_factor_db / 10)
        
        num_symbols = 1000
        shadowing_db = np.random.normal(0, sigma_shad, num_symbols)
        los_comp = np.sqrt(k_linear / (k_linear + 1))
        nlos_comp = (np.random.normal(0, 1, num_symbols) + 1j * np.random.normal(0, 1, num_symbols)) * np.sqrt(1/(2*(k_linear+1)))
        h_rician = los_comp + nlos_comp
        rician_db = 10 * np.log10(np.abs(h_rician)**2)
        total_dynamic = total_ls_loss + shadowing_db - rician_db

        # ==========================================
        # SİNYAL İŞLEME (SC-FDMA / QPSK)
        # ==========================================
        # SNR Hesabı
        rx_power_mean_dbm = tx_power_dbm - total_ls_loss
        snr_mean_db = rx_power_mean_dbm - noise_power_dbm
        snr_linear = 10 ** (snr_mean_db / 10)

        # 1. Temiz QPSK Sinyali Üretimi
        qpsk_symbols = (np.random.choice([-1, 1], num_symbols) + 1j * np.random.choice([-1, 1], num_symbols)) / np.sqrt(2)
        
        # 2. Termal Gürültü (AWGN) Üretimi (SNR'a göre ölçeklenmiş)
        noise = (np.random.normal(0, 1, num_symbols) + 1j * np.random.normal(0, 1, num_symbols)) / np.sqrt(2)
        noise_scaled = noise / np.sqrt(snr_linear)

        # 3. Kısmi Kanal (Sadece Fading ve AWGN - Doppler Yok)
        rx_symbols_fading_awgn = (qpsk_symbols * h_rician) + noise_scaled

        # 4. Tam Kanal (Fading + AWGN + Phase Rotation / Doppler)
        time_array = np.arange(num_symbols) * (1 / subcarrier_spacing)
        phase_rotation = np.exp(1j * 2 * np.pi * doppler_hz * time_array)
        rx_symbols_full_channel = (qpsk_symbols * h_rician * phase_rotation) + noise_scaled

        self.status_var.set(f"Durum: BAĞLANTI AKTİF | Elev: {elev_deg:.1f}° | Mesafe: {dist_km:.0f} km | SNR: {snr_mean_db:.1f} dB")

        # ==========================================
        # GRAFİK ÇİZİMLERİ (3x3 Grid)
        # ==========================================
        # 1. Satır: Map, Path Loss, Doppler Curve
        ax_map = self.fig.add_subplot(3, 3, 1, projection=ccrs.PlateCarree())
        ax_ls = self.fig.add_subplot(3, 3, 2)
        ax_dyn = self.fig.add_subplot(3, 3, 3)
        
        # 2. Satır: Shadowing, Rician, Total Loss Curve
        ax_shad = self.fig.add_subplot(3, 3, 4)
        ax_ric = self.fig.add_subplot(3, 3, 5)
        ax_dop_curve = self.fig.add_subplot(3, 3, 6)

        # 3. Satır: CONSTELLATION DİYAGRAMLARI (SİNYAL KALİTESİ)
        ax_const_clean = self.fig.add_subplot(3, 3, 7)
        ax_const_faded = self.fig.add_subplot(3, 3, 8)
        ax_const_full = self.fig.add_subplot(3, 3, 9)

        # --- Çizim İşlemleri ---
        subp = wgs84.subpoint(satellite.at(sim_time))
        ax_map.add_feature(cfeature.LAND); ax_map.add_feature(cfeature.COASTLINE)
        ax_map.set_extent([20, 50, 30, 50], crs=ccrs.PlateCarree())
        ax_map.plot(32.8541, 39.9208, 'r^', markersize=10, transform=ccrs.PlateCarree())
        ax_map.plot(subp.longitude.degrees, subp.latitude.degrees, 'bo', markersize=8, transform=ccrs.PlateCarree())
        ax_map.plot([32.8541, subp.longitude.degrees], [39.9208, subp.latitude.degrees], 'g--', transform=ccrs.Geodetic())
        ax_map.set_title("Geometri (Ankara - Uydu)")

        ax_ls.bar(['FSPL', 'Atmosferik'], [fspl_db, atm_loss_db], color=['blue', 'orange'])
        ax_ls.set_title(f"Büyük Ölçekli Kayıp: {total_ls_loss:.1f} dB")
        
        ax_dyn.plot(total_dynamic[:100], marker='.', color='teal')
        ax_dyn.axhline(total_ls_loss, color='black', linestyle='--')
        ax_dyn.set_title("Zaman/Paket Başına Toplam Kayıp (dB)")

        ax_shad.hist(shadowing_db, bins=20, color='purple', alpha=0.7)
        ax_shad.set_title(f"Lognormal Gölgeleme (\u03c3={sigma_shad:.1f}dB)")

        ax_ric.hist(rician_db, bins=20, color='green', alpha=0.7)
        ax_ric.set_title(f"Rician Sönümleme (K={k_factor_db:.1f}dB)")

        ax_dop_curve.bar(['Doppler (Hz)'], [abs(doppler_hz)], color='red')
        ax_dop_curve.axhline(3750, color='gray', linestyle='--', label='SCS Sınırı (3.75 kHz)')
        ax_dop_curve.set_title(f"Anlık Doppler Kayması: {doppler_hz:.0f} Hz")
        ax_dop_curve.legend()

        # --- Constellation Diyagramları (En Alt Satır) ---
        def format_constellation(ax, title):
            ax.set_title(title)
            ax.set_xlabel('In-Phase (I)')
            ax.set_ylabel('Quadrature (Q)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)

        ax_const_clean.scatter(qpsk_symbols.real, qpsk_symbols.imag, color='blue', alpha=0.5, s=10)
        format_constellation(ax_const_clean, '1. İletilen QPSK Sinyali (Tx)')

        ax_const_faded.scatter(rx_symbols_fading_awgn.real, rx_symbols_fading_awgn.imag, color='orange', alpha=0.5, s=10)
        format_constellation(ax_const_faded, f'2. Alınan Sinyal (SNR: {snr_mean_db:.1f} dB)\n(Sadece Fading + AWGN)')

        ax_const_full.scatter(rx_symbols_full_channel.real, rx_symbols_full_channel.imag, color='red', alpha=0.5, s=10)
        format_constellation(ax_const_full, f'3. Tam NTN Kanalı Sinyali\n(+ {doppler_hz:.0f} Hz Doppler Etkisi)')

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

# Uygulamayı Başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = NTNSimulatorApp(root)
    root.mainloop()