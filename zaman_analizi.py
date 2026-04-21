import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec

# ==========================================
# ITU-R Kütüphanesi Yükleme Denemesi
# ==========================================
try:
    import itur
    import itur.models.itu525 as itu525
    import itur.models.itu676 as itu676
    ITUR_AVAILABLE = True
    print("[BİLGİ] 'itur' kütüphanesi başarıyla yüklendi. Hesaplamalar ITU-R modelleri ile yapılacak.")
except ImportError:
    print("---------------------------------------------------------")
    print("[UYARI] 'itur' kütüphanesi bulunamadı!")
    print("Hesaplamalar matematiksel formüller (channel.py standardı) ile yapılacaktır.")
    print("ITU-R modellerini aktif etmek için terminalden şu komutu çalıştırın:")
    print("pip install itur")
    print("---------------------------------------------------------")
    ITUR_AVAILABLE = False

# ==========================================
# BAŞLANGIÇ: ZAMAN VE UYDU TANIMLAMA
# ==========================================
ts = load.timescale()
t_now = ts.now()

# channel.py'deki TLE verileri
tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

# Uydu ve yer istasyonunu oluştur
satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
station_lat = 39.9208
station_lon = 32.8541
ground_station = wgs84.latlon(station_lat, station_lon)

# --- UYDU GEÇİŞİNİ BULMA (ZAMAN ARALIĞI) ---
t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
pass_times, pass_events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=10.0)

rise_indices = [i for i, event in enumerate(pass_events) if event == 0]

# İlk bulduğumuz tam uydu geçişini (doğuş ve batış) hesaplama aralığı yapıyoruz
if len(rise_indices) > 0:
    idx = rise_indices[0] # İlk geçiş
    t_start = pass_times[idx]
    
    if idx + 2 < len(pass_times):
        t_end = pass_times[idx + 2] # Batış (set) anı
    else:
        t_end = ts.from_datetime(t_start.utc_datetime() + timedelta(minutes=10))
else:
    # Geçiş bulunamadıysa varsayılan 10 dakikalık bir aralık oluştur
    print("[BİLGİ] Uydu geçişi bulunamadı. Varsayılan 10 dakikalık aralık kullanılıyor.")
    t_start = t_now
    t_end = ts.from_datetime(t_now.utc_datetime() + timedelta(minutes=10))

# Toplam saniye kadar bir zaman array'i oluştur
total_seconds = int((t_end.utc_datetime() - t_start.utc_datetime()).total_seconds())
print(f"[BİLGİ] Geçiş Süresi: {total_seconds} saniye")
print(f"[BİLGİ] Başlangıç: {t_start.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"[BİLGİ] Bitiş:     {t_end.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ==========================================
# HESAPLAMA DONGUSU (ZAMAN SERİSİ)
# ==========================================
# İlgili değişkenler
carrier_freq_mhz = 2000
carrier_freq_ghz = carrier_freq_mhz / 1000.0
carrier_freq_hz = carrier_freq_mhz * 1e6
c_km_s = 299792.458

# Link Budget (Bağlantı Bütçesi) sabitleri
tx_power_dbm = 30.0 # Uydu gönderim gücü (Örn: 30 dBm = 1 Watt)
tx_gain_dbi = 20.0  # Uydu anten kazancı
rx_gain_dbi = 0.0   # Telsiz/Cihaz anten kazancı
rx_sensitivity_dbm = -125.0 # Telsiz alma hassasiyeti (NB-IoT için tipik)

# Sonuç dökümleri
time_minutes_list = []
elevation_list = []
distance_list = []
doppler_list = []
fspl_list = []
total_path_loss_list = []
k_factor_list = []
link_budget_list = []

for s in range(total_seconds):
    current_t = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=s))
    
    # 1. Geometri ve Elevasyon/Mesafe
    diff = satellite - ground_station
    alt, az, dist = diff.at(current_t).altaz()
    el_deg = alt.degrees
    d_km = dist.km
    
    # Elevasyon < 0 ise veriyi kaydetme, sadece görünen kısımları al
    if el_deg < 0:
        continue
    
    # Dakika cinsinden zamanı kaydet (başlangıç 0 olacak şekilde)
    time_minutes_list.append(s / 60.0)
    elevation_list.append(el_deg)
    distance_list.append(d_km)
    
    # 2. Doppler Kayması
    t_next = ts.from_datetime(current_t.utc_datetime() + timedelta(seconds=1))
    d_next = diff.at(t_next).distance().km
    v_rel = d_km - d_next # Yaklaşıyorsa mesafe azalır, hız (+) çıkar
    doppler_hz = (v_rel / c_km_s) * carrier_freq_hz
    doppler_list.append(doppler_hz)
    
    # 3. Path Loss & Atmosferik Kayıp (ITU-R)
    fspl = 0
    atm_loss = 0
    if ITUR_AVAILABLE:
        try:
            # ITU-R P.525 Free Space Loss
            # (Eğer model object dönerse `.value` ile çek, sayıysa direkt kullan)
            val = itu525.free_space_loss(d_km, carrier_freq_ghz)
            fspl = float(val.value) if hasattr(val, 'value') else float(val)
        except Exception:
            fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
            
        try:
            # ITU-R P.676 Slant Path Gaseous Attenuation
            # rho=7.5 g/m3 (su buharı), P=1013.25 hPa (basınç), T=15 C (Sıcaklık)
            val_atm = itu676.gaseous_attenuation_slant_path(carrier_freq_ghz, el_deg, 7.5, 1013.25, 15.0)
            atm_loss = float(val_atm.value) if hasattr(val_atm, 'value') else float(val_atm)
        except Exception:
            atm_loss = 0.5 / np.sin(np.radians(max(el_deg, 1.0)))
    else:
        # channel.py modeline fallback
        fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
        atm_loss = 0.5 / np.sin(np.radians(max(el_deg, 1.0)))
        
    total_loss = fspl + atm_loss
    fspl_list.append(fspl)
    total_path_loss_list.append(total_loss)
    
    # 4. Rician K Değeri
    # channel.py'deki dağılıma göre
    k_db = min(15.0, 2.0 + ((el_deg - 10) / 80) * 13.0) 
    k_factor_list.append(k_db)
    
    # 5. Link Bütçesi
    # Received Power (dBm) = TxPower(dBm) + TxGain(dBi) + RxGain(dBi) - TotalPathLoss(dB)
    rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
    link_budget_list.append(rx_power)


# ==========================================
# GRAFİKLERİ ÇİZME
# ==========================================
# Verileri numpy dizisine çevirme
time_arr = np.array(time_minutes_list)
el_arr = np.array(elevation_list)
dist_arr = np.array(distance_list)
dop_arr = np.array(doppler_list)
fspl_arr = np.array(fspl_list)
loss_arr = np.array(total_path_loss_list)
k_arr = np.array(k_factor_list)
lb_arr = np.array(link_budget_list)

fig = plt.figure(figsize=(16, 12))
fig.canvas.manager.set_window_title('NTN Zaman Serisi ve Senaryo Analizi (ITU-R Destekli)')
gs = GridSpec(3, 2, figure=fig)

# -- 1. Grafik: Path Loss vs Zaman --
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_arr, loss_arr, label='Toplam Path Loss (FSPL + Atmosferik)', color='red', linewidth=2)
ax1.plot(time_arr, fspl_arr, linestyle='--', label='Yalnızca FSPL', color='orange')
ax1.set_title('1. Zaman Aralığı ile Path Loss')
ax1.set_xlabel('Zaman (Dakika)')
ax1.set_ylabel('Kayıp (dB)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# -- 2. Grafik: Elevasyon ve Mesafe vs Zaman --
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Zaman (Dakika)')
ax2.set_ylabel('Elevasyon (Derece)', color='tab:blue')
ax2.plot(time_arr, el_arr, color='tab:blue', linewidth=2, label='Elevasyon')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.grid(True, linestyle='--', alpha=0.7)

ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('Mesafe / Slant Range (km)', color='tab:green')
ax2_twin.plot(time_arr, dist_arr, linestyle='-.', color='tab:green', linewidth=2, label='Mesafe')
ax2_twin.tick_params(axis='y', labelcolor='tab:green')
plt.title('2. Zaman Aralığı ile Elevasyon ve Mesafe')

# -- 3. Grafik: Doppler vs Zaman --
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_arr, dop_arr, color='purple', linewidth=2)
ax3.set_title('3. Zamana Bağlı Doppler Kayması')
ax3.set_xlabel('Zaman (Dakika)')
ax3.set_ylabel('Doppler Frekansı (Hz)')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.grid(True, linestyle='--', alpha=0.7)

# -- 4. Grafik: K Faktörü vs Zaman --
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_arr, k_arr, color='brown', linewidth=2)
ax4.set_title('4. Zamana Bağlı Rician K-Faktörü')
ax4.set_xlabel('Zaman (Dakika)')
ax4.set_ylabel('K Değeri (dB)')
ax4.grid(True, linestyle='--', alpha=0.7)

# -- 5. Grafik: Link Bütçesi (Received Power) vs Zaman --
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(time_arr, lb_arr, color='teal', linewidth=2, label='Anlık Alınan Sinyal Gücü (Received Power)')
ax5.axhline(rx_sensitivity_dbm, color='red', linestyle='--', linewidth=2, label=f'RX Hassasiyeti ({rx_sensitivity_dbm} dBm)')

# Grafiğin arka planına bağlantı koptu / devam ediyor uyarısı koyma (Opsiyonel görselleştirme)
ax5.fill_between(time_arr, -150, rx_sensitivity_dbm, color='red', alpha=0.1, label='Bağlantı Yok (Outage)')
ax5.fill_between(time_arr, rx_sensitivity_dbm, np.max(lb_arr)+5, color='green', alpha=0.1, label='Bağlantı Var (Link Active)')

ax5.set_title('5. Zamana Bağlı Link Bütçesi (Alınan Güç)')
ax5.set_xlabel('Zaman (Dakika)')
ax5.set_ylabel('Sinyal Gücü (dBm)')
ax5.set_ylim(-150, np.max(lb_arr)+5)
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.legend(loc='lower right')

plt.tight_layout()
plt.show()
