import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skyfield.api import load, EarthSatellite, wgs84
import numpy as np
from datetime import timedelta
import random

# ==========================================
# AYARLAR (SETTINGS)
# ==========================================
# Tüm grafikleri tek bir büyük pencerede (dashboard) görmek için True,
# Ayrı ayrı pencerelerde (eski usül) görmek için False yapın.
SHOW_ALL_IN_ONE_WINDOW = True 

# ==========================================
# BAŞLANGIÇ: ZAMAN VE UYDU TANIMLAMA
# ==========================================
# Uzay hesaplamalarında zaman (UTC) çok kritik olduğu için timescale yüklüyoruz.
ts = load.timescale()
t_now = ts.now() 

# Uydunun yörünge parametrelerini belirten TLE (Two-Line Element) verisi
# Bu veri uydunun uzaydaki kimlik kartıdır.
tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

# TLE verisinden bir uydu objesi türetiyoruz
satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)

# Yer istasyonunun WGS84 formatında koordinatları (Ankara)
ground_station = wgs84.latlon(39.9208, 32.8541) 

# --- ZAMANI İLERİ SARMA (GERÇEK ZAMANLI GEÇİŞ SİMÜLASYONU) ---
# Önümüzdeki 7 gün içindeki uydu geçişlerini bulmak için bir zaman aralığı belirliyoruz
t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))

# Yer istasyonu üzerinden elevasyonun en az 10 derece olduğu olayları bul
# Olay tipleri: 0 (Ufuktan Doğuş), 1 (Tepe Noktası), 2 (Ufuktan Batış)
pass_times, pass_events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=10.0)

# Sadece ufuktan doğuş (0) anlarının listesini al
rise_indices = [i for i, event in enumerate(pass_events) if event == 0]

if len(rise_indices) > 0:
    # Bulunan geçişler arasından tamamen rastgele bir tanesini seç (Farklı senaryolar görmek için)
    selected_rise_idx = random.choice(rise_indices)
    
    # Seçilen geçişin başlangıç (doğuş) ve bitiş (batış) zamanlarını al
    if selected_rise_idx + 2 < len(pass_times):
        t_start = pass_times[selected_rise_idx]
        t_end = pass_times[selected_rise_idx + 2] 
        
        # Geçişin saniye cinsinden toplam süresini bul
        pass_duration = (t_end.utc_datetime() - t_start.utc_datetime()).total_seconds()
        
        # Geçişin ortasından rastgele bir an (saniye) seç (Doppler ve mesafe sürekli değişsin diye)
        random_second = random.uniform(0, pass_duration)
        
        # Seçilen rastgele saniyeyi başlangıç zamanına ekle ve simülasyon anını (current_time) belirle
        current_time = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=random_second))
        
        print(f"\n--- YENİ SENARYO ÜRETİLDİ (GEÇİŞ İÇİNDEN RASTGELE AN) ---")
        print(f"Seçilen An: {current_time.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
else:
    # Eğer 7 gün içinde hiç geçiş yoksa hata vermemesi için manuel bir tarih ata
    print("\nGeçiş bulunamadı, varsayılan zaman atanıyor.")
    current_time = ts.utc(2026, 4, 15, 12, 0, 0)

# ==========================================
# GRAFİK PENCERESİ AYARLAMALARI
# ==========================================
if SHOW_ALL_IN_ONE_WINDOW:
    # Tüm grafikler için devasa bir figür oluştur (Grid sistemi: 3 satır, 3 sütun)
    fig = plt.figure(figsize=(20, 15))
    fig.canvas.manager.set_window_title('NTN NB-IoT Kanal Simülatörü - Kontrol Paneli')
    
    # Alt grafikleri (Axes) tanımlama (Harita için özel projection ekleniyor)
    ax_map = fig.add_subplot(3, 3, 1, projection=ccrs.PlateCarree())
    
    ax_ls_bar = fig.add_subplot(3, 3, 2)
    ax_ls_fspl = fig.add_subplot(3, 3, 3)
    ax_ls_atm = fig.add_subplot(3, 3, 4)
    
    ax_dop_bar = fig.add_subplot(3, 3, 5)
    ax_dop_curve = fig.add_subplot(3, 3, 6)
    
    ax_ss_shad = fig.add_subplot(3, 3, 7)
    ax_ss_ric = fig.add_subplot(3, 3, 8)
    ax_ss_total = fig.add_subplot(3, 3, 9)
else:
    # Grafikleri ayrı pencerelerde açmak için farklı figürler oluştur
    fig_map = plt.figure(figsize=(8, 6))
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    fig_ls, (ax_ls_bar, ax_ls_fspl, ax_ls_atm) = plt.subplots(1, 3, figsize=(18, 5))
    fig_dop, (ax_dop_bar, ax_dop_curve) = plt.subplots(1, 2, figsize=(14, 5))
    fig_ss, (ax_ss_shad, ax_ss_ric, ax_ss_total) = plt.subplots(1, 3, figsize=(18, 5))

# ==========================================
# 1. ADIM: GEOMETRİ VE GÖRSELLEŞTİRME
# ==========================================
def calculate_geometry(sat, station, time):
    """Uydu ve yer istasyonu arasındaki anlık mesafeyi ve elevasyon açısını hesaplar."""
    # Uydu konumu ile istasyon konumu arasındaki 3D fark vektörü
    diff_vector = sat - station
    # Bu vektörü belirtilen zamana göre gökyüzü koordinatlarına (Topocentric) çevirir
    sky_position = diff_vector.at(time)
    alt, az, distance = sky_position.altaz()
    # Elevasyon (altitude) derece, mesafe km olarak döner
    return alt.degrees, distance.km

# Tanımladığımız fonksiyonu o anki zaman için çalıştır
elevation_angle, distance_km = calculate_geometry(satellite, ground_station, current_time)

print(f"Elevasyon Açısı: {elevation_angle:.2f} derece")
print(f"Mesafe (Slant Range): {distance_km:.2f} km")

# Uydunun Dünya üzerindeki izdüşüm koordinatlarını hesapla
subpoint = wgs84.subpoint(satellite.at(current_time))
sat_lat = subpoint.latitude.degrees
sat_lon = subpoint.longitude.degrees
station_lat = 39.9208 
station_lon = 32.8541

# --- Harita Çizimi ---
ax_map.add_feature(cfeature.LAND)
ax_map.add_feature(cfeature.OCEAN)
ax_map.add_feature(cfeature.COASTLINE)
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.set_extent([20, 50, 30, 50], crs=ccrs.PlateCarree()) # Harita sınırları

# İstasyonu ve uyduyu işaretle
ax_map.plot(station_lon, station_lat, color='red', marker='^', markersize=10, transform=ccrs.PlateCarree(), label='Ankara')
ax_map.plot(sat_lon, sat_lat, color='blue', marker='o', markersize=8, transform=ccrs.PlateCarree(), label='Uydu İz Düşümü')
ax_map.plot([station_lon, sat_lon], [station_lat, sat_lat], color='green', linestyle='--', transform=ccrs.Geodetic(), label='Sinyal Yolu')

ax_map.set_title(f'NTN Geometrisi\nMesafe: {distance_km:.1f} km, Elev: {elevation_angle:.1f}°')
ax_map.legend(loc='lower left')

# ==========================================
# 2. ADIM: BÜYÜK ÖLÇEKLİ KAYIP (LARGE SCALE PATH LOSS)
# ==========================================
def calculate_large_scale_loss(dist_km, elev_deg, freq_mhz):
    """Mesafe ve açıya bağlı olarak Serbest Uzay (FSPL) ve Atmosferik kayıpları hesaplar."""
    if elev_deg <= 0:
        return float('inf'), float('inf'), float('inf')

    # FSPL Formülü: 32.44 + 20*log(d_km) + 20*log(f_MHz)
    fspl_db = 32.44 + 20 * np.log10(dist_km) + 20 * np.log10(freq_mhz)
    
    # Atmosferik Kayıp Formülü: L_zenith / sin(elevasyon_radyan)
    elev_rad = np.radians(elev_deg)
    l_zenith = 0.5 # S-Bandı için tam tepe noktasındaki referans kayıp
    atm_loss_db = l_zenith / np.sin(elev_rad)
    
    total_loss_db = fspl_db + atm_loss_db
    return fspl_db, atm_loss_db, total_loss_db

# NTN NB-IoT için taşıyıcı frekans (S-Bandı)
carrier_freq_mhz = 2000 
fspl, atm_loss, total_ls_loss = calculate_large_scale_loss(distance_km, elevation_angle, carrier_freq_mhz)

print(f"\n--- 2. ADIM: KANAL KAYIP ANALİZİ ---")
print(f"Kullanılan Frekans: {carrier_freq_mhz} MHz")
print(f"Serbest Uzay Kaybı (FSPL): {fspl:.2f} dB")
print(f"Atmosferik Kayıp: {atm_loss:.2f} dB")
print(f"Toplam Büyük Ölçekli Kayıp: {total_ls_loss:.2f} dB")

# --- Kayıp Grafikleri ---
# 1. Bar Grafiği
loss_labels = ['FSPL\n(Serbest Uzay)', 'Atmosferik\nKayıp']
loss_values = [fspl, atm_loss]
ax_ls_bar.bar(loss_labels, loss_values, color=['#1f77b4', '#ff7f0e'])
ax_ls_bar.set_title(f'Anlık Path Loss Durumu')
ax_ls_bar.set_ylabel('Kayıp (dB)')
for i, v in enumerate(loss_values):
    ax_ls_bar.text(i, v + 2, f"{v:.1f} dB", ha='center', fontweight='bold')
ax_ls_bar.set_ylim(0, fspl * 1.2) 

# 2. Teorik FSPL Eğrisi
theoretical_distances = np.linspace(500, 3000, 100)
theoretical_fspl = 32.44 + 20 * np.log10(theoretical_distances) + 20 * np.log10(carrier_freq_mhz)
ax_ls_fspl.plot(theoretical_distances, theoretical_fspl, color='blue', linewidth=2)
ax_ls_fspl.plot(distance_km, fspl, 'ro', markersize=8, label='Bizim Uydumuz')
ax_ls_fspl.set_title('FSPL vs Mesafe')
ax_ls_fspl.set_xlabel('Mesafe (km)')
ax_ls_fspl.set_ylabel('Serbest Uzay Kaybı (dB)')
ax_ls_fspl.grid(True, linestyle='--', alpha=0.7)
ax_ls_fspl.legend()

# 3. Teorik Atmosferik Kayıp Eğrisi
theoretical_angles = np.linspace(10, 90, 100)
l_zenith_ref = 0.5 
theoretical_atm = l_zenith_ref / np.sin(np.radians(theoretical_angles))
ax_ls_atm.plot(theoretical_angles, theoretical_atm, color='orange', linewidth=2)
ax_ls_atm.plot(elevation_angle, atm_loss, 'ro', markersize=8, label='Bizim Uydumuz')
ax_ls_atm.set_title('Atmosferik Kayıp vs Elevasyon')
ax_ls_atm.set_xlabel('Elevasyon Açısı (Derece)')
ax_ls_atm.set_ylabel('Atmosferik Kayıp (dB)')
ax_ls_atm.grid(True, linestyle='--', alpha=0.7)
ax_ls_atm.legend()

# ==========================================
# 3. ADIM: DOPPLER HESAPLAMASI
# ==========================================
# Bağıl hızı bulmak için 1 saniye sonraki uydunun konumunu hesaplıyoruz
time_next_sec = ts.from_datetime(current_time.utc_datetime() + timedelta(seconds=1))
dist_now = (satellite - ground_station).at(current_time).distance().km
dist_next = (satellite - ground_station).at(time_next_sec).distance().km

# Yaklaşıyorsa sonraki mesafe daha küçüktür, sonuç pozitif (m/s cinsinden) çıkar.
relative_velocity_km_s = dist_now - dist_next 
c_km_s = 299792.458 # Işık hızı
carrier_freq_hz = carrier_freq_mhz * 1e6

# Doppler Formülü: f_d = (v_rel / c) * f_c
doppler_shift_hz = (relative_velocity_km_s / c_km_s) * carrier_freq_hz

print(f"\n--- 3. ADIM: DOPPLER ANALİZİ ---")
print(f"Bağıl Hız (Sana doğru): {relative_velocity_km_s*1000:.2f} m/s")
print(f"Anlık Doppler Kayması: {doppler_shift_hz:.2f} Hz")

# --- Doppler Grafikleri ---
# Geçişin -5 ve +5 dakikası arasındaki teorik Doppler S-Eğrisini hesapla
minutes_array = np.linspace(-5, 5, 100)
times_array = [ts.from_datetime(current_time.utc_datetime() + timedelta(minutes=dk)) for dk in minutes_array]
theoretical_doppler = []

for z in times_array:
    z_next = ts.from_datetime(z.utc_datetime() + timedelta(seconds=1))
    d1 = (satellite - ground_station).at(z).distance().km
    d2 = (satellite - ground_station).at(z_next).distance().km
    v_rel = d1 - d2
    theoretical_doppler.append((v_rel / c_km_s) * carrier_freq_hz)

# 1. NB-IoT Alt Taşıyıcı Aralıkları ile Kıyaslama (Bar Grafiği)
scs_15kHz = 15000
scs_3_75kHz = 3750
abs_doppler = abs(doppler_shift_hz)

ax_dop_bar.bar(['Anlık Doppler', 'NB-IoT SCS\n(3.75 kHz)', 'NB-IoT SCS\n(15 kHz)'], 
               [abs_doppler, scs_3_75kHz, scs_15kHz], color=['red', 'gray', 'gray'])
ax_dop_bar.set_ylabel('Frekans (Hz)')
ax_dop_bar.set_title('Doppler vs NB-IoT Subcarrier Spacing')
ax_dop_bar.text(0, abs_doppler + 1000, f"{abs_doppler:.0f} Hz", ha='center', color='red', fontweight='bold')
ax_dop_bar.grid(axis='y', linestyle='--', alpha=0.7)

# 2. S-Eğrisi (Doppler Değişimi)
ax_dop_curve.plot(minutes_array, theoretical_doppler, 'b-', linewidth=2)
ax_dop_curve.plot(0, doppler_shift_hz, 'ro', markersize=8, label='Şu Anki Durum')
ax_dop_curve.axhline(0, color='black', linewidth=1, linestyle='--') 
ax_dop_curve.set_xlabel('Zaman (Dakika) | 0 = Şimdiki an')
ax_dop_curve.set_ylabel('Doppler Kayması (Hz)')
ax_dop_curve.set_title('Doppler S-Eğrisi')
ax_dop_curve.grid(True, alpha=0.5)
ax_dop_curve.legend()

# ==========================================
# 4. ADIM: KÜÇÜK ÖLÇEKLİ KAYIPLAR (SMALL SCALE FADING & SHADOWING)
# ==========================================
def determine_channel_parameters(elev):
    """Elevasyon açısına bağlı istatistiksel kanal model parametrelerini belirler."""
    # Elevasyon düşükse gölgeleme fazladır (Max 8 dB), yüksekse azdır (Min 2 dB)
    std_dev_shadowing = max(2.0, 8.0 - ((elev - 10) / 80) * 6.0)
    
    # Elevasyon düşükse Rician LoS gücü zayıftır (2 dB), yüksekse güçlüdür (15 dB)
    k_factor_in_db = min(15.0, 2.0 + ((elev - 10) / 80) * 13.0)
    
    return std_dev_shadowing, k_factor_in_db

sigma_shadowing_db, k_factor_db = determine_channel_parameters(elevation_angle)
k_linear = 10 ** (k_factor_db / 10) # Formüllerde kullanmak için lineer orana çevir

print(f"\n--- 4. ADIM: KÜÇÜK ÖLÇEKLİ KAYIP PARAMETRELERİ ---")
print(f"Gölgeleme (Shadowing) Std. Sapma (\u03c3): {sigma_shadowing_db:.2f} dB")
print(f"Rician K-Faktörü: {k_factor_db:.2f} dB (Lineer Oran: {k_linear:.2f})")

# Zaman/Paket bazında sinyalin maruz kalacağı rastgelelikleri üretiyoruz
num_samples = 1000

# Lognormal Gölgeleme (Bina/Ağaç arkasında kalma)
shadowing_array_db = np.random.normal(loc=0, scale=sigma_shadowing_db, size=num_samples)

# Rician Fading (Çok yollu sönümleme)
# Doğrudan LoS bileşeni
los_component = np.sqrt(k_linear / (k_linear + 1))
# Etraftan yansıyan NLoS bileşenleri (Kompleks düzlemde Gauss dağılımı)
nlos_x = np.random.normal(loc=0, scale=np.sqrt(1/(2*(k_linear+1))), size=num_samples)
nlos_y = np.random.normal(loc=0, scale=np.sqrt(1/(2*(k_linear+1))), size=num_samples)

# Toplam Kompleks Rician Kanalı (h) ve dB cinsinden kazanç/kayıp dönüşümü
h_rician = los_component + (nlos_x + 1j * nlos_y)
rician_fading_array_db = 10 * np.log10(np.abs(h_rician)**2)

# Genel Toplam Dinamik Kanal Kaybı
# FSPL sabit + Gölgeleme değişimi - Çok Yollu yansıma kazancı/kaybı
total_dynamic_loss = total_ls_loss + shadowing_array_db - rician_fading_array_db

# --- Küçük Ölçekli Kayıp Grafikleri ---
# 1. Gölgeleme Histogramı
ax_ss_shad.hist(shadowing_array_db, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
ax_ss_shad.set_title(f'Lognormal Shadowing (\u03c3 = {sigma_shadowing_db:.1f} dB)')
ax_ss_shad.set_xlabel('Gölgeleme Etkisi (dB)')
ax_ss_shad.set_ylabel('Olasılık Yoğunluğu')
ax_ss_shad.grid(True, linestyle='--', alpha=0.5)

# 2. Rician Histogramı
ax_ss_ric.hist(rician_fading_array_db, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
ax_ss_ric.set_title(f'Rician Fading (K = {k_factor_db:.1f} dB)')
ax_ss_ric.set_xlabel('Sinyal Kazancı/Kaybı (dB)')
ax_ss_ric.set_ylabel('Olasılık Yoğunluğu')
ax_ss_ric.axvline(0, color='red', linestyle='--', label='Ortalama')
ax_ss_ric.grid(True, linestyle='--', alpha=0.5)
ax_ss_ric.legend()

# 3. Toplam Kayıp Dalgalanması
ax_ss_total.plot(total_dynamic_loss[:100], marker='o', linestyle='-', color='teal', markersize=4)
ax_ss_total.axhline(total_ls_loss, color='black', linewidth=2, linestyle='--', label=f'Statik FSPL ({total_ls_loss:.1f} dB)')
ax_ss_total.set_title('Zaman İçinde Toplam Kanal Kaybı')
ax_ss_total.set_xlabel('Simülasyon İndeksi (Paket No)')
ax_ss_total.set_ylabel('Toplam Kayıp (dB)')
ax_ss_total.grid(True, linestyle='--', alpha=0.5)
ax_ss_total.legend()

# ==========================================
# SİMÜLASYONU BİTİR VE GÖSTER
# ==========================================
if SHOW_ALL_IN_ONE_WINDOW:
    # h_pad ve w_pad değerleri ile grafikler arası boşlukları açıyoruz
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.0)
    fig.subplots_adjust(top=0.92)
else:
    plt.tight_layout()

plt.show()