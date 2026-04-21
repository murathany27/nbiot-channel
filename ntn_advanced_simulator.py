import numpy as np
from datetime import datetime, timedelta
import collections
import warnings

# ==========================================
# GEREKLİ KÜTÜPHANELER
# ==========================================
# Uydu ve Yörünge Fiziği
import ephem
from sgp4.api import Satrec, SGP4_ERRORS
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, EarthLocation, CartesianRepresentation, CartesianDifferential

# ITU-R Zayıflama Modelleri
from itur.models.itu676 import gaseous_attenuation_slant_path
from itur.models.itu618 import rain_attenuation, scintillation_attenuation
from itur.models.itu840 import cloud_attenuation

# Görselleştirme
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Astropy uyarılarını konsolu kirletmemesi için gizliyoruz
warnings.filterwarnings('ignore')

# ==========================================
# 1. PARAMETRE SETİ (Kullanıcı + Bizim Parametreler)
# ==========================================
c = 299792458.0 # Işık hızı (m/s)

# Yer İstasyonu (Ankara Örneği)
gs_lat_deg = 39.9
gs_lon_deg = 32.8
gs_h_km = 0.850 # 850m rakım

station = ephem.Observer()
station.lat, station.lon = str(gs_lat_deg), str(gs_lon_deg)
station.elevation = gs_h_km * 1000.0

# Frekans ve Hava Durumu İstatistikleri
f_ghz = 3.0
f_hz = f_ghz * 1e9

T_k = 293.15      # Sıcaklık (Kelvin)
P_hpa = 1013.25   # Basınç (hPa)
rho_gm3 = 7.5     # Su buharı yoğunluğu
p = 0.01          # Yıllık olasılık
D = 0.6           # Anten Çapı (m) - Sintilasyon

# Link Bütçesi (Örnek NB-IoT NTN Değerleri)
tx_power_dbm = 30.0    # Uydu Gönderim Gücü (1 Watt)
tx_gain_dbi = 20.0     # Uydu Anten Kazancı
rx_gain_dbi = 0.0      # Alıcı (Telsiz) Anteni
rx_sensitivity_dbm = -125.0 # Telsizin koptuğu alt sınır

# Uydu Bilgileri (TLE)
tle_l1 = "1 64555U 25135AD  26106.74795786  .00001933  00000+0  92239-4 0  9993"
tle_l2 = "2 64555  97.4610 222.4026 0001997 124.5460 235.5964 15.20264790 45526"

sat = ephem.readtle("CONNECTA IOT-10", tle_l1, tle_l2)
sat_sgp4 = Satrec.twoline2rv(tle_l1, tle_l2)

# ==========================================
# 2. FİZİKSEL KATMAN VE KANAL FONKSİYONLARI
# ==========================================
state_names = {0: "LoS (Rician)", 1: "Shadowing (Log-Normal)", 2: "Blocked (Rayleigh)"}

# Gölgeleme Değişkenleri
shadow_corr_db = 0.0
shadow_rho = 0.985
num_paths = 3
initial_path_phase = 2 * np.pi * np.random.rand(num_paths)

def get_small_scale_loss(state):
    if state == 0:
        K = 10**(10/10)
        r = np.sqrt(K/(K+1)) + np.sqrt(1/(K+1)) * (np.random.normal(0,1) + 1j*np.random.normal(0,1)) / np.sqrt(2)
        return -20 * np.log10(np.abs(r))
    elif state == 1:
        r = (np.random.normal(0,1) + 1j*np.random.normal(0,1)) / np.sqrt(2)
        return -20 * np.log10(np.abs(r)) + 4.0
    else:
        r = (np.random.normal(0,1) + 1j*np.random.normal(0,1)) / np.sqrt(2)
        return -20 * np.log10(np.abs(r)) + 20.0

def get_transition_matrix(theta):
    theta_clamped = max(0, min(theta, 90))
    p_stay_los = 0.5 + 0.45 * (theta_clamped / 90)
    p_stay_blocked = 0.9 - 0.4 * (theta_clamped / 90)
    P = np.array([
        [p_stay_los, (1 - p_stay_los) * 0.8, (1 - p_stay_los) * 0.2],
        [0.3, 0.4, 0.3],
        [1 - p_stay_blocked, 0.0, p_stay_blocked]
    ])
    return P / P.sum(axis=1, keepdims=True)

def update_correlated_shadow(prev_shadow_db, theta, state):
    if state == 1:
        mu = 8.0 - (max(theta, 0.0) / 90.0) * 4.0
        innovation = np.random.normal(0.0, 3.0) # Sigma = 3.0
        new_shadow = shadow_rho * prev_shadow_db + np.sqrt(1 - shadow_rho**2) * innovation
        return mu + new_shadow
    else:
        return 0.0

def get_path_profile(state):
    if state == 0:
        excess_delays_s = np.array([0.0, 80e-9, 220e-9])
        rel_powers_db = np.array([0.0, -8.0, -15.0])
    elif state == 1:
        excess_delays_s = np.array([0.0, 120e-9, 350e-9])
        rel_powers_db = np.array([0.0, -5.0, -9.0])
    else:
        excess_delays_s = np.array([0.0, 180e-9, 500e-9])
        rel_powers_db = np.array([0.0, -3.0, -6.0])
    rel_powers_lin = 10**(rel_powers_db / 10.0)
    rel_powers_lin = rel_powers_lin / np.sum(rel_powers_lin)
    return excess_delays_s, rel_powers_lin

def get_path_dopplers(theta, base_doppler_hz):
    theta_clamped = max(0.0, min(theta, 90.0))
    spread_scale = (90.0 - theta_clamped) / 90.0
    max_extra_spread_hz = 250.0 * spread_scale
    offsets = np.array([0.0, max_extra_spread_hz, -max_extra_spread_hz])
    return base_doppler_hz + offsets

def build_tdl_channel(state, theta, base_doppler_hz, t_rel_s, propagation_delay_s):
    excess_delays_s, rel_powers_lin = get_path_profile(state)
    path_dopplers_hz = get_path_dopplers(theta, base_doppler_hz)
    abs_delays_s = propagation_delay_s + excess_delays_s
    mean_delay = np.sum(rel_powers_lin * abs_delays_s)
    rms_delay_spread_s = np.sqrt(np.sum(rel_powers_lin * (abs_delays_s - mean_delay)**2))
    coherence_bw_hz = 1.0 / (5.0 * rms_delay_spread_s) if rms_delay_spread_s > 0 else np.inf
    return abs_delays_s, path_dopplers_hz, rms_delay_spread_s, coherence_bw_hz

def doppler(satrec_obj, obs_time_dt, gs_lat_deg, gs_lon_deg, gs_h_km, f_hz):
    t = Time(obs_time_dt)
    error_code, teme_p, teme_v = satrec_obj.sgp4(t.jd1, t.jd2)
    if error_code != 0: raise RuntimeError(SGP4_ERRORS[error_code])
    teme_p = CartesianRepresentation(teme_p * u.km)
    teme_v = CartesianDifferential(teme_v * u.km / u.s)
    teme = TEME(teme_p.with_differentials(teme_v), obstime=t)
    itrs_geo = teme.transform_to(ITRS(obstime=t))
    gs = EarthLocation.from_geodetic(lon=gs_lon_deg*u.deg, lat=gs_lat_deg*u.deg, height=gs_h_km*u.km)
    sat_pos = itrs_geo.cartesian.without_differentials().xyz.to(u.km).value
    sat_vel = itrs_geo.cartesian.differentials['s'].d_xyz.to(u.km/u.s).value
    gs_pos = gs.get_itrs(t).cartesian.xyz.to(u.km).value
    los_vec = sat_pos - gs_pos
    los_hat = los_vec / np.linalg.norm(los_vec)
    radial_velocity_m_s = np.dot(sat_vel, los_hat) * 1000.0
    return -radial_velocity_m_s / c * f_hz

def find_best_pass(observer, satellite, search_hours=24):
    original_date = observer.date
    best_pass, best_alt_deg = None, -1.0
    end_date = ephem.Date(observer.date + search_hours * ephem.hour)
    while observer.date < end_date:
        info = observer.next_pass(satellite)
        if None in (info[0], info[2], info[4]): break
        max_alt = np.degrees(float(info[3]))
        if max_alt > best_alt_deg:
            best_alt_deg = max_alt
            best_pass = info
        observer.date = ephem.Date(info[4] + ephem.minute)
    observer.date = original_date
    return best_pass

# ==========================================
# 3. YÖRÜNGE ARAMA VE SİMÜLASYON DÖNGÜSÜ
# ==========================================
station.date = datetime.utcnow()
best_pass = find_best_pass(station, sat, search_hours=24)

if best_pass is None:
    raise RuntimeError("Geçerli bir uydu geçişi bulunamadı.")

rise_time, rise_az, max_time, max_alt, set_time, set_az = best_pass
analysis_start_dt = rise_time.datetime()
analysis_end_dt = set_time.datetime()

print(f"--- GEÇİŞ BULUNDU ---")
print(f"Başlangıç: {analysis_start_dt} UTC")
print(f"Bitiş: {analysis_end_dt} UTC")
print(f"Maksimum Elevasyon: {np.degrees(float(max_alt)):.1f} Derece")

# Zaman çözünürlüğü: Dinamik grafikler için 1 saniye 
time_step_seconds = 1 

# Veri toplama listeleri
data = collections.defaultdict(list)

station.date = rise_time
current_state = 0

print("Simülasyon hesaplanıyor, lütfen bekleyin... (Bu işlem gelişmiş fizik hesaplamaları nedeniyle birkaç saniye sürebilir)")
while station.date < set_time:
    sat.compute(station)
    theta = np.degrees(float(sat.alt))
    dist_km = float(sat.range) / 1000.0
    current_dt = station.date.datetime()
    t_rel_s = (current_dt - analysis_start_dt).total_seconds()
    
    # Sadece ufuk üzerindeki değerleri değerlendir
    if theta < 0:
        current_state = 2
    else:
        Pm = get_transition_matrix(theta)
        current_state = np.random.choice([0, 1, 2], p=Pm[current_state])
        
    # Free Space Path Loss (ITU-R P.525 Mantığı)
    fspl_db = 20 * np.log10(dist_km) + 20 * np.log10(f_ghz * 1000.0) + 32.44
    
    # ITUR Zayıflamaları (Gaz, Yağmur, Bulut, Sintilasyon)
    if theta > 5:
        try:
            gas_db = float(gaseous_attenuation_slant_path(f=f_ghz, el=theta, rho=rho_gm3, P=P_hpa, T=T_k, h=gs_h_km, mode='approx').value)
            rain_db = float(rain_attenuation(gs_lat_deg, gs_lon_deg, f_ghz, theta, p).value)
            cloud_db = float(cloud_attenuation(gs_lat_deg, gs_lon_deg, f_ghz, theta, p).value)
            scint_db = float(scintillation_attenuation(gs_lat_deg, gs_lon_deg, f_ghz, theta, p, D).value)
            atm_db = gas_db + rain_db + cloud_db + scint_db
        except Exception:
            atm_db = 0.5 / np.sin(np.radians(max(theta, 1.0)))
    else:
        atm_db = 0.0
        
    shadow_corr_db = update_correlated_shadow(shadow_corr_db, theta, current_state)
    small_scale_db = get_small_scale_loss(current_state)
    
    # Hassas Doppler (Astropy Radial Vector)
    doppler_hz = doppler(sat_sgp4, current_dt, gs_lat_deg, gs_lon_deg, gs_h_km, f_hz)
    
    # Link Bütçesi
    total_loss_db = fspl_db + atm_db + shadow_corr_db + small_scale_db
    rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss_db
    
    # Yayılım gecikmesi (Delay Spread) TDL Channel
    prop_delay_s = (dist_km * 1000.0) / c
    abs_delays, path_dopps, rms_ds, coherence_bw = build_tdl_channel(
        current_state, theta, doppler_hz, t_rel_s, prop_delay_s)
        
    # Veri Kaydı
    data['time'].append(current_dt)
    data['theta'].append(theta)
    data['dist'].append(dist_km)
    data['fspl'].append(fspl_db)
    data['atm'].append(atm_db)
    data['loss'].append(total_loss_db)
    data['doppler'].append(doppler_hz)
    data['state'].append(current_state) # 0, 1, 2
    data['rx_power'].append(rx_power)
    data['rms_ds_ns'].append(rms_ds * 1e9)
    
    # Gecikmeyi 1 Saniye olarak ileri atlat
    station.date += ephem.second * time_step_seconds

print("Simülasyon tamamlandı! Dashboard hazırlanıyor...")

# ==========================================
# 4. PROFESYONEL GÖRSELLEŞTİRME (DASHBOARD)
# ==========================================
time_arr = np.array(data['time'])
rx_arr = np.array(data['rx_power'])
theta_arr = np.array(data['theta'])
state_arr = np.array(data['state'])

fig = plt.figure(figsize=(18, 12))
fig.canvas.manager.set_window_title('NTN Advanced Channel Emulator Dashboard (SGP4 + Markov Fading)')
gs = GridSpec(3, 2, figure=fig)

# -- 1. Path Loss & Atmosphere --
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_arr, data['loss'], label='Total Channel Loss (Incl. States)', color='red')
ax1.plot(time_arr, data['fspl'], '--', label='FSPL Only', color='orange')
ax1.set_title('1. Total Path Loss Analizi (Tüm Kayıplar ve Fading Dahil)')
ax1.set_ylabel('Kayıp (dB)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# -- 2. Elevation & State Transitions --
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_arr, theta_arr, color='tab:blue', linewidth=2, label='Elevation')
ax2.set_ylabel('Elevasyon (Derece)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Arka planda Markov State boyamaları
ax2.fill_between(time_arr, 0, 90, where=(state_arr==0), color='green', alpha=0.15, label='State 0: LoS (Rician)')
ax2.fill_between(time_arr, 0, 90, where=(state_arr==1), color='yellow', alpha=0.15, label='State 1: Shadowing')
ax2.fill_between(time_arr, 0, 90, where=(state_arr==2), color='gray', alpha=0.15, label='State 2: Blocked')

ax2.set_title('2. Elevasyon & Markov Durum Geçişleri (State Machine)')
ax2.set_ylim(0, 90)
ax2.legend(loc='lower left')

# -- 3. Doppler Kayması (Astropy Radial Vector) --
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_arr, data['doppler'], color='purple', linewidth=2)
ax3.axhline(0, color='black', linestyle='--')
ax3.set_title('3. Hassas Doppler Kayması (SGP4 & Astropy Radial Vector)')
ax3.set_ylabel('Doppler (Hz)')
ax3.grid(True, linestyle='--', alpha=0.7)

# -- 4. Gecikme Yayılımı (RMS Delay Spread) --
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_arr, data['rms_ds_ns'], color='brown')
ax4.set_title('4. Çok Yollu Sönümleme (Gecikme Yayılımı / RMS DS)')
ax4.set_ylabel('RMS Gecikme (ns)')
ax4.grid(True, linestyle='--', alpha=0.7)

# -- 5. Link Budget (Kopma Kontrolü) --
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(time_arr, rx_arr, color='teal', linewidth=1.5, label='Anlık Alınan Sinyal (Rx Power)')
ax5.axhline(rx_sensitivity_dbm, color='red', linestyle='--', linewidth=2, label=f'Cihaz Hassasiyeti ({rx_sensitivity_dbm} dBm)')

# Başarı ve kopma alanlarının boyanması
ax5.fill_between(time_arr, -200, rx_sensitivity_dbm, color='red', alpha=0.15, label='Bağlantı Yok (Outage)')
ax5.fill_between(time_arr, rx_sensitivity_dbm, max(rx_arr)+10, color='green', alpha=0.15, label='Bağlantı Aktif (Linked)')

ax5.set_title('5. Link Bütçesi ve Bağlantı Fizibilitesi')
ax5.set_xlabel('Saat (UTC)')
ax5.set_ylabel('Güç (dBm)')
ax5.set_ylim(min(rx_arr)-10, max(rx_arr)+10)
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.legend(loc='lower right')

# Tüm x eksenlerini Saat ve Dakikaya göre formatla
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.show()
