import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- SETUP & KONSTANTEN ---
# Wir nutzen die Ergebnisse aus dem ABHC_CMB_Anchor_Final.npz
# H0 = 67.60, obs_pot = 0.084, n_late = 0.473
H0_ABHC = 67.60
OBS_POT = 0.084
N_LATE = 0.473

# Vergleichswerte Lambda-CDM (Planck 2018)
H0_LCDM = 67.4
OM_M = 0.315
OM_L = 1.0 - OM_M

# Konstanten
c = 299792.458 # km/s
Mpc_to_km = 3.086e19
Gyr_to_sec = 3.154e16 
# Umrechnung H0 (km/s/Mpc) in 1/Gyr
def h0_to_inv_gyr(h0):
    return h0 * (Mpc_to_km / 1e6) * (1.0 / Mpc_to_km) * Gyr_to_sec * 1000

H0_inv_Gyr_ABHC = H0_ABHC / 977.8 # Näherung: 1/H0 in Gyr ~ 14.4 für H0=67
H0_inv_Gyr_LCDM = H0_LCDM / 977.8 

print(f"🔭 Starte JWST-Analyse...")
print(f"Parameter: H0={H0_ABHC}, n={N_LATE}, pot={OBS_POT}")

# --- FUNKTIONEN ---

def hz_inverse_lcdm(z):
    # 1 / E(z) für LCDM
    E = np.sqrt(OM_M * (1+z)**3 + OM_L)
    return 1.0 / ((1+z) * E)

def hz_inverse_abhc(z):
    # 1 / E(z) für ABHC
    # H(z) = H0 * (1+z)^n
    # Zeit-Integral: dt = dz / (H(z)*(1+z))
    # E(z) = (1+z)^n
    E = (1+z)**N_LATE
    return 1.0 / ((1+z) * E)

def calculate_age(z_target, model="abhc"):
    # Berechnet das Alter des Universums bei Rotverschiebung z
    # Integral von z bis unendlich
    if model == "lcdm":
        integral, _ = quad(hz_inverse_lcdm, z_target, np.inf)
        t = integral * (977.8 / H0_LCDM) # in Gyr
    else:
        # Bei ABHC müssen wir den Potential-Shift beachten!
        # Das z, das wir sehen (z_obs), ist nicht das lokale z (z_cosmo)
        # z_obs = (1+z_cosmo)*(1+phi) - 1  => z_cosmo = (1+z_obs)/(1+phi) - 1
        z_cosmo = (1 + z_target) / (1 + OBS_POT) - 1
        integral, _ = quad(hz_inverse_abhc, z_cosmo, np.inf)
        t = integral * (977.8 / H0_ABHC) # in Gyr
    return t

def calculate_da(z_target, model="abhc"):
    # Winkeldurchmesser-Entfernung dA (für Galaxien-Größe)
    # dA = dL / (1+z)^2
    if model == "lcdm":
        # Comoving distance integral
        integral, _ = quad(lambda z: 1.0/np.sqrt(OM_M*(1+z)**3 + OM_L), 0, z_target)
        dm = (c / H0_LCDM) * integral
        da = dm / (1 + z_target)
    else:
        # ABHC mit Potential Shift
        z_cosmo = (1 + z_target) / (1 + OBS_POT) - 1
        # Comoving Integral
        integral, _ = quad(lambda z: 1.0/((1+z)**N_LATE), 0, z_cosmo)
        # dL Formel aus unserer Engine
        dl = (1 + z_target) * (c / H0_ABHC) * integral
        da = dl / (1 + z_target)**2
    return da

# --- BERECHNUNG & PLOT ---

z_range = np.linspace(0, 20, 100)
age_lcdm = [calculate_age(z, "lcdm") for z in z_range]
age_abhc = [calculate_age(z, "abhc") for z in z_range]

plt.figure(figsize=(10, 6))
plt.plot(z_range, age_lcdm, label="Standard $\Lambda$CDM", linestyle='--', color='gray')
plt.plot(z_range, age_abhc, label=f"ABHC (n={N_LATE:.2f})", color='cyan', linewidth=2)
plt.axvline(10, color='red', alpha=0.3, label="JWST Limit (z=10)")

plt.title("Alter des Universums: ABHC vs. LCDM")
plt.xlabel("Redshift z")
plt.ylabel("Alter (Milliarden Jahre)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/JWST_Age_Comparison.png")

# --- REPORT ---
print("\n" + "="*40)
print("   JWST ANOMALIE CHECK")
print("="*40)

z_test = [10, 15, 20]
for z in z_test:
    t_lcdm = calculate_age(z, "lcdm") * 1000 # in Mio Jahre
    t_abhc = calculate_age(z, "abhc") * 1000
    
    print(f"Bei Redshift z={z}:")
    print(f"  Alter LCDM: {t_lcdm:.0f} Mio. Jahre")
    print(f"  Alter ABHC: {t_abhc:.0f} Mio. Jahre")
    print(f"  -> GEWINN: +{t_abhc - t_lcdm:.0f} Mio. Jahre (+{(t_abhc/t_lcdm - 1)*100:.1f}%)")
    
    # Größe einer Galaxie (1 Bogensekunde)
    # da in Mpc/rad. 1 arcsec = 4.848e-6 rad
    scale_lcdm = calculate_da(z, "lcdm") * 1000 * 4.848e-6 # kpc pro arcsec
    scale_abhc = calculate_da(z, "abhc") * 1000 * 4.848e-6
    
    print(f"  Physikalische Größe (1 arcsec):")
    print(f"    LCDM: {scale_lcdm:.2f} kpc")
    print(f"    ABHC: {scale_abhc:.2f} kpc")
    print("-" * 20)

print("\n✅ Analyse abgeschlossen. Plot gespeichert.")