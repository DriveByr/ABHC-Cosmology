import numpy as np

def calculate_expected_jitter(ra, dec, spin_a=0.3819):
    """
    Berechnet den erwarteten Zeit-Jitter (in Millisekunden) 
    basierend auf der Position zum Vortex-Zentrum (165, -5).
    """
    # Vortex-Zentrum (Radianten)
    v_ra, v_dec = np.radians(165), np.radians(-5)
    p_ra, p_dec = np.radians(ra), np.radians(dec)
    
    # Winkelabstand zur Achse via Skalarprodukt
    cos_theta = (np.sin(v_dec) * np.sin(p_dec) + 
                 np.cos(v_dec) * np.cos(p_dec) * np.cos(v_ra - p_ra))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    
    # Vorhersage: Jitter ist maximal am Äquator der Rotation (theta = pi/2)
    # und minimal an den Polen (theta = 0 oder pi)
    jitter_factor = np.sin(theta) 
    
    # Basis-Jitter (für z=1 ca. 5ms)
    expected_ms = 5.0 * (1.0 + spin_a * jitter_factor)
    return expected_ms

# --- VALIDIERUNG DER DATENBANK ---
grbs = [
    {"name": "GRB 140508A", "ra": 143.7, "dec": 2.5, "obs_complexity": "High"},
    {"name": "GRB 130427A", "ra": 173.1, "dec": 27.7, "obs_complexity": "Very High"},
    {"name": "GRB 080916C", "ra": 119.8, "dec": -56.6, "obs_complexity": "Low"}
]

print("🔬 ABHC VORTEX-JITTER VORHERSAGE:")
print("="*50)
for g in grbs:
    jitter = calculate_expected_jitter(g['ra'], g['dec'])
    print(f"{g['name']}:")
    print(f"  -> Winkel zur Vortex-Achse: {jitter/5.0 - 1:.2f} rad")
    print(f"  -> Erwarteter Jitter:      {jitter:.2f} ms")
    print(f"  -> Beobachtete Komplexität: {g['obs_complexity']}")
    print("-" * 30)