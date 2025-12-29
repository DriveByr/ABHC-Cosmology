import numpy as np
import matplotlib.pyplot as plt

def simulate_grb_jitter(distance_gly, spin_a, direction_angle):
    """
    Simuliert den Ankunfts-Jitter von Photonen in einem Kerr-Vortex-Universum.
    """
    # Konstanten
    c = 299792.458
    lp = 1.616e-35 # Planck Länge
    L = distance_gly * 9.461e24 # Distanz in Metern
    
    # Der fundamentale holographische Jitter (Hogan-Skalierung)
    base_jitter = np.sqrt(L * lp) / c
    
    # Vortex-Modulation (Chiralität)
    # Der Jitter ist stärker, wenn man senkrecht zur Rotationsachse misst
    vortex_modulation = 1.0 + spin_a * np.sin(direction_angle)
    
    total_jitter_std = base_jitter * vortex_modulation
    return total_jitter_std

# --- RECHERCHE-DATEN ABGLEICH ---
# Fermi-LAT GRB Daten zeigen oft Puls-Breiten von ~10-100 ms.
# Wenn unser Modell 5 ms Jitter allein durch die Raumzeit vorhersagt,
# ist das eine signifikante Entdeckung.

def plot_jitter_prediction():
    distances = np.linspace(1, 13, 50) # Milliarden Lichtjahre
    spin = 0.382
    
    # Jitter für zwei verschiedene Richtungen am Himmel
    jitter_axis = [simulate_grb_jitter(d, spin, 0) for d in distances]
    jitter_equator = [simulate_grb_jitter(d, spin, np.pi/2) for d in distances]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, np.array(jitter_axis)*1000, label="Richtung Vortex-Achse", color='cyan')
    plt.plot(distances, np.array(jitter_equator)*1000, label="Richtung Vortex-Äquator", color='magenta')
    
    plt.title("Vorhersage der Zeit-Pixelierung (Vortex-Jitter)")
    plt.xlabel("Distanz der Quelle (Milliarden Lichtjahre)")
    plt.ylabel("Zeit-Unschärfe / Jitter (ms)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

if __name__ == "__main__":
    plot_jitter_prediction()