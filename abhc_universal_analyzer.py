import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import torch
from abhc_master_engine import ABHCMasterEngine

# --- SETUP ---
plt.style.use('seaborn-v0_8-talk')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_file():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 Ordner '{results_dir}' wurde erstellt. Bitte .npz Dateien dort ablegen.")
        return None

    files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
    if not files:
        print(f"❌ Keine .npz Dateien in '{results_dir}' gefunden!")
        return None

    print("\n--- Verfügbare ABHC Datensätze ---")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    
    try:
        choice = int(input(f"\nWelchen Datensatz analysieren? (0-{len(files)-1}): "))
        return os.path.join(results_dir, files[choice])
    except (ValueError, IndexError):
        return None

def run_analysis():
    path = select_file()
    if not path: return

    print(f"🏗️ Initialisiere Engine...")
    engine = ABHCMasterEngine(cov_softening=1e-3)
    
    print(f"📊 Analysiere: {path}...")
    data = np.load(path)
    
    # ArviZ Formatierung
    dataset = {}
    for key in data.files:
        val = data[key]
        if val.ndim == 1:
            dataset[key] = val[None, :] # Chain hinzufügen
        else:
            dataset[key] = val
    
    idata = az.from_dict(posterior=dataset)

    # 1. Statistik
    summary = az.summary(idata, hdi_prob=0.95)
    print("\n" + "="*50)
    print(f"   POSTERIOREN: {os.path.basename(path)}")
    print("="*50)
    print(summary)
    
    # 2. Physik-Check
    # Wir suchen nach H0_true oder H0 (je nach Skript-Version)
    h0_key = "H0_true" if "H0_true" in summary.index else "H0"
    
    if "obs_pot" in summary.index and h0_key in summary.index:
        pot_mean = summary.loc["obs_pot", "mean"]
        h0_mean = summary.loc[h0_key, "mean"]
        
        # Geometrische Inversion r/Rs
        r_per_rs = 1.0 / (1.0 - (1.0 / (1.0 + pot_mean)**2))
        print(f"\n🌍 PHYSIKALISCHE INTERPRETATION:")
        print(f"-> H0 (Modell): {h0_mean:.2f}")
        print(f"-> Potential-Shift: {pot_mean*100:.2f}%")
        print(f"-> Radiale Position: r ≈ {r_per_rs:.2f} * R_s")

        # 3. BIC Berechnung (Modellgüte)
        if all(k in summary.index for k in ["n_late", "M_off"]):
            print("\n🏆 Modell-Güte (BIC)...")
            nl = summary.loc["n_late", "mean"]
            moff = summary.loc["M_off", "mean"]
            ne = summary.loc["n_early", "mean"] if "n_early" in summary.index else 0.5
            zt = summary.loc["z_trans", "mean"] if "z_trans" in summary.index else 1.5

            with torch.no_grad():
                mu_model = engine.compute_mu(h0_mean, ne, nl, zt, pot_mean, moff)
                diff = engine.mb_gpu - mu_model
                inv_cov = torch.linalg.inv(engine.L_cov @ engine.L_cov.mT)
                chi2 = (diff @ inv_cov @ diff).item()
            
            bic = chi2 + len(summary.index) * np.log(len(engine.z_gpu))
            print(f"-> Chi-Square: {chi2:.2f}")
            print(f"-> BIC: {bic:.2f}")

            # 4. Residuen-Plot
            plt.figure(figsize=(10, 5))
            z_obs = engine.z_gpu.cpu().numpy()
            resid = (engine.mb_gpu - mu_model).cpu().numpy()
            plt.scatter(z_obs, resid, alpha=0.3, s=10, color='gray')
            plt.axhline(0, color='red', linestyle='--')
            plt.xscale('log')
            plt.title(f"Residuen für {os.path.basename(path)}")
            plt.savefig(path.replace(".npz", "_residuals.png"))

    # 5. Corner Plot
    vars_to_plot = [v for v in ["H0_true", "obs_pot", "n_late", "M_off"] if v in summary.index]
    az.plot_pair(idata, kind='kde', marginals=True, var_names=vars_to_plot)
    plt.savefig(path.replace(".npz", "_corner.png"))
    print(f"\n✅ Plots gespeichert unter {path.replace('.npz', '_corner.png')}")
    plt.show()

if __name__ == "__main__":
    run_analysis()