import os
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np
import time
from abhc_master_engine import ABHCMasterEngine # Importiert Teil 1

# ---------------------------------------------------------
# SETUP & INITIALISIERUNG
# ---------------------------------------------------------
engine = ABHCMasterEngine(cov_softening=1e-3) # Milde Reinigung für "Flüssigkeit"

def model():
    # --- PHYSIKALISCHE ANKER (Die ABHC-Identität) ---
    
    # H0_true: Der globale Hubble-Wert am Horizont
    H0_true = pyro.sample("H0_true", dist.Uniform(62.0, 72.0))
    pyro.sample("H0_prior", dist.Normal(67.4, 0.8), obs=H0_true) # Starker CMB-Anker
    
    # obs_pot: Unsere radiale Position (0.0 = Zentrum/LCDM, >0 = Potential-Shift)
    obs_pot = pyro.sample("obs_pot", dist.Uniform(0.0, 0.35))
    
    # Akkretions-Indizes (Rigid BHU Konfiguration)
    n_early = pyro.sample("n_early", dist.Normal(0.5, 0.1)) # Strahlung/Eddington
    n_late  = pyro.sample("n_late", dist.Normal(1.5, 0.15)) # Materie/Holographisch
    z_trans = pyro.sample("z_trans", dist.Uniform(1.0, 2.5)) # Der Phasen-Übergang
    
    # M_off: Absoluthelligkeit (MUSS bei -19.35 liegen für SH0ES-Konsistenz)
    M_off = pyro.sample("M_off", dist.Uniform(-19.5, -19.2))
    pyro.sample("M_off_prior", dist.Normal(-19.35, 0.05), obs=M_off)

    # --- DIFFERENTIABLE PHYSICS PASS ---
    mu_model = engine.compute_mu(H0_true, n_early, n_late, z_trans, obs_pot, M_off)

    # --- LIKELIHOOD (Cholesky-HPC-Modus) ---
    pyro.sample("obs", dist.MultivariateNormal(mu_model, scale_tril=engine.L_cov), obs=engine.mb_gpu)

# ---------------------------------------------------------
# EXECUTION LOOP MIT CHECKPOINTS
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🛰️ Starte ABHC Master-Sampler (Checkpoint Mode)...")
    
    if not os.path.exists("results"): os.makedirs("results")

    # NUTS Konfiguration (High-Res)
    nuts_kernel = NUTS(model, 
                       target_accept_prob=0.94, 
                       max_tree_depth=10, 
                       init_strategy=pyro.infer.autoguide.initialization.init_to_median)

    warmup_steps = 400
    total_samples = 1000
    batch_size = 50
    
    mcmc = MCMC(nuts_kernel, num_samples=batch_size, warmup_steps=warmup_steps, num_chains=1)
    
    master_samples = {}
    t_start = time.time()

    try:
        # Erster Batch (inkl. Warmup)
        print(f"🔥 Phase 1: Starte Warmup und Batch 1...")
        mcmc.run()
        master_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
        np.savez("results/ABHC_Master_Checkpoint.npz", **master_samples)
        
        # Iterativer Batch-Loop
        n_batches = total_samples // batch_size
        for i in range(1, n_batches):
            print(f"\n📦 Batch {i+1}/{n_batches} startet...")
            
            # Warmstart-Parameter vom letzten Sample
            last_params = {k: torch.tensor(v[-1], device=engine.z_gpu.device) for k, v in master_samples.items()}
            
            # Neuer Kurz-Lauf ohne erneutes Warmup
            mcmc = MCMC(nuts_kernel, num_samples=batch_size, warmup_steps=0, initial_params=last_params)
            mcmc.run()
            
            new_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
            for key in master_samples:
                master_samples[key] = np.concatenate([master_samples[key], new_samples[key]])
            
            # Sicherung
            np.savez("results/ABHC_Master_Checkpoint.npz", **master_samples)
            
            # Gradienten-Watchdog / Monitor
            p_curr = master_samples['obs_pot'][-1]
            h_curr = master_samples['H0_true'][-1]
            print(f"📊 LIVE: obs_pot={p_curr:.4f} | H0_true={h_curr:.2f} | M_off={master_samples['M_off'][-1]:.3f}")
            
            if np.isnan(p_curr):
                print("⚠️ WARNUNG: Gradient abgestürzt (NaN). Breche Batch-Loop ab.")
                break

        print(f"\n✅ Sampler beendet. Gesamtdauer: {(time.time()-t_start)/60:.2f} min")
        np.savez("results/ABHC_Master_Final.npz", **master_samples)

    except Exception as e:
        print(f"\n💥 KRITISCHER FEHLER: {e}")
        if master_samples:
            np.savez("results/ABHC_Master_Emergency_Backup.npz", **master_samples)
            print("🆘 Notfall-Sicherung erstellt.")