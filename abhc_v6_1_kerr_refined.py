import os, torch, pyro, numpy as np
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
C_LIGHT = torch.tensor(299792.458, device=device, dtype=dtype)

class ABHCEngineKerr:
    def __init__(self):
        z_np, mb_np = [], []
        with open("lcparam_full_long.txt", 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                p = line.split()
                z_np.append(float(p[1])); mb_np.append(float(p[4]))
        self.z_gpu = torch.tensor(z_np, device=device, dtype=dtype)
        self.mb_gpu = torch.tensor(mb_np, device=device, dtype=dtype)
        
        npz = np.load("cov_pantheon.npz")
        C_torch = torch.tensor(npz['cov'] if 'cov' in npz else npz[npz.files[0]], device=device, dtype=dtype)
        evals, evecs = torch.linalg.eigh(C_torch)
        evals = torch.clamp(evals, min=0.05) 
        self.L_cov = torch.linalg.cholesky(evecs @ torch.diag(evals) @ evecs.mT)

    def model(self):
        # PHYSIKALISCH GEHÄRTETE PRIOREN
        H0_true = pyro.sample("H0_true", dist.Normal(67.4, 0.5))
        obs_pot = pyro.sample("obs_pot", dist.Uniform(0.01, 0.25))
        n_late  = pyro.sample("n_late", dist.Normal(0.5, 0.1)) # Holographischer Index
        M_off   = pyro.sample("M_off", dist.Normal(-19.35, 0.05)) # ZWANG ZUR PHYSIK
        
        # SPIN-FAKTOR (Der neue Kerr-Ansatz)
        # Wir modellieren den Effekt der Rotation als eine 
        # zusätzliche Skalierung der mitbewegten Distanz
        spin_a = pyro.sample("spin_a", dist.Uniform(0.0, 0.5)) # Drehimpuls-Parameter

        # --- REFINED INTEGRATION ---
        z_eff = self.z_gpu * (1.0 + obs_pot)
        z_max = float(z_eff.max().item() + 0.1)
        z_grid = torch.linspace(0.0, z_max, 500, device=device, dtype=dtype)
        
        # H(z) Entwicklung
        Hz = H0_true * torch.pow(1.0 + z_grid, n_late)
        
        # KERR-FAKTOR: Ersetzt die extreme Schwarzschild-Dehnung
        # Erhöht die Distanz moderat (wichtig für BAO), ohne zu explodieren
        kerr_factor = 1.0 + spin_a * torch.sqrt(z_grid / (1.0 + z_grid))
        
        inv_Hz_kerr = kerr_factor / torch.clamp(Hz, min=1.0)
        dz = z_grid[1] - z_grid[0]
        cum_int = torch.cumsum(inv_Hz_kerr * dz, dim=0)
        
        idx = torch.searchsorted(z_grid, z_eff) - 1
        idx = torch.clamp(idx, 0, len(z_grid)-2)
        dist_interp = cum_int[idx]
        
        dl = (1.0 + self.z_gpu) * C_LIGHT * dist_interp
        mu_model = 5.0 * torch.log10(torch.clamp(dl, min=1e-3)) + 25.0 + M_off

        pyro.sample("obs", dist.MultivariateNormal(mu_model, scale_tril=self.L_cov), obs=self.mb_gpu)

if __name__ == "__main__":
    engine = ABHCEngineKerr()
    nuts_kernel = NUTS(engine.model, target_accept_prob=0.85, max_tree_depth=10)
    mcmc = MCMC(nuts_kernel, num_samples=600, warmup_steps=200, num_chains=1)
    mcmc.run()
    
    samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    np.savez("results/ABHC_Kerr_Refined.npz", **samples)