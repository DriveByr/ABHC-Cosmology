import os
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
C_LIGHT = torch.tensor(299792.458, device=device, dtype=dtype) # km/s

class ABHCMasterEngine:
    def __init__(self, cov_softening=1e-4):
        print(f"🏗️ Initialisiere ABHC Master-Engine auf {device}...")
        self.z_gpu, self.mb_gpu = self._load_pantheon()
        self.L_cov = self._load_and_clean_cov(cov_softening)
        print("✅ Engine bereit für MCMC.")

    def _load_pantheon(self):
        z_np, mb_np = [], []
        with open("lcparam_full_long.txt", 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                p = line.split()
                z_np.append(float(p[1])) # zhel
                mb_np.append(float(p[4])) # mb
        return torch.tensor(z_np, device=device, dtype=dtype), \
               torch.tensor(mb_np, device=device, dtype=dtype)

    def _load_and_clean_cov(self, softening):
        npz = np.load("cov_pantheon.npz")
        C_np = npz['cov'] if 'cov' in npz else npz[npz.files[0]]
        C_torch = torch.tensor(C_np, device=device, dtype=dtype)
        
        # Spektrale Reinigung: Wir stabilisieren die Matrix, 
        # ohne die Korrelationen zu zerstören
        evals, evecs = torch.linalg.eigh(C_torch)
        evals = torch.clamp(evals, min=softening) 
        C_fixed = evecs @ torch.diag(evals) @ evecs.mT
        return torch.linalg.cholesky(C_fixed)

    def get_Hz(self, z, H0, n_early, n_late, z_trans):
        # Sigmoid Phasenübergang (Physik: Radiation -> Matter Accretion)
        alpha = 15.0
        switch = torch.sigmoid(alpha * (z - z_trans))
        n_eff = n_early * switch + n_late * (1.0 - switch)
        
        # ABHC Identität: H(z) folgt dem Massenzuwachs n
        return H0 * torch.pow(1.0 + z, n_eff)

    def compute_mu(self, H0, n_early, n_late, z_trans, obs_pot, M_off):
        # 1. Potential-Shift (Der geometrische Standort-Effekt)
        z_eff = self.z_gpu * (1.0 + obs_pot)
        
        # 2. Integrations-Gitter (Differentiable Trapezoid)
        z_max = float(z_eff.max().item() + 0.1)
        z_grid = torch.linspace(0.0, z_max, 400, device=device, dtype=dtype)
        
        Hz = self.get_Hz(z_grid, H0, n_early, n_late, z_trans)
        inv_Hz = 1.0 / torch.clamp(Hz, min=1.0)
        
        dz = z_grid[1] - z_grid[0]
        cum_int = torch.cumsum(inv_Hz * dz, dim=0)
        
        # 3. Interpolation auf SN-Positionen
        idx = torch.searchsorted(z_grid, z_eff) - 1
        idx = torch.clamp(idx, 0, len(z_grid)-2)
        # Linearer Anteil für Gradienten-Fluss
        t = (z_eff - z_grid[idx]) / (z_grid[idx+1] - z_grid[idx])
        dist_interp = cum_int[idx] + t * (cum_int[idx+1] - cum_int[idx])
        
        # 4. dL in Mpc: (1+z) * c * Integral
        # WICHTIG: Hier nutzen wir das BEOBACHTETE z für die Flächenhelligkeit
        dl = (1.0 + self.z_gpu) * C_LIGHT * dist_interp
        
        # 5. Distanzmodul mu (dL muss in Mpc sein für M_off ~ -19.3)
        return 5.0 * torch.log10(torch.clamp(dl, min=1e-3)) + 25.0 + M_off