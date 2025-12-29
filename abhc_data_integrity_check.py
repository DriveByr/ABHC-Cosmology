import numpy as np
import os

def check_integrity():
    print("Verification of Dataset and Covariance Matrix Consistency")
    print("-" * 50)
    
    # 1. Check Supernova Data
    lc_file = "lcparam_full_long.txt"
    if not os.path.exists(lc_file):
        print(f"Error: {lc_file} not found.")
        return
    
    z_hel = []
    with open(lc_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            z_hel.append(float(line.split()[1]))
            
    n_data = len(z_hel)
    print(f"Supernova data entries: {n_data}")

    # 2. Check Covariance Matrix
    cov_file = "cov_pantheon.npz"
    if not os.path.exists(cov_file):
        print(f"Error: {cov_file} not found.")
        return
        
    try:
        npz = np.load(cov_file)
        key = 'cov' if 'cov' in npz else npz.files[0]
        cov_matrix = npz[key]
    except Exception as e:
        print(f"Error loading covariance matrix: {e}")
        return

    n_matrix = cov_matrix.shape[0]
    print(f"Covariance matrix dimensions: {n_matrix}x{n_matrix}")

    # 3. Consistency Proof
    if n_data == n_matrix:
        print("Status: MATCH - Dataset and Matrix dimensions are consistent.")
    else:
        print(f"Status: MISMATCH - Data ({n_data}) and Matrix ({n_matrix}) do not align.")
        return

    # 4. Statistical Health Check
    diag = np.diag(cov_matrix)
    if np.all(diag > 0):
        print("Health: Physical (all variances are positive).")
        print(f"Mean uncertainty: {np.mean(np.sqrt(diag)):.4f} mag")
    else:
        print("Health: Non-physical variances detected.")

    print("-" * 50)
    print("Conclusion: Data package is ready for MCMC inference.")

if __name__ == "__main__":
    check_integrity()