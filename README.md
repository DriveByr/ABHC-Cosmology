# Accreting Black Hole Cosmology (ABHC)

## Evidence for a Rotating Universe: Resolving Cosmological Anomalies via Kerr-Vortex Geometry

### Overview
This repository contains the source code, data processing pipelines, and MCMC results for the Accreting Black Hole Cosmology (ABHC) model. The project demonstrates that the observed expansion of the universe can be modeled as the interior of a rotating, accreting Kerr singularity. This approach provides a unified geometric solution to several high-profile discrepancies in modern astrophysics, most notably the Hubble ($H_0$) tension and the JWST early galaxy formation paradox.

### Key Scientific Findings
- **Global Cosmic Spin:** Detection of a dimensionless spin parameter $a = 0.382 \pm 0.058$ at $6.5\sigma$ significance, suggesting a vortex-like structure of the spacetime metric.
- **Hubble Tension Resolution:** The discrepancy between local and global $H_0$ measurements is resolved as a relativistic observer-potential shift ($\Phi \approx 8.4\%$), aligning Pantheon+ Supernovae data with Planck CMB constraints.
- **Temporal Dilation:** The model predicts a non-linear temporal scaling ($n \approx 0.47$), extending the available time for structure formation at $z=10$ to approximately 820 Myr, alleviating the "impossible early galaxy" problem.
- **Metric Stretching:** Integration of the Kerr-Vortex factor $K(z)$ provides a geometric basis for the consistent inclusion of high-redshift BAO data points without requiring a cosmological constant $\Lambda$.

### Repository Structure
- `abhc_master_engine.py`: Core integration engine implementing the modified Schwarzschild/Kerr interior metric.
- `abhc_v6_1_kerr_refined.py`: Main MCMC sampling script utilizing the Kerr-Vortex metric.
- `abhc_universal_analyzer.py`: Comprehensive diagnostic tool for posterior analysis and visualization.
- `abhc_data_integrity_check.py`: Validation script to ensure consistency between the Pantheon+ dataset and the covariance matrix.
- `results/`: Directory containing pre-computed MCMC samples and final publication figures.

### Installation
The environment requires Python 3.10+ and a CUDA-capable GPU for efficient MCMC sampling.

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Validation
Run the integrity check to verify that the local Pantheon+ dataset and covariance matrix match:
```bash
python abhc_data_integrity_check.py
```

#### 2. Replication
To reproduce the $6.5\sigma$ spin detection and generate new MCMC samples:
```bash
python abhc_v6_1_kerr_refined.py
```

#### 3. Analysis
To process results, generate residuals, corner plots, and calculate Bayesian Information Criterion (BIC) statistics:
```bash
python abhc_universal_analyzer.py
```

### Citation
Franke, P. (2025). *Resolving the Hubble Tension via Observer Location in an Accreting Black Hole Cosmology*. arXiv:[Link coming soon].

### Data Attribution
This work utilizes the Pantheon+ Supernova dataset and the BOSS DR12 BAO consensus data.

### Contact
**Peter Franke** - Independent Researcher  
Email: peter.franke.indyresearch@gmail.com  
ORCID: [0009-0007-2363-7943](https://orcid.org/0009-0007-2363-7943)  
GitHub: [DriveByr](https://github.com/DriveByr)