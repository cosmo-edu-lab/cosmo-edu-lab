# Cosmo-Edu-Lab — Dark Matter Analysis Notebooks & Datasets

This folder contains the commented Python notebooks and pre-formatted observational datasets accompanying:

> **Panini E., Goldoni G., Tufino E., Bertuzzo E. (2026)**, *"Dark Matter estimation as simple as possible"*.

The notebooks reproduce all figures, interactive scatter plots, and Supplementary Material LaTeX tables for both **spiral galaxies** (Sec. 2) and **galaxy clusters** (Sec. 3).

---

## Folder Structure

```text
App/notebooks/
├── README.md
├── Galaxies.ipynb                    # Sec. 2: SPARC rotation curves, Burkert halo & Differential Evolution fit
├── Clusters.ipynb                    # Sec. 3: Coma & SARS virial mass, MAD filtering & velocity histograms
└── datasets/
    ├── Rotmod_LTG/                   # Individual SPARC galaxy rotation curves (*_rotmod.dat)
    ├── Table.txt                     # SPARC metadata (D, Inc, Vflat, quality flag Q)
    ├── parameter_Burkert.txt         # Li et al. (2020) 5P/6P Burkert MCMC benchmark catalog
    ├── DE_3p.csv                     # Curated 47-galaxy sample & pre-fitted 3P/4P DE parameters
    ├── Abell1656(Coma).csv           # SDSS spectroscopic & r-band photometric data for the Coma cluster
    └── cluster_data_tot/             # Individual SARS Abell cluster catalogs (Abell*.txt)
```

---

## Option 1: Run on Google Colab (Recommended — No Local Installation)

You can run both notebooks directly in your browser using **Google Colab** in two ways.

### Method A — Quick Start via GitHub Clone (No Google Drive Required)

1. Open a notebook directly in Google Colab:
   * **Galaxies:** `https://colab.research.google.com/github/cosmo-edu-lab/cosmo-edu-lab/blob/main/App/notebooks/Galaxies.ipynb`
   * **Clusters:** `https://colab.research.google.com/github/cosmo-edu-lab/cosmo-edu-lab/blob/main/App/notebooks/Clusters.ipynb`
2. **Skip or comment out** the first cell (`drive.mount('/content/drive')`).
3. Insert a new code cell at the top and run:
   ```python
   !git clone [https://github.com/cosmo-edu-lab/cosmo-edu-lab.git](https://github.com/cosmo-edu-lab/cosmo-edu-lab.git) /content/cosmo-edu-lab
   ```
4. In **Step 0**, update `BASE_DIR` to point to the cloned datasets folder:
   ```python
   BASE_DIR = "/content/cosmo-edu-lab/App/notebooks/datasets"
   ```
5. Click **Runtime → Run all** to execute the full analysis.

### Method B — Using Google Drive (Persistent Setup)

1. Download the repository (see **Downloading the Files** below) and upload the `datasets` folder to your Google Drive, for example inside:
   ```text
   MyDrive/Cosmo-Edu-Lab/dataset/
   ```
   *(Keep all subfolders such as `Rotmod_LTG/` and `cluster_data_tot/` intact inside it.)*
2. Upload `Galaxies.ipynb` and/or `Clusters.ipynb` to Google Drive and open them with **Google Colaboratory**.
3. Run the first cell to mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. In **Step 0**, verify that `BASE_DIR` matches your exact Google Drive path:
   ```python
   BASE_DIR = "/content/drive/MyDrive/Cosmo-Edu-Lab/dataset"
   ```
   *(Tip: in Colab's left sidebar, click the **Files** folder icon, navigate to your dataset folder, click the three dots `⋮`, and select **Copy path** to paste it directly into `BASE_DIR`.)*

---

## Option 2: Download & Run Locally (Jupyter / VS Code / Anaconda)

### 1. Download the Files
* **With Git:**
  ```bash
  git clone [https://github.com/cosmo-edu-lab/cosmo-edu-lab.git](https://github.com/cosmo-edu-lab/cosmo-edu-lab.git)
  cd cosmo-edu-lab/App/notebooks
  ```
* **Without Git (Browser):** go to the [repository root](https://github.com/cosmo-edu-lab/cosmo-edu-lab), click **Code → Download ZIP**, extract the archive, and navigate to `App/notebooks/`.

### 2. Install Dependencies
Python 3.10+ is recommended. Install the required scientific packages:
```bash
pip install numpy pandas scipy matplotlib plotly pillow jupyter
```

### 3. Configure the Input Path
1. Open `Galaxies.ipynb` or `Clusters.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. **Skip or comment out** the Google Colab cell (`from google.colab import drive ...`).
3. In **Step 0**, set `BASE_DIR` according to your working directory:
   * If you launched Jupyter from inside `App/notebooks/`:
     ```python
     BASE_DIR = "datasets"
     ```
   * If you launched Jupyter from the repository root (`cosmo-edu-lab/`):
     ```python
     BASE_DIR = "App/notebooks/datasets"
     ```
   * Alternatively, provide the full absolute path on your machine (e.g., `r"C:\path\to\datasets"` on Windows or `"/path/to/datasets"` on macOS/Linux).

---

## Notebook Highlights & Configurable Options

### `Galaxies.ipynb`
* **Input files:** `Rotmod_LTG/*_rotmod.dat`, `Table.txt`, `parameter_Burkert.txt`, and `DE_3p.csv`.
* **Useful toggles (Steps 3–4):**
  * `USE_PRECOMPUTED_PARAMS = True` — loads pre-fitted Differential Evolution parameters from `DE_3p.csv` for fast classroom demos. Set to `False` to run live Differential Evolution optimization on all 47 galaxies (~30–60 s).
  * `SHOW_INDIVIDUAL_PLOTS = False` — displays only `NGC3198` inline plus the multi-page grid. Set to `True` to display all 47 individual plots inline.
  * `CONNECT_PAIRS = True` — draws dashed segments connecting our 3P/4P DE results to the 5P/6P MCMC benchmark of Li et al. (2020).
* **Generated outputs (`galaxy_plots_output/`):** individual dual-axis rotation curves (`*_plot.png`), multi-page summary grids (`FINAL_Galaxy_Grid_Page_*.png`), static and interactive DM fraction vs. mass scatter plots (`.png` and `.html`), and Supplementary Material Table S1 (`galaxies_params_table.tex`).

### `Clusters.ipynb`
* **Input files:** `Abell1656(Coma).csv` and `cluster_data_tot/Abell*.txt`.
* **Workflow:** applies the two-stage velocity membership filter ($\pm 6000\ \text{km s}^{-1}$ peak window + iterative $3\sigma_{\text{MAD}}$ clipping), estimates $(M_{200}, R_{200})$ via the Virial Theorem, computes stellar and intracluster gas masses, and evaluates reduced Pearson's $\chi^2_\nu$ with $100\times$ Monte Carlo oversampling and adjacent super-bin pooling ($E_i \ge 1$).
* **Generated outputs (`cluster_plots_output/`):** individual peculiar velocity histograms (`*_plot.png`), the 31-cluster summary grid (`FINAL_Cluster_Grid.png`), static and interactive $\%M_{\text{DM}}$ vs. $M_{200}$ scatter plots (`.png` and `.html`), and Supplementary Material Table S2 (`clusters_mass_budget_table.tex`).

---

## Original Dataset Credits

* **SPARC Galaxies:** Lelli F., McGaugh S. S., Schombert J. M. (2016), *AJ*, 152, 157.
* **Burkert MCMC Benchmark:** Li P., Lelli F., McGaugh S., Schombert J. (2020), *ApJS*, 247, 31.
* **Coma Cluster (SDSS):** Alkan M. (2021), Kaggle Coma Cluster Dataset; SDSS Collaboration.
* **SARS Abell Clusters:** Way M. J., Quintana H., Infante L., Lambas D. G., Muriel H. (2005), *AJ*, 130, 2012.