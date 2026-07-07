import os
from astropy.io import fits
import pandas as pd
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Cartella contenente i file FITS
fits_folder = os.path.join(BASE_DIR, "galaxy_fits")
csv_folder = os.path.join(BASE_DIR, "galaxy_csv")

all_data = []

for file in os.listdir(fits_folder):
    if file.endswith(".fits"):
        file_path = os.path.join(fits_folder, file)
        print(f"Processing {file_path}...")
        
        with fits.open(file_path) as hdu:
            # --- SPETTRO ---
            # su SDSS spSpec: hdu[1].data contiene flux, loglam
            flux = hdu[1].data['flux']
            loglam = hdu[1].data['loglam']
            wavelength = 10 ** loglam  # Angstrom
            
            # salva spettro completo per la galassia
            for wl, fl in zip(wavelength, flux):
                all_data.append({
                    "file": file,
                    "type": "spectrum",
                    "wave_obs": wl,
                    "flux": fl,
                    "line_name": ""
                })
            
            # --- LINEE DI EMISSIONE ---
            # SDSS spSpec DR17 ha un HDU chiamato 'EMISSION_LINES' (es. hdu[3])
            # attenzione: può variare, controllare hdu.info()
            if len(hdu) > 3:
                try:
                    line_table = hdu[3].data
                    # alcune colonne tipiche: 'line_name', 'observed_wave', 'flux'
                    for row in line_table:
                        all_data.append({
                            "file": file,
                            "type": "emission_line",
                            "wave_obs": row['observed_wave'],
                            "flux": row['flux'],
                            "line_name": row['line_name']
                        })
                except Exception as e:
                    print(f"No emission lines in {file}: {e}")

# --- CREA CSV ---
df = pd.DataFrame(all_data)
df.to_csv(csv_folder, index=False)
print(f"Done! CSV salvato in {csv_folder}")