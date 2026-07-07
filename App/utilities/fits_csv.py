import os
from astropy.io import fits
import pandas as pd
import numpy as np

# Percorso della cartella che contiene i file FITS
fits_dir = "/workspaces/Cosmo-Edu_Lab/App/galaxy_fits"
csv_dir= "/workspaces/Cosmo-Edu_Lab/App/galaxy_spectra/csv_converted"
# Crea una sottocartella per i CSV (se non esiste)
#csv_dir = os.path.join(fits_dir, "csv_converted")
#os.makedirs(csv_dir, exist_ok=True)

# Cicla su tutti i file FITS nella cartella
for file in os.listdir(fits_dir):
    if file.lower().endswith(".fits"):
        fits_path = os.path.join(fits_dir, file)
        try:
            # Apri il file FITS
            with fits.open(fits_path) as hdul:
                data = hdul[1].data  # L'HDU con la tabella spettrale

                # Estrai colonne
                flux = data["flux"]
                loglam = data["loglam"]
                wavelength = 10 ** loglam  # in Å

                # Crea DataFrame
                df = pd.DataFrame({
                    "wavelength_A": wavelength,
                    "flux": flux
                })

                # Nome del file CSV corrispondente
                base_name = os.path.splitext(file)[0]
                csv_path = os.path.join(csv_dir, f"{base_name}.csv")

                # Salva in CSV
                df.to_csv(csv_path, index=False)
                print(f"✅ Salvato: {csv_path}")

        except Exception as e:
            print(f"⚠️ Errore nel file {file}: {e}")

print("\n🎉 Conversione completata per tutti i file FITS nella cartella!")

