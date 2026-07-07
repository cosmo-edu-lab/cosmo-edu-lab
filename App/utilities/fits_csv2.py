from astropy.io import fits
import pandas as pd
import os

fits_dir = "/workspaces/Cosmo-Edu_Lab/App/galaxy_fits"
lambda_dir = "/workspaces/Cosmo-Edu_Lab/App/galaxy_spectra/lambda_obs"
os.makedirs(lambda_dir, exist_ok=True)

for file in os.listdir(fits_dir):
    if file.lower().endswith(".fits"):
        fits_path = os.path.join(fits_dir, file)
        base_name = os.path.splitext(file)[0]
        try:
            with fits.open(fits_path) as hdul:
                if "SPZLINE" in [h.name for h in hdul]:
                    line_data = hdul["SPZLINE"].data
                    cols = line_data.names  # lista colonne disponibili

                    df_dict = {"line_name": line_data["LINENAME"],
                               "lambda_obs": line_data["LINEWAVE"],
                               "z_line": line_data["LINEZ"]}

                    if "LINEEW" in cols:
                        df_dict["EW"] = line_data["LINEEW"]
                    if "LINEFLUX" in cols:
                        df_dict["flux"] = line_data["LINEFLUX"]

                    df_lines = pd.DataFrame(df_dict)

                    csv_path = os.path.join(lambda_dir, f"{base_name}_lines.csv")
                    df_lines.to_csv(csv_path, index=False)
                    print(f"✅ Linee salvate: {csv_path}")
                else:
                    print(f"⚠️ Nessuna tabella SPZLINE trovata in {file}")

        except Exception as e:
            print(f"❌ Errore con {file}: {e}")

print("\n🎉 Estrazione di λ_obs completata!")

