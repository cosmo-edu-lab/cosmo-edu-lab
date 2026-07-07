import os
import glob
import pandas as pd
import numpy as np

# --- Costanti estratte da module4.py ---
c_light = 3e5         # km/s
H0_kmsMpc = 70.0
M_sun_r = 4.67
M_to_L_ratio = 5.0    # Nel panel del cluster il M/L ratio usato di default è 5

# --- Funzione per il calcolo della massa luminosa ---
def stellar_mass_from_r_mag(app_mag_r, extinction_r, z, M_to_L_ratio=5.0):
    mag_corrected = app_mag_r - extinction_r
    D_L_Mpc = (c_light * z) / H0_kmsMpc
    dist_mod = 5 * np.log10(D_L_Mpc * 1e6) - 5
    M_galaxy_r = mag_corrected - dist_mod
    L_galaxy_solar = 10.0**(0.4 * (M_sun_r - M_galaxy_r))
    stellar_mass_msun = L_galaxy_solar * M_to_L_ratio
    return stellar_mass_msun

def crea_dataset_excel(input_folder, output_excel):
    file_list = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not file_list:
        print(f"Nessun file .txt trovato nella cartella '{input_folder}'!")
        return

    print(f"Trovati {len(file_list)} dataset. Inizio l'elaborazione...")

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for file_path in file_list:
            nome_file = os.path.basename(file_path)
            nome_cluster = os.path.splitext(nome_file)[0][:31] 
            
            try:
                # 1. Lettura del file
                df = pd.read_csv(file_path, sep=r"\s+", header=None)
                df = df.iloc[:, :8]
                df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
                
                for col in ["RAdeg", "DEdeg", "RV", "bmag"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])
                
                if df.empty:
                    print(f"  [Avviso] Il file {nome_file} è vuoto dopo la pulizia.")
                    continue

                # 2. Calcolo redshift e distanza del cluster
                v_mediana = np.nanmedian(df["RV"].values)
                z_cluster = v_mediana / c_light
                cluster_distance_mpc = v_mediana / H0_kmsMpc # Distanza in Mpc
                
                # 3. CALCOLO DEL RAGGIO IN KPC (Aggiunto!)
                # Trova il centro del cluster (media delle coordinate)
                mean_ra = df["RAdeg"].mean()
                mean_dec = df["DEdeg"].mean()
                
                # Calcola la separazione angolare (in gradi) usando l'approssimazione pitagorica
                ang_sep_deg = np.sqrt(((df["RAdeg"] - mean_ra) * np.cos(np.radians(mean_dec)))**2 + (df["DEdeg"] - mean_dec)**2)
                
                # Converti in raggio fisico (kpc)
                # Formula: r = D * theta (con theta in radianti)
                r_kpc = cluster_distance_mpc * 1000 * np.radians(ang_sep_deg)
                df["Radius_kpc"] = r_kpc
                
                # 4. Calcolo della Massa Luminosa (Barionica)
                df["Luminous_Mass_Msun"] = stellar_mass_from_r_mag(
                    app_mag_r=df["bmag"].values, 
                    extinction_r=0.0, 
                    z=z_cluster, 
                    M_to_L_ratio=M_to_L_ratio
                )
                
                # 5. Pulizia finale e salvataggio
                df_finale = df.drop(columns=["Cluster", "ID"])
                df_finale = df_finale.rename(columns={
                    "RAdeg": "Radeg",
                    "DEdeg": "Dedeg",
                    "e_RV": "err_RV"
                })
                
                df_finale.to_excel(writer, sheet_name=nome_cluster, index=False)
                print(f"  ✓ {nome_cluster} elaborato e salvato correttamente.")
                
            except Exception as e:
                print(f"  [Errore] Impossibile elaborare il file {nome_file}: {e}")

    print(f"\nOperazione completata! File salvato come: {output_excel}")

if __name__ == "__main__":
    CARTELLA_DATI = "cluster_data" 
    NOME_FILE_OUTPUT = "Cluster_Datasets_Elaborati.xlsx"
    
    if not os.path.exists(CARTELLA_DATI):
        print(f"Attenzione: La cartella '{CARTELLA_DATI}' non esiste.")
    else:
        crea_dataset_excel(CARTELLA_DATI, NOME_FILE_OUTPUT)