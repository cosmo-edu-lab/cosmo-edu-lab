import os
import glob
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- COSTANTI FISICHE ---
G_grav = 4.30091e-6
rho_crit_local = 2.775e2

def M_nfw_enclosed(r, rho_s, r_s):
    """Calcola analiticamente la massa racchiusa di Dark Matter (NFW) al raggio r."""
    x = r / r_s
    return 4.0 * np.pi * rho_s * (r_s**3) * (np.log(1.0 + x) - x / (1.0 + x))

def observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul, y):
    """Deriva la densità osservata di DM mantenendo i segni delle componenti."""
    V_bar_sq = Vgas * np.abs(Vgas) + y * (Vdisk * np.abs(Vdisk) + Vbul * np.abs(Vbul))
    V_DM_sq = np.maximum(Vobs**2 - V_bar_sq, 0) 
    q = r**2 * (V_DM_sq / r)
    dq_dr = np.gradient(q, r)
    rho_DM = (1.0 / (4.0 * np.pi * G_grav * r**2)) * dq_dr
    return rho_DM

def fit_galaxy_parameters(r, Vobs, errV, Vgas, Vdisk, Vbul):
    """
    Esegue lo scan vettorializzato del chi^2 sui range indicati,
    seguito da un refinement locale per estrarre i parametri ottimali.
    """
    dof = max(1, len(r) - 3) # Gradi di libertà
    
   
    i_vals = np.arange(1.0, 9.25, 0.25)
    j_vals = np.arange(0, 5, 1)
    base_scan = np.array([i * (10**j) for j in j_vals for i in i_vals])
    
    rho_s_grid = base_scan * rho_crit_local
    r_s_grid = base_scan
    y_grid = np.arange(0.01, 2.01, 0.01)

  
    x = r[np.newaxis, :] / r_s_grid[:, np.newaxis] 
    M_dm_rs = 4.0 * np.pi * (r_s_grid[:, np.newaxis]**3) * (np.log(1.0 + x) - x / (1.0 + x))
    
    rho_3d = rho_s_grid[:, np.newaxis, np.newaxis]
    M_3d = M_dm_rs[np.newaxis, :, :]
    r_3d = r[np.newaxis, np.newaxis, :]
    V_dm_sq_grid = G_grav * (rho_3d * M_3d) / r_3d 
    
    best_chi2 = np.inf
    best_p = (None, None, None)
    
    for y in y_grid:
        V_bar_sq = Vgas * np.abs(Vgas) + y * (Vdisk * np.abs(Vdisk) + Vbul * np.abs(Vbul))
        V_bar_sq_3d = V_bar_sq[np.newaxis, np.newaxis, :]
        
        V_tot = np.sqrt(np.maximum(V_bar_sq_3d + V_dm_sq_grid, 0))
        
        # Calcolo Chi^2
        chi2_grid = np.sum(((Vobs[np.newaxis, np.newaxis, :] - V_tot) / errV[np.newaxis, np.newaxis, :])**2, axis=2) / dof
        
        min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
        if chi2_grid[min_idx] < best_chi2:
            best_chi2 = chi2_grid[min_idx]
            best_p = (rho_s_grid[min_idx[0]], r_s_grid[min_idx[1]], y)

  
    def objective(params):
        rho, rs, y_val = params
        V_bar_sq_opt = Vgas * np.abs(Vgas) + y_val * (Vdisk * np.abs(Vdisk) + Vbul * np.abs(Vbul))
        x_opt = r / rs
        M_dm_opt = 4.0 * np.pi * rho * (rs**3) * (np.log(1.0 + x_opt) - x_opt / (1.0 + x_opt))
        V_tot_opt = np.sqrt(np.maximum(V_bar_sq_opt + (G_grav * M_dm_opt) / r, 0))
        return np.sum(((Vobs - V_tot_opt) / errV)**2) / dof

    res = minimize(objective, best_p, bounds=[(min(rho_s_grid), max(rho_s_grid)), 
                                              (min(r_s_grid), max(r_s_grid)), 
                                              (0.01, 2.0)])
    
    if res.success and res.fun < best_chi2:
        return res.fun, res.x[0], res.x[1], res.x[2]
    
    return best_chi2, best_p[0], best_p[1], best_p[2]


def run_pipeline():
    input_folder = "galaxy_data"
    output_folder = "galaxy_plot"
    export_file = "galaxy_best_parameters.csv"
    
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not files:
        print(f"Nessun file trovato in {input_folder}")
        return

    print(f" Inizio Analisi Cinematica Globale (Minimizzazione \u03C7\u00B2)")
    print("-" * 110)
    print(f"{'Galassia':<10} | {'\u03C7\u00B2 Min':<8} | {'\u03C1_s':<9} | {'r_s':<6} | {'\u03A5':<5} | {'M_bar_edg':<9} | {'M_DM_edg':<9} | {'M_tot_edg':<9} | {'% DM':<6}")
    print(f"{'':<10} | {'':<8} | {'(M_s/kpc\u00B3)':<9} | {'(kpc)':<6} | {'(M/L)':<5} | {'(M_sun)':<9} | {'(M_sun)':<9} | {'(M_sun)':<9} | {'(bordo)':<6}")
    print("-" * 110)

    results_data = []

    for file in sorted(files):
        gal_name = os.path.basename(file).replace('.txt', '')
        
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = [line.replace('#', '').strip() + '\n' if line.strip().startswith('#') and 'Rad' in line else line for line in lines if not line.strip().startswith('#') or 'Rad' in line]
                    
            df = pd.read_csv(io.StringIO("".join(cleaned_lines)), sep=r'\s+')
            if 'Rad' not in df.columns:
                print(f"{gal_name:<10} | ERRORE LETTURA COLONNE")
                continue
                
            r = df['Rad'].values
            Vobs = df['Vobs'].values
            errV = df['errV'].values if 'errV' in df.columns else np.ones_like(Vobs) * 5.0
            Vgas = df['Vgas'].values
            Vdisk = df['Vdisk'].values 
            Vbul = df['Vbul'].values 
            
          
            chi2_min, rho_s, r_s, y_opt = fit_galaxy_parameters(r, Vobs, errV, Vgas, Vdisk, Vbul)
            
           
            V_bar_sq = Vgas * np.abs(Vgas) + y_opt * (Vdisk * np.abs(Vdisk) + Vbul * np.abs(Vbul))
            V_bar = np.sqrt(np.maximum(V_bar_sq, 0))
            
            M_dm_grid = M_nfw_enclosed(r, rho_s, r_s)
            V_dm_sq = (G_grav * M_dm_grid) / r
            V_dm = np.sqrt(np.maximum(V_dm_sq, 0))
            
            V_sim = np.sqrt(np.maximum(V_bar_sq + V_dm_sq, 0))
            
        
            M_bar = (r * V_bar_sq) / G_grav
            M_dm = M_dm_grid
            M_tot = M_bar + M_dm
            
           
            r_edge = r[-1]
            M_bar_edge = M_bar[-1]
            M_dm_edge = M_dm[-1]
            M_tot_edge = M_tot[-1]
            ratio_dm = M_dm_edge / M_tot_edge
            
           
            results_data.append({
                "Galaxy": gal_name, 
                "Chi2_min": chi2_min, 
                "rho_s": rho_s, 
                "r_s": r_s, 
                "Upsilon": y_opt,
                "M_bar_edge": M_bar_edge,
                "M_DM_edge": M_dm_edge,
                "M_tot_edge": M_tot_edge,
                "DM_Fraction_Edge": ratio_dm
            })
            
            
            print(f"{gal_name:<10} | {chi2_min:<8.2f} | {rho_s:<9.2e} | {r_s:<6.2f} | {y_opt:<5.2f} | {M_bar_edge:<9.2e} | {M_dm_edge:<9.2e} | {M_tot_edge:<9.2e} | {ratio_dm*100:>5.1f}%")

          
            rho_nfw_array = rho_s / ((r / r_s) * (1.0 + r / r_s)**2)
            rho_obs_array = observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul, y_opt)
            
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'Kinematic Analysis - {gal_name} ($\chi^2=${chi2_min:.2f})', fontsize=18, fontweight='bold')
            
          
            axs[0].errorbar(r, Vobs, yerr=errV, fmt='o', color='blue', ecolor='gray',label='Observed Data')
            axs[0].plot(r, V_bar, '--', color='green', linewidth=2, label=f'Baryonic (\u03A5={y_opt:.2f})')
            axs[0].plot(r, V_dm, ':', color='purple', linewidth=2, label='Dark Matter (NFW)')
            axs[0].plot(r, V_sim, '-', color='red', linewidth=2.5, label='Simulated Total')
            axs[0].set(title='Velocity Profile', xlabel='Radius (kpc)', ylabel='Velocity (km/s)')
            axs[0].set_ylim(0, max(max(Vobs), max(V_sim)) * 1.15)
            axs[0].grid(True, linestyle='--', alpha=0.6)
            axs[0].legend()
            
          
            M_obs = (r * Vobs**2) / G_grav
            axs[1].plot(r, M_obs, '-', color='blue', linewidth=2.5, label='Observed Mass')
            axs[1].plot(r, M_bar, '--', color='green', linewidth=2, label='Baryonic Mass')
            axs[1].plot(r, M_dm, ':', color='purple', linewidth=2, label='Dark Matter (NFW)')
            axs[1].plot(r, M_tot, '-', color='red', linewidth=2.5, label='Simulated Total')
            axs[1].set(title=' Mass Profile', xlabel='Radius (kpc)', ylabel='Enclosed Mass ($M_\odot$)')
            axs[1].grid(True, linestyle='--', alpha=0.6)
            axs[1].legend()
       
          
            axs[2].plot(r, rho_nfw_array, '-', color='purple', linewidth=2.5, label='NFW Fit ($\u03C1_{NFW}$)')
            valid_mask = rho_obs_array > 0
            axs[2].plot(r[valid_mask], rho_obs_array[valid_mask], 'o', color='gray', alpha=0.7, label=' DM Density')
            axs[2].set(title='Dark Matter Density Profile', xlabel='Radius (kpc)', ylabel='Density ($M_\odot / kpc^3$)')
            axs[2].set_yscale('log')
            axs[2].set_xscale('log') 
            axs[2].grid(True, which="both", linestyle='--', alpha=0.6)
            axs[2].legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_folder, f"{gal_name}.png"), dpi=300, bbox_inches='tight')
            plt.close() 
            
        except Exception as e:
            print(f"{gal_name:<10} | ERRORE: {str(e)}")

   
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(export_file, index=False, float_format='%.4f')
    print("-" * 110)
    print(f"\nTask completato! I grafici sono salvati in '{output_folder}'.")
    print(f"Parametri fisici e masse al bordo esportati in '{export_file}'.")

if __name__ == "__main__":
    run_pipeline()