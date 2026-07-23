import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


G_grav = 4.30091e-6
c_light = 3e5
H0 = 70.0


def comoving_distance_Mpc(z, Om=0.3):
    integrand = lambda zp: 1.0 / np.sqrt(Om*(1.0+zp)**3 + (1.0-Om))
    chi, _ = quad(integrand, 0.0, z)
    return (c_light / H0) * chi

def angular_diameter_distance_Mpc(z, Om=0.3):
    return comoving_distance_Mpc(z, Om) / (1.0 + z)

def rho_crit_Msunkpc3():
    H0_kpc = H0 / 1000.0
    return 3.0 * (H0_kpc**2) / (8.0 * np.pi * G_grav)

def angsep_rad(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(ra1_deg); dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg); dec2 = np.deg2rad(dec2_deg)
    cos_theta = np.clip(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2), -1.0, 1.0)
    return np.arccos(cos_theta)


def r200_from_M200(M200, rho_crit_local):
    return (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit_local))**(1.0/3.0)

def delta_c_of_c(c):
    return (200.0 / 3.0) * c**3 / (np.log(1.0 + c) - c / (1.0 + c))

def concentration_duffy2008bis(M200_msun, z):
    return 5.71 * (M200_msun / (2e12 / 0.7))**(-0.084) * (1 + z)**(-0.47)
def concentration_duffy2008(M200_msun, z):
   
    M_safe = min(M200_msun, 5e15)
    
  
    c = 5.71 * (M_safe / (2e12 / 0.7))**(-0.084) * (1 + z)**(-0.47)
    
   
    return max(c, 3.5)

def rho_s_from_M200_and_c(M200, c, rho_crit_local):
    r200 = r200_from_M200(M200, rho_crit_local)
    r_s = r200 / c
    return delta_c_of_c(c) * rho_crit_local, r_s, r200

def estimate_M200_R200_from_sigma(sigma_obs, rho_crit_local):
    if sigma_obs <= 0: return 1e14, 1000.0
    
   
    R200 = np.sqrt((9.0 * sigma_obs**2) / (4.0 * np.pi * G_grav * 200.0 * rho_crit_local))
    
  
    M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
    
    return M200, R200


def run_cluster_analysis():
    coma_files = glob.glob("data/coma_data.csv") + glob.glob("coma_data.csv")
    abell_files = glob.glob("cluster_data/Abell*.txt") + glob.glob("Abell*.txt")
    all_files = list(set(coma_files + abell_files))
    
    output_dir = "cluster_plot"
    os.makedirs(output_dir, exist_ok=True)
    rho_crit = rho_crit_Msunkpc3()

    print(f"{'Cluster':<12} | {'χ² Min':<8} | {'M_DM (M_sun)':<11} | {'M_tot (M_sun)':<11} | {'M_bar (M_sun)':<11} | {'r_s (kpc)':<9} | {'rho_s (M/kpc3)':<14} | {'% DM'}")
    print("-" * 105)
    results_data = []
    for file in sorted(all_files):
        cluster_name = os.path.basename(file).replace('.txt', '').replace('.csv', '')
        try:
        
            if "coma" in cluster_name.lower():
                df = pd.read_csv(file, skiprows=1)
                df.columns = ['objid','ra','dec','modelmag_r','modelmagerr_r','extinction_r','redshift','zErr']
                observed_vel = c_light * df['redshift'].dropna().values
                z_cluster = np.nanmedian(df['redshift'].dropna().values)
                
                center_ra = np.nanmedian(df['ra'].dropna().values)
                center_dec = np.nanmedian(df['dec'].dropna().values)
                
                D_A_kpc = angular_diameter_distance_Mpc(z_cluster) * 1000.0
                theta_rad = angsep_rad(df['ra'].values, df['dec'].values, center_ra, center_dec)
                r = np.maximum(theta_rad * D_A_kpc, 1e-3)
                
                M_sun_r = 4.67
                mag = df['modelmag_r'].values - df['extinction_r'].values
                D_L_Mpc = (c_light * z_cluster) / H0
                dist_mod = 5 * np.log10(D_L_Mpc * 1e6) - 5
                M_abs = mag - dist_mod
                L_B = 10.0**(0.4 * (M_sun_r - M_abs))
                MLR = 2.0
                
            else:
                df = pd.read_csv(file, sep=r"\s+", header=None)
                if df.shape[1] == 7: df[7] = np.nan
                df = df.iloc[:, :8]
                df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
                for col in ["RAdeg", "DEdeg", "RV", "bmag"]: df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])
                
                members = df.copy()
                for _ in range(5):
                    v_rel = members["RV"] - np.median(members["RV"])
                    mad = np.median(np.abs(v_rel))
                    sigma_clip = 1.4826 * mad if mad > 0 else np.std(v_rel)
                    mask = np.abs(v_rel) <= 3 * sigma_clip
                    new_members = members[mask]
                    if len(new_members) == len(members): break
                    members = new_members
                    
                if len(members) < 5: continue
                
                cluster_distance = np.median(members["RV"]) / H0
                z_cluster = np.nanmedian(members["RV"] / c_light)
                
                mean_ra, mean_dec = members["RAdeg"].mean(), members["DEdeg"].mean()
                ang_sep = np.sqrt(((members["RAdeg"] - mean_ra) * np.cos(np.radians(mean_dec)))**2 + (members["DEdeg"] - mean_dec)**2)
                r = cluster_distance * 1000 * np.radians(ang_sep)
                members = members.assign(r_kpc=r).dropna(subset=["r_kpc"])
                members = members[members["r_kpc"] <= 3000]
                
                r = members["r_kpc"].values
                observed_vel = members["RV"].values
                bmag = members["bmag"].values
                
                D_pc = cluster_distance * 1e6
                dist_mod = 5*np.log10(D_pc) - 5
                M_abs = bmag - dist_mod
                L_B = 10**(-0.4*(M_abs - 5.48))
                MLR = 5.0
            
           
            order = np.argsort(r)
            r_sorted = r[order]
            observed_vel_sorted = observed_vel[order]
            L_cum = np.cumsum(L_B[order])
            M_lum_r = MLR * L_cum
            
            if "coma" in cluster_name.lower():
                sigma_global = np.std(observed_vel_sorted)
            else:
                sigma_global = np.std(observed_vel_sorted - np.median(observed_vel_sorted))
                
            M_tot_r = (3.0 * sigma_global**2 * r_sorted) / G_grav
            m_500_r = 0.7 * M_tot_r
            f_gas_r = 0.093 * ((m_500_r / 2e14)**0.21)
            m_gas_r = m_500_r * f_gas_r
            M_baryonic_r = M_lum_r + m_gas_r
            
            def positive_floor(arr):
                pos = arr[np.isfinite(arr) & (arr > 0)]
                if pos.size == 0: return arr + 1e-6
                return np.where(arr <= 0, np.nanmin(pos) * 1e-3, arr)

            M_tot_r = positive_floor(M_tot_r)
            M_baryonic_r = positive_floor(M_baryonic_r)
            
            M_bar_tot = M_baryonic_r[-1]
            M_tot_final = M_tot_r[-1]
            M_DM = M_tot_final - M_bar_tot
            perc_DM = (M_DM / M_tot_final) * 100
            
            M200, R200 = estimate_M200_R200_from_sigma(sigma_global, rho_crit)
            c = concentration_duffy2008(M200, z_cluster)
            rho_s, r_s, r200_nfw = rho_s_from_M200_and_c(M200, c, rho_crit)
            
            v_mean_obs = np.mean(observed_vel_sorted)
            x_max = observed_vel_sorted.max()
            padding = 0.15 * x_max
            bins = np.linspace(0, x_max + padding, 50)
            counts_obs, _ = np.histogram(observed_vel_sorted, bins=bins)
            
            counts_model = len(observed_vel_sorted) * (norm.cdf(bins[1:], loc=v_mean_obs, scale=sigma_global) - norm.cdf(bins[:-1], loc=v_mean_obs, scale=sigma_global))
            dof = max(1, len(counts_obs) - 1)
            chi2_min = np.sum(((counts_obs - counts_model)**2) / np.maximum(counts_model, 1.0)) / dof
            
           
            results_data.append({
                "Cluster": cluster_name,
                "Chi2_min": chi2_min,
                "M200": M200,
                "c": c,
                "rho_s": rho_s,
                "r_s": r_s,
                "M_bar": M_bar_tot,
                "M_DM": M_DM,
                "M_tot": M_tot_final,
                "Perc_DM": perc_DM
            })
            
            print(f"{cluster_name:<12} | {chi2_min:<8.2f} | {M_DM:<11.2e} | {M_tot_final:<11.2e} | {M_bar_tot:<11.2e} | {r_s:<9.1f} | {rho_s:<14.2e} | {perc_DM:.1f}%")
          
      
            f = 1.0
            
         
            x = r_sorted / r_s
            M_DM_NFW_r = 4.0 * np.pi * rho_s * (r_s**3) * (np.log(1.0 + x) - x / (1.0 + x))
            
         
            M_tot_model_r = M_baryonic_r + f * M_DM_NFW_r
            
         
            sigma_tot_nfw_local = np.sqrt(np.maximum(0.0, G_grav * M_tot_model_r / (3.0 * r_sorted)))
            sigma_bar_local = np.sqrt(np.maximum(1e-6, G_grav * M_baryonic_r / (3.0 * r_sorted)))
            
           
            rng = np.random.default_rng(seed=42)
            v_mean_obs = np.mean(observed_vel_sorted)
            sigma_mean_obs = np.std(observed_vel_sorted)
           
            v_bar = rng.normal(loc=sigma_bar_local, scale=sigma_mean_obs, size=len(r_sorted))
            
      
            v_dm  = rng.normal(loc=v_mean_obs, scale=sigma_tot_nfw_local, size=len(r_sorted))
            M_DM_plot = np.maximum(0.0, M_tot_r - M_baryonic_r)
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Cluster Analysis: {cluster_name}", fontsize=18, fontweight='bold')
            
          
            x_max = observed_vel_sorted.max()
            padding = 0.15 * x_max
            bins = np.linspace(0, x_max + padding, 50)
            
            axs[0,0].hist(observed_vel_sorted, bins=bins, alpha=1.0, color='blue', orientation='horizontal', label='Observed Velocities')
            axs[0,0].hist(v_dm, bins=bins, alpha=0.7, color='green', orientation='horizontal', label='Simulated Velocities')
            axs[0,0].hist(v_bar, bins=bins, alpha=0.6, color='red', orientation='horizontal', label='Baryonic Component')
            axs[0,0].set_title('Histogram of velocities distribution', fontsize=14, fontweight='bold')
            axs[0,0].set_ylabel('Velocity (km/s)')
            axs[0,0].set_xlabel('Number of Galaxies')
            axs[0,0].set_ylim(0, x_max + padding)
            axs[0,0].legend()
            axs[0,0].grid(True, axis='x', linestyle='--', alpha=0.5)
            
      
            axs[0,1].scatter(r_sorted, observed_vel_sorted, color='blue', s=15, alpha=0.6, label='Observed Galaxies')
            axs[0,1].scatter(r_sorted, v_bar, color='red', s=15, alpha=0.6, label='Baryonic Component')
            axs[0,1].scatter(r_sorted, v_dm, color='green', s=15, alpha=0.6, label='Simulated Galaxies')
            axs[0,1].set_title('Cluster galaxies velocities', fontsize=14, fontweight='bold')
            axs[0,1].set_xlabel('Radius (kpc)')
            axs[0,1].set_ylabel('Velocity (km/s)')
            axs[0,1].set_ylim(0, x_max + padding)
            axs[0,1].set_xlim(0, r_sorted.max() * 1.1)
            axs[0,1].legend()
            axs[0,1].grid(True, linestyle='--', alpha=0.5)
            
          
            mask_lum = M_baryonic_r > 0
            axs[1,0].plot(r_sorted[mask_lum], M_baryonic_r[mask_lum], 'r-', lw=2, label='Baryonic Mass')
            axs[1,0].plot(r_sorted, M_DM_plot, 'g-', lw=2, label='Dark Matter Mass')
            axs[1,0].plot(r_sorted, M_tot_r, 'b-', lw=2, label='Total Mass')
            axs[1,0].set_title('Mass Profile', fontsize=14, fontweight='bold')
            axs[1,0].set_xlabel('Radius (kpc)')
            axs[1,0].set_ylabel('Mass ($M_\odot$)')
            axs[1,0].set_yscale('log')
            axs[1,0].set_xscale('log')
            axs[1,0].legend()
            axs[1,0].grid(True, which="both", linestyle='--', alpha=0.5)
            
          
            Vol = (4.0/3.0) * np.pi * r_sorted**3
            axs[1,1].plot(r_sorted, M_baryonic_r/Vol, 'r-', lw=2, label='Baryonic Density')
            axs[1,1].plot(r_sorted, M_DM_plot/Vol, 'g-', lw=2, label='Dark Matter Density')
            axs[1,1].plot(r_sorted, M_tot_r/Vol, 'b-', lw=2, label='Total Density')
            axs[1,1].set_title('Density Profile', fontsize=14, fontweight='bold')
            axs[1,1].set_xlabel('Radius (kpc)')
            axs[1,1].set_ylabel('Density ($M_\odot / kpc^3$)')
            axs[1,1].set_yscale('log')
            axs[1,1].set_xscale('log')
            axs[1,1].legend()
            axs[1,1].grid(True, which="both", linestyle='--', alpha=0.5)

            for ax in [axs[0,0], axs[0,1]]:
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
            for ax in [axs[1,0], axs[1,1]]:
                ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
                ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir, f"{cluster_name}_analysis.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error on {cluster_name}: {str(e)}")

    df_results = pd.DataFrame(results_data)
    export_file = "cluster_best_parameters.csv"
    df_results.to_csv(export_file, index=False, float_format='%.4f')
    print(f"\nTask completato! Parametri fisici e χ² esportati in '{export_file}'.")
if __name__ == "__main__":
    run_cluster_analysis()