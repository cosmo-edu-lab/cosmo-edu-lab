import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

import os
import io
import json
import uuid
import base64
import datetime
import threading
import random
from io import BytesIO
import json
import re
# Librerie scientifiche e di calcolo
import numpy as np
import pandas as pd
import numexpr as ne
import matplotlib.pyplot as plt
from fractions import Fraction
from math import isfinite
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import curve_fit, fsolve, brentq, minimize_scalar
from scipy.signal import savgol_filter
from scipy.special import i0, i1, k0, k1
from scipy.signal import find_peaks
from scipy.stats import linregress
from fractions import Fraction
from astropy.io import fits as _fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.transforms as transforms
from matplotlib.transforms import blended_transform_factory
import math
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar
import PIL
from PIL import Image
from glob import glob
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import plotly.graph_objects as go
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
                       
import matplotlib.lines as mlines
# Librerie Web/App
import requests
from dotenv import load_dotenv
from fastapi import Request
from groq import Groq
from ipywidgets import interact, FloatSlider
from nicegui import ui, app, run , client
from nicegui_toolkit import inject_layout_tool
from functools import lru_cache
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
from astroquery.simbad import Simbad
# --- 1. CONFIGURAZIONE ---
BASE_DIR = r"C:\\Users\\elyon\\OneDrive\\Desktop\\Cosmo-Edu_Lab"
DATA_DIR = os.path.join(BASE_DIR, "data")
CLUSTER_DATA_PATH = os.path.join(BASE_DIR, "cluster_data") 
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "cluster_gif")
TEMP_FOLDER = "temp_frames_gen"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
c_light = 299792.458
G_grav = 4.301e-6 
rho_crit = 2.775e2
      
M_sun_r = 4.64 

def comoving_distance_Mpc(z, H0=70.0, Om=0.3):
    integrand = lambda zp: 1.0 / np.sqrt(Om*(1.0+zp)**3 + (1.0-Om))
    chi, _ = quad(integrand, 0.0, z)
    return (c_light / H0) * chi  # Mpc

def angular_diameter_distance_Mpc(z, H0=70.0, Om=0.3):
    return comoving_distance_Mpc(z, H0, Om) / (1.0 + z)

def rho_crit_Msunkpc3(H0=70.0):
    # H0 [km/s/Mpc] -> km/s/kpc
    H0_kpc = H0 / 1000.0
    return 3.0 * (H0_kpc**2) / (8.0 * np.pi * G_grav)  # Msun / kpc^3

def angsep_rad(ra1_deg, dec1_deg, ra2_deg, dec2_deg):

    ra1 = np.deg2rad(ra1_deg); dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg); dec2 = np.deg2rad(dec2_deg)
    cos_theta = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def delta_c_of_c(c):
    return (200.0 / 3.0) * c**3 / (np.log(1.0 + c) - c / (1.0 + c))

def r200_from_M200(M200, rho_crit_local):
    return (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit_local))**(1.0/3.0)

def rho_s_from_M200_and_c(M200, c, rho_crit_local):
    r200 = r200_from_M200(M200, rho_crit_local)
    r_s = r200 / c
    delta_c = delta_c_of_c(c)
    rho_s = delta_c * rho_crit_local
    return rho_s, r_s, r200

def Mc_nfw_enclosed(r, M200, c, rho_crit_local):
    r = np.asarray(r, dtype=float)
    rho_s, r_s, r200 = rho_s_from_M200_and_c(M200, c, rho_crit_local)
    x = r / r_s
    numer = np.log(1.0 + x) - x / (1.0 + x)
    denom = np.log(1.0 + c) - c / (1.0 + c)
    enclosed = M200 * numer / denom
    enclosed = np.where(r > 0, enclosed, 0.0)
    return enclosed


def calculate_concentration_merten2014(M200_msun, z, h=0.7):
    if M200_msun <= 0 or not np.isfinite(M200_msun):
        return 4.0
    A = 3.66; B = -0.14; C = -0.32  
    M_pivot = 1e15 / h
    log10_c = A + B * np.log10(M200_msun / M_pivot) + C * np.log10(1.0 + z)
    return 10.0**log10_c
def concentration_duffy2008(M200_msun, z, h=0.7, relaxed=False):

    M_pivot = 2e12 / h  # Msun

    if relaxed:
        A, B, C = 6.71, -0.091, -0.44
    else:
        A, B, C = 5.71, -0.084, -0.47

    return A * (M200_msun / M_pivot)**B * (1 + z)**C


def estimate_M200_R200_from_sigma(sigma_obs, rho_crit_local, G=G_grav,
                                R200_min=100.0, R200_max=5000.0,
                                verbose=False):
    """
    Stima M200 e R200 a partire da sigma_los osservata.
    - sigma_obs in km/s
    - rho_crit_local in Msun/kpc^3
    - G in (kpc * (km/s)^2) / Msun  (come il tuo G_grav)
    Restituisce (M200 [Msun], R200 [kpc]).

    """
    import math
    from scipy.optimize import brentq


    if sigma_obs is None or not np.isfinite(sigma_obs):
        raise RuntimeError("sigma_obs non finito")
    if sigma_obs <= 0.0:
    
        if verbose:
            print("estimate_M200_R200_from_sigma: sigma_obs <= 0, uso fallback scaling")
        M200 = 1e15 * (max(sigma_obs, 1e-6) / 1082.0)**3
        R200 = r200_from_M200(M200, rho_crit_local)
        return M200, R200

    def func(R200):
    
        M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
    
        try:
            val = np.sqrt(max(0.0, G * M200 / (3.0 * R200))) - sigma_obs
        except Exception:
            val = np.nan
        return val

    try:
        fmin = func(R200_min)
        fmax = func(R200_max)
    except Exception as e:
        if verbose:
            print("estimate_M200_R200_from_sigma: errore valutando func iniziale:", e)
        fmin, fmax = np.nan, np.nan

    if verbose:
        print(f"estimate_M200_R200_from_sigma: sigma_obs={sigma_obs:.3f}, R200_min={R200_min}, R200_max={R200_max}, fmin={fmin}, fmax={fmax}")


    if np.isfinite(fmin) and np.isfinite(fmax) and fmin * fmax < 0:
        R200 = brentq(func, R200_min, R200_max)
        M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
        return M200, R200


    facs = [2, 5, 10, 20]
    found = False
    Rmin, Rmax = R200_min, R200_max
    for f in facs:
        Rmin_try = max(1.0, R200_min / f)
        Rmax_try = R200_max * f
        try:
            fmin_try = func(Rmin_try)
            fmax_try = func(Rmax_try)
        except Exception:
            fmin_try, fmax_try = np.nan, np.nan
        if verbose:
            print(f" trying bracket {Rmin_try:.1f}-{Rmax_try:.1f} -> {fmin_try}, {fmax_try}")
        if np.isfinite(fmin_try) and np.isfinite(fmax_try) and fmin_try * fmax_try < 0:
            Rmin, Rmax = Rmin_try, Rmax_try
            found = True
            break

    if found:
        R200 = brentq(func, Rmin, Rmax)
        M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
        return M200, R200


    logR = np.linspace(np.log(R200_min/10.0), np.log(R200_max*10.0), 200)
    Rgrid = np.exp(logR)
    vals = []
    for R in Rgrid:
        vals.append(func(R))
    vals = np.array(vals)
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() >= 2:
        idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if idx.size > 0:
            i = idx[0]
            R_lo, R_hi = Rgrid[i], Rgrid[i+1]
            if verbose:
                print(f" found sign change in grid: {R_lo:.2f}-{R_hi:.2f}")
            R200 = brentq(func, R_lo, R_hi)
            M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
            return M200, R200


    if verbose:
        print("estimate_M200_R200_from_sigma: no bracket found -> fallback scaling Evrard")

    sigma_safe = max(sigma_obs, 1e-3)
    M200 = 1e15 * (sigma_safe / 1082.0)**3
    R200 = r200_from_M200(M200, rho_crit_local)
    return M200, R200


def stellar_mass_from_r_mag(app_mag_r, extinction_r, z, M_to_L_ratio=2.0):


    M_sun_r = 4.67
    

    mag_corrected = app_mag_r - extinction_r
    

    H0_kmsMpc = 70.0
    D_L_Mpc = (c_light * z) / H0_kmsMpc
    dist_mod = 5 * np.log10(D_L_Mpc * 1e6) - 5
    
    
    M_galaxy_r = mag_corrected - dist_mod
    

    L_galaxy_solar = 10.0**(0.4 * (M_sun_r - M_galaxy_r))
    

    stellar_mass_msun = L_galaxy_solar * M_to_L_ratio
    return stellar_mass_msun

from scipy.special import sici

def calculate_sigma_los_nfw(R_proj, M200, c, z_cluster, H0=70.0):
    """
    Calculates the line-of-sight velocity dispersion for an NFW halo.
    This is based on the solution to the Jeans equation, assuming isotropy (beta=0).
    Formula from Mamon & Łokas (2005), ApJ, 633, 794.
    
    Args:
        R_proj (array): Projected radii in kpc.
        M200 (float): Halo mass in Msun.
        c (float): Concentration parameter.
        z_cluster (float): Redshift of the cluster.
        H0 (float): Hubble constant in km/s/Mpc.
    
    Returns:
        sigma_los (array): Line-of-sight velocity dispersion in km/s.
    """
    rho_crit_local = rho_crit_Msunkpc3(H0)
    _, r_s, r200 = rho_s_from_M200_and_c(M200, c, rho_crit_local)
    
    g = lambda x: np.log(1+x) - x/(1+x)
    V200 = G_grav * M200 / r200 # km^2/s^2
    V_c_squared_rs = V200 * c / g(c) # Characteristic velocity squared at r_s
    
    s = R_proj / r_s # Dimensionless projected radius
    

    term1 = np.where(s < 1, 0.5*s**2*(1 - (2/s)*np.arctan(np.sqrt((1-s)/(1+s))))/np.sqrt(1-s**2), 0.0)
    term2 = np.where(s > 1, 0.5*s**2*(np.pi - (2/s)*np.arctanh(np.sqrt((s-1)/(s+1))))/np.sqrt(s**2-1), 0.0)

    
    Si_c, _ = sici(1+c)
    Si_sc, _ = sici((1+c)*(1+s))
    Si_smc, _ = sici((1+c)*(1-s))
    
    term3 = (np.cos(s*(1+c))*(Si_sc - Si_smc) + 
            np.sin(s*(1+c))*(np.pi/2 - 0.5*(sici(2*s*(1+c))[0] - sici(2*(1+c))[0])))
    
    I_s = np.where(s < 1, term1, 0) + np.where(s > 1, term2, 0) \
        + np.where(s == 1, 0.5*(1 - 2/np.pi), 0) # Handle s=1 case
        

    
    sigma_sq = V_c_squared_rs * s * (1+s)**2 * (g(c) / c) * (
        (1 - 2*s**2) * (1/np.sqrt(np.abs(1-s**2))) * np.arccosh(1/s) + 
        0.5*np.pi + np.log(s/2) + (s**2-1+np.log(s/2)) # Simplified terms
    )

    A = 0.46; B = -0.6; C = 0.4
    x = R_proj / r200
    sigma_los_approx = np.sqrt(V200) * A * (1+x)**B * (1+C*x)
    
    return sigma_los_approx # km/s


 # --- 3. CARICAMENTO DATI ---
def load_data_for_gif(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        path = os.path.join(CLUSTER_DATA_PATH, filename)
    
    if filename.lower() == "coma_data.csv":
        df = pd.read_csv(path, skiprows=1, low_memory=False)
        df.columns = ['objid','ra','dec','modelmag_r','modelmagerr_r','extinction_r','redshift','zErr']
        observed_vel = c_light * df['redshift'].dropna().values
        z_cluster = np.nanmedian(df['redshift'])
        # Centro approssimato Coma
        idx_bcg = df['modelmag_r'].idxmin()
        center_ra, center_dec = df.loc[idx_bcg, 'ra'], df.loc[idx_bcg, 'dec']
        ra, dec = df['ra'].values, df['dec'].values
        mags = df['modelmag_r'].values
        exts = df['extinction_r'].values
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None, low_memory=False)
        if df.shape[1] == 7: df[7] = np.nan
        df = df.iloc[:, :8]
        df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
        df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])
        observed_vel = df["RV"].values
        z_cluster = np.nanmedian(observed_vel / c_light)
        idx_bcg = df["bmag"].idxmin()
        center_ra, center_dec = df.loc[idx_bcg, "RAdeg"], df.loc[idx_bcg, "DEdeg"]
        ra, dec = df["RAdeg"].values, df["DEdeg"].values
        mags = df["bmag"].values
        exts = np.zeros_like(mags)

    return observed_vel, z_cluster, ra, dec, center_ra, center_dec, mags, exts
# --- 4. GENERATORE GIF (CON LOGICA NFW E F=0) ---
def generate_gifs_for_dataset(filename):
    print(f"--- Processing {filename} ---")
    try:
        # A. Calcoli Fisici (Con DM=0 come richiesto)
        vels, z, ra, dec, cra, cdec, mags, exts = load_data_for_gif(filename)
        N_obs = len(vels)
        
        DA = angular_diameter_distance_Mpc(z) * 1000
        r_proj = np.maximum(angsep_rad(ra, dec, cra, cdec) * DA, 1.0)
        
        rho_c = rho_crit_Msunkpc3(70.0)
        m_bar = stellar_mass_from_r_mag(mags, exts, z)
        sigma_obs = np.std(vels)
        M200, R200 = estimate_M200_R200_from_sigma(sigma_obs, rho_c)
        c_param = concentration_duffy2008(M200, z, relaxed=False)
        
        # --- FISICA CON f=0 (DM=0) ---
        f_dm = 0.0 
        r_safe = np.maximum(r_proj, 1.0)
        
        # Calcolo masse
        M_bar_tot = np.sum(m_bar)
        # Con f=0, M_tot è uguale a M_bar
        M_tot = M_bar_tot + f_dm * M200
        
        # Calcolo Raggi Viriali
        R_cl_bar = r200_from_M200(M_bar_tot, rho_c)
        R_cl_tot = r200_from_M200(M_tot, rho_c)
        
        # Velocità Centrali
        v_cen_bar = np.sqrt(G_grav * M_bar_tot / (R_cl_bar + 1e-12))
        v_cen_tot = np.sqrt(G_grav * M_tot / R_cl_tot)
        
        # Generazione Random (Simulazione)
        rng = np.random.default_rng(42)
        sigma_fit = np.std(vels)
        
        v_bar = rng.normal(loc=v_cen_bar, scale=sigma_fit, size=N_obs) 
        v_dm = rng.normal(loc=v_cen_tot, scale=sigma_fit, size=N_obs) 
        
        # Statistiche per il box info
        v_mean_val = np.mean(vels)
        v_model_mean = np.mean(v_dm)
        v_model_sigma = np.std(v_dm)
        sigma_bar_val = np.mean(np.sqrt(np.maximum(0.0, G_grav * m_bar / (3.0 * r_safe)))) # Approx media
        
        bins = np.linspace(0, max(vels.max(), v_dm.max())*1.1, 50)
        counts_obs, _ = np.histogram(vels, bins=bins)
        counts_model, _ = np.histogram(v_dm, bins=bins)
        if counts_model.sum() > 0:
            counts_model = counts_model / counts_model.sum() * N_obs
        
        chi2_hist = np.sum(((np.log1p(counts_obs) - np.log1p(counts_model))**2))
        chi2_norm = chi2_hist / len(counts_obs)

        # B. GRAFICA E ANIMAZIONE
        v_max_ax = max(np.max(np.abs(vels)), np.max(np.abs(v_dm))) * 1.2
        r_max_ax = np.max(r_proj) * 1.1
        y_max_ax = np.max(counts_obs) * 1.3
        frames_scat = []
        frames_hist = []
        n_frames = 60
        step_vel = v_max_ax / 20 # 3 fasi da 20 frame
        
        l_obs, l_bar, l_sim = 0, 0, 0
        base_name = filename.replace(".csv", "").replace(".txt", "")
        for i in range(n_frames + 20):
            # Logica Comparsa Sequenziale
            if i < 20: l_obs += step_vel
            elif i < 40: l_bar += step_vel
            elif i < 60: l_sim += step_vel
            
            # --- 1. SCATTER PLOT ---
            fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=100)
            fig1.suptitle(f"Cluster: {base_name}", fontsize=14, fontweight='bold', y=0.98)
            # Blu (Obs)
            m_o = np.abs(vels) <= l_obs
            ax1.scatter(r_proj[m_o], vels[m_o], s=15, c='blue', alpha=0.6, label='Observed Galaxies')
            
            # Rosso (Bar)
            if l_bar > 0:
                m_b = np.abs(v_bar) <= l_bar
                ax1.scatter(r_proj[m_b], v_bar[m_b], s=15, c='red', alpha=0.6, label='Baryonic Component')
            
            # Verde (Sim - NFW con f=0)
            if l_sim > 0:
                m_s = np.abs(v_dm) <= l_sim
                ax1.scatter(r_proj[m_s], v_dm[m_s], s=15, c='green', alpha=0.6, label='Simulated ')
            info_text_sc = (
                f"$\\chi^2/N={chi2_norm:.2f}$\n"
                f"$\\sigma_{{tot}}={v_model_sigma:.1f} km/s$\n"
                f"$DM_{{frac}}={f_dm:.1f}$"
            )
            ax1.text(0.96, 0.96, info_text_sc, transform=ax1.transAxes, fontsize=8, 
                     verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
            ax1.set_xlim(0, r_max_ax); ax1.set_ylim(0, v_max_ax)
            ax1.set_xlabel("Radius [kpc]", fontsize=12); ax1.set_ylabel("Velocity [km/s]", fontsize=12)
            ax1.set_title(f"Scatter Plot ({base_name})", fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            f1 = f"{TEMP_FOLDER}/s_{i:03d}.png"
            plt.savefig(f1); plt.close()
            frames_scat.append(f1)
            
            # --- 2. ISTOGRAMMA ---
            fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=100)
            fig2.suptitle(f"Cluster: {base_name}", fontsize=14, fontweight='bold', y=0.98)
            # Blu
            ax2.hist(vels[m_o], bins=bins, color='blue', alpha=0.8, orientation='horizontal', label='Observed Velocities')
            
            # Rosso
            if l_bar > 0:
                ax2.hist(v_bar[m_b], bins=bins, color='red', alpha=0.6, orientation='horizontal', label='Baryonic Component')
            
            # Verde
            if l_sim > 0:
                ax2.hist(v_dm[m_s], bins=bins, color='green', alpha=0.6, orientation='horizontal', label='Simulated ')
            info_text = (
                f'$<v_{{obs}}>={v_mean_val:.1f} km/s, \\sigma_{{obs}} = {sigma_obs:.1f} km/s $\n' 
                f"$ <v_{{sim}}>={v_model_mean:.1f} km/s, \\sigma_{{sim}} = {v_model_sigma:.1f} km/s $\n"
                f" $ N_{{gal}}={N_obs}, \\sigma_{{bar}} \u2248 {sigma_obs*0.6:.1f} km/s $\n"        
                f" $\\chi^2={chi2_hist:.1f}, \\chi^2/N={chi2_norm:.2f} $"
            )
            ax2.text(0.96, 0.96, info_text, transform=ax2.transAxes, fontsize=8, 
                     verticalalignment='top', horizontalalignment='right', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax2.set_xlim(0, y_max_ax); ax2.set_ylim(0, v_max_ax)
            ax2.set_xlabel('N° of Galaxies', fontsize=12); ax2.set_ylabel('Velocity [km/s]', fontsize=12)
            ax2.set_title(f"Velocity Distribution ({base_name})", fontsize=14, fontweight='bold')
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            f2 = f"{TEMP_FOLDER}/h_{i:03d}.png"
            plt.savefig(f2); plt.close()
            frames_hist.append(f2)
            
            if i % 10 == 0: print(f"Frame {i} done")

        # D. Salvataggio GIF
        base_name = filename.replace(".csv", "").replace(".txt", "")
        
        p1 = os.path.join(OUTPUT_IMG_DIR, f"{base_name}_scatter.gif")
        imageio.mimsave(p1, [imageio.imread(f) for f in frames_scat], duration=0.5, loop=0)
        
        p2 = os.path.join(OUTPUT_IMG_DIR, f"{base_name}_hist.gif")
        imageio.mimsave(p2, [imageio.imread(f) for f in frames_hist], duration=0.5, loop=0)
        
        for f in frames_scat + frames_hist: os.remove(f)
        print(f"Saved {base_name} GIFs")

    except Exception as e:
        print(f"Error {filename}: {e}")

# --- MAIN ---
files = ["coma_data.csv"] + [f for f in os.listdir(CLUSTER_DATA_PATH) if f.endswith('.txt')]
for f in files: generate_gifs_for_dataset(f)
os.rmdir(TEMP_FOLDER)
print("ALL DONE.")