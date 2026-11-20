# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.

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
#from mpl_toolkits.mplot3d import Axes3D
          
import matplotlib.lines as mlines
# Librerie Web/App
import requests
from dotenv import load_dotenv
from fastapi import Request

#from ipywidgets import interact, FloatSlider
from nicegui import ui, app, run , client
#from nicegui_toolkit import inject_layout_tool
from functools import lru_cache
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
#from astroquery.simbad import Simbad
import asyncio           
import logging
import layout
from layout import *
import core
from core import *
from huggingface_hub import HfApi
def create_page():
    #students uploads
    def handle_student_upload(e):
       
   
        file_path = os.path.join(core.SUBMISSIONS_PATH, e.name)
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
        
       
        if core.HF_TOKEN:
            try:
                api = HfApi(token=core.HF_TOKEN)
            
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"Exercises/{e.name}",
                    repo_id=core.DATASET_REPO_ID,
                    repo_type="dataset"
                )
                layout.accessible_notify(f'File "{e.name}" saved in Cloud!', type_='success')
            except Exception as ex:
                print(f"❌ Error during Cloud synchronization: {ex}")
                layout.accessible_notify("Error during Cloud synchronization.", type_='error')
        
        refreshable_submission_list.refresh()

    @ui.refreshable
    def refreshable_submission_list():
     
        files = os.listdir(core.SUBMISSIONS_PATH) if os.path.exists(core.SUBMISSIONS_PATH) else []
        if not files:
            ui.label('No submissions found.').classes('text-gray-400 italic').props('aria-live=polite')
        else:
            with ui.column().classes('w-full gap-2 p-2 bg-slate-800 rounded'):
                for name in files:
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(name).classes('text-sm text-white font-mono')
                       
                        aria_button('Download','download' ,on_click=lambda n=name: ui.download(f'/student_files/{n}')) \
                            .props('flat dense color=primary')
    def get_data_and_images(data_path, img_path, data_extensions=('.txt','.dat', '.csv'), img_extensions=('.jpg', '.png', '.jpeg')):
    
        data_files = [f for f in os.listdir(data_path) if f.endswith(data_extensions)]
        file_map = {}
        for f in data_files:
            base_name = os.path.splitext(f)[0]
            for ext in img_extensions:
                img_file = f"{base_name}{ext}"
                if os.path.exists(os.path.join(img_path, img_file)):
                    file_map[f] = os.path.join(img_path, img_file)
                    break
        return file_map

#constants
    rho_crit = 1.479e2
    G_grav = 4.30091e-6   
    c_light = 3e5         
    M_sun_r = 4.67  

    #function for galaxy
    def c_of_M200(M200, scale=0.3):
        return scale * (11.7 * (M200 / 1e11)**(-0.075))



    def observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul):
   
        V_bar_sq = Vgas * np.abs(Vgas) + Vdisk * np.abs(Vdisk) + Vbul * np.abs(Vbul)
        
       
        V_DM_sq = np.maximum(Vobs**2 - V_bar_sq, 0) 
        
        q = r**2 * (V_DM_sq / r)
        dq_dr = np.gradient(q, r)
        rho_DM = (1.0 / (4.0 * np.pi * G_grav * r**2)) * dq_dr
        return rho_DM

    def get_rhos_rs_from_observed_matching(r, Vobs, Vgas, Vdisk, Vbul, r_match, scale=0.3):
        if r.size == 0 or Vobs.size == 0:
            raise ValueError("Array r o Vobs empty!.")
        
        rho_obs_arr = observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul)
        rho_obs_at_match = np.interp(r_match, r, rho_obs_arr)

       
        if rho_obs_at_match <= 0:
            valid_rhos = rho_obs_arr[rho_obs_arr > 0]
            if valid_rhos.size > 0:
                rho_obs_at_match = valid_rhos[-1] 
            else:
                rho_obs_at_match = 1e-9 

        def to_solve(logM):
            M = 10.0**logM
            R200 = (3.0*M/(4.0*np.pi*200.0*rho_crit))**(1/3)
            c = c_of_M200(M, scale=scale)
            r_s = R200 / c
            rho_s = M / (4*np.pi*r_s**3 * (np.log(1+c) - c/(1+c)))
            x = r_match / r_s
            rho_r = rho_s / (x*(1+x)**2)
         
            return np.log10(np.maximum(rho_r, 1e-12)) - np.log10(rho_obs_at_match)

        try:
           
            logM_sol = brentq(to_solve, 6, 16)
        except (ValueError, RuntimeError):
           
            logM_sol = 11.5 

        M200 = 10.0**logM_sol
        c = c_of_M200(M200, scale=scale)
        R200 = (3.0*M200/(4.0*np.pi*200.0*rho_crit))**(1/3)
        r_s = R200 / c
        rho_s = M200 / (4*np.pi*r_s**3 * (np.log(1+c) - c/(1+c)))
        return rho_s, r_s, M200

    def M_nfw_enclosed(r, rho_s, r_s):
    
        r = np.asarray(r, dtype=float)
        x = r / r_s
        enclosed = 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
        enclosed = np.where(r > 0, enclosed, 0.0)
        return enclosed
    def M_burkert_enclosed(r, rho_s, r_s):
        r = np.asarray(r, dtype=float)
        x = r / r_s
        enclosed = 2.0 * np.pi * rho_s * (r_s**3) * (0.5 * np.log(1.0 + x**2) + np.log(1.0 + x) - np.arctan(x))
        enclosed = np.where(r > 0, enclosed, 0.0)
        return enclosed

    #function for cluster
        

    def comoving_distance_Mpc(z, H0=73.0, Om=0.315):
        integrand = lambda zp: 1.0 / np.sqrt(Om*(1.0+zp)**3 + (1.0-Om))
        chi, _ = quad(integrand, 0.0, z)
        return (c_light / H0) * chi  # Mpc

    def angular_diameter_distance_Mpc(z, H0=73.0, Om=0.315):
        return comoving_distance_Mpc(z, H0, Om) / (1.0 + z)

    def rho_crit_Msunkpc3(H0=73.0):
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
      
        if M200 <= 0:
            return np.zeros_like(r)
            
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
    def concentration_duffy2008(M200_msun, z):
   
        M_safe = min(M200_msun, 5e15)
        
    
        c = 5.71 * (M_safe / (2e12 / 0.7))**(-0.084) * (1 + z)**(-0.47)
        
    
        return max(c, 3.5)
    def concentration_duffy2008bis(M200_msun, z, h=0.7, relaxed=False):

        M_pivot = 2e12 / h  # Msun

        if relaxed:
            A, B, C = 6.71, -0.091, -0.44
        else:
            A, B, C = 5.71, -0.084, -0.47

        return A * (M200_msun / M_pivot)**B * (1 + z)**C

    def filter_cluster_members(df, vel_col="RV"):
        """Filtro a 2 stadi identico a generate_grids_over_5.py e main.tex Sec. 3.3:
        1) finestra +-6000 km/s dal picco, 2) MAD 3-sigma clipping iterativo."""
        if len(df) < 5:
            return df.copy()
        counts, bin_edges = np.histogram(df[vel_col].values, bins=50)
        v_peak = 0.5 * (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1])
        members = df[(df[vel_col] >= v_peak - 6000.0) & (df[vel_col] <= v_peak + 6000.0)].copy()
        for _ in range(5):
            if len(members) < 5:
                break
            v_rel = members[vel_col] - np.median(members[vel_col])
            mad = np.median(np.abs(v_rel))
            sigma_clip = 1.4826 * mad if mad > 0 else np.std(v_rel)
            new_members = members[np.abs(v_rel) <= 3.0 * sigma_clip]
            if len(new_members) == len(members):
                break
            members = new_members
        return members

    def stellar_mass_from_B_mag(bmag, cluster_distance_Mpc, M_to_L_ratio=5.0):
        """Massa stellare in banda B per i cluster Abell SARS (main.tex Eqs. 19-20)."""
        M_sun_B = 5.48
        dist_mod = 5.0 * np.log10(cluster_distance_Mpc * 1e6) - 5.0
        M_abs = bmag - dist_mod
        L_B = 10.0**(-0.4 * (M_abs - M_sun_B))
        return L_B * M_to_L_ratio
    
    def estimate_M200_R200_from_sigma2(sigma_obs, rho_crit_local, G=G_grav,
                                    R200_min=100.0, R200_max=5000.0,
                                    verbose=False):
        """
      
        - sigma_obs in km/s
        - rho_crit_local in Msun/kpc^3
        - G in (kpc * (km/s)^2) / Msun 
        Output (M200 [Msun], R200 [kpc]).
    
        """
        import math
        from scipy.optimize import brentq

    
        if sigma_obs is None or not np.isfinite(sigma_obs):
            raise RuntimeError("sigma_obs non finito")
        if sigma_obs <= 0.0:
        
            if verbose:
                print("estimate_M200_R200_from_sigma: sigma_obs <= 0, use fallback scaling")
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
                print("estimate_M200_R200_from_sigma: error initial func:", e)
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
    def estimate_M200_R200_from_sigma(sigma_obs, rho_crit_local, G=G_grav,
                                    R200_min=100.0, R200_max=5000.0,
                                    verbose=False):
        """
        - sigma_obs in km/s
        - rho_crit_local in Msun/kpc^3
        - G in (kpc * (km/s)^2) / Msun 
        Output (M200 [Msun], R200 [kpc]).
        """
        if sigma_obs is None or not np.isfinite(sigma_obs):
            raise RuntimeError("sigma_obs non finito")
        if sigma_obs <= 0.0:
            return 1e14, 1000.0

       
        R200 = np.sqrt((9.0 * sigma_obs**2) / (4.0 * np.pi * G * 200.0 * rho_crit_local))
        
    
        M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
        
        return M200, R200

    def stellar_mass_from_r_mag(app_mag_r, extinction_r, z, M_to_L_ratio=2.0):
    
    
        M_sun_r = 4.67
        
    
        mag_corrected = app_mag_r - extinction_r
        

        H0_kmsMpc = 73.0
        D_L_Mpc = (c_light * z) / H0_kmsMpc
        dist_mod = 5 * np.log10(D_L_Mpc * 1e6) - 5
        
        
        M_galaxy_r = mag_corrected - dist_mod
        
    
        L_galaxy_solar = 10.0**(0.4 * (M_sun_r - M_galaxy_r))
        

        stellar_mass_msun = L_galaxy_solar * M_to_L_ratio
        return stellar_mass_msun

    from scipy.special import sici

    def calculate_sigma_los_nfw(R_proj, M200, c, z_cluster, H0=73.0):
        R_proj = np.asarray(R_proj, dtype=float)
        # Protezione zero mass
        if M200 <= 0:
            return np.zeros_like(R_proj)
            
        rho_crit_local = rho_crit_Msunkpc3(H0)
        _, r_s, r200 = rho_s_from_M200_and_c(M200, c, rho_crit_local)
        
        V200 = G_grav * M200 / r200 # km^2/s^2
        
      
        A = 0.46; B = -0.6; C = 0.4
        x = R_proj / r200
        sigma_los_approx = np.sqrt(V200) * A * (1+x)**B * (1+C*x)
        
        return np.nan_to_num(sigma_los_approx, nan=0.0) # Sostituisce eventuali NaN residui con 0


    @ui.page('/module2')
    def module2():
        #add custom CSS for the page
        #ui.add_head_html('<script src="https://cdn.tailwindcss.com"></script>')
        ui.run_javascript("""
    // Intercetta la configurazione di Quasar Notify e forza un timeout massimo
    if (window.Quasar && window.Quasar.Notify) {
        window.Quasar.Notify.setDefaults({
            timeout: 3500 // Forza la scomparsa di TUTTE le notifiche (anche errori) dopo 3.5 secondi
        });
    }
""")
        ui.add_head_html('''
    <link rel="stylesheet" href="/static/github.min.css">
''')
        #ui.add_head_html('''    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">    ''')
        #<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        ui.add_head_html("""
    
    <style>
    @font-face {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 300;
        src: url('/static/roboto-v50-latin-300.woff2') format('woff2');
    }
    @font-face {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 400;
        src: url('/static/roboto-v50-latin-regular.woff2') format('woff2');
    }
    @font-face {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 500;
        src: url('/static/roboto-v50-latin-500.woff2') format('woff2');
    }
    @font-face {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 700;
        src: url('/static/roboto-v50-latin-700.woff2') format('woff2');
    }
    body {
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        color: #1a1a1a;
    }
    .q-tab {
    color: #2563eb !important;   /* blu visibile */
    }
    .q-tab--active {
    color: white !important;     /* tab attivo in bianco */
    background-color: #2563eb !important; 
    }
    .title {
        font-size: 32px !important;
        font-weight: 700;
        color: #2563eb; /* blu medio brillante */
        margin-top: 16px;
        margin-bottom: 12px;
    }

    
    .subtitle {
        font-size: 26px !important;
        font-weight: 600;
        color: #3b82f6; /* blu chiaro */
        margin-top: 14px;
        margin-bottom: 10px;
    }

    
    .section {
        font-size: 22px !important;
        font-weight: 600;
        color: #374151; /* grigio elegante */
        margin-top: 12px;
        margin-bottom: 8px;
    }

    
    .info-box {
        background: #e0f2fe;                /* azzurro molto chiaro */
        border-left: 5px solid #0284c7;     /* azzurro più acceso */
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-size: 17px;
        color: #0f172a;                     /* grigio-blu scuro per testo */
        line-height: 1.6;
        text-align: justify;  
    }
      .info-box h3 { color: #0369a1; margin-top: 15px; margin-bottom: 5px; font-weight: bold; font-size: 20px; }
    .info-box h4 { color: #0284c7; margin-top: 10px; margin-bottom: 2px; font-weight: bold; font-size: 18px; }
    .info-box ul { margin-top: 5px; padding-left: 20px; list-style-type: disc; }
    .info-box li { margin-bottom: 4px; }
    .info-box b { color: #0c4a6e; }

    
    .reference-box {
        background: #f3f4f6;                /* grigio chiarissimo, più caldo del bianco */
        border-left: 4px solid #6b7280;     /* grigio medio */
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 16px;
        color: #374151;                     /* grigio medio-scuro */
        font-style: italic;
        line-height: 1.5;
    }

    .reference-box a {
        color: #2563eb;                     /*  blu medio */
        text-decoration: underline;
    }

    
    .warning-box {
        background: #fff7ed;                /* arancione chiarissimo */
        border-left: 6px solid #f97316;     /* arancione */
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        font-size: 17px;
        color: #7c2d12;
        line-height: 1.6;
    }


    .title-on-dark {
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.45);
    }


    .plot-info-box {
    background: #f8fafc;
    border: 1px solid #e6eef8;
    padding: 8px 10px;
    border-radius: 8px;
    margin-top: 8px;
    box-shadow: 0 1px 3px rgba(2,6,23,0.04);
    font-size: 14.5px;
    color: #0f172a;
    line-height: 1.2;          
    }
    .pseudo-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        background-color: #1e1e2e;
        color: #f8f8f2;
        border-radius: 10px;
        padding: 1rem;
        line-height: 1.6;
    }
    .pseudo-code .step {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .plot-info-box .info-row {
    margin: 0;
    padding: 2px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    }


    .plot-info-box .info-row .label {
    margin: 0;
    font-size: 14px;
    color: #475569;
    font-weight: 500;
    padding-right: 8px;
    }


    .plot-info-box .info-row .value {
    margin: 0;
    font-size: 14px;
    color: #0f172a;
    font-weight: 600;
    white-space: nowrap;
    }


    .plot-info-box.compact { padding: 6px 8px; font-size: 14px; line-height: 1.1; }


    .description-on-dark {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3); /* Ombra leggermente più leggera rispetto al titolo */
}
    .success-box {
        background: #ecfdf5;                /* verde chiarissimo */
        border-left: 6px solid #10b981;     /* verde */
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        font-size: 17px;
        color: #064e3b;
        line-height: 1.6;
    }
    

    </style>
    """)
        
        #add custom JS for the page
        #<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.39/Tone.min.js"></script>
        ui.add_head_html("""
    <script src="/static/Tone.min.js"></script>
    <script>
    let synth;
    let isAudioActive = false;

    async function initAudio() {
        const button = document.getElementById('audio-button');

    
        if (isAudioActive) {
            isAudioActive = false;
            if (synth) synth.dispose();
            synth = null;

            button.innerText = 'Activate Audio';
            button.setAttribute('aria-pressed', 'false');
            console.log("🔇 Audio deactivated");
            return;
        }

    
        await Tone.start();
        synth = new Tone.Synth().toDestination();
        isAudioActive = true;

        button.innerText = 'Audio activated';
        button.setAttribute('aria-pressed', 'true');
        console.log("✅ Audio activated");
    }

    function playSeries(series, speed=400) {
        if (!isAudioActive || !synth) return;
        if (!series || series.length === 0) return;
        let i = 0;
        const loop = setInterval(() => {
            if (i >= series.length) { clearInterval(loop); return; }
            const f = 200 + Math.log10(series[i] + 1) * 600;
            synth.triggerAttackRelease(f, "16n");
            i++;
        }, speed);
    }

    function announceAudio(message) {
        const label = document.getElementById('audio_status');
        if (label) label.innerText = message;
    }
    function playSimVelCurve() { playSeries(window.vSimSeries, 400); announceAudio("Playing simulated velocity curve sound");}
    function playSimVelMean() {
        if (!isAudioActive || !synth) return;
        if (!window.vSimMean) return;
        const f = 200 + Math.log10(window.vSimMean + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing simulated velocity mean sound");
    }


    function playObservedVelCurve() { playSeries(window.vObsSeries, 400); announceAudio("Playing observed velocity curve sound");}
    function playObservedVelMean() {
        if (!isAudioActive || !synth) return;
        if (!window.vObsMean) return;
        const f = 200 + Math.log10(window.vObsMean + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing observed velocity mean sound");
    }


    function playBaryonicVelCurve() { playSeries(window.vBarySeries, 400); announceAudio("Playing baryonic velocity curve sound");}
    function playBaryonicVelMean() {
        if (!isAudioActive || !synth) return;
        if (!window.vBarMean) return;
        const f = 200 + Math.log10(window.vBarMean + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing baryonic velocity mean sound");
    }





    function playSimSigmaCurve() { playSeries(window.sigmaSimSeries, 400); announceAudio("Playing simulated velocity distribution sound");}
    function playSimSigmaMean() {
        if (!isAudioActive || !synth) return;
        if (!window.sigmaSimVal) return;
        const f = 200 + Math.log10(window.sigmaSimVal + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing simulated velocity mean sound");
    }


    function playObservedSigmaCurve() {
        playSeries(window.sigmaObsSeries, 400); announceAudio("Playing observed velocity distribution sound");
    }
    function playObservedSigmaMean() {
        if (!isAudioActive || !synth) return;
        if (!window.sigmaObsVal) return;
        const f = 200 + Math.log10(window.sigmaObsVal + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing observed velocity mean sound");
    }


    function playBaryonicSigmaCurve() {
        playSeries(window.sigmaBarSeries, 400); announceAudio("Playing baryonic velocity distribution sound");
    }
    function playBaryonicSigmaMean() {
        if (!isAudioActive || !synth) return;
        if (!window.sigmaBarVal) return;
        const f = 200 + Math.log10(window.sigmaBarVal + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing baryonic velocity mean sound");
    }
    function playDifferenceVelCurve() { playSeries(window.vDiffSeries, 400); announceAudio("Playing difference velocity curves sound");}
    function playDifferenceVelMean() {
        if (!isAudioActive || !synth) return;
        if (!window.vDiffMean && window.vDiffMean !== 0) return;
        const f = 200 + Math.log10(window.vDiffMean + 1) * 600;
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing difference velocity means sound");
    }

    function playDifferenceSigmaCurve() {
    
        playSeries(window.sigmaDiffSeries, 400); announceAudio("Playing difference velocity distributions sound");
    }

    function playDifferenceSigmaMean() {
        if (!isAudioActive || !synth) return;
        if (!window.sigmaDiffMean && window.sigmaDiffMean !== 0) return; 
    
        const f = 200 + Math.log10(window.sigmaDiffMean + 1) * 600; 
        synth.triggerAttackRelease(f, "8n"); announceAudio("Playing difference velocity means sound");
    }
    </script>
    """)
        
   
        #<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg-full.js"></script>
#functions to load and cache data
        def format_sci(val):
            if val == 0: return "0"
            s = f"{val:.2e}"
            mantissa, exp = s.split('e')
            exp = int(exp)
           
            return f"{mantissa} &times; 10<sup>{exp}</sup>"
        @lru_cache(maxsize=32) 
        def get_galaxy_data_cached(filename):
            if not filename: return None
            path = os.path.join(GALAXY_DATA_PATH, filename)
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return None
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                data_lines = []
                header_line = None
                
                for line in lines:
                    sline = line.strip()
                    if not sline: continue
                    clean_line = sline.lstrip('#').strip()
                    # Riconosce la riga di intestazione anche se non ha il #
                    if 'Rad' in clean_line and 'Vobs' in clean_line:
                        header_line = clean_line + '\n'
                    else:
                        data_lines.append(clean_line + '\n')
                
                if header_line is None:
                    header_line = "Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul\n"
                    
                full_data_str = header_line + "".join(data_lines)
                df = pd.read_csv(io.StringIO(full_data_str), sep=r'\s+')
                
                # Forza i numeri e butta via le righe corrotte o doppie intestazioni
                df['Rad'] = pd.to_numeric(df['Rad'], errors='coerce')
                df = df.dropna(subset=['Rad'])
                
                return df
            except Exception as ex:
                print(f"Error loading galaxy {filename}: {ex}")
                return None
        @lru_cache(maxsize=32)
        def get_cluster_data_cached(filename):
            if filename.lower() == "abell1656(coma).csv":
                data_path = os.path.join(DATA_DIR, "Abell1656(Coma).csv")
                if not os.path.exists(data_path):
                    return None
                try:
                    df_coma = pd.read_csv(data_path, skiprows=1)
                    df_coma.columns = ['objid', 'ra', 'dec', 'modelmag_r', 'modelmagerr_r', 'extinction_r', 'redshift', 'zErr']
                    df_coma = df_coma.dropna(subset=['ra', 'dec', 'redshift', 'modelmag_r']).copy()
                    df_coma['RV'] = c_light * df_coma['redshift']
                    return df_coma
                except Exception as e:
                    print(f"Error loading Coma {filename}: {e}")
                    return None

            data_filepath = os.path.join(CLUSTER_DATA_PATH, filename)
            if not os.path.exists(data_filepath):
                return None

            try:
                df = pd.read_csv(data_filepath, sep=r"\s+", header=None)
                if df.shape[1] == 7:
                    df[7] = np.nan
                df = df.iloc[:, :8]
                df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
                for col in ["RAdeg", "DEdeg", "RV", "bmag"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])
                return df
            except Exception as e:
                print(f"Error loading cluster {filename}: {e}")
                return None
        @lru_cache(maxsize=1)
        def get_planets_data_cached():
          
            file_path = os.path.join(DATA_DIR, 'Planets.txt')
            
            if not os.path.exists(file_path):
                print(f"Planets file not found: {file_path}")
                return None

            try:
               
                df = pd.read_csv(file_path, sep=r'\s+', engine='python')
                
             
                df = df[~df['Celestial_Body'].isin(['Sun', 'Moon'])].reset_index(drop=True)
               
                
                return df
            except Exception as e:
                print(f"Error loading planets: {e}")
                return None
        def set_clean_style():
            
            plt.style.use('default') 
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': '#FAFAFA',
                'axes.grid': True,
                'grid.color': '#DDDDDD',     
                'grid.linestyle': '--',
                'grid.alpha': 0.7,
                'axes.edgecolor': '#333333',
                'axes.linewidth': 1.2,
                'font.size': 11,
                'lines.linewidth': 2.5,      
                'figure.autolayout': True    
            })

        set_clean_style()

        main_layout("Dark Matter ")
        
        
        tab_key = 'module2_selected' 
        
        if tab_key not in app.storage.user:
        
            app.storage.user[tab_key] = 'kepler' 
            
        with ui.tabs().classes('w-full justify-center').bind_value(app.storage.user, tab_key) as tabs:
            ui.tab('kepler', label='Kepler laws planets').props('role=tab aria-selected=true')
            ui.tab('gal', label='Galaxy rotation curve').props('role=tab aria-selected=false')
            ui.tab('galdm', label='Galaxy mass & DM').props('role=tab aria-selected=false')
            ui.tab('cluster', label='Cluster velocity distribution').props('role=tab aria-selected=false')
            ui.tab('clusdm', label='Cluster mass & DM').props('role=tab aria-selected=false')

        with ui.column().classes('w-full p-4 gap-4'):
      
            with ui.tab_panels(tabs, value=app.storage.user.get(tab_key, 'kepler')).classes('w-full !bg-transparent'):
                with ui.tab_panel('kepler').props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    ui.label("Review the Kepler's Laws and planetary motion to derive the orbital velocity.").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                    with ui.dialog() as info_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-label="Detailed information about Kepler Laws activity"'):
                        html_info_box(r"""
        <h3>Kepler's Laws Exploration</h3>
        <p>Explore the orbital characteristics of planets in our solar system using Kepler's Laws.</p>
        <p>Observe the model of the Solar System, you can rotate and zoom. The colorbar represents the orbital velocity: planets near the Sun have higher velocities than those farther away.</p>
        <p>Use the dropdown menu to select a planet and see how its orbital parameters compare to the theoretical Keplerian predictions.</p>
        <ul>
            <li><b>Keplerian orbital velocity curve Plot:</b> Shows the orbital velocity vs semi-major axis for each planet, illustrating Kepler's Second Law.</li>
            <li><b>Kepler III law Plot:</b> Displays the orbital period versus semi-major axis, demonstrating Kepler's Third Law.</li>
            <li><b>Planet mass Plot:</b> Presents the mass of each planet vs radius.</li>
        </ul>
        
       
    """).props('role=dialog aria-modal=true aria-label="Descriptive text about Kepler Laws activity"')

                        aria_button("Close", "close the box",on_click=lambda:info_kepler.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    kepler_exercise_state = {'step': 0}

                   
                    kepler_exercises_html = [
                        r"""
                        <h3>Part 1: Dataset Analysis</h3>
                        <ul>
                            <li>Open the App and go to the "Kepler Panel" section of module 2.</li>
                            <li>Download or open the worksheet with the Solar System planets' data by clicking the 'dataset' button.</li>
                            <li>Analyze the columns related to "Velocity" and "Distance" (semi-major axis).</li>
                            <li>Looking at the Solar System data, how would you write the velocity-distance relationship of the Planets?</li>
                            <li>If we draw a graph with velocity on the Y-axis and distance on the X-axis, what shape does the curve take?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Part 2: Building Model </h3>
                        <ul>
                            <li>On your spreadsheet, try to find the mathematical relationship between velocity and distance.</li>
                            <li>Create three new columns and calculate the following values for each planet: <i>v &middot; d</i>, <i>v<sup>2</sup> &middot; d</i>, and <i>v<sup>3</sup> &middot; d</i>. Observe which of these three operations returns an almost constant value.</li>
                            <li>Considering the velocity-distance relationship you chose, create two new columns in the Excel sheet respectively for the square of the velocity (=(F2:F10)^2) and the reciprocal of the semi-major axis (=1/(B2:B10)). Use the data to build a scatter plot by putting the squared velocity on the Y-axis and 1/d on the X-axis. Select the two columns simultaneously and insert the scatter plot using the corresponding button in the 'insert chart' section.</li>
                            <li>What geometric shape did you find in the graph?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Part 3: Building Model </h3>
                        <ul>
                            <li>Considering the line passing through the origin, substitute the equation <i>y = mx</i> with velocity (y) and distance (x) for different values of the dataset and analyze the results.</li>
                            <li>How does the slope (angular coefficient) of the line behave for different values?</li>
                            <li>Based on your scatter plot, what does the velocity-distance relationship equal?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Part 4: Mathematical Derivation </h3>
                        <ul>
                            <li>Consider the context of the planets rotating around the Sun, where the force of gravity and centripetal acceleration act. The goal is to find the rotation velocity to understand what the constant <i>c</i> equals.</li>
                            <li>Write the formula for the gravitational force and the 2nd law force of circular motion.</li>
                            <li>How can we derive the orbital velocity using these two formulas?</li>
                            <li>Write the equivalence: <i>F<sub>grav</sub> = m*a<sub>c</sub></i>.</li>
                            <li>Perform the simplifications of the mass and distance terms on both sides of the equation.</li>
                            <li>Derive <i>v<sup>2</sup></i> by manipulating the equation and then find the velocity (without the square).</li>
                            <li>What terms does the constant include?</li>
                            <li>What relationship is there between the variables you find in the formula?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Part 5: Mathematical Derivation</h3>
                        <ul>
                            <li>To numerically verify the terms of the constant using the dataset data (velocity and distance), find the inverse formula deriving the mass <i>M</i>. Apply the formula by calculating the mass for each planet (the variables are velocity and distance) and calculate its average.</li>
                            <li>What does the obtained mass refer to?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Part 6: Comparison with the App</h3>
                        <ul>
                            <li>The obtained formula contains two constants (G and the mass of the Sun) and two variables (velocity and distance).</li>
                            <li>Create the scatter plot by putting the semi-major axis (distance) on the X-axis and the velocity on the Y-axis (select the corresponding columns).</li>
                            <li>In the App, module 2, Kepler panel, observe the orbital velocity graph ("Keplerian orbital velocity curve Plot").</li>
                            <li>Compare the curve generated by the App with the graph you just created. Are they the same?</li>
                            <li>We call this resulting model, which is valid for the Solar System: <b>Model 1</b>.</li>
                        </ul>
                        """
                    ]


                    with ui.dialog() as instr_kepler_phase1, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-label="Step-by-step exercises on Kepler Laws"'):
                        
                        @ui.refreshable
                        def kepler_exercises_content():
                            current_step = kepler_exercise_state['step']
                          
                            html_info_box(kepler_exercises_html[current_step])
                            
                          
                            with ui.row().classes('w-full justify-between items-center mt-4'):
                               
                                if current_step > 0:
                                    aria_button("Previous", "Go to previous exercise", on_click=lambda: change_step(-1)) \
                                        .classes("!bg-gray-500 hover:!bg-gray-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    ui.label() 
                                
                             
                                ui.label(f"Step {current_step + 1} of {len(kepler_exercises_html)}").classes('text-gray-500 font-bold')
                                
                               
                                if current_step < len(kepler_exercises_html) - 1:
                                    aria_button("Next", "Go to next exercise", on_click=lambda: change_step(1)) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    aria_button("Close", "close the box", on_click=lambda: instr_kepler_phase1.close()) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                        def change_step(delta):
                            kepler_exercise_state['step'] += delta
                            kepler_exercises_content.refresh()

                       
                        instr_kepler_phase1.on('open', lambda: kepler_exercise_state.update({'step': 0}) or kepler_exercises_content.refresh())

                      
                        kepler_exercises_content()
                    
                    
                    def handle_url_submission(url_input, name_input):
                        url = url_input.value
                        student_name = name_input.value
                        
                        if not url or not student_name:
                            layout.accessible_notify('Please enter your name and the sheet link!', type_='warning')
                            return
                        
                    
                        match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
                        if match:
                            sheet_id = match.group(1)
                          
                            export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx'
                            
                            try:
                               
                                response = requests.get(export_url)
                                
                                if response.status_code == 200:
                                
                                    file_name = f"{student_name.replace(' ', '_')}_exercise.xlsx"
                                    file_path = os.path.join(core.SUBMISSIONS_PATH, file_name)
                                    
                                    with open(file_path, 'wb') as f:
                                        f.write(response.content)
                                        
                                  
                                    if core.HF_TOKEN:
                                        try:
                                            api = HfApi(token=core.HF_TOKEN)
                                            api.upload_file(
                                                path_or_fileobj=file_path,
                                                path_in_repo=f"Exercises/{file_name}",
                                                repo_id=core.DATASET_REPO_ID,
                                                repo_type="dataset"
                                            )
                                            layout.accessible_notify(f'File "{file_name}" saved in Cloud!', type_='success')
                                        except Exception as ex:
                                            print(f"❌ Error during Cloud synchronization: {ex}")
                                            layout.accessible_notify("Error during Cloud synchronization.", type_='error')
                                    else:
                                        layout.accessible_notify(f'File "{file_name}" saved locally!', type_='success')
                                
                                    
                            
                                    url_input.value = ''
                                    name_input.value = ''
                                    
                                  
                                    refreshable_submission_list.refresh()
                                else:
                                    layout.accessible_notify('Error! Did you set sharing to "Anyone with the link"?', type_='error')
                            except Exception as e:
                                print(f"❌ Error during cloud download: {e}")
                                layout.accessible_notify("Google connection error.", type_='error')
                        else:
                            layout.accessible_notify('Invalid Google Sheets link.', type_='error')
                   
                    with ui.dialog() as upload_zone_dialog, ui.card().classes('w-full max-w-2xl !bg-slate-900').props('aria-label="Submission area dialog"'):
                        ui.label('Submission Area').classes('text-2xl font-bold text-cyan-400')
                        
                        with ui.tabs().classes('w-full text-white') as upload_tabs:
                            ui.tab('Local File (Offline)', icon='upload_file')
                            ui.tab('Google Sheets (Online)', icon='cloud')
                            
                        with ui.tab_panels(upload_tabs, value='Local File (Offline)').classes('w-full bg-transparent p-0 mt-4'):
                            
                          
                            with ui.tab_panel('Local File (Offline)'):
                                html_info_box(r"""
                                    <b>Offline Work with Excel/Calc</b><br>
                                    1. <b>Download the file:</b> <a href="/dataset/planets_dataset.xlsx" download style="color: #2563eb; text-decoration: underline;">Click here to download the Template</a><br>
                                    2. Work offline on your computer.<br>
                                    3. Use the button below to submit the completed <b>.xlsx</b> file.
                                """)
                              
                                def safe_offline_upload(e):
                                    file_path = os.path.join(core.SUBMISSIONS_PATH, e.name)
                                    with open(file_path, 'wb') as f:
                                        f.write(e.content.read())
                                    
                               
                                    if core.HF_TOKEN:
                                        try:
                                            api = HfApi(token=core.HF_TOKEN)
                                            api.upload_file(
                                                path_or_fileobj=file_path,
                                                path_in_repo=f"Exercises/{e.name}",
                                                repo_id=core.DATASET_REPO_ID,
                                                repo_type="dataset"
                                            )
                                            layout.accessible_notify(f'File "{e.name}" saved in locale and on Cloud!', type_='success')
                                        except Exception:
                                            layout.accessible_notify(f'You are offline. File "{e.name}" saved only on the computer!', type_='warning')
                                    else:
                                        layout.accessible_notify(f'File "{e.name}" saved locally!', type_='success')
                                    
                                    refreshable_submission_list.refresh()

                                ui.upload(
                                    label='Select your completed Excel file',
                                    on_upload=safe_offline_upload, 
                                    auto_upload=True
                                ).classes('w-full mt-2').props('tabindex=0 aria-label="Upload your dataset file"')

                         
                            with ui.tab_panel('Google Sheets (Online)'):
                                warning_box("Warning: this function requires internet connection.")
                                
                                html_info_box(r"""
                                    <b>Upload via link</b><br>
                                    1. Setting the share to <b>"Everyone with the link"</b>.<br>
                                    2. Past link here.
                                """)
                                student_name = ui.input('Name/ Group').classes('w-full bg-slate-800 text-white px-2 rounded mt-2')
                                sheet_link = ui.input('Paste Google Sheets link').classes('w-full bg-slate-800 text-white px-2 rounded mt-2')
                                
                                aria_button('Dowload and Save', 'cloud_download', 
                                            on_click=lambda: handle_url_submission(sheet_link, student_name)) \
                                    .classes('w-full !bg-green-600 hover:!bg-green-800 text-white font-bold py-2 mt-4 rounded')

                        ui.separator().classes('bg-slate-700 my-4')
                        
                        ui.label('File saved!').classes('text-lg font-bold text-white')
                        refreshable_submission_list() 
                        
                        aria_button('Close', 'close', on_click=upload_zone_dialog.close).classes('!bg-orange-500 mt-4')
                    with ui.dialog() as data_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Kepler\'s laws information"'):
                        info_box( "**Dataset variables**: Celestial_Body (name of the planet), SemiMajorAxis(km) (orbital radius in km), Velocity(km/s) (orbital velocity in km/s), Period(days) (orbital period in days), Mass(kg) (mass of the planet in kg).")
                        reference_box(
        """**Dataset reference**: [NASA-SSD](https://ssd.jpl.nasa.gov/planets/) ;[Orbital Mechanics](https://orbital-mechanics.space/reference/planetary-parameters) ;Ryan S. Park, William M. Folkner, James G. Williams, and Dale H. Boggs. The JPL Planetary and Lunar Ephemerides DE440 and DE441. The Astronomical Journal, 161(3):105, February 2021.; Brandon Rhodes. Skyfield: high precision research-grade positions for planets and Earth satellites generator. July 2019""")
                        
                        template_url ="https://docs.google.com/spreadsheets/d/1NmaAMkyIgv3Ln_6-DNEECDQfPw64kItJ/copy"
                        
                        with ui.row().classes('w-full justify-center gap-4 mt-4'):
        
                            aria_button('Build Google Sheet', 'Create copy Google Sheet', 
                on_click=lambda: ui.run_javascript(f'window.open("{template_url}", "_blank")')) \
        .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
        .props('aria-label="Create personal copy of Google Sheets"')

    
        
                            aria_button('Download Dataset', 'download', 
                                        on_click=lambda: ui.run_javascript('window.location.href = "/dataset/planets_dataset.xlsx"')) \
                                .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
                                .props('aria-label="Download the Kepler Excel dataset"')

                            aria_button('Upload Exercises', 'cloud_upload', 
                on_click=upload_zone_dialog.open) \
        .classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded') \
        .props('aria-label="Upload your completed exercises to the app folder"')
                            #aria_button('Upload Exercises', 'upload', on_click=lambda: ui.run_javascript('window.open("https://www.dropbox.com/request/RZWNDfeDXEN59yzNN8TW", "_blank")')).classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded').props('aria-label="Upload your exercises to Dropbox"')
                        aria_button("Close", "close the box",on_click=lambda:data_kepler.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.dialog() as cur_kep, ui.card().classes('p-4 w-full max-w-[600px]').props('role=dialog aria-modal=true aria-label="Interesting fact about Saturn"'):
                        html_info_box(r"""
    <h3>Saturn's Density</h3>
    <p>While Kepler's laws apply to all planets, their compositions vary wildly. Saturn is the only planet in our Solar System that is less dense than water.</p>
    <p>If you could find a bathtub large enough (and filled with water) to hold it, <b>Saturn would float!</b></p>
    """)
                        reference_box("**Source:** [NASA Science ](https://science.nasa.gov/saturn/facts/)")
                        aria_button("Close", "close", on_click=lambda:cur_kep.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                    
                
                    G = 6.67430e-20            # km^3 / kg / s^2 
                    M_sun = 1.98842e30         # kg
                    app.add_static_files('/planet_images', PLANETS_IMG_PATH)

                
                    file_path = os.path.join(DATA_DIR, 'Planets.txt')  
                    

                
                    df0 = get_planets_data_cached()

                    if df0 is None:
                        ui.label("Error loading Planets.txt").classes("text-red-500")
                        return

                
                    planets = df0['Celestial_Body'].tolist()
                    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(planets))))
                    planet_colors = {p: colors[i % len(colors)] for i, p in enumerate(planets)}

                
                    def find_planet_image(pname):
                        exts = ['.jpg', '.jpeg', '.png', '.webp']
                        for ext in exts:
                            fname = f"{pname}{ext}"
                            if os.path.exists(os.path.join(PLANETS_IMG_PATH, fname)):
                                return f"/planet_images/{fname}"

                            fname2 = f"{pname.capitalize()}{ext}"
                            if os.path.exists(os.path.join(PLANETS_IMG_PATH, fname2)):
                                return f"/planet_images/{fname2}"

                    
                        fallback = "no_image.png"
                        if os.path.exists(os.path.join(PLANETS_IMG_PATH, fallback)):
                            return f"/planet_images/{fallback}"

                        return None


                

                
                    @ui.refreshable
                    def plot_velocity(selected_planet, df=df0):
                        component = None
                        row = df.loc[df['Celestial_Body'] == selected_planet].iloc[0]
                        a_sel = float(row['SemiMajorAxis(km)'])
                        v_sel = float(row['Velocity(km/s)'])

                    
                        a_all = df.loc[(df['SemiMajorAxis(km)'] > 0) & (df['Velocity(km/s)'] > 0), 'SemiMajorAxis(km)'].values
                        a_min = a_all.min() * 0.6 if len(a_all) > 0 else 1e7
                        a_max = a_all.max() * 1.4 if len(a_all) > 0 else 1e9
                        a_range = np.logspace(np.log10(a_min), np.log10(a_max), 400)
                        v_kep_range = np.sqrt(G * M_sun / a_range)

                        with ui.pyplot(figsize=(8, 4), close=True, clear=True) as p:
                            component = p
                            plt.plot(a_range / 1e6, v_kep_range, '--', color='gray', lw=0.8,
                                    label=r'Keplerian: $v(a)=\sqrt{\dfrac{GM_\odot}{a}}$')
                            
                            for pnm in planets:
                                pa = float(df.loc[df['Celestial_Body'] == pnm, 'SemiMajorAxis(km)'].iloc[0])
                                pv = float(df.loc[df['Celestial_Body'] == pnm, 'Velocity(km/s)'].iloc[0])
                                plt.scatter(pa / 1e6, pv, color=planet_colors[pnm], s=10, zorder=5)
                                plt.text(pa / 1e6 * 1.03, pv * 1.03, pnm, fontsize=10, color=planet_colors[pnm])
                                
                            plt.scatter(a_sel / 1e6, v_sel, edgecolors='k', linewidths=0.8, facecolors='none', s=15, zorder=6)
                            plt.xlabel("Semi-major axis (10^6 km)",fontsize=10)
                            plt.ylabel("Orbital velocity (km/s)",fontsize=10)
                            plt.title("Keplerian Orbital Velocity Curve",fontsize=12, fontweight='bold')
                            plt.legend(fontsize=10)
                            plt.grid(True,alpha=0.1)
                            plt.xticks(fontsize=10)
                            plt.yticks(fontsize=10)
                            plt.tight_layout(pad=0.3)
                            return component

            
                    @ui.refreshable
                    def plot_kepler_III(selected_planet, df=df0):
                        kepler=None
                        a_sel = None 
                        T_sel = None
                        if selected_planet and selected_planet in df['Celestial_Body'].values:
                            row = df.loc[df['Celestial_Body'] == selected_planet].iloc[0]
                            a_sel = float(row['SemiMajorAxis(km)'])
                            T_sel = float(row['Period(days)'])
                        
                    
                        a_all = df.loc[(df['SemiMajorAxis(km)'] > 0), 'SemiMajorAxis(km)'].values
                        a_min = a_all.min() * 0.6 if len(a_all) > 0 else 1e7
                        a_max = a_all.max() * 1.4 if len(a_all) > 0 else 1e9
                        a_range = np.logspace(np.log10(a_min), np.log10(a_max), 400)
                        T_theory = 2 * np.pi * np.sqrt(a_range**3 / (G * M_sun)) / (3600 * 24)

                        with ui.pyplot(figsize=(8, 6), close=True, clear=True) as k:
                            kepler=k
                            
                            plt.loglog(a_range / 1e6, T_theory, '--', lw=2, color='gray',
                                label=r'Kepler III: $T(a)=2\pi\sqrt{\dfrac{a^3}{GM_\odot}}$')
                            for pnm in planets:
                                pa = float(df.loc[df['Celestial_Body'] == pnm, 'SemiMajorAxis(km)'].iloc[0])
                                pT = float(df.loc[df['Celestial_Body'] == pnm, 'Period(days)'].iloc[0])
                                plt.scatter(pa / 1e6, pT, color=planet_colors[pnm], s=70)
                                plt.text(pa / 1e6 * 1.03, pT * 1.03, pnm, fontsize=9, color=planet_colors[pnm])
                            if a_sel is not None and T_sel is not None:
                                plt.scatter(a_sel / 1e6, T_sel, facecolors='none', edgecolors='k', s=200, linewidths=1.2, zorder=6)
                            plt.xlabel("a (10^6 km)")
                            plt.ylabel("Period (days)")
                            plt.title("Kepler III: Period vs Semi-major axis", fontweight='bold')
                            plt.legend(fontsize=9)
                            plt.grid(True, which='both', ls='--', alpha=0.4)
                            plt.tight_layout()
                            return kepler

                    @ui.refreshable
                    def plot_mass(selected_planet, df=df0):
                        mass=None
                        a_sel = None
                        m_sel = None
                        if selected_planet and selected_planet in df['Celestial_Body'].values:
                            try:
                                row = df.loc[df['Celestial_Body'] == selected_planet].iloc[0]
                                a_sel = float(row['SemiMajorAxis(km)'])
                                m_sel = float(row['Mass(kg)'])
                            except IndexError:
                                pass
                        with ui.pyplot(figsize=(6, 3.5), close=True, clear=True) as m:
                            mass=m
                            for pnm in planets:
                                pa = float(df.loc[df['Celestial_Body'] == pnm, 'SemiMajorAxis(km)'].iloc[0])
                                pm = float(df.loc[df['Celestial_Body'] == pnm, 'Mass(kg)'].iloc[0])
                                plt.scatter(pa / 1e6, pm, color=planet_colors[pnm], s=70)
                                plt.text(pa / 1e6 * 1.03, pm * 1.03, pnm, fontsize=9, color=planet_colors[pnm])
                                
                        
                            
                            plt.xscale('log'); plt.yscale('log')
                            plt.xlabel("a (10^6 km)")
                            plt.ylabel("Mass (kg)")
                            plt.title("Mass vs Semi-major axis",fontweight='bold')
                            plt.grid(True, which='both', ls='--', alpha=0.4)
                            plt.tight_layout()
                            return mass
                    
                    def compute_density(row):
                        try:
                            r_km = float(row["MeanRadius(km)"])
                            m_kg = float(row["Mass(kg)"])
                        except (ValueError, TypeError):
                            return None

                        if r_km <= 0 or m_kg <= 0:
                            return None

                        r_cm = r_km * 1e5
                        volume = (4/3)*np.pi*(r_cm**3)
                        density = (m_kg * 1000) / volume   # kg → g

                        return density


                    @ui.refreshable
                    def info_panel(selected_planet, df=df0):
                    
                        row = df.loc[df['Celestial_Body'] == selected_planet].iloc[0]

                        bg_color = "#e5f3ff"
                        border_color = "#c1e0ff"
                        title_color = "#1a5f9e"
                        html = f"""
                        <div style='display:flex; gap:16px; align-items:center;
                                    background:{bg_color}; border:1px solid {border_color};
                                    padding:14px; border-radius:12px;'>
                        <div style='flex:0 0 180px;'>
                        """

                    
                        img_path = find_planet_image(selected_planet)
                        if img_path:
                        
                            img_tag = f"<img src='{img_path}' alt='{selected_planet}' style='width:180px; height:180px; object-fit:cover; border-radius:10px;'/>"

                        else:
                            img_tag = f"<div style='width:180px; height:180px; display:flex; align-items:center; justify-content:center; background:#f0f0f0; color:#666; border-radius:10px;'>No image</div>"

                        html += img_tag
                        html += "</div>"

                        density = compute_density(row)
                        density_text = f"{density:.2f} g/cm³" if density else "—"

                    
                        incl = row.get('Inclination(deg)', '')
                        ecc = row.get('Eccentricity', '')
                        gm = row.get('GM(km3/s2)', '')

                        html += f"""
                        <div style='flex:1;'>
                            <h2 style='margin:0; color:{title_color};font-size: 24px;'>{selected_planet}</h2>
                            <div style='margin-top:8px; font-size:14px; color:#111; line-height:1.6;'>
                            <b>Semi-major axis:</b> {row['SemiMajorAxis(km)']:.3e} km<br/>
                            <b>Perihelion:</b> {row['Perihelion(km)']:.3e} km &nbsp;&nbsp;
                            <b>Aphelion:</b> {row['Aphelion(km)']:.3e} km<br/>
                            <b>Orbital speed:</b> {row['Velocity(km/s)']:.3f} km/s &nbsp;&nbsp;
                            <b>Period:</b> {row['Period(days)']:.2f} days<br/>
                            <b>Mass:</b> {row['Mass(kg)']:.3e} kg &nbsp;&nbsp;
                            <b>GM:</b> {gm}<br/>
                            <b>Density:</b> {density_text} &nbsp;&nbsp;
                            <b>Inclination:</b> {incl}<br/>
                            <b>Eccentricity:</b> {ecc}
                            </div>
                        </div>
                        </div>
                        """

                
                        ui.html(html).props('role=region tabindex=0 aria-label="Planet information panel"')

                    def open_kepler_info_dialog():
                        with ui.dialog() as comp_kepler_dialog, ui.card().classes('p-0 w-full min-w-[1000px] max-w-[95vw] h-[85vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="kepler-title"'):
                            
                            with ui.column().classes('w-full h-full bg-white flex flex-col'):
                                
                                with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="kepler-title"'):
                                    ui.label('Kepler Laws & Mass Analysis').classes('text-xl font-bold').props('id="kepler-title" role=heading aria-level=2 tabindex=0')
                                    aria_button('Close', 'close', on_click=comp_kepler_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                                with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                    t_kepler3 = ui.tab('Kepler III Plot')
                                    t_mass = ui.tab('Planets Mass Plot')
                                    t_resources = ui.tab('Resources & Materials')

                                with ui.tab_panels(tabs, value=t_kepler3).classes('w-full flex-1 p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                    
                                   
                                    with ui.tab_panel(t_kepler3).classes('flex flex-col items-center justify-start gap-4'):
                                       
                                        kepler_3_plot_ref = plot_kepler_III(None)
                                        kepler_3_plot_ref.classes('w-full max-w-[800px]')

                                   
                                    with ui.tab_panel(t_mass).classes('flex flex-col items-center justify-start gap-4'):
                                      
                                        mass_plot_ref = plot_mass(None)
                                        mass_plot_ref.classes('w-full max-w-[800px]')

                                    
                                    with ui.tab_panel(t_resources).classes('flex flex-col items-center justify-start pt-10 gap-8'):
                                       
                                        with ui.column().classes('w-full max-w-[800px] gap-4'):
                                        #    with ui.card().classes('p-6 bg-white shadow-md rounded-lg flex flex-col items-center gap-4 w-full max-w-[600px] border-t-4 border-green-500'):
                                        
                                                
                                            with ui.card().classes('p-6 bg-white shadow-md rounded-lg flex flex-col items-center gap-4 w-full max-w-[600px] border-t-4 border-green-500'):
                                                ui.label("PhET Interactive Simulation").classes("text-xl font-bold text-green-700")
                                                ui.label("Explore orbital mechanics with this interactive physics simulator from the University of Colorado.").classes("text-center text-gray-600")
                                                aria_button(
                                                    'Open PhET Simulator', 
                                                    "Explore the interactive Kepler laws simulator",
                                                    on_click=lambda: ui.run_javascript("window.open('https://phet.colorado.edu/sims/html/keplers-laws/latest/keplers-laws_all.html', '_blank')")
                                                ).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-3 px-6 rounded shadow")

                                           # with ui.card().classes('p-6 bg-white shadow-md rounded-lg flex flex-col items-center gap-4 w-full max-w-[600px] border-t-4 border-blue-500'):
                                           #     ui.label("Supplementary Materials").classes("text-xl font-bold text-blue-700")
                                           #     ui.label("View the introductory presentation slides on Cosmology and Dark Matter.").classes("text-center text-gray-600")
                                            #    aria_button(
                                            #        'Open PDF Slides', 
                                            #        'Open Introduction Presentation',
                                            #        on_click=lambda: ui.run_javascript('window.open("/slides/cosmo_dark_matter.pdf", "_blank")')
                                             #   ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-3 px-6 rounded shadow")

                        comp_kepler_dialog.open()
                    with ui.row().classes('w-full gap-4 justify-center'):

                        
                       
                        aria_button("Instructions", "Instruction for Kepler panel",on_click=safe_click(lambda: [info_kepler.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                       
                        
                        aria_button(
            'Scientific Info', 
            'Open Comprehensive Kepler Information Dialog', 
            on_click=lambda: [open_kepler_info_dialog(), ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                        
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_kep.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                        

                        
                    with ui.row().classes('w-full items-center justify-center gap-8'): 
                        
                    
                        with ui.column().classes('w-full items-center justify-center gap-4'):
                        # Selettore pianeta in alto al centro
                            with ui.row().classes('w-full justify-center items-center'):
                                ui.label("Select a Planet").style("font-size:20px; font-weight:bold; margin-right: 10px;").props('id=planet_selector_label role=heading aria-level=3 tabindex=0')
                                selector = ui.select(
                                    planets,
                                    value="Earth" if "Earth" in planets else planets[0],
                                    label="Planet"
                                ).classes('w-48').props('id=planet_selector aria-labelledby=planet_selector_label role=listbox tabindex=0')

                            with ui.row().classes('w-full items-start justify-center gap-8 mt-2'): 
                             
                                with ui.column().classes('flex-1 min-w-[350px] max-w-[800px] items-center'):
                                    @ui.refreshable
                                    def render_solar_system_3d():
                                        fig = go.Figure()
                                        custom_colorbar = dict(
                                            title=dict(text="Velocity<br>(km/s)", font=dict(color='white', size=12), side='top'),
                                            tickfont=dict(color='white', size=11), x=-0.12, thickness=15
                                        )
                                        
                                        fig.add_trace(go.Scatter3d(
                                            x=[0], y=[0], z=[0], mode='markers',
                                            marker=dict(size=18, color='yellow', symbol='circle'),
                                            name='Sun (99.8% Mass)', hoverinfo='name'
                                        ))
                                        
                                        r_max_plot = 10.0
                                        if df0 is not None and not df0.empty:
                                            v_min_sol = df0['Velocity(km/s)'].min()
                                            v_max_sol = df0['Velocity(km/s)'].max()
                                            c_max_bright_sol = v_max_sol * 1.15
                                            r_max_plot = (df0['SemiMajorAxis(km)'].max() / 1e6) * 1.05

                                            for idx, row in df0.iterrows():
                                                p_name = row['Celestial_Body']
                                                a_scaled = float(row['SemiMajorAxis(km)']) / 1e6
                                                v_scaled = float(row['Velocity(km/s)'])
                                                m_kg = float(row['Mass(kg)'])
                                                
                                                theta_p = np.linspace(0, 2*np.pi, 100)
                                                fig.add_trace(go.Scatter3d(
                                                    x=a_scaled*np.cos(theta_p), y=a_scaled*np.sin(theta_p), z=np.zeros_like(theta_p), 
                                                    mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=1), 
                                                    showlegend=False, hoverinfo='skip'
                                                ))
                                                
                                                angle = np.random.uniform(0, 2*np.pi)
                                                fig.add_trace(go.Scatter3d(
                                                    x=[a_scaled*np.cos(angle)], y=[a_scaled*np.sin(angle)], z=[0], mode='markers', 
                                                    marker=dict(
                                                        size=8, color=[v_scaled], colorscale='turbo', 
                                                        cmin=v_min_sol, cmax=c_max_bright_sol, showscale=False
                                                    ), 
                                                    name=f"{p_name}", text=[p_name], customdata=[[a_scaled, m_kg]],
                                                    hovertemplate="<b>%{text}</b><br>Radius: %{customdata[0]:.1f}x10^6 km<br>Velocity: %{marker.color:.1f} km/s<br>Mass: %{customdata[1]:.2e} kg"
                                                ))
                                            
                                            fig.add_trace(go.Scatter3d(
                                                x=[None], y=[None], z=[None], mode='markers',
                                                marker=dict(colorscale='turbo', cmin=v_min_sol, cmax=c_max_bright_sol, showscale=True, colorbar=custom_colorbar),
                                                showlegend=False, hoverinfo='skip'
                                            ))

                                        fig.update_layout(
                                            title=dict(text="3D Solar System", font=dict(color='white', size=14), x=0.5, y=0.98),
                                            scene=dict(
                                                xaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), 
                                                yaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), 
                                                zaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), 
                                                bgcolor='rgb(15, 23, 42)', aspectmode='cube'
                                            ),
                                            margin=dict(l=0, r=0, b=0, t=30), 
                                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                            legend=dict(font=dict(color='white'), yanchor="top", y=0.95, xanchor="left", x=0.01)
                                        )
                                        ui.plotly(fig).classes('w-full h-[550px] rounded-lg shadow-lg')
                                    render_solar_system_3d()

                                # --- COLONNA DESTRA: Box Pianeta + Plot Velocità ---
                                with ui.column().classes('flex-1 min-w-[350px] max-w-[600px] items-center shrink-0 gap-4'):
                                    info_panel(selector.value)
                                    plot_velocity(selector.value).classes('max-w-full')
                                    
                                    with ui.row().classes('w-full justify-center items-center gap-4 mt-2'):
                                        aria_button("Activity: Kepler", "Instructions for Kepler velocity plot", on_click=lambda: instr_kepler_phase1.open()).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                        aria_button("Dataset", "Info Kepler planets dataset", on_click=safe_click(lambda: [data_kepler.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                
                        selector.on('update:model-value', lambda e: (
                            plot_velocity.refresh(selector.value),
                            info_panel.refresh(selector.value)
                        ))


                        
                                
#panel galaxy rotation curve
                with ui.tab_panel('gal').props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    ui.label(
    "Analyze the rotation speeds of spiral galaxies to solve the discrepancy between observations and predictions."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                    gal_state= {
            'r_ngc': np.array([]),
            'v_obs_ngc': np.array([]),
            'v_gas_ngc': np.array([]),
            'v_disk_ngc': np.array([]),
            'v_bul_ngc': np.array([]),
            'v_err_ngc': np.array([]),
         
            'DATA_LOADED': False,
            'current_galaxy_name': None,
            'selected_file': None,
            'chi2_points': [],      
            'manual_points': [],    
        }
                  
                 
                    G_grav = 4.30091e-6   # kpc (km/s)^2 / M_sun
                
                  
                    default_galaxy = "NGC3198.txt"
                    
                    galaxy_state = {
        "select": default_galaxy
    }
                    
                    formula_pts = {"a": None, "b": None, "c": None, "fa": None, "fb": None, "fc": None}
                    
                    parabolic_state = {
                            "points": [],
                            "history": [],
                            "plot_points": [],
                            "iteration": 0
                        }
                    
                
                    chi2_state = {
    'slider_result': " ---" 
}
                    
                    with ui.dialog() as info_galaxy, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Galaxy rotation curve information"'):
                        html_info_box(r"""
    <h3>NGC3198 Rotation Curve</h3>
    <p>Explore the rotation curve of the NGC3198 galaxy to understand the evidence for Dark Matter.</p>
    
    <h4>Legend</h4>
    <ul style="margin-bottom: 10px;">
         <li><b>Red curve:</b> Baryonic prediction (stars + gas). It follows a Keplerian decline (<span class="math">\(v \propto 1/\sqrt{r}\)</span>).</li>
         <li><b>Blue points:</b> Real observational data with error bars. It remains "flat" at large radii.</li>
         <li><b>Green curve:</b> Total simulation (Baryons + Dark Matter Halo).</li>
    </ul>
    <hr class="my-4">

    <h3>The Burkert Dark Matter Profile</h3>
    <p>To simulate the halo, we use the <b>Burkert profile</b>, which describes the density of dark matter with a cored center:</p>
    
    <div style="text-align:center; margin: 15px 0;">
        <span class="math">$$ \rho_{Burkert}(r) = \frac{\rho_s}{\left(1 + \frac{r}{r_s}\right)\left(1 + \left(\frac{r}{r_s}\right)^2\right)} $$</span>
    </div>

    <p>Where <span class="math">\(\rho_s\)</span> is the central density and <span class="math">\(r_s\)</span> is the core radius.</p>
    <h4>Adapted to the Dataset</h4>
    <p>The values for <span class="math">\(\rho_s\)</span>, <span class="math">\(r_s\)</span>, and the stellar mass-to-light ratios (<span class="math">\(\Upsilon_{\mathrm{disk}}, \Upsilon_{\mathrm{bulge}}\)</span>) are <b>fitted directly to each galaxy's SPARC rotation curve</b> via global <b>Differential Evolution</b> <span class="math">\(\chi^2\)</span> minimization with log-normal priors centered at <span class="math">\(\Upsilon_{\mathrm{disk}} = 0.5\)</span> and <span class="math">\(\Upsilon_{\mathrm{bulge}} = 0.7\)</span>.</p>

       
    <hr class="my-4">

    <h3>Activity Instructions</h3>
    <ol style="margin-top: 10px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b>Observe the Discrepancy:</b>  Notice that the <b>Red curve</b> drops at large distances, while the observed <b>Blue points</b> remain constant. This indicates that visible mass is insufficient to explain the high orbital velocities.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Add Dark Matter:</b> Move the <b>Slider</b> (fraction <span class="math">\(f\)</span>) to increase the density of the Dark Matter halo.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Listen to the Data (Sonification):</b> As you adjust the slider, listen to the pitch variations:
            <ul style="margin-top: 5px; list-style-type: disc; margin-left: 20px;">
                <li><b>Dropping Pitch:</b> Corresponds to the Keplerian decline (no Dark Matter).</li>
                <li><b>Constant High Pitch:</b> Corresponds to the flat rotation curve, indicating the correct amount of Dark Matter is present.</li>
            </ul>
        </li>
        <li style="margin-bottom: 8px;">
            <b>Fit the Data:</b> Adjust the slider until the <b>Green simulated curve</b> perfectly overlaps the observed data points.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Minimize <span class="math">\(\chi^2\)</span>:</b> Watch the Chi-squared value. The best fit corresponds to the minimum <span class="math">\(\chi^2\)</span>, representing the most statistically accurate quantity of Dark Matter.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Analyze Mass Distribution:</b> Observe the <i>Mass vs Radius</i> plot to see that while baryons dominate the center, Dark Matter dominates the outer halo.
        </li>
    </ol>
    <h4>References</h4>
    <div style="font-size: 0.9em; line-height: 1.4; color: #444;">
        <p style="margin-bottom: 5px;"><b>Burkert Profile:</b> Burkert (1995), "The Structure of Dark Matter Halos in Dwarf Galaxies".</p>
        <p style="margin-bottom: 5px;"><b>Mass Models:</b> Lelli F. et al. (2016), "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry...".</p>
        <p style="margin-bottom: 5px;"><b>Halo Fitting Benchmark:</b> Li P. et al. (2020), "A Comprehensive Catalog of Dark Matter Halo Models for SPARC Galaxies", ApJS, 247, 31.</p>
        <p style="margin-bottom: 5px;"><b>Mass-to-Light Ratios:</b> McGaugh S. et al. (2016), "Radial Acceleration Relation in Rotationally Supported Galaxies", PRL, 117, 201101.</p>
    </div>
""").props("id=ngc3198-info role=document aria-live=polite")
                        aria_button("close",'close',on_click=lambda:info_galaxy.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as data_galaxy,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('id=dataset-galaxy role=document aria-live=polite'):
                    
                        info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                    
                        reference_box(
        """**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')
                        template_url_gal ="https://docs.google.com/spreadsheets/d/1gvHeXHzKx8DcJgpZdXX1iUYvs2W7_sDd/copy"
                        
                        with ui.row().classes('w-full justify-center gap-4 mt-4'):
        
                            aria_button('Build Google Sheet', 'Create copy Google Sheet', 
                on_click=lambda: ui.run_javascript(f'window.open("{template_url_gal}", "_blank")')) \
        .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
        .props('aria-label="Create personal copy of Google Sheets"')

    
                         
        
                           
                            aria_button('Download Dataset', 'download', 
                                        on_click=lambda: ui.run_javascript('window.location.href = "/dataset/Galaxies_Datasets.xlsx"')) \
                                .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
                                .props('aria-label="Download the Excel dataset"')

                           
                            aria_button('Upload Exercises', 'cloud_upload', on_click=upload_zone_dialog.open) \
        .classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded')
                            #aria_button('Upload Exercises', 'cloud_upload',  on_click=lambda: ui.run_javascript('window.open("https://www.dropbox.com/request/RZWNDfeDXEN59yzNN8TW", "_blank")')).classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded').props('aria-label="Upload your completed exercises to Dropbox"')
                        aria_button("close",'close',on_click=lambda:data_galaxy.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
    
                    with ui.dialog() as cur_rot, ui.card().classes('p-4 w-full max-w-[600px]').props('id=curiosity-rotation role=document aria-live=polite'):
                        html_info_box(r"""
                        <h3>We live in a Dark Halo</h3>
                        <p>Our own galaxy, the Milky Way, is spinning much faster than it should be based on the visible stars and gas. At our distance from the center (26,000 light-years), the Sun orbits at ~220 km/s.</p>
                        <p>If there were no Dark Matter, the Sun would fly off into intergalactic space! We are currently embedded in a massive, invisible <b>Dark Matter Halo</b> that holds our galaxy together.</p>
                        """)
                        reference_box("**Source:** [ESA Gaia](https://sci.esa.int/web/hubble/-/61198-hubble-and-gaia-accurately-weigh-the-milky-way-heic1905)")
                        aria_button("Close", "close", on_click=lambda:cur_rot.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                    
                    
                    
                   
                        
                    @ui.refreshable
                    def update_image():
                        select = gal_state.get('selected_file') or default_galaxy
                        exts = ['.jpg', '.jpeg', '.png', '.webp']
                        base_name = os.path.splitext(select)[0]
                        image_filename = None
                        for ext in exts:
                            temp_name = base_name + ext
                            if os.path.exists(os.path.join(GALAXY_IMG_PATH, temp_name)):
                                image_filename = temp_name
                                break
                        #image_path = os.path.join(GALAXY_IMG_PATH, image_filename)

                    
                        if image_filename:
                            web_path = f"/galaxy_img/{image_filename}"
                           
                            html = f"""
    <div style='text-align:center;'>
        
        <img src="{web_path}" alt="{select}" 
            style="width:100%; max-width:400px; border-radius:10px;">
        
        <div style="margin-top:8px; font-style:italic;">
            Image reference:<br>
            <a href='https://esahubble.org/' target='_blank'>ESA Hubble</a>
        </div>

    </div>
    """

                        else:
                            html = "<p style='color:red;'>Image not found</p>"

                        ui.html(html)

                    
                        
                    @ui.refreshable
                    def update_table():
                        select = gal_state.get('selected_file') or default_galaxy
                        table_filename = os.path.splitext(select)[0] + ".csv"
                        table_path = os.path.join(GALAXY_TABLES_PATH, table_filename)

                        ui.html("<h5>Galaxy Information</h5>")

                        if os.path.exists(table_path):
                            df = pd.read_csv(table_path, header=None, names=["Property", "Value"])
                          
                            ui.table.from_pandas(df).classes("w-full max-w-xl")

                            ui.html("""
    <div style="margin-top:8px; font-style:italic; text-align:center;">
        References:<br>
        <a href='https://theskylive.com/sky' target='_blank'>The Sky Live</a><br>
        <a href='https://en.wikipedia.org/wiki/List_of_NGC_objects' target='_blank'>
            Wikipedia: List of NGC objects
        </a>
    </div>
    """)


                        else:
                            ui.html("<p style='color:red;'>Table not found</p>")
                            
                  

                  
                    with ui.dialog() as galaxy_3d_info_dialog, ui.card().classes('p-6 w-full max-w-[800px] bg-white'):
                        ui.label("Interactive 3D Tools Info").classes("text-2xl font-bold text-slate-800 mb-4")
                        html_info_box(r"""
                        <p><b>Why is the Solar System different from a Galaxy?</b></p>
                        <ul>
                            <li><b>Solar System (Point Mass):</b> 99.8% of the mass is concentrated in the Sun. As the orbital radius <span class="math">\(r\)</span> increases, the enclosed mass remains constant. Gravity weakens, causing the orbital velocity to drop: <span class="math">\( v \propto 1/\sqrt{r} \)</span>.</li>
                            <li><b>Galaxy (Extended Spherical Halo):</b> Matter (especially Dark Matter) forms a massive, diffuse spherical halo. The larger the orbit, <b>the more matter is enclosed within it</b>. This additional mass compensates for the increasing distance, keeping the outer stars' velocity constant!</li>
                        </ul>
                        <hr style="margin: 10px 0;">
                        <p><b>Interactive 3D Tools:</b></p>
                        <ul>
                            <li><i>Add dark matter moving the slider.</i></li>
                            <li><i>Rotate the 3D plot with your mouse.</i></li>
                            <li><b><i>Click on the items in the Legend to turn the Baryonic Matter or Dark Matter on and off!</i></b></li>
                        </ul>
                        """)
                        aria_button("Close", "close", on_click=galaxy_3d_info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-2")


                   
                    with ui.dialog() as galaxy_3d_dialog, ui.card().style('width: 1200px; max-width: 95vw; height: 90vh;').classes('p-6 overflow-hidden flex flex-col bg-white').props('aria-label="3D Galaxy Model" role=dialog'):
                        
                        with ui.row().classes('w-full justify-between items-center mb-1 shrink-0'):
                            ui.label("Galaxy structure and mass distribution").classes("text-2xl font-bold text-slate-800")
                            with ui.row().classes('gap-4'):
                                aria_button("Info", 'instructions', on_click=lambda: [galaxy_3d_info_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-6 rounded shadow-md")
                                aria_button("Close", 'close', on_click=lambda: galaxy_3d_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-6 rounded shadow-md")

                      
                        dialog_status_label = ui.label("").classes("text-xl font-bold text-center w-full my-1")
                        
                        with ui.row().classes('w-full items-center justify-center gap-4 mt-1 mb-4 shrink-0 bg-slate-50 p-4 rounded-xl shadow-inner border border-slate-200'):
                            ui.label("Add Dark Matter:").classes("font-bold text-lg text-slate-800")
                          
                            dialog_dm_slider = ui.slider(min=0.0, max=2.0, value=0.0, step=0.01).classes('w-1/2').props('color=purple label-always')

                       
                        def evaluate_dm_status_dialog(e=None):
                            if not gal_state.get('DATA_LOADED'): return
                            r_ngc = gal_state['r_ngc']
                            y_disk = gal_state.get('upsilon_disk', 0.5)
                            y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            v_gas_ngc = gal_state['v_gas_ngc']
                            v_disk_ngc = gal_state['v_disk_ngc']
                            v_bul_ngc = gal_state['v_bul_ngc']
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                            m_baryonic = (np.maximum(V_bar_sq, 0) * r_ngc) / G_grav
                            val_m_bar = np.nanmax(m_baryonic) if len(m_baryonic) > 0 else 0
                            
                            rho_s = gal_state.get('base_rho_s', 0.0)
                            r_s = gal_state.get('base_r_s', 1.0)
                            M_dm_grid_best = M_burkert_enclosed(r_ngc, rho_s, r_s)
                            val_m_dm_best = np.nanmax(M_dm_grid_best) if len(M_dm_grid_best) > 0 else 0
                            
                            target_fraction = val_m_dm_best / (val_m_bar + val_m_dm_best) if (val_m_bar + val_m_dm_best) > 0 else 0
                            
                            f_slider = float(dialog_dm_slider.value)
                            val_m_dm_current = val_m_dm_best * f_slider
                            current_fraction = val_m_dm_current / (val_m_bar + val_m_dm_current) if (val_m_bar + val_m_dm_current) > 0 else 0
                            
                            gal_name = gal_state.get('current_galaxy_name', 'Galaxy').removesuffix('.txt').removesuffix('.csv').removesuffix('.dat')
                            
                            if current_fraction < 0.05:
                                msg = "Dark matter missing! Move the slider to add mass."
                                color_class = "text-red-600"
                            elif current_fraction < target_fraction - 0.04:
                                msg = f"More Dark Matter needed!"
                                color_class = "text-orange-500"
                            elif current_fraction > target_fraction + 0.04:
                                msg = f"Too much Dark Matter!"
                                color_class = "text-orange-500"
                            else:
                                msg = f"Perfect! The Dark Matter halo matches {gal_name}'s real rotation curve."
                                color_class = "text-green-600"
                                
                            dialog_status_label.set_text(msg)
                            dialog_status_label.classes(replace=f"text-xl font-bold text-center w-full my-1 {color_class}")

                        plot_3d_container = ui.column().classes('w-full flex-1 items-center justify-center overflow-hidden')

                        @ui.refreshable
                        def render_3d_mass_plot():
                            plot_3d_container.clear()
                            with plot_3d_container:
                                title_label = ui.label("").classes("text-xl font-bold text-slate-800 mb-2 mt-8 text-center w-full z-10")
                                fig = go.Figure()
                                
                                custom_colorbar = dict(
                                    title=dict(text="Velocity<br>(km/s)", font=dict(color='black', size=12), side='top'),
                                    tickfont=dict(color='black', size=11), x=-0.12, thickness=15
                                )
                                r_max_plot = 10.0 

                                if gal_state.get('DATA_LOADED') and len(gal_state.get('r_ngc', [])) > 0:
                                    r_ngc = gal_state['r_ngc']
                                    current_gal = gal_state.get('current_galaxy_name', '').removesuffix('.txt')
                                    r_max_plot = np.nanmax(r_ngc) * 1.05
                                    
                                    y_disk = gal_state.get('upsilon_disk', 0.5)
                                    y_bulge = gal_state.get('upsilon_bulge', 0.7)
                                    
                                    m_bulge = (r_ngc * y_bulge * (gal_state['v_bul_ngc']**2)) / G_grav
                                    M_bulge_tot = np.nanmax(m_bulge) if np.nanmax(m_bulge) > 0 else 0
                                    
                                    m_disk = (r_ngc * y_disk * (gal_state['v_disk_ngc']**2)) / G_grav
                                    M_disk_tot = np.nanmax(m_disk) if np.nanmax(m_disk) > 0 else 0
                                    
                                    v_gas_sq = gal_state['v_gas_ngc'] * np.abs(gal_state['v_gas_ngc'])
                                    m_gas = (r_ngc * np.maximum(v_gas_sq, 0)) / G_grav
                                    M_gas_tot = np.nanmax(m_gas) if np.nanmax(m_gas) > 0 else 0
                                    
                                    f_slider = float(dialog_dm_slider.value)
                                    rho_s, r_s, M_dm_grid = get_dm_params(f_slider, r_array=r_ngc)
                                    M_dm_tot = np.nanmax(M_dm_grid) if (len(M_dm_grid) > 0 and np.nanmax(M_dm_grid) > 0) else 0
                                    
                                    m_baryonic_tot = m_bulge + m_disk + m_gas
                                    v_total_curve = np.sqrt(G_grav * (m_baryonic_tot + M_dm_grid) / r_ngc)
                                    
                                    M_bar_tot = M_bulge_tot + M_disk_tot + M_gas_tot
                                    M_tot_all = M_bar_tot + M_dm_tot

                               
                                    dm_ratio_gal = (M_dm_tot / M_tot_all) if M_tot_all > 0 else 0.0
                                    dialog_dm_slider.props(f'label-value="DM: {dm_ratio_gal * 100:.1f}%"')

                                    title_label.set_text(f"3D Mass Distribution: {current_gal}")

                                    v_min_gal = np.nanmin(v_total_curve)
                                    v_max_gal = np.nanmax(v_total_curve)
                                    c_max_bright = v_max_gal * 1.15 

                                    N_TOTAL_POINTS = 3500
                                    n_bulge = int(N_TOTAL_POINTS * (M_bulge_tot / M_tot_all)) if M_tot_all > 0 else 0
                                    n_disk  = int(N_TOTAL_POINTS * (M_disk_tot / M_tot_all)) if M_tot_all > 0 else 0
                                    n_gas   = int(N_TOTAL_POINTS * (M_gas_tot / M_tot_all)) if M_tot_all > 0 else 0
                                    n_dm    = int(N_TOTAL_POINTS * (M_dm_tot / M_tot_all)) if M_tot_all > 0 else 0

                                    def sample_component(m_array, n_pts, is_spherical=False, z_thickness=0.1):
                                        if n_pts <= 0 or np.nanmax(m_array) <= 0: 
                                            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
                                            
                                        CDF = m_array / np.nanmax(m_array)
                                        CDF_clean = np.maximum.accumulate(np.nan_to_num(CDF))
                                        U = np.random.uniform(0, 1, n_pts)
                                        r_pts = np.interp(U, CDF_clean, r_ngc)
                                        
                                        if is_spherical:
                                            phi = np.arccos(np.random.uniform(-1, 1, n_pts))
                                            theta = np.random.uniform(0, 2*np.pi, n_pts)
                                            x = r_pts * np.sin(phi) * np.cos(theta)
                                            y = r_pts * np.sin(phi) * np.sin(theta)
                                            z = r_pts * np.cos(phi)
                                        else:
                                            theta = np.random.uniform(0, 2*np.pi, n_pts)
                                            x = r_pts * np.cos(theta)
                                            y = r_pts * np.sin(theta)
                                            z = np.random.normal(0, z_thickness, n_pts)
                                            
                                        v_pts = np.interp(r_pts, r_ngc, v_total_curve)
                                        m_enclosed = np.interp(r_pts, r_ngc, m_array)
                                        return x, y, z, v_pts, np.stack((r_pts, m_enclosed), axis=-1)

                                    traces = []
                                    if n_bulge > 0:
                                        x, y, z, v, c_data = sample_component(m_bulge, n_bulge, is_spherical=True)
                                        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, symbol='square', color=v, colorscale='turbo', cmin=v_min_gal, cmax=c_max_bright, opacity=0.9), name=f'■ Bulge (M: {M_bulge_tot:.1e} M_☉)', customdata=c_data, hovertemplate="<b>Bulge</b><br>Radius: %{customdata[0]:.1f} kpc<br>Velocity: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"))
                                    if n_disk > 0:
                                        x, y, z, v, c_data = sample_component(m_disk, n_disk, is_spherical=False, z_thickness=0.2)
                                        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, symbol='diamond', color=v, colorscale='turbo', cmin=v_min_gal, cmax=c_max_bright, opacity=0.8), name=f'★ Stellar Disk (M: {M_disk_tot:.1e} M_☉)', customdata=c_data, hovertemplate="<b>Stellar Disk</b><br>Radius: %{customdata[0]:.1f} kpc<br>Velocity: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"))
                                    if n_gas > 0:
                                        x, y, z, v, c_data = sample_component(m_gas, n_gas, is_spherical=False, z_thickness=0.1)
                                        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, symbol='cross', color=v, colorscale='turbo', cmin=v_min_gal, cmax=c_max_bright, opacity=0.9), name=f'✖ Gas Disk (M: {M_gas_tot:.1e} M_☉)', customdata=c_data, hovertemplate="<b>Gas</b><br>Radius: %{customdata[0]:.1f} kpc<br>Velocity: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"))
                                    if n_dm > 0:
                                        x, y, z, v, c_data = sample_component(M_dm_grid, n_dm, is_spherical=True)
                                        traces.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, symbol='circle', color=v, colorscale='turbo', cmin=v_min_gal, cmax=c_max_bright, opacity=0.3), name=f'● DM Halo (M: {M_dm_tot:.1e} M_☉)', customdata=c_data, hovertemplate="<b>Dark Matter</b><br>Radius: %{customdata[0]:.1f} kpc<br>Velocity: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"))

                                    for trace in traces:
                                        fig.add_trace(trace)
                                        
                                    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(colorscale='turbo', cmin=v_min_gal, cmax=c_max_bright, showscale=True, colorbar=custom_colorbar), showlegend=False, hoverinfo='skip'))
                                else:
                                    title_label.set_text("Waiting for valid Galaxy Data...")
                                    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='rgba(0,0,0,0)')))

                                fig.update_layout(
                                    scene=dict(xaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), yaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), zaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), bgcolor='rgb(15, 23, 42)', aspectmode='cube'),
                                    margin=dict(l=0, r=0, b=0, t=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    legend=dict(font=dict(color='white'), yanchor="top", y=0.95, xanchor="left", x=0.01),
                                    autosize=False, width=1050, height=520
                                )
                                
                                global my_galaxy_3d_plot
                                my_galaxy_3d_plot = ui.plotly(fig).style('width: 1050px; height: 520px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);')

                        render_3d_mass_plot()
                        
                        # Aggiornamento in tempo reale mentre scorri lo slider!
                        dialog_dm_slider.on('change', lambda e: [evaluate_dm_status_dialog(), render_3d_mass_plot.refresh()])

                

                    with ui.dialog() as table_dialog2, ui.card().style('width: 1200px; max-width: 95vw;').classes('p-8 border border-gray-200 shadow-md rounded-xl bg-white').props('aria-label="Galaxy Information and Image" role=dialog'):
                        
                      
                        with ui.element('div').style('display: flex; flex-direction: row; width: 100%; gap: 2rem; align-items: flex-start; justify-content: space-between;'):
                            
                          
                            with ui.element('div').style('flex: 0 0 450px; display: flex; flex-direction: column; align-items: center;'):
                                ui.html("<h4 style='margin-top: 0; font-weight: bold; color: #1e3a8a;'>Galaxy Image</h4>")
                                update_image()
                            
                        
                            with ui.element('div').style('flex: 1 1 auto; width: 100%;'):
                                update_table()
                        
                        with ui.row().classes('w-full justify-end mt-6'):
                            aria_button("Close", 'close', on_click=lambda: table_dialog2.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-6 rounded shadow-md")
                   
                   
                 
                    with ui.dialog() as comp_physics_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="physics-title"'):
                        with ui.column().classes('w-full h-full bg-white'):
                            
                        
                            with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="physics-title"'):
                                ui.label('Physics, Math & Morphology').classes('text-xl font-bold').props('id="physics-title" role=heading aria-level=2 tabindex=0')
                                aria_button('Close', 'close', on_click=comp_physics_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                          
                            with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as physics_tabs:
                                t_comp = ui.tab('Computational Notes')
                                t_chi2 = ui.tab('χ² Minimization')
                                t_morpho = ui.tab('Morphology Top-View')
                                physics_tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))

                            with ui.tab_panels(physics_tabs, value=t_comp).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                    
                                with ui.tab_panel(t_comp):
                                    @ui.refreshable
                                    def comp_notes_content():
                                        with ui.row().classes('w-full items-start justify-between gap-6 flex-wrap lg:flex-nowrap'):
                                            
                                            # --- Colonna SINISTRA: Testo "Computational Notes" ---
                                            with ui.column().classes('flex-[3] min-w-[500px]'):
                                                html_info_box(r"""
                                                                <style>
                        .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
                        .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
                        .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
                        .math-den { padding-top: 2px; }
                    </style>
                    <h3>Computational Notes</h3>
                                                    
                                                    <ul>
                                                        <li>
                                                            <b>Step 1:</b> Baryonic velocity with Mass-to-Light ratios (\( \Upsilon_{\mathrm{disk}}, \Upsilon_{\mathrm{bulge}} \))<br>
<span class="math">\( v_{\mathrm{bar}}^2(r) = v_{\mathrm{gas}}(r)|v_{\mathrm{gas}}(r)| + \Upsilon_{\mathrm{disk}}\, v_{\mathrm{disk}}(r)|v_{\mathrm{disk}}(r)| + \Upsilon_{\mathrm{bulge}}\, v_{\mathrm{bulge}}(r)|v_{\mathrm{bulge}}(r)| \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 2:</b> Baryonic mass<br>
                                                            <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, \max(v_{\mathrm{bar}}^2(r), 0)}{G} \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 3:</b> Total observed mass from velocity data<br>
                                                            <span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2(r)}{G} \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 4:</b> Observed dark matter density (from data)<br>
                                                            <span class="math">\( \rho_{\mathrm{DM}}(r) = \frac{1}{4 \pi G r^2} \, \frac{d}{dr} \Bigg[ r^2 \Big( \frac{v_{\mathrm{obs}}^2(r) - v_{\mathrm{bar}}^2(r)}{r} \Big) \Bigg] \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                <b>Step 5:</b> Burkert dark matter profile<br>
                                <span class="math">\( \rho_{\mathrm{Burkert}}(r) = \frac{\rho_s}{\left(1 + \tfrac{r}{r_s}\right)\left(1 + \left(\tfrac{r}{r_s}\right)^2\right)} \)</span> 
                            </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 6:</b> Global Fit parameters<br>
The structural parameters \( \rho_s, r_s, \Upsilon_{\mathrm{disk}} \) (and \( \Upsilon_{\mathrm{bulge}} \) when present) are pre-calculated via global Differential Evolution \( \chi^2 \) minimization on the SPARC rotation curve.
                                                        </li>
                                                        <li style="margin-top: 8px;">
                                <b>Step 7:</b> Enclosed dark matter mass (Burkert)<br>
                                <span class="math">\( M_{\mathrm{DM}}(r) = 2 \pi \rho_s r_s^3 \left[0.5 \ln(1+x^2) + \ln(1+x) - \arctan(x)\right], \;\; x = \tfrac{r}{r_s} \)</span>
                            </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 8:</b> Dark matter rotational velocity<br>
                                                            <span class="math">\( v_{\mathrm{DM}}(r) = \sqrt{\frac{G M_{\mathrm{DM}}(r)}{r}} \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 9:</b> Total simulated velocity curve (linked to DM slider)<br>
                                                            <span class="math">\( v_{\mathrm{tot,sim}}(r) = \sqrt{\tfrac{G \, (M_{\mathrm{bar}}(r) + f\,M_{\mathrm{DM}}(r))}{r}}, \;\; f \in [0,2] \)</span>
                                                        </li>
                                                        
                                                        <li style="margin-top: 8px;">
                                                            <b>Step 10:</b> Fit quality (reduced \( \chi^2/\mathrm{dof} \))<br>
<span class="math">\( \chi^2_{\mathrm{red}} = \frac{1}{N - 3} \sum_{i=1}^N \left(\tfrac{v_{\mathrm{obs}}(r_i) - v_{\mathrm{tot,sim}}(r_i)}{\sigma_i}\right)^2 \)</span>
                                                        </li>
                                                    </ul>
                    <h4>Plot Legend</h4>
                    <ul style="margin-top: 5px;">
                        <li><b>X-axis:</b> Radius (data)</li>
                        <li><b>Y-axis:</b> <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue with grey error bars), <span class="math">\( v_{\mathrm{tot,sim}} \)</span> (green)</li>
                    </ul>
                """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the computational steps"')

                                            # --- Colonna DESTRA: Plot densità Burkert ---
                                            with ui.column().classes('flex-[2] min-w-[400px] items-center justify-center lg:mt-12'):
                                                if gal_state.get('DATA_LOADED') and len(gal_state.get('r_ngc', [])) > 0:
                                                    r_arr = gal_state['r_ngc']
                                                    rho_s = gal_state.get('base_rho_s', 0.0)
                                                    r_s = gal_state.get('base_r_s', 1.0)
                                                    
                                                    gal_name = gal_state.get('current_galaxy_name', 'Unknown').removesuffix('.txt').removesuffix('.csv').removesuffix('.dat')
                                                    
                                                    # Calcolo del profilo di Burkert in scala lineare per il raggio
                                                    rho_burkert = rho_s / ((1.0 + r_arr / r_s) * (1.0 + (r_arr / r_s)**2))
                                                    
                                                    with ui.pyplot(figsize=(5.5, 4.5), close=True, clear=True):
                                                        plt.plot(r_arr, rho_burkert, '-', color='darkorange', linewidth=2.5, label='Burkert Density')
                                                        plt.xlabel('Radius (kpc)', fontsize=10)
                                                        plt.ylabel(r'DM Density ($M_\odot / kpc^3$)', fontsize=10)
                                                        plt.yscale('log')
                                                        plt.title(f'DM Density Profile: {gal_name}', fontsize=12, fontweight='bold')
                                                        plt.grid(True, linestyle='--', alpha=0.6)
                                                        plt.legend(fontsize=10)
                                                        plt.tight_layout()
                                                else:
                                                    ui.label("Load a galaxy to see its Dark Matter density profile.").classes('text-gray-500 italic text-center')

                                    # Inizializza il componente refreshable
                                    comp_notes_content()
                               
                                with ui.tab_panel(t_chi2):
                                    html_info_box(r"""
        <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>

        <h3 class="text-xl font-bold mb-2">Parabolic interpolation and compute &chi;&sup2; minimization</h3>
        <div class="concept-box">
            <h4 class="font-bold mb-1">What is &chi;&sup2;?</h4>
            <p class="text-sm">
                The <b>&chi;&sup2; (Chi-squared)</b> statistic measures the agreement between observed data and a theoretical model. 
                Geometrically, it represents the <b>sum of the squared vertical distances</b> between each observed data point and the model curve, weighted by the experimental error (&sigma;).
            </p>
            <div class="mt-2 text-center">
                <span class="math">\( \chi^2 = \sum \frac{(O_i - T_i)^2}{\sigma_i^2} \)</span>
            </div>
            <p class="text-xs mt-1 italic">Where \( O_i \) is the observed value (blue curve), \( T_i \) is the simulated model value (green curve),at point \( i \) and \( \sigma_i \) is the experimental error from data. </p>
        </div>
        <p class="mb-2">Use the instructions below to compute the &chi;&sup2; minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)).</p>
        
        <ul class="list-disc pl-5 space-y-1">
            <li><b>1a)</b> Choose 3 values of dark matter fraction (f) with the slider above to compute the &chi;&sup2; at that point.</li>
            <li><b>1b)</b> Press <b>'Add &chi;&sup2; point'</b> button to add the point to the &chi;&sup2; plot.</li>
            <li><b>2a)</b> Input 3 values for the dark matter fraction (M_DM / M_tot, between 0 and 1).</li>
            <li><b>2b)</b> Press <b>'compute &chi;&sup2; and minimum'</b> button to calculate the &chi;&sup2; at those points and check the minimum on the plot.</li>
        </ul>

        <div class="formula-box">
            <b>Formula:</b><br><br>
            <div style="font-family: 'Times New Roman', serif; font-size: 1.1em; display: flex; align-items: center; justify-content: center;">
                x<sub>min</sub> = 
                <div class="math-frac">
                    <span class="math-num">1</span>
                    <span class="math-den">2</span>
                </div>
                &middot;
                <div class="math-frac">
                    <span class="math-num">
                        (x<sub>1</sub><sup>2</sup>&minus;x<sub>2</sub><sup>2</sup>)&chi;<sup>2</sup>(x<sub>3</sub>) + 
                        (x<sub>2</sub><sup>2</sup>&minus;x<sub>3</sub><sup>2</sup>)&chi;<sup>2</sup>(x<sub>1</sub>) + 
                        (x<sub>3</sub><sup>2</sup>&minus;x<sub>1</sub><sup>2</sup>)&chi;<sup>2</sup>(x<sub>2</sub>)
                    </span>
                    <span class="math-den">
                        (x<sub>1</sub>&minus;x<sub>2</sub>)&chi;<sup>2</sup>(x<sub>3</sub>) + 
                        (x<sub>2</sub>&minus;x<sub>3</sub>)&chi;<sup>2</sup>(x<sub>1</sub>) + 
                        (x<sub>3</sub>&minus;x<sub>1</sub>)&chi;<sup>2</sup>(x<sub>2</sub>)
                    </span>
                </div>
            </div>
        </div>
    """).props('aria-label="Parabolic interpolation formula for chi-squared minimization"')

                               
                                with ui.tab_panel(t_morpho).classes('flex flex-col items-center justify-start w-full'):
                                    
                                   
                                   
                                    morph_plot_container = ui.column().classes("w-full items-center justify-center")
                                    
                              
                                    #model_selector.on_value_change(render_3d_mass_plot.refresh) 
                    combined_exercise_state = {'step': 0}

                    combined_exercises_html = [
                        r"""
                        <h3>Phase 1: Galaxy Panel – Predicting in a New Context (Galaxies)</h3>
                        <p><b>Goal:</b> Use the Solar System Model you just created to make a prediction in the context of galaxies.</p>
                        
                        <h4>Formulate your prediction:</h4>
                        <ul>
                            <li>Before looking at the App's data, gather in your group and draw your prediction on a piece of paper: what do you expect the velocity graph of stars rotating around the center of a galaxy to look like as you move further from the center?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Observation of Real Data:</h3>
                        <ul>
                            <li>In the App, module 2, "Galaxy Panel" section, select a real galaxy (dataset) from the dropdown menu.</li>
                            <li>Observe the left graph representing the rotation velocity of stars as a function of distance from the galactic center: the blue points represent the "Observed Data" (real measurements with optical instruments), while the red line represents the theoretical Keplerian curve (what you would expect based on the formula found for the Solar System).</li>
                            <li>Observing the App's graph, why are the two curves (red line and blue points) different? Which curve do you think is correct for galaxies and why?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Phase 2: Galaxy Panel – Formulating a New Hypothesis</h3>
                        <p><b>Goal:</b> Find a mathematical solution to the discrepancy that emerged in Phase 2.</p>
                        
                        <h4>Analysis of the Discrepancy:</h4>
                        <ul>
                            <li>Take back the Keplerian orbital velocity formula that you derived in Phase 1.</li>
                            <li>Look at the observational data in the graph (blue points): the velocity does not drop, but remains almost constant even at very large distances. The Keplerian prediction (red curve) does not match the experimental data.</li>
                            <li>Why do the peripheral stars of the galaxy travel so fast?</li>
                            <li>How can the discrepancy between observations and prediction be resolved?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Building Model :</h3>
                        <ul>
                            <li>Look at the equation: if the radius in the denominator grows (as we see in the graph), what must happen to the Mass in the numerator so that the velocity remains constant and the equation remains mathematically correct?</li>
                            <li>If we had to add mass, is it luminous mass (visible stars, dust, gas) or is it something else?</li>
                        </ul>
                        """
                    ,


                    
                        
                        r"""
                        <h3>Phase 3: Galaxy Panel – Predicting in a New Context (Galaxies)</h3>
                        <p><b>Goal:</b> Use the Solar System Model you just created to make a prediction in the context of galaxies.</p>
                        
                        <h4>Formulate your prediction:</h4>
                        <ul>
                            <li>Before looking at the App's data, gather in your group and draw your prediction on a piece of paper: what do you expect the velocity graph of stars rotating around the center of a galaxy to look like as you move further from the center?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Observation of Real Data:</h3>
                        <ul>
                            <li>In the App, module 2, "Galaxy Panel" section, select a real galaxy (dataset) from the dropdown menu.</li>
                            <li>Observe the left graph representing the rotation velocity of stars as a function of distance from the galactic center: the blue points represent the "Observed Data" (real measurements with optical instruments), while the red line represents the theoretical Keplerian curve (what you would expect based on the formula found for the Solar System).</li>
                            <li>Observing the App's graph, why are the two curves (red line and blue points) different? Which curve do you think is correct for galaxies and why?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Phase 4: Galaxy Panel – Formulating a New Hypothesis</h3>
                        <p><b>Goal:</b> Find a mathematical solution to the discrepancy that emerged in Phase 2.</p>
                        
                        <h4>Analysis of the Discrepancy:</h4>
                        <ul>
                            <li>Take back the Keplerian orbital velocity formula that you derived in Phase 1.</li>
                            <li>Look at the observational data in the graph (blue points): the velocity does not drop, but remains almost constant even at very large distances. The Keplerian prediction (red curve) does not match the experimental data.</li>
                            <li>Why do the peripheral stars of the galaxy travel so fast?</li>
                            <li>How can the discrepancy between observations and prediction be resolved?</li>
                        </ul>
                        """,
                        r"""
                        <h3>Building Model :</h3>
                        <ul>
                            <li>Look at the equation: if the radius in the denominator grows (as we see in the graph), what must happen to the Mass in the numerator so that the velocity remains constant and the equation remains mathematically correct?</li>
                            <li>If we had to add mass, is it luminous mass (visible stars, dust, gas) or is it something else?</li>
                        </ul>
                        """,
                        
                    
                        r"""
                        <h3>Phase 5: Galaxy Panel – Quantifying Model </h3>
                        <p><b>Goal:</b> Verify if Model 2 fits the real data using multisensorial and statistical analysis.</p>
                        
                        <h4>Roles Distribution:</h4>
                        <p>Divide the following roles within the group (you can rotate during the activity):</p>
                        <ul>
                            <li><b>Navigator:</b> Moves the slider in the App to add dark matter; the green simulated curve moves and must reach the observations.</li>
                            <li><b>Listener:</b> Wears headphones to listen to the "Sonification" (data converted into sound) and find the sound match between the data and the simulated curve.</li>
                            <li><b>Data Analyst:</b> Monitors the mathematical values and the "Chi-Squared" generated by the App to quantify the additional mass and reduce the discrepancy between data and theory.</li>
                        </ul>

                        <hr style="margin: 15px 0;">

                        <h4>Exercise : Slider Exercise</h4>
                        <ul>
                            <li>The Navigator selects a galaxy from the dropdown menu (module 2 – Galaxy panel) and starts adding mass by moving the slider. Observe the green curve (simulated) moving. Try different mass values until you get a good match between the simulated curve and the data.</li>
                            <li>When you have reached the match between the simulated curve and the data, what dark matter value do you read on the graph? The Data Analyst checks the dark matter value and also takes care of the following statistical exercise.</li>
                        </ul>
                        """,
                        r"""
                        <h3>Exercise : Multisensory Exercise</h3>
                        <ul>
                            <li>The data is converted into sound frequencies (high pitch = high velocity values, low pitch = low values) to perceive the velocity values in a multisensory way.</li>
                            <li>The Listener clicks the 'activate audio' button and listens to the sound of the data by selecting (one at a time) 'observed velocity', 'baryonic velocity', and 'simulated velocity'. Listen to both the sound of the entire curve and the average.</li>
                            <li>Try different simulated velocity values by moving the slider, and each time listen to the sound and compare it with the observed data. Look for the audio match between the observed and simulated curves.</li>
                            <li>When the simulation sound matches the real data sound perfectly, say "Stop!" so the Navigator can select the correct mass value with the slider.</li>
                        </ul>
                        """,
                        r"""
                        <h3>Exercise : Statistical Exercise (Chi-Squared Minimization)</h3>
                        <ul>
                            <li>Let's call <b>O</b> the Observed value (blue point) and <b>E</b> the Expected value from the model. The distance/error is (O - E). If we sum all these distances (some positive above the curve, others negative below), what happens mathematically?</li>
                            <li>How do we solve the problem of negative numbers in statistics if we only want to sum positive error quantities?</li>
                            <li>Does a 10 km/s error on a very slow star weigh the same as a 10 km/s error on a very fast one? How is the formula modified to account for this?</li>
                            <li>The Chi-Squared calculates the difference between the blue points (observations) and the green line (simulated model based on a mass density distribution).</li>
                            <li>By moving the slider we can add points, how do we know we have found the absolute minimum error value? What geometric shape has a minimum value?</li>
                            <li>The Data Analyst and the Navigator must collaborate to try 3 completely different Dark Matter values on the slider and check their respective 3 Chi-Squared values.</li>
                            <li>Insert 3 points into the center graph by moving the slider and clicking 'add point' after selecting each value. What geometric shape did you get? What (minimum) value does the vertex correspond to? Find the smallest vertex value by trying different combinations of 3 values.</li>
                            <li>Why do we need to calculate exactly a parabola and need exactly 3 points to find the minimum error?</li>
                        </ul>
                        """
                    ]
                                

                    with ui.dialog() as instr_combined_galaxy, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Galaxy exercises" role=dialog'):
                        
                        @ui.refreshable
                        def combined_exercises_content():
                            current_step = combined_exercise_state['step']
                            
                            html_info_box(combined_exercises_html[current_step])
                            
                            with ui.row().classes('w-full justify-between items-center mt-4'):
                                
                                if current_step > 0:
                                 
                                    aria_button("Previous", "Go to previous exercise", on_click=lambda: change_step_galaxy_combined(-1)) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    ui.label()
                                
                                ui.label(f"Step {current_step + 1} of {len(combined_exercises_html)}").classes('text-gray-500 font-bold')
                                
                                if current_step < len(combined_exercises_html) - 1:
                                   
                                    aria_button("Next", "Go to next exercise", on_click=lambda: change_step_galaxy_combined(1)) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    aria_button("Close", "close the box", on_click=lambda: instr_combined_galaxy.close()) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                       
                        def change_step_galaxy_combined(delta):
                            new_step = combined_exercise_state['step'] + delta
                            if 0 <= new_step < len(combined_exercises_html):
                                combined_exercise_state['step'] = new_step
                                combined_exercises_content.refresh()

                        instr_combined_galaxy.on('open', lambda: combined_exercise_state.update({'step': 0}) or combined_exercises_content.refresh())

                        combined_exercises_content()

                   
                    
                    with ui.row().classes('w-full items-center justify-center gap-4'):
                        galaxy_files = [f for f in os.listdir(GALAXY_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]

                        if not galaxy_files:
                            warning_box("Nessun file di galassia trovato in GALAXY_DATA_PATH").classes("text-red-500")
                            galaxy_files = []

                        default_galaxy = "NGC3198.txt" if "NGC3198.txt" in galaxy_files else (galaxy_files[0] if galaxy_files else None)
                        

                        galaxy_select = ui.select(
                            galaxy_files,
                            value=default_galaxy,
                            label='Select a Galaxy Dataset'
                        ).classes('w-1/2 max-w-md').props('id=galaxy_selector aria-label="Galaxy dataset selector" role=listbox tabindex=0')
                        
                        aria_button("Instructions","Instruction for galaxy panel",on_click=safe_click(lambda: [info_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        
                        aria_button(
                            "Scientific Info", 
                            "Read detailed information about computational steps and morphology",
                            on_click=lambda: [comp_notes_content.refresh(), comp_physics_dialog.open(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")
            ]
                        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                        
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_rot.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                        
                    
                        
                        


                    ui.label("Move the slider to add the dark matter to the simulated velocity curve").props('id=alpha_slider_label aria-live=polite tabindex=0')
                    alpha_slider = aria_slider(min=0.0, max=2.0, value=0.0, step=0.01,
                                            aria_label="Dark matter fraction control slider").props('aria-describedby=alpha_slider_label label-always')
                    alpha_slider.bind_value(dialog_dm_slider, 'value')
                    f = float(alpha_slider.value)
                    
                    
                    def load_galaxy_data(filename):
                        try:
                            df = get_galaxy_data_cached(filename) 

                            if df is None:
                                gal_state['DATA_LOADED']= False
                                return False

                            gal_state['r_ngc'] = pd.to_numeric(df['Rad'], errors='coerce').values
                            gal_state['v_obs_ngc'] = pd.to_numeric(df['Vobs'], errors='coerce').values
                            gal_state['v_err_ngc'] = pd.to_numeric(df['errV'], errors='coerce').values
                            gal_state['v_gas_ngc'] = pd.to_numeric(df['Vgas'], errors='coerce').values
                            gal_state['v_disk_ngc'] = pd.to_numeric(df['Vdisk'], errors='coerce').values
                            gal_state['v_bul_ngc'] = pd.to_numeric(df['Vbul'], errors='coerce').values

                            if gal_state['r_ngc'].size > 0:
                                gal_name = os.path.splitext(filename)[0]
                               
                                gal_name_clean = gal_name.lower().replace('_rotmod', '').strip().upper()
                        
                                
                                csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                                if not os.path.exists(csv_path):
                                    csv_path = os.path.join(dataset_path, 'galaxy_best_parameters_2.csv')
                                    
                                df_params = pd.read_csv(csv_path)
                                
                            
                                df_params['Clean_Galaxy'] = df_params['Galaxy'].astype(str).str.lower().str.replace('_rotmod', '').str.strip().str.upper()
                                
                                row = df_params[df_params['Clean_Galaxy'] == gal_name_clean]
                                
                                if not row.empty:
                                    
                                    if 'Upsilon_disk' in row.columns and 'Upsilon_bulge' in row.columns:
                                        y_disk = float(row.iloc[0]['Upsilon_disk'])
                                        y_bulge = float(row.iloc[0]['Upsilon_bulge'])
                                    else:
                                        y_disk = float(row.iloc[0].get('Upsilon', 0.5))
                                        y_bulge = 1.4 * y_disk
                                        
                                    r_s = float(row.iloc[0]['r_s'])
                                    rho_s = float(row.iloc[0]['rho_s'])
                                else:
                                    accessible_notify(f"Error: Parameters for {gal_name} not found in CSV!", type_='error')
                                    return False

                                gal_state['base_rho_s'] = rho_s
                                gal_state['base_r_s'] = r_s
                                gal_state['upsilon_disk'] = y_disk
                                gal_state['upsilon_bulge'] = y_bulge
                                
                                gal_state['base_M_dm_grid'] = M_burkert_enclosed(gal_state['r_ngc'], rho_s, r_s)
                                
                                gal_state['DATA_LOADED'] = True
                                return True
                            else:
                                gal_state['DATA_LOADED'] = False
                                return False
                           
                        except Exception as ex:
                            print("Errore:", ex)
                            gal_state['DATA_LOADED'] = False
                            return False
                        
                        
                    def reload_galaxy(new_value):
                        if not new_value: return
            
                        loaded = load_galaxy_data(new_value)
                        
                        if loaded:
                            gal_state['selected_file'] = new_value
                            galaxy_state['select'] = new_value 
                            gal_state['current_galaxy_name'] = new_value
                            
                            new_max_gal = 2.0
                          
                            alpha_slider.max = round(new_max_gal, 1)
                            dialog_dm_slider.max = round(new_max_gal, 1) 
                            alpha_slider.step = round(new_max_gal / 100.0, 2)
                            dialog_dm_slider.step = round(new_max_gal / 100.0, 2)
                            alpha_slider.value = 0.0 
                            alpha_slider.update()
                            dialog_dm_slider.update()
                            
                            gal_state['chi2_points'].clear()
                            gal_state['manual_points'].clear()
                            chi2_state['slider_result'] = "Add points to find the minimum."
                            update_all_plots.refresh()
                            update_image.refresh()
                            update_table.refresh()
                    
                
                    @ui.refreshable
                    def update_all_plots():
                        if not gal_state['DATA_LOADED']:
                            return
                        plt.close('all')
                        try:
                            update_galaxy_rotation_plot()
                            update_mass_plot()
                            plot_chi2_user_curve()
                            update_morphology_plot()
                            #render_3d_mass_plot.refresh()
                        except Exception as ex:
                            import traceback; traceback.print_exc()
                            accessible_notify(f"Error: {ex}", type_='error')

                    loaded = load_galaxy_data(default_galaxy)
                    
                    if not loaded:
                        print(f"Error:  file {default_galaxy} not loaded correctly.")
                    else:
                        pass
                        
                    galaxy_select.on_value_change(lambda e: reload_galaxy(e.value))
                    #alpha_slider.on('update:model-value', update_all_plots.refresh)
                    alpha_slider.on('change', lambda e: update_all_plots.refresh())
                    if default_galaxy:
                        reload_galaxy(default_galaxy)


                    def get_dm_params(fraction, r_array=None):
                        r_ngc = gal_state['r_ngc']
                        rho_s = gal_state.get('base_rho_s', 0.0)
                        r_s = gal_state.get('base_r_s', 1.0)
                        M_dm_grid_base = gal_state.get('base_M_dm_grid', np.zeros_like(r_ngc))
                        
                        M_dm_grid_scaled = M_dm_grid_base * fraction

                        if r_array is None:
                            r_array = r_ngc 
                        
                        if len(r_array) != len(M_dm_grid_scaled):
                            M_dm_grid_interp = np.interp(r_array, r_ngc, M_dm_grid_scaled)
                        else:
                            M_dm_grid_interp = M_dm_grid_scaled
                        
                        return rho_s, r_s, M_dm_grid_interp

                    
                    rho_s_init, r_s_init, M_dm_grid_init = get_dm_params(f)

                    if len(M_dm_grid_init) > 0:
                        dm_max_possible = np.max(M_dm_grid_init) * 2  
                    else:
                        dm_max_possible = 10.0  
                    
                    chi2_max_possible = 10  
                    slider_max_gal = getattr(alpha_slider, 'max', 10.0)
                    
                    x_min_chi = 0.0
                    x_max_chi = 1
                    y_min_chi = 0.0
                    y_max_chi = chi2_max_possible
                        
                    unscaled_M_dm_grid_cache = {}
                    
                    def plot_chi2_user_curve():
                      
                        chi2_points = gal_state['chi2_points']
                        manual_points = gal_state['manual_points']

                        chi2_plot_container.clear()
                        plt.close()
                        with chi2_plot_container:
                            with ui.pyplot(figsize=(6,4), close=False,clear=True):
                                ax = plt.gca()
                                all_xs = []
                                all_ys = []
                                
                               
                                if chi2_points:
                                    xs_slider, ys_slider = zip(*chi2_points)
                                    all_xs.extend(xs_slider)
                                    all_ys.extend(ys_slider)
                                    ax.scatter(xs_slider, ys_slider, s=40, c="blue", alpha=0.6, label="Points")

                                    if len(chi2_points) == 3 :
                                        coeffs_slider = np.polyfit(xs_slider, ys_slider, 2)
                                        x_lo, x_hi = min(xs_slider), max(xs_slider)
                                        
                                       
                                        x_fit_slider = np.linspace(x_lo - 0.2*(x_hi-x_lo), x_hi + 0.2*(x_hi-x_lo), 400)
                                        y_fit_slider = np.polyval(coeffs_slider, x_fit_slider)
                                        
                                        all_xs.extend(x_fit_slider)
                                        all_ys.extend(y_fit_slider)
                                        
                                        if coeffs_slider[0] > 0:
                                           
                                            ymin_index = np.argmin(y_fit_slider)
                                            xmin_slider = x_fit_slider[ymin_index]
                                            ymin_slider = np.polyval(coeffs_slider, xmin_slider)
                                            
                                            ax.plot(x_fit_slider, y_fit_slider, "g-", lw=2, zorder=2, label="Parabolic Fit")
                                            ax.scatter([xmin_slider], [ymin_slider], c='red', s=140, marker='*', zorder=4, label="Min")
                                            all_xs.append(xmin_slider)
                                            all_ys.append(ymin_slider)
                                        else:
                                            ax.plot(x_fit_slider, y_fit_slider, "r--", lw=2, label="Invalid Fit")
                                
                               
                                if manual_points and len(manual_points) >= 3:
                                    xm, ym = zip(*manual_points)
                                    all_xs.extend(xm)
                                    all_ys.extend(ym)
                                    ax.scatter(xm, ym, s=60, c="blue", marker="o", label="Points")

                                    coeffs_manual = np.polyfit(xm, ym, 2)
                                    x_lo_m, x_hi_m = min(xm), max(xm)
                                    x_fit_manual = np.linspace(x_lo_m - 0.2*(x_hi_m-x_lo_m), x_hi_m + 0.2*(x_hi_m-x_lo_m), 400)
                                    y_fit_manual = np.polyval(coeffs_manual, x_fit_manual)

                                    all_xs.extend(x_fit_manual)
                                    all_ys.extend(y_fit_manual)

                                    if coeffs_manual[0] > 0:
                                        ax.plot(x_fit_manual, y_fit_manual, "green", lw=2, zorder=2, label="Parabolic Fit")
                                        xmin_manual = -coeffs_manual[1] / (2 * coeffs_manual[0])
                                        ymin_manual = np.polyval(coeffs_manual, xmin_manual)
                                        
                                        ax.scatter([xmin_manual], [ymin_manual], c='red', s=120, marker='*', zorder=4, label=" Min")
                                        all_xs.append(xmin_manual)
                                        all_ys.append(ymin_manual)
                                    else:
                                        ax.plot(x_fit_manual, y_fit_manual, "r--", lw=2, label="Invalid Fit")

                            
                                if all_xs:
                                    plot_x_max = max(all_xs) * 1.1
                                    ax.set_xlim(0, plot_x_max if plot_x_max > 0 else x_max_chi)
                                   
                                    valid_ys = [y for y in all_ys if np.isfinite(y)]
                                    if valid_ys:
                                        plot_y_max = max(max(valid_ys) * 1.1, 1.0)
                                        plot_y_min = min(0.0, min(valid_ys) * 1.1) 
                                        ax.set_ylim(plot_y_min, plot_y_max)
                                else:
                                    
                                    ax.set_xlim(x_min_chi, x_max_chi)
                                    ax.set_ylim(y_min_chi, y_max_chi)
                                
                                ax.set_xlabel(r" $M_{DM}/M_{tot}$", fontsize=10)
                                ax.set_ylabel(r"χ²/dof", fontsize=10)
                                ax.grid(True)
                                
                                handles, labels = ax.get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                if by_label: ax.legend(by_label.values(), by_label.keys())
                                
                                text_to_show = chi2_state.get('slider_result', "---")
                                plt.text(0.5, 0.92, text_to_show, 
                                         transform=ax.transAxes, 
                                         fontsize=10, 
                                         color='green',
                                         fontweight='bold',
                                         horizontalalignment='center',
                                         verticalalignment='top',    
                                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray'))
                                
                                plt.title("χ² Minimization",fontsize=12,fontweight='bold')
                                plt.tight_layout()
                                ui.element('div').props('role=status aria-live=polite tabindex=0 aria-label=Plot showing chi-square minimization curve; X axis is dark matter mass in solar masses, Y axis is chi-square per degree of freedom')

            
                    
                    def refresh_formula_from_inputs():
                       
                        chi2_points = gal_state['chi2_points']
                        manual_points = gal_state['manual_points']
                        formula_pts["a"] = input_a.value
                        formula_pts["b"] = input_b.value
                        formula_pts["c"] = input_c.value
                        formula_pts["fa"] = fa_in.value
                        formula_pts["fb"] = fb_in.value
                        formula_pts["fc"] = fc_in.value
                        a,b,c = formula_pts["a"], formula_pts["b"], formula_pts["c"]
                        fa,fb,fc = formula_pts["fa"], formula_pts["fb"], formula_pts["fc"]

                        if None not in [a,b,c,fa,fb,fc]:
                            points = sorted([(a,fa),(b,fb),(c,fc)])
                            manual_points.clear()
                            manual_points.extend(points)
                            
                            
                    
                    
                        
               
                    chi2_input_dialog = ui.dialog().props('role="dialog" aria-modal="true"')

                    with chi2_input_dialog, ui.card().classes('w-full max-w-md'):
                        ui.label('Chi² Minimization Data').classes('text-xl font-bold mb-4').props('role=heading aria-level=2 tabindex=0')
                        
                    
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            input_a = ui.number(label='Fraction 1 (0 to 1)', format='%.2f').classes('w-full')
                            fa_in = ui.number(label="χ²(x₁)", format="%.2f").classes('w-full')
                            
                            input_b = ui.number(label='Fraction 2 (0 to 1)', format='%.2f').classes('w-full')
                            fb_in = ui.number(label="χ²(x₂)", format="%.2f").classes('w-full')
                            
                            input_c = ui.number(label='Fraction 3 (0 to 1)', format='%.2f').classes('w-full')
                            fc_in = ui.number(label="χ²(x₃)", format="%.2f").classes('w-full')

                     
                        for el in [input_a, input_b, input_c, fa_in, fb_in, fc_in]:
                            el.on('update:model-value', lambda: refresh_formula_from_inputs())

                        
                        result_label = ui.label("Min: ---").classes("text-green-600 font-bold text-lg my-2")

                     
                        def compute_and_update():
                            initialize_parabolic_points() 
                      
                            accessible_notify('Plot updated!', type_='success')

                     
                        with ui.row().classes('w-full justify-end gap-2 mt-4'):
                            aria_button('Close','Close', on_click=lambda:chi2_input_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            aria_button('Compute Minimum','Compute Minimum', on_click=lambda:compute_and_update()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    points_display = ui.column()
                    history_display = ui.column()

                    with ui.row().classes('w-full items-stretch justify-between no-wrap gap-2 px-2'):
                    
                      
                        with ui.column().classes('w-[260px] flex-shrink-0 p-2 bg-blue-50 rounded-lg border border-blue-200 shadow-sm flex flex-col justify-between overflow-hidden'):
                                
                               
                            with ui.column().classes('w-full gap-1 p-0 m-0'):
                                galaxy_title_label = ui.label("Galaxy Info: ---").classes("text-xl font-bold text-blue-800 m-0 leading-tight whitespace-nowrap")
                               
                                async def open_galaxy_3d():
                                    evaluate_dm_status_dialog()
                                    target_fraction = 0.5 # Default di fallback nel caso i dati non siano pronti
                                    gal_name = gal_state.get('current_galaxy_name', 'Galaxy').removesuffix('.txt').removesuffix('.csv').removesuffix('.dat')

                                    # Calcola la frazione reale di Dark Matter per QUESTA specifica galassia
                                    if gal_state.get('DATA_LOADED') and len(gal_state.get('r_ngc', [])) > 0:
                                        r_ngc = gal_state['r_ngc']
                                        y_disk = gal_state.get('upsilon_disk', 0.5)
                                        y_bulge = gal_state.get('upsilon_bulge', 0.7)
                                        
                                        v_gas_ngc = gal_state['v_gas_ngc']
                                        v_disk_ngc = gal_state['v_disk_ngc']
                                        v_bul_ngc = gal_state['v_bul_ngc']
                                        
                                        # Massa Barionica Totale
                                        V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                                        m_baryonic = (np.maximum(V_bar_sq, 0) * r_ngc) / G_grav
                                        val_m_bar = np.nanmax(m_baryonic) if len(m_baryonic) > 0 else 0
                                        
                                        # Massa Oscura Ideale (best-fit dal dataset)
                                        rho_s = gal_state.get('base_rho_s', 0.0)
                                        r_s = gal_state.get('base_r_s', 1.0)
                                        M_dm_grid_best = M_burkert_enclosed(r_ngc, rho_s, r_s)
                                        val_m_dm_best = np.nanmax(M_dm_grid_best) if len(M_dm_grid_best) > 0 else 0
                                        
                                        # Frazione reale target (es. 0.75 -> 75% di DM sul totale)
                                        total_mass = val_m_bar + val_m_dm_best
                                        if total_mass > 0:
                                            target_fraction = val_m_dm_best / total_mass

                                    # Valore attuale impostato dallo slider dell'utente (espresso come percentuale DM/Tot)
                                    # Ricaviamo la frazione attuale visualizzata sull'etichetta dello slider o calcolata
                                    f_slider = float(alpha_slider.value)
                                    
                                    # Se lo slider controlla il fattore f, calcoliamo la frazione di DM corrispondente
                                    current_dm_mass = val_m_dm_best * f_slider
                                    current_total_mass = val_m_bar + current_dm_mass
                                    current_fraction = current_dm_mass / current_total_mass if current_total_mass > 0 else 0
                                    
                                
                                    if current_fraction < 0.05:
                                        accessible_notify("Dark matter missing! Move the slider to add mass.", type_='warning')
                                    elif current_fraction < target_fraction - 0.04:
                                        accessible_notify(f"More Dark Matter needed! For {gal_name}, the ideal DM fraction is ~{target_fraction*100:.1f}%.", type_='warning')
                                    elif current_fraction > target_fraction + 0.04:
                                        accessible_notify(f"Too much Dark Matter! For {gal_name}, the ideal DM fraction is ~{target_fraction*100:.1f}%.", type_='warning')
                                    else:
                                        accessible_notify(f"Perfect! The Dark Matter halo matches the real value for {gal_name} (~{target_fraction*100:.1f}%).", type_='success')
                                        
                                    render_3d_mass_plot.refresh()
                                    await asyncio.sleep(0.1)
                                    galaxy_3d_dialog.open()
                                    ui.run_javascript("setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 200);")
                                                               
                                galaxy_img_disp = ui.image().classes('w-full h-28 rounded cursor-pointer hover:scale-105 transition-transform').props('role=button tabindex=0 aria-label="Open 3D Galaxy Model"').on('click', open_galaxy_3d).on('keydown.enter', open_galaxy_3d)
                                #ui.html("<small class='text-gray-500'><i>Click to enlarge (Ref: ESA Hubble)</i></small>") #ui.html("<small class='text-gray-500'><i>Click to enlarge (Ref: ESA Hubble)</i></small>")
                                galaxy_info_content = ui.column().classes('w-full text-sm text-gray-700 gap-0 p-0 m-0')
                            with ui.row().classes('w-full justify-center mt-2 shrink-0'):
                                aria_button("Image & Table", "table", on_click=table_dialog2.open).classes("!bg-blue-500 text-white font-bold py-1 px-2 rounded mt-auto")

            
                        with ui.column().classes('w-full justify-between items-center'):
                            plot_container = ui.column().classes('w-full')
                            with ui.row().classes('w-full justify-center mt-2'):
                                
                                aria_button("Activity: Galaxy", "Instruction for plot galaxy panel", on_click=safe_click(lambda: [instr_combined_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                aria_button("Dataset ", "Info galaxy dataset",on_click=safe_click(lambda: [data_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded" )
                                
                    
                    
                        with ui.column().classes('w-full justify-between items-center'):
                            mass_plot_container = ui.column().classes("w-full")
                            with ui.row().classes("w-full justify-center mt-2"):
                                ui.html('''
        <style>
            .audio-menu-wrapper {
                position: relative;
                display: inline-block;
                z-index: 50 !important; /* Abbassato per stare SOTTO i dialoghi */
            }
            .audio-dropdown-content {
                display: none; 
                position: absolute;
                left: 50%;
                transform: translateX(-50%);
                bottom: 100%;
                margin-bottom: 0.5rem;
                width: 480px;
                background-color: #1e293b; 
                border: 1px solid #475569; 
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.8);
                flex-direction: column;
                gap: 1rem;
                z-index: 50 !important; /* Abbassato */
            }
            .audio-dropdown-content::after {
                content: "";
                position: absolute;
                top: 100%;  
                left: 0;
                width: 100%;
                height: 1rem; 
                background: transparent;
            }
            .audio-menu-wrapper:hover .audio-dropdown-content {
                display: flex !important;
            }
        </style>

        <div class="audio-menu-wrapper">
            <button style="background-color: #16a34a; transition: background-color 0.2s; display: flex; align-items: center; gap: 8px; color: white; font-weight: bold; padding: 0.5rem 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);" 
                onmouseover="this.style.backgroundColor='#15803d'" 
                onmouseout="this.style.backgroundColor='#16a34a'">
                🔊 Audio Tools
                <svg style="width: 16px; height: 16px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            
            <div class="audio-dropdown-content">
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #475569; padding-bottom: 0.75rem;">
                    <span style="color: white; font-weight: bold; font-size: 1.125rem;">Sonification Controls</span>
                    <button id="audio-button" role="button" aria-label="activate or deactivate audio" aria-pressed="false" onclick="initAudio()" 
                        style="background-color: #22c55e; color: black; font-weight: bold; padding: 0.4rem 0.75rem; border-radius: 0.25rem; border: none; cursor: pointer;">
                        Activate Audio
                    </button>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <button role="button" aria-label="Play observed velocity curve mean" onclick="playObservedVelMean()" style="background-color: #2563eb; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Observed Vel (mean)</button>
                    <button role="button" aria-label="Play observed velocity curve" onclick="playObservedVelCurve()" style="background-color: #60a5fa; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Observed Vel (curve)</button>
                    
                    <button role="button" aria-label="Play baryonic velocity curve mean" onclick="playBaryonicVelMean()" style="background-color: #dc2626; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Baryonic Vel (mean)</button>
                    <button role="button" aria-label="Play baryonic velocity curve" onclick="playBaryonicVelCurve()" style="background-color: #f87171; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Baryonic Vel (curve)</button>
                    
                    <button role="button" aria-label="Play simulated velocity curve mean" onclick="playSimVelMean()" style="background-color: #16a34a; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Simulated Vel (mean)</button>
                    <button role="button" aria-label="Play simulated velocity curve" onclick="playSimVelCurve()" style="background-color: #4ade80; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Simulated Vel (curve)</button>
                    
                    <button role="button" aria-label="Play difference velocity curves mean" onclick="playDifferenceVelMean()" style="background-color: #9333ea; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Difference (mean)</button>
                    <button role="button" aria-label="Play difference velocity curves" onclick="playDifferenceVelCurve()" style="background-color: #c084fc; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Difference (curve)</button>
                </div>
            </div>
        </div>
        ''').classes('inline-block')
                        
                        ui.label("").props('id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0').classes('hidden')
                
                           
                        with ui.column().classes('w-full justify-between items-center'):
                            chi2_plot_container = ui.column().classes('w-full')
                            with ui.row().classes("w-full justify-center mt-2"):
                                
                                aria_button("Add point", "Add point", on_click=lambda: add_chi2_point()).classes("!bg-green-600 text-white font-bold py-1 px-2 rounded ")
                                
                                aria_button("Reset", "Reset", on_click=lambda: refresh_chi2_plot()).classes("!bg-green-600 text-white font-bold py-1 px-2 rounded ")
                                
                                aria_button("Tool", "Tool", on_click=lambda:[chi2_input_dialog.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-green-600 text-white font-bold py-1 px-2 rounded ")
                                        
                
                
                

                    
                    with ui.row().classes("gap-4 "):
                        points_display
                        history_display
                        
                        


                        def update_galaxy_rotation_plot():
                            if not gal_state['DATA_LOADED']: return
                            r_ngc = gal_state['r_ngc']
                            v_obs_ngc = gal_state['v_obs_ngc']
                            v_gas_ngc = gal_state['v_gas_ngc']
                            v_bul_ngc = gal_state['v_bul_ngc']
                            v_disk_ngc = gal_state['v_disk_ngc']
                            v_err_ngc = gal_state['v_err_ngc']
                            current_galaxy_name = gal_state['current_galaxy_name']
                            selected_file = gal_state['selected_file']
                            
                            f = float(alpha_slider.value)
                            G_grav = 4.30091e-6
                            
                            y_disk = gal_state.get('upsilon_disk', 0.5)
                            y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                            v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            
                            rho_s, r_s, M_dm_grid_f = get_dm_params(f, r_array=r_ngc) 
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + M_dm_grid_f) / r_ngc)
                            
                          
                            M_dm_tot = np.nanmax(M_dm_grid_f) if len(M_dm_grid_f) > 0 else 0
                            m_total_model = m_baryonic + M_dm_grid_f 
                            val_m_tot = np.nanmax(m_total_model) if len(m_total_model) > 0 else 0
                            val_m_bar = np.nanmax(m_baryonic) if len(m_baryonic) > 0 else 0
                            
                            dm_ratio_gal = (M_dm_tot / val_m_tot) if val_m_tot > 0 else 0.0
                            # -----------------------------------------------------
                            
                            clean_name = current_galaxy_name.removesuffix('.txt').removesuffix('.csv').removesuffix('.dat')
                            galaxy_title_label.set_text(f"Galaxy Info: {clean_name}")
                           
                            galaxy_info_content.clear()
                            with galaxy_info_content:
                                html_info_box(f"""
                                    <ul class="list-none pl-2 space-y-1 text-sm text-gray-800">
                                        <li><b>M<sub>tot</sub>:</b> {format_sci(val_m_tot)} M<sub>☉</sub></li>
                                        <li><b>M<sub>DM</sub>:</b> {format_sci(M_dm_tot)} M<sub>☉</sub> <span style="color: #0284c7; font-weight: bold;">({dm_ratio_gal*100:.1f}%)</span></li>
                                        <li><b>M<sub>bar</sub>:</b> {format_sci(val_m_bar)} M<sub>☉</sub></li>
                                       <li><b>v<sub>obs,mean</sub>:</b> {np.nanmean(v_obs_ngc):.1f} km/s</li>
                                        <li><b>v<sub>sim,mean</sub>:</b> {np.nanmean(v_total_curve):.1f} km/s</li>
                                    </ul>
                                """)
                            exts = ['.jpg', '.jpeg', '.png', '.webp']
                            base_name = os.path.splitext(selected_file)[0]
                            found_img = None

                            for ext in exts:
                                candidate = base_name + ext
                                if os.path.exists(os.path.join(GALAXY_IMG_PATH, candidate)):
                                    found_img = candidate
                                    break

                            if found_img:
                                galaxy_img_disp.source = f"/galaxy_img/{found_img}"
                            else:
                                galaxy_img_disp.source = ""

                            v_obs_list   = [float(v) for v in v_obs_ngc if np.isfinite(v)]
                            v_bary_list  = [float(v) for v in v_baryonic if np.isfinite(v)]
                            v_total_list = [float(v) for v in v_total_curve if np.isfinite(v)]
                            v_obs_mean  = float(np.nanmean(v_obs_list)) if len(v_obs_list) > 0 else 0
                            v_bary_mean = float(np.nanmean(v_bary_list)) if len(v_bary_list) > 0 else 0
                            v_total_mean= float(np.nanmean(v_total_list)) if len(v_total_list) > 0 else 0
                            v_diff_raw = np.abs(v_obs_ngc - v_total_curve)
                            v_diff_list = [float(d) for d in v_diff_raw if np.isfinite(d)]
                            v_diff_mean = float(np.nanmean(v_diff_raw)) if v_diff_list else 0.0

                            ui.run_javascript(f"window.vObsSeries = {v_obs_list}; window.vBarySeries = {v_bary_list}; window.vSimSeries = {v_total_list}; window.vObsMean = {v_obs_mean}; window.vBarMean = {v_bary_mean}; window.vSimMean = {v_total_mean}; window.vDiffSeries = {v_diff_list}; window.vDiffMean = {v_diff_mean};")

                       
                            alpha_slider.props(f'label-value="DM / Mₜₒₜ: {dm_ratio_gal * 100:.1f}%"')

                            with plot_container:
                                plot_container.clear()
                                plt.close() 
                                with ui.pyplot(figsize=(6, 4), close=False, clear=True):
                                    ax = plt.gca()
                                    plt.plot(r_ngc, v_baryonic, color='red', linewidth=3, label=f'Keplerian velocity ')
                                    plt.plot(r_ngc, v_total_curve, linewidth=3, color='green', label="Simulated velocity" )
                                    plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', markersize=4, color='blue', ecolor='gray', capsize=2, label='Observed velocity', zorder=5)
                                    
                                    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                                    
                                    plt.xlabel('Radius (kpc)', fontsize=10)
                                    plt.ylabel('Rotation Speed (km/s)', fontsize=10)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt').removesuffix('.dat')
                                    plt.title('Galaxy rotation curve', fontsize=12,fontweight='bold')
                                
                                    max_vel_plot = np.nanmax([np.nanmax(v_obs_ngc + v_err_ngc), np.nanmax(v_total_curve), np.nanmax(v_baryonic)]) if len(v_obs_ngc) > 0 else 300
                                    max_rad_plot = np.nanmax(r_ngc) if len(r_ngc) > 0 and not np.isnan(np.nanmax(r_ngc)) else 10.0
                                    plt.ylim(0, max(300, 1.1 * max_vel_plot))
                                    plt.xlim(0, max_rad_plot * 1.1)
                                    plt.grid(True)
                                    plt.legend(loc='upper left', fontsize=10)
                                    plt.tight_layout()
                                    ui.element('div').props('role=status aria-live=polite tabindex=0')


                        
                        def update_mass_plot():
                          
                            if not gal_state['DATA_LOADED']:
                                return
                            r_ngc = gal_state['r_ngc']
                            v_obs_ngc = gal_state['v_obs_ngc']
                            v_gas_ngc = gal_state['v_gas_ngc']
                            v_bul_ngc = gal_state['v_bul_ngc']
                            v_disk_ngc = gal_state['v_disk_ngc']
                            v_err_ngc = gal_state['v_err_ngc']
                          
                            current_galaxy_name = gal_state['current_galaxy_name']
                            selected_file = gal_state['selected_file']
                            f = float(alpha_slider.value)
                            y_disk = gal_state.get('upsilon_disk', 0.5)
                            y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                            v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            
                          
                            rho_s, r_s, M_dm_grid_f = get_dm_params(f, r_array=r_ngc) 
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + M_dm_grid_f) / r_ngc)
                            
                         
                            M_dm_tot = np.nanmax(M_dm_grid_f) if len(M_dm_grid_f) > 0 else 0
                            m_total_model = m_baryonic + M_dm_grid_f
                            m_total_obs = (v_obs_ngc**2 * r_ngc) / G_grav
                            y_min_mass = 0
                            
                            # --- CORREZIONE QUI: uso nanmax per sicurezza ---
                            y_max_mass = max(np.nanmax(m_total_obs), np.nanmax(m_baryonic)*2) * 1.1 if len(m_total_obs) > 0 else 1e10
                            
                            x_min_mass = r_ngc.min() * 0.9
                            x_max_mass = r_ngc.max() * 1.1
                            with mass_plot_container:
                                mass_plot_container.clear()
                                plt.close()
                                with ui.pyplot(figsize=(6, 4), close=False,clear=True):
                                    ax = plt.gca()
                                    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                    ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                                    plt.plot(r_ngc, m_baryonic, "r-", lw=3, label=f"Baryonic mass")
                                    plt.plot(r_ngc, m_total_obs, "b-", lw=3, label=f"Total mass")
                                    plt.plot(r_ngc, m_total_model, "g-", lw=3, label=f"Simulated mass")
                                    
                                    
                                   
                                    plt.xlim(x_min_mass, x_max_mass)
                                    plt.ylim(y_min_mass, y_max_mass)

                                    plt.xlabel("Radius (kpc)", fontsize=10)
                                    plt.ylabel("Mass ($M_\\odot$)", fontsize=10)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt').removesuffix('.dat')
                                    plt.title("Galaxy mass ", fontsize=12,fontweight='bold')

                                    plt.legend(loc='upper left', fontsize=10)
                                    plt.grid(True)
                                    plt.tight_layout()

                                    ui.element('div').props(
        'role=status aria-live=polite tabindex=0 aria-label=Total mass of the galaxy (from data), DMyonic mass (computed from data) and simulated mass updated with new dark matter value'
    )


                        def update_morphology_plot():
                               
                                if not gal_state['DATA_LOADED']:
                                    return
                                r_ngc = gal_state['r_ngc']
                                v_obs_ngc = gal_state['v_obs_ngc']
                                v_gas_ngc = gal_state['v_gas_ngc']
                                v_bul_ngc = gal_state['v_bul_ngc']
                                v_disk_ngc = gal_state['v_disk_ngc']
                                v_err_ngc = gal_state['v_err_ngc']
                              
                                current_galaxy_name = gal_state['current_galaxy_name']
                                selected_file = gal_state['selected_file']
                                f = float(alpha_slider.value) 

                        
                                y_disk = gal_state.get('upsilon_disk', 0.5)
                                y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            
                                V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                                v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                                m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                                
                              
                                rho_s, r_s, M_dm_grid_f = get_dm_params(f, r_array=r_ngc) 
                                v_total_curve = np.sqrt(G_grav * (m_baryonic + M_dm_grid_f) / r_ngc)
                                
                            
                                M_dm_tot = np.nanmax(M_dm_grid_f) if len(M_dm_grid_f) > 0 else 0
                                m_total_model = m_baryonic + M_dm_grid_f 

    
                                M_bulge = np.max((v_bul_ngc**2 * r_ngc) / G_grav) if len(r_ngc) > 0 else 0
                                M_disk  = np.max((v_disk_ngc**2 * r_ngc) / G_grav) if len(r_ngc) > 0 else 0
                                M_bary_tot = M_disk + M_bulge

                                # --- CORREZIONE QUI: val_m_bar previene che R_halo diventi un array causando un crash ---
                                val_m_bar = np.nanmax(m_baryonic) if len(m_baryonic) > 0 else 0
                                R_bulge = np.average(r_ngc, weights=np.maximum(0, v_bul_ngc**2)) if np.any(v_bul_ngc > 0) else 1
                                R_disk  = np.average(r_ngc, weights=np.maximum(0, v_disk_ngc**2)) if np.any(v_disk_ngc > 0) else 5
                                R_halo  = np.sqrt(M_dm_tot / (val_m_bar+1e-12)) * np.max(r_ngc) if M_dm_tot > 0 else 0
                            

                                with morph_plot_container:
                                    morph_plot_container.clear()
                                    plt.close()
                                    with ui.pyplot(figsize=(6,6), close=False,clear=True):
                                    
                                        
                                        ax = plt.gca()
                                        def draw_gradient_circle(ax, center, radius, color, n_steps=10, max_alpha=0.4):
                                            for i in range(n_steps):
                                                r = radius * (1 - i/n_steps)
                                              
                                                alpha = (max_alpha / n_steps) 
                                                circle = plt.Circle(center, r, color=color, alpha=alpha, linewidth=0)
                                                ax.add_artist(circle)
                                          
                                            ax.add_artist(plt.Circle(center, radius, color=color, fill=False, alpha=0.5, linewidth=1, linestyle='--'))

                                        maxR = float(np.max(r_ngc)) * 1.1 if len(r_ngc) > 0 else 10.0
                                        R_halo = min(R_halo, maxR) 
                                       

                                        if f > 0 and R_halo > 0:
                                            halo = plt.Circle((0,0), R_halo, color='limegreen', alpha=0.1,zorder=1,label=f'Halo DM (M={M_dm_tot:.1e}')
                                            ax.add_artist(halo)
                                            halo_edge = plt.Circle((0,0), R_halo, color='limegreen', 
                                     fill=False, linestyle='--', alpha=0.5, zorder=1)
                                            ax.add_artist(halo_edge)
                                        disk = plt.Circle((0,0), R_disk, color='crimson', alpha=0.3, zorder=2,label=f'Disk (M={M_disk:.1e})')
                                        ax.add_artist(disk)
                                        bulge = plt.Circle((0,0), R_bulge, color='darkorange', alpha=0.85, linewidth=2,zorder=3,label='Bulge')
                                        ax.add_artist(bulge)
                                       
                                        bh = plt.Circle((0,0), 0.5, color='black', zorder=4, label='Black Hole')
                                        ax.add_artist(bh)


                                        
                                        ax.set_xlim(-maxR, maxR)
                                        ax.set_ylim(-maxR, maxR)
                                        ax.set_aspect('equal', 'box')
                                        ax.set_xlabel("x [kpc]",fontsize=14)
                                        ax.set_ylabel("y [kpc]",fontsize=14)
                                        galaxy_name_for_title = current_galaxy_name.removesuffix('.txt').removesuffix('.dat')
                                        ax.set_title(f"Galaxy structure (top-view): {galaxy_name_for_title}",fontsize=16,fontweight='bold')

                                        ax.legend(loc='upper right',fontsize=14)
                                        plt.tight_layout()
                                        ui.element('div').props(
        'role=status aria-live=polite tabindex=0 aria-label=Galaxy morphology representation (from above), showing disk with baryonic component (stars,gas) and halo appears adding dark matter (slider)'
    )

                                 
                    
                        

                    
                        
                        def chi2_function_for_minimization(target_fraction):
                            r_ngc = gal_state['r_ngc']
                            v_obs_ngc = gal_state['v_obs_ngc']
                            v_gas_ngc = gal_state['v_gas_ngc']
                            v_bul_ngc = gal_state['v_bul_ngc']
                            v_disk_ngc = gal_state['v_disk_ngc']
                            v_err_ngc = gal_state['v_err_ngc']
                          
                            current_galaxy_name = gal_state['current_galaxy_name']
                            galaxy_name = current_galaxy_name
                            if not galaxy_name or len(r_ngc) == 0:
                                return np.inf
                            
                            y_disk = gal_state.get('upsilon_disk', 0.5)
                            y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                            
                           
                            rho_s, r_s, unscaled_M_dm_grid = get_dm_params(1.0, r_array=r_ngc)
                            max_dm_unscaled = np.nanmax(unscaled_M_dm_grid) if len(unscaled_M_dm_grid) > 0 else 0
                           
                            v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            val_m_bar = np.nanmax(m_baryonic) if len(m_baryonic) > 0 else 0
                            
                            if max_dm_unscaled <= 1e-6 or target_fraction >= 1.0 or target_fraction < 0: 
                                return np.inf

                          
                            f_val = (target_fraction * val_m_bar) / (max_dm_unscaled * (1.0 - target_fraction))
                            
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + f_val * unscaled_M_dm_grid) / r_ngc)

                            mask = np.isfinite(v_obs_ngc) & np.isfinite(v_err_ngc) & (v_err_ngc > 0)
                            vobs_use = v_obs_ngc[mask]
                            verr_use = np.maximum(v_err_ngc[mask], 5.0)
                            vmodel_use = v_total_curve[mask]
                            dof = max(1, len(vobs_use) - 3) 
                            chi2_dof = np.sum(((vobs_use - vmodel_use)/verr_use)**2) / dof
                            
                            return float(chi2_dof)

                        def add_chi2_point():
                            r_ngc = gal_state['r_ngc']
                            v_obs_ngc = gal_state['v_obs_ngc']
                            v_gas_ngc = gal_state['v_gas_ngc']
                            v_bul_ngc = gal_state['v_bul_ngc']
                            v_disk_ngc = gal_state['v_disk_ngc']
                            v_err_ngc = gal_state['v_err_ngc']
                          
                            chi2_points = gal_state['chi2_points']
                          
                            f = float(alpha_slider.value)
                            y_disk = gal_state.get('upsilon_disk', 0.5)
                            y_bulge = gal_state.get('upsilon_bulge', 0.7)
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_disk * (v_disk_ngc * np.abs(v_disk_ngc)) + y_bulge * (v_bul_ngc * np.abs(v_bul_ngc))
                            v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            
                            rho_s, r_s, M_dm_grid_f = get_dm_params(f, r_array=r_ngc) 
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + M_dm_grid_f) / r_ngc)
                            
                         
                            M_dm_tot = np.nanmax(M_dm_grid_f) if len(M_dm_grid_f) > 0 else 0
                            val_m_tot = np.nanmax(m_baryonic + M_dm_grid_f) if len(m_baryonic + M_dm_grid_f) > 0 else 0
                            dm_fraction = (M_dm_tot / val_m_tot) if val_m_tot > 0 else 0.0

                            mask = np.isfinite(v_obs_ngc) & np.isfinite(v_err_ngc) & (v_err_ngc > 0)
                            vobs_use = v_obs_ngc[mask]
                            verr_use = np.maximum(v_err_ngc[mask], 5.0)
                            vmodel_use = v_total_curve[mask]

                            chi2_val = float(np.sum(((vobs_use - vmodel_use)/verr_use)**2) / max(1, len(vobs_use) - 3))
                           
                           
                            is_duplicate = any(abs(x - dm_fraction) < 1e-4 for x, y in chi2_points)
                            if is_duplicate:
                                chi2_state['slider_result'] = "Point already added! Move the slider."
                                plot_chi2_user_curve()
                                return
                                
                            chi2_points.append((dm_fraction, chi2_val))
                            
                            if len(chi2_points) > 3: chi2_points.pop(0)

                            if len(chi2_points) == 3:
                                xs, ys = zip(*chi2_points)
                                coeffs = np.polyfit(xs, ys, 2)
                                if coeffs[0] > 0:
                                    xmin = -coeffs[1] / (2*coeffs[0])  
                                    ymin = np.polyval(coeffs, xmin)
                                    if xmin < 0 or xmin >= 1.0 or ymin < 0:
                                        chi2_state['slider_result'] = "Invalid Result: Fraction must be 0 to 1"
                                    else:
                                        chi2_state['slider_result'] = f"Min Fraction ≈ {xmin*100:.1f}%, χ²_min ≈ {ymin:.2f}"
                                else:
                                    chi2_state['slider_result'] = "Invalid shape. Try different values!"
                            else:
                                chi2_state['slider_result']= "Add more points to find the minimum."

                            #update_all_plots.refresh()
                            plot_chi2_user_curve()
                            update_displays.refresh()

                        def initialize_parabolic_points():
                            manual_points = gal_state['manual_points']
                            parabolic_state["points"].clear()
                            parabolic_state["history"].clear()
                            parabolic_state["plot_points"].clear()
                            parabolic_state["iteration"] = 0

                            masses = [input_a.value, input_b.value, input_c.value]
                            if not all(isinstance(v, (int,float)) for v in masses):
                                accessible_notify("Insert 3 valid fractions (0 to 1)", type_='warning')
                                return
                        
                            chi2_state['slider_result']= "---"
                        
                            sorted_m = sorted(masses)
                            if abs(sorted_m[2]-sorted_m[0])/(abs(sorted_m[0])+1e-12)<1e-6:
                                accessible_notify("Values are too close", type_='warning')
                                return
                            
                            manual_points.clear()
                            points = [(m, chi2_function_for_minimization(m)) for m in sorted(masses)]
                        
                            fa_in.value = points[0][1]
                            fb_in.value = points[1][1]
                            fc_in.value = points[2][1]

                            formula_pts["fa"], formula_pts["fb"], formula_pts["fc"] = fa_in.value, fb_in.value, fc_in.value
                            formula_pts["a"], formula_pts["b"], formula_pts["c"] = points[0][0], points[1][0], points[2][0]

                            refresh_formula_from_inputs()
                        
                            parabolic_state["points"] = points
                            manual_points.extend(points)
                        
                            xs, ys = zip(*points)
                            coeffs = np.polyfit(xs, ys, 2)
                            
                            if coeffs[0] > 0:
                                xmin = -coeffs[1] / (2 * coeffs[0])
                                ymin = np.polyval(coeffs, xmin)
                                
                                if xmin < 0 or xmin >= 1.0 or ymin < 0:
                                    result_label.set_text("Invalid Result!")
                                else:
                                    result_label.set_text(f"Min Fraction = {xmin*100:.1f}%,  χ² min = {ymin:.2f}")
                            else:
                                result_label.set_text("Invalid shape. Try different values!")
                        
                            accessible_notify("Points computed. Check the minimum on the plot.", type_='success')
                            update_displays.refresh()
                            plot_chi2_user_curve()

                        @ui.refreshable
                        def update_displays():
                            points_display.clear()
                            with points_display:
                                if parabolic_state["points"]:
                                    section("Algorithm Points")
                                    for x,y in sorted(parabolic_state["points"], key=lambda p: p[0]):
                                        ui.label(f"Fraction = {x*100:.1f}%, χ²/dof = {y:.2f}")
                            history_display.clear()
                            with history_display:
                                if parabolic_state["history"]:
                                    section("History step")
                                    for entry in parabolic_state["history"]:
                                        ui.markdown(entry)

                        def refresh_chi2_plot():
                            chi2_points = gal_state['chi2_points']
                            manual_points = gal_state['manual_points']
                            chi2_points.clear()
                            parabolic_state["points"].clear()
                            parabolic_state["plot_points"].clear()
                            parabolic_state["history"].clear()
                            parabolic_state["iteration"] = 0
                            manual_points.clear()
                    
                            formula_pts["auto_update"] = False 
                            chi2_state['slider_result']= "---"
                            update_all_plots.refresh()

               
                    update_all_plots()
                    

                        


#panel galaxy exercise

        
                with ui.tab_panel('galdm').props('role=tabpanel'):
                # with ui.card().classes("p-4 !bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg"):
                    ui.label(" Explore the influence of dark matter on galaxy rotation curves of different datasets.  ").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')

                    with ui.dialog() as info_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Information about the galaxy rotation curve activity" role=dialog'):
                        html_info_box(r"""
        <h3>Analysis Overview</h3>
        <p>You are an astrophysicist investigating the <b>presence of dark matter</b> in a galaxy.</p>
        <ul>
            <li>Choose a dataset and complete the missing fields with the correct formulas.</li>
            <li>Click <b>Run Analysis</b> to compare the baryonic mass with the total mass and the observed velocity with the Keplerian-like velocity.</li>
        </ul>
    """).props('aria-label="Descriptive text about galaxy velocity and mass activities"')
                        aria_button("Close", "close the box",on_click=lambda:info_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as cur_mass, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="Curiosity about ghost galaxies" role=dialog'):
                        html_info_box(r"""
                        <h3>Ghost Galaxies</h3>
                        <p>In 2016, astronomers discovered <b>Dragonfly 44</b>, a galaxy that has the same mass as the Milky Way but only 1% of the stars.</p>
                        <p>It is made almost entirely of <b>99.9% Dark Matter</b>. It is a "failed" galaxy that never formed many stars but retained its massive dark halo.</p>
                        """)
                        reference_box("**Source:** [Space](https://www.space.com/33850-weird-galaxy-is-mostly-dark-matter.html)")
                        aria_button("Close", "close", on_click=lambda:cur_mass.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                    def open_baryonic_analysis_dialog():
                        global plots_popup_container
                        
                     
                        with ui.dialog() as comp_baryonic_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="baryonic-title"'):
                            
                            with ui.column().classes('w-full h-full bg-white flex flex-col'):
                                
                                with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="baryonic-title"'):
                                    ui.label('Baryonic Analysis & Galaxy Plots').classes('text-xl font-bold').props('id="baryonic-title" role=heading aria-level=2 tabindex=0')
                                    aria_button('Close', 'close', on_click=comp_baryonic_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                                with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                    
                              
                                    t_math = ui.tab('Baryonic Math')
                                    t_legend = ui.tab('Legend and Units')
                                   
                                    t_plots = ui.tab('Galaxy Plots')
                                   
                                    tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))

                                with ui.tab_panels(tabs, value=t_math).classes('w-full flex-1 overflow-y-auto bg-gray-50 text-slate-900 p-6'):
                                    
                                 
                                    with ui.tab_panel(t_plots).classes('flex flex-col items-center justify-start'):
                                        ui.label("Galaxy Velocity and Mass Profiles").classes("text-2xl font-bold mb-4 text-slate-800")
                                        ui.select(options=galaxy_select.options, label='Select a Galaxy Dataset') \
                                            .bind_value(galaxy_select, 'value') \
                                            .classes('w-64 mb-6 bg-slate-800 text-white rounded shadow-sm') \
                                            .props('behavior="menu" outlined popup-content-class="bg-slate-800 text-white"') \
                                            .on_value_change(lambda e: update_plots_popup.refresh())
                                        plots_popup_container = ui.column().classes("w-full items-center justify-center")
                                        update_plots_popup()
                                        
                                      
                                      
                                        info_box("**Dataset variables**: Rad (radius), Vobs (observed velocity), errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity), SBdisk (surface brightness disk), SBbul (surface brightness bulge)")
                                        reference_box("""**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')

                                        
                                    
                                  
                                 
                                    with ui.tab_panel(t_math):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>
                                        <h3>Baryonic Calculation Steps</h3>
                                        
                                        <ul>
                                            <li>
                                                <b>Step 1:</b> Compute the baryonic velocity from data (conserving the gas rotation sign and applying the disk and bulge Mass-to-Light ratios \(\Upsilon_{\mathrm{disk}}, \Upsilon_{\mathrm{bulge}}\)):<br>
<span class="math">\( v_{\mathrm{bar}}^2 = v_{\mathrm{gas}}|v_{\mathrm{gas}}| + \Upsilon_{\mathrm{disk}}\,v_{\mathrm{disk}}|v_{\mathrm{disk}}| + \Upsilon_{\mathrm{bulge}}\,v_{\mathrm{bulge}}|v_{\mathrm{bulge}}| \)</span>
                                            </li>
                                            
                                            <li style="margin-top: 10px;">
                                                <b>Step 2:</b> Derive the baryonic mass:<br>
                                                <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2}{G} \)</span>
                                            </li>
                                            
                                            <li style="margin-top: 10px;">
                                                <b>Step 3:</b> Compute the total mass from observations:<br>
                                                <span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2}{G} \)</span>
                                            </li>
                                            
                                          
                                        </ul>
                                        """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the steps to compute baryonic velocity and mass"')

                                  
                                   
                                    with ui.tab_panel(t_legend):
                                        html_info_box(r"""
       <style>
        .formula-box { 
            background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; 
            color: #111; margin-top: 15px; overflow-x: auto; 
        }
        
        /* Eliminiamo i pallini e forziamo la riga singola */
        .math-list { 
            list-style: none !important; 
            padding: 0 !important; 
            margin: 0 !important; 
            white-space: nowrap !important;
        }
        
        .math-list li { 
            white-space: nowrap !important; 
            margin-bottom: 8px !important;
            display: block !important;
        }
        
        /* Forziamo MathJax a non andare a capo */
        mjx-container { 
            display: inline-block !important;
            white-space: nowrap !important;
        }
    </style>

        <div class="formula-box">
            <h3 class="text-xl font-bold mb-2">Legend of Symbols and Units</h3>
            <ul class="list-disc pl-5 space-y-1 math-list">
                <li><b>Rad</b> = Radius [\(\mathrm{kpc}\)]</li>
                <li><b>V_obs</b> = Observed rotation velocity [\(\mathrm{km/s}\)]</li>
                <li><b>V_gas, V_disk, V_bul</b> = Velocity contributions from gas, stellar disk, bulge [\(\mathrm{km/s}\)]</li>
                <li><b>G</b> = \( 4.30091 \times 10^{-6} \, \mathrm{kpc \cdot (km/s)^2 \cdot M_\odot^{-1}} \)</li>
                <li><b>M_baryonic</b> = Baryonic mass [\(\mathrm{M_\odot}\)]</li>
                <li><b>M_total</b> = Total mass (from observations) [\(\mathrm{M_\odot}\)]</li>
                <li><b>V_baryonic</b> = Baryonic velocity (Keplerian-like prediction) [\(\mathrm{km/s}\)]</li>
            </ul>
            
            <h3 class="text-xl font-bold mb-2 mt-4">Units Conversion</h3>
            <ul class="list-disc pl-5 space-y-1 math-list">
                <li><b>1 kpc</b> = \( 3.086 \times 10^{16} \, \mathrm{m} \) = \( 3.26 \times 10^3 \, \mathrm{light\text{-}years} \)</li>
                <li><b>1 pc</b> = \( 3.086 \times 10^{13} \, \mathrm{km} \) = \( 3.26 \, \mathrm{light\text{-}years} \)</li>
                <li><b>1 Mpc</b> = \( 10^6 \, \mathrm{pc} \) = \( 3.086 \times 10^{19} \, \mathrm{km} \)</li>
                <li><b>1 km/s</b> = \( 3.6 \times 10^3 \, \mathrm{km/h} \) = \( 10^3 \, \mathrm{m/s} \)</li>
                <li><b>1 \( \mathrm{M_\odot} \)</b> = \( 1.989 \times 10^{30} \, \mathrm{kg} \) (solar mass)</li>
                <li><b>1 \( \mathrm{L_\odot} \)</b> = \( 3.828 \times 10^{26} \, \mathrm{W} \) (solar luminosity)</li>
            </ul>
        </div>
""")

                                   
                                   

                        comp_baryonic_dialog.open()
                    with ui.row().classes('w-full gap-8 justify-center'):
                        galaxy_files = [f for f in os.listdir(GALAXY_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                        galaxy_file_map = get_data_and_images(GALAXY_DATA_PATH, GALAXY_IMG_PATH)
                        
                        if not galaxy_file_map:
                            ui.label("No galaxy data files found in 'App/galaxy_data/' directory.").classes('text-red-500')
                        else:
                        
                            galaxy_select = ui.select(galaxy_files, label='Select a Galaxy Dataset').classes('flex-1 max-w-md items-start text-lg')
                        
                        
                        aria_button("Instructions", "Read instructions",on_click=safe_click(lambda: [info_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button(
    "Scientific Info", 
    "Open comprehensive dialog for galaxy plots, data, and physics", 
    on_click=lambda: [
        open_baryonic_analysis_dialog(), 
        ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")
    ]
).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_mass.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")


                

                    
                    ui.add_head_html('''
    <style>
        .math-text { font-family: 'Times New Roman', serif; font-style: italic; font-size: 1.25rem; display: flex; align-items: center; flex-wrap: nowrap; }
        .math-sub { vertical-align: sub; font-size: 0.7em; margin-left: -2px; }
        .math-sup { vertical-align: super; font-size: 0.7em; }
        .fraction { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 4px; }
        .numerator { border-bottom: 2px solid white; padding-bottom: 2px; }
        .denominator { padding-top: 2px; }
        
        /* Modifica per la radice quadrata */
        .sqrt-content { 
            border-top: 2px solid white; 
            padding-top: 4px; 
            display: inline-flex; 
            align-items: center; 
            margin-top: 0.5em;
            white-space: nowrap; /* Impedisce a capo */
        }

        /* Input pulito */
        .formula-input .q-field__control { 
            height: 32px !important; min-height: 32px !important; 
            padding: 0 8px !important;
            background: rgba(255,255,255,0.1) !important;
        }
        .formula-input .q-field__native { color: #90caf9; text-align: center; padding: 0; }
    </style>
''')

                    ui.add_head_html('''
    <style>
        .sqrt-box {
            border-top: 2px solid #e2e8f0; /* Grigio chiaro per risaltare sul fondo scuro */
            border-left: 2px solid #e2e8f0;
            padding-left: 8px;
            padding-top: 4px;
            padding-bottom: 4px;
            display: flex;
            align-items: center; /* Fondamentale per il centraggio verticale */
            flex-wrap: nowrap;
        }
        .compact-select {
            width: 110px !important; 
        }
        .math-text {
            white-space: nowrap; /* Impedisce che una parentesi vada a capo da sola */
            display: flex;
            align-items: center;
        }
    </style>
''')
                    vars_options_gal = ['v_bar', 'v_obs', 'rad', 'g', 'm_tot', 'm_bar', 'v_gas', 'v_disk', 'v_bulge']
                    answer_vb, answer_vo,answer_rad,answer_g,answer_Mtot,answer_Mbar,answer_vgas,answer_vdisk,answer_vbulge = {}, {},{},{},{},{},{},{},{}
                    @ui.refreshable
                    def show_galaxy_pseudocode():
                    

                     with ui.card().classes("items-center justify-center p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                        ui.markdown("Galaxy Mass Exercise: fill the missing parts").classes("text-2xl font-bold text-blue-300 mb-4")
                     
                        with ui.row().classes('w-full gap-8 max-w-7xl flex-wrap items-start justify-center'):
                            
                     
                            with ui.column().classes('flex-[4] gap-4 min-w-[600px]'):
                                
                                ui.label("1) Load galaxy dataset").classes('text-blue-200 font-bold text-lg')
                                ui.code(f'data = load_data("{galaxy_select.value}")', language='python')

                                ui.label("2) Compute Baryonic Velocity").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center no-wrap gap-2'):
                                    ui.label('Vbar = ').classes('math-text')
                                    
                                    with ui.row().classes('sqrt-box items-center no-wrap gap-2'):
                                        ui.label('[').classes('math-text')
                                        answer_vgas['el'] = aria_select_input(vars_options_gal, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vgas| + ϒ · (').classes('math-text')
                                        answer_vdisk['el'] = aria_select_input(vars_options_gal, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vdisk| + ').classes('math-text')
                                        answer_vbulge['el'] = aria_select_input(vars_options_gal, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vbul|) ]').classes('math-text')

                                ui.label("3) Compute Luminous Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">bar</span> = ')
                                    with ui.column().classes('fraction'):
                                        with ui.row().classes('numerator items-center gap-1'):
                                            ui.html('(')
                                            answer_vb['el'] = aria_select_input(vars_options_gal, "Select variable for baryonic velocity")
                                            ui.html(')<span class="math-sup">2</span> &middot; ')
                                            answer_rad['el'] = aria_select_input(vars_options_gal, "Select variable for radius in luminous mass")
                                        with ui.row().classes('denominator w-full justify-center'):
                                            answer_g['el'] = aria_select_input(vars_options_gal, "Select variable for gravitational constant")


                          
                            with ui.column().classes('flex-[3] gap-6 min-w-[350px]'):
                                
                                ui.label("4) Compute Total Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">tot</span> = ')
                                    with ui.column().classes('fraction'):
                                        with ui.row().classes('numerator items-center gap-1'):
                                            ui.html('(')
                                            answer_vo['el'] = aria_select_input(vars_options_gal, "Select variable for observed velocity")
                                            ui.html(')<span class="math-sup">2</span> &middot; ')
                                            answer_rad['el2'] = aria_select_input(vars_options_gal, "Select variable for radius in total mass")
                                        with ui.row().classes('denominator w-full justify-center'):
                                            answer_g['el2'] = aria_select_input(vars_options_gal, "Select variable for gravitational constant in total mass")

                                ui.label("5) Compute Dark Matter Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">DM</span> = ')
                                    answer_Mtot['el'] = aria_select_input(vars_options_gal, "Select variable for total mass")
                                    ui.html('&minus;')
                                    answer_Mbar['el'] = aria_select_input(vars_options_gal, "Select variable for baryonic mass")

                                
                    show_galaxy_pseudocode()
                    galaxy_select.on('update:model-value', lambda e: show_galaxy_pseudocode.refresh())

                    
                    
                    
                    
                    plots_and_image_container = ui.column().classes('w-full items-center')
                    @ui.refreshable
                    def update_galaxy_mass_analysis():
                        plots_and_image_container.clear()
                        
                        with plots_and_image_container:
                            try:
                                results_data = []
                                csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                                if not os.path.exists(csv_path):
                                    csv_path = os.path.join(dataset_path, 'galaxy_best_parameters_2.csv')
                                
                                df_params = pd.read_csv(csv_path)
                                df_params['Clean_Galaxy'] = df_params['Galaxy'].astype(str).str.replace('_rotmod', '', case=False).str.strip().str.upper()

                               
                                for file in galaxy_files:
                                    gal_name = os.path.splitext(file)[0]
                                    gal_name_clean = gal_name.replace('_rotmod', '').strip().upper()
                                    
                                    row = df_params[df_params['Clean_Galaxy'] == gal_name_clean]
                                    if row.empty: continue
                                    
                                    if 'Upsilon_disk' in row.columns and 'Upsilon_bulge' in row.columns:
                                        y_disk = float(row.iloc[0]['Upsilon_disk'])
                                        y_bulge = float(row.iloc[0]['Upsilon_bulge'])
                                    else:
                                        y_disk = float(row.iloc[0].get('Upsilon', 0.5))
                                        y_bulge = 1.4 * y_disk
                                    
                                    r_s = float(row.iloc[0]['r_s'])
                                    rho_s = float(row.iloc[0]['rho_s'])

                                    data_ngc = get_galaxy_data_cached(file)
                                    if data_ngc is None or data_ngc.empty: continue
                                        
                                    r = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                                    vgas = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values if 'Vgas' in data_ngc.columns else np.zeros_like(r)
                                    vdisk = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values if 'Vdisk' in data_ngc.columns else np.zeros_like(r)
                                    vbul = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values if 'Vbul' in data_ngc.columns else np.zeros_like(r)
                                    
                                    V_bar_sq = vgas * np.abs(vgas) + y_disk * (vdisk * np.abs(vdisk)) + y_bulge * (vbul * np.abs(vbul))
                                    M_bar = (r * V_bar_sq) / G_grav
                                    
                                    # Massa Dark Matter con BURKERT
                                    x_arr = r / r_s
                                    M_dm = 2.0 * np.pi * rho_s * (r_s**3) * (0.5 * np.log(1.0 + x_arr**2) + np.log(1.0 + x_arr) - np.arctan(x_arr))
                                    
                                    M_tot = M_bar + M_dm
                                    
                                    if len(M_tot) > 0 and M_tot[-1] > 0:
                                        dm_fraction = M_dm[-1] / M_tot[-1]
                                        results_data.append({
                                            "Galaxy": gal_name_clean,
                                            "M_tot_edge": M_tot[-1],
                                            "DM_Perc": dm_fraction * 100.0
                                        })
                                
                                
                                if results_data:
                                    df_res = pd.DataFrame(results_data)
                                    fig = px.scatter(df_res, x="M_tot_edge", y="DM_Perc", hover_name="Galaxy",
                                                     log_x=True, color_discrete_sequence=['royalblue'],
                                                     labels={"M_tot_edge": "Total Mass (M_sun)", "DM_Perc": "Dark Matter Fraction (%)"},
                                                     title="Galaxies: Dark Matter Fraction vs Total Mass")
                                    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')), opacity=0.8)
                                    ui.plotly(fig).classes('w-full max-w-4xl h-96')
                                    ui.element('div').props('role=img tabindex=0 aria-label="Interactive scatter plot showing the dark matter fraction vs total mass for all analyzed galaxies"')

                                # Mostra Immagine e Tabella della singola galassia selezionata
                                selected_file = galaxy_select.value
                                if selected_file:
                                    image_filepath = galaxy_file_map.get(selected_file)
                                    with ui.row().classes('w-full gap-4 justify-start mt-6'):
                                        with ui.column().classes('flex-1 items-center'):
                                            if image_filepath and os.path.exists(image_filepath):
                                                file_name = os.path.basename(image_filepath)
                                                web_path = f'/galaxy_img/{file_name}'
                                                aria_image(web_path, f"Image of the selected galaxy {selected_file}").classes('w-full max-w-xl')
                                                reference_box("""
    **Image reference:**  
    - [ESA Hubble](https://esahubble.org/)
    """).classes('w-full text-center text-base italic mt-2')
                                            else:
                                                ui.label("Image not found").classes('text-red-500')

                                        with ui.column().classes('flex-1 items-center'):
                                            table_filepath = os.path.join(GALAXY_TABLES_PATH, os.path.splitext(selected_file)[0] + ".csv")
                                            if os.path.exists(table_filepath):
                                                df_table = pd.read_csv(table_filepath)
                                                ui.table.from_pandas(df_table).props('aria-label=Table of selected galaxy characteristics role=table').classes('w-full max-w-xl')
                                                reference_box("""
    **Galaxy information references:**  
    - [The Sky Live](https://theskylive.com/sky)  
    - [Wikipedia: List of NGC objects](https://en.wikipedia.org/wiki/List_of_NGC_objects)
    """).classes('w-full text-center text-base italic mt-2')
                                            else:
                                                ui.label("Table not found").classes('text-red-500')
                            except Exception as e:
                                warning_box(f"Error processing galaxies: {e}").classes('text-red-500')
                    
                    @ui.refreshable
                    def update_plots_popup():
                        global plots_popup_container 
                        plots_popup_container.clear()
                        
                        selected_file = galaxy_select.value
                        if not selected_file:
                            return
                        
                        data_ngc = get_galaxy_data_cached(selected_file)
                        if data_ngc is None or data_ngc.empty: return
                       
                        r = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                        vobs = pd.to_numeric(data_ngc['Vobs'], errors='coerce').values
                        verr = pd.to_numeric(data_ngc['errV'], errors='coerce').values
                        vgas = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values if 'Vgas' in data_ngc.columns else np.zeros_like(r)
                        vdisk = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values if 'Vdisk' in data_ngc.columns else np.zeros_like(r)
                        vbul = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values if 'Vbul' in data_ngc.columns else np.zeros_like(r)

                        gal_name = os.path.splitext(selected_file)[0]
                        gal_name_clean = gal_name.replace('_rotmod', '').strip().upper()
                        
                        try:
                            # Cerca di leggere il primo o il secondo csv
                            csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                            if not os.path.exists(csv_path):
                                csv_path = os.path.join(dataset_path, 'galaxy_best_parameters_2.csv')
                            
                            df_params = pd.read_csv(csv_path)
                            df_params['Clean_Galaxy'] = df_params['Galaxy'].astype(str).str.replace('_rotmod', '', case=False).str.strip().str.upper()
                            
                            row = df_params[df_params['Clean_Galaxy'] == gal_name_clean]
                            
                            if not row.empty:
                                if 'Upsilon_disk' in row.columns and 'Upsilon_bulge' in row.columns:
                                    y_disk = float(row.iloc[0]['Upsilon_disk'])
                                    y_bulge = float(row.iloc[0]['Upsilon_bulge'])
                                else:
                                    y_disk = float(row.iloc[0].get('Upsilon', 0.5))
                                    y_bulge = 1.4 * y_disk
                                    
                                r_s = float(row.iloc[0]['r_s'])
                                rho_s = float(row.iloc[0]['rho_s'])
                            else:
                                accessible_notify(f"Error: Parameters for {gal_name} not found in CSV!", type_='error')
                                return
                        except Exception as e:
                            accessible_notify(f"CSV Read Error: {e}", type_='error')
                            return

                     
                        V_bar_sq = vgas * np.abs(vgas) + y_disk * (vdisk * np.abs(vdisk)) + y_bulge * (vbul * np.abs(vbul))
                        V_bar = np.sqrt(np.maximum(V_bar_sq, 0))
                        
                       
                        x_arr = r / r_s
                        M_dm_grid = 2.0 * np.pi * rho_s * (r_s**3) * (0.5 * np.log(1.0 + x_arr**2) + np.log(1.0 + x_arr) - np.arctan(x_arr))
                        V_dm_sq = (G_grav * M_dm_grid) / r
                        V_dm = np.sqrt(np.maximum(V_dm_sq, 0))
                        
                       
                        V_sim = np.sqrt(np.maximum(V_bar_sq + V_dm_sq, 0))
                        
                     
                        rho_burkert_array = rho_s / ((1.0 + r / r_s) * (1.0 + (r / r_s)**2))

                        with plots_popup_container:
                            with ui.row().classes('w-full justify-center gap-8 flex-wrap'):
                                with ui.pyplot(figsize=(10, 6)):
                                    fig = plt.gcf()
                                    ax1 = plt.gca()
                                    
                                    ax1.errorbar(r, vobs, yerr=verr, fmt='o', color='blue', ecolor='gray', label='Observations')
                                    ax1.plot(r, V_bar, '-', color='red', linewidth=2, label='Baryonic')
                                    ax1.plot(r, V_dm, '-', color='purple', linewidth=2, label='Dark Matter')
                                    ax1.plot(r, V_sim, '-', color='green', linewidth=2.5, label='Total Simulated')
                                    
                                    ax1.set_xlabel('Radius (kpc)', fontsize=12)
                                    ax1.set_ylabel('Velocity (km/s)', fontsize=12)
                                    ax1.set_ylim(0, max(np.nanmax(vobs), np.nanmax(V_sim)) * 1.15)
                                    ax1.grid(True, linestyle='--', alpha=0.6)
                                    
                                    ax2 = ax1.twinx()
                                    ax2.plot(r, rho_burkert_array, '-', color='darkorange', linewidth=2.5, label='Burkert Density')
                                    ax2.set_ylabel(r'DM Density ($M_\odot / kpc^3$)', fontsize=12, color='darkorange')
                                    ax2.set_yscale('log')
                                    ax2.tick_params(axis='y', labelcolor='darkorange')
                                    
                                    plt.title(f"{gal_name_clean} Rotation Curve & DM Density", fontweight='bold', fontsize=14)
                                    
                                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=10)
                                    plt.tight_layout()
                    
                 
                    galaxy_select.on('update:model-value', lambda e: [update_plots_popup.refresh(), update_galaxy_mass_analysis.refresh()])

                    def check_and_run_galaxy():

                        v1 = (answer_vb.get('el') and answer_vb['el'].value or '').strip().lower()
                        v2 = (answer_vo.get('el') and answer_vo['el'].value or '').strip().lower()
                        r  = (answer_rad.get('el') and answer_rad['el'].value or '').strip().lower()
                        g  = (answer_g.get('el') and answer_g['el'].value or '').strip().lower()
                        m_tot = (answer_Mtot.get('el') and answer_Mtot['el'].value or '').strip().lower()
                        m_bar = (answer_Mbar.get('el') and answer_Mbar['el'].value or '').strip().lower()
                        v_gas = (answer_vgas.get('el') and answer_vgas['el'].value or '').strip().lower()
                        v_disk= (answer_vdisk.get('el') and answer_vdisk['el'].value or '').strip().lower()
                        v_bul = (answer_vbulge.get('el') and answer_vbulge['el'].value or '').strip().lower()
                        r2 = (answer_rad.get('el2') and answer_rad['el2'].value or '').strip().lower()
                        g2 = (answer_g.get('el2') and answer_g['el2'].value or '').strip().lower()

                        ok_r2 = r2 in ("rad", "radius", "r")
                        ok_g2 = g2 in ("g", "gravitational_constant")

                        ok_v1    = v1 in ('v_bar','vbar','v_baryonic','v baryonic','v-baryonic')
                        ok_v2    = v2 in ('v_obs','vobs','v_obs_ngc','v_observations')
                        ok_r     = r in ("rad","radius","r")
                        ok_g     = g in ("g","gravitational_constant","G")
                        ok_Mtot  = m_tot in ('mtot','m_tot','m_total','m total','m_total_ngc')
                        ok_Mbar  = m_bar in ('m_bar','mbar','m_baryonic','m baryonic','m_baryonic_ngc')
                        ok_vgas  = v_gas in ('vgas','v_gas','v gas','v_gas_ngc')
                        ok_vdisk = v_disk in ('vdisk','v_disk','v disk','v_disk_ngc')
                        ok_vbul  = v_bul in ('v_bulge','vbulge','vbul','v_bul','v bul','v_bul_ngc')


                        for ans in [answer_vb, answer_vo, answer_rad, answer_g,
            answer_Mtot, answer_Mbar, answer_vgas, answer_vdisk, answer_vbulge]:
                            for k in ('el', 'el2'):
                                el = ans.get(k)
                                if el:
                                    el.error = None
                                    el.classes(remove='border-red-500 border-green-500')

                        def mark(ans, ok, key='el'):
                            el = ans.get(key)
                            if el:
                                el.classes(add='border-green-500' if ok else 'border-red-500')

                        mark(answer_vb, ok_v1)
                        mark(answer_vo, ok_v2)
                        mark(answer_rad, ok_r, 'el')
                        mark(answer_g, ok_g, 'el')
                        mark(answer_Mtot, ok_Mtot)
                        mark(answer_Mbar, ok_Mbar)
                        mark(answer_vgas, ok_vgas)
                        mark(answer_vdisk, ok_vdisk)
                        mark(answer_vbulge, ok_vbul)
                        mark(answer_rad, ok_r2, 'el2')
                        mark(answer_g, ok_g2, 'el2')

                        if all([ok_v1, ok_v2, ok_r, ok_g, ok_Mtot, ok_Mbar, ok_vgas, ok_vdisk, ok_vbul, ok_r2, ok_g2]):
                            accessible_notify('All correct! Running analysis...', type_='success')
                            update_galaxy_mass_analysis()
                        else:
                            accessible_notify('Wrong, try again!', type_='error')


                    with ui.dialog() as plots_popup, ui.card().classes('p-4 w-full max-w-[1300px]').props('aria-label="Galaxy plots popup" role=dialog'):
                        ui.label("Galaxy Plots").classes("text-2xl font-bold mb-4").props('role=heading aria-level=2 tabindex=0')
                        popup_plot_container = ui.column().classes("w-full items-center")
                        aria_button("Close", "close the plots popup", on_click=lambda:plots_popup.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.row().classes('w-full justify-center gap-4 '):
                    
                        aria_button("Run Analysis", "Run the analysis to reproduce the plots",on_click=lambda:check_and_run_galaxy()).classes("!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded")
                     
                    
                        

#panel cluster
                with ui.tab_panel('cluster').props('role=tabpanel'):
                 
                    ui.label("Analyze the velocity distribution to unveil the reason of larger mass in galaxy clusters.").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                    plot_state = {'id': 0}  
                    cluster_state = {
                        "select": "Abell1656(Coma).csv", 
                        
"chi2_points": [],
"chi2_slider_result": "---",
                        "observed_vel": np.array([]),
                        "r_proj_kpc": np.array([]),
                        "m_bary_at_gal": np.array([]),
                        "N_obs": 0,
                        "v_mean_val": 0,
                        "sigma_obs": 0,
                        "M200": 0, "R200": 0, "c": 0, "rho_crit": 0,
                        "obs_mean": 0,
                        "bins": [], "counts_obs": [], "bin_centers": [],
                       
                        "x_min": 0, "x_max": 0, "y_max": 0, "padding": 0,
                        "r_min": 0, "r_max": 0, "r_padding": 0,
                        "ylim_min": 0, "ylim_max": 0,
                       
                        "M_bar_tot": 0, "R200_tot": 0,
                   
                        "sim_state": {
                            "indices": [], "rng": None,
                            "x_obs": [], "y_obs": [], "vx_obs": [], "vy_obs": [],
                            "x_model": [], "y_model": [], "vx_model": [], "vy_model": []
                        }
                    }
                    def compute_cluster_slider_physics(slider_val, slider_max=None):
                        """Calcola masse e dispersioni globali (senza raggio proiettato) dallo slider f in [0, 2]."""
                        f = float(slider_val)  # f=0: solo barioni; f=1: M200 esatta; f=2: 2x DM
                        M_bar_tot = cluster_state.get('M_bar_tot', 0.0)
                        M_dm_true = cluster_state.get('M_dm_true', 0.0)
                        R200 = cluster_state.get('R200', 1000.0)

                        M_dm_sim = f * M_dm_true
                        M_tot_sim = max(M_bar_tot + M_dm_sim, 1e-6)
                        dm_ratio_display = (M_dm_sim / M_tot_sim) if M_tot_sim > 0 else 0.0

                        sigma_bar = cluster_state.get('sigma_bar_global', 0.0)
                        sigma_sim = float(np.sqrt((G_grav * M_tot_sim) / (3.0 * R200))) if R200 > 0 else 0.0

                        return M_tot_sim, M_dm_sim, M_bar_tot, dm_ratio_display, sigma_bar, sigma_sim
                    with ui.dialog() as instruction_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Cluster simulation instructions" role=dialog'):
                        html_info_box(r"""
    <h3> Cluster Simulation</h3>
    <p>Explore the dynamics of the Coma Cluster and the evidence for Dark Matter using the Virial Theorem.</p>

   <h4>Legend</h4>
    <ul style="margin-bottom: 10px;">
        <li><b>Blue Histogram:</b> Real observed peculiar velocity distribution (centered at 0 km/s). It is wide, indicating high velocity dispersion (<span class="math">\(\sigma_{obs}\)</span>).</li>
        <li><b>Red Histogram:</b> Prediction based on <b>visible baryonic mass</b> only (stars + hot gas). It is narrow, implying galaxies should move slowly.</li>
        <li><b>Green Histogram:</b> Simulation that starts overlapping the red baryonic distribution and widens in real-time as you add Dark Matter.</li>
    </ul>
<hr class="my-4">

    <h3>Simulating the Cluster Halo</h3>
    <p>The <b>Green Histogram</b> represents the theoretical velocity distribution of galaxies within the cluster. It is constructed using a Gaussian distribution whose width (velocity dispersion, <span class="math">\(\sigma_{sim}\)</span>) depends on the total mass.</p>

    <h4>The Virial Connection</h4>
    <p>We assume the cluster is in virial equilibrium. According to the <b>Virial Theorem</b> applied to clusters, the velocity dispersion is related to the total mass <span class="math">\(M_{tot}\)</span> and the virial radius <span class="math">\(R_{200}\)</span> by:</p>
    
    <div style="text-align:center; margin: 15px 0;">
        <span class="math">$$ \sigma_{sim} = \sqrt{\frac{G M_{tot}}{3 R_{200}}} $$</span>
    </div>
<h4>How the Slider Works</h4>
    <p>When you move the slider, you inject a fraction <span class="math">\(f\)</span> of the virial Dark Matter mass <span class="math">\(M_{\mathrm{DM}} = M_{200} - M_{\mathrm{bar}}\)</span> into the total mass:</p>
    <div style="text-align:center; margin: 10px 0;">
        <span class="math">$$ M_{\mathrm{tot,sim}} = M_{\mathrm{bar}} + f \cdot M_{\mathrm{DM}} $$</span>
    </div>
    <p>At slider = 0 (<span class="math">\(f = 0\)</span>), <span class="math">\(M_{\mathrm{tot,sim}} = M_{\mathrm{bar}}\)</span> and the green distribution overlaps the narrow red baryonic curve. Increasing <span class="math">\(f\)</span> raises <span class="math">\(\sigma_{\mathrm{sim}}\)</span>, widening the green histogram until it matches the blue observed data.</p>
    
    <hr class="my-4">

    <h3>Activity Instructions</h3>
    <ol style="margin-top: 10px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b>Observe the Discrepancy:</b> Compare the Red and Blue histograms. Visible matter alone provides insufficient gravity to hold the fast-moving galaxies together; without extra mass, the cluster would fly apart.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Apply Virial Theorem:</b> Recall that for a stable cluster, Kinetic Energy and Potential Energy must balance (<span class="math">\(2K + U = 0\)</span>). High velocities (<span class="math">\(K\)</span>) require a deep potential well (<span class="math">\(U\)</span>), implying more mass.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Add Dark Matter:</b> Use the slider to inject Dark Matter into the simulation. This increases the total mass and broadens the velocity distribution.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Fit the Data:</b> Adjust the slider until the <b>Green simulated histogram</b> perfectly overlaps the <b>Blue observed histogram</b>.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Visual & Audio Check:</b> Watch the animation of galaxy movements and (if enabled) listen to the sonification: a high, constant pitch indicates a stable, bound cluster.
        </li>
    </ol>
    <hr class="my-4">
    
    <h4>References</h4>
    <div style="font-size: 0.9em; line-height: 1.4; color: #444;">
        <p style="margin-bottom: 5px;"><b>Distance Measures:</b> Hogg, D. W. (1999), "Distance measures in cosmology". (Used for comoving/angular distances).</p>
        <p style="margin-bottom: 5px;"><b>Coma Kinematics:</b> Lokas & Mamon (2003), "Dark matter distribution in the Coma cluster from galaxy kinematics".</p>
        <p style="margin-bottom: 5px;"><b>Virial Mass Estimators:</b> Mamon & Łokas (2005), "Dark matter in elliptical galaxies - II. Estimating the mass within the virial radius".</p>
        <p style="margin-bottom: 5px;"><b>Baryon Fractions:</b> Giodini, S. et al. (2009), "Stellar and Total Baryon Mass Fractions in Groups and Clusters Since Redshift 1", The Astrophysical Journal, 703:982-993. </p>
    </div>
""").props('aria-label="Descriptive text about galaxy cluster activity"')
                        aria_button("Close", "Close the box", on_click=lambda: instruction_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    combined_cluster_state = {'step': 0}

                    combined_cluster_html = [
                        
                        r"""
                        <h3>Phase 1: Cluster Panel – The Virial Theorem in Galaxy Clusters</h3>

                        <p><b>Goal:</b> Verify if Dark Matter is also present in galaxy clusters, using different mathematical tools.</p>

                       

                        <h4>1. Formulation of Predictions:</h4>

                        <ul>

                            <li>Considering a Cluster, formed by many galaxies rotating around the center of mass of the cluster, is Model 2 valid in this context as well?</li>

                            <li>Is the presence of dark matter an isolated case concerning individual galaxies or is it a general rule of the Universe? Justify your answer.</li>

                        </ul>

                        """,

                        r"""

                        <h3>Phase 2: Cluster Panel – The Virial Theorem in Galaxy Clusters</h3>

                        <p><b>Goal:</b> Verify if Dark Matter is also present in galaxy clusters, using different mathematical tools.</p>

                       

                        <h4>Derivation of the Virial Theorem:</h4>

                        <ul>

                            <li>Consider a satellite of mass <i>m</i> in a circular orbit of radius <i>r</i> around a mass <i>M</i> (planet). On your notebook or in the 'Reflection' section in the App's side menu, write the formulas for Kinetic Energy (<i>K</i>) and Gravitational Potential Energy (<i>U</i>).</li>

                            <li>Write Newton's equation of dynamics (<i>F = ma</i>) by inserting the relationship between Gravitational Force and circular motion (replace <i>F</i> with the gravitational force and '<i>m*a</i>' with the centripetal acceleration).</li>

                            <li>Multiply both sides of the force equation by <i>r</i>.</li>

                            <li>Use the result obtained to manipulate the formula for Kinetic Energy and compare it with the formula for Gravitational Potential to find the mathematical relationship that links them (Virial Theorem).</li>

                        </ul>


                        """,
                         r"""

                        <h3>Phase 3: Calculating Cluster Mass</h3>

                        <h4>Exercise:</h4>

                        <ul>

                            <li>Open the App, module 2 in the "Cluster Panel" section, and select a Galaxy Cluster (dataset) from the dropdown menu.</li>

                            <li>From the Virial formula just found, replace <i>K</i> with the kinetic energy formula and <i>U</i> with the potential energy formula, and perform the appropriate simplifications.</li>

                            <li>Derive the inverse formula to find the Total Mass.</li>

                            <li><b>Note:</b> The cluster is made up of many galaxies, each with a different velocity. Therefore, we have a velocity distribution, and usually, the velocity dispersion (standard deviation) is considered instead of the sum of the velocities.</li>

                        </ul>

                        """,

                      r"""
                        <h3>Phase 4: Calculating Cluster Mass</h3>

                        <h4>Exercise :</h4>

                        <ul>
                            <li>Use the data from the dataset to calculate the dynamical mass enclosed within the visible galaxies and the luminous mass (stars, dust, gas).</li>
                            
                            <li>Download the cluster dataset from the App (module 2 - Cluster panel - dataset). Analyze the data (velocity RV and radius of each galaxy in the cluster) by selecting a cluster dataset from the Excel spreadsheet.</li>

                            <li>To calculate the Dynamical Mass within the observed region using the formula <span class="math">\( M(r) = \frac{3 \cdot r \cdot \sigma^2}{G} \)</span>, first find the maximum visible radius of the cluster by applying the Excel formula <code>=MAX(H2:H108)</code>.</li>

                            <li>Calculate the standard deviation of the velocity (velocity dispersion) using the Excel formula <code>=STDEV.P(C2:C108)</code>.</li>

                            <li>To obtain the dynamical mass, multiply 3 times the maximum radius by the squared standard deviation and divide by the constant G <code>=(3*L2*K2^2)/I2</code>.</li>
                            
                            <li style="background-color: #e0f2fe; padding: 10px; border-radius: 5px; margin: 10px 0;"><i><b>Note for astrophysicists:</b> This Excel calculation gives you the mass stopping at the last visible galaxy. In cosmology, the true Total Mass of the cluster (<span class="math">\(M_{200}\)</span>) extends much further into the dark halo! The App will automatically calculate the full <span class="math">\(M_{200}\)</span> for you solving the mathematical equations.</i></li>

                            <li>Calculate the total stellar mass of the cluster by summing the values of each galaxy in the Stellar Mass column of the dataset <code>=SUM(G2:G108)</code>.</li>

                            <li>Calculate the gas mass of the cluster using the pre-defined formula in Excel column<code>=(0.7*L2) * 0.093 * (((0.7*L2) / 200000000000000) ^ 0.21)</code>, where <code>L2</code> is the mass calculated in the previous step.</li>

                            <li>Calculate the Luminous Mass (stars, gas) by summing the total stellar mass and the gas mass. </li>

                            <li>Compare the results of the luminous mass and dynamical mass. Calculate the ratio: Mass / Luminous Mass. What value do you get? Which of the two masses is larger and why?</li>

                           

                        </ul>

                        """,
                         r"""

                        <h3>Phase 5: Slider exercise</h3>

                        <h4>Exercise :</h4>

                        <ul>

                            <li>Look at the histogram generated by the App (velocity vs number of galaxies) and the scatter plot (velocity vs distance). Compare the plots representing the observed data (blue) with those of the predicted velocities using only the visible mass (red).</li>

                            <li>Looking at the red and blue histograms on the App, are the real galaxies in the cluster moving slower or faster than they should be, and why?</li>

                            <li>Use the slider to add mass; the simulated graph (green) will move. The goal is to find the amount of mass to add to make the simulated graph match the observations. Check the dark matter value (plot label) needed to obtain the match with the data.</li>

                            <li>In conclusion, is dark matter present in clusters? Why?</li>

                        </ul>

                        """,

                        r"""

                        <h3>Phase 6: Minimization exercise</h3>

                        <h4>Exercise :</h4>

                        <ul>

                        <li>This exercise use the chi2 minimization method to find the best fit of the simulated distribution (green) to the observed distribution (blue).  </li>

                       <li>The reduced &chi;&sup2; value compares the observed galaxy counts (O<sub>i</sub>, blue) and simulated galaxy counts (E<sub>i</sub>, green) across the velocity bins of the histogram.</li>

                        <li>Select a DM mass value using the slider and click 'Add point' (button below the central plot), repeat three times changing the value.</li>

                        <li>A parabola appears with the minimum value. Check the chi2 value obtained as minimum and find the best value to reduce the distance between observations (blue)and predictions(green).</li>

                            <li>Compare the dark matter values obtained in the context of galaxies and the cluster by analyzing the plot labels (Galaxy Panel and Cluster Panel). Verify in which context there is more dark matter and justify your answer.</li>

                        </ul>

                        """
                    ]

              
                    with ui.dialog() as instr_cluster_combined, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Cluster activity" role=dialog'):
                        
                        @ui.refreshable
                        def cluster_combined_content():
                            current_step = combined_cluster_state['step']
                            
                            html_info_box(combined_cluster_html[current_step])
                            
                            with ui.row().classes('w-full justify-between items-center mt-4'):
                             
                                if current_step > 0:
                                    aria_button("Previous", "Go to previous step", on_click=lambda: change_step_combined(-1)) \
                                        .classes("!bg-gray-500 hover:!bg-gray-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    ui.label() 
                                
                                ui.label(f"Step {current_step + 1} of {len(combined_cluster_html)}").classes('text-gray-500 font-bold')
                                
                               
                                if current_step < len(combined_cluster_html) - 1:
                                    aria_button("Next", "Go to next step", on_click=lambda: change_step_combined(1)) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                else:
                                    aria_button("Close", "Close the activity", on_click=lambda: instr_cluster_combined.close()) \
                                        .classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                        def change_step_combined(delta):
                            new_step = combined_cluster_state['step'] + delta
                            if 0 <= new_step < len(combined_cluster_html):
                                combined_cluster_state['step'] = new_step
                                cluster_combined_content.refresh()

                        instr_cluster_combined.on('open', lambda: combined_cluster_state.update({'step': 0}) or cluster_combined_content.refresh())
                        cluster_combined_content()
                    with ui.dialog() as dataset_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto ').props('aria-label="Dataset info" role=dialog'):
                
                
                        info_box("**Coma variables (SDSS):** objid, ra, dec, modelmag_r, modelmagerr_r, extinction_r, redshift (z), zErr.  \n**Abell SARS variables:** Cluster, ID, RAdeg, DEdeg, RV (radial velocity in km/s), e_RV, q_RV, bmag (B-band magnitude).")
                        reference_box("""**Dataset references**: [Kaggle: Coma cluster](https://www.kaggle.com/datasets/mertalkan98/coma-cluster); [SDSS](https://www.sdss.org/science/data-release-publications); Way M.J. et al. (2005), *Redshifts in the Southern Abell Redshift Survey Clusters*.""").classes('text-base italic')
                        template_url_clu ="https://docs.google.com/spreadsheets/d/1LTZNHpg2BlkKZyMrsV_KxcaTGasa7EhD/copy"
                        with ui.row().classes('w-full justify-center gap-4 mt-4'):
        
                            aria_button('Build Google Sheet', 'Create copy Google Sheet', 
                on_click=lambda: ui.run_javascript(f'window.open("{template_url_clu}", "_blank")')) \
        .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
        .props('aria-label="Create personal copy of Google Sheets"')

    
        
        
                           
                            aria_button('Download Dataset', 'download', 
                                        on_click=lambda: ui.run_javascript('window.location.href = "/dataset/Cluster_dataset.xlsx"')) \
                                .classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded') \
                                .props('aria-label="Download the Excel dataset"')

                           
                            aria_button('Upload Exercises', 'cloud_upload', on_click=upload_zone_dialog.open) \
        .classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded')
                          
                    
                        aria_button("Close","Close the box", on_click=lambda:dataset_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                   
                            
                    with ui.dialog() as cur_clust, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="Information about a galaxy without dark matter" role=dialog'):
                        html_info_box(r"""
                        <h3>A Galaxy Without Dark Matter?</h3>
    <p>Astronomers using the Hubble Space Telescope have confirmed that the galaxy <b>NGC 1052-DF2</b> is almost completely missing its Dark Matter.</p>
    <p>This "ultra-diffuse" galaxy is as wide as the Milky Way but contains only 1/200th the number of stars and at most 1/400th the expected amount of Dark Matter. This discovery challenges theories of galaxy formation but paradoxically <b>proves Dark Matter is real</b>.</p>
    <p><i>Why?</i> If Dark Matter were just an illusion caused by modified gravity, it should appear everywhere. Finding a galaxy <i>without</i> it proves that Dark Matter is a distinct substance that can be separated from normal matter!</p>""")
                        reference_box("**Source:** [NASA](https://science.nasa.gov/missions/hubble/mystery-of-galaxys-missing-dark-matter-deepens/)")
                        aria_button("Close", "close", on_click=lambda:cur_clust.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")
                 
                        

                 
                    
                    
                        
                    
              
                    def open_cluster_analysis_dialog2():
                      
                        with ui.dialog() as comp_cluster_dialog2, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="cluster-title"'):
                            
                            with ui.column().classes('w-full h-full bg-white'):
                              
                                with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0'):
                                    ui.label('Cluster Analysis: Physics & Simulation').classes('text-xl font-bold').props('id="cluster-title" role=heading aria-level=2 tabindex=0')
                                    aria_button('Close', 'close', on_click=comp_cluster_dialog2.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                               
                                with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                    t_comp = ui.tab('Computational Notes')
                                    t_chi2 = ui.tab('χ² Minimization')
                                    t_sim = ui.tab('Velocity Simulation')
                                    
                                    tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))

                              
                                with ui.tab_panels(tabs, value=t_comp).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                    
                                 
                                    with ui.tab_panel(t_comp):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>
                                        <h3>Computational Notes</h3>
<ul>
    <li><b>Step 1 — Radial velocities:</b> Direct \( v_i \) from SARS (Way et al., 2005) or \( v_i = c \cdot z_i \) from Coma SDSS redshifts.</li>
    <li><b>Step 2 — Membership filtering (Beers et al., 1990):</b> Retain galaxies within \( \pm 6000\,\mathrm{km/s} \) of the velocity peak, then apply iterative \( 3\sigma_{\mathrm{MAD}} \) clipping (\( \sigma_{\mathrm{MAD}} = 1.4826 \cdot \mathrm{MAD} \)) to remove foreground/background interlopers, leaving \( N \) cluster members.</li>
    <li><b>Step 3 — Peculiar velocities &amp; dispersion:</b> Center velocities in the cluster rest frame:
        <span class="math">\( v_{\mathrm{pec},i} = v_i - \bar{v}, \quad \sigma_{\mathrm{obs}} = \sqrt{\frac{1}{N}\sum_{i=1}^N v_{\mathrm{pec},i}^2} \)</span></li>
    <li><b>Step 4 — Virial mass \( M_{200} \) and radius \( R_{200} \):</b> Using \( \rho_{\mathrm{crit}} = \frac{3 H_0^2}{8 \pi G} \) (\( H_0 = 73\,\mathrm{km\,s^{-1}\,Mpc^{-1}} \)), solve simultaneously:
        <span class="math">\( \sigma_{\mathrm{obs}}^2 = \frac{G M_{200}}{3 R_{200}}, \quad M_{200} = \frac{4}{3}\pi (200\rho_{\mathrm{crit}}) R_{200}^3 \)</span></li>
    <li><b>Step 5 — Luminosity distance &amp; stellar mass:</b> Using the linear Hubble law \( D_L \approx c z_{\mathrm{cluster}} / H_0 \) and distance modulus \( \mu = 5\log_{10}(D_L / 10\,\mathrm{pc}) \):
        <ul style="margin-top:4px; list-style-type:circle;">
            <li><b>Coma (r-band):</b> \( M_r = m_r - A_r - \mu \), \( M_{\star,\mathrm{tot}} = \Upsilon_r \sum_i 10^{0.4(4.67 - M_{r,i})} \) with \( \Upsilon_r = 2.0 \) (Bell et al., 2003).</li>
            <li><b>SARS Abell (B-band):</b> \( M_B = m_B - \mu \), \( M_{\star,\mathrm{tot}} = \Upsilon_B \sum_i 10^{0.4(5.48 - M_{B,i})} \) with \( \Upsilon_B = 5.0 \).</li>
        </ul>
    </li>
    <li><b>Step 6 — Gas mass (Giodini et al., 2009):</b> With \( M_{500} \approx 0.7 M_{200} \):
        <span class="math">\( M_{\mathrm{gas}} = (0.7 M_{200}) \cdot 0.093 \left(\frac{0.7 M_{200}}{2 \times 10^{14}\,M_\odot}\right)^{0.21} \)</span></li>
    <li><b>Step 7 — Total baryonic &amp; Dark Matter mass:</b>
        <span class="math">\( M_{\mathrm{bar}} = M_{\star,\mathrm{tot}} + M_{\mathrm{gas}}, \quad M_{\mathrm{DM}} = M_{200} - M_{\mathrm{bar}} \)</span></li>
    <li><b>Step 8 — Slider-controlled mass &amp; global 1D velocity dispersions:</b>
        <span class="math">\( M_{\mathrm{tot,sim}} = M_{\mathrm{bar}} + f \cdot M_{\mathrm{DM}} \)</span>
        <ul style="margin-top:4px; list-style-type:circle;">
            <li><b>Baryonic (red):</b> \( \sigma_{\mathrm{bar}} = \sqrt{\frac{G M_{\mathrm{bar}}}{3 R_{200}}} \)</li>
            <li><b>Simulated (green):</b> \( \sigma_{\mathrm{sim}} = \sqrt{\frac{G M_{\mathrm{tot,sim}}}{3 R_{200}}} \)</li>
        </ul>
    </li>
    <li><b>Step 9 — Histograms &amp; phase-space scatter plot:</b> Synthetic peculiar velocities are drawn from \( \mathcal{N}(0, \sigma_{\mathrm{bar}}) \) and \( \mathcal{N}(0, \sigma_{\mathrm{sim}}) \) using 100&times; Monte Carlo oversampling (each point weighted by 1/100) over 39 symmetric bins. The projected radius \( r_{\mathrm{proj},i} = \theta_i D_A \) from the BCG is computed solely to plot the <i>Peculiar Velocity vs. Projected Radius</i> scatter plot and the 3D cluster view.</li>
</ul>
    """).props('role=dialog aria-modal=true aria-label="Computational notes"')

                                   
                                    with ui.tab_panel(t_chi2):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>
                                         <h3>Chi-Square (χ²) Minimization for Histograms</h3>
<p>The agreement between observed and simulated peculiar velocity distributions is quantified using <b>Pearson's reduced binned Chi-Square</b> (Pearson, 1900; Taylor, 1997).</p>
<ul>
    <li><b>Step 1 — Symmetric binning:</b> Peculiar velocities \( v_{\mathrm{pec},i} = v_i - \bar{v} \) are grouped into <b>39 equally spaced bins</b> (40 symmetric edges) centered at \( 0\,\mathrm{km/s} \), giving observed galaxy counts \( O_i \).</li>
    <li><b>Step 2 — 100&times; Monte Carlo oversampling:</b> To obtain smooth expected counts \( E_i \) without random fluctuations from a single \( N \)-galaxy draw (Barlow &amp; Beeston, 1993), we generate \( 100 N \) velocities from \( \mathcal{N}(0, \sigma_{\mathrm{sim}}) \) and weight each point by \( 1/100 \) so that \( \sum E_i = N \).</li>
    <li><b>Step 3 — Reduced Chi-Square formula:</b> Evaluated only over the \( B \) valid bins with \( E_i \ge 0.5 \):<br>
        <span class="math">\( \chi^2_{\nu} = \frac{1}{B - 3} \sum_{i:\, E_i \ge 0.5} \frac{(O_i - E_i)^2}{E_i} \)</span><br>
        Here \( k = 3 \) constraints are subtracted for the mean velocity (centered at 0), the global velocity dispersion \( \sigma_{\mathrm{sim}} \), and the total sample size \( N \) (Taylor, 1997, Sec. 12.3, p. 269).</li>
    <li><b>Why divide by \( E_i \)?</b> Bin counts follow Poisson statistics where the variance equals the expected count (\( \sigma_{\mathrm{counts}}^2 = E_i \)).</li>
</ul>

                        """).props('role=dialog aria-modal=true aria-label="Chi-Square explanation"')

                                
                                    with ui.tab_panel(t_sim).classes('flex flex-col items-center justify-start'):
                                        ui.label("Velocity Simulation").classes("text-2xl font-bold mb-4 text-slate-800")
                                   
                                        aria_button('Galaxies velocities', 'show cluster galaxies velocity',on_click=on_galaxies_click, icon='visibility').classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                    
                        comp_cluster_dialog2.open()
                    @ui.refreshable
                    def update_cluster_image():
                        
                    
                        select = cluster_state["select"]

                        if select.lower() == "abell1656(coma).csv":
                            img_path = os.path.join(CLUSTER_IMG_PATH, "coma_img.jpg")
                        else:
                            img_path = os.path.join(CLUSTER_IMG_PATH, os.path.splitext(select)[0] + ".jpg")
                        if os.path.exists(img_path):
                            filename = os.path.basename(img_path)
                            web = f"/cluster_img/{filename}"
                            
                            ui.html(f"""
    <div style="text-align:center; width:100%">
        
        <img src="{web}" style="width:100%; max-width:450px; border-radius:12px;">

        <p><em>
            Image references:<br>
            - <a href='https://esahubble.org/' target='_blank'>ESA Hubble</a><br>
            - Bonnarel, F. et al. (2000). <i>The ALADIN interactive sky atlas</i>, Astronomy & Astrophysics Supplement Series<br>
            - Digitized Sky Survey 2 (DSS2), via CDS HiPS  
            (<a href='https://alasky.cds.unistra.fr' target='_blank'>Aladin HiPS</a>)
        </em></p>

    </div>
    """)

                        else:
                            ui.label("Image not found").classes('text-red-500 text-center')

            
            
                    
                    
                    @ui.refreshable
                    def update_cluster_table():
                        
                    
                        select = cluster_state["select"]
                        ui.html("<h5>Cluster Information</h5>")
                        if select.lower() == "abell1656(coma).csv":
                            table_path = os.path.join(CLUSTER_TABLES_PATH, "coma_table.csv")
                        else:
                            table_path = os.path.join(CLUSTER_TABLES_PATH, os.path.splitext(select)[0] + ".csv")
                        if os.path.exists(table_path):
                            df = pd.read_csv(table_path, header=None,names=["Property", "Value"])
                            
                            ui.table.from_pandas(df).classes("w-full max-w-xl")
                            #html_table = df.to_html(index=False, classes="nice-table", border=0)

                            ui.html(f"""
    <div style="text-align:center; width:100%">

        <p><em>
            Cluster data references:<br>
            - <a href='https://en.wikipedia.org/wiki/List_of_galaxy_groups_and_clusters' target='_blank'>Wikipedia: List of galaxy clusters</a><br>
            - Wenger, M., Ochsenbein, F. et al. (2000). <i>The SIMBAD astronomical database</i>. Astronomy & Astrophysics Supplement Series. 
            <a href='https://simbad.u-strasbg.fr' target='_blank'>SIMBAD</a><br>
            - Helou, G., Madore, B. F. (1988). <i>The NASA/IPAC Extragalactic Database (NED)</i>. 
            <a href='https://ned.ipac.caltech.edu' target='_blank'>NED</a><br>
            - Abell, G. O. et al. (1989). <i>A catalog of rich clusters of galaxies</i>. Astrophysical Journal Supplement Series<br>
            - VizieR Online Data Catalog: VII/110A. 
            <a href='https://vizier.u-strasbg.fr' target='_blank'>VizieR</a>
        </em></p>

    </div>
    """)

                        else:
                            ui.label("Table not found").classes('text-red-500 text-center')
                    # --- 1. DIALOG UNITO PER IMMAGINE E TABELLA (Cluster) ---
                    with ui.dialog() as table_dialog_cluster, ui.card().style('width: 1200px; max-width: 95vw;').classes('p-8 border border-gray-200 shadow-md rounded-xl bg-white').props('aria-label="Cluster Information and Image" role=dialog'):
                        with ui.element('div').style('display: flex; flex-direction: row; width: 100%; gap: 2rem; align-items: flex-start; justify-content: space-between;'):
                            # Colonna Sinistra (Immagine)
                            with ui.element('div').style('flex: 0 0 450px; display: flex; flex-direction: column; align-items: center;'):
                                ui.html("<h4 style='margin-top: 0; font-weight: bold; color: #1e3a8a;'>Cluster Image</h4>")
                                update_cluster_image()
                            # Colonna Destra (Tabella)
                            with ui.element('div').style('flex: 1 1 auto; width: 100%;'):
                                update_cluster_table()
                        with ui.row().classes('w-full justify-end mt-6'):
                            aria_button("Close", 'close', on_click=lambda: table_dialog_cluster.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-6 rounded shadow-md")

                    # --- 2. DIALOG 3D INTERATTIVO (Apre al click dell'immagine nel panel) ---
                    with ui.dialog() as cluster_3d_info_dialog, ui.card().classes('p-6 w-full max-w-[800px] bg-white'):
                        ui.label("Interactive 3D Tools Info").classes("text-2xl font-bold text-slate-800 mb-4")
                        html_info_box(r"""
                        <p><b>Visualizing the Cluster's Components</b></p>
                        <ul>
                            <li><b>Stellar Mass (Galaxies):</b> The visible galaxies that make up the cluster. Despite their brightness, they contribute only a tiny fraction (1-2%) to the total mass.</li>
                            <li><b>Intracluster Gas (X-ray):</b> A massive cloud of extremely hot gas filling the space between galaxies. It represents the bulk of the "luminous" (baryonic) matter in the cluster (~13%).</li>
                            <li><b>Dark Matter Halo:</b> The invisible scaffold holding everything together, representing roughly ~85% of the cluster's mass!</li>
                        </ul>
                        <hr style="margin: 10px 0;">
                        <p><b>Interactive 3D Tools:</b></p>
                        <ul>
                            <li><i>Add dark matter moving the slider.</i></li>
                            <li><i>Rotate the 3D plot with your mouse.</i></li>
                            <li><b><i>Click on the items in the Legend to turn the Components on and off!</i></b></li>
                        </ul>
                        """)
                        aria_button("Close", "close", on_click=cluster_3d_info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-2")

                    with ui.dialog() as cluster_3d_dialog, ui.card().style('width: 1200px; max-width: 95vw; height: 90vh;').classes('p-6 overflow-hidden flex flex-col bg-white').props('aria-label="3D Cluster Model" role=dialog'):
                        
                        with ui.row().classes('w-full justify-between items-center mb-1 shrink-0'):
                            ui.label("Cluster structure and mass distribution").classes("text-2xl font-bold text-slate-800")
                            with ui.row().classes('gap-4'):
                                aria_button("Info", 'instructions', on_click=lambda: [cluster_3d_info_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-6 rounded shadow-md")
                                aria_button("Close", 'close', on_click=lambda: [cluster_3d_dialog.close(), refresh_cluster_plots(full_anim=False)]).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-6 rounded shadow-md")

                        dialog_status_label_cluster = ui.label("").classes("text-xl font-bold text-center w-full my-1")
                        
                        with ui.row().classes('w-full items-center justify-center gap-4 mt-1 mb-4 shrink-0 bg-slate-50 p-4 rounded-xl shadow-inner border border-slate-200'):
                            ui.label("Add Dark Matter:").classes("font-bold text-lg text-slate-800")
                            dialog_cluster_dm_slider = ui.slider(min=0.0, max=2.0, value=0.0, step=0.01).classes('w-1/2').props('color=purple label-always')

                        def evaluate_cluster_dm_status_dialog(e=None):
                            try:
                                if len(cluster_state.get('observed_vel', [])) == 0:
                                    return
                                f_slider = float(dialog_cluster_dm_slider.value)
                                S_max = float(getattr(dialog_cluster_dm_slider, 'max', 60.0))
                                cluster_name = cluster_state.get('select', 'Cluster').removesuffix('.csv').removesuffix('.txt')

                                _, _, _, current_fraction, _, _ = compute_cluster_slider_physics(f_slider, slider_max=S_max)
                                target_fraction = cluster_state.get('true_dm_ratio', 0.0)

                                if current_fraction < 0.05:
                                    msg = "Dark matter missing! Move the slider to add mass."
                                    color_class = "text-red-600"
                                elif current_fraction < target_fraction - 0.04:
                                    msg = "More Dark Matter needed!"
                                    color_class = "text-orange-500"
                                elif current_fraction > target_fraction + 0.04:
                                    msg = "Too much Dark Matter!"
                                    color_class = "text-orange-500"
                                else:
                                    msg = f"Perfect! The Dark Matter matches {cluster_name}'s virial mass."
                                    color_class = "text-green-600"

                                dialog_status_label_cluster.set_text(msg)
                                dialog_status_label_cluster.classes(replace=f"text-xl font-bold text-center w-full my-1 {color_class}")
                            except Exception as ex:
                                print(f"Error in cluster evaluate: {ex}")

                        plot_3d_cluster_container = ui.column().classes('w-full flex-1 items-center justify-center overflow-hidden')

                        @ui.refreshable
                        def render_3d_cluster_plot():
                            plot_3d_cluster_container.clear()
                            with plot_3d_cluster_container:
                                title_label = ui.label("").classes("text-xl font-bold text-slate-800 mb-2 mt-8 text-center w-full z-10")
                                fig = go.Figure()

                                custom_colorbar = dict(
                                    title=dict(text="Velocity<br>Dispersion<br>(km/s)", font=dict(color='black', size=12), side='top'),
                                    tickfont=dict(color='black', size=11), x=-0.12, thickness=15
                                )
                                r_max_plot = 2000.0

                                if cluster_state.get('r_proj_kpc') is not None and len(cluster_state.get('r_proj_kpc', [])) > 0:
                                    r_kpc = cluster_state['r_proj_kpc']
                                    current_cluster = cluster_state.get('select', '').removesuffix('.csv').removesuffix('.txt')
                                    r_max_plot = cluster_state.get('R200', np.nanmax(r_kpc)) * 1.05

                                    f_slider = float(dialog_cluster_dm_slider.value)
                                    S_max = float(getattr(dialog_cluster_dm_slider, 'max', 60.0))

                                    current_M_tot, current_M_DM, _, dm_ratio_display, _, _ = compute_cluster_slider_physics(
                                        f_slider, slider_max=S_max
                                    )
                                    m_stars_tot = cluster_state.get('M_stars_tot', float(np.sum(cluster_state['m_bary_at_gal'])))
                                    m_gas_global_fixed = cluster_state.get('M_gas_tot', 0.0)
                                    M_dm_true = cluster_state.get('M_dm_true', 0.0)
                                    progress = (current_M_DM / M_dm_true) if M_dm_true > 0 else 0.0
                                    c_nfw = cluster_state.get('c', 4.0)

                                    dialog_cluster_dm_slider.props(f'label-value="DM: {dm_ratio_display * 100:.1f}%"')
                                    title_label.set_text(f"3D Mass Distribution: {current_cluster}")

                                    n_gas = 1500 if m_gas_global_fixed > 0 else 0
                                    n_dm = int(4000 * progress) if current_M_DM > 0 else 0

                                    r_array = np.linspace(0.1, cluster_state.get('R200', 2000), 500)

                                    if m_gas_global_fixed > 0:
                                        M_gas_enclosed = Mc_nfw_enclosed(r_array, m_gas_global_fixed, max(c_nfw * 0.8, 0.1), cluster_state['rho_crit'])
                                    else:
                                        M_gas_enclosed = np.zeros_like(r_array)

                                    rho_s = cluster_state.get('rho_s', 1.0)
                                    r_s = cluster_state.get('r_s', 100.0)
                                    M_dm_base = M_burkert_enclosed(r_array, rho_s, r_s)
                                    if np.nanmax(M_dm_base) > 0 and current_M_DM > 0:
                                        M_dm_enclosed = (M_dm_base / np.nanmax(M_dm_base)) * current_M_DM
                                    else:
                                        M_dm_enclosed = np.zeros_like(r_array)

                                    if current_M_tot > 0:
                                        sigma_array = calculate_sigma_los_nfw(r_array, current_M_tot, max(c_nfw, 0.1), cluster_state.get('z_cluster', 0.02), 73.0)
                                    else:
                                        sigma_array = np.zeros_like(r_array)

                                    c_min_plot = np.nanmin(sigma_array) if len(sigma_array) > 0 else 0
                                    c_max_plot = np.nanmax(sigma_array) * 1.15 if (len(sigma_array) > 0 and np.nanmax(sigma_array) > 0) else 1000

                                    def sample_cluster_component(m_array_enclosed, n_pts):
                                        if n_pts <= 0 or np.nanmax(m_array_enclosed) <= 0:
                                            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
                                        CDF = m_array_enclosed / np.nanmax(m_array_enclosed)
                                        CDF_clean = np.maximum.accumulate(np.nan_to_num(CDF))
                                        U = np.random.uniform(0, 1, n_pts)
                                        r_pts = np.interp(U, CDF_clean, r_array)
                                        phi = np.arccos(np.random.uniform(-1, 1, n_pts))
                                        theta = np.random.uniform(0, 2 * np.pi, n_pts)
                                        x = r_pts * np.sin(phi) * np.cos(theta)
                                        y = r_pts * np.sin(phi) * np.sin(theta)
                                        z = r_pts * np.cos(phi)
                                        vel_pts = np.interp(r_pts, r_array, sigma_array)
                                        mass_enc = np.interp(r_pts, r_array, m_array_enclosed)
                                        return x, y, z, vel_pts, np.stack((r_pts, mass_enc), axis=-1)

                                    traces = []
                                    if len(cluster_state.get('x_kpc', [])) > 0:
                                        x_gal = cluster_state['x_kpc']
                                        y_gal = cluster_state['y_kpc']
                                        r_gal = cluster_state['r_proj_kpc']
                                        std_r = np.std(r_gal) if len(r_gal) > 1 else 10.0
                                        z_gal = np.random.normal(0, std_r, len(x_gal))
                                        v_gal_peculiar = np.abs(cluster_state['observed_vel_pec'])
                                        c_data_gal = np.stack((r_gal, np.full_like(r_gal, m_stars_tot)), axis=-1)
                                        traces.append(go.Scatter3d(
                                            x=x_gal, y=y_gal, z=z_gal, mode='markers',
                                            marker=dict(size=3, symbol='square', color=v_gal_peculiar, colorscale='turbo', cmin=c_min_plot, cmax=c_max_plot, opacity=0.9),
                                            name=f'■ Galaxies (M: {m_stars_tot:.1e} M_☉)', customdata=c_data_gal,
                                            hovertemplate="<b>Galaxies (Real Data)</b><br>Radius: %{customdata[0]:.0f} kpc<br>Dispersion: %{marker.color:.0f} km/s<br>Total Stellar Mass: %{customdata[1]:.1e} M_☉"
                                        ))

                                    if n_gas > 0:
                                        x, y, z, v, c_data = sample_cluster_component(M_gas_enclosed, n_gas)
                                        traces.append(go.Scatter3d(
                                            x=x, y=y, z=z, mode='markers',
                                            marker=dict(size=3, symbol='cross', color=v, colorscale='turbo', cmin=c_min_plot, cmax=c_max_plot, opacity=0.4),
                                            name=f'✖ Hot Gas (M: {m_gas_global_fixed:.1e} M_☉)', customdata=c_data,
                                            hovertemplate="<b>Intracluster Gas</b><br>Radius: %{customdata[0]:.0f} kpc<br>Dispersion: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"
                                        ))

                                    if n_dm > 0:
                                        x, y, z, v, c_data = sample_cluster_component(M_dm_enclosed, n_dm)
                                        traces.append(go.Scatter3d(
                                            x=x, y=y, z=z, mode='markers',
                                            marker=dict(size=2, symbol='circle', color=v, colorscale='turbo', cmin=c_min_plot, cmax=c_max_plot, opacity=0.15),
                                            name=f'● DM Halo (M: {current_M_DM:.1e} M_☉)', customdata=c_data,
                                            hovertemplate="<b>Dark Matter Halo</b><br>Radius: %{customdata[0]:.0f} kpc<br>Dispersion: %{marker.color:.0f} km/s<br>Enclosed Mass: %{customdata[1]:.1e} M_☉"
                                        ))

                                    for trace in traces:
                                        fig.add_trace(trace)

                                    fig.add_trace(go.Scatter3d(
                                        x=[None], y=[None], z=[None], mode='markers',
                                        marker=dict(colorscale='turbo', cmin=c_min_plot, cmax=c_max_plot, showscale=True, colorbar=custom_colorbar),
                                        showlegend=False, hoverinfo='skip'
                                    ))
                                else:
                                    title_label.set_text("Waiting for valid Cluster Data...")
                                    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='rgba(0,0,0,0)')))

                                fig.update_layout(
                                    scene=dict(xaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), yaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), zaxis=dict(visible=False, range=[-r_max_plot, r_max_plot]), bgcolor='rgb(15, 23, 42)', aspectmode='cube'),
                                    margin=dict(l=0, r=0, b=0, t=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    legend=dict(font=dict(color='white'), yanchor="top", y=0.95, xanchor="left", x=0.01),
                                    autosize=False, width=1050, height=520
                                )

                                global my_cluster_3d_plot
                                my_cluster_3d_plot = ui.plotly(fig).style('width: 1050px; height: 520px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);')
                        render_3d_cluster_plot()
                        dialog_cluster_dm_slider.on('change', lambda e: [evaluate_cluster_dm_status_dialog(), render_3d_cluster_plot.refresh()])
                    
                    chart_obs = None
                    chart_model = None
                    sim_timer = ui.timer(0.05, lambda: None, active=False)
                    def reset_simulation_data():
                        observed_vel = cluster_state.get('observed_vel', [])
                        if len(observed_vel) == 0:
                            return

                        N_gal = min(80, len(observed_vel))
                        cluster_radius = 1.0
                        rng = np.random.default_rng()

                        x_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        x_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)

                        sigma_obs_val = cluster_state['sigma_obs']
                        scale_factor = 0.0005
                        vx_obs = rng.normal(0.0, sigma_obs_val, N_gal) * scale_factor
                        vy_obs = rng.normal(0.0, sigma_obs_val, N_gal) * scale_factor

                        dir_x_model = rng.normal(0.0, 1.0, N_gal)
                        dir_y_model = rng.normal(0.0, 1.0, N_gal)

                        cluster_state['sim_state'] = {
                            "x_obs": x_init_obs, "y_obs": y_init_obs,
                            "x_model": x_init_mod, "y_model": y_init_mod,
                            "vx_obs": vx_obs, "vy_obs": vy_obs,
                            "dir_x_model": dir_x_model, "dir_y_model": dir_y_model,
                            "rng": rng,
                            "current_dataset": cluster_state.get("select", "")
                        }

                    async def update_cluster_points():
                        try:
                            sim_state = cluster_state.get('sim_state')

                            if not sim_state or sim_state.get("current_dataset") != cluster_state.get("select"):
                                reset_simulation_data()
                                sim_state = cluster_state.get('sim_state')
                                if not sim_state:
                                    return

                            cluster_radius = 1.0
                            dt = 0.05
                            VELOCITY_SCALE = 0.0005
                            L = 2 * cluster_radius

                            # Fisica globale coerente con lo slider (senza raggio proiettato)
                            _, _, _, dm_ratio_display, _, sigma_model_current = compute_cluster_slider_physics(float(dm_slider.value))

                            vx_model = sim_state["dir_x_model"] * sigma_model_current * VELOCITY_SCALE
                            vy_model = sim_state["dir_y_model"] * sigma_model_current * VELOCITY_SCALE

                            sim_state["x_obs"] = (sim_state["x_obs"] + sim_state["vx_obs"] * dt + cluster_radius) % L - cluster_radius
                            sim_state["y_obs"] = (sim_state["y_obs"] + sim_state["vy_obs"] * dt + cluster_radius) % L - cluster_radius
                            sim_state["x_model"] = (sim_state["x_model"] + vx_model * dt + cluster_radius) % L - cluster_radius
                            sim_state["y_model"] = (sim_state["y_model"] + vy_model * dt + cluster_radius) % L - cluster_radius

                            chart_obs.options['series'][0]['data'] = np.stack((sim_state["x_obs"], sim_state["y_obs"]), axis=1).tolist()
                            chart_model.options['series'][0]['data'] = np.stack((sim_state["x_model"], sim_state["y_model"]), axis=1).tolist()
                            chart_obs.update()
                            chart_model.update()

                            label_stats_obs.set_text(f"σ_obs: {cluster_state['sigma_obs']:.0f} km/s")
                            label_stats_model.set_text(f"Simulated σ: {sigma_model_current:.0f} km/s | DM: {dm_ratio_display*100:.1f}%")

                        except Exception as e:
                            print(f"Update Error: {e}")

                
                    sim_timer.callback = update_cluster_points

                    async def start_simulation_logic():
                        
                        reset_simulation_data()
                        
                        
                        await asyncio.sleep(0.5)
                        
                        
                        if chart_obs: chart_obs.run_chart_method(':resize')
                        if chart_model: chart_model.run_chart_method(':resize')
                        
                        
                        sim_timer.activate()
                    with ui.dialog() as sim_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto').props('aria-label="Cluster galaxies velocity simulation" role=dialog aria-modal=true'):
                        ui.label('Cluster Galaxies Velocities View').classes('text-xl font-bold').props('role=heading aria-level=2 tabindex=0 aria-label=Galaxy Simulation View')
                        
                        with ui.row().classes('w-full justify-center gap-4 p-4'):
                            
                            with ui.column().classes('flex-1 items-center'):
                                chart_obs = ui.echart({
                                    'title': {'text': 'Observed Galaxies', 'left': 'center', 'textStyle': {'fontWeight': 'bold'}},
                                    'grid': {'left': '5%', 'right': '5%', 'top': '10%', 'bottom': '10%'},
                                    'xAxis': {'min': -1.0, 'max': 1.0, 'show': False},
                                    'yAxis': {'min': -1.0, 'max': 1.0, 'show': False},
                                    'series': [{'type': 'scatter', 'symbolSize': 8, 'itemStyle': {'color': 'blue', 'opacity': 0.8}, 'data': [], 'animation': False}]
                                }).classes('w-full h-64')
                                label_stats_obs = ui.label('').classes('text-sm font-bold text-blue-700')

                            
                            with ui.column().classes('flex-1 items-center'):
                                chart_model = ui.echart({
                                    'title': {'text': 'Simulated Galaxies', 'left': 'center', 'textStyle': {'fontWeight': 'bold'}},
                                    'grid': {'left': '5%', 'right': '5%', 'top': '10%', 'bottom': '10%'},
                                    'xAxis': {'min': -1.0, 'max': 1.0, 'show': False},
                                    'yAxis': {'min': -1.0, 'max': 1.0, 'show': False},
                                    'series': [{'type': 'scatter', 'symbolSize': 8, 'itemStyle': {'color': 'green', 'opacity': 0.8}, 'data': [], 'animation': False}]
                                }).classes('w-full h-64')
                                label_stats_model = ui.label('').classes('text-sm font-bold text-green-700')
                        
                        aria_button("close",'close',on_click=lambda:[sim_dialog.close(), sim_timer.deactivate()]).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                    async def on_galaxies_click():
                        sim_dialog.open()          
                        await start_simulation_logic() 
                        
                        
                    with ui.row().classes('w-full justify-center gap-8'):
                        DEFAULT_CLUSTER = "Abell1656(Coma).csv" 
                        #DEFAULT_CLUSTER = "Abell0080.txt"
                        cluster_name = None


                        dataset_files = [
                            f for f in os.listdir(CLUSTER_DATA_PATH)
                            if f.lower().endswith(('.txt', '.dat', '.csv'))
                        ]
                        if DEFAULT_CLUSTER not in dataset_files:
                            dataset_files.insert(0, DEFAULT_CLUSTER)
                            
                            
                    
                        dataset_selector = ui.select(
                            options=dataset_files,
                            value=DEFAULT_CLUSTER,
                            label='Select a Cluster Dataset'
                        
                        ).classes('w-1/2 max-w-md').props('id=cluster_selector aria-label="Cluster dataset selector" role=listbox tabindex=0')
                                
                                
                    
                        select=dataset_selector.value
                        

                    
                        cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)
                        aria_button("Instructions", "Read the instructions",on_click=safe_click(lambda: [instruction_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")  
                            
                        aria_button(
    "Scientific Info", 
    "Open the full cluster analysis including notes, Chi2, and simulations", 
    on_click=lambda: [open_cluster_analysis_dialog2(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")]
    ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                        

                        
                        
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_clust.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded") 
                         
                    
                            
                            
                    

                

                    def _finalize_cluster_state(observed_vel):
                        """Popola tutte le chiavi di cluster_state in modo allineato a generate_grids_over_5.py,
                        preservando tutte le chiavi richieste da istogramma, scatter, chi2, animazione 2D e vista 3D."""
                        cluster_state['observed_vel'] = observed_vel
                        cluster_state['N_obs'] = len(observed_vel)
                        cluster_state['obs_mean'] = float(np.mean(observed_vel))
                        cluster_state['v_mean_val'] = cluster_state['obs_mean']
                        cluster_state['sigma_obs'] = float(np.std(observed_vel))

                        # Velocità peculiari centrate in 0 (sistema di riferimento del cluster)
                        v_peculiar = observed_vel - cluster_state['obs_mean']
                        cluster_state['observed_vel_pec'] = v_peculiar

                        # Massa viriale M200, R200 (Teorema del Viriale, H0 = 73)
                        rho_crit = rho_crit_Msunkpc3(H0=73.0)
                        M200, R200 = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], rho_crit)
                        cluster_state['rho_crit'] = rho_crit
                        cluster_state['M200'] = M200
                        cluster_state['R200'] = R200

                        # Parametri ausiliari mantenuti solo per la vista 3D del cluster
                        cluster_state['c'] = concentration_duffy2008(M200, cluster_state['z_cluster'])
                        cluster_state['rho_s'], cluster_state['r_s'], _ = rho_s_from_M200_and_c(
                            M200, cluster_state['c'], rho_crit
                        )

                        # Bilancio di massa globale (Stelle + Gas Giodini 2009 + DM per sottrazione)
                        m_stars_tot = float(np.sum(cluster_state['m_bary_at_gal']))
                        f_gas_global = 0.093 * (((0.7 * M200) / 2e14)**0.21)
                        m_gas_tot = float((0.7 * M200) * f_gas_global)
                        M_bar_tot = m_stars_tot + m_gas_tot
                        M_dm_true = max(0.0, M200 - M_bar_tot)

                        cluster_state['M_stars_tot'] = m_stars_tot
                        cluster_state['M_gas_tot'] = m_gas_tot
                        cluster_state['M_bar_tot'] = M_bar_tot
                        cluster_state['M_dm_true'] = M_dm_true
                        cluster_state['true_dm_ratio'] = (M_dm_true / M200) if M200 > 0 else 0.0

                        # Dispersione barionica globale (larghezza iniziale della curva rossa e della verde a slider=0)
                        cluster_state['sigma_bar_global'] = float(np.sqrt((G_grav * M_bar_tot) / (3.0 * R200)))

                        # Binning simmetrico centrato in 0 (39 bin = 40 estremi, come generate_grids_over_5.py)
                        max_abs = max(float(np.abs(v_peculiar).max()), 3.5 * cluster_state['sigma_obs'])
                        x_lim = max_abs * 1.10
                        bins = np.linspace(-x_lim, x_lim, 40)
                        counts_obs, _ = np.histogram(v_peculiar, bins=bins)

                        cluster_state['x_lim'] = x_lim
                        cluster_state['x_min'] = -x_lim
                        cluster_state['x_max'] = x_lim
                        cluster_state['padding'] = 0.0
                        cluster_state['bins'] = bins
                        cluster_state['bin_centers'] = 0.5 * (bins[1:] + bins[:-1])
                        cluster_state['counts_obs'] = counts_obs

                        # Calcolo y_max includendo il picco stretto della curva rossa barionica
                        rng_tmp = np.random.default_rng(seed=42)
                        v_bar_tmp = rng_tmp.normal(0.0, cluster_state['sigma_bar_global'], size=len(v_peculiar) * 100)
                        bar_counts_tmp, _ = np.histogram(v_bar_tmp, bins=bins, weights=np.full(len(v_bar_tmp), 1.0 / 100.0))
                        cluster_state['y_max'] = float(max(counts_obs.max(), bar_counts_tmp.max()) * 1.3)

                        # Limiti assi per lo scatter plot (r_proj_kpc sull'asse X, v_peculiar sull'asse Y)
                        cluster_state['r_min'] = float(cluster_state['r_proj_kpc'].min())
                        cluster_state['r_max'] = float(cluster_state['r_proj_kpc'].max())
                        cluster_state['r_padding'] = 0.1 * (cluster_state['r_max'] - cluster_state['r_min'])
                        cluster_state['v_min_global'] = -x_lim
                        cluster_state['ylim_min'] = -x_lim
                        cluster_state['ylim_max'] = x_lim

                    def load_coma_dataset(filename):
                        data_path = os.path.join(DATA_DIR, "Abell1656(Coma).csv")
                        df_coma = pd.read_csv(data_path, skiprows=1)
                        df_coma.columns = ['objid', 'ra', 'dec', 'modelmag_r', 'modelmagerr_r', 'extinction_r', 'redshift', 'zErr']
                        df_coma = df_coma.dropna(subset=['ra', 'dec', 'redshift', 'modelmag_r']).copy()
                        df_coma['RV'] = c_light * df_coma['redshift']

                        # Filtro a 2 stadi (+-6000 km/s + MAD 3-sigma clipping)
                        members = filter_cluster_members(df_coma, vel_col="RV")
                        if len(members) < 5:
                            members = df_coma.copy()

                        observed_vel = members['RV'].values
                        cluster_state['z_cluster'] = float(np.nanmedian(members['redshift'].values))

                        # Raggio proiettato e coordinate (solo per scatter plot Velocity vs Radius e vista 3D)
                        idx_bcg = members['modelmag_r'].idxmin()
                        cluster_state['center_ra'] = float(members.loc[idx_bcg, 'ra'])
                        cluster_state['center_dec'] = float(members.loc[idx_bcg, 'dec'])

                        cluster_distance_Mpc = float(np.median(observed_vel)) / 73.0
                        D_A_kpc = cluster_distance_Mpc * 1000.0
                        theta_rad = angsep_rad(
                            members['ra'].values, members['dec'].values,
                            cluster_state['center_ra'], cluster_state['center_dec']
                        )
                        cluster_state['r_proj_kpc'] = np.maximum(theta_rad * D_A_kpc, 1e-3)
                        cluster_state['x_kpc'] = np.radians(members['ra'].values - cluster_state['center_ra']) * np.cos(np.radians(cluster_state['center_dec'])) * D_A_kpc
                        cluster_state['y_kpc'] = np.radians(members['dec'].values - cluster_state['center_dec']) * D_A_kpc

                        # Massa stellare in banda r (M_sun,r = 4.67, Upsilon_r = 2.0)
                        cluster_state['m_bary_at_gal'] = stellar_mass_from_r_mag(
                            members['modelmag_r'].values,
                            members['extinction_r'].values,
                            cluster_state['z_cluster'],
                            M_to_L_ratio=2.0
                        )

                        _finalize_cluster_state(observed_vel)
                        return df_coma

                    def initialize_cluster_from_df(df):
                        if df is None or df.empty: return
                        df_clean = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"]).copy()
                        df_clean = df_clean[df_clean["bmag"] > 0.0]

                        # Filtro a 2 stadi (+-6000 km/s + MAD 3-sigma clipping)
                        members = filter_cluster_members(df_clean, vel_col="RV")
                        if len(members) < 5:
                            members = df_clean.copy()

                        observed_vel = members["RV"].values
                        cluster_state['z_cluster'] = float(np.nanmedian(observed_vel / c_light))

                        # Raggio proiettato e coordinate (solo per scatter plot Velocity vs Radius e vista 3D)
                        idx_bcg = members["bmag"].idxmin()
                        cluster_state['center_ra'] = float(members.loc[idx_bcg, "RAdeg"])
                        cluster_state['center_dec'] = float(members.loc[idx_bcg, "DEdeg"])

                        cluster_distance_Mpc = float(np.median(observed_vel)) / 73.0
                        D_A_kpc = cluster_distance_Mpc * 1000.0
                        theta_rad = angsep_rad(
                            members["RAdeg"].values, members["DEdeg"].values,
                            cluster_state['center_ra'], cluster_state['center_dec']
                        )
                        cluster_state['r_proj_kpc'] = np.maximum(theta_rad * D_A_kpc, 1e-3)
                        cluster_state['x_kpc'] = np.radians(members["RAdeg"].values - cluster_state['center_ra']) * np.cos(np.radians(cluster_state['center_dec'])) * D_A_kpc
                        cluster_state['y_kpc'] = np.radians(members["DEdeg"].values - cluster_state['center_dec']) * D_A_kpc

                        # Massa stellare in banda B per SARS Abell (M_sun,B = 5.48, Upsilon_B = 5.0)
                        cluster_state['m_bary_at_gal'] = stellar_mass_from_B_mag(
                            members["bmag"].values,
                            cluster_distance_Mpc,
                            M_to_L_ratio=5.0
                        )

                        _finalize_cluster_state(observed_vel)

                    def recompute_axis_limits():
                        # I limiti simmetrici centrati in 0 sono già calcolati in _finalize_cluster_state()
                        pass
                    def plot_cluster_chi2_curve():
                        cluster_chi2_plot_container.clear()
                        plt.close()
                        with cluster_chi2_plot_container:
                            with ui.pyplot(figsize=(6, 4), close=False, clear=True):
                                ax = plt.gca()
                                chi2_points = cluster_state['chi2_points']
                                all_xs = []
                                all_ys = []
                                
                               
                                x_max_chi_fallback = 1.0
                                
                                if chi2_points:
                                    xs, ys = zip(*chi2_points)
                                    all_xs.extend(xs)
                                    all_ys.extend(ys)
                                   
                                    ax.scatter(xs, ys, s=40, c="blue", alpha=0.6, zorder=3, label="Points")

                                    if len(chi2_points) == 3:
                                        coeffs = np.polyfit(xs, ys, 2)
                                        
                                        x_lo, x_hi = min(xs), max(xs)
                                        span = x_hi - x_lo
                                        if span == 0: span = x_hi * 0.1
                                        
                                        x_fit = np.linspace(max(0, x_lo - 0.5*span), x_hi + 0.5*span, 400)
                                        y_fit = np.polyval(coeffs, x_fit)
                                        
                                        all_xs.extend(x_fit)
                                        all_ys.extend(y_fit)
                                        
                                        if coeffs[0] > 1e-5: 
                                            xmin = -coeffs[1] / (2 * coeffs[0])
                                            ymin = np.polyval(coeffs, xmin)

                                            ax.plot(x_fit, y_fit, "g-", lw=2, zorder=2, label="Parabolic Fit")
                                            
                                           
                                            if -0.5 <= xmin <= 2.0:
                                                ax.scatter([xmin], [ymin], c='red', s=140, marker='*', zorder=4, label="Min")
                                            
                                            all_xs.append(xmin)
                                            all_ys.append(ymin)
                                        else:
                                            ax.plot(x_fit, y_fit, "r--", lw=2, label="Invalid Fit ")
                                
                               
                                if all_xs:
                                    plot_x_max = max(all_xs) * 1.1
                                    ax.set_xlim(0, plot_x_max if plot_x_max > 0 else x_max_chi_fallback)
                                    
                               
                                    valid_ys = [y for y in all_ys if np.isfinite(y)]
                                    if valid_ys:
                                        plot_y_max = max(max(valid_ys) * 1.1, 1.0)
                                        
                                        plot_y_min = min(0.0, min(valid_ys) * 1.1) 
                                        ax.set_ylim(plot_y_min, plot_y_max)
                                else:
                                  
                                    ax.set_xlim(0, x_max_chi_fallback)
                                    ax.set_ylim(0, 10.0)
                                    
                                ax.set_xlabel(r" $M_{DM}/M_{tot}$")
                                ax.set_ylabel(r"χ²/dof", fontsize=10)
                                ax.grid(True, alpha=0.3)
                                
                                handles, labels = ax.get_legend_handles_labels()
                                if handles: 
                                    ax.legend(handles, labels, fontsize=8)
                                    
                                text_to_show = cluster_state.get('chi2_slider_result', "---")
                                plt.text(0.5, 0.92, text_to_show, 
                                        transform=ax.transAxes, 
                                        fontsize=9, 
                                        color='green',
                                        fontweight='bold',
                                        horizontalalignment='center',
                                        verticalalignment='top',     
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray'))
                                
                                plt.title("χ² Minimization", fontsize=12, fontweight='bold')
                                plt.tight_layout()
                    
                   
                    def add_cluster_chi2_point():
                        f = float(dm_slider.value)
                        v_peculiar = cluster_state['observed_vel_pec']
                        bins = cluster_state['bins']
                        counts_obs = cluster_state['counts_obs']

                        # 1. Usa la fisica globale dallo slider (nessun raggio proiettato)
                        _, _, _, dm_ratio_display, _, sigma_sim = compute_cluster_slider_physics(f)

                        # 2. Oversampling Monte Carlo 100x centrato in 0 (identico a generate_grids_over_5.py)
                        oversample = 100
                        rng = np.random.default_rng(seed=42)
                        v_sim_tot = rng.normal(loc=0.0, scale=sigma_sim, size=len(v_peculiar) * oversample)
                        pesi_sim = np.ones_like(v_sim_tot) / oversample
                        counts_model, _ = np.histogram(v_sim_tot, bins=bins, weights=pesi_sim)

                        # 3. Chi-quadrato ridotto su bin validi (E_i >= 0.5) con k = 3 vincoli (N, media, sigma)
                        valid_bins = counts_model >= 0.5
                        if np.sum(valid_bins) > 3:
                            chi2_raw = np.sum(((counts_obs[valid_bins] - counts_model[valid_bins])**2) / counts_model[valid_bins])
                            dof = max(1, np.sum(valid_bins) - 3)
                            chi2_val = float(chi2_raw / dof)
                        else:
                            chi2_val = 0.0

                        # 4. Gestione punti e fit parabolico (logica dell'app invariata)
                        is_duplicate = any(abs(x - dm_ratio_display) < 1e-4 for x, y in cluster_state['chi2_points'])
                        if is_duplicate:
                            cluster_state['chi2_slider_result'] = "Point already added! Move the slider."
                            plot_cluster_chi2_curve()
                            return

                        cluster_state['chi2_points'].append((dm_ratio_display, chi2_val))
                        if len(cluster_state['chi2_points']) > 3:
                            cluster_state['chi2_points'].pop(0)

                        if len(cluster_state['chi2_points']) == 3:
                            xs, ys = zip(*cluster_state['chi2_points'])
                            coeffs = np.polyfit(xs, ys, 2)

                            if coeffs[0] > 1e-5:
                                xmin = -coeffs[1] / (2 * coeffs[0])
                                ymin = np.polyval(coeffs, xmin)

                                if xmin < 0:
                                    cluster_state['chi2_slider_result'] = "Invalid Result: Ratio cannot be negative!"
                                elif xmin >= 1.0:
                                    cluster_state['chi2_slider_result'] = "Invalid Result: Ratio cannot exceed 100%!"
                                elif ymin < 0:
                                    cluster_state['chi2_slider_result'] = f"Invalid Fit: χ² min < 0 (≈ {ymin:.2f})"
                                else:
                                    cluster_state['chi2_slider_result'] = f"Best Fit: DM/Tot ≈ {xmin*100:.1f}%, χ²_min ≈ {ymin:.2f}"
                            else:
                                cluster_state['chi2_slider_result'] = "Invalid shape. Select 3 points that surround the minimum!"
                        else:
                            cluster_state['chi2_slider_result'] = "Add more points to find the minimum."

                        plot_cluster_chi2_curve()
                    
                        
                    def refresh_cluster_chi2_plot():
                        cluster_state['chi2_points'].clear()
                        cluster_state['chi2_slider_result'] = "---"
                        plot_cluster_chi2_curve()

                   
                  

                    with ui.column().classes('w-full items-center '):
                        dm_slider_min, dm_slider_max = 0.0, 2.0
                        ui.label("Move the slider to add dark matter to the simulated velocity distribution in the cluster").props('id=dm_slider_label')
                        dm_slider = aria_slider(min=dm_slider_min, max=dm_slider_max,
                        value=0.0, step=0.01,  aria_label="Dark matter fraction control slider").props('aria-describedby=dm_slider_label label-always')
                        dm_slider.bind_value(dialog_cluster_dm_slider, 'value')
                        
                        
                        
                        with ui.row().classes('w-full items-stretch justify-between no-wrap gap-2 px-2'):
                    
                      
                            with ui.column().classes('w-[300px] flex-shrink-0 p-2 bg-blue-50 rounded-lg border border-blue-200 shadow-sm flex flex-col justify-between overflow-hidden'):
                            
                                with ui.column().classes('w-full gap-1 p-0 m-0'):
                                    cluster_title_label = ui.label("Cluster Info: ---").classes("text-xl font-bold text-blue-800 mb-2 whitespace-nowrap")
                                    
                                    async def open_cluster_3d():
                                        evaluate_cluster_dm_status_dialog() 
                                        render_3d_cluster_plot.refresh()
                                        await asyncio.sleep(0.1)
                                        cluster_3d_dialog.open()
                                        ui.run_javascript("setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 200);")
                                        
                                    cluster_img_display = ui.image().classes('w-full h-28 rounded cursor-pointer hover:scale-105 transition-transform').props('role=button tabindex=0 aria-label="Open 3D Cluster Model"').on('click', open_cluster_3d).on('keydown.enter', open_cluster_3d)
                                    cluster_info_content = ui.column().classes('w-full text-sm text-gray-700 gap-0 p-0 m-0')
                                
                                with ui.row().classes('w-full justify-center mt-2 shrink-0'):
                                  
                                    aria_button("Image & Table", "table", on_click=table_dialog_cluster.open).classes("!bg-blue-500 text-white font-bold py-1 px-4 rounded shadow-md")
                            with ui.column().classes('w-full justify-between items-center'):
                                plot_container_histo = ui.column().classes('w-full')
                                with ui.row().classes('w-full justify-center mt-2'):
                                    aria_button(
        "Activity: Cluster ", 
        "Start the cluster analysis activity", 
        on_click=lambda: [instr_cluster_combined.open(), ui.run_javascript("MathJax.typesetPromise()")]
    ).classes("!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-6 rounded shadow-md")
                                    aria_button("Dataset", "Read the info about dataset",on_click=safe_click(lambda: [dataset_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded") 
                                    
                                  
    
                                    
                                    
                                    
                            with ui.column().classes('w-full justify-between items-center'):
                                plot_container_scatter = ui.column().classes('w-full')
                                with ui.row().classes('w-full justify-center mt-2'):
                                    ui.html('''
        <style>
            .sigma-audio-menu-wrapper {
                position: relative;
                display: inline-block;
                z-index: 50 !important; /* Abbassato per stare SOTTO i dialoghi */
            }
            .sigma-audio-dropdown-content {
                display: none;
                position: absolute;
                left: 50%;
                transform: translateX(-50%);
                bottom: 100%;
                margin-bottom: 0.5rem;
                width: 480px;
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
                flex-direction: column;
                gap: 1rem;
                z-index: 50 !important; /* Abbassato */
            }
            .sigma-audio-dropdown-content::after {
                content: "";
                position: absolute;
                top: 100%;  
                left: 0;
                width: 100%;
                height: 1rem; 
                background: transparent;
            }
            .sigma-audio-menu-wrapper:hover .sigma-audio-dropdown-content {
                display: flex !important;
            }
        </style>

        <div class="sigma-audio-menu-wrapper">
            <button style="background-color: #16a34a; transition: background-color 0.2s; display: flex; align-items: center; gap: 8px; color: white; font-weight: bold; padding: 0.5rem 1.5rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);" 
                onmouseover="this.style.backgroundColor='#15803d'" 
                onmouseout="this.style.backgroundColor='#16a34a'">
                🔊 Audio Tools
                <svg style="width: 16px; height: 16px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            
            <div class="sigma-audio-dropdown-content">
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #475569; padding-bottom: 0.75rem;">
                    <span style="color: white; font-weight: bold; font-size: 1.125rem;">Sonification Controls</span>
                    
                    <button id="audio-button-cluster" role="button" aria-label="activate or deactivate audio" aria-pressed="false" onclick="initAudio()" 
                        style="background-color: #22c55e; color: black; font-weight: bold; padding: 0.4rem 0.75rem; border-radius: 0.25rem; border: none; cursor: pointer;">
                        Activate Audio
                    </button>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <button role="button" aria-label="Play observed dispersion velocity mean" onclick="playObservedSigmaMean()" style="background-color: #2563eb; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Observed σ (mean)</button>
                    <button role="button" aria-label="Play observed dispersion velocity" onclick="playObservedSigmaCurve()" style="background-color: #60a5fa; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Observed σ (curve)</button>
                    
                    <button role="button" aria-label="Play baryonic dispersion velocity mean" onclick="playBaryonicSigmaMean()" style="background-color: #dc2626; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Baryonic σ (mean)</button>
                    <button role="button" aria-label="Play baryonic dispersion velocity" onclick="playBaryonicSigmaCurve()" style="background-color: #f87171; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Baryonic σ (curve)</button>
                    
                    <button role="button" aria-label="Play simulated dispersion velocity mean" onclick="playSimSigmaMean()" style="background-color: #16a34a; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Simulated σ (mean)</button>
                    <button role="button" aria-label="Play simulated dispersion velocity" onclick="playSimSigmaCurve()" style="background-color: #4ade80; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Simulated σ (curve)</button>
                    
                    <button role="button" aria-label="Play differences dispersion velocities mean" onclick="playDifferenceSigmaMean()" style="background-color: #9333ea; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Difference σ (mean)</button>
                    <button role="button" aria-label="Play differences dispersion velocities" onclick="playDifferenceSigmaCurve()" style="background-color: #c084fc; color: white; padding: 0.5rem; border-radius: 0.25rem; text-align: left; font-size: 0.875rem; border: none; cursor: pointer;">▶ Difference σ (curve)</button>
                </div>
            </div>
        </div>
        ''').classes('inline-block')

                            ui.label("").props('id=audio_status_cluster role=status aria-live=polite aria-atomic=true tabindex=0').classes('hidden')
                                
                                  
                            with ui.column().classes('w-full justify-between items-center'):
                                cluster_chi2_plot_container = ui.column().classes('w-full')
                                
                          
                                plot_cluster_chi2_curve() 
                                
                                with ui.row().classes("w-full justify-center mt-2"):
                                    
                                    aria_button("Add point", "Add point", on_click=lambda: add_cluster_chi2_point()).classes("!bg-green-600 text-white font-bold py-1 px-2 rounded ")
                                    aria_button("Reset", "Reset", on_click=lambda: refresh_cluster_chi2_plot()).classes("!bg-green-600 text-white font-bold py-1 px-2 rounded ")
                                    
                    def refresh_cluster_plots(full_anim=False):
                           
                        
                        plot_state['id'] += 1             
    
                        if full_anim:
                            plot_container_histo.clear()      
                            plot_container_scatter.clear()
                        
                        #asyncio.create_task(update_coma_histogram_v2(plot_state['id'], full_anim))
                        ui.timer(0, lambda: update_coma_histogram_v2(plot_state['id'], full_anim), once=True)
                    
                
                        
                    if DEFAULT_CLUSTER.lower() == "abell1656(coma).csv":
                        df = load_coma_dataset(DEFAULT_CLUSTER)
                        cluster_name = DEFAULT_CLUSTER
                        #select = dataset_selector.value
                
                    else:
                        df = get_cluster_data_cached(DEFAULT_CLUSTER)
                        initialize_cluster_from_df(df)
                    recompute_axis_limits()
                    
                    dm_slider.value = 0.0

                   
                    
                    
                    ui.timer(0.5, lambda: refresh_cluster_plots(full_anim=True), once=True)

                
                    def on_dataset_change(new_value):
                        if not isinstance(new_value, str):
                            return

                        cluster_state["select"] = new_value
                        refresh_cluster_chi2_plot()

                        if new_value.lower() == "abell1656(coma).csv":
                            load_coma_dataset(new_value)
                        else:
                            df = get_cluster_data_cached(new_value)
                            initialize_cluster_from_df(df)

                        dm_slider.value = 0.0
                        dialog_cluster_dm_slider.value = 0.0
                        dm_slider.update()
                        dialog_cluster_dm_slider.update()

                        cluster_state['chi2_points'].clear()
                        cluster_state['chi2_slider_result'] = "Add points to find the minimum."
                        cluster_state["sim_state"] = {}
                        refresh_cluster_plots(full_anim=True)
                        update_cluster_image.refresh()
                        update_cluster_table.refresh()

                    #dm_slider.on_value_change(lambda: refresh_cluster_plots(full_anim=False))
                    dm_slider.on('change', lambda e: refresh_cluster_plots(full_anim=False))
                    #dm_slider.on('update:model-value', refresh_cluster_plots.refresh)
                    #dataset_selector.on('update:model-value',  on_dataset_change)
                    dataset_selector.on_value_change(lambda e: on_dataset_change(e.value))
                



                
                            
                    async def update_coma_histogram_v2(token=None, full_anim=True):
                        if token is None:
                            token = plot_state['id']
                        if token != plot_state['id']:
                            return

                        v_peculiar = cluster_state['observed_vel_pec']
                        r_proj_kpc = cluster_state['r_proj_kpc']
                        if len(v_peculiar) == 0:
                            return

                        bins = cluster_state['bins']
                        x_lim = cluster_state['x_lim']
                        y_max = cluster_state['y_max']
                        r_max = cluster_state['r_max']
                        r_padding = cluster_state['r_padding']
                        sigma_obs = cluster_state['sigma_obs']

                        try:
                            f = float(dm_slider.value)

                            # 1. Fisica globale dallo slider (nessuna dipendenza da r_proj_kpc)
                            displayed_M_tot, displayed_M_DM, M_bar_tot_fixed, dm_ratio_display, sigma_bar, sigma_sim = (
                                compute_cluster_slider_physics(f)
                            )

                            # 2. Istogramma: oversampling Monte Carlo 100x centrato in 0
                            # Usiamo lo stesso z_hist così a slider=0 il verde parte identico e sovrapposto al rosso
                            oversample = 100
                            rng_hist = np.random.default_rng(seed=42)
                            z_hist = rng_hist.normal(loc=0.0, scale=1.0, size=len(v_peculiar) * oversample)
                            v_bar_hist = z_hist * sigma_bar
                            v_dm_hist = z_hist * sigma_sim
                            pesi_over = np.ones_like(z_hist) / oversample

                            # 3. Scatter plot: esattamente N punti abbinati a r_proj_kpc, centrati in 0
                            # Con lo stesso z_scat i punti verdi partono sopra i rossi e si espandono con lo slider
                            rng_scat = np.random.default_rng(seed=42)
                            z_scat = rng_scat.normal(loc=0.0, scale=1.0, size=len(v_peculiar))
                            v_bar_scat = z_scat * sigma_bar
                            v_dm_scat = z_scat * sigma_sim

                            # 4. Sonificazione (aggiornamento variabili JS per Audio Tools)
                            obs_sub = v_peculiar[::max(1, len(v_peculiar) // 20)]
                            sigma_obs_series = [float(abs(v)) for v in obs_sub if np.isfinite(v)]
                            sigma_bar_series = [float(sigma_bar)] * len(sigma_obs_series)
                            sigma_sim_list = [float(sigma_sim)] * len(sigma_obs_series)
                            sigma_diff_series = np.abs(np.array(sigma_obs_series) - np.array(sigma_sim_list))
                            sigma_diff_mean = float(np.mean(sigma_diff_series)) if len(sigma_diff_series) > 0 else 0.0

                            ui.run_javascript(f"""
                                window.sigmaObsSeries = {sigma_obs_series};
                                window.sigmaBarSeries = {sigma_bar_series};
                                window.sigmaSimSeries = {sigma_sim_list};
                                window.sigmaObsVal = {float(sigma_obs)};
                                window.sigmaBarVal = {float(sigma_bar)};
                                window.sigmaSimVal = {float(sigma_sim)};
                                window.sigmaDiffSeries = {sigma_diff_series.tolist()};
                                window.sigmaDiffMean = {sigma_diff_mean};
                            """)

                            # 5. Aggiornamento etichetta slider, pannello informativo e immagine
                            dm_slider.props(f'label-value="DM / Mₜₒₜ: {dm_ratio_display * 100:.1f}%"')

                            select_name = cluster_state.get("select", "Abell1656(Coma).csv")
                            clean_name = select_name.removesuffix('.csv').removesuffix('.txt').removesuffix('.dat')
                            cluster_title_label.set_text(f"Cluster Info: {clean_name}")

                            cluster_info_content.clear()
                            with cluster_info_content:
                                html_info_box(f"""
                                    <ul class="list-none pl-2 space-y-1 text-sm text-gray-800">
                                        <li><b>M<sub>tot</sub>:</b> {format_sci(displayed_M_tot)} M<sub>☉</sub></li>
                                        <li><b>M<sub>DM</sub>:</b> {format_sci(displayed_M_DM)} M<sub>☉</sub> <span style="color: #0284c7; font-weight: bold;">({dm_ratio_display*100:.1f}%)</span></li>
                                        <li><b>M<sub>bar</sub>:</b> {format_sci(M_bar_tot_fixed)} M<sub>☉</sub></li>
                                        <li><b>σ<sub>obs</sub>:</b> {sigma_obs:.1f} km/s</li>
                                        <li><b>σ<sub>sim</sub>:</b> {sigma_sim:.1f} km/s</li>
                                    </ul>
                                """)

                            img_name = (
                                "coma_img.jpg"
                                if select_name.lower() == "abell1656(coma).csv"
                                else os.path.splitext(select_name)[0] + ".jpg"
                            )
                            if os.path.exists(os.path.join(CLUSTER_IMG_PATH, img_name)):
                                cluster_img_display.source = f"/cluster_img/{img_name}"
                            else:
                                cluster_img_display.source = ""

                            # Ordinamento radiale SOLO per lo scatter plot e l'animazione progressiva
                            sort_idx = np.argsort(r_proj_kpc)
                            r_sorted = r_proj_kpc[sort_idx]
                            obs_vel_sorted = v_peculiar[sort_idx]
                            v_bar_scat_sorted = v_bar_scat[sort_idx]
                            v_dm_scat_sorted = v_dm_scat[sort_idx]

                            n_chunks = 10
                            chunk_size = int(np.ceil(len(r_sorted) / n_chunks))
                            sleep_time = 0.15

                            def render_frame(
                                current_obs, current_bar_scat, current_dm_scat,
                                curr_r_obs, curr_r_bar, curr_r_dm,
                                curr_bar_hist, curr_bar_w,
                                curr_dm_hist, curr_dm_w
                            ):
                                try:
                                    with plot_container_histo:
                                        plot_container_histo.clear()
                                        plt.close('all')
                                        with ui.pyplot(figsize=(6, 4)):
                                            ax = plt.gca()
                                            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                                            plt.hist(
                                                current_obs, bins=bins, alpha=1.0, color='blue',
                                                orientation='horizontal', label='Observed Velocities',
                                                rasterized=True
                                            )
                                            if len(curr_dm_hist) > 0:
                                                plt.hist(
                                                    curr_dm_hist, bins=bins, weights=curr_dm_w,
                                                    alpha=0.7, color='green', orientation='horizontal',
                                                    label='Simulated Velocities', rasterized=True
                                                )
                                            if len(curr_bar_hist) > 0:
                                                plt.hist(
                                                    curr_bar_hist, bins=bins, weights=curr_bar_w,
                                                    alpha=0.6, color='red', orientation='horizontal',
                                                    label='Baryonic Component', rasterized=True
                                                )

                                            plt.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                                            plt.xlim(0, float(y_max))
                                            plt.ylim(-float(x_lim), float(x_lim))
                                            plt.ylabel('Peculiar Velocity [km/s]', fontsize=10)
                                            plt.xlabel('N° of Galaxies', fontsize=10)
                                            plt.title('Histogram of velocities distribution', fontsize=12, fontweight='bold')
                                            plt.legend(fontsize=10, loc='upper right')
                                            plt.tight_layout()
                                            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                                            ui.element('div').props(
                                                'role=status aria-live=polite tabindex=0 aria-label=Histogram showing the observed peculiar velocities of galaxies in the cluster in blue, the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
                                            )

                                    with plot_container_scatter:
                                        plot_container_scatter.clear()
                                        with ui.pyplot(figsize=(6, 4)):
                                            ax = plt.gca()
                                            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                                            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

                                            plt.scatter(curr_r_obs, current_obs, s=10, color='blue', alpha=0.6, label='Observed Galaxies', rasterized=True)
                                            plt.scatter(curr_r_bar, current_bar_scat, s=10, color='red', alpha=0.6, label='Baryonic Component', rasterized=True)
                                            plt.scatter(curr_r_dm, current_dm_scat, s=10, color='green', alpha=0.6, label='Simulated Galaxies', rasterized=True)

                                            plt.axhline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                                            plt.xlim(0, r_max + r_padding)
                                            plt.ylim(-float(x_lim), float(x_lim))
                                            plt.xlabel('Projected Radius [kpc]', fontsize=10)
                                            plt.ylabel('Peculiar Velocity [km/s]', fontsize=10)
                                            plt.title('Cluster galaxies velocities', fontsize=12, fontweight='bold')
                                            plt.legend(fontsize=10, loc='upper right')
                                            plt.grid(True, linestyle='--', alpha=0.7)
                                            plt.tight_layout()
                                            ui.element('div').props(
                                                'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the observed peculiar velocities of galaxies in the cluster in blue, the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
                                            )
                                except Exception as ex:
                                    print(f"Render error in update_coma_histogram_v2: {ex}")
                                    return

                            if full_anim:
                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']:
                                        return
                                    limit = i * chunk_size
                                    render_frame(
                                        obs_vel_sorted[:limit], [], [],
                                        r_sorted[:limit], [], [],
                                        [], [], [], []
                                    )
                                    await asyncio.sleep(sleep_time)

                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']:
                                        return
                                    limit = i * chunk_size
                                    limit_over = limit * oversample
                                    render_frame(
                                        obs_vel_sorted, v_bar_scat_sorted[:limit], [],
                                        r_sorted, r_sorted[:limit], [],
                                        v_bar_hist[:limit_over], pesi_over[:limit_over],
                                        [], []
                                    )
                                    await asyncio.sleep(sleep_time)

                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']:
                                        return
                                    limit = i * chunk_size
                                    limit_over = limit * oversample
                                    render_frame(
                                        obs_vel_sorted, v_bar_scat_sorted, v_dm_scat_sorted[:limit],
                                        r_sorted, r_sorted, r_sorted[:limit],
                                        v_bar_hist, pesi_over,
                                        v_dm_hist[:limit_over], pesi_over[:limit_over]
                                    )
                                    await asyncio.sleep(sleep_time)
                            else:
                                render_frame(
                                    obs_vel_sorted, v_bar_scat_sorted, v_dm_scat_sorted,
                                    r_sorted, r_sorted, r_sorted,
                                    v_bar_hist, pesi_over,
                                    v_dm_hist, pesi_over
                                )

                        except Exception as ex:
                            print(f"Error in update_coma_histogram_v2: {ex}")
                            return



                    

                def handle_tab_change(e):
                      
                    if e.value == 'cluster':
                          
                        ui.timer(0.1, lambda: refresh_cluster_plots(full_anim=True), once=True)

                   
                tabs.on_value_change(handle_tab_change)
                
                
#panel cluster exercise
                with ui.tab_panel('clusdm').props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg"):
                    ui.label("Explore how virial theorem reveals dark matter in galaxy clusters. ").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                    with ui.dialog() as inst_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label=Instructions for cluster mass activities'):
                        html_info_box(r"""
        <h3>Cluster Mass Analysis</h3>
        <p>Imagine analyzing a <b>galaxy cluster</b> to reveal dark matter.</p>
        <ul>
            <li>Choose a dataset and complete the missing fields.</li>
            <li>Click <b>Run Analysis</b> to compare <b>luminous mass</b> vs <b>total (virial) mass</b>.</li>
            <li>Review the additional information provided in the "Additional Information: Numerical Explanation" PDF. This document offers a detailed numerical explanation of the concepts related to the dark matter.</li>
        </ul>
    """).props('aria-label="Descriptive text about galaxy cluster mass activities"')
                        aria_button("Close", "Close the box",on_click=lambda:inst_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as cur_el_gordo, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="Information about El Gordo galaxy cluster"'):
                        html_info_box(r"""
                        <h3>"El Gordo": The Cosmic Heavyweight</h3>
                        <p>Did you know that astronomers discovered a galaxy cluster so massive they officially nicknamed it <b>"El Gordo"</b> (Spanish for <i>"The Fat One"</i>)?</p>
                        
                        <p>Located over 7 billion light-years away, it is the largest, hottest, and brightest X-ray cluster ever discovered in the distant Universe. It contains the mass of roughly <b>3 million billion Suns</b> ($3 \times 10^{15} M_\odot$)!</p>
                        
                        <p>Like the Bullet Cluster, "El Gordo" is actually the site of a violent collision between two smaller clusters, making it an excellent laboratory for studying Dark Matter.</p>
                        """)
                        
                        reference_box("**Source:** [NASA Chandra](https://chandra.harvard.edu/photo/2012/elgordo/)")
                        
                        aria_button("Close", "close", on_click=lambda:cur_el_gordo.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                   
                    cluster_popup_container = None
                    @ui.refreshable
                    def update_cluster_popup_plots():
                        global cluster_popup_container
                        if cluster_popup_container is None: return
                        cluster_popup_container.clear()

                        selected_file = cluster_select.value
                        if not selected_file:
                            return

                        df = get_cluster_data_cached(selected_file)

                        if df is None or df.empty:
                            return

                        is_coma = "coma" in selected_file.lower()
                        members = df.copy()
                        if "RV" not in members.columns and "redshift" in members.columns:
                            members["RV"] = members["redshift"] * c_light
                        if not is_coma and "bmag" in members.columns:
                            members = members[members["bmag"] > 0.0]

                        members = filter_cluster_members(members, vel_col="RV")
                        if len(members) < 5:
                            ui.label(f"Not enough valid galaxies in {selected_file}.").classes('text-orange-500')
                            return

                        observed_vel = members["RV"].values
                        cluster_distance_Mpc = float(np.median(observed_vel)) / 73.0
                        z_cluster = float(np.nanmedian(members["redshift"].values)) if is_coma else float(np.nanmedian(observed_vel / c_light))
                        rho_crit = rho_crit_Msunkpc3(73.0)

                        if is_coma:
                            m_star_arr = stellar_mass_from_r_mag(members["modelmag_r"].values, members["extinction_r"].values, z_cluster, M_to_L_ratio=2.0)
                        else:
                            m_star_arr = stellar_mass_from_B_mag(members["bmag"].values, cluster_distance_Mpc, M_to_L_ratio=5.0)

                        M_star_tot = float(np.sum(m_star_arr))
                        sigma_global = float(np.std(observed_vel))
                        M_tot, R200 = estimate_M200_R200_from_sigma(sigma_global, rho_crit)

                        f_gas_global = 0.093 * (((0.7 * M_tot) / 2e14)**0.21)
                        M_gas_tot = float((0.7 * M_tot) * f_gas_global)
                        sigma_bar = float(np.sqrt((G_grav * (M_star_tot + M_gas_tot)) / (3.0 * R200)))

                        v_peculiar = observed_vel - np.mean(observed_vel)
                        oversample = 100
                        rng = np.random.default_rng(seed=42)
                        v_sim_tot = rng.normal(loc=0.0, scale=sigma_global, size=len(v_peculiar) * oversample)
                        v_bar = rng.normal(loc=0.0, scale=sigma_bar, size=len(v_peculiar) * oversample)
                        pesi = np.ones_like(v_sim_tot) / oversample

                        x_lim = max(float(np.abs(v_peculiar).max()), 3.5 * sigma_global) * 1.10
                        bins = np.linspace(-x_lim, x_lim, 40)

                        with cluster_popup_container:
                            with ui.row().classes('w-full justify-center gap-8 flex-wrap'):
                                with ui.pyplot(figsize=(8, 6)):
                                    import matplotlib.ticker as ticker
                                    ax = plt.gca()
                                    ax.hist(v_peculiar, bins=bins, alpha=0.4, color='blue', label='Observations')
                                    ax.hist(v_bar, bins=bins, weights=pesi, histtype='step', linewidth=2, color='red', label='Baryonic')
                                    ax.hist(v_sim_tot, bins=bins, weights=pesi, histtype='step', linewidth=3, color='green', label='Total Simulated')
                                    ax.set_title(f"{os.path.splitext(selected_file)[0]}", fontsize=14, fontweight='bold')
                                    ax.set_xlabel('Velocity (km/s)', fontsize=12)
                                    ax.set_ylabel('Number of Galaxies', fontsize=12)
                                    ax.set_xlim(-x_lim, x_lim)
                                    ax.legend(fontsize=11)
                                    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    def open_cluster_analysis_dialog():
                        global cluster_popup_container
                        
                        with ui.dialog() as comp_cluster_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="cluster-main-title"'):
                            
                            with ui.column().classes('w-full h-full bg-white'):
                                
                                with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0'):
                                    ui.label('Cluster Analysis: Physics & Simulation').classes('text-xl font-bold').props('id="cluster-main-title" role=heading aria-level=2 tabindex=0')
                                    aria_button('Close', 'close', on_click=comp_cluster_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                                
                                with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                    
                                   
                                    
                                    t_virial = ui.tab('Virial Theorem')
                                    t_legend = ui.tab('Legend & Units')
                                    t_plots = ui.tab('Cluster Plots')
                                    t_math = ui.tab('Dark Matter Evidence')
                                    

                                    tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))

                                with ui.tab_panels(tabs, value=t_virial).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                    
                                   
                                    with ui.tab_panel(t_plots).classes('flex flex-col items-center justify-start'):
                                        ui.label("Cluster Analysis Plots").classes("text-2xl font-bold mb-4")
                                        
                                        ui.select(options=cluster_select.options, label='Select a Cluster Dataset') \
                                            .bind_value(cluster_select, 'value') \
                                            .classes('w-64 mb-6 bg-slate-800 text-white rounded shadow-sm') \
                                            .props('behavior="menu" outlined popup-content-class="bg-slate-800 text-white"') \
                                            .on_value_change(lambda e: update_cluster_popup_plots.refresh())
                                  
                                        cluster_popup_container = ui.column().classes("w-full items-center justify-center")
                                        update_cluster_popup_plots()
                                      

                                        info_box("**Dataset variables**: Cluster (name), ID (galaxy ID), RAdeg (right ascension), DEdeg (declination), RV (radial velocity), e_RV (velocity error), q_RV (quality flag), Nint (n.lines), bmag (B band magnitude).")
                                        reference_box("""**Dataset reference**: Way M.J. et al., *Redshifts in the Southern Abell Redshift Survey Clusters*.""").classes('text-base italic')

                                   
                                    with ui.tab_panel(t_math):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>

    <h3>Further Evidence: Lensing & Collisions</h3>

    <p>While rotation curves provided the first hint, other independent observations confirm the existence of Dark Matter.</p>



    <hr class="my-4">



    <h4>1. Gravitational Lensing</h4>

   

    <div style="text-align: center; margin: 10px 0;">

        <img src="/images/gravitational_lensing.jpg" alt="Gravitational Lensing Diagram" style="width: 100%; max-width: 400px; border-radius: 8px; border: 1px solid #ccc;">

        <p style="font-size: 0.85em; color: #555;"><i>Light bending around a massive object.</i></p>

    </div>



    <p>According to Einstein's General Relativity, mass bends space-time, causing light paths to curve. This effect is called <b>Gravitational Lensing</b>.</p>

    <ul style="list-style-type: disc; margin-left: 20px;">

        <li>Astronomers observe strong distortions (arcs and rings) of background galaxies around massive clusters.</li>

        <li>Calculations show that the <b>visible mass</b> (stars and gas) is insufficient to cause such strong bending.</li>

        <li>The extra bending implies a huge amount of invisible mass: <b>Dark Matter</b>.</li>

    </ul>



    <hr class="my-4">



    <h4>2. The Bullet Cluster (1E 0657-56)</h4>



    <div style="text-align: center; margin: 10px 0;">

        <img src="/images/bullet_lensing.jpg" alt="Bullet Cluster Composite Image" style="width: 100%; max-width: 400px; border-radius: 8px; border: 1px solid #ccc;">

        <p style="font-size: 0.85em; color: #555;"><i>Composite image: X-ray (pink) vs. Mass density (blue).</i></p>

    </div>



    <p>The coloured cloud for Dark Matter comes from the collision of two galaxy clusters, known as the <b>Bullet Cluster</b>.</p>

    <ul style="list-style-type: disc; margin-left: 20px;">

        <li><b>Baryonic Matter (Pink/Red):</b> The hot gas, observed in X-rays, constitutes most of the normal matter. During the collision, the gas clouds interacted electromagnetically (friction), slowed down, and lagged behind.</li>

        <li><b>Total Mass (Blue):</b> Mapped via gravitational lensing, the majority of the mass passed right through the collision without slowing down.</li>

    </ul>

    <p><b>Conclusion:</b> The mass (Blue) is separated from the gas (Pink). This proves that the majority of matter is <b>collisionless</b> and interacts only via gravity—fundamental properties of Dark Matter.</p>



    <hr class="my-4">



    <h4>References</h4>

    <div style="font-size: 0.9em; line-height: 1.4; color: #444;">

        <p style="margin-bottom: 5px;">

            <b>[Wikipedia]</b>

            <a href="https://en.wikipedia.org/wiki/Gravitational_lens" target="_blank" style="color: #2563eb; text-decoration: underline;">Gravitational Lens</a>

        </p>

        <p style="margin-bottom: 5px;">

            <b>[Wikipedia]</b>

            <a href="https://en.wikipedia.org/wiki/Bullet_Cluster" target="_blank" style="color: #2563eb; text-decoration: underline;">Bullet Cluster (1E 0657-56)</a>

        </p>

        <p style="margin-bottom: 5px;">

            <b>[NASA]</b>

            <a href="https://chandra.harvard.edu/press/06_releases/press_082106.html" target="_blank" style="color: #2563eb; text-decoration: underline;">Chandra Press Release (2006) - "NASA Finds Direct Proof of Dark Matter"</a>

        </p>

    </div>

""")
                                        
                                    with ui.tab_panel(t_virial):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>

       <h3>Computational Notes</h3>
      <ul>
          <li><b>Step 1:</b> Filter cluster members using a \( \pm 6000\,\mathrm{km/s} \) window around the velocity peak followed by iterative \( 3\sigma_{\mathrm{MAD}} \) clipping (Beers et al., 1990).</li>
          <li><b>Step 2:</b> Compute cluster distance from the linear Hubble law (\( H_0 = 73\,\mathrm{km\,s^{-1}\,Mpc^{-1}} \)):<br>
              <span class="math">\( D_L \approx \frac{\mathrm{median}(v)}{H_0} \)</span> (Mpc) and distance modulus <span class="math">\( \mu = 5 \log_{10}(D_{\mathrm{pc}}) - 5 \)</span>.</li>
          <li><b>Step 3:</b> Compute stellar mass from galaxy magnitudes:<br>
              <b>SARS Abell (B-band):</b> <span class="math">\( M_B = m_B - \mu, \; L_B = 10^{-0.4(M_B - 5.48)}, \; M_{\mathrm{stars}} = 5.0 \sum_i L_{B,i} \)</span><br>
              <b>Coma (r-band):</b> <span class="math">\( M_r = m_r - A_r - \mu, \; L_r = 10^{0.4(4.67 - M_r)}, \; M_{\mathrm{stars}} = 2.0 \sum_i L_{r,i} \)</span></li>
          <li><b>Step 4:</b> Compute global velocity dispersion <span class="math">\( \sigma_{\mathrm{obs}} \)</span> and solve the Virial Theorem with \( \rho_{\mathrm{crit}} = \frac{3H_0^2}{8\pi G} \) to obtain <span class="math">\( M_{200} \)</span> and <span class="math">\( R_{200} \)</span>.</li>
          <li><b>Step 5:</b> Gas mass (Giodini et al., 2009, with \( M_{500} \approx 0.7 M_{200} \)):<br>
              <span class="math">\( M_{\mathrm{gas}} = (0.7 M_{200}) \cdot 0.093 \left( \frac{0.7 M_{200}}{2 \times 10^{14}\,M_\odot} \right)^{0.21} \)</span></li>
          <li><b>Step 6:</b> Total baryonic and Dark Matter mass:<br>
              <span class="math">\( M_{\mathrm{lum}} = M_{\mathrm{stars}} + M_{\mathrm{gas}}, \quad M_{\mathrm{DM}} = M_{200} - M_{\mathrm{lum}} \)</span></li>
      </ul>



        <h4>Virial Theorem Setup</h4>

       

        <ul>

            <li><b>Step 7:</b> Gravitational potential energy: <span class="math">\( U(r) = - \frac{G M m}{r} \)</span></li>

            <li><b>Step 8:</b> Gravitational force: <span class="math">\( F_{\mathrm{grav}} = \frac{G M m}{r^2} \)</span></li>

            <li><b>Step 9:</b> Newton 2nd law force (for circular motion): <span class="math">\( F_{\mathrm{newton}} = \frac{m v^2}{r} \)</span></li>

            <li><b>Step 10:</b> Equating: <span class="math">\( v^2(r) = \frac{G M}{r} \)</span></li>

            <li><b>Step 11:</b> Kinetic energy: <span class="math">\( K = \tfrac{1}{2} m v^2 = \tfrac{1}{2} \frac{G M m}{r} \)</span></li>

            <li><b>Step 12:</b> Relation K vs U: <span class="math">\( K = -\tfrac{1}{2} U \)</span></li>

            <li><b>Step 13:</b> Virial theorem: <span class="math">\( 2K + U = 0 \)</span></li>

            <li><b>Step 14:</b> System treated as <span class="math">\( N \)</span> a satellite orbiting a planet.</li>

        </ul>



        <h4>Cluster Mass </h4>

        <ul>

            <li><b>Step 15:</b> 1-D velocity dispersion: <span class="math">\( \sigma_v = \sqrt{\tfrac{1}{N} \sum_{i=0}^N (v_i - \bar{v})^2} \)</span></li>

            <li><b>Step 16:</b> 3-D velocity dispersion (isotropic): <span class="math">\( \sigma_{3D} = \sqrt{3} \, \sigma_v \)</span></li>

            <li><b>Step 17:</b> Total cluster mass: <span class="math">\( M_{\mathrm{tot}} = \frac{3 \, \sigma_v^2 \, r}{G} \)</span></li>

        </ul>

    """)

                                 
                                    with ui.tab_panel(t_legend):
                                        html_info_box(r"""
                                                      <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>
                                                      <h3 class="text-xl font-bold mb-2">Legend of Symbols and Units</h3>

        <ul class="list-disc pl-5 space-y-1">

            <li><b>RV</b> = Radial velocity of cluster galaxies [km/s] (dataset)</li>

            <li><b>&sigma;<sub>v</sub></b> = Velocity dispersion of cluster galaxies [km/s]</li>

            <li><b>N</b> = Number of galaxies in the cluster</li>

            <li><b>D</b> = Distance to the cluster [Mpc] (fixed or from redshift <i>D = z &middot; c / H<sub>0</sub></i>)</li>

            <li><b>z</b> = redshift</li>

            <li><b>r</b> = Cluster radius [kpc] (computed from angular separation in dataset and distance)</li>

            <li><b>M<sub>total</sub>(r)</b> = Total mass (Virial theorem) [M<sub>&odot;</sub>]</li>

            <li><b>M<sub>lum</sub>(r)</b> = Luminous mass (B-band luminosity &times; M/L) [M<sub>&odot;</sub>]</li>

            <li><b>M/L</b> = 5 (Mass-to-light ratio in B-band) [M<sub>&odot;</sub>/L<sub>&odot;</sub>]</li>

            <li><b>L(r)</b> = Cumulative B-band luminosity within radius r [L<sub>&odot;</sub>]</li>

            <li><b>M<sub>sun,B</sub></b> = 5.48 (absolute magnitude of the Sun in B-band)</li>

            <li><b>H<sub>0</sub></b> = 73 km/s/Mpc (Hubble constant)</li>

            <li><b>G</b> = 4.30091 &times; 10<sup>-6</sup> kpc&middot;(km/s)&sup2;&middot;M<sub>&odot;</sub><sup>-1</sup></li>

            <li><b>c</b> = 3 &times; 10<sup>5</sup> km/s (Speed of light)</li>

            <li><b>U</b> = Gravitational potential energy  [km&sup2;/s&sup2;]</li>

            <li><b>F<sub>grav</sub></b> = Gravitational force [M<sub>&odot;</sub>&middot;kpc/s&sup2;]</li>

            <li><b>F<sub>Newton</sub></b> = Newton's 2nd law force [M<sub>&odot;</sub>&middot;kpc/s&sup2;]</li>

            <li><b>v&sup2;(R)</b> = Squared orbital velocity for circular orbit [km&sup2;/s&sup2;]</li>

            <li><b>K</b> = Kinetic energy [km&sup2;/s&sup2;]</li>

            <li><b>2K + U = 0</b> = Virial theorem for bound system</li>

            <li><b>M<sub>circ</sub></b> = Mass corresponding to circular orbit of a satellite [M<sub>&odot;</sub>]</li>

            <li><b>M<sub>tot</sub>(r)</b> = Total virial mass of the cluster [M<sub>&odot;</sub>]</li>

            <li><b>L<sub>B</sub></b> = Luminosity of a single galaxy in B-band [L<sub>&odot;</sub>]</li>

  

            <li><b>M<sub>lum</sub>(r)</b> = Luminous mass within radius r [M<sub>&odot;</sub>]</li>

            <li><b>D<sub>pc</sub></b> = Cluster distance in parsecs [pc]</li>

            <li><b>dist<sub>mod</sub></b> = Distance modulus</li>

            <li><b>M<sub>abs</sub></b> = Absolute magnitude of a galaxy [mag]</li>

            <li><b>m</b> = Mass of the satellite or galaxy [M<sub>&odot;</sub>]</li>

            <li><b>M</b> = Mass inside radius r of the planet or cluster [M<sub>&odot;</sub>]</li>

           


            <li><b>L<sub>B,i</sub></b> = Luminosity of a single galaxy [L<sub>&odot;</sub>]</li>

            <li><b>bmag</b> = Apparent magnitude in B band [mag] (dataset)</li>

        </ul>

        <h3 class="text-xl font-bold mb-2">Units Conversion</h3>

        <ul class="list-disc pl-5 space-y-1">

            <li><b>1 kpc</b> = 3.086 &times; 10<sup>16</sup> m = 3.26 &times; 10&sup3; light-years</li>

            <li><b>1 pc</b> = 3.086 &times; 10<sup>13</sup> km = 3.26 light-years</li>

            <li><b>1 Mpc</b> = 10<sup>6</sup> pc = 3.086 &times; 10<sup>19</sup> km</li>

            <li><b>1 km/s</b> = 3.6 &times; 10&sup3; km/h = 10&sup3; m/s</li>

            <li><b>1 M<sub>&odot;</sub></b> = 1.989 &times; 10<sup>30</sup> kg (solar mass)</li>

            <li><b>1 L<sub>&odot;</sub></b> = 3.828 &times; 10<sup>26</sup> W (solar luminosity)</li>

            <li><b>Distance modulus:</b> <i>m - M = 5 log<sub>10</sub>(d / 10 pc)</i></li>

            <li><b>Small angle:</b> 1 arcsec at distance D &rarr; <i>D &middot; tan(1") &asymp; D / 206265</i></li>

        </ul>

    """)

                        comp_cluster_dialog.open()
                        
                        ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")
                    with ui.row().classes('w-full justify-center '):
                        cluster_files = [f for f in os.listdir(CLUSTER_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                        cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)

                        if not cluster_file_map:
                            ui.label("No cluster data files found.").classes('text-red-500')
                        else:
                        
                            cluster_select = ui.select(cluster_files, label='Select a Cluster Dataset').classes('flex-1 max-w-md items-start text-lg')
                        aria_button("Instructions", "Read instruction for cluster activities",on_click=safe_click(lambda: [inst_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button(
    "Scientific Info", 
    "Open the full cluster analysis dialog with plots and math", 
    on_click=lambda: [open_cluster_analysis_dialog(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")]
).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_el_gordo.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                        
                        ui.add_head_html('''
    <style>
        .math-text { font-family: 'Times New Roman', serif; font-style: italic; font-size: 1.25rem; display: flex; align-items: center; flex-wrap: nowrap; }
        .math-sub { vertical-align: sub; font-size: 0.7em; margin-left: -2px; }
        .math-sup { vertical-align: super; font-size: 0.7em; }
        .fraction { display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 4px; }
        .numerator { border-bottom: 2px solid white; padding-bottom: 2px; }
        .denominator { padding-top: 2px; }
        
        /* Modifica per la radice quadrata */
        .sqrt-content { 
            border-top: 2px solid white; 
            padding-top: 4px; 
            display: inline-flex; 
            align-items: center; 
            margin-top: 0.5em;
            white-space: nowrap; /* Impedisce a capo */
        }

        /* Input pulito */
        .formula-input .q-field__control { 
            height: 32px !important; min-height: 32px !important; 
            padding: 0 8px !important;
            background: rgba(255,255,255,0.1) !important;
        }
        .formula-input .q-field__native { color: #90caf9; text-align: center; padding: 0; }
    </style>
''')
                        vars_options = [
    'sigma_v', 'r', 'G', 'G*m/r', '0.5*m*v^2', '-G*m/r',
    'G*m/r^2', 'm*v^2/r', 'K', 'U', 'v^2*r/G',
    'M_abs', 'L_B', 'L(r)', 'd*1e6', 'd_pc', 'bmag', 'rv','(v^2)*r/g'
]
                        answer_sigma,answer_r,answer_g2,answer_vel,answer_vel_mean = {}, {},{},{},{}
                        ans_U, ans_Fg, ans_Fc, ans_K, ans_v2 = {}, {}, {}, {}, {}
                        ans_virialLHS, ans_virialRHS, ans_Mcirc, ans_Mvir,answer_LB = {}, {}, {}, {}, {}
                        answer_D, answer_Dmod, answer_Mabs, answer_Lsum,answer_Mlum = {}, {}, {}, {},{}
                        @ui.refreshable
                        def show_cluster_pseudocode():
                         
                            with ui.card().classes("items-center justify-center p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                                ui.markdown("Cluster Mass Exercise: fill in the missing parts in the pseudo-code below")\
                                    .classes("text-2xl font-bold text-blue-300 mb-4 item-center")\
                                    .props('aria-label="cluster exercises fill the formulas in the boxes" role=heading aria-level=2 tabindex=0')
                                
                              
                                with ui.row().classes('w-full max-w-6xl justify-center gap-6 no-wrap items-start'):
                                    
                                  
                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):

                                        # 0) LOAD
                                        ui.label("0) Load cluster dataset").classes('text-blue-200 font-bold text-lg')
                                        ui.code(f'data = load_data("{cluster_select.value}")', language='python').classes('w-full')

                                        # 1) ORBITAL DYNAMICS
                                        ui.label("1) Circular motion (e.g. satellite around planet)").classes('text-blue-200 font-bold text-lg')

                                        # 2) POTENTIAL ENERGY
                                        ui.label("2) Compute the gravitational potential energy").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('U(r) = ')
                                            ans_U['el'] = aria_select_input(vars_options, "Select variable for potential energy")

                                        # 3) FORCE GRAV
                                        ui.label("3) Compute the gravitational force").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('F<span class="math-sub">grav</span> = ')
                                            ans_Fg['el'] = aria_select_input(vars_options, "Select variable for gravitational force")

                                        # 4) FORCE CENT
                                        ui.label("4) Compute the Newton's 2nd law force").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('F<span class="math-sub">Newton</span> = ')
                                            ans_Fc['el'] = aria_select_input(vars_options, "Select variable for Newton's 2nd law force")

                                        # 5) EQUATE
                                        ui.label("5) Equate: F_grav = m*centripetal acceleration → derive v²(R)").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('v<span class="math-sup">2</span>(R) = ')
                                            ans_v2['el'] = aria_select_input(vars_options, "Select variable for squared velocity")

                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 6) KINETIC
                                        ui.label("6) Kinetic energy ").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('K = ')
                                            ans_K['el'] = aria_select_input(vars_options, "Select variable for kinetic energy")

                                        # 7) VIRIAL THEOREM
                                        ui.label("7) Virial theorem (time-averaged, bound system)").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('2&lang;') 
                                            ans_virialLHS['el'] = aria_select_input(vars_options, "Select variable for kinetic energy in virial theorem")
                                            ui.html('&rang; + &lang;') 
                                            ans_virialRHS['el'] = aria_select_input(vars_options, "Select variable for potential energy in virial theorem")
                                            ui.html('&rang; = 0')

                                        # 8) CIRCULAR MASS
                                        ui.label("8) Compute circular orbit mass").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">circ</span>(r) = ')
                                            ans_Mcirc['el'] = aria_select_input(vars_options, "Select formula for circular mass")

                                        # 9) TOTAL MASS SYSTEM
                                        ui.label("9) Compute the total mass for a system of N galaxies:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">tot</span>(r) &asymp; &Sigma;')
                                            ans_Mvir['el'] = aria_select_input(vars_options, "Select formula for total mass system")

                                        # 10) VELOCITY DISPERSION
                                        ui.label("10) Compute velocity dispersion").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('&sigma;<span class="math-sub">v</span><span class="math-sup">2</span> = ')
                                            ui.html(' (1/N) &middot; ')
                                            ui.html('&Sigma; |')
                                            answer_vel['el'] = aria_select_input(vars_options, "Select variable for radial velocity")
                                            ui.html('&minus; mean(')
                                            answer_vel_mean['el'] = aria_select_input(vars_options, "Select variable for mean radial velocity")
                                            ui.html(')|<span class="math-sup">2</span>')

                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 11) VIRIAL TOTAL MASS
                                        ui.label("11) Compute virial/total mass").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">total</span>(r) = ')
                                            with ui.column().classes('fraction'):
                                                with ui.row().classes('numerator items-center gap-1'):
                                                    ui.html('3 &middot; ')
                                                    answer_sigma['el'] = aria_select_input(vars_options, "Select variable for velocity dispersion")
                                                    ui.html('<span class="math-sup">2</span> &middot; ')
                                                    answer_r['el'] = aria_select_input(vars_options, "Select variable for radius")
                                                with ui.row().classes('denominator w-full justify-center'):
                                                    answer_g2['el'] = aria_select_input(vars_options, "Select variable for gravitational constant")

                                        # 12) CLUSTER DISTANCE
                                        ui.label("12) Compute cluster distance in pc:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('D<span class="math-sub">pc</span> = ')
                                            answer_D['el'] = aria_select_input(vars_options, "Select formula for cluster distance")

                                        # 13) DISTANCE MODULUS
                                        ui.label("13) Compute distance modulus:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('dist<sub>mod</sub> = 5 &middot; log<span class="math-sub">10</span>(')
                                            answer_Dmod['el'] = aria_select_input(vars_options, "Select variable for distance modulus in pc")
                                            ui.html(') &minus; 5')

                                        # 14) ABSOLUTE MAGNITUDE
                                        ui.label("14) Compute absolute magnitude from apparent magnitude (dataset) and distance:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">abs</span> = ')
                                            answer_Mabs['el'] = aria_select_input(vars_options, "Select variable for apparent magnitude")
                                            ui.html('&minus; dist<sub>mod</sub>')

                                        # 15) LUMINOSITY
                                        ui.label("15) Compute luminosity for each galaxy:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('L<span class="math-sub">B</span> = 10<span class="math-sup">-0.4 &middot; (</span>')
                                            answer_LB['el'] = aria_select_input(vars_options, "Select variable for absolute magnitude")
                                            ui.html('<span class="math-sup"> &minus; M<span class="math-sub">sun,B</span>)</span>')

                                        # 16) TOTAL LUMINOSITY
                                        ui.label("16) Compute total luminosity:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('L(r) = &Sigma;(')
                                            answer_Lsum['el'] = aria_select_input(vars_options, "Select variable for galaxy luminosity")
                                            ui.html(') <span class="text-sm not-italic ml-2">for galaxies with r<span class="math-sub">i</span> &le; r</span>')

                                        # 17) LUMINOUS MASS
                                        ui.label("17) Compute luminous mass:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">lum</span>(r) = (M/L) &middot ')
                                            answer_Mlum['el'] = aria_select_input(vars_options, "Select variable for total luminosity")
                                            ui.html('+ M<span class="math-sub">gas</span>(r); ')

                                      
                                                        
                        show_cluster_pseudocode()
                        cluster_select.on('update:model-value', lambda e: show_cluster_pseudocode.refresh())

                        
                        cluster_plot_container = ui.row().classes('w-full gap-4 justify-center')
                        @ui.refreshable
                        def update_cluster_mass_profile():
                            cluster_plot_container.clear()
                            
                            with cluster_plot_container:
                                try:
                                    results_data = []
                                    # Calcola massa e percentuale DM per tutti i cluster presenti nel dataset
                                    for file in cluster_files:
                                        df = get_cluster_data_cached(file)
                                        if df is None or df.empty: continue
                                        
                                        members = df.copy()
                                        is_coma = "coma" in file.lower()
                                        
                                        if is_coma:
                                            if 'redshift' in members.columns:
                                                members['RV'] = members['redshift'] * c_light
                                            mag_col = 'modelmag_r'
                                        else:
                                            mag_col = 'bmag'
                                            
                                        if 'RV' not in members.columns: continue
                                            
                                      
                                        for _ in range(5):
                                            v_rel = members["RV"] - np.median(members["RV"])
                                            mad = np.median(np.abs(v_rel))
                                            sigma = 1.4826 * mad if mad > 0 else np.std(v_rel)
                                            mask = np.abs(v_rel) <= 3 * sigma
                                            new_members = members[mask]
                                            if len(new_members) == len(members): break
                                            members = new_members

                                        if len(members) < 5: continue
                                        
                                        observed_vel = members["RV"].values
                                        if is_coma:
                                            sigma_global = np.std(observed_vel)
                                        else:
                                            sigma_global = np.std(observed_vel - np.median(observed_vel))
                                            
                                        rho_crit = rho_crit_Msunkpc3(73.0)
                                        M200, R200 = estimate_M200_R200_from_sigma(sigma_global, rho_crit)
                                        M_tot = M200
                                        
                                       
                                        #if M_tot > 5e15 or M_tot < 1e13: continue

                                        cluster_distance = np.median(observed_vel) / 73.0
                                        if is_coma:
                                            z_cluster = np.nanmedian(members['redshift'])
                                            cluster_distance = (c_light * z_cluster) / 73.0
                                            dist_mod = 5 * np.log10(cluster_distance * 1e6) - 5
                                            ext_val = members['extinction_r'].values if 'extinction_r' in members.columns else 0.0
                                            M_abs = members[mag_col].values - ext_val - dist_mod
                                            L_B = 10**(-0.4 * (M_abs - 4.67))
                                            MLR = 2.0
                                        else:
                                            D_pc = cluster_distance * 1e6
                                            dist_mod = 5 * np.log10(D_pc) - 5
                                            M_abs = members[mag_col].values - dist_mod
                                            L_B = 10**(-0.4 * (M_abs - 5.48))
                                            MLR = 5.0
                                        
                                        M_star_tot = MLR * np.sum(L_B)
                                        f_gas_global = 0.093 * (((0.7 * M_tot) / 2e14)**0.21)
                                        M_gas_tot = (0.7 * M_tot) * f_gas_global
                                        
                                        # Sottrazione pura
                                        M_DM_tot = np.maximum(0.0, M_tot - M_star_tot - M_gas_tot)
                                        perc_dm = (M_DM_tot / M_tot) * 100.0

                                        results_data.append({
                                            "Cluster": os.path.splitext(file)[0],
                                            "M_tot": M_tot,
                                            "Perc_DM": perc_dm
                                        })
                                        
                                    if results_data:
                                        df_res = pd.DataFrame(results_data)
                                        fig = px.scatter(df_res, x="M_tot", y="Perc_DM", hover_name="Cluster",
                                                         log_x=True, color_discrete_sequence=['forestgreen'],
                                                         labels={"M_tot": "Total Mass (M_sun)", "Perc_DM": "Dark Matter Fraction (%)"},
                                                         title="Clusters: Dark Matter Fraction vs Total Mass")
                                        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')), opacity=0.8)
                                        ui.plotly(fig).classes('w-full max-w-4xl h-96')
                                        
                                        ui.element('div').props('role=img tabindex=0 aria-label=Interactive scatter plot showing the dark matter fraction vs total mass for all analyzed clusters')

                                    # Sezione dedicata a Immagine e Tabella del Cluster attualmente selezionato
                                    selected_file = cluster_select.value
                                    if selected_file:
                                        image_filepath = cluster_file_map.get(selected_file)
                                        with ui.row().classes('w-full gap-4 justify-start mt-6'):
                                            with ui.column().classes('flex-1 items-center '):
                                                if image_filepath and os.path.exists(image_filepath):
                                                    file_name = os.path.basename(image_filepath)
                                                    web_path = f'/cluster_img/{file_name}'
                                                    aria_image(web_path, f"Image of the selected galaxy cluster {selected_file} ").classes('w-full max-w-xl')
                                                    reference_box("""
        **Image references:**  
        - [ESA Hubble](https://esahubble.org/)  
        - Bonnarel, F. et al. (2000). *The ALADIN interactive sky atlas*. Astronomy and Astrophysics Supplement Series  
        - Digitized Sky Survey 2 (DSS2), via CDS HiPS. [Aladin HiPS](https://alasky.cds.unistra.fr)
        """).classes('w-full text-center text-base italic mt-2')
                                                else:
                                                    ui.label("Image not found").classes('text-red-500')

                                            with ui.column().classes('flex-1 items-center '):
                                                table_filepath = os.path.join(CLUSTER_TABLES_PATH, os.path.splitext(selected_file)[0] + ".csv")
                                                if os.path.exists(table_filepath):
                                                    df_table = pd.read_csv(table_filepath)
                                                    ui.table.from_pandas(df_table).props('aria-label=Table of the selected galaxy cluster characteristics role=table').classes('w-full max-w-xl')
                                                    reference_box("""
        **Cluster data references:**  
        - [Wikipedia: List of galaxy clusters](https://en.wikipedia.org/wiki/List_of_galaxy_groups_and_clusters)  
        - Wenger, M., Ochsenbein, F. et al. (2000). *The SIMBAD astronomical database*. Astronomy & Astrophysics Supplement Series. [SIMBAD](https://simbad.u-strasbg.fr)  
        - Helou, G., Madore, B. F. (1988). *The NASA/IPAC Extragalactic Database (NED)*. [NED](https://ned.ipac.caltech.edu)  
        - Abell, G. O. et al. (1989). *A catalog of rich clusters of galaxies*. Astrophysical Journal Supplement Series  
        - VizieR Online Data Catalog: VII/110A. [VizieR](https://vizier.u-strasbg.fr)
        """).classes('w-full text-center text-base italic mt-2')
                                                else:
                                                    ui.label("Table not found").classes('text-red-500')
                                                    
                                except Exception as e:
                                    warning_box(f"Error processing clusters: {e}").classes('text-red-500')
                                    
                                    
                        


                        def check_and_run_cluster():
                            def norm(x: str) -> str:
                                return (x or "").replace(" ", "").lower()


                            s     = norm(answer_sigma.get('el') and answer_sigma['el'].value)
                            r     = norm(answer_r.get('el')     and answer_r['el'].value)
                            g     = norm(answer_g2.get('el')    and answer_g2['el'].value)
                            v2    = norm(ans_v2.get('el')       and ans_v2['el'].value)
                            K     = norm(ans_K.get('el')        and ans_K['el'].value)
                            U     = norm(ans_U.get('el')        and ans_U['el'].value)
                            Fg    = norm(ans_Fg.get('el')       and ans_Fg['el'].value)
                            Fc    = norm(ans_Fc.get('el')       and ans_Fc['el'].value)
                            virLHS= norm(ans_virialLHS.get('el')and ans_virialLHS['el'].value)
                            virRHS= norm(ans_virialRHS.get('el')and ans_virialRHS['el'].value)
                            Mcirc = norm(ans_Mcirc.get('el')    and ans_Mcirc['el'].value)
                            Mvir  = norm(ans_Mvir.get('el')     and ans_Mvir['el'].value)
                            LB    = norm(answer_LB.get('el')    and answer_LB['el'].value)
                            Lsum  = norm(answer_Lsum.get('el')  and answer_Lsum['el'].value)
                            Mlum  = norm(answer_Mlum.get('el')  and answer_Mlum['el'].value)
                            D     = norm(answer_D.get('el')     and answer_D['el'].value)
                            Dmod  = norm(answer_Dmod.get('el')  and answer_Dmod['el'].value)
                            Mabs  = norm(answer_Mabs.get('el')  and answer_Mabs['el'].value)
                            vel   = norm(answer_vel.get('el')   and answer_vel['el'].value)
                            vel_mean = norm(answer_vel_mean.get('el') and answer_vel_mean['el'].value)

    
                            checks_list = [
                                (answer_sigma, s in {"sigma_v","sigmav","σv","σ_v"}),
                                (answer_r,     r in {"r","radius","rad"}),
                                (answer_g2,    g in {"g","gravitational_constant"}),
                                (ans_v2,       v2 in {"g*m/r","g*m*m/r","gm/r"}),
                                (ans_K,        K in {"0.5*m*v^2","0.5*m*v**2","1/2*m*v^2","1/2*m*v**2",
                                                    "g*m/(2*r)","g*m*m/(2*r)"}),
                                (ans_U,        U in {"-g*m/r","-g*m*m/r"}),
                                (ans_Fg,       Fg in {"g*m/r^2","g*m*m/r^2"}),
                                (ans_Fc,       Fc in {"m*v^2/r","m*v**2/r"}),
                                (ans_virialLHS, virLHS in {"k"}),
                                (ans_virialRHS, virRHS in {"u"}),
                                (ans_Mcirc,    Mcirc in {"v^2*r/g","v**2*r/g","(v**2)*r/g","r*v^2/g"}),
                                (ans_Mvir,     Mvir in {"(v^2)*r/g","(rv^2)*r/g","(rv**2)*r/g","(rv^2)r/g","(rv**2)r/g"}),
                                (answer_LB,    LB in {"m_abs","mabs","mag_abs"}),
                                (answer_Lsum,  Lsum in {"l_b","lb","l_b(r)","lb(r)"}),
                                (answer_Mlum,  Mlum in {"(l(r))","l(r)"}),
                                (answer_D,     D in {"d*1e6","dist*1e6","1e6*d"}),
                                (answer_Dmod,  Dmod in {"d_pc","dpc","dist_pc"}),
                                (answer_Mabs,  Mabs in {"bmag"}),
                                (answer_vel,   vel in {"rv","rv_i","rv(i)","v_obs"}),
                                (answer_vel_mean, vel_mean in {"rv", "rv_i", "rv(i)", "v_obs"})
                            ]
    
                        
                            for ans_dict, _ in checks_list:
                                el = ans_dict.get('el')
                                if el:
                                    el.classes(remove='border-red-500 border-green-500')

    
                    
                            all_ok = True
                            for ans_dict, ok in checks_list:
                                el = ans_dict.get('el')
                                if el:
                                    if ok:
                                        el.classes(add='border-green-500')
                                    else:
                                        el.classes(add='border-red-500')
                                        all_ok = False

                            if all_ok:
                                accessible_notify('Correct! Analysis running...', type_='success')
                                update_cluster_mass_profile()
                            else:
                                accessible_notify('Wrong! Try again!', type_='error')

                     
                        with ui.row().classes('w-full justify-center'):
                            with ui.column().classes('flex-1 items-center '):

                                aria_button("Run Analysis", "Ruon the analysis to make the plots",on_click=lambda:check_and_run_cluster()).classes("!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded")
                          
                        cluster_select.on('update:model-value', lambda e: update_cluster_popup_plots.refresh())


                
            






    



