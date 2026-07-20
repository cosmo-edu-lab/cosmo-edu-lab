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
    rho_crit = 2.775e2
    G_grav = 4.30091e-6   
    c_light = 3e5         
    M_sun_r = 4.64  

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


    #function for cluster
        

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


    @ui.page('/module2')
    def module2():
        #add custom CSS for the page
        #ui.add_head_html('<script src="https://cdn.tailwindcss.com"></script>')
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
           
            if not filename:
                return None

            path = os.path.join(GALAXY_DATA_PATH, filename)

            if not os.path.exists(path):
                print(f"File not found: {path}")
                return None

            try:
                
                df = pd.read_csv(path, comment='#', sep=r'\s+', header=0, engine='python')

                
                df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff','', regex=False)

        
                if all(col.replace('.','',1).isdigit() for col in df.columns):
                    
                    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, engine='python')
                    df.columns = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

                return df

            except Exception as ex:
                print(f"Error loading galaxy {filename}: {ex}")
                return None
        @lru_cache(maxsize=32)
        def get_cluster_data_cached(filename):
           
          
            if filename.lower() == "coma_data.csv":
             
                return load_coma_dataset("coma_data.csv")

         
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
                            <li>Write the formula for the gravitational force and the centripetal force of circular motion.</li>
                            <li>How can we derive the orbital velocity using these two formulas?</li>
                            <li>Write the equivalence: <i>F<sub>grav</sub> = F<sub>cent</sub></i>.</li>
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
                        with ui.pyplot(figsize=(8, 6), close=True, clear=True) as m:
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
                        
                    
                        with ui.column().classes('w-full md:w-auto min-w-[280px] items-center'):
                            with ui.row().classes('flex-1'):
                            
                                ui.label("Select a Planet").style("font-size:20px; font-weight:bold; margin-bottom:6px;").props('id=planet_selector_label role=heading aria-level=3 tabindex=0')
                        

                                selector = ui.select(
                            planets,
                            value="Earth" if "Earth" in planets else planets[0],
                            label="Planet"
                        ).classes('w-48').props('id=planet_selector aria-labelledby=planet_selector_label role=listbox tabindex=0')
                                
                            with ui.row().classes('flex-1'):
                                info_panel(selector.value)
                            
                    
                        with ui.column().classes('w-full md:w-auto shrink-0'):
                        
                            plot_velocity(selector.value).classes('max-w-full')
                            with ui.row().classes('w-full justify-center items-center gap-4 mt-4'):
                                aria_button(
                                    "Activity: Kepler", 
                                    "Instructions for Kepler velocity plot", 
                                    on_click=lambda: instr_kepler_phase1.open()
                                ).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                
                                aria_button(
                                    "Dataset", 
                                    "Info Kepler planets dataset",
                                    on_click=safe_click(lambda: [data_kepler.open(), ui.run_javascript("MathJax.typesetPromise()")])
                                ).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        
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

    <h3>The NFW Dark Matter Profile</h3>
    <p>To simulate the halo, we use the <b>Navarro-Frenk-White (NFW) profile</b>, which describes how the density of cold dark matter varies with distance from the center:</p>
    
    <div style="text-align:center; margin: 15px 0;">
        <span class="math">$$ \rho_{NFW}(r) = \frac{\rho_{s}}{(r/r_{s})(1+r/r_{s})^{2}} $$</span>
    </div>

    <p>Where <span class="math">\(\rho_s\)</span> is the characteristic density and <span class="math">\(r_s\)</span> is the scale radius.</p>

    <h4>Adapted to the Dataset</h4>
    <p>The values for <span class="math">\(\rho_s\)</span> and <span class="math">\(r_s\)</span> are not arbitrary; they are <b>adapted to the specific galaxy dataset</b> (Observational Matching):</p>
    <ul style="list-style-type: disc; margin-left: 20px;">
        <li>We calculate the "required" dark matter density <span class="math">\(\rho_{DM}\)</span> derived directly from the observed velocity data at a specific outer radius (<span class="math">\(r_{match}\)</span>), where the dark matter effect is dominant.</li>
        <li>We force the NFW model to match this observation: <span class="math">\(\rho_{NFW}(r_{match}) = \rho_{DM}(r_{match})\)</span>.</li>
    </ul>
    <p>This ensures that the halo structure (defined by its virial mass <span class="math">\(M_{200}\)</span> and concentration <span class="math">\(c\)</span>) is physically consistent with the real galaxy properties you are analyzing.</p>
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
        <p style="margin-bottom: 5px;"><b> Derivation of DM Distribution:</b> Karukes, Salucci & Gentile (2015), "The Dark Matter Distribution in the Spiral NGC 3198 out to 0.22 R_vir".</p>
        <p style="margin-bottom: 5px;"><b>NFW Profile:</b> Navarro, Frenk, White (1996), "The Structure of Cold Dark Matter Halos".</p>
        <p style="margin-bottom: 5px;"><b> Mass Models:</b> Lelli F. et al. (2016), "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry...".</p>
        <p style="margin-bottom: 5px;"><b> Halo Density:</b> Li et al. (2019), "A constant characteristic volume density of dark matter haloes from SPARC rotation curves".</p>
        <p style="margin-bottom: 5px;"><b> Rotation Curve:</b> Karukes E. V., Salucci P. (2016), "The Universal Rotation Curve of Dwarf Disk Galaxies".</p>
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
                            
                    with ui.dialog() as image_dialog2, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl').props('aria-label="Galaxy image" role=dialog'):
                        ui.html("<h5>Galaxy Image</h5>")
                        update_image()
                        aria_button("close",'close',on_click=lambda:image_dialog2.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as table_dialog2, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl').props('aria-label="Galaxy information table" role=dialog'):
                        update_table()
                        aria_button("close",'close',on_click=lambda:table_dialog2.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                   
                    with ui.dialog() as comp_physics_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="physics-title"'):
                        with ui.column().classes('w-full h-full bg-white'):
                            
                        
                            with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="physics-title"'):
                                ui.label('Physics, Math & Morphology').classes('text-xl font-bold').props('id="physics-title" role=heading aria-level=2 tabindex=0')
                                aria_button('Close', 'close', on_click=comp_physics_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                          
                            with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                t_comp = ui.tab('Computational Notes')
                                t_chi2 = ui.tab('χ² Minimization')
                                t_morpho = ui.tab('Morphology Top-View')
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
                                            <li>
                                                <b>Step 1:</b> Baryonic velocity with Mass-to-Light ratio (\( \Upsilon \))<br>
                                                <span class="math">\( v_{\mathrm{bar}}^2(r) = v_{\mathrm{gas}}(r)|v_{\mathrm{gas}}(r)| + \Upsilon \left[ v_{\mathrm{disk}}(r)|v_{\mathrm{disk}}(r)| + v_{\mathrm{bulge}}(r)|v_{\mathrm{bulge}}(r)| \right] \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 2:</b> Baryonic mass<br>
                                                <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, \max(v_{\mathrm{bar}}^2(r), 0)}{G} \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 3:</b> Total observed mass from velocity data<br>
                                                <span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2(r)}{G} \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 4:</b> Observed dark matter density (from data)<br>
                                                <span class="math">\( \rho_{\mathrm{DM}}(r) = \frac{1}{4 \pi G r^2} \, \frac{d}{dr} \Bigg[ r^2 \Big( \frac{v_{\mathrm{obs}}^2(r) - v_{\mathrm{bar}}^2(r)}{r} \Big) \Bigg] \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 5:</b> NFW dark matter profile<br>
                                                <span class="math">\( \rho_{\mathrm{NFW}}(r) = \frac{\rho_s}{\left(\tfrac{r}{r_s}\right)\left(1 + \tfrac{r}{r_s}\right)^2} \)</span> 
                                            </li>
                                            
                                            <li>
                                                <b>Step 6:</b> Global Fit parameters<br>
                                                The structural parameters \( \rho_s, r_s \) and \( \Upsilon \) are pre-calculated via a global \( \chi^2 \) minimization on the entire rotation curve dataset.
                                            </li>
                                            
                                            <li>
                                                <b>Step 7:</b> Enclosed dark matter mass (NFW)<br>
                                                <span class="math">\( M_{\mathrm{DM}}(r) = 4 \pi \rho_s r_s^3 \left[\ln(1+x) - \frac{x}{1+x}\right], \;\; x = \tfrac{r}{r_s} \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 8:</b> Dark matter rotational velocity<br>
                                                <span class="math">\( v_{\mathrm{DM}}(r) = \sqrt{\frac{G M_{\mathrm{DM}}(r)}{r}} \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 9:</b> Total simulated velocity curve (linked to DM slider)<br>
                                                <span class="math">\( v_{\mathrm{tot,sim}}(r) = \sqrt{\tfrac{G \, (M_{\mathrm{bar}}(r) + f\,M_{\mathrm{DM}}(r))}{r}}, \;\; f \in [0,2] \)</span>
                                            </li>
                                            
                                            <li>
                                                <b>Step 10:</b> Fit quality (χ²/d.o.f)<br>
                                                <span class="math">\( \chi^2 = \sum_i \left(\tfrac{v_{\mathrm{obs}}(r_i) - v_{\mathrm{tot,sim}}(r_i)}{\sigma_i}\right)^2 \)</span>
                                            </li>
                                        </ul>
        <h4>Plot Legend</h4>
        <ul>
            <li><b>X-axis:</b> Radius (data)</li>
            <li><b>Y-axis:</b> <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue with grey error bars), <span class="math">\( v_{\mathrm{tot,sim}} \)</span> (green)</li>
        </ul>
    """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the computational steps"')

                               
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
            <p class="text-xs mt-1 italic">Where \( O_i \) is the observed value (blue curve), \( T_i \) is the theoretical value (red curve),at point \( i \) and \( \sigma_i \) is the experimental error from data. </p>
        </div>
        <p class="mb-2">Use the instructions below to compute the &chi;&sup2; minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)).</p>
        
        <ul class="list-disc pl-5 space-y-1">
            <li><b>1a)</b> Choose 3 values of dark matter fraction (f) with the slider above to compute the &chi;&sup2; at that point.</li>
            <li><b>1b)</b> Press <b>'Add &chi;&sup2; point'</b> button to add the point to the &chi;&sup2; plot.</li>
            <li><b>2a)</b> Input 3 values for dark matter mass points (x-coordinates).</li>
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
                            on_click=lambda: [comp_physics_dialog.open(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")
            ]
                        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                       
                        
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_rot.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                        
                    
                        
                        


                    ui.label("Move the slider to add the dark matter to the simulated velocity curve").props('id=alpha_slider_label aria-live=polite tabindex=0')
                    alpha_slider = aria_slider(min=0.0, max=2.0, value=0.0, step=0.01,
                                            aria_label="Dark matter fraction control slider").props('aria-describedby=alpha_slider_label label-always debounce=300')
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
                               
                                try:
                                  
                                    csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                                    df_params = pd.read_csv(csv_path)
                                    row = df_params[df_params['Galaxy'] == gal_name]
                                    
                                    if not row.empty:
                                        rho_s = float(row.iloc[0]['rho_s'])
                                        r_s = float(row.iloc[0]['r_s'])
                                        y_opt = float(row.iloc[0]['Upsilon'])
                                    else:
                                        accessible_notify(f"Error: Parameters for {gal_name} not found in CSV!", type_='error')
                                        gal_state['DATA_LOADED'] = False
                                        return False 
                                except Exception as e:
                                    accessible_notify(f"CSV Read Error: {e}", type_='error')
                                    gal_state['DATA_LOADED'] = False
                                    return False

                                gal_state['base_rho_s'] = rho_s
                                gal_state['base_r_s'] = r_s
                                gal_state['upsilon'] = y_opt
                                
                                gal_state['base_M_dm_grid'] = M_nfw_enclosed(gal_state['r_ngc'], rho_s, r_s)
                                
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
                            alpha_slider.step = round(new_max_gal / 100.0, 2)
                            alpha_slider.value = 0.0 
                            alpha_slider.update()
                            
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
                    x_max_chi = slider_max_gal * 1.05 if slider_max_gal > 0 else 10.0
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
                                galaxy_img_disp = ui.image().classes('w-full h-28 rounded cursor-pointer hover:scale-105 transition-transform').props('role=button tabindex=0 aria-label="Enlarge galaxy image"').on('click', image_dialog2.open).on('keydown.enter', image_dialog2.open)
                                #ui.html("<small class='text-gray-500'><i>Click to enlarge (Ref: ESA Hubble)</i></small>")
                                galaxy_info_content = ui.column().classes('w-full text-sm text-gray-700 gap-0 p-0 m-0')
                            with ui.row().classes('w-full justify-center mt-2 shrink-0'):
                                aria_button(" Table", "table", on_click=table_dialog2.open).classes("!bg-blue-500 text-white font-bold py-1 px-2 rounded mt-auto")

            
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
                z-index: 99999; 
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
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
                flex-direction: column;
                gap: 1rem;
            }
            
            /* --- IL PONTE INVISIBILE PER IL MENU GALASSIE --- */
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
                            
                            y_opt = gal_state.get('upsilon', 1.0)
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
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
                            
                            clean_name = current_galaxy_name.removesuffix('.txt').removesuffix('.csv')
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
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                    plt.title('Galaxy rotation curve', fontsize=12,fontweight='bold')
                                
                                    max_vel_plot = np.nanmax([np.nanmax(v_obs_ngc + v_err_ngc), np.nanmax(v_total_curve), np.nanmax(v_baryonic)])
                                    plt.ylim(0, max(300, 1.1 * max_vel_plot))
                                    plt.xlim(0, r_ngc.max() * 1.1)
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
                            y_opt = gal_state.get('upsilon', 1.0)
    
                          
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
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
                                    plt.plot(r_ngc, m_total_model, "g-", lw=3, label=f"Simulated mass DM")
                                    
                                    
                                   
                                    plt.xlim(x_min_mass, x_max_mass)
                                    plt.ylim(y_min_mass, y_max_mass)

                                    plt.xlabel("Radius (kpc)", fontsize=10)
                                    plt.ylabel("Mass ($M_\\odot$)", fontsize=10)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
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

                        
                                y_opt = gal_state.get('upsilon', 1.0)
    
                              
                                V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
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
                                        galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
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
                            
                            y_opt = gal_state.get('upsilon', 1.0)
                            
                           
                            rho_s, r_s, unscaled_M_dm_grid = get_dm_params(1.0, r_array=r_ngc)
                            max_dm_unscaled = np.nanmax(unscaled_M_dm_grid) if len(unscaled_M_dm_grid) > 0 else 0
                            
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
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
                            y_opt = gal_state.get('upsilon', 1.0)
    
                            V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
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
                                                <b>Step 1:</b> Compute the baryonic velocity from data (conserving the gas rotation sign and applying the Mass-to-Light ratio \(\Upsilon\)):<br>
                                                <span class="math">\( v_{\mathrm{bar}}^2 = v_{\mathrm{gas}}|v_{\mathrm{gas}}| + \Upsilon \left[ v_{\mathrm{disk}}|v_{\mathrm{disk}}| + v_{\mathrm{bulge}}|v_{\mathrm{bulge}}| \right] \)</span>
                                            </li>
                                            
                                            <li style="margin-top: 10px;">
                                                <b>Step 2:</b> Derive the baryonic mass:<br>
                                                <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2}{G} \)</span>
                                            </li>
                                            
                                            <li style="margin-top: 10px;">
                                                <b>Step 3:</b> Compute the total mass from observations:<br>
                                                <span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2}{G} \)</span>
                                            </li>
                                            
                                            <li style="margin-top: 10px;">
                                                <b>Step 4 (Plotting):</b>
                                                <ul style="margin-top:5px; list-style-type: circle;">
                                                    <li><b>X-axis:</b> radius (data)</li>
                                                    <li><b>Y-axis Velocities:</b> \( v_{\mathrm{bar}} \) (red), \( v_{\mathrm{obs}} \) (blue)</li>
                                                    <li><b>Y-axis Masses:</b> \( M_{\mathrm{bar}} \) (red), \( M_{\mathrm{tot}} \) (blue)</li>
                                                </ul>
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
                    vars_options = ['v_bar', 'v_obs', 'rad', 'g', 'm_tot', 'm_bar', 'v_gas', 'v_disk', 'v_bulge']
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
                                        answer_vgas['el'] = aria_select_input(vars_options, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vgas| + ϒ · (').classes('math-text')
                                        answer_vdisk['el'] = aria_select_input(vars_options, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vdisk| + ').classes('math-text')
                                        answer_vbulge['el'] = aria_select_input(vars_options, "...").classes('compact-select').props('dense hide-bottom-space')
                                        ui.label('· |Vbul|) ]').classes('math-text')

                                ui.label("3) Compute Luminous Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">bar</span> = ')
                                    with ui.column().classes('fraction'):
                                        with ui.row().classes('numerator items-center gap-1'):
                                            ui.html('(')
                                            answer_vb['el'] = aria_select_input(vars_options, "Select variable for baryonic velocity")
                                            ui.html(')<span class="math-sup">2</span> &middot; ')
                                            answer_rad['el'] = aria_select_input(vars_options, "Select variable for radius in luminous mass")
                                        with ui.row().classes('denominator w-full justify-center'):
                                            answer_g['el'] = aria_select_input(vars_options, "Select variable for gravitational constant")


                          
                            with ui.column().classes('flex-[3] gap-6 min-w-[350px]'):
                                
                                ui.label("4) Compute Total Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">tot</span> = ')
                                    with ui.column().classes('fraction'):
                                        with ui.row().classes('numerator items-center gap-1'):
                                            ui.html('(')
                                            answer_vo['el'] = aria_select_input(vars_options, "Select variable for observed velocity")
                                            ui.html(')<span class="math-sup">2</span> &middot; ')
                                            answer_rad['el2'] = aria_select_input(vars_options, "Select variable for radius in total mass")
                                        with ui.row().classes('denominator w-full justify-center'):
                                            answer_rad['el2'] = aria_select_input(vars_options, "Select variable for radius in total mass")

                                ui.label("5) Compute Dark Matter Mass").classes('text-blue-200 font-bold text-lg')
                                with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                    ui.html('M<span class="math-sub">DM</span> = ')
                                    answer_Mtot['el'] = aria_select_input(vars_options, "Select variable for total mass")
                                    ui.html('&minus;')
                                    answer_Mbar['el'] = aria_select_input(vars_options, "Select variable for baryonic mass")

                                ui.label("6) Plot Results").classes('text-blue-200 font-bold text-lg')
                                with ui.column().classes('bg-gray-800 p-3 rounded font-mono text-sm text-green-300 w-auto inline-block border border-gray-700'):
                                    ui.html('plt.plot(R, M<sub>bar</sub>, label="Baryonic mass")')
                                    ui.html('plt.plot(R, M<sub>tot</sub>, label="Total mass")')
                                    ui.html('plt.plot(R, V<sub>bar</sub>, label="Baryonic velocity")')
                                    ui.html('plt.plot(R, V<sub>obs</sub>, label="Observed velocity")')
                    show_galaxy_pseudocode()
                    galaxy_select.on('update:model-value', lambda e: show_galaxy_pseudocode.refresh())

                    
                    
                    
                    
                    plots_and_image_container = ui.column().classes('w-full items-center')
                    @ui.refreshable
                    def update_galaxy_mass_analysis():
                        selected_file = galaxy_select.value
                        if not selected_file:
                            plots_and_image_container.clear()
                            return

                       
                        image_filepath = galaxy_file_map.get(selected_file)
                        plots_and_image_container.clear()
                        
                        try:
                                data_ngc = get_galaxy_data_cached(selected_file) 

                                if data_ngc is None:
                                    with plots_and_image_container:
                                        ui.label(f"Error loading {selected_file}").classes("text-red-500")
                                    return
                                r_ngc = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                                v_obs_ngc = pd.to_numeric(data_ngc['Vobs'], errors='coerce').values
                                v_gas_ngc = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values
                                v_disk_ngc = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values
                                v_bul_ngc = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values
                                v_err_ngc = pd.to_numeric(data_ngc['errV'], errors='coerce').values

                                
                               
                                gal_name = os.path.splitext(selected_file)[0]
                                try:
                                    csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                                    df_params = pd.read_csv(csv_path)
                                    row = df_params[df_params['Galaxy'] == gal_name]
                                    if not row.empty:
                                        y_opt = float(row.iloc[0]['Upsilon'])
                                    else:
                                        accessible_notify(f"Error: Parameters for {gal_name} not found in CSV!", type_='error')
                                        return
                                except Exception as e:
                                    accessible_notify(f"CSV Read Error: {e}", type_='error')
                                    return

                                V_bar_sq = v_gas_ngc * np.abs(v_gas_ngc) + y_opt * (v_disk_ngc * np.abs(v_disk_ngc) + v_bul_ngc * np.abs(v_bul_ngc))
                                v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                                m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                                m_total    = (v_obs_ngc**2 * r_ngc) / G_grav
                                with plots_and_image_container:
                                    with ui.row().classes('w-full justify-center gap-4'):
                                        with ui.column().classes('flex-1 items-center'):
                                            with ui.pyplot(figsize=(8, 6)):
                                        
                                                
                                                plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', ms=4, color='blue', ecolor='lightblue', capsize=2, label=f'Observed ({selected_file})', zorder=5)
                                                plt.plot(r_ngc, v_baryonic, color='red', lw=2, label='Baryonic Velocity')
                                                plt.xlabel('Radius (kpc)'); plt.ylabel('Velocity (km/s)'); plt.title(f'{os.path.splitext(selected_file)[0]} Rotation Curve',fontweight='bold')
                                                plt.ylim(0, max(np.nanmax(v_obs_ngc+v_err_ngc), np.nanmax(v_baryonic))*1.1); plt.xlim(0, np.nanmax(r_ngc)*1.1); plt.grid(True); plt.legend()
                                                ui.element('div').props(
    'role=img tabindex=0 aria-label=Rotational velocity curve of the selected galaxy (from data) compared with the baryonic velocity'
    )

                                        with ui.column().classes('flex-1 items-center'):
                                            with ui.pyplot(figsize=(8, 6)):
                                            
                                                plt.plot(r_ngc, m_baryonic/1e9, color='red', lw=2, label='Baryonic Mass')
                                                plt.plot(r_ngc, m_total/1e9,    color='blue', lw=2, label='Total Mass')
                                                plt.xlabel('Radius (kpc)'); plt.ylabel(' Mass ($10^9$ $M_\\odot$)'); plt.title(f'{os.path.splitext(selected_file)[0]} Enclosed Mass',fontweight='bold')
                                                max_m = np.nanmax([np.nanmax(m_total/1e9), np.nanmax(m_baryonic/1e9)])
                                                plt.ylim(0, max_m*1.2 if not np.isnan(max_m) and max_m>0 else 500); plt.xlim(0, np.nanmax(r_ngc)*1.1 if not np.isnan(np.nanmax(r_ngc)) else 50); plt.grid(True); plt.legend()
                                                ui.element('div').props(
    'role=img tabindex=0 aria-label=Total mass and baryonic mass of the selected galaxy (from data)'
    )

                                        
                                    if image_filepath and os.path.exists(image_filepath):
                                        with ui.row().classes('w-full gap-4 justify-start'):
                    
            
                                            with ui.column().classes('flex-1 items-center'):
                                                file_name = os.path.basename(image_filepath)
                                                web_path = f'/galaxy_img/{file_name}'
                                                aria_image(web_path, f"Image of the selected galaxy {selected_file}").classes('w-full max-w-xl')
                                                reference_box("""
    **Image reference:**  
    - [ESA Hubble](https://esahubble.org/)
    """).classes('w-full text-center text-base italic mt-2')

            
                                            with ui.column().classes('flex-1 items-center '):
                                                table_filepath = os.path.join(                    GALAXY_TABLES_PATH,                    os.path.splitext(selected_file)[0] + ".csv"                )
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
                            with plots_and_image_container:
                                warning_box(f"Error processing {selected_file}: {e}").classes('text-red-500')

                    
                    @ui.refreshable
                    def update_plots_popup():
                        
                        global plots_popup_container 
                        plots_popup_container.clear()
                        
                        selected_file = galaxy_select.value
                        if not selected_file:
                            return
                        
                        data_ngc = get_galaxy_data_cached(selected_file)
                       
                        r = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                        vobs = pd.to_numeric(data_ngc['Vobs'], errors='coerce').values
                        verr = pd.to_numeric(data_ngc['errV'], errors='coerce').values
                        vgas = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values
                        vdisk = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values
                        vbul = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values

                        gal_name = os.path.splitext(selected_file)[0]
                        try:
                            csv_path = os.path.join(dataset_path, 'galaxy_best_parameters.csv')
                            df_params = pd.read_csv(csv_path)
                            row = df_params[df_params['Galaxy'] == gal_name]
                            if not row.empty:
                                y_opt = float(row.iloc[0]['Upsilon'])
                            else:
                                accessible_notify(f"Error: Parameters for {gal_name} not found in CSV!", type_='error')
                                return
                        except Exception as e:
                            accessible_notify(f"CSV Read Error: {e}", type_='error')
                            return

                        V_bar_sq = vgas * np.abs(vgas) + y_opt * (vdisk * np.abs(vdisk) + vbul * np.abs(vbul))
                        v_baryonic = np.sqrt(np.maximum(V_bar_sq, 0))
                        m_baryonic = (v_baryonic**2 * r) / G_grav
                        mtot = (vobs**2 * r) / G_grav

                        with plots_popup_container:
                          
                            with ui.row().classes('w-full justify-center gap-8 flex-wrap'):
                                
                                
                                with ui.pyplot(figsize=(6, 5)):
                                    plt.errorbar(r, vobs, yerr=verr, fmt='o', color='blue', ms=4,
                                                ecolor='lightblue', capsize=2, label='Observed')
                                    plt.plot(r, v_baryonic, color='red', lw=2, label='Baryonic')
                                    plt.xlabel("Radius (kpc)"); plt.ylabel("Velocity (km/s)")
                                    plt.title("Rotation Curve", fontweight='bold')
                                    plt.grid(True); plt.legend()

                                
                                with ui.pyplot(figsize=(6, 5)):
                                    plt.plot(r, m_baryonic/1e9, color='red', lw=2, label='Baryonic Mass')
                                    plt.plot(r, mtot/1e9, color='blue', lw=2, label='Total Mass')
                                    plt.xlabel("Radius (kpc)"); plt.ylabel(r"Mass ($10^9$ $M_\odot$)")
                                    plt.title("Enclosed Mass", fontweight='bold')
                                    plt.grid(True); plt.legend()
                    
                 
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
                            el = ans.get('el')
                            if el:
                                el.error = None
                                el.classes(remove='border-red-500 border-green-500')

                        def mark(ans, ok):
                            if ans.get('el'):
                                if ok:
                                    ans['el'].classes(add='border-green-500')
                                else:
                                    ans['el'].classes(add='border-red-500')

                        mark(answer_vb, ok_v1)
                        mark(answer_vo, ok_v2)
                        mark(answer_rad, ok_r)
                        mark(answer_g, ok_g)
                        mark(answer_Mtot, ok_Mtot)
                        mark(answer_Mbar, ok_Mbar)
                        mark(answer_vgas, ok_vgas)
                        mark(answer_vdisk, ok_vdisk)
                        mark(answer_vbulge, ok_vbul)

                        if all([ok_v1, ok_v2, ok_r, ok_g, ok_Mtot, ok_Mbar, ok_vgas, ok_vdisk, ok_vbul]):
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
                        "select": "coma_data.csv", 
                        
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
                    with ui.dialog() as instruction_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label="Cluster simulation instructions" role=dialog'):
                        html_info_box(r"""
    <h3> Cluster Simulation</h3>
    <p>Explore the dynamics of the Coma Cluster and the evidence for Dark Matter using the Virial Theorem.</p>

    <h4>Legend</h4>
    <ul style="margin-bottom: 10px;">
        <li><b>Blue Histogram:</b> Real observed velocity distribution. It is wide, indicating high velocity dispersion (<span class="math">\(\sigma_{obs}\)</span>).</li>
        <li><b>Red Histogram:</b> Prediction based on <b>visible baryonic mass</b> only. It is narrow, implying galaxies should move slowly.</li>
        <li><b>Green Histogram:</b> Simulation that updates in real-time as you add Dark Matter.</li>
    </ul>
<hr class="my-4">

    <h3>Simulating the Cluster Halo</h3>
    <p>The <b>Green Histogram</b> represents the theoretical velocity distribution of galaxies within the cluster. It is constructed using a Gaussian distribution whose width (velocity dispersion, <span class="math">\(\sigma_{sim}\)</span>) depends on the total mass.</p>

    <h4>The Virial Connection</h4>
    <p>We assume the cluster is in virial equilibrium. According to the <b>NFW Dark Matter profile</b> applied to clusters, the velocity dispersion is related to the total mass <span class="math">\(M_{tot}\)</span> and the virial radius <span class="math">\(R_{200}\)</span> by:</p>
    
    <div style="text-align:center; margin: 15px 0;">
        <span class="math">$$ \sigma_{sim} = \sqrt{\frac{G M_{tot}}{3 R_{200}}} $$</span>
    </div>

    <h4>How the Slider Works</h4>
    <p>When you move the slider, you are changing the fraction <span class="math">\(f\)</span> of Dark Matter in the total mass equation:</p>
    <div style="text-align:center; margin: 10px 0;">
        <span class="math">$$ M_{tot} = M_{bar} + f \cdot M_{200} $$</span>
    </div>
    <p>As you increase <span class="math">\(f\)</span>, the total mass <span class="math">\(M_{tot}\)</span> increases. Consequently, the simulated velocity dispersion <span class="math">\(\sigma_{sim}\)</span> increases, causing the green histogram to become <b>wider</b> until it matches the observed blue histogram.</p>
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
        <p style="margin-bottom: 5px;"><b>NFW Profile:</b> Navarro, Frenk & White (1996/97), "A Universal Density Profile from Hierarchical Clustering".</p>
        <p style="margin-bottom: 5px;"><b>Coma Kinematics:</b> Lokas & Mamon (2003), "Dark matter distribution in the Coma cluster from galaxy kinematics".</p>
        <p style="margin-bottom: 5px;"><b>Virial Mass Estimators:</b> Mamon & Łokas (2005), "Dark matter in elliptical galaxies - II. Estimating the mass within the virial radius".</p>
        <p style="margin-bottom: 5px;"><b>Halo Concentrations:</b> Duffy et al. (2008), "Dark matter halo concentrations in the WMAP5 cosmology".</p>
    <p style="margin-bottom: 5px;"><b>Baryon Fractions:</b> Giodini, S. et al. (2009), "Stellar and Total Baryon Mass Fractions in Groups and Clusters Since Redshift 1", The Astrophysical Journal, 703:982-993. </p>
    </div>
""").props('aria-label="Descriptive text about galaxy cluster activity"')
                        aria_button("Close","Close the box", on_click=lambda:instruction_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                  
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

                            <li>Write Newton's equation of dynamics (<i>F = ma</i>) by inserting the relationship between Gravitational Force and Centripetal Force (replace <i>F</i> with the gravitational force and '<i>m*a</i>' with the centripetal force).</li>

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

                            <li>Use the data from the dataset downloadable from the App (radius and average velocities of the galaxies) to calculate the total mass of the cluster and the luminous mass (only stars, dust, gas).</li>

                            <li>Download the cluster dataset from the App (module 2 - Cluster panel - dataset). Analyze the data (velocity RV and radius of each galaxy in the cluster) by selecting a cluster dataset from the Excel spreadsheet.</li>

                            <li>To calculate the Total Mass using the formula <span class="math">\( M = \frac{r \cdot \sigma^2}{G} \)</span>, first perform the following steps. Find the maximum radius of the cluster by applying the Excel formula <code>=MAX(H2:H108)</code>.</li>

                            <li>Calculate the standard deviation of the velocity using the Excel formula <code>=STDEV.P(C2:C108)</code>.</li>

                            <li>To obtain the total mass of the cluster, multiply the mean radius by the squared standard deviation and divide by the constant G <code>=(L2*K2^2)/I2</code>.</li>

                            <li>Calculate the total stellar mass of the cluster by summing the values of each galaxy in the Stellar Mass column of the dataset <code>=SUM(G2:G108)</code>.</li>

                            <li>Calculate the gas mass of the cluster using the pre-defined formula in Excel column <code>=(0.7*L2) * 0.093 * (((0.7*L2) / (200000000000000 / 0.7)) ^ 0.21)</code>, where <code>L2</code> is the total mass of the cluster calculated in the previous step.</li>

                            <li>Calculate the Luminous Mass (stars, gas) by summing the total stellar mass and the gas mass. </li>

                            <li>Compare the results of the luminous mass and total mass. Calculate the ratio: Total Mass / Luminous Mass. What value do you get? Which of the two masses is larger and why?</li>

                            <li>Visualize the total and luminous-only mass and density (Mass/spherical_volume) plots in the App (Cluster Mass & DM panel – 'Open Cluster Plots') after selecting a cluster from the dropdown menu.</li>

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

                        <li>The chi2 value is calculated using the difference squared bewtween the mean observed velocity and the mean simulated velocity, divided by the experimental error.  </li>

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
                
                
                        info_box("**Dataset variables**: objid (galaxy ID),ra (right ascension),dec (declination),modelmag_r (apparent magnitude in r band),modelmagerr_r (magnitude error),extinction_r,redshift (z),zErr (redshift error)")
                                                    
                        reference_box("""**Dataset reference**: [Kaggle:Coma cluster](https://www.kaggle.com/datasets/mertalkan98/coma-cluster) ; [SDSS](https://www.sdss.org/science/data-release-publications)""").classes('text-base italic')
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

            <li><b>Step 1:</b> Velocity of each galaxy from redshift data:<br>

            <span class="math">\( v_i = c \cdot z_i \)</span></li>



            <li><b>Step 2:</b> Number of observed galaxies: <span class="math">\( N \)</span></li>



            <li><b>Step 3:</b> Compute the mean velocity:<br>

            <span class="math">\( \bar{v} = \frac{1}{N} \sum_i v_i \)</span></li>



            <li><b>Step 4:</b> Observed velocity dispersion:<br>

            <span class="math">\( \sigma_{\mathrm{obs}} = \sqrt{ \frac{1}{N} \sum_i (v_i - \bar{v})^2 } \)</span></li>



            <li><b>Step 5:</b> Comoving distance:<br>

            <span class="math">\( \chi(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m (1+z')^3 + (1-\Omega_m)}} \)</span></li>



            <li><b>Step 6:</b> Angular diameter distance:<br>

            <span class="math">\( D_A(z) = \frac{\chi(z)}{1+z} \)</span></li>



            <li><b>Step 7:</b> Define cluster center from BCG or median RA/DEC:<br>

            <span class="math">\( (\mathrm{center\_ra}, \mathrm{center\_dec}) = (ra[idx_{bcg}], dec[idx_{bcg}]) \;\; \mathrm{or} \;\; (\mathrm{median}(ra), \mathrm{median}(dec)) \)</span></li>



            <li><b>Step 8:</b> Angular separation between galaxies:<br>

            <span class="math">\( \theta = \arccos(\sin \delta_1 \sin \delta_2 + \cos \delta_1 \cos \delta_2 \cos(\alpha_1 - \alpha_2)) \)</span></li>



            <li><b>Step 9:</b> Critical density:<br>

            <span class="math">\( \rho_{\mathrm{crit}} = \frac{3 H_0^2}{8 \pi G} \)</span></li>



            <li><b>Step 10:</b> Projected radius:<br>

            <span class="math">\( r_{\mathrm{proj},i} = \max(\theta_i \cdot D_A) \)</span></li>



            <li><b>Step 11:</b> Virial theorem and <span class="math">\( M_{200} \)</span>:<br>

            <span class="math">\( \sigma_{\mathrm{obs}}^2 = \frac{G M_{200}}{3 R_{200}}, \;\; M_{200} = \frac{4}{3} \pi 200 \rho_{\mathrm{crit}} R_{200}^3 \)</span></li>



            <li><b>Step 12:</b> Concentration parameter:<br>

            <span class="math">\( c = A \left(\frac{M_{200}}{M_{\mathrm{pivot}}}\right)^B (1+z)^C \)</span></li>



            <li><b>Step 13:</b> NFW density factor:<br>

            <span class="math">\( \delta_c(c) = \frac{200}{3} \frac{c^3}{\ln(1+c) - c/(1+c)} \)</span></li>



            <li><b>Step 14:</b> Characteristic radius and density:<br>

            <span class="math">\( r_{200} = \left(\frac{3M_{200}}{4 \pi 200 \rho_{\mathrm{crit}}}\right)^{1/3}, \;\; r_s = \frac{r_{200}}{c}, \;\; \rho_s = \delta_c(c)\,\rho_{\mathrm{crit}} \)</span></li>



            <li><b>Step 15:</b> Dark matter mass (NFW):<br>

            <span class="math">\( M_{\mathrm{NFW}}(r_{\mathrm{proj},i}) = M_{200} \cdot \frac{\ln(1+x_i) - x_i/(1+x_i)}{\ln(1+c) - c/(1+c)}, \;\; x_i = \frac{r_{\mathrm{proj},i}}{r_s} \)</span></li>



            <li><b>Step 16:</b> Luminosity distance and distance modulus:<br>

            <span class="math">\( D_L \approx \frac{c \cdot z_{\mathrm{cluster}}}{H_0}, \;\; \mathrm{distmod} = 5 \log_{10}(D_L) - 5 \)</span></li>



            <li><b>Step 17:</b> Magnitude and luminosity:<br>

            <span class="math">\( M_r = m_r - A_r - (5 \log_{10}(D_L/10\,pc)), \;\; L_r = 10^{0.4(M_{r,\odot} - M_r)} \)</span></li>



            <li><b>Step 18:</b> Stellar/baryonic mass:<br>

            <span class="math">\( M_{\mathrm{bar}} = (M/L)\,L_r, \;\; (M/L = 2) \)</span></li>

            <li><b>Step 19:</b> Gas mass (Giodini 2009):<br> <span class="math">\( M_{\mathrm{gas}} = M_{500} \cdot 0.093 \left( \frac{M_{500}}{2 \cdot 10^{14} / h} \right)^{0.21} \)</span> (where <span class="math">\( M_{500} \approx 0.7 \cdot M_{\mathrm{tot}} \)</span>)</li>

    <li><b>Step 20:</b> Total Baryonic Mass:<br> <span class="math">\( M_{\mathrm{lum}} = M_{\mathrm{stars}} + M_{\mathrm{gas}} \)</span></li>





            <li><b>Step 21:</b> Total mass (DM + baryonic, linked to slider):<br>

            <span class="math">\( M_{\mathrm{tot}}(i) = M_{\mathrm{bar}}(i) + f \cdot M_{\mathrm{NFW}}(r_{\mathrm{proj},i}) \)</span></li>



            <li><b>Step 22:</b> Velocity dispersions:

                <ul style="margin-top:5px; list-style-type:circle;">

                    <li>Baryonic: <span class="math">\( \sigma_{\mathrm{bar}}(i) = \sqrt{ \frac{G M_{\mathrm{bar}}(i)}{3 r_{\mathrm{proj},i}} } \)</span></li>

                    <li>Total (simulated): <span class="math">\( \sigma_{\mathrm{sim}}(i) = \sqrt{ \frac{G M_{\mathrm{tot}}(i)}{3 r_{\mathrm{proj},i}} } \)</span></li>

                </ul>

            </li>



            <li><b>Step 21:</b> Plot histograms:

                <ul style="margin-top:5px; list-style-type:circle;">

                    <li>Observed histogram: <span class="math">\( \mathrm{plt.hist}(v_{\mathrm{obs}}, bins) \)</span> (blue)</li>

                    <li>Simulated histogram: <span class="math">\( \mathrm{plt.hist}(\sigma_{\mathrm{tot}}, bins) \)</span> (green)</li>

                </ul>

            </li>

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

                        <p>In the cluster panel, the goodness-of-fit between the observed and simulated velocity distributions is calculated using <b>Pearson's Chi-Square Test for binned data</b>.</p>

                       

                        <ul>

                            <li><b>Step 1: Binning the data</b><br>

                            Instead of comparing individual galaxy velocities point-by-point, we group them into velocity intervals (bins) to form a Gaussian bell curve. We then count how many galaxies fall into each bin.</li>



                            <li><b>Step 2: Pearson's Formula</b><br>

                            <span class="math">\( \chi^2 = \sum_{i=1}^{N_{\mathrm{bins}}} \frac{(O_i - E_i)^2}{\max(E_i, 1.0)} \)</span><br><br>

                            Where:<br>

                            - <span class="math">\( O_i \)</span> = <b>Observed counts</b> (number of real galaxies in bin <i>i</i>).<br>

                            - <span class="math">\( E_i \)</span> = <b>Expected counts</b> (number of simulated galaxies in bin <i>i</i>).

                            </li>



                            <li><b>Why divide by Expected Counts (\( E_i \))? (Poisson Statistics)</b><br>

                            Counting discrete random events (like how many galaxies end up in a specific velocity bin) follows a <b>Poisson distribution</b>. A fundamental property of Poisson statistics is that the variance (the statistical error squared) is equal to the expected value: <span class="math">\( \sigma_{\mathrm{counts}}^2 \approx E_i \)</span>. Therefore, dividing the squared difference by \( E_i \) correctly weights the statistical significance of the error.</li>



                            <li><b>Why don't we use the telescope's measurement error?</b><br>

                            The instrumental error of the telescope (e.g., ±30 km/s) tells us the uncertainty of a single galaxy's speed. However, galaxies inside a massive cluster move with an enormous intrinsic velocity dispersion (often > 1000 km/s). This massive statistical spread completely "swallows" the small instrumental error, making it negligible when comparing the shapes of the overall distributions.</li>



                            <li><b>Step 3: Division by zero protection</b><br>

                            Because our simulated distribution is generated using a random number generator, some bins in the extremes (the tails of the bell curve) might contain exactly 0 galaxies (\( E_i = 0 \)). To prevent the software from crashing due to a division by zero, we enforce a mathematical floor: <span class="math">\( \max(E_i, 1.0) \)</span>.</li>



                            <li><b>Step 4: Reduced Chi-Square</b><br>

                            <span class="math">\( \chi^2_{\mathrm{red}} = \frac{\chi^2}{\mathrm{dof}} \)</span><br>

                            Finally, we divide the total sum by the Degrees of Freedom (dof), which relates to the number of bins. By tracking this reduced value, you can find the exact Dark Matter mass where the simulated curve best matches the real universe!</li>

                        </ul>

                        """).props('role=dialog aria-modal=true aria-label="Chi-Square explanation"')

                                
                                    with ui.tab_panel(t_sim).classes('flex flex-col items-center justify-start'):
                                        ui.label("Velocity Simulation").classes("text-2xl font-bold mb-4 text-slate-800")
                                   
                                        aria_button('Galaxies velocities', 'show cluster galaxies velocity',on_click=on_galaxies_click, icon='visibility').classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                    
                        comp_cluster_dialog2.open()
                    @ui.refreshable
                    def update_cluster_image():
                        
                    
                        select = cluster_state["select"]

                        if select.lower() == "coma_data.csv":
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
                        if select.lower() == "coma_data.csv":
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
                    with ui.dialog() as image_dialog, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl').props('aria-label="Cluster image" role=dialog aria-modal=true'):
                        ui.html("<h5>Cluster Image</h5>")
                        update_cluster_image()
                        aria_button("close",'close',on_click=lambda:image_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as table_dialog, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl').props('aria-label="Cluster information table" role=dialog'):
                        update_cluster_table()
                        aria_button("close",'close',on_click=lambda:table_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    chart_obs = None
                    chart_model = None
                    sim_timer = ui.timer(0.05, lambda: None, active=False)
                    def reset_simulation_data():
                        observed_vel = cluster_state.get('observed_vel', [])
                        if len(observed_vel) == 0: return

                    
                        r_proj_kpc = cluster_state['r_proj_kpc']
                        m_bary_at_gal = cluster_state['m_bary_at_gal']
                        M200 = cluster_state['M200']
                        c = cluster_state['c']
                        rho_crit = cluster_state['rho_crit']
                        
                        N_gal = min(80, len(observed_vel))
                        cluster_radius = 1.0
                        rng = np.random.default_rng() 

                    
                        x_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        x_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)

                        
                        indices = rng.choice(len(observed_vel), size=N_gal, replace=False)
                        observed_vel_sub = observed_vel[indices]
                        sigma_obs_val = np.std(observed_vel_sub)
                        
                        
                        scale_factor = 0.0005
                        vx_obs = rng.normal(0.0, sigma_obs_val, N_gal) * scale_factor
                        vy_obs = rng.normal(0.0, sigma_obs_val, N_gal) * scale_factor

                        
                        r_safe = np.maximum(r_proj_kpc, 1.0)
                        
                        #M_dm_base = Mc_nfw_enclosed(r_safe, M200, c, rho_crit)
                        rho_s = cluster_state['rho_s']
                        r_s = cluster_state['r_s']
                        M_dm_base = M_nfw_enclosed(r_safe, rho_s, r_s)
                        M_bary_base = np.maximum(m_bary_at_gal, 1e-6)
                        
                        
                        dir_x_model = rng.normal(0.0, 1.0, N_gal)
                        dir_y_model = rng.normal(0.0, 1.0, N_gal)

                        cluster_state['sim_state'] = {
                            "x_obs": x_init_obs, "y_obs": y_init_obs,
                            "x_model": x_init_mod, "y_model": y_init_mod,
                            "vx_obs": vx_obs, "vy_obs": vy_obs,
                            "dir_x_model": dir_x_model, "dir_y_model": dir_y_model, 
                            "M_dm_base": M_dm_base, "M_bary_base": M_bary_base,    
                            "r_safe": r_safe,
                            "rng": rng, "indices": indices,
                            "current_dataset": cluster_state.get("select", "")
                        }

                    async def update_cluster_points():
                        try:
                            sim_state = cluster_state.get('sim_state')
                            
                            if not sim_state or sim_state.get("current_dataset") != cluster_state.get("select"):
                                reset_simulation_data()
                                sim_state = cluster_state.get('sim_state')
                                if not sim_state: return

                            
                            cluster_radius = 1.0
                            dt = 0.05
                            VELOCITY_SCALE = 0.0005
                            L = 2 * cluster_radius
                            #TARGET_SLIDER_VALUE = 20.0
                            
                            #raw_slider_value = float(dm_slider.value)
                            #f = raw_slider_value / TARGET_SLIDER_VALUE
                            f= float(dm_slider.value)
                            M_bary = sim_state["M_bary_base"]
                            M_dm = sim_state["M_dm_base"]
                            r_safe = sim_state["r_safe"]
                            indices = sim_state["indices"]
                            
                            
                            M_total_withDM = np.maximum(M_bary + f * M_dm, 1e-6)
                            
                            
                            # sigma^2 ~ G * M / R
                            sigma_sq = (G_grav * M_total_withDM) / (3.0 * r_safe)
                            sigma_local = np.sqrt(np.maximum(0.0, sigma_sq))
                            
                            
                            sigma_model_current = sigma_local[indices].mean()

                            
                            # v = direzione * magnitudine(sigma) * fattore_scala
                            vx_model = sim_state["dir_x_model"] * sigma_model_current * VELOCITY_SCALE
                            vy_model = sim_state["dir_y_model"] * sigma_model_current * VELOCITY_SCALE

                            
                            sim_state["x_obs"] += sim_state["vx_obs"] * dt
                            sim_state["y_obs"] += sim_state["vy_obs"] * dt
                            
                        
                            sim_state["x_model"] += vx_model * dt
                            sim_state["y_model"] += vy_model * dt

                        
                            sim_state["x_obs"] = (sim_state["x_obs"] + cluster_radius) % L - cluster_radius
                            sim_state["y_obs"] = (sim_state["y_obs"] + cluster_radius) % L - cluster_radius
                            sim_state["x_model"] = (sim_state["x_model"] + cluster_radius) % L - cluster_radius
                            sim_state["y_model"] = (sim_state["y_model"] + cluster_radius) % L - cluster_radius

                            
                            data_obs = np.stack((sim_state["x_obs"], sim_state["y_obs"]), axis=1).tolist()
                            data_model = np.stack((sim_state["x_model"], sim_state["y_model"]), axis=1).tolist()

                            chart_obs.options['series'][0]['data'] = data_obs
                            chart_model.options['series'][0]['data'] = data_model
                            
                            chart_obs.update()
                            chart_model.update()

                            
                            observed_vel = cluster_state['observed_vel']
                            sigma_obs_display = np.std(observed_vel[indices])
                            
                    
                            sigma_model_display = sigma_model_current 
                            
                            label_stats_obs.set_text(f"σ_obs: {sigma_obs_display:.0f} km/s")
                            label_stats_model.set_text(f"Simulated σ: {sigma_model_display:.0f} km/s | DM: {f:.2f}")

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
                        DEFAULT_CLUSTER = "coma_data.csv" 
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
                         
                    
                            
                            
                    

                

                    def load_coma_dataset(filename): 
                  
                        data_path = os.path.join(DATA_DIR, "coma_data.csv")
                        df_coma = pd.read_csv(data_path, skiprows=1)
                        df_coma.columns = ['objid','ra','dec','modelmag_r','modelmagerr_r','extinction_r','redshift','zErr']
                        cluster_state['observed_vel'] = c_light * df_coma['redshift'].dropna().values
                        cluster_state['N_obs'] = len(cluster_state['observed_vel'])
                        cluster_state['v_mean_val'] = np.mean(cluster_state['observed_vel'])
                        cluster_state['sigma_obs'] = np.std(cluster_state['observed_vel'])
             

                        z_arr = df_coma['redshift'].dropna().values
                        if len(z_arr) == 0:
                            raise RuntimeError("No redshifts in df")
                        cluster_state['z_cluster'] = np.nanmedian(z_arr)

                        try:
                            idx_bcg = df_coma['modelmag_r'].dropna().idxmin()
                            cluster_state['center_ra'] = float(df_coma.loc[idx_bcg, 'ra'])
                            cluster_state['center_dec'] = float(df_coma.loc[idx_bcg, 'dec'])
                        except Exception:
                            cluster_state['center_ra'] = float(np.nanmedian(df_coma['ra'].dropna().values))
                            cluster_state['center_dec'] = float(np.nanmedian(df_coma['dec'].dropna().values))
                       

                  
                        

                        D_A_kpc = angular_diameter_distance_Mpc(cluster_state['z_cluster']) * 1000.0
                        theta_rad = angsep_rad(df_coma['ra'].values, df_coma['dec'].values, cluster_state['center_ra'], cluster_state['center_dec'])
                        r_proj_kpc = np.maximum(theta_rad * D_A_kpc, 1e-3)
                        cluster_state['r_proj_kpc'] = np.maximum(theta_rad * D_A_kpc, 1e-3)

                        cluster_state['rho_crit'] = rho_crit_Msunkpc3(H0=70.0)
                        cluster_state['m_bary_at_gal'] = stellar_mass_from_r_mag(df_coma['modelmag_r'].values, df_coma['extinction_r'].values, cluster_state['z_cluster'])

                        #M200, R200 = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                        #cluster_state['M200'] = M200
                        #cluster_state['R200'] = R200
                        #cluster_state['c'] = concentration_duffy2008(M200, cluster_state['z_cluster'], h=0.7, relaxed=False)
                        try:
                            csv_path = os.path.join(dataset_path, 'cluster_best_parameters.csv')
                            df_params = pd.read_csv(csv_path)
                            # Usa 'coma_data' in load_coma_dataset, usa gal_name in initialize_cluster_from_df
                            row = df_params[df_params['Cluster'] == 'coma_data'] 
                            if not row.empty:
                                cluster_state['M200'] = float(row.iloc[0]['M200'])
                                cluster_state['c'] = float(row.iloc[0]['c'])
                                cluster_state['r_s'] = float(row.iloc[0]['r_s'])
                                cluster_state['rho_s'] = float(row.iloc[0]['rho_s'])
                                cluster_state['R200'] = cluster_state['r_s'] * cluster_state['c']
                            else:
                                cluster_state['M200'], cluster_state['R200'] = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                                cluster_state['c'] = concentration_duffy2008(cluster_state['M200'], cluster_state['z_cluster'])
                                cluster_state['rho_s'], cluster_state['r_s'], _ = rho_s_from_M200_and_c(cluster_state['M200'], cluster_state['c'], cluster_state['rho_crit'])
                        except Exception as e:
                            print(f"CSV Read Error: {e}")
                            cluster_state['M200'], cluster_state['R200'] = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                            cluster_state['c'] = concentration_duffy2008(cluster_state['M200'], cluster_state['z_cluster'])
                            cluster_state['rho_s'], cluster_state['r_s'], _ = rho_s_from_M200_and_c(cluster_state['M200'], cluster_state['c'], cluster_state['rho_crit'])
                        cluster_state['obs_mean'] = np.mean(cluster_state['observed_vel'])
                        cluster_state['observed_vel_pec'] = cluster_state['observed_vel'] - cluster_state['obs_mean']

                       
                        x_min = cluster_state['observed_vel'].min()
                        x_max = cluster_state['observed_vel'].max()
                        padding = 0.15 * (x_max - x_min)
                        cluster_state['x_min'] = x_min
                        cluster_state['x_max'] = x_max
                        cluster_state['padding'] = padding
                        cluster_state['bins'] = np.linspace(0, x_max + padding, 50)
                       
                        cluster_state['bin_centers'] = 0.5 * (cluster_state['bins'][1:] + cluster_state['bins'][:-1])
                        counts, _ = np.histogram(cluster_state['observed_vel'], bins=cluster_state['bins'])
                        cluster_state['counts_obs'] = counts
                        
                        cluster_state['y_max'] = int(np.ceil(np.max(counts) * 1.15))
                        cluster_state['r_min'] = cluster_state['r_proj_kpc'].min()
                        cluster_state['r_max'] = cluster_state['r_proj_kpc'].max()
                        cluster_state['r_padding'] = 0.1 * (cluster_state['r_max'] - cluster_state['r_min'])
                        
                        cluster_state['v_min_global'] = 0.0
                        cluster_state['ylim_min'] = 0.0
                        cluster_state['ylim_max'] = x_max + 0.2 * abs(x_max)
                        r_min, r_max = r_proj_kpc.min(), r_proj_kpc.max()
                    


                        return df_coma

                   

                    def initialize_cluster_from_df(df):
           
                        cluster_state['observed_vel'] = df["RV"].values
                        cluster_state['N_obs'] = len(cluster_state['observed_vel'])
                        cluster_state['v_mean_val'] = np.mean(cluster_state['observed_vel'])
                        cluster_state['sigma_obs'] = np.std(cluster_state['observed_vel'])

                        cluster_state['z_cluster'] = np.nanmedian(cluster_state['observed_vel'] / c_light)

                        idx_bcg = df["bmag"].idxmin()
                        cluster_state['center_ra'] = float(df.loc[idx_bcg, "RAdeg"])
                        cluster_state['center_dec'] = float(df.loc[idx_bcg, "DEdeg"])

                        D_A_kpc = angular_diameter_distance_Mpc(cluster_state['z_cluster']) * 1000.0
                        theta_rad = angsep_rad(df["RAdeg"], df["DEdeg"], cluster_state['center_ra'], cluster_state['center_dec'])
                        
                        cluster_state['r_proj_kpc'] = np.maximum(theta_rad * D_A_kpc, 1e-3)

                        cluster_state['rho_crit'] = rho_crit_Msunkpc3(H0=70.0)

                        cluster_state['m_bary_at_gal'] = stellar_mass_from_r_mag(
                            df["bmag"].values,
                            0.0,
                            cluster_state['z_cluster']
                        )

                        #M200, R200 = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                        #cluster_state['M200'] = M200
                        ##cluster_state['R200'] = R200

                        #cluster_state['c'] = concentration_duffy2008(M200, cluster_state['z_cluster'], h=0.7, relaxed=False)
                        gal_name = os.path.splitext(cluster_state['select'])[0]
                        try:
                            csv_path = os.path.join(dataset_path, 'cluster_best_parameters.csv')
                            df_params = pd.read_csv(csv_path)
                         
                            row = df_params[df_params['Cluster'] == gal_name] 
                            if not row.empty:
                                cluster_state['M200'] = float(row.iloc[0]['M200'])
                                cluster_state['c'] = float(row.iloc[0]['c'])
                                cluster_state['r_s'] = float(row.iloc[0]['r_s'])
                                cluster_state['rho_s'] = float(row.iloc[0]['rho_s'])
                                cluster_state['R200'] = cluster_state['r_s'] * cluster_state['c']
                            else:
                                cluster_state['M200'], cluster_state['R200'] = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                                cluster_state['c'] = concentration_duffy2008(cluster_state['M200'], cluster_state['z_cluster'])
                                cluster_state['rho_s'], cluster_state['r_s'], _ = rho_s_from_M200_and_c(cluster_state['M200'], cluster_state['c'], cluster_state['rho_crit'])
                        except Exception as e:
                            print(f"CSV Read Error: {e}")
                            cluster_state['M200'], cluster_state['R200'] = estimate_M200_R200_from_sigma(cluster_state['sigma_obs'], cluster_state['rho_crit'])
                            cluster_state['c'] = concentration_duffy2008(cluster_state['M200'], cluster_state['z_cluster'])
                            cluster_state['rho_s'], cluster_state['r_s'], _ = rho_s_from_M200_and_c(cluster_state['M200'], cluster_state['c'], cluster_state['rho_crit'])
                        cluster_state['obs_mean'] = np.mean(cluster_state['observed_vel'])
                        cluster_state['observed_vel_pec'] = cluster_state['observed_vel'] - cluster_state['obs_mean']
        
                        x_min = cluster_state['observed_vel'].min()
                        x_max = cluster_state['observed_vel'].max()
                        padding = 0.15 * (x_max - x_min)
                        cluster_state['x_min'] = x_min
                        cluster_state['x_max'] = x_max
                        cluster_state['padding'] = padding

                        cluster_state['bins'] = np.linspace(0, x_max + padding, 50)
                      
                        cluster_state['bin_centers'] = 0.5 * (cluster_state['bins'][1:] + cluster_state['bins'][:-1])
                        counts, _ = np.histogram(cluster_state['observed_vel'], bins=cluster_state['bins'])
                        cluster_state['counts_obs'] = counts
                        cluster_state['y_max'] = int(np.ceil(counts.max() * 1.3))

                        cluster_state['r_min'] = cluster_state['r_proj_kpc'].min()
                        cluster_state['r_max'] = cluster_state['r_proj_kpc'].max()
                        cluster_state['r_padding'] = 0.1 * (cluster_state['r_max'] - cluster_state['r_min'])

                        
                        cluster_state['dm_slider_min'], cluster_state['dm_slider_max'] = 0.0, 0.10
                        f_max = cluster_state['dm_slider_max']

                        cluster_state['M_bar_tot'] = np.sum(cluster_state['m_bary_at_gal'])
                        cluster_state['R_cluster_bar'] = r200_from_M200(cluster_state['M_bar_tot'], cluster_state['rho_crit'])
                        cluster_state['M_tot_max'] = cluster_state['M_bar_tot'] + f_max * cluster_state['M200']
                        cluster_state['R200_tot'] = r200_from_M200(cluster_state['M_tot_max'], cluster_state['rho_crit'])

                        cluster_state['v_max_expected'] = np.sqrt(G_grav * cluster_state['M_tot_max'] / cluster_state['R200_tot'])
                        cluster_state['v_max_bar'] = np.sqrt(G_grav * cluster_state['M_bar_tot'] / cluster_state['R_cluster_bar'])

                        cluster_state['v_min_global'] = 0.0
                        cluster_state['v_max_global'] = max(
                            cluster_state['observed_vel'].max(),
                            cluster_state['v_max_expected'] * 1.2,
                            cluster_state['v_max_bar'] * 1.5
                        )

                        cluster_state['v_padding'] = 0.1 * (cluster_state['v_max_global'] - cluster_state['v_min_global'])
                        cluster_state['ylim_min'] = 0.0
                        cluster_state['ylim_max'] = cluster_state['observed_vel'].max() + 0.2 * abs(cluster_state['observed_vel'].max())
                    
                      
                        
                    def recompute_axis_limits():
                       

                        x_min = cluster_state['observed_vel'].min()
                        x_max = cluster_state['observed_vel'].max()
                        padding = 0.15 * (x_max - x_min)
                        cluster_state['x_min'] = x_min
                        cluster_state['x_max'] = x_max
                        cluster_state['padding'] = padding

                        cluster_state['y_max'] = cluster_state['counts_obs'].max() * 1.3

                        cluster_state['r_min'] = cluster_state['r_proj_kpc'].min()
                        cluster_state['r_max'] = cluster_state['r_proj_kpc'].max()
                        cluster_state['r_padding'] = 0.1 * (cluster_state['r_max'] - cluster_state['r_min'])
                        
                        cluster_state['v_min_global'] = 0.0
                        cluster_state['ylim_min'] = 0.0
                        cluster_state['ylim_max'] = cluster_state['observed_vel'].max() + 0.2 * abs(cluster_state['observed_vel'].max())
                    def plot_cluster_chi2_curve():
                        cluster_chi2_plot_container.clear()
                        plt.close()
                        with cluster_chi2_plot_container:
                            with ui.pyplot(figsize=(6, 4), close=False, clear=True):
                                ax = plt.gca()
                                chi2_points = cluster_state['chi2_points']
                                all_xs = []
                                all_ys = []
                                
                               
                                M200 = cluster_state.get('M200', 1e14)
                                slider_max_val = getattr(dm_slider, 'max', 10.0) 
                              
                                x_max_chi_fallback = slider_max_val * 1.05 if slider_max_val > 0 else 10.0
                                
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
                        S_max = getattr(dm_slider, 'max', 60.0)
                    
                        
                        
                        observed_vel = cluster_state['observed_vel']
                        rho_crit = cluster_state['rho_crit']
                        r_proj_kpc = cluster_state['r_proj_kpc']
                        m_bary_at_gal = cluster_state['m_bary_at_gal']
                        bins = cluster_state['bins']
                        counts_obs = cluster_state['counts_obs']
                        N_obs = cluster_state['N_obs']

                        r_safe = np.maximum(r_proj_kpc, 1)
                        rho_s = cluster_state['rho_s']
                        r_s = cluster_state['r_s']
                        M_dm_nfw_arr = M_nfw_enclosed(r_safe, rho_s, r_s)
                        
                        h = 0.7
                        m_vir_global_fixed = np.sum(m_bary_at_gal)
                        m_500_global_fixed = 0.7 * m_vir_global_fixed
                        f_gas_global_fixed = 0.093 * ((m_500_global_fixed / (2e14 / h))**0.21)
                        m_gas_global_fixed = m_500_global_fixed * f_gas_global_fixed
                        ratio_stars_gas_fixed = (m_vir_global_fixed + m_gas_global_fixed) / m_vir_global_fixed
                        m_bary_tot_at_gal_fixed = m_bary_at_gal * ratio_stars_gas_fixed
                        
                        M_bar_tot_fixed = np.sum(m_bary_tot_at_gal_fixed)
                        M_DM_csv_tot = np.max(M_dm_nfw_arr)
                        M_tot_csv = M_bar_tot_fixed + M_DM_csv_tot
                        true_dm_ratio = (M_DM_csv_tot / M_tot_csv) if M_tot_csv > 0 else 0.0
                        if cluster_state.get('select', '').lower() == 'coma_data.csv':
                            max_progress = 0.99 / true_dm_ratio if true_dm_ratio > 0 else 1.0
                            progress = float(f / S_max) * max_progress
                        else:
                            progress = float(f / S_max)
                        dm_ratio_display = float(progress * true_dm_ratio)
                        
                        v_center_bar = np.sqrt(np.maximum(0.0, G_grav * M_bar_tot_fixed / (r200_from_M200(M_bar_tot_fixed, rho_crit) + 1e-12)))
                        v_mean_obs = np.mean(observed_vel)
                        if v_mean_obs > v_center_bar:
                            v_center_tot = v_center_bar + (v_mean_obs - v_center_bar) * progress
                        else:
                            v_center_tot = v_mean_obs

                        sigma_fit = np.std(observed_vel)
                        counts_model = N_obs * (norm.cdf(bins[1:], loc=v_center_tot, scale=sigma_fit) - norm.cdf(bins[:-1], loc=v_center_tot, scale=sigma_fit))

                        chi2_hist = np.sum(((counts_obs - counts_model)**2) / np.maximum(counts_model, 1.0))
                        chi2_val = chi2_hist / max(1, len(counts_obs))

                        is_duplicate = any(abs(x - dm_ratio_display) < 1e-4 for x, y in cluster_state['chi2_points'])
                        if is_duplicate:
                            cluster_state['chi2_slider_result'] = "Point already added! Move the slider."
                            plot_cluster_chi2_curve()
                            return  
                            
                        cluster_state['chi2_points'].append((dm_ratio_display, chi2_val))
                        
                        if len(cluster_state['chi2_points']) > 3: cluster_state['chi2_points'].pop(0)
                      
                        if len(cluster_state['chi2_points']) == 3:
                            xs, ys = zip(*cluster_state['chi2_points'])
                            coeffs = np.polyfit(xs, ys, 2)
                            
                        
                            if coeffs[0] > 1e-5:
                                xmin = -coeffs[1] / (2 * coeffs[0])
                                ymin = np.polyval(coeffs, xmin) 
                                
                              
                                #ymin_display = max(0.0, ymin)
                             
                                if xmin < 0:
                                    cluster_state['chi2_slider_result'] = "Invalid Result: Ratio cannot be negative!"
                                elif xmin >= 1.0:
                                    cluster_state['chi2_slider_result'] = "Invalid Result: Ratio cannot exceed 100%!"
                                elif ymin < 0:
                                  
                                    cluster_state['chi2_slider_result'] = f"Invalid Fit: χ² min < 0 (≈ {ymin:.2f})"
                                else:
                                    cluster_state['chi2_slider_result'] = f"Best Fit: DM/Tot ≈ {xmin*100:.1f}%, χ²_min ≈ {ymin:.2f}"
                            else:    
                               
                                cluster_state['chi2_slider_result'] = "Invalid shape. Select 3 points that surrond the minimum!"

                        plot_cluster_chi2_curve()

                    
                        
                    def refresh_cluster_chi2_plot():
                        cluster_state['chi2_points'].clear()
                        cluster_state['chi2_slider_result'] = "---"
                        plot_cluster_chi2_curve()

                   
                  

                    with ui.column().classes('w-full items-center '):
                        dm_slider_min, dm_slider_max = 0.0, 60
                        ui.label("Move the slider to add dark matter to the simulated velocity distribution in the cluster").props('id=dm_slider_label')
                        dm_slider = aria_slider(min=dm_slider_min, max=dm_slider_max,
                        value=0.0, step=0.01,  aria_label="Dark matter fraction control slider").props('aria-describedby=dm_slider_label label-always debounce=300')

                        
                        
                        
                        with ui.row().classes('w-full items-stretch justify-between no-wrap gap-2 px-2'):
                    
                      
                            with ui.column().classes('w-[260px] flex-shrink-0 p-2 bg-blue-50 rounded-lg border border-blue-200 shadow-sm flex flex-col justify-between overflow-hidden'):
                            
                                with ui.column().classes('w-full gap-1 p-0 m-0'):
                                    cluster_title_label = ui.label("Cluster Info: ---").classes("text-xl font-bold text-blue-800 mb-2 whitespace-nowrap")
                                    cluster_img_display = ui.image().classes('w-full h-28 rounded cursor-pointer hover:scale-105 transition-transform').props('role=button tabindex=0 aria-label="Enlarge cluster image"').on('click', image_dialog.open).on('keydown.enter', image_dialog.open)
                                    cluster_info_content = ui.column().classes('w-full text-sm text-gray-700 gap-0 p-0 m-0')
                                
                           
                                with ui.row().classes('w-full justify-center mt-2 shrink-0'):
                                    aria_button(" Table", "table", on_click=table_dialog.open).classes("!bg-blue-500 text-white font-bold py-1 px-2 rounded ")
                          
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
                z-index: 99999; 
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
            }
            
            /* --- LA MAGIA: IL PONTE INVISIBILE --- */
            .sigma-audio-dropdown-content::after {
                content: "";
                position: absolute;
                top: 100%;  /* Si attacca sotto al pannello nero */
                left: 0;
                width: 100%;
                height: 1rem; /* Copre interamente il buco fino al bottone verde */
                background: transparent;
            }
            /* ------------------------------------ */

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
                    
                
                        
                    if DEFAULT_CLUSTER.lower() == "coma_data.csv":
                        df = load_coma_dataset(DEFAULT_CLUSTER)
                        cluster_name = DEFAULT_CLUSTER
                        #select = dataset_selector.value
                
                    else:
                        initialize_cluster_from_df(df)
                    recompute_axis_limits()
                    
                    dm_slider.value = 0.0

                   
                    
                    
                    ui.timer(0.5, lambda: refresh_cluster_plots(full_anim=True), once=True)

                
                    def on_dataset_change(new_value):
                        if not isinstance(new_value, str):
                            return
                        
                        cluster_state["select"] = new_value
                        
                        refresh_cluster_chi2_plot()
                        if new_value.lower() == "coma_data.csv":
                            df = load_coma_dataset(new_value)
                            new_max_clu = 60.0
                        else:
                            df = get_cluster_data_cached(new_value)
                            initialize_cluster_from_df(df)
                            new_max_clu = 10.0
                        #new_max_clu = 0.99
                        recompute_axis_limits()
                        # cluster_name = new_value
                        
                        #dm_slider.value = 0.0
                        sigma_obs = cluster_state['sigma_obs']
                        
                       
                        #new_max_clu = max(5.0, (sigma_obs / 100.0) * 3.0)
                      
                        
                 
                        dm_slider.max = round(new_max_clu, 1)
                        dm_slider.step = round(new_max_clu / 100.0, 2)
                        dm_slider.value = 0.0
                        dm_slider.props(f'max="{new_max_clu}" step="{round(new_max_clu / 100.0, 2)}"')
                        dm_slider.update()
                        
                 
                        cluster_state['chi2_points'].clear()
                        cluster_state['chi2_slider_result'] = "Add points to find the minimum."
                        refresh_cluster_plots(full_anim=True)
                        update_cluster_image.refresh()
                        update_cluster_table.refresh()
                        
                     
                        observed_vel = cluster_state['observed_vel']
                        r_proj_kpc = cluster_state['r_proj_kpc']
                        m_bary_at_gal = cluster_state['m_bary_at_gal']

                        N_gal = min(80, len(observed_vel))
                        cluster_radius = 1.0
                        rng = np.random.default_rng(123)

                        x_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_obs = rng.uniform(-cluster_radius, cluster_radius, N_gal)

                        x_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)
                        y_init_mod = rng.uniform(-cluster_radius, cluster_radius, N_gal)

                        r_safe = np.maximum(r_proj_kpc, 1.0)
                        M_total_bary = np.maximum(m_bary_at_gal, 1e-6)
                        sigma_bary = np.sqrt(G_grav * M_total_bary / (3.0 * r_safe))
                        
                        indices = rng.choice(len(observed_vel), size=N_gal, replace=False)
                        
                        sigma_bary_sub = sigma_bary[indices]
                        observed_vel_sub = observed_vel[indices]
                        sigma_obs_val = np.std(observed_vel_sub)
                        
                        vx_obs = rng.normal(0.0, sigma_obs_val, N_gal)
                        vy_obs = rng.normal(0.0, sigma_obs_val, N_gal)

                        vx_model = rng.normal(0.0, sigma_bary_sub, N_gal)
                        vy_model = rng.normal(0.0, sigma_bary_sub, N_gal)

                   
                        cluster_state["sim_state"] = {
                            "x_obs": x_init_obs.copy(),
                            "y_obs": y_init_obs.copy(),
                            "x_model": x_init_mod.copy(),
                            "y_model": y_init_mod.copy(),
                            "vx_obs": vx_obs.copy(),
                            "vy_obs": vy_obs.copy(),
                            "vx_model": vx_model.copy(),
                            "vy_model": vy_model.copy(),
                            "rng": rng,
                            "indices": indices,
                            "current_dataset": new_value
                        }
                    

                    #dm_slider.on_value_change(lambda: refresh_cluster_plots(full_anim=False))
                    dm_slider.on('change', lambda e: refresh_cluster_plots(full_anim=False))
                    #dm_slider.on('update:model-value', refresh_cluster_plots.refresh)
                    #dataset_selector.on('update:model-value',  on_dataset_change)
                    dataset_selector.on_value_change(lambda e: on_dataset_change(e.value))
                



                
                            
                    async def update_coma_histogram_v2(token=None,full_anim=True):
                        
                    
                        if token is None: token = plot_state['id']
                        if token != plot_state['id']: return 
                        observed_vel = cluster_state['observed_vel']
                        r_proj_kpc = cluster_state['r_proj_kpc']
                        m_bary_at_gal = cluster_state['m_bary_at_gal']
                        M200 = cluster_state['M200']
                        c = cluster_state['c']
                        rho_crit = cluster_state['rho_crit']
                        obs_mean = cluster_state['obs_mean']
                        bins = cluster_state['bins']
                        counts_obs = cluster_state['counts_obs']
                        N_obs = cluster_state['N_obs']
                        r_max = cluster_state['r_max']
                        r_padding = cluster_state['r_padding']
                        ylim_max = cluster_state['ylim_max']
                        x_max = cluster_state['x_max']
                        padding = cluster_state['padding']
                        y_max = cluster_state['y_max']
                        try:
        
                            f = float(dm_slider.value) 
                            S_max = getattr(dm_slider, 'max', 60.0)
                            #progress = float(np.clip(f / S_max, 0.0, 1.0))
                            
                            
                            r_safe = np.maximum(r_proj_kpc, 1) 
                            rho_s = cluster_state['rho_s']
                            r_s = cluster_state['r_s']
                            M_dm_at_gal = M_nfw_enclosed(r_safe, rho_s, r_s)

                            h = 0.7
                            m_vir_global_fixed = np.sum(m_bary_at_gal)
                            m_500_global_fixed = 0.7 * m_vir_global_fixed
                            f_gas_global_fixed = 0.093 * ((m_500_global_fixed / (2e14 / h))**0.21)
                            m_gas_global_fixed = m_500_global_fixed * f_gas_global_fixed
                            ratio_stars_gas_fixed = (m_vir_global_fixed + m_gas_global_fixed) / m_vir_global_fixed
                            
                            m_bary_tot_at_gal_fixed = m_bary_at_gal * ratio_stars_gas_fixed
                            
                            # Calcolo proporzione reale
                            M_bar_tot_fixed = np.sum(m_bary_tot_at_gal_fixed)
                            M_DM_csv_tot = np.max(M_dm_at_gal)
                            M_tot_csv = M_bar_tot_fixed + M_DM_csv_tot
                            true_dm_ratio = (M_DM_csv_tot / M_tot_csv) if M_tot_csv > 0 else 0.0
                        
                            if cluster_state.get('select', '').lower() == 'coma_data.csv':
                                max_progress = 0.99 / true_dm_ratio if true_dm_ratio > 0 else 1.0
                                progress = float(f / S_max) * max_progress
                            else:
                                progress = float(f / S_max)
                            dm_ratio_display = progress * true_dm_ratio
                            
                            M_total_at_gal = np.maximum(m_bary_tot_at_gal_fixed + progress * M_dm_at_gal, 1e-6)
                        
                            sigma_los_loc = np.sqrt(np.maximum(0.0, G_grav * M_total_at_gal / (3.0 * r_safe))) 
                            sigma_fit = np.std(observed_vel) 
                            
                           
                            v_center_bar = np.sqrt(np.maximum(0.0, G_grav * M_bar_tot_fixed / (r200_from_M200(M_bar_tot_fixed, rho_crit) + 1e-12)))
                            v_mean_obs = np.mean(observed_vel)
                            
                            if v_mean_obs > v_center_bar:
                                v_center_tot = v_center_bar + (v_mean_obs - v_center_bar) * progress
                            else:
                                v_center_tot = v_mean_obs

                            rng = np.random.default_rng(seed=42)
                            v_bar = rng.normal(loc=v_center_bar, scale=sigma_fit, size=len(r_proj_kpc))
                            v_dm  = rng.normal(loc=v_center_tot, scale=sigma_fit, size=len(r_proj_kpc))
                            
                            sigma_bar = np.sqrt(np.maximum(0.0, G_grav * m_bary_tot_at_gal_fixed / (3.0 * r_safe)))
                            
                            obs_sub = observed_vel[::max(1, len(observed_vel)//20)]
                            bar_sub = sigma_bar[::max(1, len(sigma_bar)//20)] 
                            sim_sub = sigma_los_loc[::max(1, len(sigma_los_loc)//20)]

                            sigma_obs_series = [float(abs(v - obs_mean)) for v in obs_sub if np.isfinite(v)]
                            sigma_bar_series = [float(s) for s in bar_sub if np.isfinite(s)]
                            sigma_sim_list = [float(s) for s in sim_sub if np.isfinite(s)]


                            sigma_obs_val = float(np.std(observed_vel))
                            sigma_bar_val = float(np.nanmean(sigma_bar))
                            sigma_sim_val = float(np.nanmean(sigma_los_loc))
                            min_len = min(len(sigma_obs_series), len(sigma_sim_list))
                            sigma_diff_series = np.abs(np.array(sigma_obs_series[:min_len]) - np.array(sigma_sim_list[:min_len]))
                            sigma_diff_series_list = sigma_diff_series.tolist()
                            sigma_diff_mean = np.mean(sigma_diff_series)

                            ui.run_javascript(f"window.sigmaObsSeries = {sigma_obs_series};")
                            ui.run_javascript(f"window.sigmaBarSeries = {sigma_bar_series};")
                            ui.run_javascript(f"window.sigmaSimSeries = {sigma_sim_list};")
                            ui.run_javascript(f"window.sigmaObsVal = {sigma_obs_val};")
                            ui.run_javascript(f"window.sigmaBarVal = {sigma_bar_val};")
                            ui.run_javascript(f"window.sigmaSimVal = {sigma_sim_val};")
                            ui.run_javascript(f"window.sigmaDiffSeries = {sigma_diff_series_list};")
                            ui.run_javascript(f"window.sigmaDiffMean = {sigma_diff_mean};") 


                    



                            
                            
                            counts_model, _ = np.histogram(v_dm, bins=bins)
                            if counts_model.sum() > 0:
                                counts_model = counts_model / counts_model.sum() * N_obs
                
                        
                        
                        
                            #chi2_hist = np.sum(((np.log1p(counts_obs) - np.log1p(counts_model))**2))
                            chi2_hist = np.sum(((counts_obs - counts_model)**2) / np.maximum(counts_model, 1.0))
                            #chi2_hist = np.sum(((counts_obs - counts_model)**2) / np.maximum(counts_model, 1.0)**2)
                            chi2_norm = chi2_hist / len(counts_obs)
                            
                            
                            v_mean_val = np.mean(observed_vel)
                            sigma_obs = np.std(observed_vel)
                            v_model_mean = np.mean(v_dm)
                            v_model_sigma = np.std(v_dm)
                            peak_obs = counts_obs.max()
                            peak_model = counts_model.max()





                            
                            sigma_bar_val = sigma_bar.mean()
                            sigma_dm_val = sigma_los_loc.mean()

                            N_sim = len(v_dm) 
                            N_bar=len(v_bar)
                            M_dm_mean = np.mean(f * M_dm_at_gal)
                            dm_fraction = np.mean((f * M_dm_at_gal) / M_total_at_gal)



                        



                            #mass_ratio = M_tot / M_bar_tot_fixed
                            counts_dm, _ = np.histogram(v_dm, bins=bins)
                            counts_dm = counts_dm / counts_dm.sum() * N_obs
                            chi2_scatter = np.sum(((np.log1p(counts_obs) - np.log1p(counts_dm))**2))

                            chi2_scatter_norm = chi2_scatter / len(counts_obs)


                          
                       
                            M_DM_current = f * M200

                         
                            #dm_ratio = M_DM_current / M_tot if M_tot > 0 else 0.0

                            # Aggiorna slider ed etichette
                            dm_slider.props(f'label-value="DM / Mₜₒₜ: {dm_ratio_display * 100:.1f}%"')
                            
                            select_name = cluster_state.get("select", "coma_data.csv")
                            clean_name = select_name.removesuffix('.csv').removesuffix('.txt')
                            cluster_title_label.set_text(f"Cluster Info: {clean_name}")
                          
                            cluster_info_content.clear()
                            with cluster_info_content:
                                displayed_M_tot = M_bar_tot_fixed + progress * M_DM_csv_tot
                                displayed_M_DM = progress * M_DM_csv_tot
                                
                                html_info_box(f"""
                                    <ul class="list-none pl-2 space-y-1 text-sm text-gray-800">
                                        <li><b>M<sub>tot</sub>:</b> {format_sci(displayed_M_tot)} M<sub>☉</sub></li>
                                        <li><b>M<sub>DM</sub>:</b> {format_sci(displayed_M_DM)} M<sub>☉</sub> <span style="color: #0284c7; font-weight: bold;">({dm_ratio_display*100:.1f}%)</span></li>
                                        <li><b>M<sub>bar</sub>:</b> {format_sci(M_bar_tot_fixed)} M<sub>☉</sub></li>
                                        <li><b>σ<sub>obs</sub>:</b> {sigma_obs:.1f} km/s</li>
                                        <li><b>σ<sub>sim</sub>:</b> {sigma_los_loc.mean():.1f} km/s</li>
                                    </ul>
                                """)
                            select_name = cluster_state["select"]
                            img_name = "coma_img.jpg" if select_name.lower() == "coma_data.csv" else os.path.splitext(select_name)[0] + ".jpg"
                            if os.path.exists(os.path.join(CLUSTER_IMG_PATH, img_name)):
                                cluster_img_display.source = f"/cluster_img/{img_name}"
                            else:
                                cluster_img_display.source = ""
                            sort_idx = np.argsort(r_proj_kpc)
        
                            r_sorted = r_proj_kpc[sort_idx]
                            obs_vel_sorted = observed_vel[sort_idx]
                            v_bar_sorted = v_bar[sort_idx]
                            v_dm_sorted = v_dm[sort_idx]

                      
                            n_chunks = 10     
                            chunk_size = int(np.ceil(len(r_sorted) / n_chunks))
                            sleep_time = 0.15

                            def render_frame(current_obs, current_bar, current_dm, 
                                            curr_r_obs, curr_r_bar, curr_r_dm):
                                try:
                                    with plot_container_histo:
                                        plot_container_histo.clear()
                                        plt.close('all')
                                        with ui.pyplot(figsize=(6, 4)):
                                            ax = plt.gca()
                                            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                          
                                            plt.hist(current_obs, bins=bins, alpha=1.0,color='blue', orientation='horizontal',
                        label='Observed Velocities', 
                        rasterized=True)
                
                                            if len(current_dm) > 0:
                                        
                                                plt.hist(current_dm, bins=bins, alpha=0.7, color='green', orientation='horizontal',
                                                        label='Simulated Velocities' , 
                                                        rasterized=True)
                                            
                                            if len(current_bar) > 0:
                                            
                                                plt.hist(current_bar, bins=bins, alpha=0.6, color='red', orientation='horizontal',
                                                        label='Baryonic Component', 
                                                        rasterized=True)
                                                
                                            
                                           
                                            plt.xlim(0, float(y_max))
                                            #plt.yscale('log')
                                            plt.ylim(0, float(x_max + padding))
                                            plt.ylabel('Velocity [km/s]',fontsize=10)
                                            plt.xlabel('N° of Galaxies',fontsize=10)
                                            #if cluster_name is not None:
                                            #    plt.title(f'Galaxy Velocity Distribution ({cluster_name})')
                                            #else:
                                            #    plt.title('Galaxy Velocity Distribution (Cluster)')
                                            plt.title('Histogram of velocities distribution ',fontsize=12,fontweight='bold')
                                            plt.legend(fontsize=10, loc='upper left')
                                            plt.tight_layout()
                                            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                                            ui.element('div').props(
            'role=status aria-live=polite tabindex=0 aria-label=Histogram showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
            )

                                

                                    with plot_container_scatter:
                                        plot_container_scatter.clear()
                                        #plt.close()
                                        with ui.pyplot(figsize=(6, 4)):
                                    
                                        
                                            ax = plt.gca()
                                            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                                            ax.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                                            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                                            plt.scatter(curr_r_obs, current_obs, s=10, color='blue', alpha=0.6, label="Observed Galaxies", rasterized=True)

                                            plt.scatter(curr_r_bar, current_bar, s=10, color='red', alpha=0.6, 
                            label='Baryonic Component'   , 
                            rasterized=True)
                

                                            plt.scatter(curr_r_dm, current_dm, s=10, color='green', alpha=0.6, 
                            label="Simulated Galaxies" , 
                            rasterized=True)

                                    
                                            
                                            

                                            plt.xlim(0, r_max + r_padding)
                                            plt.ylim(0, ylim_max)


                                            plt.xlabel("Radius [kpc]",fontsize=10)
                                            plt.ylabel("Velocity [km/s]",fontsize=10)
                                            #if cluster_name is not None:
                                            #    plt.title(f"Phase-space diagram of cluster galaxies ({cluster_name})")
                                            #else:
                                            #    plt.title("Phase-space diagram of cluster galaxies")
                                            plt.title("Cluster galaxies velocities",fontsize=12,fontweight='bold')
                                            plt.legend(fontsize=10, loc='upper left')
                                            plt.grid(True, linestyle="--", alpha=0.7)
                                            
                                            
                                            
                                            ui.element('div').props(
            'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
            )
                               
                                except Exception:
                                       
                                        return
                            if full_anim:
                             
                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']: return 
                                    limit = i * chunk_size
                                    render_frame(obs_vel_sorted[:limit], [], [], 
                                                 r_sorted[:limit], [], [])
                                    await asyncio.sleep(sleep_time)

                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']: return 
                                    limit = i * chunk_size
                                    render_frame(obs_vel_sorted, v_bar_sorted[:limit], [], 
                                                 r_sorted, r_sorted[:limit], [])
                                    await asyncio.sleep(sleep_time)
                            
                          
                                for i in range(1, n_chunks + 1):
                                    if token != plot_state['id']: return 
                                    limit = i * chunk_size
                                    render_frame(obs_vel_sorted, v_bar_sorted, v_dm_sorted[:limit], 
                                                r_sorted, r_sorted, r_sorted[:limit])
                                
                                    pause = sleep_time if full_anim else 0.01 
                                    await asyncio.sleep(pause)
                        
                            else:
                               
                                render_frame(obs_vel_sorted, v_bar_sorted, v_dm_sorted, r_sorted, r_sorted, r_sorted)

                        except Exception:
                           
                            return
                       
                        

                 
                    #ui.timer(dt, update_cluster_points)



                    

                def handle_tab_change(e):
                      
                    if e.value == 'cluster':
                          
                        ui.timer(0.1, lambda: refresh_cluster_plots(full_anim=True), once=True)

                   
                tabs.on_value_change(handle_tab_change)
                
                
#panel cluster exercise
                with ui.tab_panel('clusdm').props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg"):
                    ui.label("Explore how virial theorem reveals dark matter in galaxy clusters. ").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                    with ui.dialog() as inst_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label=Instructions for cluster mass and density activities'):
                        html_info_box(r"""
        <h3>Cluster Mass Analysis</h3>
        <p>Imagine analyzing a <b>galaxy cluster</b> to reveal dark matter.</p>
        <ul>
            <li>Choose a dataset and complete the missing fields.</li>
            <li>Click <b>Run Analysis</b> to compare <b>luminous mass</b> vs <b>total (virial) mass</b>.</li>
            <li>Review the additional information provided in the "Additional Information: Numerical Explanation" PDF. This document offers a detailed numerical explanation of the concepts related to the dark matter.</li>
        </ul>
    """).props('aria-label="Descriptive text about galaxy cluster mass and density activities"')
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

                        

                        H0 = 70.0
                        G = 4.302e-6
                        M_sun_B = 5.48
                        MLR_B = 5.0
                        FIXED_DISTANCE_MPC = None

                        members = df.copy()
                        # sigma clipping
                        for _ in range(5):
                            v_rel = members["RV"] - np.median(members["RV"])
                            mad = np.median(np.abs(v_rel))
                            sigma = 1.4826 * mad if mad > 0 else np.std(v_rel)
                            mask = np.abs(v_rel) <= 3 * sigma
                            new_members = members[mask]
                            if len(new_members) == len(members):
                                break
                            members = new_members

                        if len(members) < 10:
                            ui.label(f"Not enough valid galaxies in {selected_file}.").classes('text-orange-500')
                            return

                        cluster_distance = float(FIXED_DISTANCE_MPC) if FIXED_DISTANCE_MPC else np.median(members["RV"]) / H0

                        mean_ra, mean_dec = members["RAdeg"].mean(), members["DEdeg"].mean()
                        ang_sep = np.sqrt(((members["RAdeg"] - mean_ra) * np.cos(np.radians(mean_dec)))**2 +
                                        (members["DEdeg"] - mean_dec)**2)
                        r_kpc = cluster_distance * 1000 * np.radians(ang_sep)
                        members = members.assign(r_kpc=r_kpc).replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc"])
                        members = members[members["r_kpc"] <= 3000]
                        r = members["r_kpc"].values

                        D_pc = cluster_distance * 1e6
                        dist_mod = 5*np.log10(D_pc) - 5
                        M_abs = members["bmag"] - dist_mod
                        L_B = 10**(-0.4*(M_abs - M_sun_B))
                        L_B_np = L_B.values

                        order = np.argsort(r)
                        R_cum = r[order]
                        L_cum = np.cumsum(L_B_np[order])
                        M_lum_r = MLR_B * L_cum

                        sigma_global = np.std(members["RV"] - np.median(members["RV"]))
                        M_tot_r = (3.0 * sigma_global**2 * R_cum) / G
                        h = 0.7  
                        m_500_r = 0.7 * M_tot_r
                        f_gas_r = 0.093 * ((m_500_r / (2e14 / h))**0.21)
                        m_gas_r = m_500_r * f_gas_r
                        M_baryonic_r = M_lum_r + m_gas_r
                        # positive floor to avoid zeros
                        def positive_floor(arr):
                            pos = arr[np.isfinite(arr) & (arr > 0)]
                            if pos.size == 0:
                                return arr + 1e-6
                            floor = np.nanmin(pos) * 1e-3
                            return np.where(arr <= 0, floor, arr)

                        M_tot_r = positive_floor(M_tot_r)
                        M_baryonic_r = positive_floor(M_baryonic_r)

                        rho_lum_cum = M_baryonic_r / ((4.0/3.0) * np.pi * R_cum**3)
                        rho_tot_cum = M_tot_r / ((4.0/3.0) * np.pi * R_cum**3)

                        with cluster_popup_container:
                            with ui.row().classes('w-full justify-center gap-8 flex-wrap'):
                                with ui.pyplot(figsize=(8, 6)):
                                    plt.plot(R_cum, M_baryonic_r/1e9, label="Luminous Mass", color="red", lw=2)
                                    plt.plot(R_cum, M_tot_r/1e9, label="Total Mass", color="blue", lw=2)
                                    plt.xscale("log"); plt.yscale("log")
                                    plt.xlabel("Radius (kpc)"); plt.ylabel("Mass (10^9 M☉)")
                                    plt.title(f"Mass Profile — {selected_file}",fontweight='bold')
                                    plt.grid(True, which="both", ls="--")
                                    plt.legend()

                                with ui.pyplot(figsize=(8, 6)):
                                    plt.plot(R_cum, rho_lum_cum, label="Luminous Density", color="red")
                                    plt.plot(R_cum, rho_tot_cum, label="Total Density", color="blue")
                                    plt.xscale("log"); plt.yscale("log")
                                    plt.xlabel("Radius (kpc)"); plt.ylabel("Density (M☉/kpc³)")
                                    plt.title(f"Density Profile — {selected_file}",fontweight='bold')
                                    plt.grid(True, which="both", ls="--")
                                    plt.legend()
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

            <li><b>Step 1:</b> Compute angular separation (radians):<br> <span class="math">\( \Delta \theta \)</span></li>

            <li><b>Step 2:</b> Compute radius (<span class="math">\( D = 100 \,\mathrm{Mpc} \)</span>):<br> <span class="math">\( r = D \cdot \Delta \theta \)</span> (kpc)</li>

            <li><b>Step 3:</b> Compute observed velocity:<br> <span class="math">\( v_i = c \, z_i \)</span></li>

            <li><b>Step 4:</b> Distance modulus:<br> <span class="math">\( \text{distmod} = 5 \log_{10}(D_{\mathrm{pc}}) - 5 \)</span></li>

            <li><b>Step 5:</b> Absolute magnitude:<br> <span class="math">\( M_{\mathrm{abs}} = b_{\mathrm{mag}} - \text{distmod} \)</span></li>

            <li><b>Step 6:</b> Luminosity (<span class="math">\( M_{\odot,B} = 5.48 \)</span>):<br> <span class="math">\( L_B = 10^{-0.4 (M_{\mathrm{abs}} - M_{\odot,B})} \)</span></li>

            <li><b>Step 7:</b> Total cluster luminosity:<br> <span class="math">\( L_r = \sum_i L_{B,i} \)</span></li>

            <li><b>Step 8:</b> Stars mass (<span class="math">\( M/L = 5 \)</span>):<br> <span class="math">\( M_{\mathrm{stars}} = (M/L) \cdot L_r \)</span></li>

           

<li><b>Step 9:</b> Gas mass (Giodini,2009):<br> <span class="math">\( M_{\mathrm{gas}} = M_{500} \cdot 0.093 \left( \frac{M_{500}}{2 \cdot 10^{14} / h} \right)^{0.21} \)</span> (where <span class="math">\( M_{500} \approx 0.7 \cdot M_{\mathrm{tot}} \)</span>)</li>

    <li><b>Step 10:</b> Total Baryonic Mass:<br> <span class="math">\( M_{\mathrm{lum}} = M_{\mathrm{stars}} + M_{\mathrm{gas}} \)</span></li>

 </ul>



        <h4>Virial Theorem Setup</h4>

       

        <ul>

            <li><b>Step 9:</b> Gravitational potential energy: <span class="math">\( U(r) = - \frac{G M m}{r} \)</span></li>

            <li><b>Step 10:</b> Gravitational force: <span class="math">\( F_{\mathrm{grav}} = \frac{G M m}{r^2} \)</span></li>

            <li><b>Step 11:</b> Centripetal force: <span class="math">\( F_{\mathrm{cent}} = \frac{m v^2}{r} \)</span></li>

            <li><b>Step 12:</b> Equating forces: <span class="math">\( v^2(r) = \frac{G M}{r} \)</span></li>

            <li><b>Step 13:</b> Kinetic energy: <span class="math">\( K = \tfrac{1}{2} m v^2 = \tfrac{1}{2} \frac{G M m}{r} \)</span></li>

            <li><b>Step 14:</b> Relation K vs U: <span class="math">\( K = -\tfrac{1}{2} U \)</span></li>

            <li><b>Step 15:</b> Virial theorem: <span class="math">\( 2K + U = 0 \)</span></li>

            <li><b>Step 16:</b> System treated as <span class="math">\( N \)</span> light particles orbiting a heavy center of mass.</li>

        </ul>



        <h4>Cluster Mass & Density</h4>

        <ul>

            <li><b>Step 17:</b> 1-D velocity dispersion: <span class="math">\( \sigma_v = \sqrt{\tfrac{1}{N} \sum_{i=0}^N (v_i - \bar{v})^2} \)</span></li>

            <li><b>Step 18:</b> 3-D velocity dispersion (isotropic): <span class="math">\( \sigma_{3D} = \sqrt{3} \, \sigma_v \)</span></li>

            <li><b>Step 19:</b> Total cluster mass: <span class="math">\( M_{\mathrm{tot}} = \frac{3 \, \sigma_v^2 \, r}{G} \)</span></li>

            <li><b>Step 20:</b> Cluster volume: <span class="math">\( V = \tfrac{4}{3} \pi r^3 \)</span></li>

            <li><b>Step 21:</b> Luminous density: <span class="math">\( \rho_{\mathrm{lum}} = \frac{M_{\mathrm{lum}}}{V} \)</span></li>

            <li><b>Step 22:</b> Total density: <span class="math">\( \rho_{\mathrm{tot}} = \frac{M_{\mathrm{tot}}}{V} \)</span></li>

            <li><b>Step 23 (Plot):</b> X-axis: radius. Y-axis: <span class="math">\( M_{\mathrm{lum}}, M_{\mathrm{tot}}, \rho_{\mathrm{lum}}, \rho_{\mathrm{tot}} \)</span></li>

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

            <li><b>H<sub>0</sub></b> = 70 km/s/Mpc (Hubble constant)</li>

            <li><b>G</b> = 4.302 &times; 10<sup>-6</sup> kpc&middot;(km/s)&sup2;&middot;M<sub>&odot;</sub><sup>-1</sup></li>

            <li><b>c</b> = 299792 km/s (Speed of light)</li>

            <li><b>U</b> = Gravitational potential energy of a particle [km&sup2;/s&sup2;]</li>

            <li><b>F<sub>grav</sub></b> = Gravitational force [M<sub>&odot;</sub>&middot;kpc/s&sup2;]</li>

            <li><b>F<sub>cent</sub></b> = Centripetal force [M<sub>&odot;</sub>&middot;kpc/s&sup2;]</li>

            <li><b>v&sup2;(R)</b> = Squared orbital velocity for circular orbit [km&sup2;/s&sup2;]</li>

            <li><b>K</b> = Kinetic energy of a particle [km&sup2;/s&sup2;]</li>

            <li><b>2K + U = 0</b> = Virial theorem for bound system</li>

            <li><b>M<sub>circ</sub></b> = Mass corresponding to circular orbit of one particle [M<sub>&odot;</sub>]</li>

            <li><b>M<sub>tot</sub>(r)</b> = Total virial mass of the cluster [M<sub>&odot;</sub>]</li>

            <li><b>L<sub>B</sub></b> = Luminosity of a single galaxy in B-band [L<sub>&odot;</sub>]</li>

            <li><b>L(r)</b> = Cumulative luminosity up to radius r [L<sub>&odot;</sub>]</li>

            <li><b>M<sub>lum</sub>(r)</b> = Luminous mass within radius r [M<sub>&odot;</sub>]</li>

            <li><b>D<sub>pc</sub></b> = Cluster distance in parsecs [pc]</li>

            <li><b>dist<sub>mod</sub></b> = Distance modulus</li>

            <li><b>M<sub>abs</sub></b> = Absolute magnitude of a galaxy [mag]</li>

            <li><b>m</b> = Mass of a light single particle or galaxy [M<sub>&odot;</sub>]</li>

            <li><b>M</b> = Mass inside radius r of heavy particle or total enclosed mass [M<sub>&odot;</sub>]</li>

            <li><b>M<sub>circ</sub></b> = Mass of a particle in circular orbit [M<sub>&odot;</sub>]</li>

            <li><b>v&sup2;</b> = Squared orbital velocity [km&sup2;/s&sup2;]</li>

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
    'M_abs', 'L_B', 'L(r)', 'd*1e6', 'd_pc', 'bmag', 'rv'
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
                                        ui.label("1) Orbital dynamics (single light particle around heavy one)").classes('text-blue-200 font-bold text-lg')

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
                                        ui.label("4) Compute the centripetal force").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('F<span class="math-sub">cent</span> = ')
                                            ans_Fc['el'] = aria_select_input(vars_options, "Select variable for centripetal force")

                                        # 5) EQUATE
                                        ui.label("5) Equate forces: F_grav = F_cent → derive v²(R)").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('v<span class="math-sup">2</span>(R) = ')
                                            ans_v2['el'] = aria_select_input(vars_options, "Select variable for squared velocity")

                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 6) KINETIC
                                        ui.label("6) Kinetic energy of the light particle").classes('text-blue-200 font-bold text-lg')
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
                                        ui.label("8) Compute circular orbit mass of one particle").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">circ</span>(r) = ')
                                            ans_Mcirc['el'] = aria_select_input(vars_options, "Select formula for circular mass")

                                        # 9) TOTAL MASS SYSTEM
                                        ui.label("9) Compute the total mass for a system of N particles:").classes('text-blue-200 font-bold text-lg')
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

                                        # 18) PLOT
                                        ui.label("18) Compare mass and density profiles").classes('text-blue-200 font-bold text-lg')
                                        with ui.column().classes('bg-gray-800 p-2 rounded font-mono text-md text-green-300 w-auto inline-block border border-gray-700'):
                                            ui.html('plot(r, M<sub>lum</sub> and M<sub>total</sub>, label="Luminous and Total Mass")')
                                            ui.html('plot(r, D<sub>lum</sub>=M<sub>lum</sub>/V and D<sub>tot</sub>=M<sub>tot</sub>/V, label="Luminous and Total Density")')
                                                        
                        show_cluster_pseudocode()
                        cluster_select.on('update:model-value', lambda e: show_cluster_pseudocode.refresh())

                        
                        cluster_plot_container = ui.row().classes('w-full gap-4 justify-center')
                        @ui.refreshable
                        def update_cluster_mass_profile():
                            selected_file = cluster_select.value
                            if not selected_file:
                                cluster_plot_container.clear()
                                return

                       
                            image_filepath = cluster_file_map.get(selected_file)
                            
                            cluster_plot_container.clear()

                            with cluster_plot_container:
                                try:
                                    
                                    df = get_cluster_data_cached(selected_file)

                                    H0 = 70.0
                                    G = 4.302e-6
                                    M_sun_B = 5.48
                                    MLR_B = 5.0
                                    FIXED_DISTANCE_MPC = None  

                                
                                    members = df.copy()
                                    for _ in range(5):
                                        v_rel = members["RV"] - np.median(members["RV"])
                                        mad = np.median(np.abs(v_rel))
                                        sigma = 1.4826 * mad if mad > 0 else np.std(v_rel)
                                        mask = np.abs(v_rel) <= 3 * sigma
                                        new_members = members[mask]
                                        if len(new_members) == len(members):
                                            break
                                        members = new_members

                                    if len(members) < 10:
                                        ui.label(f"Not enough valid galaxies in {selected_file}.").classes('text-orange-500')
                                        return

                            
                                    if FIXED_DISTANCE_MPC is not None:
                                        cluster_distance = float(FIXED_DISTANCE_MPC)
                                    else:
                                        cluster_distance = np.median(members["RV"]) / H0

                                
                                    mean_ra, mean_dec = members["RAdeg"].mean(), members["DEdeg"].mean()
                                    ang_sep = np.sqrt(((members["RAdeg"] - mean_ra) * np.cos(np.radians(mean_dec)))**2 +
                                                    (members["DEdeg"] - mean_dec)**2)
                                    r_kpc = cluster_distance * 1000 * np.radians(ang_sep)
                                    members = members.assign(r_kpc=r_kpc).replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc"])
                                    members = members[members["r_kpc"] <= 3000]
                                    r = members["r_kpc"].values

                                
                                    D_pc = cluster_distance * 1e6
                                    dist_mod = 5*np.log10(D_pc) - 5
                                    M_abs = members["bmag"] - dist_mod
                                    L_B = 10**(-0.4*(M_abs - M_sun_B))
                                    L_B_np = L_B.values

                                
                                    order = np.argsort(r)
                                    R_cum = r[order]
                                    L_cum = np.cumsum(L_B_np[order])
                                    M_lum_r = MLR_B * L_cum

                                
                                    rmin = max(10.0, np.percentile(r, 2))
                                    rmax = np.percentile(r, 98)
                                    if rmax <= rmin:
                                        rmin = max(1.0, np.min(r)*0.9)
                                        rmax = np.max(r)*1.1
                                    n_bins = min(25, max(10, len(r)//8))
                                    edges = np.geomspace(rmin, rmax, n_bins + 1)
                                    R_mid = 0.5 * (edges[:-1] + edges[1:])

                                    sigma_global = np.std(members["RV"] - np.median(members["RV"]))
                                    M_tot_r = (3.0 * sigma_global**2 * R_cum) / G 
                                    h = 0.7  
                                    m_500_r = 0.7 * M_tot_r
                                    f_gas_r = 0.093 * ((m_500_r / (2e14 / h))**0.21)
                                    m_gas_r = m_500_r * f_gas_r
                                    M_baryonic_r = M_lum_r + m_gas_r

                                    def positive_floor(arr):
                                        pos = arr[np.isfinite(arr) & (arr > 0)]
                                        if pos.size == 0:
                                            return arr + 1e-6
                                        floor = np.nanmin(pos) * 1e-3
                                        return np.where(arr <= 0, floor, arr)

                                    M_tot_r = positive_floor(M_tot_r)
                                    M_baryonic_r = positive_floor(M_baryonic_r)

                                    with ui.column().classes('flex-1 items-center'):  
                                        with ui.pyplot(figsize=(8,6)):
                                    
                                            mask_lum = M_baryonic_r > 0
                                            plt.plot(R_cum[mask_lum], M_baryonic_r[mask_lum], label='Luminous Mass', color='red')
                                            plt.plot(R_cum, M_tot_r, label='Total Mass (Virial)', color='blue')
                                            plt.xscale('log'); plt.yscale('log')
                                            plt.xlabel('Radius (kpc)'); plt.ylabel('Mass ($M_\\odot$)')
                                            plt.title(f'Mass Profile {os.path.splitext(selected_file)[0]}',fontweight='bold')
                                            plt.grid(True, which="both", ls="--"); plt.legend()
                                            ui.element('div').props(
        'role=img tabindex=0 aria-label=Plot showing the mass profile (in solar mass) in function to the radius (in kpc) of the selected galaxy cluster, comparing luminous mass and total mass'
    )
                                
                                        rho_lum_cum = M_baryonic_r / ((4.0/3.0) * np.pi * R_cum**3)
                                        rho_tot_cum = M_tot_r / ((4.0/3.0) * np.pi * R_cum**3)
                                    with ui.column().classes('flex-1 items-center'):
                                        with ui.pyplot(figsize=(8,6)):
                                        
                                            plt.plot(R_cum, rho_lum_cum, label=' Luminous Density', color='red')
                                            plt.plot(R_cum, rho_tot_cum, label=' Total Density', color='blue')
                                            plt.xscale('log'); plt.yscale('log')
                                            plt.xlabel('Radius (kpc)')
                                            plt.ylabel('Density ($M_\\odot$ / kpc³)')
                                            plt.title(f'Density Profile {os.path.splitext(selected_file)[0]}',fontweight='bold')
                                            plt.grid(True, which="both", ls="--")
                                            plt.legend()
                                            ui.element('div').props(
        'role=img tabindex=0 aria-label=Plot showing the mass density profile in function to the radius of the selected galaxy cluster, comparing luminous mass and total mass'
    )


                                    with ui.row().classes('w-full gap-4 justify-start'):

    
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
    """).classes('w-full text-center text-base  italic mt-2')
                                            else:
                                                ui.label("Image not found").classes('text-red-500')

            
                                        with ui.column().classes('flex-1 items-center '):
                                            table_filepath = os.path.join(
                    CLUSTER_TABLES_PATH,
                    os.path.splitext(selected_file)[0] + ".csv"
                )
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
                                    warning_box(f"Error processing {selected_file}: {e}").classes('text-red-500')
                                    
                                    
                                    
                        


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


                
            






    



