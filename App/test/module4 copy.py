

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
import asyncio           
import logging
from layout import *
from core import *
def create_page():
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


    rho_crit = 2.775e2
    G_grav = 4.30091e-6   
    c_light = 3e5         
    M_sun_r = 4.64  

    #function for galaxy
    def c_of_M200(M200, scale=0.3):
        return scale * (11.7 * (M200 / 1e11)**(-0.075))



    def observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul):
    
        aD = np.where(r>0, Vdisk**2 / r, 0.0)
        aHI = np.where(r>0, Vgas**2 / r, 0.0)
        q = r**2 * ((np.where(r>0, Vobs**2 / r, 0.0)) - aD - aHI)
        dqdr = np.gradient(q, r)
        return dqdr / (4.0*np.pi*G_grav*r**2)

    def get_rhos_rs_from_observed_matching(r, Vobs, Vgas, Vdisk, Vbul, r_match, scale=0.3):
        if r.size == 0 or Vobs.size == 0:
            raise ValueError("Array r o Vobs vuoti! Controlla il caricamento della galassia.")
        rho_obs_arr = observed_rho_from_RC(r, Vobs, Vgas, Vdisk, Vbul)
        rho_obs_at_match = np.interp(r_match, r, rho_obs_arr)

        def to_solve(logM):
            M = 10.0**logM
            R200 = (3.0*M/(4.0*np.pi*200.0*rho_crit))**(1/3)
            c = c_of_M200(M, scale=scale)
            r_s = R200 / c
            rho_s = M / (4*np.pi*r_s**3 * (np.log(1+c) - c/(1+c)))
            x = r_match / r_s
            rho_r = rho_s / (x*(1+x)**2)
            return np.log10(rho_r) - np.log10(rho_obs_at_match)

        logM_sol = brentq(to_solve, 10, 13)
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


    @ui.page('/module4')
    def module4():
        #ui.add_head_html('<script src="https://cdn.tailwindcss.com"></script>')
        
        ui.add_head_html('''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    ''')
        ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
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
    background-color: #2563eb !important; /* sfondo blu acceso per l’attivo */
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
        ui.add_head_html("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.39/Tone.min.js"></script>
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
        
        ui.add_head_html('''
    <script>
    window.MathJax = {
        tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]},
        svg: {fontCache: 'global'}
    };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    ''')
        


        def set_clean_style():
            # Resetta e imposta uno stile base chiaro
            plt.style.use('default') 
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': '#FAFAFA', # Grigio chiarissimo per l'area del grafico
                'axes.grid': True,
                'grid.color': '#DDDDDD',     # Griglia leggera
                'grid.linestyle': '--',
                'grid.alpha': 0.7,
                'axes.edgecolor': '#333333',
                'axes.linewidth': 1.2,
                'font.size': 11,
                'lines.linewidth': 2.5,      # Linee più corpose
                'figure.autolayout': True    # Gestione automatica spazi
            })

        set_clean_style()

        main_layout("Module 4: Dark Matter ")
        

        with ui.column().classes('w-full p-4 gap-4'):
            with ui.tabs().classes('w-full') as tabs:
                zero,one, two, three, four,five = ui.tab('Kepler laws planets').props('role=tab aria-selected=true'), ui.tab('Galaxy rotation curve').props('role=tab aria-selected=false'), ui.tab('Galaxy mass & DM').props('role=tab aria-selected=false'), ui.tab('Cluster velocity distribution').props('role=tab aria-selected=false'), ui.tab('Cluster mass & DM').props('role=tab aria-selected=false'),ui.tab('CMB').props('role=tab aria-selected=false')
        

            with ui.tab_panels(tabs, value=zero).classes('w-full'):
                
                
                with ui.tab_panel(zero).props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Kepler's Laws and Planetary Data")
                    with ui.dialog() as info_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Kepler's Laws Exploration</h3>
        <p>Explore the orbital characteristics of planets in our solar system using Kepler's Laws.</p>
        <ul>
            <li><b>First Plot:</b> Shows the orbital velocity vs semi-major axis for each planet, illustrating Kepler's Second Law.</li>
            <li><b>Second Plot:</b> Displays the orbital period versus semi-major axis, demonstrating Kepler's Third Law.</li>
            <li><b>Third Plot:</b> Presents the mass of each planet vs radius.</li>
        </ul>
    """).props('role=dialog aria-modal=true aria-label="Descriptive text about Kepler Laws activity"')

                        aria_button("Close", "close the box",on_click=info_kepler.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.dialog() as data_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        info_box( "**Dataset variables**: Celestial_Body (name of the planet), SemiMajorAxis(km) (orbital radius in km), Velocity(km/s) (orbital velocity in km/s), Period(days) (orbital period in days), Mass(kg) (mass of the planet in kg).")
                        reference_box(
        """**Dataset reference**: [NASA-SSD](https://ssd.jpl.nasa.gov/planets/) ;[Orbital Mechanics](https://orbital-mechanics.space/reference/planetary-parameters) ;Ryan S. Park, William M. Folkner, James G. Williams, and Dale H. Boggs. The JPL Planetary and Lunar Ephemerides DE440 and DE441. The Astronomical Journal, 161(3):105, February 2021.; Brandon Rhodes. Skyfield: high precision research-grade positions for planets and Earth satellites generator. July 2019""")
                        aria_button("Close", "close the box",on_click=data_kepler.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                
                    G = 6.67430e-20            # km^3 / kg / s^2 (unità coerenti con km, kg)
                    M_sun = 1.98842e30         # kg
                    app.add_static_files('/planet_images', PLANETS_IMG_PATH)

                
                    file_path = os.path.join(DATA_DIR, 'Planets.txt')   # usa il tuo DATA_DIR
                    

                
                    df0 = pd.read_csv(file_path, sep=r'\s+', engine='python')
                
                    df0 = df0[~df0['Celestial_Body'].isin(['Sun', 'Moon'])].reset_index(drop=True)

                
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

                
                        ui.html(html).props('role=region tabindex=0 aria-label=Planet information panel')

                    def open_slides():
                        ui.run_javascript('window.open("/slides/cosmo_dark_matter.pdf", "_blank")')
                    kepler_3_plot_ref = None
                    mass_plot_ref = None
                
                    with ui.dialog() as kepler_3_dialog, ui.card().classes('p-4 w-full max-w-[900px]'):
                        ui.label("Kepler III: Period vs Semi-major axis").classes("text-lg font-bold").props('aria-label=kepler III law plot role=heading aria-level=3 tabindex=0')
                        kepler_3_plot_ref = plot_kepler_III(None)
                        kepler_3_plot_ref.classes('w-full')
                        aria_button("Close","close", on_click=kepler_3_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.dialog() as mass_plot_dialog, ui.card().classes('p-4 w-full max-w-[900px]'):
                        ui.label("Mass vs Semi-major axis").classes("text-lg font-bold").props('aria-label=kepler mass plot role=heading aria-level=3 tabindex=0')
                        mass_plot_ref = plot_mass(None)
                        mass_plot_ref.classes('w-full') 
                        aria_button("Close","close", on_click=mass_plot_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        
                    with ui.row().classes('w-full gap-4 justify-center'):

                        
                       
                        aria_button("Instruction", "Instruction for Kepler panel",on_click=safe_click(lambda: [info_kepler.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Dataset", "Info Kepler planets dataset",on_click=safe_click(lambda: [data_kepler.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button('Explore the Kepler Laws Phet Simulator',"Explore the interactive Kepler laws simulator",
        on_click=safe_click(lambda: ui.run_javascript("window.open('https://phet.colorado.edu/sims/html/keplers-laws/latest/keplers-laws_all.html', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        

                        
                        aria_button('Kepler III law plot  ', "Kepler III law plot (Period vs semi-major axis)",on_click=kepler_3_dialog.open).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            
                        
                        aria_button('Planets Mass plot ','Mass Plot (M vs a)', on_click=mass_plot_dialog.open).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        aria_button('Open Introduction Presentation', 'Open Introduction Presentation',on_click=open_slides).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        
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
                            
                            
                    
                    selector.on('update:model-value', lambda e: (
                        plot_velocity.refresh(selector.value),
                    
                        info_panel.refresh(selector.value)
                    ))


                        
                                

                with ui.tab_panel(one).props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Rotation Curve")
                    DATA_LOADED=False
                    global galaxy_file_map, selected_file, image_filepath
                    G_grav = 4.30091e-6   # kpc (km/s)^2 / M_sun
                
                    r_ngc = np.array([])
                    v_obs_ngc = np.array([])
                    v_gas_ngc = np.array([])
                    v_disk_ngc = np.array([])
                    v_bul_ngc = np.array([])
                    v_err_ngc = np.array([])
                    r_match = None
                    current_galaxy_name = None
                    default_galaxy = "NGC3198.txt"
                    
                    galaxy_state = {
        "select": default_galaxy
    }
                    chi2_points = []
                    manual_points = []
                    
                    formula_pts = {"a": None, "b": None, "c": None, "fa": None, "fb": None, "fc": None}
                    
                    parabolic_state = {
                            "points": [],
                            "history": [],
                            "plot_points": [],
                            "iteration": 0
                        }
                    
                
                
                    
                    with ui.dialog() as info_galaxy, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>NGC3198 Rotation Curve</h3>
        <p>Explore the rotation curve of the NGC3198 galaxy.</p>
        <ul>
            <li><b>Red curve:</b> Shows the <b>baryonic prediction</b> with a Keplerian-like trend (initial rise followed by a decrease).</li>
            <li><b>Green curve:</b> Simulates the addition of a <b>dark matter halo</b> to flatten the curve and match observations.</li>
            <li><b>Slider Control:</b> 0 = total baryonic curve, 1 = total curve with full dark matter halo contribution.</li>
        </ul>
    """)
                        aria_button("close",'close',on_click=info_galaxy.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as data_galaxy,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    
                        info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                    
                        reference_box(
        """**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')
                        aria_button("close",'close',on_click=data_galaxy.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
    

                    
                    
                    with ui.row().classes('w-full '):
                        with ui.dialog() as velocity_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                            html_info_box(r"""
        <h3>Computational Notes</h3>
        
        <ul>
            <li>
                <b>Step 1:</b> Baryonic velocity (from data)<br>
                <span class="math">\( v_{\mathrm{bar}}^2(r) = v_{\mathrm{gas}}^2(r) + v_{\mathrm{disk}}^2(r) + v_{\mathrm{bulge}}^2(r) \)</span>
            </li>
            
            <li>
                <b>Step 2:</b> Baryonic mass<br>
                <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2(r)}{G} \)</span>
            </li>
            
            <li>
                <b>Step 3:</b> Total observed mass from velocity data<br>
                <span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2(r)}{G} \)</span>
            </li>
            
            <li>
                <b>Step 4:</b> Observed dark matter density (from data)<br>
                <span class="math">\( \rho_{\mathrm{DM}}(r) = \frac{1}{4 \pi G r^2} \, \frac{d}{dr} \Bigg[ r^2 \Big( \frac{v_{\mathrm{obs}}^2(r)}{r} - \frac{v_{\mathrm{gas}}^2(r)}{r} - \frac{v_{\mathrm{disk}}^2(r)}{r} - \frac{v_{\mathrm{bulge}}^2(r)}{r} \Big) \Bigg] \)</span>
            </li>
            
            <li>
                <b>Step 5:</b> NFW dark matter profile<br>
                <span class="math">\( \rho_{\mathrm{NFW}}(r) = \frac{\rho_s}{\left(\tfrac{r}{r_s}\right)\left(1 + \tfrac{r}{r_s}\right)^2} \)</span> <br>
                with <span class="math">\( \alpha=1, \beta=3, \gamma=1 \)</span>, <span class="math">\( R_{200} = \left[\tfrac{3 M_{200}}{4\pi \cdot 200 \rho_{\mathrm{crit}}}\right]^{1/3}, \;\; r_s = \tfrac{R_{200}}{c} \)</span>, <br>
                <span class="math">\( c = 0.3 \times 11.7 \left(\tfrac{M_{200}}{10^{11} M_\odot}\right)^{-0.075} \)</span>
            </li>
            
            <li>
                <b>Step 6:</b> Match observed density with NFW<br>
                <span class="math">\( \rho_{\mathrm{NFW}}(r_{\mathrm{match}}) = \rho_{\mathrm{DM}}(r_{\mathrm{match}}) \)</span> → Obtain <span class="math">\( M_{200}, \rho_s, r_s \)</span>
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
                <span class="math">\( v_{\mathrm{tot,sim}}(r) = \sqrt{\tfrac{G \, (M_{\mathrm{bar}}(r) + f\,M_{\mathrm{DM}}(r))}{r}}, \;\; f \in [0,1] \)</span>
            </li>
            
            <li>
                <b>Step 10:</b> Fit quality (χ²/d.o.f)<br>
                <span class="math">\( \chi^2 = \sum_i \left(\tfrac{v_{\mathrm{obs}}(r_i) - v_{\mathrm{tot,sim}}(r_i)}{\sigma_i}\right)^2, \;\; \chi^2_{\mathrm{dof}} = \tfrac{\chi^2}{N_{\mathrm{obs}} - N_{\mathrm{params}}} \)</span>
            </li>
        </ul>

        <h4>Plot Legend</h4>
        <ul>
            <li><b>X-axis:</b> Radius (data)</li>
            <li><b>Y-axis:</b> <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue with grey error bars), <span class="math">\( v_{\mathrm{tot,sim}} \)</span> (green)</li>
        </ul>
    """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the computational steps"')

                            aria_button("Close", "close the box",on_click=velocity_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    @ui.refreshable
                    def update_image():
                        select = galaxy_state["select"]
                        image_filename = os.path.splitext(select)[0] + ".jpg"
                        image_path = os.path.join(GALAXY_IMG_PATH, image_filename)

                    
                        if os.path.exists(image_path):
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
                        select = galaxy_state["select"]
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
                            
                    with ui.dialog() as image_dialog2, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl'):
                        ui.html("<h5>Galaxy Image</h5>")
                        update_image()
                        aria_button("close",'close',on_click=image_dialog2.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as table_dialog2, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl'):
                        update_table()
                        aria_button("close",'close',on_click=table_dialog2.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                    with ui.dialog() as morpho, ui.card().classes('w-full max-w-xl mx-auto h-auto '):
                        morph_plot_container = ui.column().classes("w-full h-auto")
                        aria_button('close','close', on_click=lambda:morpho.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as chi2_info_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <style>
            .formula-box { background: rgba(255,255,255,0.9); padding: 15px; border: 1px solid #0284c7; border-radius: 8px; color: #111; margin-top: 15px; overflow-x: auto; }
            .math-frac { display: inline-flex; flex-direction: column; text-align: center; vertical-align: middle; margin: 0 4px; }
            .math-num { border-bottom: 1px solid black; padding-bottom: 2px; }
            .math-den { padding-top: 2px; }
        </style>

        <h3 class="text-xl font-bold mb-2">Parabolic interpolation and compute &chi;&sup2; minimization</h3>
        <p class="mb-2">Use the formula below to compute the &chi;&sup2; minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)).</p>
        
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
                        aria_button("close",'close',on_click=chi2_info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
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
                        ).classes('w-1/2 max-w-md').props('id=galaxy_selector aria-label=Galaxy dataset selector role=listbox tabindex=0')
                       
                        aria_button("Galaxy panel Instruction","Instruction for galaxy panel",on_click=safe_click(lambda: [info_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Dataset info", "Info galaxy dataset",on_click=safe_click(lambda: [data_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [velocity_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
            )
                        aria_button("Image", "Show galaxy image",on_click=lambda: image_dialog2.open()).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Table", "Show galaxy data table",on_click=lambda: table_dialog2.open()).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Info chi2", "Read detailed information about chi2 ", on_click=chi2_info_dialog.open).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Plot galaxy ", "Plot galaxy ", 
                                            on_click=lambda: morpho.open()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        
                    
                    with ui.row().classes('w-full items-center justify-center gap-4'):
                    
                        ui.html('''
<div class="flex space-x-2">
<button id="audio-button" role="button" aria-label=" or deactivate audio" aria-pressed="false" onclick="initAudio()" 
            class="!bg-blue-500 hover:!bg-gray-700 text-white font-bold py-1 px-2 rounded">
        Activate Audio
    </button>''')
                        ui.label("").props(
            'id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0'
        )
                        ui.html('''
        <div class="flex space-x-2">
        <button role="button" aria-label="Play observed velocity curve mean" onclick="playObservedVelMean()" class="!bg-blue-500 hover:!bg-blue-700 text-white px-2 py-1 rounded">▶ Observed Vel (mean)</button>
        <button role="button" aria-label="Play observed velocity curve" onclick="playObservedVelCurve()" class="!bg-blue-300 hover:!bg-blue-500 text-white px-2 py-1 rounded">▶ Observed Vel (curve)</button>
        <button role="button" aria-label="Play baryonic velocity curve mean" onclick="playBaryonicVelMean()" class="!bg-red-500 hover:!bg-red-700 text-white px-2 py-1 rounded">▶ Baryonic Vel (mean)</button>
        <button role="button" aria-label="Play baryonic velocity curve" onclick="playBaryonicVelCurve()" class="!bg-red-300 hover:!bg-red-500 text-white px-2 py-1 rounded">▶ Baryonic Vel (curve)</button>
        <button role="button" aria-label="Play simulated velocity curve mean" onclick="playSimVelMean()" class="!bg-green-500 hover:!bg-green-700 text-white px-2 py-1 rounded">▶ Simulated Vel (mean)</button>
        <button role="button" aria-label="Play simulated velocity curve" onclick="playSimVelCurve()" class="!bg-green-300 hover:!bg-green-500 text-white px-2 py-1 rounded">▶ Simulated Vel (curve)</button>
        <button role="button" aria-label="Play difference velocity curves mean" onclick="playDifferenceVelMean()" class="!bg-yellow-500 hover:!bg-yellow-700 text-black px-2 py-1 rounded">▶ Difference (mean)</button>
        <button role="button" aria-label="Play difference velocity curves" onclick="playDifferenceVelCurve()" class="!bg-yellow-300 hover:!bg-yellow-500 text-black px-2 py-1 rounded">▶ Difference (curve)</button>
        </div>
        ''')
                        ui.label("").props(
            'id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0'
        )
                    

                


                    ui.label("Move the slider to add the dark matter to the simulated velocity curve").props('id=alpha_slider_label aria-live=polite tabindex=0')
                    alpha_slider = aria_slider(min=0.0, max=2.0, value=0.0, step=0.01,
                                            aria_label="Dark matter fraction control slider").props('aria-describedby=alpha_slider_label label-always')
                    f = float(alpha_slider.value)
                    
                    
                    def load_galaxy_data(filename):
                        global DATA_LOADED, selected_file

                        if not filename:
                            #print("load_galaxy_data: filename è None")
                            DATA_LOADED = False
                            return None

                        path = os.path.join(GALAXY_DATA_PATH, filename)

                        if not os.path.exists(path):
                            #print(f"load_galaxy_data: File not found: {path}")
                            DATA_LOADED = False
                            return None

                        try:
                        
                            df = pd.read_csv(path, comment='#', sep=r'\s+', header=0, engine='python')

                        
                            df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff','', regex=False)

                        
                            if all(col.replace('.','',1).isdigit() for col in df.columns):
                                #print("File senza header, ricarico:", filename)
                                df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, engine='python')
                                df.columns = ['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul']

                            #print("COLONNE FINALI:", list(df.columns))

                        
                            r_ngc     = pd.to_numeric(df['Rad'],  errors='coerce').values
                            vobs_ngc  = pd.to_numeric(df['Vobs'], errors='coerce').values
                            vgas_ngc  = pd.to_numeric(df['Vgas'], errors='coerce').values
                            vdisk_ngc = pd.to_numeric(df['Vdisk'],errors='coerce').values
                            vbul_ngc  = pd.to_numeric(df['Vbul'], errors='coerce').values
                            verr_ngc  = pd.to_numeric(df['errV'], errors='coerce').values

                        
                            if r_ngc.size == 0 or vobs_ngc.size == 0:
                                #print("load_galaxy_data: colonne vuote → fallito")
                                DATA_LOADED = False
                                return None

                            r_match = r_ngc[-1]
                            
                            DATA_LOADED = True
                        
                            #print(f"load_galaxy_data: caricato {filename}, {len(r_ngc)} punti")
                            return r_ngc, vobs_ngc, vgas_ngc, vdisk_ngc, vbul_ngc, verr_ngc, r_match

                        except Exception as ex:
                            print("load_galaxy_data: errore:", ex)
                            DATA_LOADED = False
                            return None
                    def reload_galaxy(new_value):
                    
                        global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name 
                        global selected_file, image_filepath

                        if not new_value:
                            #print("reload_galaxy: filename is None, abort")
                            return
                        galaxy_state["select"] = new_value
                        filename = new_value
                        #print("reload_galaxy: filename scelto ->", filename)

                        loaded = load_galaxy_data(filename)
                        if loaded is None:
                            #print("reload_galaxy: load fallito")
                            DATA_LOADED = False
                            return

                        r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match = loaded
                        DATA_LOADED = True
                        selected_file = filename
                        current_galaxy_name = filename
                        #image_filepath = galaxy_file_map.get(filename, None) 
                        #print(f"reload_galaxy: current_galaxy_name impostato -> {current_galaxy_name!r}")
                        alpha_slider.value = 0.0
                        update_all_plots.refresh()
                        update_image.refresh()
                        update_table.refresh()

                


                    @ui.refreshable
                    def update_all_plots():
                        if not DATA_LOADED:
                            #print("update_all_plots: DATA_LOADED == False -> skip")
                            return
                        plt.close('all')
                        try:
                            update_galaxy_rotation_plot()
                            update_mass_plot()
                            plot_chi2_user_curve()
                            update_morphology_plot()
                            #update_morphology_plot_3d()
                        except Exception as ex:
                            #print("update_all_plots: eccezione durante i plot:", ex)
                            import traceback; traceback.print_exc()
                            accessible_notify(f"Errore nei plot: {ex}", type_='error')

                    loaded = load_galaxy_data(default_galaxy)
                    if loaded is None:
                        print("Errore: il file non è stato caricato")
                    else:
                        r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match = loaded

                    
                        if r_ngc.size == 0 or v_obs_ngc.size == 0:
                            #print("Errore: r o v_obs sono vuoti! Non posso calcolare get_rhos_rs_from_observed_matching")
                            DATA_LOADED = False
                        else:
                            DATA_LOADED = True
                    galaxy_select.on_value_change(lambda e: reload_galaxy(e.value))

                    alpha_slider.on('update:model-value', update_all_plots.refresh)
                    
                    
                    if default_galaxy:
                        reload_galaxy(default_galaxy)
                    
                    

                        #print("startup: current_galaxy_name impostato a default ->", current_galaxy_name)

                    

                    def get_dm_params(fraction, r_array=None):
                    
                            if r_array is None:
                                r_array = r_ngc 
                            
                            rho_s, r_s, _ = get_rhos_rs_from_observed_matching(
                                r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match
                            )
                            
                            M_dm_grid_full = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                            
                        
                            if len(r_array) != len(M_dm_grid_full):
                                M_dm_grid_interp = np.interp(r_array, r_ngc, M_dm_grid_full)
                            else:
                                M_dm_grid_interp = M_dm_grid_full
                            
                            return rho_s, r_s, M_dm_grid_interp
                    
                    #rho_s_init, r_s_init, _ = get_rhos_rs_from_observed_matching(                       r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                    #M_dm_grid_init = M_nfw_enclosed(r_ngc, rho_s=rho_s_init, r_s=r_s_init)
                    rho_s_init, r_s_init, M_dm_grid_init = get_dm_params(f, r_array=r_ngc)

                    dm_max_possible = np.max(M_dm_grid_init) * 2  
                    chi2_max_possible = 10  

                    x_min_chi = 0
                    x_max_chi = dm_max_possible * 1.05
                    y_min_chi = 0
                    y_max_chi = chi2_max_possible

                        
                    unscaled_M_dm_grid_cache = {}
                    #chi2_plot_container = ui.column().classes('w-full')
                    chi2_state = {
    'slider_result': " ---" 
}
                    def plot_chi2_user_curve():
                        global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED
                        

                        
                        chi2_plot_container.clear()
                        plt.close()
                        with chi2_plot_container:
                            with ui.pyplot(figsize=(6,3.5), close=False,clear=True):
                                ax = plt.gca()

                                
                                if chi2_points:
                                    xs_slider, ys_slider = zip(*chi2_points)
                                    ax.scatter(xs_slider, ys_slider, s=40, c="blue", alpha=0.6, label="Points")

                                    if len(chi2_points) == 3 :
                                        coeffs_slider = np.polyfit(xs_slider, ys_slider, 2)
                                        x_fit_slider = np.linspace(min(xs_slider), max(xs_slider), 200)
                                        y_fit_slider = np.polyval(coeffs_slider, x_fit_slider)
                                        
                                        x_lo, x_hi = min(xs_slider), max(xs_slider)
                                        x_fit_slider = np.linspace(x_lo - 0.1*(x_hi-x_lo), x_hi + 0.1*(x_hi-x_lo), 400)
                                        y_fit_slider = np.polyval(coeffs_slider, x_fit_slider)
                                        ymin_index = np.argmin(y_fit_slider)
                                        xmin_slider = x_fit_slider[ymin_index]
                                        ymin_slider = y_fit_slider[ymin_index]

                                        ax.plot(x_fit_slider, y_fit_slider, "g-", lw=2, label="Parabolic Fit ")
                                        ax.scatter([xmin_slider], [ymin_slider], c='red', s=140, marker='*', label="Min ")
                                        
                                        

                                

                                                                
                                if manual_points and len(manual_points) >= 3:
                                    xm, ym = zip(*manual_points)
                                    ax.scatter(xm, ym, s=60, c="blue", marker="o", label="Points")

                                    coeffs_manual = np.polyfit(xm, ym, 2)
                                    x_fit_manual = np.linspace(min(xm), max(xm), 400)
                                    y_fit_manual = np.polyval(coeffs_manual, x_fit_manual)

                                    ax.plot(x_fit_manual, y_fit_manual, "green", lw=2, label="Parabolic Fit")

                                    
                                    xmin_manual = -coeffs_manual[1] / (2 * coeffs_manual[0])
                                    ymin_manual = np.polyval(coeffs_manual, xmin_manual)
                                    ax.scatter([xmin_manual], [ymin_manual], c='red', s=120, marker='*', label=" Min")

                                    result_label.set_text(f"DM min ≈ {xmin_manual:.3e}, χ² min ≈ {ymin_manual:.4f}")
                                    
                            


                                all_ys = []
                                if chi2_points: all_ys += [p[1] for p in chi2_points]
                                if parabolic_state["points"]: all_ys += [p[1] for p in parabolic_state["points"]]
                                if parabolic_state["plot_points"]: all_ys += [p[1] for p in parabolic_state["plot_points"]]
                                y_max_plot = y_max_chi if not all_ys else max(y_max_chi, max(all_ys)*1.1)

                                ax.set_xlim(x_min_chi, x_max_chi)
                                ax.set_ylim(y_min_chi, y_max_plot)
                                ax.set_xlabel("DM mass ($M_{\\odot}$)")
                                ax.set_ylabel(r"χ²/dof")
                                ax.grid(True)
                                handles, labels = ax.get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                if by_label: ax.legend(by_label.values(), by_label.keys())
                                text_to_show = chi2_state['slider_result']
                               #slider_result_label = ui.label("Results: ---").classes("text-green-600 font-bold text-sm my-1")
                                plt.text(0.5, 0.92, text_to_show, 
                 transform=plt.gca().transAxes, 
                 fontsize=10, 
                 color='green',
                 fontweight='bold',
                 horizontalalignment='center',
                 verticalalignment='top',     
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray'))
                                plt.title("χ² Minimization",fontsize=14,fontweight='bold')
                                plt.tight_layout()
                                ui.element('div').props(
        'role=status aria-live=polite tabindex=0 aria-label=Plot showing chi-square minimization curve; X axis is dark matter mass in solar masses, Y axis is chi-square per degree of freedom'
        )


            
                    
                    def refresh_formula_from_inputs():
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
                        ui.label('Chi² Minimization Data').classes('text-xl font-bold mb-4')
                        
                    
                        with ui.grid(columns=2).classes("w-full gap-4"):
                          
                            input_a = ui.number(label='Mass 1 (x₁)', format='%.2e').classes('w-full')
                            fa_in = ui.number(label="χ²(x₁)", format="%.4f").classes('w-full')
                           
                            input_b = ui.number(label='Mass 2 (x₂)', format='%.2e').classes('w-full')
                            fb_in = ui.number(label="χ²(x₂)", format="%.4f").classes('w-full')
                          
                            input_c = ui.number(label='Mass 3 (x₃)', format='%.2e').classes('w-full')
                            fc_in = ui.number(label="χ²(x₃)", format="%.4f").classes('w-full')

                     
                        for el in [input_a, input_b, input_c, fa_in, fb_in, fc_in]:
                            el.on('update:model-value', lambda: refresh_formula_from_inputs())

                        
                        result_label = ui.label("Min: ---").classes("text-green-600 font-bold text-lg my-2")

                     
                        def compute_and_update():
                            initialize_parabolic_points() 
                      
                            accessible_notify('Plot updated!', type_='positive')

                     
                        with ui.row().classes('w-full justify-end gap-2 mt-4'):
                            aria_button('Close','Close', on_click=chi2_input_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            aria_button('Compute Minimum','Compute Minimum', on_click=compute_and_update).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    points_display = ui.column()
                    history_display = ui.column()

                    with ui.row().classes('w-full no-wrap items-start justify-center gap-x-8'):
     
                        with ui.column().classes('w-full md:w-[30%] min-w-[300px] items-center'):
                            plot_container = ui.column().classes('w-full')
                           
                        with ui.column().classes('w-full items-center justify-center md:w-[30%] min-w-[300px] gap-1'):
            
                      

                        
                            chi2_plot_container = ui.column().classes('w-full p-0 m-0')
                            plot_chi2_user_curve() 
                            
                        
                            

                            with ui.row().classes("w-full justify-center items-center gap-4 px-1 "):
                
                              
                                aria_button("Add point", "Add point", on_click=lambda: add_chi2_point()).classes("bg-green-600 text-white font-bold font-bold py-2 px-4 roundedm")
                                
                                aria_button("Tool", "Open dialog", on_click=chi2_input_dialog.open).classes("bg-green-600 text-white font-bold font-bold py-2 px-4 rounded")
                                
                                aria_button("Reset", "Refresh plot", on_click=lambda: refresh_chi2_plot()).classes("bg-red-600 text-white font-bold font-bold py-2 px-4 rounded")

                
                                        
                                
                        mass_plot_container = ui.column().classes("w-full md:w-[30%] min-w-[300px]")
                                        
                    
                   
                    

                    
                    with ui.row().classes("gap-4 "):
                        points_display
                        history_display
                        
                        


                        def update_galaxy_rotation_plot():
                            global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name
                            if not DATA_LOADED:
                                return
                            #print("update_galaxy_rotation_plot: current_galaxy_name =", repr(current_galaxy_name),        " galaxy_select.value =", getattr(galaxy_select, "value", None))
                            
                            f = float(alpha_slider.value)
                            
                            
                                
                                #rho_s_dynamic, rs_dynamic, alpha_dm, beta_dm, gamma_dm = dm_params_from_slider(f)
                            G_grav = 4.30091e-6
                            v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 + np.maximum(0, v_disk_ngc)**2 + np.maximum(0, v_bul_ngc)**2)
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            m_total    = (v_obs_ngc**2 * r_ngc) / G_grav
                        
                            
                            #rho_s, r_s, M200 = get_rhos_rs_from_observed_matching(r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)

                            #M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                            rho_s, r_s, M_dm_grid = get_dm_params(f, r_array=r_ngc)

                            
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + f * M_dm_grid) / r_ngc)
                            
                        
                            v_obs_list   = [float(v) for v in v_obs_ngc if np.isfinite(v)]
                            v_bary_list  = [float(v) for v in v_baryonic if np.isfinite(v)]
                            v_total_list = [float(v) for v in v_total_curve if np.isfinite(v)]

                            v_obs_mean  = float(np.nanmean(v_obs_list)) if len(v_obs_list) > 0 else 0
                            v_bary_mean = float(np.nanmean(v_bary_list)) if len(v_bary_list) > 0 else 0
                            v_total_mean= float(np.nanmean(v_total_list)) if len(v_total_list) > 0 else 0
                            v_diff_raw = np.abs(v_obs_ngc - v_total_curve)

                            v_diff_list = [float(d) for d in v_diff_raw if np.isfinite(d)]
                            v_diff_mean = float(np.nanmean(v_diff_raw)) if v_diff_list else 0.0

                            ui.run_javascript(f"window.vObsSeries = {v_obs_list};")
                            ui.run_javascript(f"window.vBarySeries = {v_bary_list};")
                            ui.run_javascript(f"window.vSimSeries = {v_total_list};")
                            ui.run_javascript(f"window.vObsMean = {v_obs_mean};")
                            ui.run_javascript(f"window.vBarMean = {v_bary_mean};")
                            ui.run_javascript(f"window.vSimMean = {v_total_mean};")
                            ui.run_javascript(f"window.vDiffSeries = {v_diff_list};")
                            ui.run_javascript(f"window.vDiffMean = {v_diff_mean};")


                        

                        


                            M_vis_tot = np.max(m_baryonic)
                            M_dm_tot = np.max(M_dm_grid) * f
                            dm_fraction = 100 * M_dm_tot / (M_vis_tot + 1e-12)

                            

                            
                            mask = np.isfinite(v_obs_ngc) & np.isfinite(v_err_ngc) & (v_err_ngc > 0)
                            r_use = r_ngc[mask]
                            vobs_use = v_obs_ngc[mask]
                            verr_use = v_err_ngc[mask]
                            vmodel_use = v_total_curve[mask]  

                            chi2 = np.sum(((vobs_use - vmodel_use)/np.maximum(verr_use, 5.0))**2)


                            N_used = len(vobs_use)
                            N_params = 1  
                            dof = max(1, N_used - N_params)
                            chi2_dof = chi2 / dof
                        
                            r_ref = 20  # kpc
                            if len(r_ngc) == 0:
                                dm_frac_r = 0
                            else:
                                idx_r = np.searchsorted(r_ngc, r_ref)
                                idx_r = min(idx_r, len(r_ngc)-1)
                                dm_frac_r = 100 * f * M_dm_grid[idx_r] / (m_baryonic[idx_r] + f * M_dm_grid[idx_r])

                            residui = (vobs_use - vmodel_use) / verr_use
                            v_gas_frac = 100 * np.mean(v_gas_ngc**2 / v_total_curve**2)
                            v_disk_frac = 100 * np.mean(v_disk_ngc**2 / v_total_curve**2)
                            v_bul_frac = 100 * np.mean(v_bul_ngc**2 / v_total_curve**2)





                            with plot_container:
                                plot_container.clear()
                                plt.close() 
                                with ui.pyplot(figsize=(10, 6), close=False, clear=True):
                                    info_text = (
            f"$M_{{DM}} = {M_dm_tot:.2e} \\, M_{{\\odot}}$\n"
            f"$M_{{bar}} = {M_vis_tot:.2e} \\, M_{{\\odot}}$\n"
            f"DM Fraction: ${dm_fraction:.1f} \\,\\%$\n"
            f"$v_{{obs}} = {v_obs_ngc.max():.1f} \\, km/s$\n"
            f"$v_{{sim}} = {v_total_curve.max():.1f} \\, km/s$"
        )
                                    
                                    plt.plot(r_ngc, v_baryonic, color='red', linewidth=3, label=f'Keplerian velocity ')
                                    plt.plot(r_ngc, v_total_curve, linewidth=3, color='green', label="Simulated velocity" )
                                   
                                    plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', markersize=4, color='blue', ecolor='gray', capsize=2, label='Observed velocity', zorder=5)
                                    #plt.text(0.05, 0.95,  f"DM mass: {M_dm_tot:.2e} M☉, {dm_fraction:.1f}% of visible " f'V_obs{v_obs_ngc.max():.1f} km/s 'f"Simulated velocity: {v_total_curve.max():.1f} km/s " '(Baryonic mass: {M_vis_tot:.2e} M☉)', transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
                                    
                                    plt.text(0.96, 0.96, info_text, 
                 transform=plt.gca().transAxes, 
                 fontsize=12, 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 linespacing=1.8, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                                    plt.xlabel('Radius (kpc)', fontsize=16)
                                    plt.ylabel('Rotation Speed (km/s)', fontsize=16)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                    plt.title(f'Galaxy Rotation Curve: {galaxy_name_for_title}', fontsize=18,fontweight='bold')
                                  

                                    plt.ylim(0, max(300, 1.1 * np.max(v_obs_ngc + v_err_ngc)))
                                    plt.xlim(0, r_ngc.max() * 1.1)
                                    plt.grid(True)
                                    plt.legend(loc='upper left', fontsize=16)
                                    plt.tight_layout()

                                    ui.element('div').props(
        'role=status aria-live=polite tabindex=0 aria-label=Rotational velocity curve of the galaxy (from data),baryonic velocity curve (computed from data) and simulated velocity curve updated with new dark matter value (slider moved)'
    )


                        
                        def update_mass_plot():
                            global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name
                            if not DATA_LOADED:
                                return
                            #print("update_mass_plot: current_galaxy_name =", repr(current_galaxy_name), " galaxy_select.value =", getattr(galaxy_select, "value", None))
                            f = float(alpha_slider.value)
                            v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 +
                            np.maximum(0, v_disk_ngc)**2 +
                            np.maximum(0, v_bul_ngc)**2)
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            m_total_obs = (v_obs_ngc**2 * r_ngc) / G_grav

                            #rho_s, r_s, _ = get_rhos_rs_from_observed_matching(        r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                            #M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                            rho_s, r_s, M_dm_grid = get_dm_params(f, r_array=r_ngc)

                            m_total_model = m_baryonic + f * M_dm_grid
                            y_min_mass = 0
                            y_max_mass = max(np.max(m_total_obs), np.max(m_baryonic)*2) * 1.1
                            x_min_mass = r_ngc.min() * 0.9
                            x_max_mass = r_ngc.max() * 1.1
                            with mass_plot_container:
                                mass_plot_container.clear()
                                plt.close()
                                with ui.pyplot(figsize=(10, 6), close=False,clear=True):
                            
                                
                                    plt.plot(r_ngc, m_baryonic, "r-", lw=3, label=f"Baryonic mass")
                                    plt.plot(r_ngc, m_total_obs, "b-", lw=3, label=f"Total mass")
                                    plt.plot(r_ngc, m_total_model, "g-", lw=3, label=f"Simulated mass DM")
                                    
                                    info_text = (
            f"$M_{{tot}} = {np.max(m_total_obs):.2e} \\, M_{{\\odot}}$\n"
            f"$M_{{bar}} = {np.max(m_baryonic):.2e} \\, M_{{\\odot}}$\n"
             f"$M_{{DM}} ={np.max(m_total_model):.2e} \\,M_{{\\odot}}$\n"
           
        )
                                    plt.text(0.96, 0.96, info_text, 
                 transform=plt.gca().transAxes, 
                 fontsize=12, 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 linespacing=1.8, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                                    plt.xlim(x_min_mass, x_max_mass)
                                    plt.ylim(y_min_mass, y_max_mass)

                                    plt.xlabel("Radius (kpc)", fontsize=16)
                                    plt.ylabel("Mass ($M_\\odot$)", fontsize=16)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                    plt.title(f"Mass vs radius: {galaxy_name_for_title}", fontsize=18,fontweight='bold')

                                    plt.legend(loc='upper left', fontsize=16)
                                    plt.grid(True)
                                    plt.tight_layout()

                                    ui.element('div').props(
        'role=status aria-live=polite tabindex=0 aria-label=Total mass of the galaxy (from data), DMyonic mass (computed from data) and simulated mass updated with new dark matter value'
    )

                                    #plot_info_box_compact2({"Visible baryonic mass": f"{np.max(m_baryonic):.2e} M☉",        "Total mass": f"{np.max(m_total_obs):.2e} M☉",        f"Simulated mass DM (α={f:.2f})": f"{np.max(m_total_model):.2e} M☉" })
                        def update_morphology_plot():
                                global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name
                                if not DATA_LOADED:
                                    return
                                #print("update_morphology_plot: current_galaxy_name =", repr(current_galaxy_name),         " galaxy_select.value =", getattr(galaxy_select, "value", None))
                                f = float(alpha_slider.value) 

                        
                                v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 + np.maximum(0, v_disk_ngc)**2 + np.maximum(0, v_bul_ngc)**2)
                                m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                                M_vis_tot = np.max(m_baryonic)

                                #rho_s, r_s, _ = get_rhos_rs_from_observed_matching(r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                                #M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                                rho_s, r_s, M_dm_grid = get_dm_params(f, r_array=r_ngc)

                                M_dm_tot = np.max(M_dm_grid) * f

    
                                M_bulge = np.max((v_bul_ngc**2 * r_ngc) / G_grav)
                                M_disk  = np.max((v_disk_ngc**2 * r_ngc) / G_grav)
                                M_bary_tot = M_disk + M_bulge


    
                                R_bulge = np.average(r_ngc, weights=np.maximum(0, v_bul_ngc**2)) if np.any(v_bul_ngc > 0) else 1
                                R_disk  = np.average(r_ngc, weights=np.maximum(0, v_disk_ngc**2)) if np.any(v_disk_ngc > 0) else 5
                                R_halo  = np.sqrt(M_dm_tot / (M_vis_tot+1e-12)) * np.max(r_ngc) if M_dm_tot > 0 else 0
                            

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

                                        #maxR = max(R_halo, R_disk, R_bulge) * 1.2
                                        maxR = float(np.max(r_ngc)) * 1.1
                                        R_halo = min(R_halo, maxR) 
                                        #if f > 0 and R_halo > 0:
                                            #draw_gradient_circle(ax, (0,0), min(R_halo, maxR*1.5), 'green', n_steps=15, max_alpha=0.2)
             
                                            #ax.plot([], [], 'o', color='green', alpha=0.3, label=f'DM Halo (Sphere)')

         
                                        #draw_gradient_circle(ax, (0,0), R_disk, 'crimson', n_steps=8, max_alpha=0.6)
                                        #ax.plot([], [], 'o', color='crimson', alpha=0.5, label='Stellar Disk')

                                      
                                        #bulge = plt.Circle((0,0), R_bulge, color='gold', alpha=0.9, zorder=10)
                                        #ax.add_artist(bulge)
                                        #ax.plot([], [], 'o', color='gold', label='Bulge')

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
                                       
                                        #plt.scatter(0, 0, s=5, c='black', marker='o',   linewidths=1, zorder=4, label='Black Hole')
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

                                 
                    
                        

                    
                        
                        def chi2_function_for_minimization(M_dm_tot):
                            global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name
                            galaxy_name = current_galaxy_name
                            if not galaxy_name:
                                return np.inf
                            if galaxy_name not in unscaled_M_dm_grid_cache:
                                #rho_s, r_s, _ = get_rhos_rs_from_observed_matching(                                r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                                rho_s, r_s, M_dm_grid = get_dm_params(f, r_array=r_ngc)

                                unscaled_M_dm_grid_cache[galaxy_name] = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)

                            unscaled_M_dm_grid = unscaled_M_dm_grid_cache[galaxy_name]
                            max_dm_unscaled = np.max(unscaled_M_dm_grid)
                            if max_dm_unscaled <= 1e-6: return np.inf

                            f_val = M_dm_tot / max_dm_unscaled
                            v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 +
                                                np.maximum(0, v_disk_ngc)**2 +
                                                np.maximum(0, v_bul_ngc)**2)
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + f_val * unscaled_M_dm_grid) / r_ngc)

                            mask = np.isfinite(v_obs_ngc) & np.isfinite(v_err_ngc) & (v_err_ngc > 0)
                            vobs_use = v_obs_ngc[mask]
                            verr_use = np.maximum(v_err_ngc[mask], 5.0)
                            vmodel_use = v_total_curve[mask]
                            dof = max(1, len(vobs_use) - 1)
                            chi2_dof = np.sum(((vobs_use - vmodel_use)/verr_use)**2) / dof
                            return chi2_dof

                        def add_chi2_point():
                            global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED
                            f = float(alpha_slider.value)
                            #rho_s, r_s, _ = get_rhos_rs_from_observed_matching(                           r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                            #M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                            rho_s, r_s, M_dm_grid = get_dm_params(f, r_array=r_ngc)

                            M_dm_tot = np.max(M_dm_grid) * f

                            v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 +
                                                np.maximum(0, v_disk_ngc)**2 +
                                                np.maximum(0, v_bul_ngc)**2)
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            v_total_curve = np.sqrt(G_grav * (m_baryonic + f * M_dm_grid) / r_ngc)

                            mask = np.isfinite(v_obs_ngc) & np.isfinite(v_err_ngc) & (v_err_ngc > 0)
                            vobs_use = v_obs_ngc[mask]
                            verr_use = np.maximum(v_err_ngc[mask], 5.0)
                            vmodel_use = v_total_curve[mask]

                            chi2_val = np.sum(((vobs_use - vmodel_use)/verr_use)**2) / max(1, len(vobs_use)-1)
                            chi2_points.append((M_dm_tot, chi2_val))
                            
                            if len(chi2_points) > 3:
                                chi2_points.pop(0)

                        
                            if len(chi2_points) == 3:
                                xs, ys = zip(*chi2_points)
                                #if abs(np.polyfit(xs, ys, 2)[0]) > 1e-9:
                                coeffs = np.polyfit(xs, ys, 2)
                                xmin = -coeffs[1] / (2*coeffs[0])  
                                ymin = np.polyval(coeffs, xmin)
                                chi2_state['slider_result'] = f"Result: Min: $M_{{DM}} \\approx {xmin:.2e} \\, M_{{\\odot}},\\chi^2_{{min}} \\approx {ymin:.2f}$"
                                #else:
                                #    slider_result_label.set_text("Points are collinear.")
                            else:
                                chi2_state['slider_result']= "Add more points to find the minimum."

                            update_all_plots.refresh()
                            plot_chi2_user_curve()
                            update_displays.refresh()


                        
                        
                    

                        def initialize_parabolic_points():
                    
                            parabolic_state["points"].clear()
                            parabolic_state["history"].clear()
                            parabolic_state["plot_points"].clear()
                            parabolic_state["iteration"] = 0

                            masses = [input_a.value, input_b.value, input_c.value]
                            if not all(isinstance(v, (int,float)) for v in masses):
                                accessible_notify("Insert 3 values for DM masses", type_='warning')
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
                            #if abs(np.polyfit(xs, ys, 2)[0]) > 1e-9:
                            coeffs = np.polyfit(xs, ys, 2)
                            xmin = -coeffs[1] / (2 * coeffs[0])
                            ymin = np.polyval(coeffs, xmin)
                            result_label.set_text(f"DM min = {xmin:.3e} M☉,  χ² min = {ymin:.4f}")
                            #else:
                                #result_label.set_text("Points are collinear.")
                        
                            accessible_notify("Points computed. Check the minimum on the plot.", type_='success')

                            #accessible_notify(" Points initialized. Ready for χ² computation.", type='positive')
                        

                            update_displays.refresh()
                            #update_all_plots.refresh()
                            plot_chi2_user_curve()

                    

                        @ui.refreshable
                        def update_displays():
                            points_display.clear()
                            with points_display:
                                if parabolic_state["points"]:
                                    section("Algorithm Points")
                                    for x,y in sorted(parabolic_state["points"], key=lambda p: p[0]):
                                        ui.label(f"Mass = {x:.3e} M☉, χ²/dof = {y:.4f}")
                            history_display.clear()
                            with history_display:
                                if parabolic_state["history"]:
                                    section("History step")
                                    for entry in parabolic_state["history"]:
                                        ui.markdown(entry)

                        
                        
                    
                        
    
                        

                        
                        def refresh_chi2_plot():
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
                    

                        


                    

        
                with ui.tab_panel(two).props('role=tabpanel'):
                # with ui.card().classes("p-4 !bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Mass ")
                    
                    with ui.dialog() as info_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Analysis Overview</h3>
        <p>You are an astrophysicist investigating the <b>presence of dark matter</b> in a galaxy.</p>
        <ul>
            <li>Choose a dataset and complete the missing fields with the correct formulas.</li>
            <li>Click <b>Run Analysis</b> to compare the baryonic mass with the total mass and the observed velocity with the Keplerian-like velocity.</li>
        </ul>
    """).props('aria-label="Descriptive text about galaxy velocity and mass activities"')
                        aria_button("Close", "close the box",on_click=info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.dialog() as data_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Dataset description and references'):
                        info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                        
                
                        reference_box(
        """**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')
                        aria_button("Close", "close the box",on_click=info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as baryonic_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Baryonic Calculation Steps</h3>
        
        <ul>
            <li>
                <b>Step 1:</b> Compute the baryonic velocity as the sum of each component from data:<br>
                <span class="math">\( v_{\mathrm{bar}}^2 = v_{\mathrm{gas}}^2 + v_{\mathrm{disk}}^2 + v_{\mathrm{bulge}}^2 \)</span>
            </li>
            
            <li>
                <b>Step 2:</b> Derive the baryonic mass:<br>
                <span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2}{G} \)</span>
            </li>
            
            <li>
                <b>Step 3:</b> Compute the total mass from observations:<br>
                <span class="math">\( M(r) = \frac{r \, v_{\mathrm{obs}}^2}{G} \)</span>
            </li>
            
            <li>
                <b>Step 5 (Plotting):</b>
                <ul style="margin-top:5px; list-style-type: circle;">
                    <li><b>X-axis:</b> radius (data)</li>
                    <li><b>Y-axis:</b> <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue)</li>
                    <li><b>Masses:</b> <span class="math">\( M_{\mathrm{bar}} \)</span> (red), <span class="math">\( M_{\mathrm{tot}} \)</span> (blue)</li>
                </ul>
            </li>
        </ul>
    """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the steps to compute baryonic velocity and mass"')
                        aria_button("Close", "close the box",on_click=baryonic_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"') as legend_dialog, ui.card().classes("p-4 w-full max-w-2xl overflow-x-auto"):
                        html_info_box(r"""
        <h3 class="text-xl font-bold mb-2">Legend of Symbols and Units</h3>
        <ul class="list-disc pl-5 space-y-1">
            <li><b>Rad</b> = Radius [kpc]</li>
            <li><b>V_obs</b> = Observed rotation velocity [km/s]</li>
            <li><b>V_gas, V_disk, V_bul</b> = Velocity contributions from gas, stellar disk, bulge [km/s]</li>
            <li><b>G</b> = 4.30091 &times; 10<sup>-6</sup> kpc&middot;(km/s)&sup2;&middot;M<sub>&odot;</sub><sup>-1</sup></li>
            <li><b>M_baryonic</b> = Baryonic mass [M<sub>&odot;</sub>]</li>
            <li><b>M_total</b> = Total mass (from observations) [M<sub>&odot;</sub>]</li>
            <li><b>V_baryonic</b> = Baryonic velocity (Keplerian-like prediction) [km/s]</li>
        </ul>
    """).props('id="legend-desc"')
                        aria_button("Close", "close the box",on_click=legend_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                
                    with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog, ui.card().classes("p-4 w-full max-w-2xl overflow-x-auto"):
                        html_info_box(r"""
        <h3 class="text-xl font-bold mb-2">Units Conversion</h3>
        <ul class="list-disc pl-5 space-y-1">
            <li><b>1 kpc</b> = 3.086 &times; 10<sup>16</sup> m = 3.26 &times; 10<sup>3</sup> light-years</li>
            <li><b>1 pc</b> = 3.086 &times; 10<sup>13</sup> km = 3.26 light-years</li>
            <li><b>1 Mpc</b> = 10<sup>6</sup> pc = 3.086 &times; 10<sup>19</sup> km</li>
            <li><b>1 km/s</b> = 3.6 &times; 10<sup>3</sup> km/h = 10<sup>3</sup> m/s</li>
            <li><b>1 M<sub>&odot;</sub></b> = 1.989 &times; 10<sup>30</sup> kg (solar mass)</li>
            <li><b>1 L<sub>&odot;</sub></b> = 3.828 &times; 10<sup>26</sup> W (solar luminosity)</li>
        </ul>
    """).props('id="units-desc"')
                        aria_button("Close","close the box", on_click=units_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.row().classes('w-full gap-8 justify-center'):
                        galaxy_files = [f for f in os.listdir(GALAXY_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                        galaxy_file_map = get_data_and_images(GALAXY_DATA_PATH, GALAXY_IMG_PATH)
                        
                        if not galaxy_file_map:
                            ui.label("No galaxy data files found in 'App/galaxy_data/' directory.").classes('text-red-500')
                        else:
                        
                            galaxy_select = ui.select(galaxy_files, label='Select a Galaxy Dataset').classes('flex-1 max-w-md items-start text-lg')
                        aria_button("Instruction", "Read instructions",on_click=safe_click(lambda: [info_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("Dataset", "Read dataset info",on_click=safe_click(lambda: [data_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )

                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [baryonic_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📘 Legend", "Read the legend of symbols and units",on_click=legend_dialog.open).classes(
    "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📐 Units","Read the units conversion", on_click=units_dialog.open).classes(
    "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )


                

                    
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

                   
                    
                    answer_vb, answer_vo,answer_rad,answer_g,answer_Mtot,answer_Mbar,answer_vgas,answer_vdisk,answer_vbulge = {}, {},{},{},{},{},{},{},{}
                    @ui.refreshable
                    def show_galaxy_pseudocode():
                    

                        with ui.card().classes("items-center justify-center p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown("Galaxy Mass Exercise: fill the missing parts").classes("text-2xl font-bold text-blue-300 mb-4")
                          
                            with ui.row().classes('w-full gap-4 max-w-6xl no-wrap items-start justify-center'):
                                
                               
                                with ui.column().classes('flex-1 gap-4 min-w-[300px]'):
                                    
                                 
                                    ui.label("1) Load galaxy dataset").classes('text-blue-200 font-bold text-lg')
                                   
                                    ui.code(f'data = load_data("{galaxy_select.value}")', language='python')

                                  
                                    ui.label("2) Compute Baryonic Velocity").classes('text-blue-200 font-bold text-lg')
                                    with ui.row().classes('items-center gap-1 mt-1 flex-nowrap math-text'):
                                       
                                        ui.html('V<span class="math-sub">bar</span> = &radic;')
                                        
                                       
                                        with ui.row().classes('sqrt-content gap-1'):
                                            ui.html('(')
                                            answer_vgas['el'] = aria_formula_input().props('dense filled').classes('w-24')
                                            ui.html(')<span class="math-sup">2</span> + (')
                                            answer_vdisk['el'] = aria_formula_input().props('dense filled').classes('w-24')
                                            ui.html(')<span class="math-sup">2</span> + (')
                                            answer_vbulge['el'] = aria_formula_input().props('dense filled').classes('w-24')
                                            ui.html(')<span class="math-sup">2</span>')

                                   
                                    ui.label("3) Compute Luminous Mass").classes('text-blue-200 font-bold text-lg')
                                    with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                      
                                        ui.html('M<span class="math-sub">bar</span> = ')
                                        
                                       
                                        with ui.column().classes('fraction'):
                                        
                                            with ui.row().classes('numerator items-center gap-1'):
                                                ui.html('(')
                                                answer_vb['el'] = aria_formula_input().props('dense filled').classes('w-24')
                                                ui.html(')<span class="math-sup">2</span> &middot; ')
                                                answer_rad['el'] =aria_formula_input().props('dense filled').classes('w-24')
                                           
                                            with ui.row().classes('denominator w-full justify-center'):
                                                answer_g['el'] = aria_formula_input().props('dense filled').classes('w-24')

                           
                                with ui.column().classes('flex-1 gap-2'):
                                    
                                   
                                    ui.label("4) Compute Total Mass").classes('text-blue-200 font-bold text-lg')
                                    with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                         
                                        ui.html('M<span class="math-sub">tot</span> = ')
                                        
                                     
                                        with ui.column().classes('fraction'):
                                          
                                            with ui.row().classes('numerator items-center gap-1'):
                                                ui.html('(')
                                                answer_vo['el'] =aria_formula_input().props('dense filled').classes('w-24')
                                                ui.html(')<span class="math-sup">2</span> &middot; ')
                                               
                                                answer_rad['el2'] = aria_formula_input().props('dense filled').classes('w-24')
                                      
                                            with ui.row().classes('denominator w-full justify-center'):
                                                answer_rad['el2'] = aria_formula_input().props('dense filled').classes('w-24')

                               
                                    ui.label("5) Compute Dark Matter Mass").classes('text-blue-200 font-bold text-lg')
                                    with ui.row().classes('items-center gap-1 mt-1 math-text'):
                                       
                                        ui.html('M<span class="math-sub">DM</span> = ')
                                        answer_Mtot['el'] = aria_formula_input().props('dense filled').classes('w-24')
                                        ui.html('&minus;')
                                        answer_Mbar['el'] = aria_formula_input().props('dense filled').classes('w-24')

                                 
                                    ui.label("6) Plot Results").classes('text-blue-200 font-bold text-lg')
                                    with ui.column().classes('bg-gray-800 p-3 rounded font-mono text-sm text-green-300 w-auto inline-block border border-gray-700'):
                                        ui.label('plt.plot(R, M_bar, label="Baryonic")')
                                        ui.label('plt.plot(R, M_tot, label="Total")')
                                        ui.label('plt.plot(R, V_bar, label="V_bar")')
                                        ui.label('plt.plot(R, V_obs, label="V_obs")')

                    show_galaxy_pseudocode()
                    galaxy_select.on('update:model-value', lambda e: show_galaxy_pseudocode.refresh())

                    
                    
                    
                    
                    plots_and_image_container = ui.column().classes('w-full items-center')
                    @ui.refreshable
                    def update_galaxy_mass_analysis():
                        selected_file = galaxy_select.value
                        if not selected_file:
                            plots_and_image_container.clear()
                            return

                        data_filepath = os.path.join(GALAXY_DATA_PATH, selected_file)
                        image_filepath = galaxy_file_map.get(selected_file)
                        plots_and_image_container.clear()
                        
                        try:
                                data_ngc = pd.read_csv(data_filepath, comment='#', sep=r'\s+', header=0, engine='python')
                                r_ngc = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                                v_obs_ngc = pd.to_numeric(data_ngc['Vobs'], errors='coerce').values
                                v_gas_ngc = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values
                                v_disk_ngc = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values
                                v_bul_ngc = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values
                                v_err_ngc = pd.to_numeric(data_ngc['errV'], errors='coerce').values

                                
                                v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 + np.maximum(0, v_disk_ngc)**2 + np.maximum(0, v_bul_ngc)**2)
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
                        popup_plot_container.clear()
                        selected_file = galaxy_select.value
                        if not selected_file:
                            return
                        
                        data_path = os.path.join(GALAXY_DATA_PATH, selected_file)
                        data_ngc = pd.read_csv(data_path, comment='#', sep=r'\s+', header=0, engine='python')

                        r = pd.to_numeric(data_ngc['Rad'], errors='coerce').values
                        vobs = pd.to_numeric(data_ngc['Vobs'], errors='coerce').values
                        verr = pd.to_numeric(data_ngc['errV'], errors='coerce').values
                        vgas = pd.to_numeric(data_ngc['Vgas'], errors='coerce').values
                        vdisk = pd.to_numeric(data_ngc['Vdisk'], errors='coerce').values
                        vbul = pd.to_numeric(data_ngc['Vbul'], errors='coerce').values

                        vbar = np.sqrt(np.maximum(0, vgas)**2 + np.maximum(0, vdisk)**2 + np.maximum(0, vbul)**2)
                        mbar = (vbar**2 * r) / G_grav
                        mtot = (vobs**2 * r) / G_grav

                        with popup_plot_container:
                        
                            with ui.pyplot(figsize=(8, 6)):
                                plt.errorbar(r, vobs, yerr=verr, fmt='o', color='blue', ms=4,
                                            ecolor='lightblue', capsize=2, label='Observed')
                                plt.plot(r, vbar, color='red', lw=2, label='Baryonic')
                                plt.xlabel("Radius (kpc)"); plt.ylabel("Velocity (km/s)")
                                plt.title("Rotation Curve",fontweight='bold')
                                plt.grid(True); plt.legend()

                            
                            with ui.pyplot(figsize=(8, 6)):
                                plt.plot(r, mbar/1e9, color='red', lw=2, label='Baryonic Mass')
                                plt.plot(r, mtot/1e9, color='blue', lw=2, label='Total Mass')
                                plt.xlabel("Radius (kpc)"); plt.ylabel("Mass (10^9 M☉)")
                                plt.title("Enclosed Mass",fontweight='bold')
                                plt.grid(True); plt.legend()
                    galaxy_select.on('update:model-value', lambda e: update_plots_popup.refresh())


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


                    with ui.dialog() as plots_popup, ui.card().classes('p-4 w-full max-w-[1300px]'):
                        ui.label("Galaxy Plots").classes("text-2xl font-bold mb-4")
                        popup_plot_container = ui.column().classes("w-full items-center")
                        aria_button("Close", "close the plots popup", on_click=plots_popup.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.row().classes('w-full justify-center gap-4 '):
                    
                        aria_button("Run Analysis", "Run the analysis to reproduce the plots",on_click=check_and_run_galaxy).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                     
                    
                        aria_button("Open Plots", "Open the two galaxy plots in a popup window",            on_click=lambda: [update_plots_popup(), plots_popup.open()]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")



                with ui.tab_panel(three).props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Velocity Distribution")
                    
                    with ui.dialog() as instruction_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Coma Cluster Simulation</h3>
        <ul>
            <li>The <b>blue histogram</b> shows the observed velocities of galaxies in the Coma cluster (fixed data).</li>
            <li>Use the slider to <b>add Dark Matter (DM)</b> to your simulation.</li>
            <li>With <b>DM = 0</b>, the simulated galaxies are gravitationally unbound.</li>
            <li>As you increase the <b>DM</b>, the gravitational pull keeps the galaxies bound and shapes their velocity distribution to match the observed histogram.</li>
        </ul>
        
    """).props('aria-label="Descriptive text about galaxy cluster activity"')
                        aria_button("Close","Close the box", on_click=instruction_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as dataset_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Dataset info'):
                
                
                        info_box("**Dataset variables**: objid (galaxy ID),ra (right ascension),dec (declination),modelmag_r (apparent magnitude in r band),modelmagerr_r (magnitude error),extinction_r,redshift (z),zErr (redshift error)")
                                                    
                        reference_box("""**Dataset reference**: [Kaggle:Coma cluster](https://www.kaggle.com/datasets/mertalkan98/coma-cluster) ; [SDSS](https://www.sdss.org/science/data-release-publications)""").classes('text-base italic')
                        aria_button("Close","Close the box", on_click=dataset_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                    with ui.row().classes('w-full justify-center '):
                        with ui.dialog() as cluster_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                            html_info_box(r"""
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

            <li><b>Step 19:</b> Total mass (DM + baryonic, linked to slider):<br>
            <span class="math">\( M_{\mathrm{tot}}(i) = M_{\mathrm{bar}}(i) + f \cdot M_{\mathrm{NFW}}(r_{\mathrm{proj},i}) \)</span></li>

            <li><b>Step 20:</b> Velocity dispersions:
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
    """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of the steps to compute cluster properties"')
                            aria_button("Close","Close the box", on_click=cluster_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                            
                            
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
                        
                        ).classes('w-1/2 max-w-md').props('id=galaxy_selector aria-label=Galaxy dataset selector role=listbox tabindex=0')
                                
                                
                    
                        select=dataset_selector.value
                        cluster_state = {
            "select": DEFAULT_CLUSTER
        }

                    
                        cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)
                        
                    
                        
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
                        with ui.dialog() as image_dialog, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl'):
                            ui.html("<h5>Cluster Image</h5>")
                            update_cluster_image()
                            aria_button("close",'close',on_click=image_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        
                        with ui.dialog() as table_dialog, ui.card().classes('w-full max-w-xl p-0 border border-gray-200 shadow-md rounded-xl'):
                            update_cluster_table()
                            aria_button("close",'close',on_click=table_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        
                        with ui.dialog() as sim_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
                            ui.label('Cluster Galaxies Velocities View').classes('text-xl font-bold').props('aria-label=Galaxy Simulation View')
                            aria_button("close",'close',on_click=sim_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            with ui.row().classes('w-full justify-center gap-4 p-4'):
                                
                                with ui.column().classes('flex-1 items-center'):
                                    sim_container_obs = ui.column()
                                with ui.column().classes('flex-1 items-center'):
                                    sim_container_model = ui.column()
                
                
                

                        aria_button("Instruction", "Read the instructions",on_click=safe_click(lambda: [instruction_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")  
                        aria_button("Dataset", "Read the info about dataset",on_click=safe_click(lambda: [dataset_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")  
                        aria_button("Info", "Read the detailed information about computational steps from data to plots",on_click=safe_click(lambda: [cluster_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")  
                    
                        aria_button("Image", "Show the cluster image corresponding to the selected dataset", on_click=safe_click(lambda: [
                            update_cluster_image.refresh(), 
                            image_dialog.open()
                        ])).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")

                        aria_button("Table", "Show the data table corresponding to the selected dataset", on_click=safe_click(lambda: [
                            update_cluster_table.refresh(), 
                            table_dialog.open()
                        ])).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        aria_button('Galaxies velocities', 'show cluster galaxies velocity',on_click=sim_dialog.open, icon='visibility').classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        
                    
                    with ui.row().classes('w-full justify-center items-center gap-2'):
                        ui.html('''
        <div class="flex space-x-2">
        <button id="audio-button" role="button" aria-label="Activate or deactivate audio" aria-pressed="false"
                    onclick="initAudio()" 
                    class="!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded">
                Activate Audio
            </button>
        <button role="button" aria-label="Play observed dispersion velocity mean" onclick="playObservedSigmaMean()" class="!bg-blue-500 hover:!bg-blue-700 text-white px-3 py-1 rounded">▶ Observed σ (mean)</button>
        <button role="button" aria-label="Play observed dispersion velocity" onclick="playObservedSigmaCurve()" class="!bg-blue-300 hover:!bg-blue-500 text-white px-3 py-1 rounded">▶ Observed σ (curve)</button>
        <button role="button" aria-label="Play baryonic dispersion velocity mean" onclick="playBaryonicSigmaMean()" class="!bg-red-500 hover:!bg-red-700 text-white px-3 py-1 rounded">▶ Baryonic σ (mean)</button>
        <button role="button" aria-label="Play baryonic dispersion velocity" onclick="playBaryonicSigmaCurve()" class="!bg-red-300 hover:!bg-red-500 text-white px-3 py-1 rounded">▶ Baryonic σ (curve)</button>
        <button role="button" aria-label="Play simulated dispersion velocity mean" onclick="playSimSigmaMean()" class="!bg-green-500 hover:!bg-green-700 text-white px-3 py-1 rounded">▶ Simulated σ (mean)</button>
        <button role="button" aria-label="Play simulated dispersion velocity" onclick="playSimSigmaCurve()" class="!bg-green-300 hover:!bg-green-500 text-white px-3 py-1 rounded">▶ Simulated σ (curve)</button>
        <button role="button" aria-label="Play differences dispersion velocities mean" onclick="playDifferenceSigmaMean()" class="!bg-purple-700 hover:!bg-purple-900 text-white px-3 py-1 rounded">▶ Difference σ (mean)</button>
        <button role="button" aria-label="Play differences dispersion velocities" onclick="playDifferenceSigmaCurve()" class="!bg-purple-500 hover:!bg-purple-700 text-white px-3 py-1 rounded">▶ Difference σ (curve)</button>

        </div>
        ''')
                        ui.label("").props(
            'id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0'
        )
                    
                            
                            
                    

                

                    def load_coma_dataset(filename): 
                        global observed_vel, N_obs, v_mean_val, sigma_obs
                        global z_cluster, center_ra, center_dec
                        global r_proj_kpc, rho_crit, m_bary_at_gal
                        global M200, R200, c
                        global obs_mean, observed_vel_pec
                        global bins, counts_obs, bin_centers
                        global r_min, r_max, r_padding
                        global dm_slider_min, dm_slider_max, f_max
                        global M_bar_tot, R_cluster_bar, M_tot_max, R200_tot
                        global v_max_expected, v_max_bar
                        global v_min_global, v_max_global, v_padding
                        global ylim_min, ylim_max
                        global x_min, x_max, padding, y_max
                        data_path = os.path.join(DATA_DIR, "coma_data.csv")
                        df_coma = pd.read_csv(data_path, skiprows=1)
                        df_coma.columns = ['objid','ra','dec','modelmag_r','modelmagerr_r','extinction_r','redshift','zErr']

                    
                        observed_vel = c_light * df_coma['redshift'].dropna().values
                        N_obs = len(observed_vel)
                        v_mean_val = np.mean(observed_vel)
                        sigma_obs = np.std(observed_vel)

                        z_arr = df_coma['redshift'].dropna().values
                        if len(z_arr) == 0:
                            raise RuntimeError("No redshifts in df")
                        z_cluster = np.nanmedian(z_arr)

                        try:
                            idx_bcg = df_coma['modelmag_r'].dropna().idxmin()
                            center_ra = float(df_coma.loc[idx_bcg, 'ra'])
                            center_dec = float(df_coma.loc[idx_bcg, 'dec'])
                        except Exception:
                            center_ra = float(np.nanmedian(df_coma['ra'].dropna().values))
                            center_dec = float(np.nanmedian(df_coma['dec'].dropna().values))

                        D_A_kpc = angular_diameter_distance_Mpc(z_cluster) * 1000.0
                        theta_rad = angsep_rad(df_coma['ra'].values, df_coma['dec'].values, center_ra, center_dec)
                        r_proj_kpc = np.maximum(theta_rad * D_A_kpc, 1e-3)

                        rho_crit = rho_crit_Msunkpc3(H0=70.0)
                        m_bary_at_gal = stellar_mass_from_r_mag(df_coma['modelmag_r'].values, df_coma['extinction_r'].values, z_cluster)

                        M200, R200 = estimate_M200_R200_from_sigma(sigma_obs, rho_crit)
                        c = concentration_duffy2008(M200, z_cluster, h=0.7, relaxed=False)

                        obs_mean = np.mean(observed_vel)
                        observed_vel_pec = observed_vel - obs_mean

                
                        x_min = observed_vel.min()
                        x_max = observed_vel.max()
                        padding = 0.15 * (x_max - x_min)
                        bins = np.linspace(0, x_max + padding, 50)
                        bin_centers = 0.5 * (bins[1:] + bins[:-1])
                        counts_obs, _ = np.histogram(observed_vel, bins=bins)
                        y_max = int(np.ceil(np.max(counts_obs) * 1.15))
                        r_min, r_max = r_proj_kpc.min(), r_proj_kpc.max()
                    
                    
                        
                        
                        y_max = int(np.ceil(counts_obs.max() * 1.3))

                    
                        r_min = r_proj_kpc.min()
                        r_max = r_proj_kpc.max()
                        r_padding = 0.1 * (r_max - r_min)
                        v_min_global = 0.0
                        ylim_min=v_min_global
                        #ylim_min = observed_vel.min() - 0.2 * abs(observed_vel.min())
                        ylim_max = observed_vel.max() + 0.2 * abs(observed_vel.max())

                        return df_coma

                    def load_cluster_dataset(filename):
                        if filename.lower() == "coma_data.csv":
                            return load_coma_dataset(DEFAULT_CLUSTER) 
                        data_filepath = os.path.join(CLUSTER_DATA_PATH, filename)

                        df = pd.read_csv(data_filepath, sep=r"\s+", header=None)


                        if df.shape[1] == 7:
                            df[7] = np.nan

                        df = df.iloc[:, :8]
                        df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]

                        for col in ["RAdeg", "DEdeg", "RV", "bmag"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                        df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])

                        return df

                    def initialize_cluster_from_df(df):
                        global observed_vel, N_obs, v_mean_val, sigma_obs
                        global z_cluster, center_ra, center_dec
                        global r_proj_kpc, rho_crit, m_bary_at_gal
                        global M200, R200, c
                        global obs_mean, observed_vel_pec
                        global bins, counts_obs, bin_centers
                        global r_min, r_max, r_padding
                        global dm_slider_min, dm_slider_max, f_max
                        global M_bar_tot, R_cluster_bar, M_tot_max, R200_tot
                        global v_max_expected, v_max_bar
                        global v_min_global, v_max_global, v_padding
                        global ylim_min, ylim_max
                        global x_min, x_max, padding, y_max

                    
                        observed_vel = df["RV"].values
                        N_obs = len(observed_vel)
                        v_mean_val = np.mean(observed_vel)
                        sigma_obs = np.std(observed_vel)

                    
                
                        z_cluster = np.nanmedian(observed_vel / c_light)

                    
                        idx_bcg = df["bmag"].idxmin()
                        center_ra = float(df.loc[idx_bcg, "RAdeg"])
                        center_dec = float(df.loc[idx_bcg, "DEdeg"])

                    
                        D_A_kpc = angular_diameter_distance_Mpc(z_cluster) * 1000.0
                        theta_rad = angsep_rad(df["RAdeg"], df["DEdeg"], center_ra, center_dec)
                        r_proj_kpc = np.maximum(theta_rad * D_A_kpc, 1e-3)

                    
                        rho_crit = rho_crit_Msunkpc3(H0=70.0)

                        m_bary_at_gal = stellar_mass_from_r_mag(
                            df["bmag"].values,
                            0.0,
                            z_cluster
                        )

                
                        M200, R200 = estimate_M200_R200_from_sigma(sigma_obs, rho_crit)

                    
                        c = concentration_duffy2008(M200, z_cluster, h=0.7, relaxed=False)

                    
                        obs_mean = np.mean(observed_vel)
                        observed_vel_pec = observed_vel - obs_mean
        
                        x_min = observed_vel.min()
                        x_max = observed_vel.max()
                        padding = 0.15 * (x_max - x_min)
                        


                        bins = np.linspace(0, x_max + padding, 50)
                        bin_centers = 0.5 * (bins[1:] + bins[:-1])
                        counts_obs, _ = np.histogram(observed_vel, bins=bins)
                        y_max = int(np.ceil(counts_obs.max() * 1.3))

                        r_min, r_max = r_proj_kpc.min(), r_proj_kpc.max()
                        r_padding = 0.1 * (r_max - r_min)

                    
                        dm_slider_min, dm_slider_max = 0.0, 50.0
                        f_max = dm_slider_max

                        M_bar_tot = np.sum(m_bary_at_gal)
                        R_cluster_bar = r200_from_M200(M_bar_tot, rho_crit)
                        M_tot_max = M_bar_tot + f_max * M200
                        R200_tot = r200_from_M200(M_tot_max, rho_crit)

                        v_max_expected = np.sqrt(G_grav * M_tot_max / R200_tot)
                        v_max_bar = np.sqrt(G_grav * M_bar_tot / R_cluster_bar)

                        v_min_global = 0.0
                        v_max_global = max(
                            observed_vel.max(),
                            v_max_expected * 1.2,
                            v_max_bar * 1.5
                        )

                        v_padding = 0.1 * (v_max_global - v_min_global)
                        ylim_min = v_min_global
                        #ylim_min = observed_vel.min() - 0.2 * abs(observed_vel.min())
                        ylim_max = observed_vel.max() + 0.2 * abs(observed_vel.max())
                        
                    def recompute_axis_limits():
                        global x_min, x_max, padding, y_max
                        global r_min, r_max, r_padding
                        global ylim_min, ylim_max

                        x_min = observed_vel.min()
                        x_max = observed_vel.max()
                        padding = 0.15 * (x_max - x_min)

                        y_max = counts_obs.max() * 1.3

                        r_min = r_proj_kpc.min()
                        r_max = r_proj_kpc.max()
                        r_padding = 0.1 * (r_max - r_min)
                        v_min_global = 0.0
                    
                        ylim_min = v_min_global
                        #ylim_min = observed_vel.min() - 0.2 * abs(observed_vel.min())
                        ylim_max = observed_vel.max() + 0.2 * abs(observed_vel.max())


                    with ui.column().classes('w-full items-center '):
                        dm_slider_min, dm_slider_max = 0.0, 50.0
                        ui.label("Move the slider to add dark matter to the simulated velocity distribution in the cluster").props('id=dm_slider_label')
                        dm_slider = aria_slider(min=dm_slider_min, max=dm_slider_max,
                        value=0.0, step=0.01,  aria_label="Dark matter fraction control slider").props('aria-describedby=dm_slider_label label-always')

                
                        
                        with ui.row().classes('w-full justify-center gap-4'):
                            with ui.column().classes('flex-1 items-center'):
                                
                                plot_container_scatter = ui.column()
                            
    
                            with ui.column().classes('flex-1 items-center'):
                                plot_container_histo = ui.column()
                                
                            
                    
                    

                    @ui.refreshable
                    def refresh_cluster_plots():
                            global observed_vel, N_obs, v_mean_val, sigma_obs
                            global z_cluster, center_ra, center_dec
                            global r_proj_kpc, rho_crit, m_bary_at_gal
                            global M200, R200, c
                            global obs_mean, observed_vel_pec
                            global bins, counts_obs, bin_centers
                            global r_min, r_max, r_padding
                            global dm_slider_min, dm_slider_max, f_max
                            global M_bar_tot, R_cluster_bar, M_tot_max, R200_tot
                            global v_max_expected, v_max_bar
                            global v_min_global, v_max_global, v_padding
                            global ylim_min, ylim_max
                            global x_min, x_max, padding, y_max
                            #global df, cluster_name
                        
                            plt.close()
                            update_coma_histogram_v2()
                    
                
                        
                    if DEFAULT_CLUSTER.lower() == "coma_data.csv":
                        df = load_coma_dataset(DEFAULT_CLUSTER)
                        cluster_name = DEFAULT_CLUSTER
                        #select = dataset_selector.value
                
                    else:
                        initialize_cluster_from_df(df)
                    recompute_axis_limits()
                    
                    dm_slider.value = 0.0
                    refresh_cluster_plots.refresh()

                
                    def on_dataset_change(new_value):
                        global cluster_name, df
                        global observed_vel, N_obs, v_mean_val, sigma_obs
                        global z_cluster, center_ra, center_dec
                        global r_proj_kpc, rho_crit, m_bary_at_gal
                        global M200, R200, c
                        global obs_mean, observed_vel_pec
                        global bins, counts_obs, bin_centers
                        global r_min, r_max, r_padding
                        global dm_slider_min, dm_slider_max, f_max
                        global M_bar_tot, R_cluster_bar, M_tot_max, R200_tot
                        global v_max_expected, v_max_bar
                        global v_min_global, v_max_global, v_padding
                        global ylim_min, ylim_max
                        global x_min, x_max, padding, y_max
                        filename = new_value
                        if not isinstance(filename, str):
                            return
                        cluster_state["select"] = new_value
                        if new_value.lower() == "coma_data.csv":
                            df = load_coma_dataset(new_value)
                        
                        else:
                            df = load_cluster_dataset(new_value)
                            initialize_cluster_from_df(df)
                        recompute_axis_limits()
                        cluster_name = filename
                        select = new_value
                    
                        dm_slider.value = 0.0
                        refresh_cluster_plots.refresh()
                        update_cluster_image.refresh()
                        update_cluster_table.refresh()
                        #ui.timer(0.1, refresh_cluster_plots.refresh, once=True)
                        global state, N_gal, vx_obs, vy_obs
                    
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
                        #observed_vel_sub = np.take(observed_vel, indices)
                        #sigma_bary_sub = np.take(sigma_bary, indices)
                        

                        sigma_bary_sub = sigma_bary[indices]

                        observed_vel_sub = observed_vel[indices]
                        sigma_obs_val = np.std(observed_vel_sub)
                        vx_obs = rng.normal(0.0, sigma_obs_val, N_gal)
                        vy_obs = rng.normal(0.0, sigma_obs_val, N_gal)

                        vx_model = rng.normal(0.0, sigma_bary_sub, N_gal)
                        vy_model = rng.normal(0.0, sigma_bary_sub, N_gal)

                        state = {
                            "x_obs": x_init_obs.copy(),
                            "y_obs": y_init_obs.copy(),
                            "x_model": x_init_mod.copy(),
                            "y_model": y_init_mod.copy(),
                            "vx_obs": vx_obs.copy(),
                            "vy_obs": vy_obs.copy(),
                            "vx_model": vx_model.copy(),
                            "vy_model": vy_model.copy(),
                            "rng": rng,
                            "indices": indices
                        }
                


                    


                        dt = 0.05
                    

                        
                    dm_slider.on('update:model-value', refresh_cluster_plots.refresh)
                    #dataset_selector.on('update:model-value',  on_dataset_change)
                    dataset_selector.on_value_change(lambda e: on_dataset_change(e.value))
                



                
                            
                    def update_coma_histogram_v2():
                        global observed_vel, N_obs, v_mean_val, sigma_obs
                        global z_cluster, center_ra, center_dec
                        global r_proj_kpc, rho_crit, m_bary_at_gal
                        global M200, R200, c
                        global obs_mean, observed_vel_pec
                        global bins, counts_obs, bin_centers
                        global r_min, r_max, r_padding
                        global dm_slider_min, dm_slider_max, f_max
                        global M_bar_tot, R_cluster_bar, M_tot_max, R200_tot
                        global v_max_expected, v_max_bar
                        global v_min_global, v_max_global, v_padding
                        global ylim_min, ylim_max
                        global x_min, x_max, padding, y_max
                        global cluster_name
                        try:
        
                            f = float(dm_slider.value) 
                
                        
                        
                            r_safe = np.maximum(r_proj_kpc, 1) 
                        

                            M_dm_at_gal = Mc_nfw_enclosed(r_safe, M200, c, rho_crit)


                            M_total_at_gal = np.maximum(m_bary_at_gal + f * M_dm_at_gal, 1e-6)
                        
                        
                            sigma_los_loc = np.sqrt(np.maximum(0.0, G_grav * M_total_at_gal / (3.0 * r_safe))) 
                            sigma_fit = np.std(observed_vel) 
                            
                            M_bar_tot = np.sum(m_bary_at_gal)
                            M_tot = M_bar_tot + f * M200
                            R_cluster = r200_from_M200(M_tot, rho_crit)
                            v_center_tot = np.sqrt(np.maximum(0.0, G_grav * M_tot / R_cluster))
                        
                            sigma_bar = np.sqrt(np.maximum(0.0, G_grav * m_bary_at_gal / (3.0 * r_safe)))
                    
                            R_cluster_bar = r200_from_M200(M_bar_tot, rho_crit)
                            v_center_bar = np.sqrt(np.maximum(0.0, G_grav * M_bar_tot / (R_cluster_bar + 1e-12)))

                            rng = np.random.default_rng(seed=42)
                        
                            v_bar = rng.normal(loc=v_center_bar, scale=sigma_fit, size=len(r_proj_kpc))
                            v_dm  = rng.normal(loc=v_center_tot, scale=sigma_fit, size=len(r_proj_kpc))

                        
                        
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
                
                        
                        
                        
                            chi2_hist = np.sum(((np.log1p(counts_obs) - np.log1p(counts_model))**2))
                            chi2_norm = chi2_hist / len(counts_obs)
                            
                            
                            
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



                        



                            mass_ratio = np.mean(m_bary_at_gal / M_total_at_gal)
                            counts_dm, _ = np.histogram(v_dm, bins=bins)
                            counts_dm = counts_dm / counts_dm.sum() * N_obs
                            chi2_scatter = np.sum(((np.log1p(counts_obs) - np.log1p(counts_dm))**2))

                            chi2_scatter_norm = chi2_scatter / len(counts_obs)





                    
                            with plot_container_histo:
                                plot_container_histo.clear()
                                plt.close()
                                with ui.pyplot(figsize=(8, 4)):
                                
                                    plt.hist(observed_vel, bins=bins, alpha=1.0,color='blue', orientation='horizontal',
                label='Observed Velocities', 
                rasterized=True)
        
                                    if len(v_dm) > 0:
                                  
                                        plt.hist(v_dm, bins=bins, alpha=0.7, color='green', orientation='horizontal',
                                                label='Simulated Velocities' , 
                                                rasterized=True)
                                    
                                    if len(v_bar) > 0:
                                    
                                        plt.hist(v_bar, bins=bins, alpha=0.6, color='red', orientation='horizontal',
                                                label='Baryonic Component', 
                                                rasterized=True)
                                        
                                    info_text = (     f'$<v_{{obs}}>={v_mean_val:.1f} km/s \\,   \\sigma_{{obs}} = {sigma_obs:.1f} km/s $\n' 
                   
                                                 f"$ <v_{{sim}}>={v_model_mean:.1f} km/s\\ ,\\sigma_{{sim}} = {v_model_sigma:.1f} km/s $\n"
          
                        f" $ N_{{gal}}={N_obs}   \\,\\sigma_{{bar}} = {sigma_bar_val:.1f} km/s $\n"         
                         f" $\\chi^2={chi2_hist:.1f}\\ ,\\chi^2/N_{{obs}}={chi2_norm:.2f} $")
                                    plt.text(0.96, 0.96, info_text, 
                 transform=plt.gca().transAxes, 
                 fontsize=8, 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 linespacing=1.8, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                                    plt.xlim(0,y_max )
                                    #plt.yscale('log')
                                    plt.ylim(0, x_max + padding )
                                    plt.ylabel('Velocity [km/s]',fontsize=12)
                                    plt.xlabel('N° of Galaxies',fontsize=12)
                                    #if cluster_name is not None:
                                    #    plt.title(f'Galaxy Velocity Distribution ({cluster_name})')
                                    #else:
                                    #    plt.title('Galaxy Velocity Distribution (Cluster)')
                                    plt.title('Galaxy Velocity Distribution (Cluster)',fontsize=14,fontweight='bold')
                                    plt.legend(fontsize=10, loc='upper left')
                                    plt.tight_layout()
                                    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Histogram showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
    )

                        

                                
                                


                    
                        

                            with plot_container_scatter:
                                plot_container_scatter.clear()
                                #plt.close()
                                with ui.pyplot(figsize=(8, 4)):
                            
                                

                                    plt.scatter(r_proj_kpc, observed_vel, s=10, color='blue', alpha=0.6, label="Observed Galaxies", rasterized=True)

                                    plt.scatter(r_proj_kpc, v_bar, s=10, color='red', alpha=0.6, 
                    label='Baryonic Component'   , 
                    rasterized=True)
        

                                    plt.scatter(r_proj_kpc, v_dm, s=10, color='green', alpha=0.6, 
                    label="Simulated Galaxies" , 
                    rasterized=True)

                             
                                    info_text = (
    f"$\\chi^2={chi2_scatter:.1f}\\ ,\\chi^2/N_{{gal}}={chi2_scatter_norm:.2f}$\n"
    f"$\\sigma_{{bar}} ={sigma_bar_val:.1f} \\, km/s\\ ,N_{{gal\\_bar}}={N_bar}$\n"
    f"$\\sigma_{{tot}}= {sigma_dm_val:.1f} \\, km/s\\,N_{{gal\\_sim}}={N_sim}$\n" 
    f"$DM_{{mean}}= {M_dm_mean:.2e} \\, M_\\odot\\ ,M_{{bar}}/M_{{tot}}={mass_ratio:.4e}$"
)
                                    plt.text(0.96, 0.96, info_text, 
                 transform=plt.gca().transAxes, 
                 fontsize=8, 
                 verticalalignment='top', 
                 horizontalalignment='right',
                 linespacing=1.8, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

                                    plt.xlim(0, r_max + r_padding)
                                    plt.ylim(0, ylim_max)


                                    plt.xlabel("Radius [kpc]",fontsize=12)
                                    plt.ylabel("Velocity [km/s]",fontsize=12)
                                    #if cluster_name is not None:
                                    #    plt.title(f"Phase-space diagram of cluster galaxies ({cluster_name})")
                                    #else:
                                    #    plt.title("Phase-space diagram of cluster galaxies")
                                    plt.title("Scatter plot of cluster galaxies velocities",fontsize=14,fontweight='bold')
                                    plt.legend(fontsize=10, loc='upper left')
                                    plt.grid(True, linestyle="--", alpha=0.7)
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
    )
                    
                            

                        except Exception as e:
                            warning_box(f"An error occurred: {e}").classes('text-red-500')
                        
                        
                        
                    
                    

                    
                
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
                    #observed_vel_sub = np.take(observed_vel, indices)
                    #sigma_bary_sub = np.take(sigma_bary, indices)

                    sigma_obs_val = np.std(observed_vel_sub)
                    vx_obs = rng.normal(0.0, sigma_obs_val, N_gal)
                    vy_obs = rng.normal(0.0, sigma_obs_val, N_gal)

                
                    vx_model = rng.normal(0.0, sigma_bary_sub, N_gal)
                    vy_model = rng.normal(0.0, sigma_bary_sub, N_gal)

                    state = {
                        "x_obs": x_init_obs.copy(),
                        "y_obs": y_init_obs.copy(),
                        "x_model": x_init_mod.copy(),
                        "y_model": y_init_mod.copy(),
                        "vx_obs": vx_obs.copy(),
                        "vy_obs": vy_obs.copy(),
                        "vx_model": vx_model.copy(),
                        "vy_model": vy_model.copy(),
                        "rng": rng,
                        "indices": indices
                    }

                    dt = 0.05

            
                    def update_cluster_points():
                        try:
                            f = float(dm_slider.value)
                            cluster_radius = 1.0
                            current_dataset = cluster_state["select"]
                            if state.get("current_dataset") != current_dataset:
                                state["current_dataset"] = current_dataset

                                N_gal_new = min(80, len(observed_vel))
                                cluster_radius = 1.0

                                indices = state["rng"].choice(len(observed_vel), size=N_gal_new, replace=False)

                                x_obs = state["rng"].uniform(-cluster_radius, cluster_radius, N_gal_new)
                                y_obs = state["rng"].uniform(-cluster_radius, cluster_radius, N_gal_new)
                                x_model = state["rng"].uniform(-cluster_radius, cluster_radius, N_gal_new)
                                y_model = state["rng"].uniform(-cluster_radius, cluster_radius, N_gal_new)

                                r_safe = np.maximum(r_proj_kpc, 1.0)
                                M_dm_at_gal = Mc_nfw_enclosed(r_safe, M200, c, rho_crit)
                                M_total_bary = np.maximum(m_bary_at_gal, 1e-6)
                                M_total_withDM = np.maximum(m_bary_at_gal + f*M_dm_at_gal, 1e-6)

                                sigma_bary = np.sqrt(np.maximum(0.0, G_grav * M_total_bary / (3.0 * r_safe)))
                                sigma_withDM = np.sqrt(np.maximum(0.0, G_grav * M_total_withDM / (3.0 * r_safe)))

                                sigma_total_sub = sigma_withDM[indices].mean()
                                sigma_obs_val = np.std(observed_vel[indices])

                                vx_model = state["rng"].normal(0.0, sigma_total_sub, N_gal_new)*0.01
                                vy_model = state["rng"].normal(0.0, sigma_total_sub, N_gal_new)*0.01
                                vx_obs = state["rng"].normal(0.0, sigma_obs_val, N_gal_new)
                                vy_obs = state["rng"].normal(0.0, sigma_obs_val, N_gal_new)

                            
                                state.update({
                                    "indices": indices,
                                    "x_obs": x_obs,
                                    "y_obs": y_obs,
                                    "x_model": x_model,
                                    "y_model": y_model,
                                    "vx_model": vx_model,
                                    "vy_model": vy_model,
                                    "vx_obs": vx_obs,
                                    "vy_obs": vy_obs,
                                    "N_gal": N_gal_new
                                })
                            r_safe = np.maximum(r_proj_kpc, 1.0)
                            M_dm_at_gal = Mc_nfw_enclosed(r_safe, M200, c, rho_crit)
                            M_total_bary = np.maximum(m_bary_at_gal, 1e-6)
                            M_total_withDM = np.maximum(m_bary_at_gal + f*M_dm_at_gal, 1e-6)

                            sigma_bary = np.sqrt(np.maximum(0.0, G_grav * M_total_bary / (3.0 * r_safe)))
                            sigma_withDM = np.sqrt(np.maximum(0.0, G_grav * M_total_withDM / (3.0 * r_safe)))
                            rng = state["rng"] 
                        

                            sigma_total_sub = sigma_withDM[state["indices"]].mean()
                            
                            

                            N_gal = len(state["indices"])
                            state["vx_model"][:] = state["rng"].normal(0.0, sigma_total_sub, N_gal)*0.01
                            state["vy_model"][:] = state["rng"].normal(0.0, sigma_total_sub, N_gal)*0.01
                            sigma_total_sub2 = sigma_total_sub*0.2

                    
                            sigma_obs_val = np.std(observed_vel[state['indices']])
                        
                            state["vx_obs"][:] = state["rng"].normal(0.0, sigma_obs_val, N_gal)
                            state["vy_obs"][:] = state["rng"].normal(0.0, sigma_obs_val, N_gal)

                    
                            state["x_obs"] += state["vx_obs"] * dt
                            state["y_obs"] += state["vy_obs"] * dt
                            state["x_model"] += state["vx_model"] * dt
                            state["y_model"] += state["vy_model"] * dt

                            L = 2*cluster_radius  
                        
                            state["x_obs"] = (state["x_obs"] + cluster_radius) % L - cluster_radius
                            state["y_obs"] = (state["y_obs"] + cluster_radius) % L - cluster_radius
                            state["x_model"] = (state["x_model"] + cluster_radius) % L - cluster_radius
                            state["y_model"] = (state["y_model"] + cluster_radius) % L - cluster_radius

                        
                            with sim_container_obs:
                                sim_container_obs.clear()
                                plt.close()
                                with ui.pyplot(figsize=(4, 4)):
                                    plt.scatter(state["x_obs"], state["y_obs"], c="blue", s=20, alpha=0.8, label="Observed")
                                    plt.xlim(-cluster_radius, cluster_radius)
                                    plt.ylim(-cluster_radius, cluster_radius)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Observed Galaxies",fontweight='bold')
                                    plt.legend(loc="upper right")
                                plot_info_box_compact({
                                    "σ_obs": f"{sigma_obs_val:.0f} km/s"
                                
                                })

                            with sim_container_model:
                                sim_container_model.clear()
                                plt.close()
                                with ui.pyplot(figsize=(4, 4)):
                                    plt.scatter(state["x_model"], state["y_model"], c="green", s=20, alpha=0.8, label="Simulated")
                                    plt.xlim(-cluster_radius, cluster_radius)
                                    plt.ylim(-cluster_radius, cluster_radius)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Simulated Galaxies",fontweight='bold')
                                    plt.legend(loc="upper right")
                                plot_info_box_compact({
                                    "Simulated σ": f"{sigma_total_sub2:.0f} km/s",
                                    "DM factor": f"{f:.2f}"
                                })

                        except Exception as e:
                            accessible_notify(f"Simulator error: {e}", type_="warning")


                    refresh_cluster_plots()
                        #ui.timer(0.05, refresh_cluster_plots.refresh, once=True)
                


                    ui.timer(dt, update_cluster_points)



                    

        
                with ui.tab_panel(four).props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Mass & Virial Theorem")
                    with ui.dialog() as inst_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Cluster Mass Analysis</h3>
        <p>Imagine analyzing a <b>galaxy cluster</b> to reveal dark matter.</p>
        <ul>
            <li>Choose a dataset and complete the missing fields.</li>
            <li>Click <b>Run Analysis</b> to compare <b>luminous mass</b> vs <b>total (virial) mass</b>.</li>
        </ul>
    """).props('aria-label="Descriptive text about galaxy cluster mass and density activities"')
                        aria_button("Close", "Close the box",on_click=inst_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.dialog() as datac_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label=Info about cluster dataset'):
                        info_box("**Dataset variables**: Cluster (name), ID (galaxy ID), RAdeg (right ascension in degrees), DEdeg (declination in degrees), RV (radial velocity in km/s), e_RV (velocity error in km/s), q_RV (quality flag),Nint (n.emission/absorbed lines), bmag (apparent magnitude in B band).")
                                            
                        reference_box("""**Dataset reference**: Way M.J. et al., *Redshifts in the Southern Abell Redshift Survey Clusters*. I. The Data'""").classes('text-base italic')
                        aria_button("Close", "Close the box",on_click=datac_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as formula_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Computational Notes</h3>

        <ul>
            <li><b>Step 1:</b> Compute angular separation (radians):<br> <span class="math">\( \Delta \theta \)</span></li>
            <li><b>Step 2:</b> Compute radius (<span class="math">\( D = 100 \,\mathrm{Mpc} \)</span>):<br> <span class="math">\( r = D \cdot \Delta \theta \)</span> (kpc)</li>
            <li><b>Step 3:</b> Compute observed velocity:<br> <span class="math">\( v_i = c \, z_i \)</span></li>
            <li><b>Step 4:</b> Distance modulus:<br> <span class="math">\( \text{distmod} = 5 \log_{10}(D_{\mathrm{pc}}) - 5 \)</span></li>
            <li><b>Step 5:</b> Absolute magnitude:<br> <span class="math">\( M_{\mathrm{abs}} = b_{\mathrm{mag}} - \text{distmod} \)</span></li>
            <li><b>Step 6:</b> Luminosity (<span class="math">\( M_{\odot,B} = 5.48 \)</span>):<br> <span class="math">\( L_B = 10^{-0.4 (M_{\mathrm{abs}} - M_{\odot,B})} \)</span></li>
            <li><b>Step 7:</b> Total cluster luminosity:<br> <span class="math">\( L_r = \sum_i L_{B,i} \)</span></li>
            <li><b>Step 8:</b> Luminous mass (<span class="math">\( M/L = 5 \)</span>):<br> <span class="math">\( M_{\mathrm{lum}} = (M/L) \cdot L_r \)</span></li>
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
    """).props('role=dialog aria-modal=true aria-label="Mathematical explanation of virial theorem"')
                        aria_button("Close", "Close the box",on_click=formula_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"') as legend_dialog, ui.card().classes("p-4 w-full max-w-2xl overflow-x-auto"):
                        html_info_box(r"""
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
            <li><b>dist_mod</b> = Distance modulus</li>
            <li><b>M<sub>abs</sub></b> = Absolute magnitude of a galaxy [mag]</li>
            <li><b>m</b> = Mass of a light single particle or galaxy [M<sub>&odot;</sub>]</li>
            <li><b>M</b> = Mass inside radius r of heavy particle or total enclosed mass [M<sub>&odot;</sub>]</li>
            <li><b>M<sub>circ</sub></b> = Mass of a particle in circular orbit [M<sub>&odot;</sub>]</li>
            <li><b>v&sup2;</b> = Squared orbital velocity [km&sup2;/s&sup2;]</li>
            <li><b>L<sub>B,i</sub></b> = Luminosity of a single galaxy [L<sub>&odot;</sub>]</li>
            <li><b>bmag</b> = Apparent magnitude in B band [mag] (dataset)</li>
        </ul>
    """).props('id="legend-desc"')
                        aria_button("Close","Close the box", on_click=legend_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                
                    with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog, ui.card().classes("p-4 w-full max-w-2xl overflow-x-auto"):
                        html_info_box(r"""
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
    """).props('id="units-desc"')
                
                        aria_button("Close", "Close the box",on_click=units_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.row().classes('w-full justify-center '):
                        cluster_files = [f for f in os.listdir(CLUSTER_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                        cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)

                        if not cluster_file_map:
                            ui.label("No cluster data files found.").classes('text-red-500')
                        else:
                        
                            cluster_select = ui.select(cluster_files, label='Select a Cluster Dataset').classes('flex-1 max-w-md items-start text-lg')
                        aria_button("Instruction", "Read instruction for cluster activities",on_click=safe_click(lambda: [inst_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("Dataset", "Read dataset info",on_click=safe_click(lambda: [datac_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [formula_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📘 Legend", "Read the legend of symbols and units",on_click=legend_dialog.open).classes(
    "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📐 Units","Read units conversion", on_click=units_dialog.open).classes(
        "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                
                

                    
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
                        answer_sigma,answer_r,answer_g2,answer_vel,answer_vel_mean = {}, {},{},{},{}
                        ans_U, ans_Fg, ans_Fc, ans_K, ans_v2 = {}, {}, {}, {}, {}
                        ans_virialLHS, ans_virialRHS, ans_Mcirc, ans_Mvir,answer_LB = {}, {}, {}, {}, {}
                        answer_D, answer_Dmod, answer_Mabs, answer_Lsum,answer_Mlum = {}, {}, {}, {},{}
                        @ui.refreshable
                        def show_cluster_pseudocode():
                         
                            with ui.card().classes("items-center justify-center p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                                ui.markdown("Cluster Mass Exercise: fill in the missing parts in the pseudo-code below")\
                                    .classes("text-2xl font-bold text-blue-300 mb-4 item-center")\
                                    .props('aria-label=cluster exercises fill the formulas in the boxes')
                                
                              
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
                                            ans_U['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 3) FORCE GRAV
                                        ui.label("3) Compute the gravitational force").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('F<span class="math-sub">grav</span> = ')
                                            ans_Fg['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 4) FORCE CENT
                                        ui.label("4) Compute the centripetal force").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('F<span class="math-sub">cent</span> = ')
                                            ans_Fc['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 5) EQUATE
                                        ui.label("5) Equate forces: F_grav = F_cent → derive v²(R)").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('v<span class="math-sup">2</span>(R) = ')
                                            ans_v2['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 6) KINETIC
                                        ui.label("6) Kinetic energy of the light particle").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('K = ')
                                            ans_K['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 7) VIRIAL THEOREM
                                        ui.label("7) Virial theorem (time-averaged, bound system)").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('2&lang;') 
                                            ans_virialLHS['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html('&rang; + &lang;') 
                                            ans_virialRHS['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html('&rang; = 0')

                                        # 8) CIRCULAR MASS
                                        ui.label("8) Compute circular orbit mass of one particle").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">circ</span>(r) = ')
                                            ans_Mcirc['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 9) TOTAL MASS SYSTEM
                                        ui.label("9) Compute the total mass for a system of N particles:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">tot</span>(r) &asymp; &Sigma;') # Simboli ≈ e Somma
                                            ans_Mvir['el'] = aria_formula_input().props('dense filled').classes('w-12')

                            
                                    

                                        # 10) VELOCITY DISPERSION
                                        ui.label("10) Compute velocity dispersion").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('&sigma;<span class="math-sub">v</span><span class="math-sup">2</span> = ')
                                            ui.html(' (1/N) &middot; ')
                                          
                                            ui.html('&Sigma; |')
                                            answer_vel['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html('&minus; mean(')
                                            answer_vel_mean['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html(')|<span class="math-sup">2</span>')
                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 11) VIRIAL TOTAL MASS
                                        ui.label("11) Compute virial/total mass").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">total</span>(r) = ')
                                          
                                            with ui.column().classes('fraction'):
                                                with ui.row().classes('numerator items-center gap-1'):
                                                    ui.html('3 &middot; ')
                                                    answer_sigma['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                                    ui.html('<span class="math-sup">2</span> &middot; ')
                                                    answer_r['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                                with ui.row().classes('denominator w-full justify-center'):
                                                    answer_g2['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 12) CLUSTER DISTANCE
                                        ui.label("12) Compute cluster distance in pc:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('D<span class="math-sub">pc</span> = ')
                                            answer_D['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 13) DISTANCE MODULUS
                                        ui.label("13) Compute distance modulus:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('dist_mod = 5 &middot; log<span class="math-sub">10</span>(')
                                            answer_Dmod['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html(') &minus; 5')

                                        # 14) ABSOLUTE MAGNITUDE
                                        ui.label("14) Compute absolute magnitude from apparent magnitude (dataset) and distance:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">abs</span> = ')
                                            answer_Mabs['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html('&minus; dist_mod')

                                        # 15) LUMINOSITY
                                        ui.label("15) Compute luminosity for each galaxy:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            
                                            ui.html('L<span class="math-sub">B</span> = 10<span class="math-sup">-0.4 &middot; (</span>')
                                            answer_LB['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html('<span class="math-sup"> &minus; M<span class="math-sub">sun,B</span>)</span>')
                                    with ui.column().classes('flex-1 min-w-[400px] gap-2'):
                                        # 16) TOTAL LUMINOSITY
                                        ui.label("16) Compute total luminosity:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('L(r) = &Sigma;(')
                                            answer_Lsum['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                            ui.html(') <span class="text-sm not-italic ml-2">for galaxies with r<span class="math-sub">i</span> &le; r</span>')

                                        # 17) LUMINOUS MASS
                                        ui.label("17) Compute luminous mass:").classes('text-blue-200 font-bold text-lg')
                                        with ui.row().classes('items-center gap-1 math-text'):
                                            ui.html('M<span class="math-sub">lum</span>(r) = (M/L) &middot; ')
                                            answer_Mlum['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                        # 18) PLOT
                                        ui.label("18) Compare mass and density profiles").classes('text-blue-200 font-bold text-lg')
                                        with ui.column().classes('bg-gray-800 p-2 rounded font-mono text-md text-green-300 w-auto inline-block border border-gray-700'):
                                            ui.label('plot(r, M_lum and M_total, label="Luminous and Total Mass")')
                                            ui.label('plot(r, D_lum=M_lum/V and D_total=M_tot/V, label="Luminous and Total Density")')
                            

                                
                        show_cluster_pseudocode()
                        cluster_select.on('update:model-value', lambda e: show_cluster_pseudocode.refresh())

                        
                        cluster_plot_container = ui.row().classes('w-full gap-4 justify-center')
                        @ui.refreshable
                        def update_cluster_mass_profile():
                            selected_file = cluster_select.value
                            if not selected_file:
                                cluster_plot_container.clear()
                                return

                            data_filepath = os.path.join(CLUSTER_DATA_PATH, selected_file)
                            image_filepath = cluster_file_map.get(selected_file)
                            
                            cluster_plot_container.clear()

                            with cluster_plot_container:
                                try:
                                    
                                    df = pd.read_csv(data_filepath, sep=r'\s+', header=None)
                                    if df.shape[1] == 7:
                                        df[7] = np.nan
                                    df = df.iloc[:, :8]
                                    df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
                                    for col in ["RAdeg", "DEdeg", "RV", "bmag"]:
                                        df[col] = pd.to_numeric(df[col], errors="coerce")
                                    df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])

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

                                    def positive_floor(arr):
                                        pos = arr[np.isfinite(arr) & (arr > 0)]
                                        if pos.size == 0:
                                            return arr + 1e-6
                                        floor = np.nanmin(pos) * 1e-3
                                        return np.where(arr <= 0, floor, arr)

                                    M_tot_r = positive_floor(M_tot_r)
                                    M_lum_r = positive_floor(M_lum_r)

                                    with ui.column().classes('flex-1 items-center'):  
                                        with ui.pyplot(figsize=(8,6)):
                                    
                                            mask_lum = M_lum_r > 0
                                            plt.plot(R_cum[mask_lum], M_lum_r[mask_lum], label='Luminous Mass', color='red')
                                            plt.plot(R_cum, M_tot_r, label='Total Mass (Virial)', color='blue')
                                            plt.xscale('log'); plt.yscale('log')
                                            plt.xlabel('Radius (kpc)'); plt.ylabel('Mass ($M_\\odot$)')
                                            plt.title(f'Mass Profile {os.path.splitext(selected_file)[0]}',fontweight='bold')
                                            plt.grid(True, which="both", ls="--"); plt.legend()
                                            ui.element('div').props(
        'role=img tabindex=0 aria-label=Plot showing the mass profile (in solar mass) in function to the radius (in kpc) of the selected galaxy cluster, comparing luminous mass and total mass'
    )
                                
                                        rho_lum_cum = M_lum_r / ((4.0/3.0) * np.pi * R_cum**3)
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
                                    
                                    
                                    
                        @ui.refreshable
                        def update_cluster_popup_plots():
                            cluster_popup_container.clear()

                            selected_file = cluster_select.value
                            if not selected_file:
                                return

                            data_filepath = os.path.join(CLUSTER_DATA_PATH, selected_file)
                            df = pd.read_csv(data_filepath, sep=r'\s+', header=None)

                            if df.shape[1] == 7:
                                df[7] = np.nan
                            df = df.iloc[:, :8]
                            df.columns = ["Cluster", "ID", "RAdeg", "DEdeg", "RV", "e_RV", "q_RV", "bmag"]
                            for col in ["RAdeg", "DEdeg", "RV", "bmag"]:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            df = df.dropna(subset=["RAdeg", "DEdeg", "RV", "bmag"])

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

                            # positive floor to avoid zeros
                            def positive_floor(arr):
                                pos = arr[np.isfinite(arr) & (arr > 0)]
                                if pos.size == 0:
                                    return arr + 1e-6
                                floor = np.nanmin(pos) * 1e-3
                                return np.where(arr <= 0, floor, arr)

                            M_tot_r = positive_floor(M_tot_r)
                            M_lum_r = positive_floor(M_lum_r)

                            rho_lum_cum = M_lum_r / ((4.0/3.0) * np.pi * R_cum**3)
                            rho_tot_cum = M_tot_r / ((4.0/3.0) * np.pi * R_cum**3)

                            with cluster_popup_container:
                                with ui.pyplot(figsize=(8, 6)):
                                    plt.plot(R_cum, M_lum_r/1e9, label="Luminous Mass", color="red", lw=2)
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

                                
                        with ui.dialog() as cluster_plots_popup, ui.card().classes("p-4 w-full max-w-[1300px]"):
                            ui.label("Cluster Plots").classes("text-2xl font-bold mb-4")
                            cluster_popup_container = ui.column().classes("w-full items-center")
                            aria_button("Close", "close popup", on_click=cluster_plots_popup.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        with ui.row().classes('w-full justify-center'):
                            with ui.column().classes('flex-1 items-center '):

                                aria_button("Run Analysis", "Ruon the analysis to make the plots",on_click=check_and_run_cluster).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                            with ui.column().classes('flex-1 items-center '):
                                
                                aria_button("Open Cluster Plots",             "Open cluster mass & density plots",    on_click=lambda: [update_cluster_popup_plots(), cluster_plots_popup.open()]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        cluster_select.on('update:model-value', lambda e: update_cluster_popup_plots.refresh())


                with ui.tab_panel(five).props('role=tabpanel'):
                    #with ui.card().classes("p-4 !bg-gradient-to-r from-pink-500 to-red-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cosmic Microwave Background (CMB) & Universe Composition")
                    with ui.dialog() as ref_dialog, ui.card().classes("p-4 w-full max-w-[1300px]").props('role=dialog aria-label=References'):
                        reference_box("""
    **Planck Simulator References:**  
    - [Chris North Planck Simulator](https://chrisnorth.github.io/planckapps/Simulator)  
    - [UCSB Planck Project](https://www.deepspace.ucsb.edu/projects/planck)  
    
    """)
                        
                        reference_box("""

    **Images References:**  
    - [ESA Planck Overview](https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck)  
    - [Planck Image Archive](https://planck.ipac.caltech.edu/page/contacts)  
    """)
                        aria_button("Close", "close popup", on_click=ref_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    def open_slides_DM():
                        ui.run_javascript('window.open("/slides/Dark_matter.pdf", "_blank")')


                    with ui.row().classes('w-full gap-8 justify-center'):
                    
                        aria_button('Additional Information:Numerical Explanation',"Additional Information:Numerical Explanation", on_click=open_slides_DM).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        aria_button('References',"References", on_click=ref_dialog.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        
                
                        
                            
                        aria_button('Explore the Planck CMB Simulator',"Explore the interactive Plank CMB simulator",
            on_click=safe_click(lambda: ui.run_javascript("window.open('https://chrisnorth.github.io/planckapps/Simulator', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                            
                        aria_button('ESA: The Universe According to Planck', "ESA reference website",on_click=safe_click(lambda: ui.run_javascript("window.open('https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck','_blank')"))).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        
                    with ui.row().classes('w-full mx-auto place-items-center justify-center gap-8'):
                        
                        aria_image('/images/univ_composition.jpg', "Image of the universe composition with dark energy,dark matter and ordinary matter").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                        aria_image('/images/power_spectrum.jpg', "Image of the power spectrum of cosmic microwave background").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                        aria_image('/images/CMB_asymmetry.jpg', "Image of CMB asymmetry in the universe").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                        aria_image('/images/Planck_freq.jpg', "Image of the plank frequency spectrum").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
            
            
            






    



