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
# Librerie Web/App
import requests
from dotenv import load_dotenv
from fastapi import Request
from groq import Groq
#from ipywidgets import interact, FloatSlider
from nicegui import ui, app, run , client
#from nicegui_toolkit import inject_layout_tool
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
    @ui.page('/module4')
    def module4():
        
#load custom CSS styles
        ui.add_head_html('''
    <link rel="stylesheet" href="/static/github.min.css">
''')
        #<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        #ui.add_head_html('''    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">''')
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
    
    
        #load datasets in cache
        def wavelength_to_rgb(wl):
    # wl in Angstroms. Convert to nm
            nm = wl / 10.0
            r = g = b = 0.0
            if 380 <= nm <= 440:
                r = -(nm - 440) / (440 - 380)
                g = 0.0
                b = 1.0
            elif 440 < nm <= 490:
                r = 0.0
                g = (nm - 440) / (490 - 440)
                b = 1.0
            elif 490 < nm <= 510:
                r = 0.0
                g = 1.0
                b = -(nm - 510) / (510 - 490)
            elif 510 < nm <= 580:
                r = (nm - 510) / (580 - 510)
                g = 1.0
                b = 0.0
            elif 580 < nm <= 645:
                r = 1.0
                g = -(nm - 645) / (645 - 580)
                b = 0.0
            elif 645 < nm <= 750:
                r = 1.0
                g = 0.0
                b = 0.0
            else:
                r = g = b = 0.5
        
            if nm < 420:
                factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
            elif nm > 700:
                factor = 0.3 + 0.7 * (750 - nm) / (750 - 700)
            else:
                factor = 1.0
                r *= factor; g *= factor; b *= factor
            return (r, g, b)
        
       
    #script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
        @lru_cache(maxsize=1)
        def get_supernovae_data_cached():
           
            sn_file = os.path.join(DATA_DIR, "supernovae.txt")
            
            if not os.path.exists(sn_file):
                return None

            try:
              
                df = pd.read_csv(sn_file, sep=r"\s+", comment="#", engine="python")
                
              
                if 'zcmb' not in df.columns and 'z' in df.columns:
                    df['zcmb'] = df['z']
                    
          
                df['zcmb'] = pd.to_numeric(df['zcmb'], errors='coerce')
                df['mb'] = pd.to_numeric(df['mb'], errors='coerce')
                
               
                df = df.dropna(subset=['zcmb', 'mb']).reset_index(drop=True)
                
               
                c = 299792.458
                df['v_nonrel'] = c * df['zcmb']
                term = (1 + df['zcmb'])
                df['v_rel'] = c * ((term**2 - 1) / (term**2 + 1))
                
                return df

            except Exception as e:
                print(f"Error loading supernovae data: {e}")
                return None
        @lru_cache(maxsize=1)
        def get_gal_spectra_catalog_cached():
        
            gal_file = os.path.join(DATA_DIR, "gal_spectra_data.csv")
            
            if not os.path.exists(gal_file):
                print(f"Catalog file not found: {gal_file}")
                return None

            try:
                cols = [
                    "plate", "mjd", "fiberid", "run2d", "specobj_id", "ra", "dec",
                    "sn_median_r", "z", "zerr", "zwarning", "class", "subclass"
                ]
              
                df = pd.read_csv(gal_file, comment='#', sep=',', names=cols, engine='python', dtype=str)

          
                for c in ["plate", "mjd", "fiberid"]:
                    if c in df.columns:
                       
                        df[c] = (df[c].astype(str)
                                    .str.strip()
                                    .str.strip("'\""))
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

               
                if "z" in df.columns:
                    df["z"] = pd.to_numeric(
                        df["z"].astype(str).str.strip().str.strip("'\""), 
                        errors="coerce"
                    )
                
               
                if "zerr" in df.columns:
                    df["zerr"] = pd.to_numeric(
                        df["zerr"].astype(str).str.strip().str.strip("'\""), 
                        errors="coerce"
                    )

               
                df = df.dropna(subset=["plate", "mjd", "fiberid"])
                
                return df

            except Exception as e:
                print(f"Error loading spectra catalog: {e}")
                return None
        
        
        @lru_cache(maxsize=32)
        def get_observed_lines_cached(file_select_value):
            if not file_select_value:
                return None, None
            base_dir = GALAXY_LINE_PATH
            base_name = os.path.splitext(os.path.basename(file_select_value))[0]
            expected_name = f"{base_name}_lines.csv"
            expected_path = os.path.join(base_dir, expected_name)

          
            if os.path.exists(expected_path):
                return pd.read_csv(expected_path), expected_path

           
            prefix = base_name.split("_")[0]
            try:
                candidates = [f for f in os.listdir(base_dir) if f.startswith(prefix) and f.endswith("_lines.csv")]
                if candidates:
                    chosen = os.path.join(base_dir, candidates[0])
                    #print(f"✅ Found file in cache: {candidates[0]}")
                    return pd.read_csv(chosen), chosen
            except OSError:
                pass
                
            return None, None 

        @lru_cache(maxsize=32)
        def get_spectrum_data_cached(filename):
      
            if not filename:
                return None
                
            
            path = os.path.join(GALAXY_SPECTRA_PATH, filename)
            
            if not os.path.exists(path):
                return None

            try:
                df = pd.read_csv(path)
                wl = np.array(df["wavelength_A"], dtype=float)
                flux = np.array(df["flux"], dtype=float)
                
            
                mask = np.isfinite(wl) & np.isfinite(flux)
                return wl[mask], flux[mask]
            except Exception as e:
                print(f"Error caching spectrum {filename}: {e}")
                return None
            
        def get_velocity_dataset_cached():
            
           
            df_base = get_gal_spectra_catalog_cached()
            
            if df_base is None:
                return None

          
            df = df_base.dropna(subset=["z", "zerr"]).copy()

          
            c_km_s = 299792.458
            
           
            df["v_nr"] = c_km_s * df["z"]
            df["v_nr_err"] = c_km_s * df["zerr"]
            
          
            # v = c * ((1+z)^2 - 1) / ((1+z)^2 + 1)
            z = df["z"]
            term = (1 + z) ** 2
            df["v_rel"] = c_km_s * ((term - 1) / (term + 1))
            
            # Derivata: dv/dz = c * 4(1+z) / ((1+z)^2 + 1)^2
            df["v_rel_err"] = c_km_s * (4 * (1 + z) * df["zerr"]) / ((term + 1) ** 2)

            return df
        
    
        main_layout("Redshift & Universe Expansion")
     
        tab_key = 'module4_selected'
        if tab_key not in app.storage.user:
            app.storage.user[tab_key] = 'redshift'
        with ui.tabs().classes('w-full justify-center').bind_value(app.storage.user, tab_key) as tabs:
            ui.tab('redshift', label='Cosmological Redshift').props('aria-label="Activity 1: Cosmological Redshift"')
            ui.tab('hubble', label='Hubble law').props('aria-label="Activity 2: Hubble law"')
            ui.tab('cmb', label='CMB & Dark Energy ').props('aria-label="Activity 3: Cosmic Microwave Background"')
    
    
   #panel redshift         
        spectra_csv_dir = GALAXY_SPECTRA_PATH
       
    

    
        state = {"zs_points": []} 
        def add_redshift_activity(container):

        
            
            with container:
                description_on_dark(
    "Investigate how the stretching of light waves reveals the motion of celestial objects and the expansion of space itself. "
)
                with ui.dialog() as datalam,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    reference_box("""**Dataset reference**: [SDSS-Galaxy spectra](https://www.sdss4.org/dr16/ , https://dr17.sdss.org/ , https://classic.sdss.org/dr6/algorithms/linestable.php), [NIST Atomic Spectra Database](https://www.nist.gov/pml/atomic-spectra-database)""")
                    info_box("**Dataset**: wavelength (Å), flux (W·m⁻²·Å⁻¹).")
                    aria_button("close","close",on_click=lambda: datalam.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

    
                
                with ui.dialog() as info_red, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props("id=redshift-info role=document aria-live=polite"):
                    html_info_box(r"""
    <h3>Galaxy Spectra Exploration</h3>
    <p>Explore how galaxy spectra reveal the cosmic redshift by analyzing emission lines and calculating recession velocities.</p>
    
    <ol style="margin-top: 10px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b>Visualize the Spectrum:</b> Plot the data with <b>Flux</b> on the y-axis and <b>Wavelength</b> on the x-axis for a selected galaxy.
        </li>
        
        <li style="margin-bottom: 8px;">
            <b>Identify Emission Lines:</b> Locate the observed peaks (<span class="math">\(\lambda_{\rm obs}\)</span>) in the spectrum and match them to standard laboratory lines (<span class="math">\(\lambda_0\)</span>). Common tracers include:
            <ul style="list-style-type: disc; margin-left: 20px; margin-top: 5px;">
                <li><b>H<span class="math">\(\alpha\)</span>:</b> 6563 Å</li>
                <li><b>H<span class="math">\(\beta\)</span>:</b> 4861 Å</li>
                <li><b>[O III]:</b> 5007 Å</li>
                <li><b>Ca II K:</b> 3933 Å</li>
            </ul>
        </li>
        
        <li style="margin-bottom: 8px;">
            <b>Calculate Redshift (<span class="math">\(z\)</span>):</b> Verify that the observed lines are shifted toward longer wavelengths (<span class="math">\(\lambda_{\rm obs} > \lambda_0\)</span>). Compute the redshift using the formula:
            <div style="text-align:center; margin: 5px 0;">
                <span class="math">\(z = \frac{\lambda_{\rm obs} - \lambda_0}{\lambda_0}\)</span>
            </div>
            Compare your result with the value reported in the dataset.
        </li>
        
        <li style="margin-bottom: 8px;">
            <b>Compute Velocities:</b> Calculate the recession velocity using two different models.
            <ul style="list-style-type: disc; margin-left: 20px; margin-top: 5px;">
                <li><b>Non-relativistic:</b> <span class="math">\(v = cz\)</span></li>
                <li><b>Relativistic:</b> <span class="math">\(v = c \frac{(1+z)^2 - 1}{(1+z)^2 + 1}\)</span></li>
            </ul>
        </li>
        
        <li style="margin-bottom: 8px;">
            <b>Compare Models:</b> Plot both velocities against redshift. Observe that for small redshift (<span class="math">\(z < 0.1\)</span>) the values are similar, but they diverge significantly at large <span class="math">\(z\)</span>, where relativistic effects become crucial.
        </li>
    </ol>
""").props("id=redshift-info-content role=document aria-live=polite")
                    aria_button("close","close",on_click=lambda: info_red.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as cur_red, ui.card().classes('p-4 w-full max-w-[600px]').props("id=redshift-curiosity role=document aria-live=polite"):
                    html_info_box(r"""
                    <h3>Not Everything is Moving Away!</h3>
                    <p>While most galaxies are redshifting away from us due to cosmic expansion, our neighbor, the <b>Andromeda Galaxy (M31)</b>, shows a <b>Blueshift</b>.</p>
                    <p>Gravity is pulling it towards the Milky Way at 110 km/s. In about 4.5 billion years, the two galaxies will collide and merge.</p>
                    """)
                    reference_box("**Source:** [Space](https://www.space.com/15590-andromeda-galaxy-m31.html)")
                    aria_button("Close", "close", on_click=cur_red.close).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                
                with ui.dialog() as redshift_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props("id=redshift-dialog role=document aria-modal=true aria-labelledby=redshift-dialog-title"):
                    html_info_box(r"""
        <h3>Redshift and Flux Relations</h3>

        <h4>Redshift and Velocity</h4>
        <p>When a galaxy moves away due to the expansion of the Universe, its spectral lines shift toward longer wavelengths (<b>redshift</b>):</p>
        




        <div style="text-align:center; margin: 10px 0;">
            $$z = \frac{\lambda_{\rm obs} - \lambda_0}{\lambda_0}$$
        </div>

        <ul>
            <li><b>Small redshifts:</b> <span class="math">\( v = c \, z \)</span></li>
            <li><b>Relativistic:</b> <span class="math">\( v = c \, \frac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span></li>
        </ul>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Flux–Luminosity Relation</h4>
        <p>The flux <span class="math">\( F(\lambda) \)</span> is the energy received per unit area, time, and wavelength (W·m⁻²·Å⁻¹). For isotropic emission from a source with intrinsic luminosity <span class="math">\( L \)</span> at luminosity distance <span class="math">\( d_L \)</span>:</p>
        
        <p style="text-align:center;"><span class="math">\( F = \frac{L}{4 \pi d_L^2} \)</span></p>
        
        <p>Including the effect of redshift:</p>
        <p style="text-align:center;"><span class="math">\( F(\lambda_{\rm obs}) = \frac{L(\lambda_{\rm emit})}{4 \pi d_L^2 (1+z)} \)</span></p>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Relativistic Doppler Effect</h4>
        <p>For light, the observed frequency changes if the source and/or observer are moving relative to each other.</p>
        <ul>
            <li><b>Source moving away:</b> <span class="math">\( f_{\rm obs} = f_{\rm emit} \sqrt{\frac{1 - v/c}{1 + v/c}} \)</span></li>
            <li><b>Source approaching:</b> <span class="math">\( f_{\rm obs} = f_{\rm emit} \sqrt{\frac{1 + v/c}{1 - v/c}} \)</span></li>
        </ul>
        <p>Redshift corresponds to a wavelength increase (<span class="math">\( \lambda_{\rm obs} > \lambda_{\rm emit} \)</span>), blueshift corresponds to a wavelength decrease.</p>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Common Spectral Lines (SDSS)</h4>
        <p>SDSS provides galaxy spectra (<span class="math">\( \lambda_{\rm obs}, F \)</span>) with key emission and absorption lines:</p>

        <table style="width:100%; border-collapse: collapse; margin-top: 10px; text-align:center;">
            <tr style="background-color: #bae6fd; color: #0c4a6e;">
                <th style="border: 1px solid #0284c7; padding: 8px;">Element / Transition</th>
                <th style="border: 1px solid #0284c7; padding: 8px;">Rest Wavelength (Å)</th>
            </tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">Hα (Hydrogen Balmer)</td><td style="border: 1px solid #0284c7; padding: 8px;">6562.8</td></tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">Hβ (Hydrogen Balmer)</td><td style="border: 1px solid #0284c7; padding: 8px;">4861.3</td></tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">[O III] (Forbidden Oxygen)</td><td style="border: 1px solid #0284c7; padding: 8px;">5006.8</td></tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">[O II] Doublet</td><td style="border: 1px solid #0284c7; padding: 8px;">3727</td></tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">Ca II H &amp; K</td><td style="border: 1px solid #0284c7; padding: 8px;">3968.5 / 3933.7</td></tr>
            <tr><td style="border: 1px solid #0284c7; padding: 8px;">Na I D</td><td style="border: 1px solid #0284c7; padding: 8px;">5890 / 5896</td></tr>
        </table>
        <p style="text-align:center; font-size: 0.9em; margin-top: 5px;"><i>Source: NIST Atomic Spectra Database</i></p>
    """).props('tabindex=0 role=document aria-label="Redshift and Flux Relations information"')
                    
                    aria_button("Close", label="Close the box", on_click=lambda: redshift_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                with ui.row().classes("w-full items-center justify-center"):
                    csv_files = sorted([f for f in os.listdir(spectra_csv_dir) if f.lower().endswith(".csv")])
                    default_file = csv_files[0] if csv_files else None
                    if not csv_files:
                        ui.label("No CSV spectra found in folder: " + spectra_csv_dir).classes("text-red-600")

                
                    file_select = ui.select(csv_files, label="Choose spectrum CSV",value=default_file).classes("w-80").props('aria-describedby=tabs_desc aria-label="choose spectrum file"')
                    plot_btn = aria_button("Plot Spectrum", "plot galaxy spectrum").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Instructions","Instruction",on_click=lambda: [info_red.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Dataset","Data",on_click=lambda: datalam.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Info",label="Read detailed information about Redshift and Flux Relations",on_click=lambda: [redshift_dialog.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Curiosity", "Open curiosity", on_click=cur_red.open).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
            
            

            
                    known_lines = {
            "Hα": 6562.8,
            "Hβ": 4861.3,
            "Hγ": 4340.5,
            "Hδ": 4101.7,
            "[O III]": 5006.8,
            "[N II]": 6583.5,
            "[S II]": 6716.4,
            "[He II]": 4685.7,
            "[O II]": 3727.0,
            "[Ne III]": 3869.0,
            "[O I]": 6300.3,
            "[S II] 6731": 6730.8,
            "He I 5876": 5875.6,
            "Mg II": 2798.0,
            "Na D": 5892.0,
            "Ca K": 3933.7,
            "Ca H": 3968.5,
        }
                    def load_redshift_for_file(file_select_value):
                        base = os.path.basename(file_select_value)
                        parts = os.path.splitext(base)[0].split("-")   # spec-0266-51602-0004
                        if len(parts) < 4:
                            raise ValueError(f"Filename {base} doesn't match expected pattern 'spec-PLATE-MJD-FIBERID'")
                        _, plate_s, mjd_s, fiberid_s = parts
                        try:
                            plate = int(str(plate_s).strip().strip("'\""))
                            mjd = int(str(mjd_s).strip().strip("'\""))
                            fiberid = int(str(fiberid_s).strip().strip("'\""))
                        except ValueError:
                            raise ValueError(f"Invalid numeric ID in filename: {base}")

                   
                        df = get_gal_spectra_catalog_cached()
    
                        if df is None:
                            raise ValueError("Could not load galaxy spectra catalog.")

                    

                        match = df[
                            (df["plate"] == int(plate)) &
                            (df["mjd"] == int(mjd)) &
                            (df["fiberid"] == int(fiberid))
                        ]

                        if match.empty:
                            raise ValueError(f"No matching redshift entry for {file_select_value} (plate={plate}, mjd={mjd}, fiberid={fiberid})")

                        z_val = match["z"].iloc[0]
                        if pd.isna(z_val):
                            raise ValueError(f"Matching row found but z is missing for {file_select_value}")
                        return float(z_val)

                    def get_z_for_current_file():
                        if not file_select.value:
                            return None
                        try:
                            return load_redshift_for_file(file_select.value)
                        except Exception as e:
                            accessible_notify(f"Error loading redshift: {e}", type_="warning")
                            return None

                    def load_observed_lines(file_select_value):
                      
                        result = get_observed_lines_cached(file_select_value)
                        
                  
                        if result is None or result[0] is None:
                            if file_select_value:
                                base_name = os.path.splitext(os.path.basename(file_select_value))[0]
                            else:
                                base_name = "unknown"
                          
                            raise FileNotFoundError(f"No file *_lines.csv found for {base_name}")
                            
                        return result
                       
                    def plot_spectrum():
                        spec_plot_container.clear()
                        if not file_select.value:
                            accessible_notify("Select a spectrum CSV first.", type_="warning")
                            return

                        data = get_spectrum_data_cached(file_select.value)
    
                        if data is None:
                            accessible_notify("Error loading spectrum file.", type_="error")
                            return
                        wl, flux = data
                      
                   
                        if wl.size == 0:
                            accessible_notify("No usable data in spectrum.", type_="error")
                            return

                        measured = []
                        for line_name, lambda0 in known_lines.items():
                            if z_global is None:
                                lambda_obs_calc = ""
                            else:
                                lambda_obs_calc = lambda0 * (1 + z_global)

                            if wl.min() <= lambda_obs_calc <= wl.max():
                                measured.append({"line": line_name, "lambda_obs": lambda_obs_calc})

                        with spec_plot_container:
                            plt.close()
                            with ui.pyplot(figsize=(15, 5)):
                                plt.plot(wl, flux, lw=1, color="blue", label="Flux")
                                for m in measured:
                                    λ = m["lambda_obs"]
                                    label = m["line"]
                                    plt.axvline(λ, color="black", ls="--", alpha=0.6)
                                    plt.text(λ, np.nanmax(flux) * 0.95, f"{label}\n({λ:.1f} Å)",
                                            rotation=90, va="top", ha="center", fontsize=8)
                                plt.xlabel("Wavelength [Å]")
                                plt.ylabel("Flux (raw units)")
                                plt.title(f"Spectrum: {file_select.value}",fontweight='bold')
                                plt.grid(alpha=0.3)
                                plt.tight_layout()
                                aria_chart_label("Galaxy spectrum plot with flux vs wavelength and marked spectral lines")

                            with ui.pyplot(figsize=(15, 0.3)):
                                xmin, xmax = wl.min(), wl.max()
                                ncols = 400
                                xs = np.linspace(xmin, xmax, ncols)
                                grad = np.zeros((1, ncols, 3))

                                def wavelength_to_rgb(wavelength_A):
                                    nm = wavelength_A / 10.0
                                    if nm < 380 or nm > 750:
                                        return (0.5, 0.5, 0.5)
                                    if 380 <= nm < 440:
                                        r = -(nm - 440) / (440 - 380); g = 0.0; b = 1.0
                                    elif 440 <= nm < 490:
                                        r = 0.0; g = (nm - 440) / (490 - 440); b = 1.0
                                    elif 490 <= nm < 510:
                                        r = 0.0; g = 1.0; b = -(nm - 510) / (510 - 490)
                                    elif 510 <= nm < 580:
                                        r = (nm - 510) / (580 - 510); g = 1.0; b = 0.0
                                    elif 580 <= nm < 645:
                                        r = 1.0; g = -(nm - 645) / (645 - 580); b = 0.0
                                    else:
                                        r = 1.0; g = 0.0; b = 0.0
                                    return (r, g, b)

                                for j, xv in enumerate(xs):
                                    grad[0, j, :] = wavelength_to_rgb(xv)

                                plt.imshow(grad, aspect='auto', extent=(xmin, xmax, 0, 1))
                                plt.axis('off')

                    

                    

                    with ui.dialog() as tab_spectra,ui.card().props('aria-label="spectral lines table" role=dialog'):
                        

                        try:
                            if file_select.value:
                                df_obs, obs_path = load_observed_lines(file_select.value)
                                df_obs.columns = df_obs.columns.str.strip().str.lower()
                            
                                #lambda_obs_dict = dict(zip(df_obs["line"], df_obs["lambda_obs"]))
                            
                                    
                            else:
                                lambda_obs_dict = {}
                        except FileNotFoundError:
                            accessible_notify("No matching *_lines.csv file found for the selected spectrum.", type_="warning")
                            #lambda_obs_dict = {}

                        z_global = get_z_for_current_file()

                        def update_z_on_file_change(e):
                            nonlocal z_global
                            z_global = get_z_for_current_file()

                        file_select.on("update:model-value", update_z_on_file_change)

                        student_rows = []
                        for line, lambda0 in known_lines.items():
                        
                            student_rows.append({
                                "line": line,
                                "lambda0": f"{lambda0:.1f}",
                                "lambda_obs": "",
                                "z": ""
                            })

                        student_table = aria_table(
                            columns=[
                                {"name": "line", "label": "Line", "field": "line"},
                                {"name": "lambda0", "label": "λ₀ [Å]", "field": "lambda0"},
                                {"name": "lambda_obs", "label": "λ_obs [Å]", "field": "lambda_obs"},
                                {"name": "z", "label": "z = (λ_obs−λ₀)/λ₀", "field": "z"},
                            ],
                            rows=student_rows,
                            label="Table of known spectral lines for redshift calculation",
                            row_key="line"
                        ).classes("w-full")

                        with ui.row().classes(" items-center gap-3"):
                            selected_line = ui.select([r["line"] for r in student_rows], label="Select line").classes("w-44")
                            z_input = aria_input(label="Enter redshift z", aria_label="Enter redshift z", placeholder="e.g. 0.02").classes("w-32")
                            update_z_btn = aria_button("Add z value", "add z redshift value").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            check_btn = aria_button("Check table results", "check table results").classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")

                        def update_z_table():
                            line = selected_line.value
                            if not line:
                                accessible_notify("Select a spectral line first.", type_="warning")
                                return
                            try:
                                z = float(z_input.value)
                            except Exception:
                                accessible_notify("Enter a valid numeric z value.", type_="warning")
                                return
                            for r in student_table.rows:
                                if r["line"] == line:
                                    r["z"] = f"{z:.5f}"
                            student_table.update()
                            z_input.value = ""

                        update_z_btn.on("click", update_z_table)

                        with ui.dialog() as result_dialog, ui.card().classes("w-2/3 max-w-3xl").props('aria-label="Measured spectral lines results" role=dialog'):
                            result_dialog_card_label = ui.label("Measured Spectral Lines").classes("text-lg font-bold mb-2").props(
                                'tabindex=0 role=heading aria-level=2'
                            )

                            result_table = aria_table(
                                columns=[
                                    {"name": "line", "label": "Line", "field": "line"},
                                    {"name": "lambda0", "label": "λ₀ [Å]", "field": "lambda0"},
                                    {"name": "lambda_obs", "label": "λ_obs [Å]", "field": "lambda_obs"},
                                    {"name": "z", "label": "z (auto)", "field": "z"},
                                ],
                                rows=[],
                                label="Measured spectral lines results table"
                            ).classes("w-full")
                            with ui.row().classes("justify-end mt-3"):
                                aria_button("Close", "close dialog", on_click=result_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                                
                        def show_solution():
                            if not file_select.value:
                                accessible_notify("Select a spectrum CSV first.", type_="warning")
                                return

                            result_dialog_card_label.set_text(f"Measured Spectral Lines for: {file_select.value}")

                            result_rows = []
                            for line, lambda0 in known_lines.items():
                                if z_global is None:
                                    lambda_obs_calc = ""
                                else:
                                    lambda_obs_calc = lambda0 * (1 + z_global)

                                result_rows.append({
                                    "line": line,
                                    "lambda0": f"{lambda0:.1f}",
                                    "lambda_obs": f"{lambda_obs_calc:.3f}",
                                    "z": f"{z_global:.6f}"
                                })

                            result_table.rows = result_rows
                            result_table.update()
                            result_dialog.open()

                        check_btn.on("click", show_solution)
                        aria_button("Close","close",on_click=lambda: tab_spectra.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Table redshift exercise","Table exercise",on_click=lambda: tab_spectra.open()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    
                    
                   
                    with ui.dialog() as velocity,ui.card().props('aria-label="velocity exercise" role=dialog'):
                        with ui.row().classes('w-full items-center'):
                            html_info_box(r"""
    <p>Insert a value of redshift to compute the velocities:</p>
    <ul>
        <li><b>Non relativistic velocity:</b> <span class="math">\( v = c z \)</span></li>
        <li><b>Relativistic velocity:</b> <span class="math">\( v = c \frac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span></li>
    </ul>
""")
                            z_input = aria_input(label="Enter redshift z", aria_label="Enter redshift z", placeholder="e.g. 0.02").classes("w-48")
                            calc_vel_btn = aria_button("Compute velocity", "compute velocity for single z").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                    
                            vel_table = aria_table(
                                columns=[
                                    {"name": "z", "label": "z", "field": "z"},
                                    {"name": "v_nr", "label": "v_non-rel [km/s]", "field": "v_nr"},
                                    {"name": "v_rel", "label": "v_rel [km/s]", "field": "v_rel"},
                                ],
                                rows=[],
                                label="Computed velocities from redshift table"
                            ).classes("w-128")

                            def compute_single_velocity():
                                try:
                                    z = float(z_input.value)
                                except Exception:
                                    accessible_notify("Enter a valid redshift (numeric).", type_="warning")
                                    return
                                c_km_s = 299792.458
                                v_nr = c_km_s * z
                                v_rel = c_km_s * (((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1))
                                vel_table.rows = [{"z": f"{z:.6f}", "v_nr": f"{v_nr:.1f}", "v_rel": f"{v_rel:.1f}"}]
                                vel_table.update()

                            calc_vel_btn.on("click", compute_single_velocity)

                            plot_vel_btn = aria_button("Plot velocities (dataset)", "plot velocity dataset automatically").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            vel_plot_container = ui.column()

                            def plot_velocity_dataset():
                                vel_plot_container.clear()

                                df = get_velocity_dataset_cached()
    
                                if df is None:
                                    accessible_notify("Error loading velocity data.", type_="error")
                                    return

                                c_km_s = 299792.458
                                

                                with vel_plot_container:
                                    plt.close()
                                    with ui.pyplot(figsize=(7, 5)):
                                        df_plot = df.dropna(subset=["v_nr_err", "v_rel_err"])
                                        df_plot = df_plot[(df_plot["v_nr_err"] >= 0) & (df_plot["v_rel_err"] >= 0)]

                                        plt.errorbar(df_plot["z"], df_plot["v_nr"], yerr=df_plot["v_nr_err"], fmt='o', capsize=4,
                                                        color='tab:blue', label='v_non-rel')
                                        plt.errorbar(df_plot["z"], df_plot["v_rel"], yerr=df_plot["v_rel_err"], fmt='o', capsize=4,
                                                        color='tab:green', label='v_relativistic')
                                        

                                        zs_plot = np.linspace(0, max(df["z"].max() * 1.2, 0.2), 300)
                                        v_nr_curve = c_km_s * zs_plot
                                        v_rel_curve = c_km_s * (((1 + zs_plot) ** 2 - 1) / ((1 + zs_plot) ** 2 + 1))
                                        plt.plot(zs_plot, v_nr_curve, '--', color='blue', alpha=0.5)
                                        plt.plot(zs_plot, v_rel_curve, '--', color='green', alpha=0.5)

                                        plt.xlabel("Redshift z")
                                        plt.ylabel("Velocity [km/s]")
                                        plt.title("Galaxy velocities from SDSS dataset",fontweight='bold')
                                        plt.legend()
                                        plt.grid(alpha=0.4)
                                        plt.tight_layout()
                                        aria_chart_label("Galaxy velocities plot from redshift dataset with non-relativistic and relativistic velocities")

                        
                            plot_vel_btn.on("click", plot_velocity_dataset)
                        aria_button("Close","close",on_click=lambda:velocity.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Velocity exercise","velocity exercise",on_click=lambda: [velocity.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    
                    spec_plot_container = ui.column().classes('w-full items-center justify-center')
                    plot_btn.on("click", plot_spectrum)
                    file_select.on("update:model-value", plot_spectrum)
                    if default_file:
                        ui.timer(0.1, plot_spectrum, once=True)

                        

                    




      
#functions for cosmological calculations
        def omega_r_from_Tcmb_h(Tcmb=2.7255, h=0.7, Neff=3.046):
            # Omega_gamma h^2 ≈ 2.472e-5 * (Tcmb/2.7255)^4
            Omega_gamma_h2 = 2.472e-5 * (Tcmb / 2.7255)**4
            # neutrino relativistic contribution: ≈ Omega_gamma * 0.2271 * Neff
            Omega_r_h2 = Omega_gamma_h2 * (1.0 + 0.2271 * Neff)
            return Omega_r_h2 / (h**2)


        #Omega_r = omega_r_from_Tcmb_h(Tcmb=2.7255, h=0.7, Neff=3.046)  # ≈ 8e-5-9e-5

        def dL_Mpc_LCDM_full(z, H0=70.0, Omega_m=0.3, Omega_Lambda=0.7, Omega_r=9e-5, Omega_k=None):
            
            c = 3e5  # km/s
            if Omega_k is None:
                Omega_k = 1.0 - (Omega_m + Omega_Lambda + Omega_r)
            def integrand(zp):
                Ez = np.sqrt(Omega_r*(1+zp)**4 + Omega_m*(1+zp)**3 + Omega_k*(1+zp)**2 + Omega_Lambda)
                return 1.0 / Ez
            integral, _ = quad(integrand, 0.0, float(z))
            return (1.0 + z) * (c / float(H0)) * integral
        def dL_Mpc_noLambda(z, H0):#model Einstein de sitter
            c = 299792.458
            z = np.atleast_1d(z)
            dL = np.zeros_like(z, dtype=float)
            for i, zi in enumerate(z):
                integrand = lambda zp: 1.0 / np.sqrt(1*(1+zp)**3 + 0.0)  
                integral, _ = quad(integrand, 0.0, float(zi))
                dL[i] = (c / H0) * (1 + zi) * integral
            return dL if len(dL) > 1 else dL[0]

        def dL_Mpc_LCDM(z, H0):
            c = 299792.458
            z = np.atleast_1d(z)
            dL = np.zeros_like(z, dtype=float)
            for i, zi in enumerate(z):
                integrand = lambda zp: 1.0 / np.sqrt(0.3*(1+zp)**3 + 0.7)  # Ωm=0.3, ΩΛ=0.7
                integral, _ = quad(integrand, 0.0, float(zi))
                dL[i] = (c / H0) * (1 + zi) * integral
            return dL if len(dL) > 1 else dL[0]




        h = 0.7
        Tcmb = 2.7255
        Neff = 3.046
        #Omega_r = omega_r_from_Tcmb_h(Tcmb=Tcmb, h=h, Neff=Neff)
        #Omega_r = 8.6e-5  
        #Omega_m_h2 = 0.142   #  da CMB / fit
        #Omega_m = Omega_m_h2 / (h**2)
        #Omega_Lambda = 1.0 - (Omega_r + Omega_m)
        G = 6.67430e-11
        #H0_km_s_Mpc = 70.0
        #H0 = H0_km_s_Mpc * 1000.0 / 3.085677581e22  # s^-1
        #rho_crit = 3 * H0**2 / (8 * np.pi * G)
        
  #panel for hubble law and cosmological distance measurements      
        def add_hubble_activity(container):
        

            with container:
                description_on_dark(
    "Use Type Ia Supernovae as 'standard candles' to measure vast cosmological distances and uncover the accelerating expansion of the Universe."
)
                
                
                with ui.dialog() as datas_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto').props('aria-label="information about supernovae dataset"'):
                    reference_box("""**Dataset reference**: [Supernovae data Pantheon-SH0ES](https://github.com/dscolnic/Pantheon , https://pantheonplussh0es.github.io/)""")
        
                    info_box("**Dataset**: zcmb (CMB frame redshift), mb (apparent B magnitude), err_mag.")
                    aria_button("Close", label="Close", on_click=lambda: datas_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as cur_hub, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="curiosity about faster than light recession"'):
                    html_info_box(r"""
                    <h3>Faster Than Light?</h3>
                    <p>Hubble's Law ($v = H_0 d$) implies that galaxies sufficiently far away are receding from us faster than the speed of light ($c$).</p>
                    <p><b>Does this violate relativity?</b> No! Special relativity says nothing can move *through* space faster than light. However, space itself can expand at any speed.</p>
                    """)
                    reference_box("**Source:** [UCLA Cosmology ](https://www.astro.ucla.edu/~wright/cosmology_faq.html#FTL)")
                    aria_button("Close", "close", on_click=lambda:cur_hub.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                
                with ui.dialog() as cosmology_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto').props('aria-label="information about cosmology relations"'):
                    html_info_box(r"""
        <h3>Fundamental Cosmological Relations</h3>
        
        <ul>
            <li>
                <b>Hubble Law:</b><br>
                <span class="math">\( v = H_0 \cdot d_L \)</span><br>
                For small redshifts (<span class="math">\( z \ll 1 \)</span>), <span class="math">\( v \approx c \cdot z \)</span>.
            </li>
            
            <li>
                <b>Distance Modulus:</b><br>
                <span class="math">\( \mu = m_B - M_B = 5 \log_{10}(d_L [\rm pc]) - 5 \)</span><br>
                → connects apparent and absolute magnitude.
            </li>
            
            <li>
                <b>Luminosity Distance:</b><br>
                <span class="math">\( d_L = (1+z)\frac{c}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m (1+z')^3 + \Omega_\Lambda}} \)</span><br>
                → accounts for cosmological expansion (<span class="math">\(\Lambda\)</span>CDM model).
            </li>
            
            <li>
                <b>Scale Factor Relation:</b><br>
                <span class="math">\( 1 + z = \frac{1}{a(t)} \)</span> → expansion of the Universe.
            </li>
        </ul>
    """).props('aria-label="information about cosmology relations"')
                    aria_button("Close", label="Close", on_click=lambda: cosmology_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

        
                    

            
                with ui.dialog() as lcdm_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto').props('aria-label="information about cosmology relations in model with cosmological constant"'):
                    html_info_box(r"""
        <h3>\(\Lambda\)CDM Explanation</h3>
        
        <ul>
            <li><b>Velocity (non-relativistic):</b> <span class="math">\( v = c \cdot z \)</span></li>
            <li><b>Relativistic velocity:</b> <span class="math">\( v = c \frac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span></li>
            <li><b>Distance modulus:</b> <span class="math">\( \mu = m_B - M_B \)</span></li>
            <li><b>Luminosity distance:</b> <span class="math">\( d_L = 10^{(\mu+5)/5} [\rm pc] \)</span></li>
            <li><b>Hubble constant estimate:</b> <span class="math">\( H_0 = \frac{\sum_i v_i d_{L,i}}{\sum_i d_{L,i}^2} \)</span></li>
            <li><b>Residuals:</b> <span class="math">\( r = v - H_0 \cdot d_L \)</span></li>
            <li><b>Scale factor relation:</b> <span class="math">\( 1 + z = \frac{1}{a(t)} \)</span></li>
            <li><b>\(\Lambda\)CDM distance:</b> <span class="math">\( d_L(z) = \frac{c (1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}, \quad E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda} \)</span></li>
            <li><b>Typical parameters:</b> <span class="math">\(\Omega_m = 0.3, \Omega_\Lambda = 0.7, \Omega_k = 0\)</span> (flat Universe)</li>
        </ul>
    """).props('aria-label="information about cosmology relations in model with cosmological constant"')
                    aria_button("Close", label="Close", on_click=lambda: lcdm_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                with ui.dialog() as info_dlg, ui.card().props('aria-modal="true" role="dialog" aria-labelledby="info-title" aria-describedby="info-content"'):
                    ui.label("Information").classes("text-lg font-semibold text-blue-400").props("id=info-title ole=heading aria-level=2 tabindex=0")
    
                    html_info_box(r"""
    
    <h3>Activity Instructions</h3>
    <p>In this exercise, you will analyze Supernovae Ia data to understand the expansion of the Universe:</p>
    <ol class="list-decimal ml-4 space-y-2">
        <li>
            <b>Calculate Velocities & Distance:</b> Using the dataset, compute the non-relativistic velocity (<span class="math">\(v=cz\)</span>) and convert the distance modulus (<span class="math">\(\mu\)</span>) into luminosity distance (<span class="math">\(d_L\)</span>).
        </li>
        <li>
            <b>Determine the Hubble Constant (\(H_0\)):</b> Plot Velocity vs. Distance. Filter the data for low redshift (<span class="math">\(z < 0.1\)</span>) and perform a linear fit to find the slope, which corresponds to <span class="math">\(H_0\)</span>.
        </li>
        <li>
            <b>Relativistic Comparison:</b> For high redshift data (<span class="math">\(z > 0.1\)</span>), calculate the relativistic velocity and observe the discrepancy with the non-relativistic approximation.
        </li>
        <li>
            <b>Compare Cosmological Models:</b> Plot the Distance Modulus vs. Redshift. Compute and overlay the theoretical curves for two models to see which fits the data:
            <ul class="list-disc ml-6 mt-1">
                <li><b>\(\Lambda\)CDM Model (Dark Energy):</b> Use <span class="math">\(\Omega_m = 0.3\)</span> and <span class="math">\(\Omega_\Lambda = 0.7\)</span>.</li>
                <li><b>Matter-Only Model:</b> Use <span class="math">\(\Omega_m = 1.0\)</span> and <span class="math">\(\Omega_\Lambda = 0\)</span>.</li>
            </ul>
        </li>
        <li>
            <b>Conclusion:</b> Identify that the observed supernovae appear dimmer than expected in a matter-only universe, providing evidence for accelerated expansion.
        </li>
    </ol>
     <hr class="my-4">
    <h3>Cosmological Concepts Summary</h3>
    
    <h4>Redshift and scale factor</h4>
    <p><span class="math">\( 1+z = \dfrac{a(t_0)}{a(t_{\rm emit})} \)</span> with <span class="math">\( a(t_0)=1 \Rightarrow 1+z = 1/a(t) \)</span>.</p>

    <h4>Flux and luminosity</h4>
    <p>For isotropic emission: <span class="math">\( F = \dfrac{L}{4\pi d_L^2} \)</span>.<br>
    In wavelengths: <span class="math">\( F(\lambda_{\rm obs}) = \dfrac{1}{1+z}\dfrac{L(\lambda_{\rm emit})}{4\pi d_L^2} \)</span>.</p>

    <h4>Hubble relation & Hubble time</h4>
    <p><span class="math">\( v = \dot a\, r = H_0 r \Rightarrow H_0 = \dot a / a \)</span>.<br>
    Hubble time <span class="math">\( t_0 \approx 1/H_0 \)</span> (in appropriate units; e.g. <span class="math">\( H_0=70 \)</span> km/s/Mpc → <span class="math">\( t_0 \approx 14 \)</span> Gyr).</p>

    <h4>\(\Lambda\)CDM luminosity distance (flat)</h4>
    <p><span class="math">\( d_L(z) = \dfrac{c(1+z)}{H_0}\int_0^z \dfrac{dz'}{E(z')} \)</span>, with <span class="math">\( E(z)=\sqrt{\Omega_m(1+z)^3 + \Omega_k(1+z)^2 + \Omega_\Lambda} \)</span>.<br>
    Typical values: <span class="math">\( \Omega_m=0.3, \Omega_k=0, \Omega_\Lambda=0.7 \)</span>.</p>

    <h4>Velocities</h4>
    <p>Non-relativistic: <span class="math">\( v = c z \)</span>.<br>
    Relativistic: <span class="math">\( v = c\,\dfrac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span>.</p>

   

""").props("id=planck-info-content role=document aria-live=polite aria-describedby=info-content aria-label='Detailed information about redshift, flux, Hubble relation, Lambda CDM luminosity distance, velocities, and activity instructions.'")
                    aria_button("Close", label="Close", on_click=lambda: info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                with ui.dialog() as leg_dlg, ui.card().props('aria-modal="true" role="dialog" aria-labelledby="legend-title" aria-describedby="legend-content"'):
                    ui.label("Legend: Symbols and Units").classes("text-lg font-semibold text-blue-400").props("id=info-title ole=heading aria-level=2 tabindex=0")
    
                    html_info_box(r"""
        <h3>Symbols and Units</h3>
        <ul>
            <li><b>c</b> = 299,792.458 km/s — speed of light</li>
            <li><b>H₀</b> — Hubble constant [km/s/Mpc]</li>
            <li><b>z</b> — redshift (dimensionless)</li>
            <li><b>v<sub>nr</sub>, v<sub>rel</sub></b> — velocities [km/s]</li>
            <li><b>d<sub>L</sub></b> — luminosity distance [Mpc]</li>
            <li><b>μ</b> — distance modulus (mag)</li>
            <li><b>m<sub>B</sub>, M<sub>B</sub></b> — apparent and absolute magnitude</li>
            <li><b>Ωₘ, Ω_Λ</b> — density parameters (matter, dark energy)</li>
            <li><b>a(t)</b> — cosmological scale factor</li>
        </ul>
    """).props("role=document aria-live=polite aria-describedby=info-content aria-label='Legend explaining symbols and units used in the Hubble Law with Type Ia Supernovae activity.'")
                    aria_button("Close", label="Close", on_click=lambda: [leg_dlg.close(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            

                with ui.row().classes("items-center gap-2 justify-center"):
                    
                    aria_button("Instructions ", "Information",on_click=lambda: [info_dlg.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                    aria_button("Dataset","data info",on_click=lambda: [datas_dialog.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                    aria_button("Legend ", "Variable and constants legend",on_click=lambda: [leg_dlg.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                    aria_button(
                        "Fundamental Cosmology Relations",
                        label="Show Fundamental Cosmological Relations",
                        on_click=lambda: [
                            cosmology_dialog.open(),
                            ui.run_javascript("MathJax.typesetPromise()")
                        ]
                    ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                    aria_button(
                        "ΛCDM",
                        label="Show ΛCDM Explanation",
                        on_click=lambda: [
                            lcdm_dialog.open(),
                            ui.run_javascript("MathJax.typesetPromise()")
                        ]
                    ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                    aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_hub.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                    
                    H0_box = ui.number(label="H₀ [km/s/Mpc]", format="%.3f").classes("w-44").props('aria-label="Hubble constant H0 in km per second per Megaparsec"')
                
                    M_B_input = ui.number(label="Absolute Magnitude M_B (SNe Ia)", value=-19.3, format="%.3f").classes("w-52").props('aria-label="Absolute Magnitude of Type Ia Supernovae M sub B"')
                    with ui.column().classes("items-start gap-2"):
                        ui.label("Compute H₀ from data (linear fit v = H₀·dₗ)").classes("text-sm text-blue-600")
                        result_label = ui.label("").classes("text-lg font-semibold text-green-600").props("aria-live=polite ")
                        aria_button("Compute H₀", "compute H0",on_click=lambda: compute_H0()).props("color=primary outline").classes(" w-32 !bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    
                    df_base = get_supernovae_data_cached()
                    c = 299792.458
                    if df_base is None:
                        ui.label(f"File not found or error: supernovae.txt").classes("text-red-600")
                        return
                    df = df_base.copy()
                    try:
                        current_M_B = float(M_B_input.value)
                    except (ValueError, TypeError):
                        current_M_B = -19.3 

                    df['mu'] = df['mb'] - current_M_B
                    df['d_L_Mpc'] = 10**((df['mu'] + 5) / 5) / 1e6

                    state = {'show_vrel': False, 'show_hubble_line': False, 'show_mag_hubble': False,'show_lcdm': False, 'show_noLambda': False}

            
                    plots_row = ui.row().classes("gap-4 mt-4")
                    with plots_row:
                    
                        vdL_col = ui.column().style("flex: 1")
                        vz_col = ui.column().style("flex: 1")
                
                        #mB_col = ui.column().style("flex: 1")
                        mu_col = ui.column().style("flex: 1")
                        
                    with ui.dialog() as tablex,ui.card().props('aria-label="table exercise" role=dialog aria-modal=true'):
                        info_box(" Exercise Table: Fill in missing values applying the correct formulas ")
                        sample = df.sample(10, random_state=2).reset_index(drop=True)
                        rows = [{"name": f"SN{i}", "z": round(float(r['zcmb']),4), "mB": round(float(r['mb']),3),
                                "mu": "", "d_L_Mpc": "", "v_nonrel": "", "v_rel": "", "residual": ""} for i,r in sample.iterrows()]
                        table = aria_table(
                            columns=[{"name":c,"label":c,"field":c} for c in rows[0].keys()],
                            rows=rows,
                            label="Supernovae Data Table"
                        ).classes("w-full")
                        with ui.row().classes("items-center gap-3 mt-3"):
                            sel = ui.select([r['name'] for r in rows], label="Select SN").classes("w-44").props('aria-label="Select supernova from table to update values"')
                            mu_input = aria_input(label="μ [mag]", aria_label="insert distance modulus").classes("w-28")
                            dL_input = aria_input(label="d_L [Mpc]",aria_label="insert distance modulus").classes("w-28")
                            vnr_input = aria_input(label="v_nonrel [km/s]",aria_label="insert non relativistic velocity").classes("w-28")
                            vrel_input = aria_input(label="v_rel [km/s]",aria_label="insert relativistic velocity").classes("w-28")
                            

                        def update_row():
                            for r in table.rows:
                                if r["name"] == sel.value:
                                    if mu_input.value: r["mu"] = mu_input.value
                                    if dL_input.value: r["d_L_Mpc"] = dL_input.value
                                    if vnr_input.value: r["v_nonrel"] = vnr_input.value
                                    if vrel_input.value: r["v_rel"] = vrel_input.value
                            table.update()
                            accessible_notify("Row updated.", type_="success")
                        

                
                        def show_results():
                            result=[]
                            for r in table.rows:
                                row=r.copy()
                            
                                try:
                                    z=float(row['z'])
                                except:
                                    z=None

                            
                                if z is not None:
                                    v_nonrel = c*z
                                    v_rel = c*((1+z)**2 - 1)/((1+z)**2 + 1)
                                    row['v_nonrel'] = f"{v_nonrel:.2f}"
                                    row['v_rel'] = f"{v_rel:.2f}"

                                try:
                                    if row['mu'] and not row['d_L_Mpc']:
                                        mu = float(row['mu'])
                                        dL = 10**((mu+5)/5)/1e6
                                        row['d_L_Mpc'] = f"{dL:.3f}"
                                    elif row['d_L_Mpc'] and not row['mu']:
                                        dL = float(row['d_L_Mpc'])
                                        mu = 5*math.log10(dL*1e6) - 5
                                        row['mu'] = f"{mu:.3f}"
                                    elif (not row['mu']) and (not row['d_L_Mpc']) and z is not None:
                                    
                                        H0v = get_H0_value()

                                        dL = c*z/H0v
                                        mu = 5*math.log10(dL*1e6) - 5
                                        row['d_L_Mpc'] = f"{dL:.3f}"
                                        row['mu'] = f"{mu:.3f}"
                                except:
                                    pass

                                
                                try:
                                    row['residual']=f"{float(row['v_nonrel'])-float(row['v_rel']):.2f}"
                                except:
                                    row['residual']=""

                                result.append(row)

                        
                            with ui.dialog() as dlg, ui.card().props('aria-modal="true" role="dialog" aria-labelledby="completed-table-title" aria-describedby="completed-table-content"'):
                                ui.label("Completed Table ").classes("text-lg font-semibold")
                                aria_table(
                                    columns=[{"name":k,"label":k,"field":k} for k in result[0].keys()],
                                    rows=result,
                                    label="Completed Supernovae Data Table"
                                ).classes("w-full")
                                aria_button("Close", "close dialog",on_click=lambda: dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            dlg.open()
                            
                        with ui.row().classes("gap-3"):
                            aria_button("Add / Update row","add row in table", on_click=update_row).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            aria_button("Show results","show table with results", on_click=show_results).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                        aria_button("Close","close",on_click=lambda: tablex.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    aria_button("Table Exercise","Table exercise",on_click=lambda: tablex.open()).props("color=primary outline").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                    
                    def get_H0_value():
                        try:
                        
                            val = H0_box.value
                            if val is None or (isinstance(val, str) and val.strip() == ""):
                                return 70.0
                            return float(val)
                        except (TypeError, ValueError):
                            return 70.0



            
                    def compute_H0():
                        c = 299792.458
                        df_lowz = df[df['zcmb'] < 0.1]
                        x = df_lowz['d_L_Mpc']
                        y = df_lowz['v_nonrel']
                        valid = (x > 0) & (y > 0)
                        x = x[valid]; y = y[valid]
                        #coeffs = np.polyfit(x, y, 1)
                        #H0_est = coeffs[0]
                        H0_est = np.sum(x * y) / np.sum(x**2)

                        result_label.set_text(f"Estimated H₀ ≈ {H0_est:.2f} km/s/Mpc")
                        H0_box.value = round(H0_est, 2)

                        with ui.dialog() as dlg, ui.card().props('aria-modal="true" role="dialog" aria-labelledby="h0-result-title" aria-describedby="h0-result-content"'):
                            ui.label(f"Linear fit result: H₀ = {H0_est:.2f} km/s/Mpc").classes("text-lg font-semibold text-blue-700")
                            with ui.pyplot():
                                plt.subplot(2, 1, 1)
                                plt.scatter(x, y, s=10, color="blue", label="Data (z < 0.1)")
                                plt.plot(x, H0_est * x, 'r--', label=f'Fit H₀={H0_est:.2f}')
                                plt.xlabel("d_L [Mpc]"); plt.ylabel("v [km/s]")
                                plt.legend(); plt.title("Hubble fit (low-z)",fontweight='bold')

                                plt.subplot(2, 1, 2)
                                residuals = y - H0_est * x
                                plt.scatter(x, residuals, s=10, color="gray")
                                plt.axhline(0, color='r', linestyle='--')
                                plt.xlabel("d_L [Mpc]"); plt.ylabel("Residuals [km/s]")
                                plt.tight_layout()
                                aria_chart_label("Hubble fit plot and residuals")
                            aria_button("Close", "close dialog box",on_click=lambda: dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        dlg.open()


                    def chi2(data_mb, model_mb, sigma_mb):
                        mask = (sigma_mb > 0) & np.isfinite(sigma_mb)
                        return np.sum(((data_mb[mask] - model_mb[mask])**2) / (sigma_mb[mask]**2))
                    def chi2_model(H0_try, z_data, mB_data, sigma_mB, model_func):
                        mu_model = 5*np.log10(model_func(z_data, H0_try)*1e6)-5
                        return np.sum((mB_data - (mu_model + M_B_input.value))**2 / sigma_mB**2)

                    def plot_all():
                        vdL_col.clear(); vz_col.clear(); mu_col.clear() # ;mB_col.clear()
                        M_B = float(M_B_input.value)
                        df['mu'] = df['mb'] - M_B
                        df['d_L_Mpc'] = 10**((df['mu']+5)/5)/1e6
                        H0v = get_H0_value()

                        # 1) v_nonrel vs d_L
                        with vdL_col, ui.pyplot():
                            plt.errorbar(df['d_L_Mpc'], df['v_nonrel'], fmt='o', color="blue",ms=4, alpha=0.7, label="v_nonrel")
                            if state['show_vrel']:
                                plt.errorbar(df['d_L_Mpc'], df['v_rel'], fmt='o', color="green", alpha=0.7, label="v_rel")
                            if state['show_hubble_line']:
                                H0v = get_H0_value()

                                x = np.logspace(np.log10(df['d_L_Mpc'].min()), np.log10(df['d_L_Mpc'].max()*1.1), 300)

                                plt.plot(x, H0v*x, '--', color="red", label=f'v=H₀d ({H0v:.1f})')
                            
                            plt.xlabel("d_L [Mpc]"); plt.ylabel("v [km/s]")
                            plt.title("velocity vs luminosity distance ",fontweight='bold')
                            plt.legend(); plt.tight_layout()
                            aria_chart_label("Plot of velocity v versus luminosity distance d sub L")
                        # 2) v vs z
                        with vz_col, ui.pyplot():
                            plt.errorbar(df['zcmb'], df['v_nonrel'], fmt='o',  color="blue",ms=4, alpha=0.7, label="v_nonrel")
                            if state['show_vrel']:
                                plt.errorbar(df['zcmb'], df['v_rel'], fmt='o', color="green", alpha=0.7, label="v_rel")
                            if state['show_hubble_line']:
                                H0v = get_H0_value()
                                z = np.linspace(0, df['zcmb'].max(), 300)
                                dL = (c / H0v) * z  
                                v_hubble = H0v * dL 
                                plt.plot(z, v_hubble, '--', color='red', lw=2, label=f'Hubble line H₀={H0v:.1f}')

                            plt.xlabel("z"); plt.ylabel("v [km/s]")
                            plt.title("velocity vs redshift",fontweight='bold')
                            plt.legend(); plt.tight_layout()
                            aria_chart_label("Plot of velocity v versus redshift z")
                            

                        # 4) μ vs z
                        with mu_col, ui.pyplot():
                            plt.errorbar(df['zcmb'], df['mu'], yerr=df['dmb'], fmt='o', color="blue", ms=4, alpha=0.7, capsize=2, label="μ (data)")

                            if state['show_mag_hubble']:
                                try:
                                    H0v = get_H0_value()

                                except (TypeError, ValueError):
                                    H0v = 70.0
                                #z_hubble = np.linspace(0, df['zcmb'].max(), 300)
                                z_hubble = np.linspace(1e-4, df['zcmb'].max()*1.05, 300)
                                dL_h = (c / H0v) * z_hubble
                                mu_h = 5 * np.log10(dL_h * 1e6) - 5
                                plt.plot(z_hubble, mu_h, '--', color='red', lw=2, label=f'μ (Hubble law, H₀={H0v:.1f})')
                                model_mu_interp = interp1d(z_hubble, mu_h, bounds_error=False, fill_value="extrapolate")
                
                                valid_z = (df['zcmb'].values >= z_hubble.min()) & (df['zcmb'].values <= z_hubble.max())
                                
                            
                                chi2_val = chi2(df['mu'][valid_z].values, model_mu_interp(df['zcmb'][valid_z].values), df['dmb'][valid_z].values)
                                plt.text(0.9, 0.05, f"χ² = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="red",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))

                            if state['show_lcdm']:
                                H0v = get_H0_value()
                                #z = np.linspace(0, df['zcmb'].max(), 300)
                                z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                                dL = np.array([dL_Mpc_LCDM(zv, H0v) for zv in z])
                                mu = 5*np.log10(dL*1e6)-5
                                plt.plot(z, mu,color="green", label='μ(ΛCDM)')
                                model_mu_interp = interp1d(z, mu, bounds_error=False, fill_value="extrapolate")
                
                                valid_z = (df['zcmb'].values >= z.min()) & (df['zcmb'].values <= z.max())
                                
                                
                                chi2_val = chi2(df['mu'][valid_z].values, model_mu_interp(df['zcmb'][valid_z].values), df['dmb'][valid_z].values)

                                plt.text(0.1, 0.05, f"χ²_LCDM = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="green",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))
                                
                            if state['show_noLambda']:
                                H0v = get_H0_value()
                                #z = np.linspace(0, df['zcmb'].max(), 300)
                                z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                                dL_noL = np.array([dL_Mpc_noLambda(zv, H0v) for zv in z])
                                mu_noL = 5 * np.log10(dL_noL * 1e6) - 5
                                plt.plot(z, mu_noL, color="orange", lw=2,
                                        label='μ (ΩΛ=0, matter-only)')
                                model_mu_interp = interp1d(z, mu_noL, bounds_error=False, fill_value="extrapolate")
                
                                valid_z = (df['zcmb'].values >= z.min()) & (df['zcmb'].values <= z.max())
                                
                                
                                chi2_val = chi2(df['mu'][valid_z].values, model_mu_interp(df['zcmb'][valid_z].values), df['dmb'][valid_z].values)
                                plt.text(0.5, 0.05, f"χ²_noΛ = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="orange",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))

                            plt.xlabel("z"); plt.ylabel("μ [mag]")
                            plt.title("distance modulus vs redshift",fontweight='bold')
                            #plt.xscale('log'); plt.yscale('log')
                            plt.legend(); plt.tight_layout()
                            aria_chart_label("Plot of distance modulus μ versus redshift z")
                    aria_button("Add v_rel","add relativistic velocity" ,on_click=lambda: (state.update(show_vrel=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Add Hubble line", "add Hubble line",on_click=lambda: (state.update(show_hubble_line=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Add Hubble mag","add Hubble magnitude", on_click=lambda: (state.update(show_mag_hubble=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Add CDM", "add only matter model",on_click=lambda: (state.update(show_noLambda=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Add ΛCDM", "add cosmological model",on_click=lambda: (state.update(show_lcdm=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                plot_all()


   #panel cosmic microwave background     
        def add_cmb_activity(container):
            with container:
                description_on_dark("Investigate how the Cosmic Microwave Background (CMB) affects the universe composition.")
                with ui.dialog() as ref_dialog, ui.card().classes("p-4 w-full max-w-[1300px]").props('role=dialog aria-label="References"'):
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
                    aria_button("Close", "close popup", on_click=lambda:ref_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.dialog() as instruction_d , ui.card().classes("p-4 w-full max-w-[800px]").props('role=dialog aria-label="Instructions"'):
                    html_info_box("""
    <h2 class="text-2xl font-bold mb-4">CMB Instructions</h2>
    <ol class="list-decimal list-inside space-y-2"> 
    <li>Explore the interactive Planck CMB simulator by clicking the corresponding button below. This simulator allows you to visualize the Cosmic Microwave Background and understand its significance in cosmology.</li>

    <li>Utilize the references section to access further reading materials and resources about the Planck mission and the CMB.</li>
    <li>Examine the images provided below, which illustrate various aspects of the universe's composition, the power spectrum of the CMB, and other relevant phenomena.</li>
    </ol>
                    """)
                    aria_button("Close", "close popup", on_click=lambda:instruction_d.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                


                with ui.row().classes('w-full gap-8 justify-center'):
                    aria_button('Instructions',"Module Instructions", on_click=lambda:instruction_d.open()).classes("!bg-blue-600 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                
                    aria_button('References',"References", on_click=lambda:ref_dialog.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
            
                    
                        
                    aria_button('Explore the Planck CMB Simulator',"Explore the interactive Plank CMB simulator",
        on_click=safe_click(lambda: ui.run_javascript("window.open('https://chrisnorth.github.io/planckapps/Simulator', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                        
                    aria_button('ESA: The Universe According to Planck', "ESA reference website",on_click=safe_click(lambda: ui.run_javascript("window.open('https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck','_blank')"))).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
                with ui.row().classes('w-full mx-auto place-items-center justify-center gap-8'):
                    
                    aria_image('/images/univ_composition.jpg', "Image of the universe composition with dark energy,dark matter and ordinary matter").classes('w-80 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/power_spectrum.jpg', "Image of the power spectrum of cosmic microwave background").classes('w-80 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/CMB_asymmetry.jpg', "Image of CMB asymmetry in the universe").classes('w-80 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/Planck_freq.jpg', "Image of the plank frequency spectrum").classes('w-80 h-auto rounded-lg shadow-lg border border-gray-300')
        
    
    
            
        with ui.tab_panels(tabs, value='redshift').classes('w-full !bg-transparent'):
            with ui.tab_panel('redshift') as redshift_panel:
                add_redshift_activity(redshift_panel)


            with ui.tab_panel('hubble') as hubble_panel:
                add_hubble_activity(hubble_panel)
            with ui.tab_panel('cmb').props('role=tabpanel') as cmb_panel:
                add_cmb_activity(cmb_panel)
                
                
                
                
        '''                            
                        # 3) mB vs z
                        with mB_col, ui.pyplot():
                            plt.errorbar(df['zcmb'], df['mb'], yerr=df['dmb'], fmt='o', color="blue", ms=4, alpha=0.7, capsize=2, label="m_B (data)")

                            if state['show_mag_hubble']:
                                try:
                                    H0v = get_H0_value()

                                except (TypeError, ValueError):
                                    H0v = 70.0
                                #z_hubble = np.linspace(0, df['zcmb'].max(), 300)
                                z_hubble = np.linspace(1e-4, df['zcmb'].max(), 300)
                                dL_h = (c / H0v) * z_hubble
                                mu_h = 5 * np.log10(dL_h * 1e6) - 5
                                
                                plt.plot(z_hubble, mu_h + M_B, '--', color="red", label=f'm_B (Hubble law, H₀={H0v:.1f})')
                                model_mb = mu_h + M_B
                                z_data_for_interp = df['zcmb'].values
                
                                # Per calcolare chi^2, usiamo solo i dati che sono all'interno del range del modello interpolato
                                valid_z = (z_data_for_interp >= z_hubble.min()) & (z_data_for_interp <= z_hubble.max())
                                
                                model_mb_interp = interp1d(z_hubble, model_mb, bounds_error=False, fill_value="extrapolate")
                                
                                # Calcolo chi2 corretto: O = m_B, E = m_B_model
                                chi2_val = chi2(df['mb'][valid_z].values, 
                                                model_mb_interp(z_data_for_interp[valid_z]), 
                                                df['dmb'][valid_z].values)
                                plt.text(0.9, 0.05, f"χ² = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="red",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))
                                

                            if state['show_lcdm']:
                                H0v = get_H0_value()
                                #z = np.linspace(0, df['zcmb'].max(), 300)
                                z = np.linspace(1e-4, df['zcmb'].max(), 250)
                                dL = np.array([dL_Mpc_LCDM(zv, H0v) for zv in z])
                                mu = 5*np.log10(dL*1e6)-5
                                plt.plot(z, mu+M_B,color="green", label='m_B(ΛCDM)')
                                model_mb = mu + M_B
                                model_mb_interp = interp1d(z, model_mb, bounds_error=False, fill_value="extrapolate")
                
                                valid_z = (df['zcmb'].values >= z.min()) & (df['zcmb'].values <= z.max())
                                
                                chi2_val = chi2(df['mb'][valid_z].values, model_mb_interp(df['zcmb'][valid_z].values), df['dmb'][valid_z].values)
                                plt.text(0.1, 0.05, f"χ²_LCDM = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="green",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))
                                
                            if state['show_noLambda']:
                                H0v = get_H0_value()
                                z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                                #z = np.linspace(0, df['zcmb'].max(), 300)
                                dL_noL = np.array([dL_Mpc_noLambda(zv, H0v) for zv in z])
                                mu_noL = 5 * np.log10(dL_noL * 1e6) - 5
                                plt.plot(z, mu_noL + M_B, color="orange", lw=2,                               label='m_B (ΩΛ=0, matter-only)')
                                model_mb = mu_noL + M_B
                                model_mb_interp = interp1d(z, model_mb, bounds_error=False, fill_value="extrapolate")
                
                                valid_z = (df['zcmb'].values >= z.min()) & (df['zcmb'].values <= z.max())
                                
                                chi2_val = chi2(df['mb'][valid_z].values, model_mb_interp(df['zcmb'][valid_z].values), df['dmb'][valid_z].values)
                                plt.text(0.5, 0.05, f"χ²_noΛ = {chi2_val:.1f}",
                                        transform=plt.gca().transAxes,
                                        fontsize=11, color="orange",
                                        ha='right', va='bottom',
                                        bbox=dict(facecolor="white", alpha=0.7))

                            plt.xlabel("z"); plt.ylabel("m_B [mag]")
                            plt.title("magnitude vs redshift",fontweight='bold')
                            #plt.xscale('log'); plt.yscale('log')
                            plt.legend(); plt.tight_layout()
                            aria_chart_label("Plot of apparent magnitude m sub B versus redshift z")
            '''            
        
                 
                    