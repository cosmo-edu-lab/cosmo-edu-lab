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
    @ui.page('/module3')
    def module3():
        
#load custom CSS 
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
    
       
    #s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg-full.js';
#load datasets with caching
        @lru_cache(maxsize=1)
        def get_cobe_data_cached():
          
            cobefile = os.path.join(DATA_DIR, "cobe_firas_monopole.txt")
            
            if not os.path.exists(cobefile):
                return None

            try:
                df_cobe = pd.read_csv(cobefile, sep=r"\s+", header=0, engine="python")

               
                freq_cm1 = pd.to_numeric(df_cobe["frequency(cm^-1)"], errors="coerce").values
                I_MJy_sr = pd.to_numeric(df_cobe["monopole_spectrum(MJy/sr)"], errors="coerce").values
                err_kJy_sr = pd.to_numeric(df_cobe["uncertainty(kJy/sr)"], errors="coerce").values

               
                mask = np.isfinite(freq_cm1) & np.isfinite(I_MJy_sr)
                freq_cm1, I_MJy_sr, err_kJy_sr = freq_cm1[mask], I_MJy_sr[mask], err_kJy_sr[mask]

            
                c = 2.99792458e8
                
               
                cobewl = 1.0 / (freq_cm1 * 100.0) 
                
        
                I_W_m2_Hz_sr = I_MJy_sr * 1e-20           # MJy -> W
                err_W_m2_Hz_sr = err_kJy_sr * 1e-23       # kJy -> W

                # B_lambda = B_nu * c / lambda^2
                cobeI = I_W_m2_Hz_sr * (c / (cobewl ** 2))
                cobeErr = err_W_m2_Hz_sr * (c / (cobewl ** 2))

              
                sort_idx = np.argsort(cobewl)
                cobewl = cobewl[sort_idx]
                cobeI = cobeI[sort_idx]
                cobeErr = cobeErr[sort_idx]

                return cobewl, cobeI, cobeErr

            except Exception as e:
                print(f"Error processing COBE data: {e}")
                return None
        @lru_cache(maxsize=1)
        def get_cmb_data_cached():
       
            cmb_path = os.path.join(DATA_DIR, "dataset_CMB.txt")
            
            if not os.path.exists(cmb_path):
                return None

            try:
               
                df = pd.read_csv(cmb_path, comment="#", sep=r"\s+", header=None, engine="python")
                
              
                z_obs = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
                T_obs = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
                T_err = pd.to_numeric(df.iloc[:, 3], errors="coerce").values

               
                valid = np.isfinite(z_obs) & np.isfinite(T_obs) & np.isfinite(T_err)
                
                return z_obs[valid], T_obs[valid], T_err[valid]

            except Exception as e:
                print(f"Error loading CMB dataset: {e}")
                return None
            
            
        main_layout("Universe History & CMB")
    
        
        tab_key = 'module3_selected_tab'
        if tab_key not in app.storage.user:
            app.storage.user[tab_key] =' planck'
        with ui.tabs().classes('w-full justify-center').bind_value(app.storage.user, tab_key) as tabs:
            ui.tab('planck',label=' Planck spectrum').props('aria-label="Activity 1: Planck Spectrum"')
            ui.tab('index',label='CMB Adiabatic Index').props('aria-label="Activity 2: Adiabatic Index"')
            ui.tab('radmat',label='Radiation & Matter').props('aria-label="Activity 3: Radiation & Matter"')
        

#panel planck spectrum
            def add_planck_activity(container):
    
                h = 6.62607015e-34
                c = 2.99792458e8
                kB = 1.380649e-23

                def B_lambda(wl_m, T):
                    """Planck spectral radiance Bλ(λ,T) [W·m⁻²·sr⁻¹·m⁻¹] """
                    wl = np.array(wl_m, dtype=np.float64)
            
                    wl = np.where(wl <= 0, np.nan, wl)
                    a = 2.0 * h * c**2 / (wl**5)
                    x = (h * c) / (wl * kB * T)       
                
                    denom = np.expm1(x)              
            
                    denom = np.where(np.isfinite(denom) & (denom != 0), denom, np.inf)
                    return a / denom


                cobe_data = get_cobe_data_cached()

                if cobe_data is not None:
                    cobewl, cobeI, cobeErr = cobe_data
                    cobedata_loaded = True
                else:
                    cobewl, cobeI, cobeErr = np.array([]), np.array([]), np.array([])
                    cobedata_loaded = False
                    coberead_error = "Could not load or process data"


                user_rows = []         
                generated_rows = []    
                current_temp = None    
                auto_fill = False      


            
                with container:
                    description_on_dark(
    "Explore the nature of thermal radiation to understand how the Black Body spectrum describes the afterglow of the Big Bang."
)
                    
                    with ui.dialog() as plank,ui.card().props('aria-label="COBE-FIRAS dataset information dialog" role=dialog aria-modal=true'):
                    
                        reference_box("""**Dataset reference**: Fixsen et al. (1996), The Astrophysical Journal. [COBE-FIRAS monopole spetrum](https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_spect.html)""")
                        info_box("**Dataset**:frequency(cm^-1) monopole_spectrum(MJy/sr) uncertainty(kJy/sr)")
                        aria_button("Close","close",on_click=lambda:plank.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        
                        
                
                        
                    with ui.dialog() as cur_planck, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="The Big Bang on Your TV" role=dialog aria-modal=true'):
                        html_info_box(r"""
                        <h3>The Big Bang on Your TV</h3>
                        <p>Before digital television, when you tuned an old analog TV to a dead channel, about <b>1% of the static "snow"</b> you saw was actually caused by the Cosmic Microwave Background (CMB).</p>
                        <p>You were literally watching the afterglow of the Big Bang interacting with your TV antenna!</p>
                        """)
                        reference_box("**Source:** [Forbes](https://www.forbes.com/sites/startswithabang/2019/11/13/this-is-how-your-old-television-set-can-prove-the-big-bang/)")
                        aria_button("Close", "close", on_click=lambda:cur_planck.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                   
                    
                    with ui.dialog().props('aria-modal="true" role="dialog" aria-labelledby="planck-info-title" aria-describedby="planck-info-content"') as info_dlg, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
                        ui.label("Information").classes("text-lg font-semibold text-blue-400").props("id=planck-info-title role=heading aria-level=2 tabindex=0")
    
                        html_info_box(r"""
        
        <h3>Activity Instructions</h3>
        <p>In this activity, you will determine the present-day temperature (<span class="math">\(T_0\)</span>) of the Cosmic Microwave Background:</p>
        <ol class="list-decimal ml-4 space-y-2">
            <li>
                <b>Generate Theoretical Curves:</b> Use Planck's Law to calculate and plot radiation intensity vs. wavelength for different peak temperatures (e.g., 3000 K, 5000 K).
            </li>
            <li>
               <b>Compare with Observations:</b> Overlay your theoretical curves with the real COBE/FIRAS dataset (blue points) representing the CMB spectrum.
            </li>
            <li>
                <b>Analyze the Peak:</b> Observe that higher temperatures shift the peak to shorter wavelengths (left) and increase intensity.Verify this using Wien's Displacement Law.
            </li>
            <li>
                <b>Find the Match:</b> Adjust the temperature in your simulation until the theoretical curve perfectly overlaps the observed data. The temperature that fits the data is the current temperature of the Universe, <span class="math">\(T_0 \approx 2.725\)</span> K.
            </li>
        </ol>
         <hr class="my-4">
        <h3>Planck’s Law and Blackbody Radiation</h3>
        <p>Planck’s law describes the spectral radiance of an ideal blackbody:</p>
        
        <div style="text-align:center; margin: 15px 0;">
            $$B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5} \frac{1}{e^{hc/(\lambda k_B T)} - 1}$$
        </div>

        <p><strong>where:</strong></p>
        <ul>
            <li><span class="math">\(B_\lambda\)</span> = spectral radiance [W·m⁻²·sr⁻¹·m⁻¹]</li>
            <li><span class="math">\(h\)</span> = Planck constant = <span class="math">\(6.626\times10^{-34}\)</span> J·s</li>
            <li><span class="math">\(c\)</span> = speed of light = <span class="math">\(2.998\times10^{8}\)</span> m/s</li>
            <li><span class="math">\(k_B\)</span> = Boltzmann constant = <span class="math">\(1.381\times10^{-23}\)</span> J/K</li>
        </ul>

        <p>The distribution peaks at a wavelength inversely proportional to T:</p>
        <div style="text-align:center; margin: 10px 0;">
            $$\lambda_{max} T = b$$
        </div>
        <p>with <span class="math">\(b = 2.897771955\times10^{-3}\)</span> m·K.</p>

        <p>The COBE/FIRAS measurement confirmed the CMB blackbody at T ≈ 2.725 K.</p>

       

    """).props("id=planck-info-content role=document aria-live=polite")

                        aria_button("Close","close",on_click=lambda:info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")



                    
                    with ui.dialog().props('aria-modal="true" role="dialog" aria-labelledby="planck-legend-title" aria-describedby="planck-legend-content"') as legend_dlg, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
                        ui.label("Legend").classes("text-lg font-semibold text-blue-400").props("id=planck-legend-title role=heading aria-level=2 tabindex=0")
    
                        html_info_box(r"""
        <h3>Symbols and Units</h3>
        <ul>
            <li><b>λ</b> = wavelength [m]</li>
            <li><b>ν</b> = frequency [Hz]</li>
            <li><b>Bλ</b> = spectral radiance [W·m⁻²·sr⁻¹·m⁻¹]</li>
            <li><b>T</b> = absolute temperature [K]</li>
            <li><b>h</b> = Planck constant = <span class="math">\( 6.626\times10^{-34} \)</span> J·s</li>
            <li><b>c</b> = speed of light = <span class="math">\( 2.998\times10^{8} \)</span> m/s</li>
            <li><b>kB</b> = Boltzmann constant = <span class="math">\( 1.381\times10^{-23} \)</span> J/K</li>
            <li><b>b</b> = Wien constant = <span class="math">\( 2.898\times10^{-3} \)</span> m·K</li>
            <li><b>I</b> = intensity [W·m⁻²·sr⁻¹·m⁻¹]</li>
            <li><b>z</b> = redshift (dimensionless)</li>
        </ul>
    """).props("id=planck-legend-content role=document aria-live=polite")

                        aria_button("Close","close",on_click=lambda:legend_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")



                
                    with ui.dialog() as temp, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto').props('aria-label="Comparison instructions" role=dialog aria-modal=true'):
                        html_info_box(r"""
        <h3>Comparison Instructions</h3>
        <p>Compare theoretical blackbody curves at different temperatures with the COBE/FIRAS CMB spectrum.</p>
        <ul>
            <li>Select or enter a temperature, then build a table of wavelength–intensity pairs.</li>
            <li>Make sure to use wavelengths in meters and intensities in compatible units.</li>
            <li>Click <b>'Generate Plot'</b> to overlay your theoretical curve on the COBE data.</li>
        </ul>
        
        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">
        
        <h4>Interactive Blackbody Calculators</h4>
        <ul>
            <li><a href="https://www.opticsthewebsite.com/OpticsCalculators" target="_blank" style="color:#0284c7; text-decoration:underline;">OpticsTheWebsite</a></li>
            <li><a href="https://ncc.nesdis.noaa.gov/Planck.php" target="_blank" style="color:#0284c7; text-decoration:underline;">NOAA Planck Calculator</a></li>
            <li><a href="https://www.bodkindesign.com/reference-library/blackbody-spectral-radiance-calculator/" target="_blank" style="color:#0284c7; text-decoration:underline;">Bodkin Design Blackbody Calculator</a></li>
            <li><a href="https://spectralcalc.com/blackbody_calculator/blackbody.php" target="_blank" style="color:#0284c7; text-decoration:underline;">SpectralCalc Blackbody Tool</a></li>
        </ul>
    """)
                        aria_button("Close","close",on_click=lambda:temp.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    @ui.refreshable
                    def draw_plot():
                
                        with ui.pyplot(figsize=(6,4)):
                            
                            try:
                                if cobedata_loaded and getattr(cobewl, "size", 0) > 0:
                                    
                                    wl = np.array(cobewl, dtype=float)
                                    I = np.array(cobeI, dtype=float)
                                    err = np.array(cobeErr, dtype=float) if 'cobeErr' in globals() or 'cobeErr' in locals() else None
                                    finite_mask = np.isfinite(wl) & np.isfinite(I)
                                    if err is not None:
                                        finite_mask &= np.isfinite(err)
                                    wl_plot = wl[finite_mask]
                                    I_plot = I[finite_mask]
                                    if err is not None:
                                        err_plot = err[finite_mask]
                                    else:
                                        err_plot = None

                                
                                    order = np.argsort(wl_plot)
                                    wl_plot = wl_plot[order]
                                    I_plot = I_plot[order]
                                    if err_plot is not None:
                                        err_plot = err_plot[order]

                                    if err_plot is not None:
                                        plt.errorbar(wl_plot, I_plot, yerr=err_plot,
                                                    fmt='o', ms=4, elinewidth=1, capsize=2,
                                                    color='blue', ecolor='gray', alpha=0.85,
                                                    label="COBE/FIRAS (observed)")
                                        
                                    else:
                                        plt.scatter(wl_plot, I_plot, s=20, color='blue', label="COBE/FIRAS (observed)")

                                else:
                                    plt.text(0.1, 0.5, "COBE/FIRAS data not available.", transform=plt.gca().transAxes)

                            except Exception as _e:
                                
                                plt.text(0.1, 0.5, "Error plotting COBE data.", transform=plt.gca().transAxes)

                        
                            if current_temp is not None:
                            
                                if cobedata_loaded and getattr(cobewl, "size", 0) > 0:
                                    wl_min = float(np.nanmin(cobewl))
                                    wl_max = float(np.nanmax(cobewl))
                                else:
                                    wl_min, wl_max = 1e-4, 1e-2

                            
                                wl_grid = np.logspace(np.log10(max(1e-12, wl_min*0.5)),
                                                    np.log10(wl_max*2.0), 1000)
                                Igrid = B_lambda(wl_grid, current_temp)

                                mask_grid = np.isfinite(wl_grid) & np.isfinite(Igrid) & (Igrid > 0)
                                wl_plot_grid = wl_grid[mask_grid]
                                I_plot_grid = Igrid[mask_grid]

                            
                                order = np.argsort(wl_plot_grid)
                                wl_plot_grid = wl_plot_grid[order]
                                I_plot_grid = I_plot_grid[order]

                            
                                plt.plot(wl_plot_grid, I_plot_grid, '-', lw=2, color='tab:green',
                                        label=f"Planck T={current_temp:.3f} K", alpha=0.95)


                                
                                b_const = 2.897771955e-3
                                lambda_max_theory = b_const / current_temp
                                

                            
                            if len(user_rows) > 0:
                                wls = [r['wavelength_m'] for r in user_rows]
                                Is = [r['intensity'] for r in user_rows]
                                plt.scatter(wls, Is, s=40, marker='x', c='green', label='Input Point', zorder=8)
                        
                            plt.xscale('log')
                            plt.yscale('log')
                            plt.xlabel("Wavelength λ [m]")
                            plt.ylabel("Radiance intensity Bλ [W·m⁻²·sr⁻¹·m⁻¹]")
                            plt.title("COBE/FIRAS CMB Spectrum vs Planck Curve",fontweight='bold')
                            plt.grid(True, which='both', ls='--', alpha=0.4)
                            plt.legend()
                            plt.tight_layout()

                            aria_chart_label("Plot of COBE/FIRAS CMB spectrum and Planck blackbody curve")
                    with ui.dialog() as wien,ui.card().props('aria-label="Wien\'s Law Calculator" role=dialog aria-modal=true'):
                            
                        info_box("Wien’s Law Calculator: T=λₘₐₓ/b"
                        
                        "\n\nInsert the observed peak wavelength 'λₘₐₓ' to compute the temperature")

                        lambda_input = aria_input("λₘₐₓ [m]", "Enter peak wavelength in meters").classes("w-full")
                        T_output = ui.label("").classes("text-md font-medium text-blue-600 mt-2").props('aria-live=polite')

                        def compute_T_from_lambda():
                            try:
                                b = 2.897771955e-3
                                lam = float(lambda_input.value)
                                if lam <= 0:
                                    raise ValueError
                                Tcalc = b / lam
                                T_output.set_text(f"Calculated temperature: T = {Tcalc:.3f} K")
                            except Exception:
                                accessible_notify("Please enter a valid positive wavelength.", type_='warning')

                        def reset_wien_calc():
                            lambda_input.set_value("")
                            T_output.set_text("")
                            accessible_notify("Wien’s law calculator reset.", type_='info')

                        with ui.row().classes("items-center gap-3 mt-2"):
                            aria_button("Calculate", "Compute temperature", on_click=compute_T_from_lambda).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            aria_button("Reset", "Clear input and result", on_click=reset_wien_calc).classes("negative !bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")

                        aria_button("Close","close",on_click=lambda:wien.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.dialog() as gen_data_dlg, ui.card().classes('w-full h-full').props('aria-label="Generated spectrum data" role=dialog aria-modal=true'):
                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label("Generated spectrum from input peak temperature ").classes('text-lg font-bold').props('aria-label="Generated spectrum table title"')
    
                        gen_table = ui.table(
                            columns=[
                                {'name':'wavelength_m','label':'λ [m]','field':'wavelength_m'},
                                {'name':'intensity','label':'Intensity','field':'intensity'}
                            ], 
                            rows=[]
                        ).classes('w-full').props('role=table tabindex=0 aria-label="Generated spectrum table"')
                        aria_button("Close","close",on_click=lambda:gen_data_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.row().classes("w-full justify-center items-center gap-2"):
                        
                        aria_button("Instructions ", "Information",on_click=lambda:[info_dlg.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        aria_button("Dataset","dataset",on_click=lambda:[plank.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        aria_button("Legend ", "Variable and constants legend",on_click=lambda:[legend_dlg.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                        aria_button("CMB spectrum exercise","CMB spectrum exercise",on_click=lambda:[temp.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        aria_button("Wien law","Wien law",on_click=lambda:[wien.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                        aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_planck.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                    

                        

                        
                    
            
                    
                    with ui.row().classes('w-full justify-center items-center gap-8'):
                        with ui.column().classes('flex-1'):
                            draw_plot()
                            lambda_peak_box = ui.label("").classes("text-md font-semibold text-blue-400 ").props('aria-live=polite')
                        
                        
                    
                        with ui.column().classes('flex-1'):
                            with ui.grid(columns=2):
                                temp_select = ui.select(
                                    [1,2,3,4,5,6,7,8,9,10,100,200,300,400,500,600,700,800,900,1000, 2000, 3000, 4000, 5000, 6000, 10000],
                                    label="Peak temperature (K)"
                                ).props('dense outlined aria-label="Select peak temperature"').classes('w-64')

                                temp_custom = ui.number(
                                    label="Or enter custom temperature [K]",
                                    format="%.6g"
                                ).props('dense outlined aria-label="Custom temperature in Kelvin"').classes('w-64')

                                lambda_min_input = ui.number(
                                    label="λ_min [m]",
                                    value=float(np.nanmin(cobewl) if cobedata_loaded and cobewl.size > 0 else 1e-4),
                                    format="%.3e"
                                ).props('dense outlined aria-label="Minimum wavelength"').classes('w-64')

                                lambda_max_input = ui.number(
                                    label="λ_max [m]",
                                    value=float(np.nanmax(cobewl) if cobedata_loaded and cobewl.size > 0 else 1e-2),
                                    format="%.3e"
                                ).props('dense outlined aria-label="Maximum wavelength"').classes('w-64')


                        
                            with ui.row().classes("items-center gap-3 "):
                                wl_input = aria_input("Wavelength [m]", "Enter wavelength in meters").classes("w-24")
                                I_input = aria_input("Intensity", "Enter corresponding intensity value").classes("w-24")
                                aria_button("Add Row", "Add new row to table", on_click=lambda: add_row_wl_I()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                aria_button("Generate Plot", "Overlay Planck curve for selected T", on_click=lambda: generate_and_plot()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                                aria_button("Reset", "Clear table and curve", on_click=lambda: reset_all()).classes("negative !bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")

                        with ui.column().classes('flex-1'):
                            with ui.expansion("Table Planck Spectrum", icon="table_chart", value=True).classes("w-128 border rounded-lg justify-center"):
                                table_container = ui.column().classes("w-full justify-center")



                            
                        
                            def refresh_table():
                                nonlocal table_container
                                table_container.clear()  
                                rows = []
                                for r in user_rows:
                                    rows.append({'wavelength_m': f"{r['wavelength_m']:.6e}", 'intensity': f"{r['intensity']:.6e}"})
                            

                            
                                with table_container:
                                    aria_table(
                                        columns=[
                                            {'name':'wavelength_m','label':'λ [m]','field':'wavelength_m'},
                                            {'name':'intensity','label':'Intensity','field':'intensity'}
                                            
                                        ],
                                        rows=rows,
                                        label="Wavelengths and intensities "
                                    ).classes('w-full')



                            def add_row_wl_I():
                                try:
                                    wl_val = float(wl_input.value)
                                    I_val = float(I_input.value)
                                except Exception:
                                    accessible_notify("Please enter valid numerical values for wavelength and intensity.", type_='warning')
                                    return
                                user_rows.append({'wavelength_m': wl_val, 'intensity': I_val})
                                wl_input.set_value("")
                                I_input.set_value("")
                                refresh_table()
                                draw_plot.refresh()

    


                            def parse_selected_temp():
                                try:
                                    if temp_custom.value is not None and float(temp_custom.value) > 0:
                                        return float(temp_custom.value)
                                    elif temp_select.value:
                                        return float(temp_select.value)
                                    else:
                                        return None
                                except Exception:
                                    return None

                            def generate_and_plot():
                                nonlocal current_temp, generated_rows
                                Tsel = parse_selected_temp()
                                if Tsel is None:
                                    accessible_notify("Please select or enter a valid temperature (>0).", type_='warning')
                                    return
                                current_temp = Tsel

                        
                                try:
                                    lam_min = float(lambda_min_input.value)
                                    lam_max = float(lambda_max_input.value)
                                    if lam_min <= 0 or lam_max <= lam_min:
                                        raise ValueError
                                except Exception:
                                    accessible_notify("Invalid wavelength range. Check λ_min and λ_max.", type_='warning')
                                    return

                            
                                npts = max(50, len(cobewl) if cobedata_loaded and cobewl.size>0 else 200)
                                wl_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), npts)
                                I_grid = B_lambda(wl_grid, Tsel)

                                generated_rows = [{'wavelength_m': float(wl), 'intensity': float(I)} for wl, I in zip(wl_grid, I_grid)]
                                gen_table.rows = [
                                    {'wavelength_m': f"{r['wavelength_m']:.6e}", 'intensity': f"{r['intensity']:.6e}"} 
                                    for r in generated_rows
                                ]
                                gen_table.update() 
                                gen_data_dlg.open() 
                            
                            
                                b_const = 2.897771955e-3
                                lambda_max_theory = b_const / Tsel
                                lambda_max_obs = (cobewl[np.nanargmax(cobeI)] if cobedata_loaded and cobewl.size>0 else np.nan)

                            
                                try:
                                    lambda_peak_box.set_text(f"λₘₐₓ (theory) = {lambda_max_theory:.3e} m — λₘₐₓ (obs) = {lambda_max_obs:.3e} m")
                                except Exception:
                                    pass

                                refresh_table()
                                draw_plot.refresh()




                            def reset_all():
                                nonlocal current_temp, generated_rows, user_rows
                                current_temp = None
                                generated_rows.clear()
                                user_rows.clear()
                                lambda_peak_box.set_text("")  
                                refresh_table()
                                draw_plot.refresh()
                                accessible_notify("Table and curve fully reset.", type_='info')



                            refresh_table()

#panel cmb adiabatic index
            def add_cmb_activity(container):

                cmb_data = get_cmb_data_cached()

                if cmb_data is not None:
                    z_obs, T_obs, T_err = cmb_data
                else:
                    with container:
                        ui.label("Error loading CMB dataset (dataset_CMB.txt).").classes("text-red-600")
                    return

                T0_default = 2.72548  # K
                gamma_default = 0.0

                def T_model(z, T0, gamma):
                    return T0 * (1.0 + z) ** (3.0 * (gamma - 1.0))

                def chi2_for_gamma(gamma_val, T0_val):
                    Tpred = T_model(z_obs, T0_val, gamma_val)
                    valid = np.isfinite(T_obs) & np.isfinite(T_err)
                    chi2 = np.sum(((T_obs[valid] - Tpred[valid]) / T_err[valid]) ** 2)
                    dof = max(1, len(T_obs[valid]) - 1)
                    return chi2 / dof

                

                def float_to_fraction_str(x, max_denominator=24):
                    try:
                        frac = Fraction(x).limit_denominator(max_denominator)
                        if abs(float(frac) - x) < 1e-6:
                            return f"{frac.numerator}/{frac.denominator}"
                        else:
                            return f"{frac.numerator}/{frac.denominator} ≈ {x:.4f}"
                    except Exception:
                        return f"{x:.4f}"

                z_min, z_max = np.nanmin(z_obs), np.nanmax(z_obs)
                z_grid = np.linspace(max(0, z_min * 0.98), z_max * 1.02, 400)

                with container:
                    description_on_dark(
    "Analyze the thermodynamics of the expanding Universe to understand how it cooled down from a hot plasma to the transparent cosmos we see today."
)
                    with ui.dialog() as info_dlg, ui.card().classes('p-4 w-full overflow-x-auto').props('aria-label="CMB thermodynamics activity instructions" role=dialog aria-modal=true'):
                        html_info_box(r"""
 

    <h3>Activity Instructions</h3>
    <ol style="margin-top: 10px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b>Review Thermodynamics:</b> Recall the first law of thermodynamics and how the adiabatic index relates to the degrees of freedom of a gas.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Analyze the Plot:</b> Observe the graph showing CMB Temperature vs. Redshift based on observational data.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Fit the Model:</b> Move the slider to adjust <span class="math">\(\gamma\)</span>. This changes the theoretical curve <span class="math">\(T \propto (1+z)^{3(\gamma-1)}\)</span>.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Find the Match:</b> Identify the value of <span class="math">\(\gamma\)</span> where the simulated curve perfectly overlaps the data points. 
        </li>
        <li style="margin-bottom: 8px;">
            <b>Conclusion:</b> This result confirms that the primordial universe was dominated by radiation (massless photons) rather than standard gas.
        </li>
    </ol>
   

    
""").props("id=cmb-thermo-info role=document aria-live=polite")
                        aria_button("Close","close",on_click=lambda:info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                  
                    with ui.dialog() as cur_adia, ui.card().classes('p-4 w-full max-w-[600px]').props('aria-label="Is the Universe perfectly adiabatic?" role=dialog aria-modal=true'):
                        html_info_box(r"""
                        <h3>Is the Universe Perfectly Adiabatic?</h3>
                        <p>We model the expansion of the Universe as <b>adiabatic</b> (constant entropy), but technically, stars and black holes produce entropy, making the process irreversible.</p>
                        <p><b>So why does the model work?</b></p>
                        <p>The entropy of the Cosmic Microwave Background (CMB) photons is enormous—about <b>1 billion photons for every proton</b>. The entropy produced by all the stars and galaxies combined is negligible compared to the "fossil" entropy of the Big Bang. Thus, the Universe remains <i>effectively</i> adiabatic.</p>
                        """)
                        reference_box("**Source:** [Egan & Lineweaver (2010)](https://arxiv.org/abs/0909.3983)")
                        aria_button("Close", "close", on_click=lambda:cur_adia.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

               
                    
                    with ui.dialog() as ref_dialog, ui.card().classes('p-4 w-full overflow-x-auto').props('aria-label="Reference data for CMB analysis" role=dialog aria-modal=true'):
                        info_box(
                        
                        "**Dataset**: Redshift (z), CMB Temperature (K), T/z+1, T_error"
                    )
                        reference_box(
                        """**Dataset reference**: Riechers et al. 2022, *Nature*, 'Microwave background temperature at a redshift of 6.34 from H2O absorption'."""
                    )
                        aria_button("Close","close",on_click=lambda:ref_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as thermo_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
                            <h3>Thermodynamics and Adiabatic Transformations</h3>

                            <p>Imagine the large-scale Universe as an isolated, closed system — it does not exchange heat or energy with any 'outside'. Its total internal energy can be approximated as conserved over time. In this context, the early Universe was hot, dense, and dominated by photon radiation. The cosmic microwave background (CMB) discovered by Penzias and Wilson is the remnant of that radiation, now cooled to about 2.725 K. Its spectrum is an almost perfect blackbody, and its temperature has decreased with the expansion of the Universe.</p>

                            <p>The expansion behaves as an <b>adiabatic process</b>: total energy is conserved, the system expands, and the temperature drops as the volume increases.</p>
                            

                            <ul>
                                <li>
                                    <b>First Law of Thermodynamics:</b><br>
                                    <span class="math">\( \Delta U = Q - L \)</span> → <span class="math">\( dQ = dU + P dV \)</span><br>
                                    → for adiabatic processes <span class="math">\( dQ = 0 \Rightarrow dU + P dV = 0 \)</span>
                                </li>
                                
                                <li>
                                    <b>For an ideal gas:</b><br>
                                    <span class="math">\( P V^\gamma = \text{const} \)</span> → <span class="math">\( T V^{\gamma-1} = \text{const} \)</span> → <span class="math">\( T \propto V^{1-\gamma} \)</span>
                                </li>
                            </ul>

                            <p>The adiabatic index <span class="math">\( \gamma = C_p / C_v \)</span> depends on the degrees of freedom of the gas (equipartition of energy):</p>
                            <ul>
                                <li>Monatomic gas: <span class="math">\( \gamma = 5/3 \)</span></li>
                                <li>Diatomic gas: <span class="math">\( \gamma = 7/5 \)</span></li>
                                <li>Polyatomic (photon gas): <span class="math">\( \gamma = 4/3 \)</span></li>
                            </ul>

                            <h4>In Cosmology</h4>
                            <p>Using the Stefan–Boltzmann law for radiation energy density, <span class="math">\( u = \alpha_{\rm rad} T^4 \)</span> with <span class="math">\( \alpha_{\rm rad} = 4\sigma/c \approx 7.56\times10^{-16} \, \rm J \, m^{-3} K^{-4} \)</span>, and <span class="math">\( P = u/3 \)</span>,</p>
                            
                            <p>we obtain <span class="math">\( \frac{1}{T} \frac{dT}{dt} = -\frac{1}{3V} \frac{dV}{dt} \)</span>. With the scale factor <span class="math">\( a(t) \)</span>, since <span class="math">\( V \propto a^3 \)</span>, we get <span class="math">\( T \propto a^{-1} \)</span>.</p>
                            
                            <p>Because redshift relates to the scale factor as <span class="math">\( 1+z = 1/a \)</span>, we have <span class="math">\( T(z) \propto (1+z) \)</span>.</p>
                            
                            <p>In general form, for any <span class="math">\( \gamma \)</span>: <span class="math">\( T(z) = T_0 (1+z)^{3(\gamma-1)} \)</span>.</p>
                        """).props('tabindex=0 role=document aria-label="Thermodynamics and Adiabatic Transformations information"')
                        aria_button("Close", label="Close the box", on_click=lambda: thermo_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                            


                        
                    with ui.dialog() as legend_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Legend: Symbols and Units</h3>
        <ul>
            <li><b>T</b> = Temperature [K]</li>
            <li><b>V</b> = Volume [m³]</li>
            <li><b>P</b> = Pressure [Pa]</li>
            <li><b>γ</b> = Adiabatic index (Cp/Cv)</li>
            <li><b>Cp, Cv</b> = Specific heats [J·mol⁻¹·K⁻¹]</li>
            <li><b>kB</b> = Boltzmann constant = <span class="math">\( 1.380649\times10^{-23} \)</span> J·K⁻¹</li>
            <li><b>R</b> = Gas constant = 8.314 J·mol⁻¹·K⁻¹</li>
            <li><b>a_rad</b> = Radiation density const. = <span class="math">\( 7.56\times10^{-16} \)</span> J·m⁻³·K⁻⁴</li>
            <li><b>σ</b> = Stefan–Boltzmann const. = <span class="math">\( 5.67\times10^{-8} \)</span> W·m⁻²·K⁻⁴</li>
            <li><b>z</b> = Cosmological redshift (dimensionless)</li>
            <li><b>λ_obs, λ_em</b> = Observed / Emitted wavelength [m]</li>
            <li><b>a(t)</b> = Scale factor (dimensionless)</li>
        </ul>
    """).props('tabindex=0 role=document aria-label="Legend of variables and constants used in the exercise"')

                        aria_button("Close", label="Close the box", on_click=lambda: legend_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    with ui.row().classes("w-full justify-center gap-1 "):

                
                        with ui.column().classes("flex-1"):

                            with ui.row().classes("justify-center gap-1 w-full"):
                                aria_button("Instructions","instruction",on_click=lambda:[info_dlg.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button("Dataset","dataset",on_click=lambda:[ref_dialog.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button(                            "Info Thermodynamics",
                                "Read detailed information about thermodynamics and adiabatic transformations",
                                on_click=lambda:    [                             thermo_dialog.open() ,ui.run_javascript("MathJax.typesetPromise()") ]                      ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button(                            "Legend","Open legend of symbols and units",
                                on_click=lambda: [
                                    legend_dialog.open()                     ,ui.run_javascript("MathJax.typesetPromise()")]      
                            ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_adia.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                            with ui.row().classes("justify-center gap-1 w-full"):
                                T0_input = ui.number(value=T0_default, label="T₀ [K] (present-day)", format="%.6g") \
                                .props('aria-label="Present-day CMB temperature input in Kelvin"')
                                gamma_box = info_box("").classes("p-2 border rounded ").props("aria-live=polite")
                                chi2_box = info_box("").classes("p-2 border rounded").props("aria-live=polite")
                            with ui.row().classes("justify-center gap-1 w-full"):
                                gamma_slider = aria_slider(min=0.0, max=2.0, value=gamma_default, step=0.0001,aria_label="Gamma adiabatic index slider").props('aria-describedby=gamma_slider_label label-always color="light-blue" debounce=300')
                                

                        
                            @ui.refreshable
                            def plot_area():
                                g = float(gamma_slider.value)
                                T0_val = float(T0_input.value)
                                T_grid = T_model(z_grid, T0_val, g)

                                y_min = min(np.nanmin(T_obs), np.nanmin(T_model(z_grid, T0_default, gamma_default)))
                                y_max = max(np.nanmax(T_obs), np.nanmax(T_model(z_grid, T0_default, gamma_default)))
                                y_pad = 0.12 * (y_max - y_min) if (y_max - y_min) > 0 else 1.0

                                frac_str = float_to_fraction_str(g, max_denominator=12)

                                with ui.pyplot(figsize=(6,4)):
                                    plt.errorbar(z_obs, T_obs, yerr=T_err, fmt="o", markersize=4, color="blue",
                                                ecolor="gray", capsize=2, label="Observed data", zorder=5)
                                    plt.plot(z_grid, T_grid, lw=2.2, color="green",
                                            label=f"Model: γ={g:.4f} ({frac_str})", zorder=4)
                                    plt.xlabel("Redshift z")
                                    plt.ylabel("CMB Temperature [K]")
                                    plt.title("CMB Temperature vs Redshift",fontweight='bold')
                                    plt.grid(alpha=0.4, linestyle="--")
                                    plt.legend()
                                    plt.xlim(z_min * 0.98, z_max * 1.02)
                                    plt.ylim(y_min - y_pad, y_max + y_pad)
                                    plt.tight_layout()
                                    aria_chart_label("Plot of CMB temperature versus redshift with model overlay")
                                chi2_val = chi2_for_gamma(g, T0_val)
                                gamma_box.content = f"**Adiabatic index γ:** {g:.6f}  \n**As fraction:** {frac_str}"
                                chi2_box.content = f"**χ² :** {chi2_val:.6g}"
                                gamma_box.update()
                                chi2_box.update()
                            with ui.row().classes("justify-center gap-1 w-full"):
                                plot_area()

                            def update_plot(*_):
                                plot_area.refresh()

                            gamma_slider.on("update:model-value", update_plot)
                            T0_input.on("update:model-value", update_plot)

                        ui.add_head_html('''
    <style>
      
        .math-text { 
            font-family: 'Times New Roman', serif; 
            font-style: italic; 
            font-size: 1.3rem; 
            display: flex; 
            align-items: baseline; /* Allinea il testo sulla riga di scrittura */
            flex-wrap: wrap; 
            gap: 5px; 
            line-height: 1.6;
        }

        /* Testo descrittivo normale */
        .math-normal { 
            font-family: sans-serif;
            font-style: normal; 
            font-size: 0.9rem; 
            margin-right: 6px; 
            color: #9ca3af; 
        }
        
        /* Apici e Pedici posizionati in modo relativo per non rompere la riga */
        .math-sub { 
            font-size: 0.7em; 
            position: relative; 
            top: 0.3em; 
            margin-right: 1px;
        }
        .math-sup { 
            font-size: 0.7em; 
            position: relative; 
            top: -0.4em; 
            margin-left: 1px;
        }
        
        /* Input ottimizzato per allinearsi alla baseline */
        .formula-input {
             display: inline-flex;
             align-items: baseline;
        }
        .formula-input .q-field__control { 
            height: 28px !important; 
            min-height: 28px !important; 
            padding: 0 4px !important;
            background: rgba(255,255,255,0.1) !important;
            border-radius: 5px;
            border: 1px solid #4b5563;
            transform: translateY(3px); /* Piccola correzione per allinearlo al testo Times New Roman */
        }
        .formula-input .q-field__native { 
            color: #90caf9; 
            text-align: center; 
            padding: 0; 
            font-family: sans-serif;
            font-size: 0.9rem;
        }
    </style>
''')
                        with ui.column().classes("flex-1"):
                            with ui.card().classes("p-6 w-full max-w-4xl min-w-[600px] !bg-gray-900 text-white rounded-xl shadow-xl"):
                                ui.markdown(" CMB Thermodynamics Exercise: fill in the missing variables").classes("text-xl font-bold text-blue-300").props("role=heading aria-level=2 tabindex=0")
                            

                                answer_du, answer_dq, answer_pdv, answer_pv, answer_gamma, answer_cp, answer_cv, answer_meno1, answer_zero, answer_zrel, answer_gamma_uno, answer_p, answer_v, answer_32, answer_T, answer_52, answer_75, answer_53, answer_four, answer_3, answer_1, answer_lambda_obs, answer_lambda_emit, answer_1z = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

                                @ui.refreshable
                                def show_adiabatic_exercise():
                                    with ui.row().classes('w-full items-start justify-between gap-1'):
                                        with ui.column().classes('flex-1 gap-1'):

                                            # 1) First Law Thermodynamics
                                            ui.label("1) First Law Thermodynamics").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                answer_dq["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('=')
                                                answer_du["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('+')
                                                answer_pdv["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            # 2) Adiabatic condition
                                            ui.label("2) Adiabatic condition").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                answer_dq["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('=')
                                                answer_zero["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                            
                                            with ui.row().classes("math-text"):
                                                answer_du["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('+')
                                                answer_pdv["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('= 0')

                                            # 3) Adiabatic Transformation
                                            ui.label("3) Adiabatic Transformation").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                answer_p["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_v["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">')
                                                answer_gamma["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span> = const')
                                            
                                            with ui.row().classes("math-text"):
                                                ui.html('T &propto;')
                                                answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">(')
                                                answer_gamma_uno["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html(')</span>')

                                            # 4) Adiabatic Index
                                            ui.label("4) Adiabatic Index").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                ui.html('&gamma; = ')
                                                answer_cp["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_cv["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                        with ui.column().classes('flex-1 gap-4 min-w-[300px]'):
                                            
                                            # 5) Energy Equipartition
                                            ui.label("5) Energy Equipartition").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                           
                                            ui.label("Monoatomic gas:").classes('text-md')
                                            with ui.row().classes("math-text"):
                                                ui.html('<span class="math-normal"></span> K =')
                                                answer_32["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('k<span class="math-sub">B</span> &middot;')
                                                answer_T["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&rarr; &gamma; = ')
                                                answer_53["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                            with ui.row().classes('items-center flex-wrap'):
                                                ui.label("Biatomic gas:").classes('text-md')
                                            with ui.row().classes("math-text"):
                                                ui.html('<span class="math-normal"></span> U =')
                                                answer_52["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('nR &middot;')
                                                answer_T["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&rarr; &gamma; = ')
                                                answer_75["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            # 6) Stefan-Boltzmann
                                            ui.label("6) Stefan-Boltzmann law for radiation").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                ui.html('U = a<span class="math-sub">rad</span> &middot;')
                                                answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">')
                                                answer_four["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span> &middot;')
                                                answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            with ui.row().classes("math-text"):
                                                ui.html('P = ')
                                                answer_1["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('/')
                                                answer_3["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('&middot;')
                                                answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">')
                                                answer_four["el2"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span> &middot;')
                                                answer_v["el4"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                      
                                        with ui.column().classes('flex-1 gap-4 min-w-[300px]'):
                                            
                                            # 7) Doppler–Redshift relation
                                            ui.label("7) Doppler–Redshift relation").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            with ui.row().classes("math-text"):
                                                ui.html('1 + z = ')
                                                answer_lambda_obs["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_lambda_emit["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                            
                                            ui.label("λ_emitted ∝ a(t) → universe expansion factor").classes("text-md ").props("tabindex=0")

                                            with ui.row().classes("math-text"):
                                                ui.html('a(t) = ')
                                                answer_1["el2"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('/')
                                                answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            # 8) Cosmological Application
                                            ui.label("8) Cosmological Application").classes("text-blue-200 font-bold text-lg").props('role=heading aria-level=3 tabindex=0')
                                            ui.label("The relation temperature-redshift: T(z) = T₀·(1+z)").classes("text-md ").props("tabindex=0")
                                            
                                            with ui.row().classes("math-text"):
                                                ui.html('T &propto; a(t)<span class="math-sup">(')
                                                answer_meno1["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html(')</span>')

                                            ui.label("Considering a section of the expanding universe V ∝ a(t)³").classes("text-md ").props("tabindex=0")
                                            with ui.row().classes('items-center flex-wrap'):
                                                ui.label("Adiabatic expansion:").classes('text-md')
                                            with ui.row().classes("math-text"):
                                                ui.html('<span class="math-normal"></span> T &propto; a(t)<span class="math-sup">(3(')
                                                answer_gamma_uno["el2"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('))</span>')

                                            with ui.row().classes("math-text"):
                                                ui.html('T &propto; ')
                                                answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">(')
                                                answer_zrel["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html(')</span>')

                                show_adiabatic_exercise()
                                def check_and_run_adiabatic():
                                    def _get(ans):
                                        return (ans.get("el") and ans["el"].value or "").strip().lower()

                                    dU = _get(answer_du)
                                    dQ = _get(answer_dq)
                                    PdV = _get(answer_pdv)
                                    pv = _get(answer_pv)
                                    gam = _get(answer_gamma)
                                    cp = _get(answer_cp)
                                    cv = _get(answer_cv)
                                    Ta = _get(answer_meno1)
                                    zrel = _get(answer_zrel)
                                    gamma_uno = _get(answer_gamma_uno)
                                    four = _get(answer_four)
                                    three = _get(answer_3)
                                    one = _get(answer_1)
                                    lambda_obs = _get(answer_lambda_obs)
                                    lambda_emit = _get(answer_lambda_emit)
                                    one_z = _get(answer_1z)
                                    tre_due = _get(answer_32)
                                    cinque_due = _get(answer_52)
                                    sette_cinque = _get(answer_75)
                                    cinque_tre = _get(answer_53)
                                    vol = _get(answer_v)
                                    temp = _get(answer_T)
                                    pres = _get(answer_p)
                                    zero = _get(answer_zero)

                                    def norm(s): return s.replace(" ", "").replace("−", "-").replace("–", "-")

                                    ok_dQ = norm(dQ) in ("dq", "d_q", "dq")
                                    ok_dU = norm(dU) in ("du", "Δu", "d_u", "deltau")
                                    ok_PdV = norm(PdV) in ("pdv", "p*dv", "pdv", "p dv")
                                    ok_pv = norm(pv) in ("γ", "gamma")
                                    ok_gam = norm(gam) in ("1-γ", "1-gamma", "1-g")
                                    ok_cp = norm(cp) in ("c_p", "cp")
                                    ok_cv = norm(cv) in ("c_v", "cv")
                                    ok_Ta = norm(Ta) in ("-1", "-1.0", "minus1")
                                    ok_zrel = norm(zrel) in ("3(γ-1)", "3(gamma-1)", "3(g-1)", "3*(γ-1)")
                                    ok_gamma_uno = norm(gamma_uno) in ("1-γ", "1-gamma", "1-g")
                                    ok_four = norm(four) in ("4",)
                                    ok_3 = norm(three) in ("3",)
                                    ok_1 = norm(one) in ("1",)
                                    ok_lambda_obs = norm(lambda_obs) in ("λ_observed", "lambda_observed", "lambdaobs", "lambda_obs", "λobs")
                                    ok_lambda_emit = norm(lambda_emit) in ("λ_emitted", "lambda_emitted", "lambdaemitted", "lambda_emit")
                                    ok_one_z = norm(one_z) in ("1+z", "1plusz", "1+redshift", "z+1")
                                    ok_32 = norm(tre_due) in ("3/2", "1.5")
                                    ok_52 = norm(cinque_due) in ("5/2", "2.5")
                                    ok_75 = norm(sette_cinque) in ("7/5", "1.4")
                                    ok_53 = norm(cinque_tre) in ("5/3", "1.6667")
                                    ok_v = norm(vol) in ("v", "volume")
                                    ok_T = norm(temp) in ("t", "temperature")
                                    ok_p = norm(pres) in ("p", "pressure")
                                    ok_zero = norm(zero) in ("0", "zero")

                                    def reset_styles(ans_dicts):
                                        for ans in ans_dicts:
                                            for el in ans.values():
                                                if el:
                                                
                                                    el.classes(remove='border-red-500 border-green-500 !bg-red-900/10 !bg-green-900/10')
                                                
                                                    el.classes(add='border border-gray-600')
                                                    el.update()

                                    def mark(ans, ok):
                                        for el in ans.values():
                                            if el:
                                            
                                                el.classes(remove='border-red-500 border-green-500 !bg-red-900/10 !bg-green-900/10')
                                            
                                                if ok:
                                                    el.classes(add='border-green-500 !bg-green-900/10')
                                                else:
                                                    el.classes(add='border-red-500 !bg-red-900/10')
                                            
                                                el.update()



                                
                                    reset_styles([
                                        answer_du, answer_dq, answer_pdv, answer_pv, answer_gamma, answer_cp, answer_cv,
                                        answer_meno1, answer_zero, answer_zrel, answer_gamma_uno, answer_p, answer_v,
                                        answer_32, answer_T, answer_52, answer_75, answer_53, answer_four, answer_3,
                                        answer_1, answer_lambda_obs, answer_lambda_emit, answer_1z
                                    ])


                                    mark(answer_dq, ok_dQ)
                                    mark(answer_du, ok_dU)
                                    mark(answer_pdv, ok_PdV)
                                    mark(answer_pv, ok_pv)
                                    mark(answer_gamma, ok_gam)
                                    mark(answer_cp, ok_cp)
                                    mark(answer_cv, ok_cv)
                                    mark(answer_meno1, ok_Ta)
                                    mark(answer_zrel, ok_zrel)
                                    mark(answer_gamma_uno, ok_gamma_uno)
                                    mark(answer_four, ok_four)
                                    mark(answer_3, ok_3)
                                    mark(answer_1, ok_1)
                                    mark(answer_lambda_obs, ok_lambda_obs)
                                    mark(answer_lambda_emit, ok_lambda_emit)
                                    mark(answer_1z, ok_one_z)
                                    mark(answer_32, ok_32)
                                    mark(answer_52, ok_52)
                                    mark(answer_75, ok_75)
                                    mark(answer_53, ok_53)
                                    mark(answer_v, ok_v)
                                    mark(answer_T, ok_T)
                                    mark(answer_p, ok_p)
                                    mark(answer_zero, ok_zero)

                                    if all([ok_dQ, ok_dU, ok_PdV, ok_pv, ok_gam, ok_cp, ok_cv, ok_Ta, ok_zrel, ok_gamma_uno, ok_four, ok_3, ok_1, ok_lambda_obs, ok_lambda_emit, ok_one_z, ok_32, ok_52, ok_75, ok_53, ok_v, ok_T, ok_p, ok_zero]):
                                        accessible_notify("✅ All correct! Well done.", type_="success")
                                    else:
                                        accessible_notify("❌ Some formulas are incorrect — check the red boxes.", type_="error")

                
                                aria_button("Check Exercise", "Check all formulas for correctness", on_click=check_and_run_adiabatic).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                        
#panel radiation-matter
            def add_radiation_matter_activity(container):
            
                H0 = 2.27e-18  # s^-1
                G = 6.67430e-11  # m³/kg/s²
                rho_c = 9.2e-27  # kg/m³
                Omega_m0 = 0.315
                Omega_r0 = 9.24e-5
                T0 = 2.725  # K

            
                z_grid = np.logspace(-2, 5, 500)
                rho_m = Omega_m0 * rho_c * (1 + z_grid) ** 3
                rho_r = Omega_r0 * rho_c * (1 + z_grid) ** 4
                z_eq_true = Omega_m0 / Omega_r0 - 1
                T_eq_true = T0 * (1 + z_eq_true)

            
                answer_u, answer_rhoc, answer_Om0, answer_Or0, answer_a, answer_z, answer_G, answer_H0, answer_pi, answer_r, answer_T0, \
                answer_eq, answer_1z, answer_m, answer_v, answer_c, answer_E, answer_3, answer_h, answer_nu, answer_lambda, answer_T, \
                answer_n, answer_m4, answer_4, answer_m3, answer_rhom, answer_rhor, answer_rhom0, answer_rhor0 = \
                {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

                with container:
                    description_on_dark(
    "Compare the evolution of matter and radiation densities to identify the key transition epochs in the history of the Universe."
)
                    with ui.dialog() as i_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
    

    <h3>Activity Instructions</h3>
    <ol style="margin-top: 10px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b>Derive the Laws:</b> Fill in the missing power laws in the formula box. Recall that matter dilutes with volume (<span class="math">\(V \propto a^3\)</span>), while radiation loses energy due to both volume expansion and redshift (<span class="math">\(E \propto 1/a\)</span>).
        </li>
        <li style="margin-bottom: 8px;">
            <b>Generate the Plot:</b> Once inputs are correct, the curves for Matter and Radiation density vs. Redshift will appear.
        </li>
        <li style="margin-bottom: 8px;">
            <b>Find the Equivalence Epoch:</b> Locate the intersection point where <span class="math">\(\rho_m = \rho_r\)</span>. Identify the equivalence redshift <span class="math">\(z_{eq}\)</span> from the plot (expected <span class="math">\(z_{eq} \approx 3400\)</span>).
        </li>
        <li style="margin-bottom: 8px;">
            <b>Calculate Temperature:</b> Use the relation <span class="math">\(T(z) = T_0(1+z)\)</span> to find the temperature of the Universe at that epoch (<span class="math">\(T_{eq}\)</span>).
        </li>
    </ol>
""").props("id=equivalence-info role=document aria-live=polite")
                        aria_button("Close", label="Close the box", on_click=lambda: i_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    with ui.dialog() as eq_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        html_info_box(r"""
        <h3>Equivalence between Matter and Radiation Densities</h3>

        <p>After the Big Bang, the Universe passed through three main eras:</p>
        <ul>
            <li><b>Radiation-dominated epoch:</b> <span class="math">\( \rho_r > \rho_m \)</span></li>
            <li><b>Matter–radiation equality:</b> <span class="math">\( \rho_r = \rho_m \)</span></li>
            <li><b>Matter-dominated epoch:</b> <span class="math">\( \rho_m > \rho_r \)</span></li>
        </ul>
        



        <p>During the cosmic expansion, the density of matter decreases as <span class="math">\( a^{-3} \)</span>, while radiation decreases faster <span class="math">\( a^{-4} \)</span> since photons lose energy by redshifting (<span class="math">\( \lambda \propto a \Rightarrow E \propto 1/a \)</span>).</p>

        <p>From classical physics:</p>
        <ul>
            <li><b>Non-relativistic matter:</b> <span class="math">\( E \approx mc^2 \Rightarrow \rho_m \propto a^{-3} \)</span></li>
            <li><b>Radiation:</b> <span class="math">\( E_\gamma = h\nu = hc/\lambda \Rightarrow \rho_r \propto a^{-4} \)</span></li>
            <li>Using Stefan–Boltzmann law <span class="math">\( \rho_r = \alpha_{\rm rad} T^4 \)</span> and adiabatic expansion <span class="math">\( T \propto 1/a \)</span>, we again get <span class="math">\( \rho_r \propto a^{-4} \)</span></li>
        </ul>

        <p>The equality redshift <span class="math">\( z_{\rm eq} \)</span> satisfies <span class="math">\( \rho_m(z_{\rm eq}) = \rho_r(z_{\rm eq}) \)</span>.</p>
    """).props('tabindex=0 role=document aria-label="Equivalence between matter and radiation densities information"')

                        aria_button("Close", label="Close the box", on_click=lambda: eq_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                    
                    with ui.dialog() as cur_rad, ui.card().classes('p-4 w-full max-w-[600px]').props("id=goldilocks-info role=document aria-live=polite"):
                        html_info_box(r"""
                        <h3>The "Goldilocks" Epoch</h3>
                        <p>Today the CMB temperature is a freezing 2.7 K. However, about 10-17 million years after the Big Bang ($z \approx 100$), the temperature of the Universe was between <b>0°C and 100°C</b>.</p>
                        <p>Some scientists speculate that for a few million years, the entire Universe was a habitable zone where liquid water could exist anywhere.</p>
                        """)
                        reference_box("**Source:** [Loeb (2014)](https://arxiv.org/pdf/1312.0613)")
                        aria_button("Close", "close", on_click=lambda:cur_rad.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                    

                    with ui.dialog() as eq_legend_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props("id=legend-info role=document aria-live=polite"):
                        html_info_box(r"""
        <h3>Legend: Symbols and Constants</h3>
        <ul>
            <li><b>ρ_m, ρ_r</b> = matter / radiation energy density [kg·m⁻³]</li>
            <li><b>Ω_m₀, Ω_r₀</b> = present-day density parameters (dimensionless)</li>
            <li><b>ρ_c</b> = critical density = <span class="math">\( 3 H_0^2 / 8 \pi G \)</span></li>
            <li><b>H₀</b> = Hubble constant = <span class="math">\( 2.27\times10^{-18} \)</span> s⁻¹</li>
            <li><b>G</b> = gravitational constant = <span class="math">\( 6.67\times10^{-11} \)</span> m³·kg⁻¹·s⁻²</li>
            <li><b>T₀</b> = present-day CMB temperature = 2.725 K</li>
            <li><b>z</b> = redshift (dimensionless)</li>
            <li><b>a(t)</b> = scale factor = 1/(1+z)</li>
            <li><b>α_rad</b> = radiation density constant = <span class="math">\( 7.56\times10^{-16} \)</span> J·m⁻³·K⁻⁴</li>
            <li><b>h</b> = Planck constant = <span class="math">\( 6.626\times10^{-34} \)</span> J·s</li>
            <li><b>c</b> = speed of light = <span class="math">\( 2.998\times10^{8} \)</span> m/s</li>
            <li><b>λ</b> = photon wavelength [m]</li>
            <li><b>ν</b> = photon frequency [Hz]</li>
        </ul>
    """).props('tabindex=0 role=document aria-label="Legend of symbols and constants used in the exercise"')
                        aria_button("Close", label="Close the box", on_click=lambda: eq_legend_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    ui.add_head_html('''
    <style>
 
        .math-text { 
            font-family: 'Times New Roman', serif; 
            font-style: italic; 
            font-size: 1.3rem; 
            display: flex; 
            align-items: baseline; /* Allinea il testo sulla riga di scrittura */
            flex-wrap: wrap; 
            gap: 5px; 
            line-height: 1.6;
        }

        /* Testo descrittivo normale */
        .math-normal { 
            font-family: sans-serif;
            font-style: normal; 
            font-size: 0.9rem; 
            margin-right: 6px; 
            color: #9ca3af; 
        }
        
        /* Apici e Pedici posizionati in modo relativo per non rompere la riga */
        .math-sub { 
            font-size: 0.7em; 
            position: relative; 
            top: 0.3em; 
            margin-right: 1px;
        }
        .math-sup { 
            font-size: 0.7em; 
            position: relative; 
            top: -0.4em; 
            margin-left: 1px;
        }
        
        /* Input ottimizzato per allinearsi alla baseline */
        .formula-input {
             display: inline-flex;
             align-items: baseline;
        }
        .formula-input .q-field__control { 
            height: 28px !important; 
            min-height: 28px !important; 
            padding: 0 4px !important;
            background: rgba(255,255,255,0.1) !important;
            border-radius: 5px;
            border: 1px solid #4b5563;
            transform: translateY(3px); /* Piccola correzione per allinearlo al testo Times New Roman */
        }
        .formula-input .q-field__native { 
            color: #90caf9; 
            text-align: center; 
            padding: 0; 
            font-family: sans-serif;
            font-size: 0.9rem;
        }
    </style>
''')

                    with ui.row().classes("w-full justify-center items-start gap-6 "):
                        with ui.column().classes("flex-1"):
                            with ui.card().classes("p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                                ui.markdown(" Radiation & Matter Density Evolution").classes("text-2xl font-bold mb-4 text-blue-300").props("role=heading aria-level=2 tabindex=0")
                                
                                @ui.refreshable
                                def show_formulas():
                                    with ui.row().classes('w-full  justify-center gap-1'):
                                    
                                    
                                        with ui.column().classes("flex-1 gap-1"):
    
                                            # 1) Energy density definition
                                            ui.label("1) Energy density definition").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes("math-text"):
                                                ui.html('&rho; = ')
                                                answer_u["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_v["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            # 2) Matter particles energy
                                            ui.label("2) Matter particles energy").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("The energy of non relativistic matter particles:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('E<span class="math-sub">m</span> = (')
                                                answer_m["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_c["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">2</span>) + (1/2 &middot;')
                                                answer_m["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">2</span>)')

                                            # 3) Matter energy density
                                            ui.label("3) Matter energy density").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('&rho;<span class="math-sub">m</span> = ')
                                                answer_E["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("Considering a portion of universe:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                
                                                ui.html('<span class="math-normal"> </span> V &propto; a(t)<span class="math-sup">3</span>')

                                            with ui.row().classes("math-text"):
                                                ui.html('&rho;<span class="math-sub">m</span> = &Omega;<span class="math-sub">m</span> &middot; &rho;<span class="math-sub">c</span> <span class="math-normal">with</span> &rho;<span class="math-sub">c</span> = (3 &middot; H<span class="math-sub">0</span><span class="math-sup">2</span>) / (8 &middot; &pi; &middot; G)')

                                            with ui.row().classes("math-text"):
                                                ui.html('&rho;<span class="math-sub">m</span>(a) = &rho;<span class="math-sub">m0</span> &middot; a<span class="math-sup">(')
                                                answer_m3["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('<span class="math-sup">)</span> = &rho;<span class="math-sub">m0</span> &middot; (')
                                                answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html(')<span class="math-sup">')
                                                answer_3["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span>')

                                            with ui.row().classes("items-center math-text"):
                                                ui.html('<span class="math-normal">with</span> &rho;<span class="math-sub">m0</span> = &Omega;<span class="math-sub">m0</span> &middot; &rho;<span class="math-sub">c</span>')
                                  
                                            # 4) Radiation energy density
                                            ui.label("4) Radiation energy density").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("The energy for photons:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes("math-text"):
                                                
                                                ui.html('<span class="math-normal"> </span> E<span class="math-sub">r</span> =')
                                                answer_h["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_nu["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('=')
                                                answer_h["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_c["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_lambda["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("The wavelength stretches with universe expansion: ").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                
                                                ui.html('<span class="math-normal"></span> &lambda; &propto; a(t)')
                                         
                                            
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("The number of photons per volume decreases with expansion: ").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('<span class="math-normal"> </span> n &propto; a(t)<span class="math-sup">-3</span>')
                                        with ui.column().classes("flex-1 gap-1"): 
                                            
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("Every photon energy decreases with expansion:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('<span class="math-normal"> </span> E<span class="math-sub">r</span> &propto; a(t)<span class="math-sup">-1</span>')

                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("From the energy equation:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('<span class="math-normal"> </span> &rho;<span class="math-sub">r</span> = ')
                                                answer_n["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot;')
                                                answer_E["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&propto; a(t)<span class="math-sup">(')
                                                answer_m4["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('<span class="math-sup">)</span>')

                                        
                                            with ui.row().classes("items-center flex-wrap"):
                                                ui.label("From the Stefan-Boltzmann law:").classes("text-md").props("tabindex=0")
                                            with ui.row().classes("items-center math-text"):
                                                ui.html('<span class="math-normal"> </span> &rho;<span class="math-sub">r</span> = a<span class="math-sub">rad</span> &middot;')
                                                answer_T["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('<span class="math-sup">')
                                                answer_4["el"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span>')

                                            with ui.row().classes(" math-text"):
                                                ui.html('<span class="math-normal">with</span> T &propto; a(t)<span class="math-sup">-1</span>')

                                            with ui.row().classes(" math-text"):
                                                ui.html('&rho;<span class="math-sub">r</span> = &Omega;<span class="math-sub">r</span> &middot; &rho;<span class="math-sub">c</span>')
                                        
                                            with ui.row().classes("math-text"):
                                                ui.html('&rho;<span class="math-sub">r</span>(a) = &rho;<span class="math-sub">r0</span> &middot; a<span class="math-sup">(')
                                                answer_m4["el2"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('<span class="math-sup">)</span> = &rho;<span class="math-sub">r0</span> &middot; (')
                                                answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html(')<span class="math-sup">')
                                                answer_4["el2"] = aria_formula_input().props("dense filled").classes("w-10 formula-input")
                                                ui.html('</span>')

                                            with ui.row().classes(" math-text"):
                                                ui.html('<span class="math-normal">with</span> &rho;<span class="math-sub">r0</span> = &Omega;<span class="math-sub">r0</span> &middot; &rho;<span class="math-sub">c</span>')
                                        
                                            # 5) Equivalence condition
                                            ui.label("5) Equivalence condition").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes("math-text"):
                                                answer_rhom["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('(z<span class="math-sub">eq</span>) = ')
                                                answer_rhor["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('(z<span class="math-sub">eq</span>)')

                                            # 6) Redshift of equivalence
                                            ui.label("6) Redshift of equivalence").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('1 + z<span class="math-sub">eq</span> = ')
                                                answer_rhom0["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('/')
                                                answer_rhor0["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")

                                            # 7) Temperature–redshift relation
                                            ui.label("7) Temperature–redshift relation").classes("text-blue-200 font-bold text-lg").props("role=heading aria-level=3 tabindex=0")
                                            with ui.row().classes(" math-text"):
                                                ui.html('T(z) = ')
                                                answer_T0["el"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html('&middot; (')
                                                answer_1z["el3"] = aria_formula_input().props("dense filled").classes("w-12 formula-input")
                                                ui.html(')')

                                

                                show_formulas()
                                formula_button_slot = ui.row().classes(" w-full items-center justify-center")

                        with ui.column().classes("flex-1 justify-start !bg-gray-900 p-4 rounded-xl shadow-lg"):

                
                            with ui.row().classes("justify-center w-full gap-2 flex-wrap"):
                                aria_button("Instructions","instruction",on_click=lambda:[i_dialog.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button(
                                        "Info: Matter-Radiation",
                                        "Read detailed information about matter-radiation equivalence",
                                        on_click=lambda: [
                                            eq_dialog.open(),
                                            ui.run_javascript("MathJax.typesetPromise()")
                                        ]
                                    ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button(
                                        "Legend",
                                        "Open legend of symbols and constants",
                                        on_click=lambda: [
                                            eq_legend_dialog.open(),
                                            ui.run_javascript("MathJax.typesetPromise()")
                                        ]
                                    ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                                generate_plot_btn = aria_button("Generate Plot", "Plot densities").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded ")
                                aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_rad.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                            with ui.row().classes("w-full justify-center items-center gap-12 flex-wrap"):

    
                                with ui.column().classes("items-center gap-4"):
                                    
                                    unit_select = ui.select(
                                        {
                                            "kg/m³": "kg/m³",
                                            "J/m³": "J/m³",
                                            "erg/cm³": "erg/cm³",
                                            "GeV/cm³": "GeV/cm³",
                                            "M⊙/Mpc³": "M⊙/Mpc³",
                                        },
                                        value="GeV/cm³",
                                        label="Density units",
                                    ).props("dense filled aria-label='Select density units'")

                                    xaxis_select = ui.select(
                                        {
                                            "z": "Redshift (z)",
                                            "a": "Scale factor a(t)",
                                            "t": "Cosmic time (t)",
                                        },
                                        value="z",
                                        label="X-axis variable",
                                    ).props("dense filled aria-label='Select x-axis variable'")

                                with ui.column().classes("items-center gap-3"):
                
            
                                    with ui.row().classes("items-center justify-center gap-1 flex-nowrap"):
                                        ui.label("ρₘ = ρₘ₀ · (")
                                        rho_m_z_input = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                        ui.label(")^")
                                        rho_m_power_input = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")

                                    with ui.row().classes("items-center justify-center gap-1 flex-nowrap"):
                                        ui.label("ρᵣ = ρᵣ₀ · (")
                                        rho_r_z_input = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                        ui.label(")^")
                                        rho_r_power_input = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                        
                            

                        
                            @ui.refreshable
                            def plot_curves(show_curves=False):
                                with ui.pyplot(figsize=(6, 4)):
                                
                                    c = 2.998e8
                                    rho_m_plot = np.copy(rho_m)
                                    rho_r_plot = np.copy(rho_r)
                                    ylabel = "Density [kg/m³]"

                                    if unit_select.value == "J/m³":
                                        rho_m_plot *= c**2
                                        rho_r_plot *= c**2
                                        ylabel = "Energy Density [J/m³]"
                                    elif unit_select.value == "erg/cm³":
                                        rho_m_plot *= c**2 * 1e-7
                                        rho_r_plot *= c**2 * 1e-7
                                        ylabel = "Energy Density [erg/cm³]"
                                    elif unit_select.value == "GeV/cm³":
                                        rho_m_plot *= 5.62e23
                                        rho_r_plot *= 5.62e23
                                        ylabel = "Energy Density [GeV/cm³]"
                                    elif unit_select.value == "M⊙/Mpc³":
                                        Msun = 1.989e30
                                        Mpc = 3.086e22
                                        rho_m_plot = rho_m_plot / Msun * Mpc**3
                                        rho_r_plot = rho_r_plot / Msun * Mpc**3
                                        ylabel = "Density [M⊙/Mpc³]"

                                
                                    if xaxis_select.value == "z":
                                        x = 1 + z_grid
                                        xlabel = "1 + z"
                                    elif xaxis_select.value == "a":
                                        x = 1 / (1 + z_grid)
                                        xlabel = "Scale factor a(t)"
                                    elif xaxis_select.value == "t":
                                    
                                        t_grid = (2 / (3 * H0)) * (1 + z_grid) ** (-1.5)
                                        #x = t_grid / (1e17)  # in 10^17 s ≈ Gyr
                                        #x = t_grid / 3.154e7 #yr
                                        x = t_grid / 3.154e13 #Myr
                                        
                                        xlabel = "Cosmic time [Myr]"

                            
                                    plt.title("Matter and Radiation Density Evolution",fontweight='bold')
                                    plt.xlabel(xlabel)
                                    plt.ylabel(ylabel)
                                    plt.grid(True, which="both", ls="--", alpha=0.5)
                                    plt.xscale("log")
                                    plt.yscale("log")

                                    if show_curves:
                                        plt.plot(x, rho_m_plot, label="Matter density ρₘ", color="blue", lw=2)
                                        plt.plot(x, rho_r_plot, label="Radiation density ρᵣ", color="red", lw=2)
                                        plt.legend()

                                    plt.tight_layout()
                                    aria_chart_label("Plot showing the evolution of matter and radiation densities")
                            
                                with ui.row().classes("gap-4 "):
                                    z_eq_input = ui.number(label="z_eq", value=None).props("dense filled aria-label='Input for redshift of equivalence epoch'")
                                    T_eq_input = ui.number(label="T_eq [K]", value=None).props("dense filled aria-label='Input for temperature at equivalence epoch'")

                                    def check_results():
                                        try:
                                            z_val = float(z_eq_input.value)
                                            T_val = float(T_eq_input.value)
                                        except Exception:
                                            accessible_notify("Please enter numeric values.", type_="error")
                                            return

                                        ok_z = abs(z_val - z_eq_true) / z_eq_true < 0.1
                                        ok_T = abs(T_val - T_eq_true) / T_eq_true < 0.1

                                        if ok_z and ok_T:
                                            accessible_notify("✅ Both z_eq and T_eq are correct!", type_="success")
                                        elif ok_z:
                                            accessible_notify("⚠️ z_eq is correct, but T_eq is off.", type_="warning")
                                        elif ok_T:
                                            accessible_notify("⚠️ T_eq is correct, but z_eq is off.", type_="warning")
                                        else:
                                            accessible_notify("❌ Both results are incorrect.", type_="error")

                                    aria_button("Check Results", "Verify z_eq and T_eq", on_click=check_results).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")

                            with ui.row().classes("gap-2 justify-center "):
                                plot_curves(show_curves=False)

                    
                            def on_generate_plot():
                                zm = (rho_m_z_input.value or "").strip().lower()
                                zr = (rho_r_z_input.value or "").strip().lower()
                                pm = (rho_m_power_input.value or "").strip()
                                pr = (rho_r_power_input.value or "").strip()

                                ok_zm = zm in ("1+z", "1plusz", "z+1")
                                ok_zr = zr in ("1+z", "1plusz", "z+1")
                                ok_pm = pm in ("3", "³")
                                ok_pr = pr in ("4", "⁴")

                        
                                for el in [rho_m_z_input, rho_r_z_input, rho_m_power_input, rho_r_power_input]:
                                    el.classes(remove='border-red-500 border-green-500')

                            
                                rho_m_z_input.classes(add='border-green-500' if ok_zm else 'border-red-500')
                                rho_m_power_input.classes(add='border-green-500' if ok_pm else 'border-red-500')
                                rho_r_z_input.classes(add='border-green-500' if ok_zr else 'border-red-500')
                                rho_r_power_input.classes(add='border-green-500' if ok_pr else 'border-red-500')

                                if ok_zm and ok_zr and ok_pm and ok_pr:
                                    accessible_notify("✅ Formulas correct! Plot generated.", type_="success")
                                    plot_curves.refresh(show_curves=True)
                                else:
                                    accessible_notify("❌ Check the red boxes for errors.", type_="error")

                            generate_plot_btn.on("click", on_generate_plot)
                
                        def check_and_run_equivalence():
                                def _get(ans):
                                    return (ans.get("el") and ans["el"].value or "").strip().lower()

                                def norm(s): return s.replace(" ", "").replace("−", "-")

                                ok_Om0 = norm(_get(answer_Om0)) in ("ωm0", "omega_m0", "omegam0", "Ωm0")
                                ok_Or0 = norm(_get(answer_Or0)) in ("ωr0", "omega_r0", "omegar0", "Ωr0")
                                ok_a = norm(_get(answer_a)) in ("-3", "-4", "−3", "−4")
                                ok_G = norm(_get(answer_G)) in ("g", "gravitationalconstant")
                                ok_H0 = norm(_get(answer_H0)) in ("h0", "h_0")
                                ok_pi = norm(_get(answer_pi)) in ("π", "pi")
                                ok_r = norm(_get(answer_r)) in ("ρr", "rho_r")
                                ok_T0 = norm(_get(answer_T0)) in ("t0", "t_0")
                                ok_eq = norm(_get(answer_eq)) in ("z_eq", "zeq")
                                ok_m= norm(_get(answer_m)) in ("m", "mass")
                                ok_u= norm(_get(answer_u)) in ("u", "energy")
                                ok_v= norm(_get(answer_v)) in ("v", "volume")
                                ok_c= norm(_get(answer_c)) in ("c", "speedoflight")
                                ok_E= norm(_get(answer_E)) in ("e", "energy")
                                ok_3= norm(_get(answer_3)) in ("3",)
                                ok_h= norm(_get(answer_h)) in ("h", "planckconstant")   
                                ok_nu= norm(_get(answer_nu)) in ("ν", "nu", "frequency")
                                ok_lambda= norm(_get(answer_lambda)) in ("λ", "lambda", "wavelength")
                                ok_T= norm(_get(answer_T)) in ("t", "temperature")
                                ok_n= norm(_get(answer_n)) in ("n", "numberdensity")
                                ok_m4= norm(_get(answer_m4)) in ("-4", "−4")
                                ok_4= norm(_get(answer_4)) in ("4",)
                                ok_m3= norm(_get(answer_m3)) in ("-3", "−3")
                                ok_rhom= norm(_get(answer_rhom)) in ("ρm", "rhom", "rhomatter")
                                ok_rhor= norm(_get(answer_rhor)) in ("ρr", "rhor", "rhoradiation")
                                ok_rhom0= norm(_get(answer_rhom0)) in ("ρm0", "rhom0", "rhomatter0")
                                ok_rhor0= norm(_get(answer_rhor0)) in ("ρr0", "rhor0", "rhoradiation0")
                                ok_T0= norm(_get(answer_T0)) in ("t0", "t_0")   
                                ok_1z= norm(_get(answer_1z)) in ("1+z", "1plusz", "1+redshift", "z+1")
                                all_ok = all([ok_1z,ok_Om0, ok_Or0, ok_a, ok_G, ok_H0, ok_pi, ok_r, ok_T0, ok_eq, ok_m, ok_u, ok_v, ok_c, ok_E, ok_3, ok_h, ok_nu, ok_lambda, ok_T, ok_n, ok_m4, ok_4, ok_m3, ok_rhom, ok_rhor, ok_rhom0, ok_rhor0])

                                all_dicts = [answer_Om0, answer_Or0, answer_a, answer_G, answer_H0,
                                            answer_pi, answer_r, answer_T0, answer_eq, answer_m, answer_u, answer_v,
                                            answer_c, answer_E, answer_3, answer_h, answer_nu, answer_lambda, answer_T,
                                            answer_n, answer_m4, answer_4, answer_m3, answer_rhom, answer_rhor,
                                            answer_rhom0, answer_rhor0,answer_1z]
                                for ans in all_dicts:
                                    for el in ans.values():
                                        if el:
                                            el.classes(remove='border-red-500 border-green-500')

                                def mark(ans, ok):
                                    for el in ans.values():
                                        if el:
                                            el.classes(remove='border-red-500 border-green-500')
                                            el.classes(add='border-green-500' if ok else 'border-red-500')
                                            el.update()




                                mark(answer_Om0, ok_Om0)
                                mark(answer_Or0, ok_Or0)
                                mark(answer_a, ok_a)
                                mark(answer_G, ok_G)
                                mark(answer_H0, ok_H0)
                                mark(answer_pi, ok_pi)
                                mark(answer_r, ok_r)
                                mark(answer_T0, ok_T0)
                                mark(answer_eq, ok_eq)
                                mark(answer_m, ok_m)
                                mark(answer_u, ok_u)    
                                mark(answer_v, ok_v)
                                mark(answer_c, ok_c)
                                mark(answer_E, ok_E)
                                mark(answer_3, ok_3)
                                mark(answer_h, ok_h)    
                                mark(answer_nu, ok_nu)
                                mark(answer_lambda, ok_lambda)
                                mark(answer_T, ok_T)
                                mark(answer_n, ok_n)
                                mark(answer_m4, ok_m4)
                                mark(answer_4, ok_4)
                                mark(answer_m3, ok_m3)
                                mark(answer_rhom, ok_rhom)
                                mark(answer_rhor, ok_rhor)
                                mark(answer_rhom0, ok_rhom0)
                                mark(answer_rhor0, ok_rhor0)
                                mark(answer_1z, ok_1z)

                                if all_ok:
                                    accessible_notify("✅ All formulas are correct! Curves are now displayed.", type_="success")
                                    plot_curves.refresh(show_curves=True)
                                else:
                                    accessible_notify("❌ Some formulas are incorrect — check the red boxes.", type_="error")

                        with formula_button_slot:
                            aria_button(
                                "Check Formulas",
                                "Validate all formulas",
                                on_click=check_and_run_equivalence
                            ).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")

                
                    

            
                    

                
                        



            
        with ui.tab_panels(tabs, value='planck').classes('w-full !bg-transparent'):

            with ui.tab_panel('planck').props('role=tabpanel aria-label="Planck spectrum "'):
                planck_container = ui.column().classes('w-full p-4')
                add_planck_activity(planck_container)
            with ui.tab_panel('index').props('role=tabpanel aria-label="CMB Adiabatic Index "'):
                cmb_container = ui.column().classes('w-full p-4')
                add_cmb_activity(cmb_container)
                
            with ui.tab_panel('radmat').props('role=tabpanel aria-label="Radiation & Matter"'):
               
                radmat_container = ui.column().classes('w-full p-4')
                add_radiation_matter_activity(radmat_container)
        


