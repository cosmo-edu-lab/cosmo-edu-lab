# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.

from asyncio import events
import os
import io
import json
import textwrap
import uuid
import base64
import datetime
import threading
import random
from io import BytesIO
import json
import re
import time
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
from plotly.subplots import make_subplots
from PIL import Image
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
from astroquery.simbad import Simbad
import asyncio           
import logging
from nicegui.events import MouseEventArguments
from layout import *
from core import *
def create_page():
    @ui.page('/module1')
    def introduction():
    #add CSS styles
        ui.add_head_html('''
    <link rel="stylesheet" href="/static/github.min.css">
''')
        #ui.add_head_html('''    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"> ''')
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

    .description-on-dark {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3); /* Ombra leggermente più leggera rispetto al titolo */
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


    .plot-info-box.compact2 {
        /* Aspetto generale */
        background-color: rgba(255, 255, 255, 0.85); /* Bianco semitrasparente */
        backdrop-filter: blur(4px); /* Effetto sfocatura per leggibilità */
        padding: 6px 10px; /* Padding interno ridotto */
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        max-width: 200px; /* Larghezza massima per non coprire troppo */
        font-size: 0.7rem; /* Testo piccolo */
    }

    .plot-info-box.compact2 .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px; /* Spazio tra etichetta e valore */
        line-height: 1.4;
    }

    .plot-info-box.compact2 .info-row .label {
        /* Rende l'etichetta meno invadente */
        font-weight: 400; 
        white-space: nowrap; /* Non spezzare l'etichetta */
    }

    .plot-info-box.compact2 .info-row .value {
        /* Allinea il valore a destra e lo mette in evidenza */
        text-align: right;
        flex-shrink: 0; /* Impedisce al valore di restringersi */
    }

    </style>
    """)

      

       
        #s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg-full.js';
        main_layout("Introduction to Cosmology")
        tab_key = 'module1_selected'
        if tab_key not in app.storage.user:
            app.storage.user[tab_key] = 'intro'
        with ui.tabs().classes('w-full justify-center').bind_value(app.storage.user, tab_key) as tabs: 
            ui.tab('intro', label='Structures of Universe').props('aria-label="Structures of Universe"')
            ui.tab('discovery', label='Discoveries Timeline').props('aria-label="Discoveries Timeline"')
            ui.tab('universe', label='Universe Evolution').props('aria-label="Universe Evolution"')
            ui.tab('instrument', label='Observational Methods').props('aria-label="Observational Instruments and Methods"')
            ui.tab('planets', label='Planets ').props('aria-label="Planets and Milky Way"')
            ui.tab('galaxy', label='Galaxies ').props('aria-label="Galaxy Map"')
            ui.tab('stars', label='Stars ').props('aria-label="Stars Map"')
            ui.tab('particles', label='Fundamental Particles').props('aria-label="Fundamental Particles"')
#functions for observational methods and instruments dialog        
        def open_scientific_info_dialog():
            with ui.dialog() as scientific_dialog, ui.card().classes('p-0 w-full max-w-[1400px] h-[85vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="title-dialog"'):
                
                with ui.column().classes('w-full h-full bg-white flex flex-col'):
                   
                    with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="title-dialog"'):
                        ui.label('Scientific Information').classes('text-xl font-bold').props('id="title-dialog" role=heading aria-level=2 tabindex=0')
                        aria_button('Close', 'close', on_click=scientific_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                   
                    with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                        t_ccd = ui.tab('CCD Concepts')
                        t_math = ui.tab('Signal-Noise Math')
                        t_methods = ui.tab('Methods')
                        t_instr = ui.tab('Instruments')
                        t_env = ui.tab('Environments')
                        t_biblio = ui.tab('Bibliography')

                        tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))
                    with ui.tab_panels(tabs, value=t_ccd).classes('w-full flex-1 overflow-y-auto bg-gray-50 text-slate-900 p-6'):
                        
                        with ui.tab_panel(t_ccd):
                            html_info_box(r"""
                        <h3 class="text-slate-800 mb-4">🔭 CCD Concepts & Physics</h3>

                        <div class="space-y-8">

                            <div>
                                <h4 class="text-indigo-900 font-bold border-b border-indigo-100 pb-1 mb-2">1. Context: What is a CCD?</h4>
                                
                                <p class="text-gray-700 text-sm mb-2">
                                    The <b>CCD (Charge-Coupled Device)</b> is the "electronic retina" of modern telescopes. It connects the optical world to the digital world.
                                </p>

                                <div class="bg-blue-50 p-3 rounded border-l-4 border-blue-500 text-sm text-gray-800">
                                    <strong>The "Rain & Buckets" Analogy:</strong><br>
                                    Imagine a telescope as a funnel and the CCD as a grid of buckets placed underneath.
                                    <ul class="list-disc ml-5 mt-1 space-y-1">
                                        <li><b>The Rain (Photons):</b> Light travels from the star and is collected by the telescope mirror.</li>
                                        <li><b>The Buckets (Pixels):</b> The CCD captures these photons. Through the <i>photoelectric effect</i>, each photon that hits a pixel knocks loose an electron.</li>
                                        <li><b>The Count (Data):</b> At the end of the exposure, the computer counts how many electrons are in each bucket.</li>
                                    </ul>
                                </div>
                            </div>

                            <div>
                                <h4 class="text-indigo-900 font-bold border-b border-indigo-100 pb-1 mb-2">2. The Battle: Signal vs. Noise</h4>
                                <p class="text-gray-700 text-sm">The quality of an image is defined by the <b>Signal-to-Noise Ratio (SNR)</b>.</p>
                                <ul class="list-disc ml-6 mt-2 text-sm text-gray-800">
                                    <li><b>Signal (\(S\)):</b> The "good" photons from the star/galaxy.</li>
                                    <li><b>Noise (\(N\)):</b> The "bad" fluctuations (Poisson Noise + Sky Background).</li>
                                </ul>

                                <div style="margin: 15px 0; padding: 10px; background-color: rgba(2, 132, 199, 0.1); border-left: 4px solid #0284c7; border-radius: 4px;">
                                    <p style="margin:0; font-style:italic; color:#0369a1; font-size: 0.9em;">
                                        <b>The Golden Rule:</b> Collect enough photons so the Signal stands out above the Noise.
                                    </p>
                                </div>
                            </div>

                        </div>
                        """)

                        with ui.tab_panel(t_math):
                            html_info_box(r"""
                <h3 class="text-slate-800 mb-4">📊 Scientific Background & Logic</h3>

                <div class="space-y-8">

                    <div class="mb-8">
                        <h4 class="text-indigo-900 font-bold text-lg border-b border-indigo-100 pb-1 mb-2">1. How this Simulation Works</h4>
                        
                        <div class="float-right ml-4 mb-2 w-1/3">
                            <img src="/images/eso9845d.jpg" class="rounded shadow-md border border-gray-300 w-full" alt="Target Galaxy">
                            <p class="text-[10px] text-gray-500 text-center mt-1">
                                Target Example: NGC 1232 (Credit: <a href="https://www.eso.org/public/images/eso9845d/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline">ESO</a>)
                            </p>
                        </div>

                        <ol class="list-decimal ml-6 space-y-2 text-sm text-gray-800">
                            <li>
                                <b>The Input (Truth):</b> We start by reading the pixel values from the real image shown on the right.
                            </li>
                            <li>
                                <b>Physics Scaling:</b> We multiply these values by the <b>Telescope Area (\(D^2\))</b> and <b>Exposure Time (\(t\))</b>.
                            </li>
                            <li>
                                <b>Adding Noise (The CCD Equation):</b> 
                                We define the Signal-to-Noise Ratio (SNR) according to Howell (2006):
                                <div class="my-4 py-3 bg-gray-100 text-center rounded overflow-x-auto text-lg font-bold">
                                    \[ SNR = \frac{S_{star} \cdot t}{\sqrt{ (S_{star} + B_{sky} + D_{dark}) \cdot t + R_{read}^2 }} \]
                                </div>
                                We use <code>np.random.poisson()</code> to simulate the random arrival of photons based on this formula.
                            </li>
                        </ol>
                        <div style="clear: both;"></div>
                    </div>

                    <div class="mb-4">
                        <h4 class="text-indigo-900 font-bold text-lg border-b border-indigo-100 pb-1 mb-2">2. Simulation Parameters</h4>
                        <p class="text-sm mb-2 text-gray-600">Standard values for a cooled amateur CCD:</p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div class="p-3 bg-gray-100 rounded border-l-4 border-gray-500">
                                <b>Dark Current = 1.0 \(e^-\)/s</b><br>
                                <span class="text-xs text-gray-600">Thermal noise at approx -20°C.</span>
                            </div>
                            <div class="p-3 bg-gray-100 rounded border-l-4 border-gray-500">
                                <b>Read Noise = 10.0 \(e^-\)</b><br>
                                <span class="text-xs text-gray-600">Electronic readout error.</span>
                            </div>
                        </div>
                    </div>

                    <div class="bg-gray-100 p-4 rounded-lg border border-gray-300 mt-6">
                        <h4 class="text-indigo-900 font-bold mb-3 border-b border-gray-300 pb-1">3. Behind the Simulation: Sky & Optics</h4>

                        <div class="mb-4">
                            <h5 class="font-bold text-slate-800 text-sm mb-1">A. The 3 Sky Levels (\(B_{sky}\)) explained</h5>
                            <p class="text-xs text-gray-600 mb-2">We use specific photon flux rates (\(e^-/\mathrm{pixel}/\mathrm{sec}\)) based on real sky brightness magnitudes:</p>
                            <ul class="list-disc ml-5 text-sm text-gray-700 space-y-2">
                                <li>
                                    <b>Dark Mountain (Value = 10):</b> <br>
                                    Corresponds to a dark site (~21.5 mag/arcsec²). Only natural airglow exists. The noise is minimal (\(\sqrt{10} \approx 3.1\)).
                                </li>
                                <li>
                                    <b>Suburban (Value = 100):</b> <br>
                                    Corresponds to ~19.5 mag/arcsec². Light pollution increases the background by <b>10x</b>, making the noise \(\sqrt{100} = 10\) times higher.
                                </li>
                                <li>
                                    <b>City Center (Value = 500):</b> <br>
                                    Corresponds to a bright city sky (~17.5 mag/arcsec²). The sky is <b>50x brighter</b> than the mountain. The resulting "Shot Noise" (\(\sqrt{500} \approx 22\)) completely buries the faint galaxy signal.
                                </li>
                            </ul>
                        </div>

                        <div>
                            <h5 class="font-bold text-slate-800 text-sm mb-1">B. Linking Sliders to the Formula</h5>
                            
                            <ul class="list-disc ml-5 text-sm text-gray-700 space-y-2">
                                <li>
                                    <b>Telescope Diameter Slider:</b> 
                                    Increases <b>\(S_{star}\)</b>. Since Area \(\propto D^2\), doubling the diameter captures <b>4x more photons</b>.
                                </li>
                                <li>
                                    <b>Exposure Time Slider:</b> 
                                    Increases <b>\(t\)</b>. While Noise grows as \(\sqrt{t}\), Signal grows as \(t\). Thus, longer exposures always improve the image quality.
                                </li>
                            </ul>
                        </div>
                    </div>

                </div>
                """)
                        with ui.tab_panel(t_methods):
                            html_info_box(r"""
                                <h3 class="text-2xl font-bold text-slate-800 mb-6">Key Observational Methods</h3>
                                <div class="grid grid-cols-1 gap-6">
                                    
                                    <div class="bg-white p-4 rounded-lg shadow-sm border-l-4 border-indigo-500">
                                        <div class="flex items-start">
                                            <i class="material-icons text-indigo-600 mr-3 text-3xl">spectrum</i>
                                            <div>
                                                <strong class="text-indigo-900 text-xl block mb-1">Spectroscopy</strong>
                                                <p class="text-gray-700 leading-relaxed">
                                                    The study of light dispersed into its constituent wavelengths (spectrum). 
                                                    By analyzing spectral lines (emission/absorption), astronomers can measure:
                                                    <ul class="list-disc ml-5 mt-2 text-sm text-gray-600">
                                                        <li><b>Chemical Composition:</b> Each element leaves a unique fingerprint.</li>
                                                        <li><b>Radial Velocity:</b> Via the Doppler Shift (\(z\)).</li>
                                                        <li><b>Physical Properties:</b> Temperature, density, and pressure of the source.</li>
                                                    </ul>
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="bg-white p-4 rounded-lg shadow-sm border-l-4 border-blue-500">
                                        <div class="flex items-start">
                                            <i class="material-icons text-blue-600 mr-3 text-3xl">brightness_6</i>
                                            <div>
                                                <strong class="text-blue-900 text-xl block mb-1">Photometry</strong>
                                                <p class="text-gray-700 leading-relaxed">
                                                    The precise measurement of the intensity of light (flux) from astronomical objects.
                                                    It is often performed using standard filters (U, B, V, R, I).
                                                    <ul class="list-disc ml-5 mt-2 text-sm text-gray-600">
                                                        <li><b>Variable Stars:</b> Measuring light curves to find periods (e.g., Cepheids).</li>
                                                        <li><b>Exoplanets:</b> Detecting transits (dips in brightness).</li>
                                                        <li><b>Supernovae:</b> Tracking the explosion's evolution over time.</li>
                                                    </ul>
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="bg-white p-4 rounded-lg shadow-sm border-l-4 border-purple-500">
                                        <div class="flex items-start">
                                            <i class="material-icons text-purple-600 mr-3 text-3xl">settings_input_antenna</i>
                                            <div>
                                                <strong class="text-purple-900 text-xl block mb-1">Interferometry</strong>
                                                <p class="text-gray-700 leading-relaxed">
                                                    A technique that combines signals from multiple telescopes (acting as an array) to interfere with each other.
                                                    This simulates a telescope with a diameter equal to the maximum separation (baseline) between the antennas, drastically increasing <b>Angular Resolution</b>.
                                                    <br><i>Examples: VLA (Radio), VLTI (Optical), EHT (Black Hole Imaging).</i>
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="bg-white p-4 rounded-lg shadow-sm border-l-4 border-green-500">
                                        <div class="flex items-start">
                                            <i class="material-icons text-green-600 mr-3 text-3xl">blur_on</i>
                                            <div>
                                                <strong class="text-green-900 text-xl block mb-1">Adaptive Optics (AO)</strong>
                                                <p class="text-gray-700 leading-relaxed">
                                                    A technology used in ground-based telescopes to correct the blurring effects of Earth's atmosphere (seeing) in real-time.
                                                    It uses deformable mirrors that change shape hundreds of times per second, guided by a laser "artificial star".
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                </div>
                                """)

                        with ui.tab_panel(t_instr):
                            html_info_box(r"""
                            <h3 class="text-2xl font-bold text-slate-800 mb-6">Major Instruments & Technology</h3>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                
                                <div class="bg-slate-50 p-5 rounded-lg border border-slate-200 flex flex-col justify-between">
                                    <div>
                                        <h4 class="text-xl font-bold text-slate-900 mb-3 border-b pb-2">🔭 Telescopes (The Collectors)</h4>
                                        <p class="text-gray-700 mb-3">Their main job is to collect light ("Light Buckets"), not just to magnify.</p>
                                        <ul class="list-none space-y-2 text-gray-800">
                                            <li class="flex items-start"><span class="text-blue-500 font-bold mr-2">➤</span> 
                                                <b>Refractors:</b> Use lenses. Limited in size due to lens weight/sagging.
                                            </li>
                                            <li class="flex items-start"><span class="text-blue-500 font-bold mr-2">➤</span> 
                                                <b>Reflectors:</b> Use mirrors. Modern giant telescopes (8m-39m) are all reflectors, often using segmented hexagonal mirrors (e.g., JWST, ELT).
                                            </li>
                                            <li class="flex items-start"><span class="text-blue-500 font-bold mr-2">➤</span> 
                                                <b>Radio Telescopes:</b> Huge dishes (or arrays) to focus long-wavelength radio waves.
                                            </li>
                                        </ul>
                                    </div>
                                    
                                    <div class="mt-4 pt-4 border-t border-slate-200">
                                        <img src="/images/hubble_tel.jpg" class="w-full rounded shadow-sm border border-gray-300 mb-2" alt="Hubble Telescope">
                                        <div class="text-right">
                                            <a href="https://science.nasa.gov/mission/hubble/observatory/design/instruments/" target="_blank" class="text-xs text-blue-600 hover:text-blue-800 hover:underline flex items-center justify-end gap-1">
                                                Source: NASA Hubble Instruments <i class="material-icons text-[10px]">open_in_new</i>
                                            </a>
                                        </div>
                                    </div>
                                </div>

                                <div class="bg-slate-50 p-5 rounded-lg border border-slate-200 flex flex-col justify-between">
                                    <div>
                                        <h4 class="text-xl font-bold text-slate-900 mb-3 border-b pb-2">📷 Detectors (The Eyes)</h4>
                                        <p class="text-gray-700 mb-3">Devices that convert photons into digital signals.</p>
                                        <ul class="list-none space-y-2 text-gray-800">
                                            <li class="flex items-start"><span class="text-green-500 font-bold mr-2">➤</span> 
                                                <b>CCDs & CMOS:</b> Charge-Coupled Devices. They have high Quantum Efficiency (>90%), meaning they detect almost every photon that hits them (compared to <2% for the human eye).
                                            </li>
                                            <li class="flex items-start"><span class="text-green-500 font-bold mr-2">➤</span> 
                                                <b>Spectrographs:</b> Use prisms, grisms, or diffraction gratings to disperse light onto the detector.
                                            </li>
                                            <li class="flex items-start"><span class="text-green-500 font-bold mr-2">➤</span> 
                                                <b>Bolometers:</b> Used in Infrared/Microwave astronomy (like Planck) to measure the heat generated by incoming radiation.
                                            </li>
                                        </ul>
                                    </div>

                                    <div class="mt-4 pt-4 border-t border-slate-200">
                                        <img src="/images/ccd_detector.jpg" class="w-full rounded shadow-sm border border-gray-300 mb-2" alt="CCD Detector">
                                        <div class="text-right">
                                            <a href="https://noirlab.edu/public/blog/50-years-ccds/" target="_blank" class="text-xs text-blue-600 hover:text-blue-800 hover:underline flex items-center justify-end gap-1">
                                                Source: NOIRLab (50 Years of CCDs) <i class="material-icons text-[10px]">open_in_new</i>
                                            </a>
                                        </div>
                                    </div>
                                </div>

                            </div>
                            """)

                        with ui.tab_panel(t_env):
                            html_info_box(r"""
                                <h3 class="text-2xl font-bold text-slate-800 mb-4">Observational Environments</h3>
                                <p class="italic text-gray-600 mb-6">Why do we put telescopes in weird places? To escape the Earth's atmosphere.</p>
                                
                                <div class="space-y-4">
                                    <div class="bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition">
                                        <h4 class="text-lg font-bold text-purple-800">🏔️ Ground-based</h4>
                                        <p class="text-gray-700 text-sm">Located on high mountains (Atacama Desert, Mauna Kea, Canary Islands) to minimize atmospheric turbulence and water vapor absorption.</p>
                                        <p class="text-xs text-gray-500 mt-1">Best for: Optical, Radio, Near-Infrared.</p>
                                    </div>

                                    <div class="bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition">
                                        <h4 class="text-lg font-bold text-blue-800">✈️ Airborne</h4>
                                        <p class="text-gray-700 text-sm">Telescopes inside stratospheric aircraft (e.g., SOFIA). They fly above 99% of the water vapor.</p>
                                        <p class="text-xs text-gray-500 mt-1">Best for: Far-Infrared.</p>
                                    </div>

                                    <div class="bg-white border rounded-lg p-4 shadow-sm hover:shadow-md transition">
                                        <h4 class="text-lg font-bold text-black">🚀 Space-based</h4>
                                        <p class="text-gray-700 text-sm">The ultimate view. No atmosphere means perfect seeing and access to all wavelengths.</p>
                                        <p class="text-xs text-gray-500 mt-1">Best for: UV, X-ray, Gamma-ray, and ultra-deep Optical/IR (Hubble, JWST).</p>
                                    </div>
                                </div>
                                """)

                        with ui.tab_panel(t_biblio):
                            html_info_box(r"""
                <div>
                    <h4 class="text-slate-800 font-bold text-lg mb-2">📚 Scientific Bibliography</h4>
                    <p class="text-gray-600 mb-4 text-sm">The simulations and concepts in this module are based on standard astrophysical literature.</p>
                    
                    <div class="space-y-6">
                        
                        <div>
                            <h5 class="text-indigo-900 font-bold border-b border-indigo-100 pb-1 mb-2 text-sm uppercase">1. CCD Physics & Signal-to-Noise Ratio</h5>
                            <ul class="list-disc ml-6 text-gray-800 space-y-2 text-sm">
                                <li>
                                    <b>Howell, S. B. (2006).</b> <i>Handbook of CCD Astronomy</i>. Cambridge University Press.<br>
                                    <span class="text-gray-500 italic">Source for the "CCD Equation" and the photon-counting logic used in the simulation (\( SNR = \frac{S_{star} \cdot t}{\sqrt{ (S_{star} + B_{sky} + D_{dark}) \cdot t + R_{read}^2 }} \)).</span>
                                </li>
                                <li>
                                    <b>Bevington, P. R., & Robinson, D. K. (2003).</b> <i>Data Reduction and Error Analysis for the Physical Sciences</i>. McGraw-Hill.<br>
                                    <span class="text-gray-500 italic">Mathematical basis for Poisson Statistics (the square root noise rule).</span>
                                </li>
                            </ul>
                        </div>

                        <div>
                            <h5 class="text-indigo-900 font-bold border-b border-indigo-100 pb-1 mb-2 text-sm uppercase">2. Observational Methods & Optics</h5>
                            <ul class="list-disc ml-6 text-gray-800 space-y-2 text-sm">
                                <li>
                                    <b>Kitchin, C. R. (2013).</b> <i>Astrophysical Techniques</i>. CRC Press.<br>
                                    <span class="text-gray-500 italic">Reference for Spectroscopy, Photometry, and Interferometry definitions.</span>
                                </li>
                                <li>
                                    <b>Carroll, B. W., & Ostlie, D. A. (2017).</b> <i>An Introduction to Modern Astrophysics</i>. Cambridge University Press.<br>
                                    <span class="text-gray-500 italic">Reference for Telescope Light Gathering Power (\(\propto D^2\)).</span>
                                </li>
                            </ul>
                        </div>

                    </div>
                </div>
        """)
                
                scientific_dialog.open()


   #panel for observational methods and instruments
        def instrument_page(container):
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            
            with container:
                ui.label(
    "Discover how modern technology and sky surveys allow us to observe the deep universe and map the structure of the cosmos."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                with ui.dialog() as instruction_d , ui.card().classes("p-4 w-full max-w-[800px]").props('role=dialog aria-label=Instructions'):
                    html_info_box(r"""
    <h2 class="text-2xl font-bold mb-4">Observational methods and instruments instructions</h2>
    <ol class="list-decimal list-inside space-y-2"> 
        <li> Explore the various observational techniques used in astronomy, such as spectroscopy, photometry, and interferometry.</li>
        <li> Learn about the different types of instruments, including telescopes and detectors, and their roles in data collection.</li>
        <li> Understand the environments where observations are made: ground-based, airborne, and space-based.</li>
        <li> Click on the "Observational Techniques & Instruments" button to open detailed information.</li>
      
        <li> Move the sliders to adjust telescope diameter and exposure time in order to reveal the correct galaxy image on the plot.</li> 
        <li> Use the knowledge from the scientific background provided to optimize your settings for the best image quality.</li>
    </ol>
                        """)
                    aria_button("Close", "close popup", on_click=lambda:instruction_d.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
               
                with ui.dialog() as curiosity_d, ui.card().classes("p-6 w-full max-w-[600px]").props('role=dialog aria-label=Curiosity'):
                    html_info_box(r"""
                <h2 class="text-2xl font-bold mb-4 text-purple-800">💡 Did you know?</h2>
                <p class="text-gray-700 leading-relaxed mb-4">
                    Before the invention of the CCD, astronomers had to rely on <b>glass photographic plates</b> to capture images of the night sky. These plates were incredibly inefficient, capturing only about <b>1% to 2%</b> of the photons that hit them! Astronomers had to expose a plate for hours in the freezing cold just to record a faint smudge of a galaxy.
                </p>
                <p class="text-gray-700 leading-relaxed mb-4">
                    In 1969, Willard Boyle and George E. Smith invented the CCD at Bell Labs. Modern astronomical CCDs have a "Quantum Efficiency" of over <b>90%</b>, meaning they record almost every single particle of light they receive. This revolutionary leap completely transformed astrophysics, allowing us to peer deeper into the universe than ever before.
                </p>
                <p class="text-gray-700 leading-relaxed">
                    For this world-changing invention, Boyle and Smith were awarded the <b>Nobel Prize in Physics in 2009</b>!
                </p>
                                    """)
                    aria_button("Close", "close curiosity popup", on_click=curiosity_d.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-4")
                
                with ui.row().classes('w-full justify-center gap-6 mb-4'):
                    aria_button(
                        'Instructions', 
                        'Open Instructions Dialog', 
                        on_click=lambda: [instruction_d.open(), ui.run_javascript("MathJax.typesetPromise()")]
                    ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
                    aria_button(
                        'Scientific Info', 
                        'Open Scientific Information Dialog', 
                        on_click=lambda: [open_scientific_info_dialog(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")]
                    ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
                    aria_button(
                        'Curiosity', 
                        'Open Curiosity Dialog', 
                        on_click=curiosity_d.open
                    ).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                
             
                with ui.row().classes('w-full justify-center items-start gap-8 mt-4'):
                    
                   
                    with ui.column().classes('w-full md:w-1/3 space-y-4 text-left'):
                        ui.label("Instrument Controls").classes('text-2xl font-bold text-white border-b border-gray-500 pb-2 w-full')

                        with ui.column().classes('w-full gap-1'):
                            ui.label("Telescope Diameter (m)").classes('font-bold text-white text-lg')
                            diam_slider = aria_slider(
                                min=0.1, max=10.0, value=0.5, step=0.1, 
                                aria_label="Telescope Diameter"
                            ).props('label-always color="blue" dark').classes('w-full')

                        with ui.column().classes('w-full gap-1'):
                            ui.label("Exposure Time (seconds)").classes('font-bold text-white text-lg')
                            time_slider = aria_slider(
                                min=1, max=300, value=10, step=10, 
                                aria_label="Exposure Time"
                            ).props('label-always color="green" dark').classes('w-full') 

                        with ui.column().classes('w-full gap-1'):
                            ui.label("Sky Background Level").classes('font-bold text-white text-lg')
                            sky_select = ui.select(
                                options={10: 'Dark Mountain Sky (Low Noise)', 100: 'Suburban Sky', 500: 'City Center (High Noise)'}, 
                                value=100
                            ).classes('w-full bg-slate-800 text-white border border-gray-600 rounded shadow-sm')\
                            .props('outlined behavior="menu" dark label-color="white"')

                        with ui.column().classes('w-full gap-2'):
                            ui.label("CCD Visualization Mode").classes('font-bold text-white text-lg')
                            cmap_select = ui.select(
                                options={
                                    'gray': 'Grayscale (Raw CCD Reality)', 
                                    'inferno': 'Inferno Map (Intensity Analysis)', 
                                    'viridis': 'Viridis Map (Science Standard)'
                                }, 
                                value='gray' 
                            ).classes('w-full bg-slate-800 text-white border border-gray-600 rounded shadow-sm').props('outlined behavior="menu" dark label-color="white"')
                    
                 
                    with ui.column().classes('w-full md:w-1/2 flex flex-col items-center gap-1'):
                        visibility_status = ui.label("Image status: Waiting for parameters...").classes(
                            "text-xl font-bold text-green-400 text-center w-full bg-slate-800 p-2 rounded border border-green-500/50 m-0"
                        ).props('aria-live=polite')
                        
                        plot_container = ui.column().classes('w-full items-center justify-start p-0 m-0')
                
                    def update_simulation():
                        plot_container.clear()
                        
                        D = diam_slider.value       
                        t = time_slider.value       
                        bg_level = sky_select.value 
                        current_cmap = cmap_select.value
                   
                        dark_current = 1.0 
                        read_noise = 10.0  
                        
                   
                        size = 300 
                        
                        image_file_path = os.path.join(BASE_DIR, 'images', 'eso9845d.jpg')
                        true_flux = None

                        if os.path.exists(image_file_path):
                            try:
                                img = Image.open(image_file_path).convert('L').resize((size, size))
                                img_array = np.array(img)
                                true_flux = (img_array / 255.0) * 60.0 
                            except Exception as e:
                                print(f"Errore caricamento immagine: {e}")

                        if true_flux is None:
                            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
                            r = np.sqrt(x**2 + y**2)
                            spiral = np.sin(5*r + np.arctan2(y, x))
                            true_flux = 50 * np.exp(-r/0.2) + 10 * np.exp(-r/0.5) * (spiral**2)

                        area_factor = (D / 2.0)**2 
                        
                        S_map = true_flux * area_factor * t
                        B_map = np.ones_like(S_map) * bg_level * t
                        D_map = np.ones_like(S_map) * dark_current * t
                        
                  
                        MAX_DIAMETER_FACTOR = (10.0 / 2.0)**2
                        MAX_EXPOSURE = 300
                        max_possible_signal = np.max(true_flux) * MAX_DIAMETER_FACTOR * MAX_EXPOSURE
                        
                        fixed_vmin = 0
                        fixed_vmax = max_possible_signal

                        poisson_source = S_map + B_map + D_map
                        poisson_source = np.maximum(poisson_source, 0) 
                        
                        image_poisson = np.random.poisson(poisson_source)
                        image_read_noise = np.random.normal(0, read_noise, size=S_map.shape)
                        
                        observed_image = image_poisson + image_read_noise
                        
                        S_peak = np.max(S_map)
                        noise_variance = S_peak + (bg_level * t) + (dark_current * t) + (read_noise**2)
                        snr = S_peak / np.sqrt(noise_variance) if noise_variance > 0 else 0

                        total_signal = int(np.sum(S_map))  
                        total_bg_noise = int(np.sum(B_map + D_map)) 
                        
                        if snr < 5:
                            status_text = "Image status: Dominated by noise. The galaxy is completely invisible."
                        elif snr < 15:
                            status_text = "Image status: Faint detection. A blurry central core is barely visible."
                        elif snr < 30:
                            status_text = "Image status: Good resolution. The core is bright and faint spiral arms are emerging."
                        else:
                            status_text = "Image status: Excellent visibility! The spiral structure and arms are perfectly clear."
                            
                        visibility_status.set_text(status_text)
                        
                        with plot_container:
                            with ui.pyplot(figsize=(8, 6), facecolor='none').classes('w-full max-w-[800px] mx-auto') as plot_element:
                                fig = plot_element.fig
                                fig.patch.set_alpha(0.0)
                                ax = fig.gca()
                                
                                im = ax.imshow(
                                    observed_image, 
                                    cmap=current_cmap, 
                                    origin='lower',
                                    vmin=fixed_vmin,
                                    vmax=fixed_vmax
                                )
                                ax.set_title(r"$\mathbf{SNR}$: " + f"{snr:.1f} | " + r"$\mathbf{Total\ Photons}$: " + f"{int(np.sum(observed_image))}\n" +
                                r"$\mathbf{Signal}$: " + f"{total_signal} e- | " + r"$\mathbf{Noise}$: " + f"{total_bg_noise} e-",
                                fontsize=12, pad=20, color='white', loc='center')
                                ax.axis('off')
                                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        
                                cbar.set_label('Photons Counts (ADU/pixel)', fontsize=12, color='white', labelpad=20)
                                cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=11)
                                cbar.outline.set_edgecolor('white')
                                
                                fig.subplots_adjust(left=0.05, right=0.82, top=0.80, bottom=0.05)

                    diam_slider.on_value_change(update_simulation)
                    time_slider.on_value_change(update_simulation)
                    sky_select.on_value_change(update_simulation)
                    cmap_select.on_value_change(update_simulation)
                    
                    update_simulation()
#panel structures of universe                  
        IMAGES_MACRO = [
            {'file': 'images/spiral_galaxy.jpg', 'title': 'Spiral Galaxy',
            'description': 'A spiral galaxy rich in gas, dust, and billions of stars. Image of Messier 77,Source: [NASA](https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/messier-77/).', 'size': 9.5e17},

            {'file': 'images/elliptical_galaxy.jpg', 'title': 'Elliptical Galaxy',
            'description': 'A smooth, featureless galaxy composed mostly of old stars. Image of Messier 32,Source: [UniverseToday](https://www.universetoday.com/articles/messier-32).', 'size': 3e17},

            {'file': 'images/irregular_galaxy.jpg', 'title': 'Irregular Galaxy',
            'description': 'A chaotic galaxy without a defined structure. Image of NGC 55,Source: [ESO](https://www.eso.org/public/italy/images/eso0914a/).', 'size': 6e17},
            {'file': 'images/lenticular_galaxy.jpg', 'title': 'Lenticular Galaxy',
            'description': 'A galaxy with a central bulge and disk but no spiral arms. Image of NGC 6861,Source: [SciNews](https://www.sci.news/astronomy/science-hubble-space-telescope-lenticular-galaxy-ngc6861-02416.html).', 'size': 5e17},

            {'file': 'images/nebula_emission.jpg', 'title': 'Emission Nebula',
            'description': 'Clouds of ionized gas glowing due to nearby young stars. Image of N44,Source: [Wikipedia-ESA](https://en.wikipedia.org/wiki/N44_%28emission_nebula%29).', 'size': 9e16},

            {'file': 'images/nebula_dark.jpg', 'title': 'Dark Nebula',
            'description': 'Dense, cold clouds of dust blocking background starlight. Image of Bernard 3,Source: [AstroCarballada](https://astro.carballada.com/barnard_3/).', 'size': 3e16},

            {'file': 'images/protoplanetary_disk.jpg', 'title': 'Protoplanetary Disk',
            'description': 'A disk of gas and dust where new planets are forming. Image of HL Tauri,Source: [Wikipedia-ALMA](https://en.wikipedia.org/wiki/Protoplanetary_disk).', 'size': 3e12},

            {'file': 'images/star.jpg', 'title': 'Main Sequence Star',
            'description': 'A stable star fusing hydrogen into helium in its core. Image of the Sun,Source: [BBC](https://www.skyatnightmagazine.com/space-science/main-sequence-stars).', 'size': 1.39e6},

            {'file': 'images/red_giant.jpg', 'title': 'Red Giant',
            'description': 'A dying star that has swollen to enormous size. Image of CW Leonis,Source: [SciNews](https://www.sci.news/astronomy/hubble-cw-leonis-10219.html).', 'size': 1e8},

            {'file': 'images/supernova.jpg', 'title': 'Supernova Explosion',
            'description': 'A catastrophic stellar explosion releasing heavy elements. Image of Cassiopea A,Source: [NASA-JLP-Caltech](https://www.schoolsobservatory.org/discover/projects/supernovae/examples).', 'size': 1e17},

            {'file': 'images/black_hole.jpg', 'title': 'Black Hole',
            'description': 'A region where gravity is so strong that nothing can escape. Image of Messier 87,Source: [Wikipedia](https://en.wikipedia.org/wiki/Black_hole).', 'size': 3e10},

            {'file': 'images/quasar.jpg', 'title': 'Quasar',
            'description': 'An extremely luminous active galactic nucleus powered by a supermassive black hole. Image of J0749 2255,Source: [NASA](https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-quasars/).', 'size': 1e14},

            {'file': 'images/exoplanet.jpg', 'title': 'Exoplanet',
            'description': 'A planet orbiting a star outside our solar system. Image of 2M1207b,Source: [NASA](https://science.nasa.gov/resource/2m1207-b-first-image-of-an-exoplanet/).', 'size': 1e5},

            {'file': 'images/planet.jpg', 'title': 'Planet',
            'description': 'A world orbiting a star, possibly with an atmosphere and moons. Image of Jupiter,Source: [ESA](https://www.esa.int/Science_Exploration/Space_Science/Juice/Hello_Jupiter!_How_to_observe_a_gas_giant).', 'size': 1.4e5},

            {'file': 'images/moon.jpeg', 'title': 'Moon',
            'description': 'A natural satellite orbiting a planet. Image of the Moon,Source: [NASA](https://science.nasa.gov/moon/viewing-guide/).', 'size': 3474},

            {'file': 'images/comet.jpg', 'title': 'Comet',
            'description': 'Ice-rich bodies that form glowing tails when near a star. Image of Halley,Source: [Wikipedia](https://en.wikipedia.org/wiki/Halley%27s_Comet).', 'size': 20},

            {'file': 'images/asteroid.jpg', 'title': 'Asteroid',
            'description': 'A small rocky body left over from the formation of the solar system. Image of Psyche,Source: [NASA](https://science.nasa.gov/solar-system/asteroids/16-psyche/).', 'size': 226},

            {'file': 'images/meteor.jpg', 'title': 'Meteor',
            'description': 'A piece of debris burning in a planet’s atmosphere. Image of a typical meteor,Source: [BBC](https://www.skyatnightmagazine.com/astrophotography/observe-photograph-meteor-train)', 'size': 0.01},

            {'file': 'images/globular_cluster.jpg', 'title': 'Globular Cluster',
            'description': 'A dense spherical group of very old stars. Image of NGC 2298,Source:[NASA](https://science.nasa.gov/image-detail/hubble-ngc2298-acs-wfc3-v3-5fcont-final/).', 'size': 3e15},

            {'file': 'images/galaxy_cluster.jpg', 'title': 'Galaxy Cluster',
            'description': 'A massive structure containing hundreds or thousands of galaxies. Image of Abell 370,Source: [NASA-ESA-Harvard](https://www.cfa.harvard.edu/research/topic/galaxy-clusters).', 'size': 3e22},
            {'file': 'images/galaxy_group.jpg', 'title': 'Galaxy Group',
            'description': 'A small collection of galaxies bound by gravity. Image of the Local Group,Source: [BBC](https://www.skyatnightmagazine.com/space-science/local-group-guide-galaxy-neighbourhood).', 'size': 3e19},

            {'file': 'images/filaments.jpg', 'title': 'Cosmic Filament (Simulation)',
            'description': 'One of the largest structures in the universe, forming the cosmic web. Simulation of a cosmic filament,Source: [SciTech](https://scitechdaily.com/first-direct-image-of-the-cosmic-web-reveals-the-universes-hidden-highways/).', 'size': 3e23},
            {'file': 'images/voids.jpg', 'title': 'Cosmic Void (Simulation)',
            'description': 'Vast, empty regions of space with very few galaxies. Simulation of a cosmic void,Source: [LiveScience](https://www.livescience.com/65928-stare-into-the-fuzzy-dark-void.html).', 'size': 3e24},
            {'file': 'images/cosmic_web.jpg', 'title': 'Cosmic Web (Simulation)',
            'description': 'The large-scale structure of the universe, composed of filaments and voids. Simulation of the cosmic web,Source: [UniverseToday](https://www.universetoday.com/articles/astronomy-without-a-telescope-the-edge-of-greatness).', 'size': 1e25}
        ]
        def markdown_to_html(md_text):
            import re
            # Trasforma [testo](url) in <a href="url" target="_blank">testo</a>
            return re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', md_text)

        def create_interactive_tile(image_file: str, title: str, description: str, size: int):
            safe_id = title.replace(" ", "_")
            with ui.dialog() as popup, ui.card().props('role=dialog '):
                ui.label(title).classes('text-h5').props('role=heading aria-level=3 tabindex=0')
                aria_image(image_file, title).classes('w-96 h-auto')
                info_box(markdown_to_html(description)).props('tabindex=0 aria-label="Description" role=document')
                aria_button('Close','Close image popup', on_click=lambda: popup.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            
            with ui.card().classes('p-2 hover:!bg-gray-200 transition-all cursor-pointer'):
                thumb = aria_image(image_file, title).classes('w-40 h-40 object-cover rounded-md cursor-pointer')
                thumb.props(f'role=button tabindex=0 aria-label="Open details for {title}"').on('click', lambda: popup.open()).on('keydown.enter', lambda: popup.open())
            
                thumb.props('draggable="true"')
                thumb.props(f'id="img-{safe_id}"')
                
                
                thumb.props(f'ondragstart="event.dataTransfer.setData(\'application/json\', JSON.stringify({{file:\'{image_file}\', title:\'{title}\', safe_id:\'{safe_id}\', size:{size}}}))"')
        

        def open_external_and_units_dialog():
            with ui.dialog() as ext_units_dialog, ui.card().classes('p-0 w-full max-w-[1400px] h-[85vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="ext-units-title"'):
                
                with ui.column().classes('w-full h-full bg-white flex flex-col'):
                    
                   
                    with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="ext-units-title"'):
                        ui.label('Resources & Cosmic Scales').classes('text-xl font-bold').props('id="ext-units-title" role=heading aria-level=2 tabindex=0')
                        aria_button('Close', 'close', on_click=ext_units_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                   
                    with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                        t_ext_res = ui.tab('External Resources')
                        t_units = ui.tab('Cosmic Distance Scales')
                        tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))
              
                    with ui.tab_panels(tabs, value=t_ext_res).classes('w-full flex-1 overflow-y-auto bg-gray-50 text-slate-900 p-6'):
                        
                  
                        with ui.tab_panel(t_ext_res):
                            html_info_box(r"""
                    <div style="font-family: sans-serif; padding: 10px;">
                        <h3 style="margin-top: 0; color: #166534; border-bottom: 2px solid #22c55e; padding-bottom: 10px; margin-bottom: 20px;">
                            Explore External Resources
                        </h3>
                        <p style="margin-bottom: 20px; color: #4b5563;">
                            Interactive simulators and visualization tools to deepen your understanding of the cosmos.
                        </p>

                        <div style="display: flex; flex-direction: column; gap: 15px;">

                            <a href="https://www.solarsystemscope.com/" target="_blank" style="text-decoration: none; color: inherit;">
                                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; transition: background-color 0.2s; cursor: pointer;" 
                                    onmouseover="this.style.backgroundColor='#f0fdf4'" onmouseout="this.style.backgroundColor='transparent'">
                                    <div style="font-weight: bold; color: #15803d; font-size: 1.1em;">Solar System Scope ↗</div>
                                    <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">
                                        Interactive 3D model of the Solar System and night sky.
                                    </div>
                                </div>
                            </a>

                            <a href="https://scaleofuniverse.com/en" target="_blank" style="text-decoration: none; color: inherit;">
                                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; transition: background-color 0.2s; cursor: pointer;" 
                                    onmouseover="this.style.backgroundColor='#f0fdf4'" onmouseout="this.style.backgroundColor='transparent'">
                                    <div style="font-weight: bold; color: #15803d; font-size: 1.1em;">Scale of the Universe ↗</div>
                                    <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">
                                        Zoom from the Planck length to the edge of the observable Universe.
                                    </div>
                                </div>
                            </a>

                            <a href="https://theskylive.com/3dsolarsystem" target="_blank" style="text-decoration: none; color: inherit;">
                                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; transition: background-color 0.2s; cursor: pointer;" 
                                    onmouseover="this.style.backgroundColor='#f0fdf4'" onmouseout="this.style.backgroundColor='transparent'">
                                    <div style="font-weight: bold; color: #15803d; font-size: 1.1em;">The Sky Live 3D ↗</div>
                                    <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">
                                        Real-time positions of planets and comets in the solar system.
                                    </div>
                                </div>
                            </a>

                            <a href="https://stellarium-web.org/" target="_blank" style="text-decoration: none; color: inherit;">
                                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; transition: background-color 0.2s; cursor: pointer;" 
                                    onmouseover="this.style.backgroundColor='#f0fdf4'" onmouseout="this.style.backgroundColor='transparent'">
                                    <div style="font-weight: bold; color: #15803d; font-size: 1.1em;">Stellarium Web ↗</div>
                                    <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">
                                        Realistic planetarium to see stars and constellations from any location.
                                    </div>
                                </div>
                            </a>

                            <a href="https://science.nasa.gov/eyes/" target="_blank" style="text-decoration: none; color: inherit;">
                                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; transition: background-color 0.2s; cursor: pointer;" 
                                    onmouseover="this.style.backgroundColor='#f0fdf4'" onmouseout="this.style.backgroundColor='transparent'">
                                    <div style="font-weight: bold; color: #15803d; font-size: 1.1em;">NASA Eyes ↗</div>
                                    <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">
                                        Official NASA visualization apps for Earth, Solar System, and Exoplanets.
                                    </div>
                                </div>
                            </a>

                        </div>
                    </div>
                    """).props('tabindex=0 role=document aria-label="List of external astronomy resources"')

                      
                        with ui.tab_panel(t_units):
                            html_info_box(r"""
        <h3>Cosmic Distance Scales: km, AU, and pc</h3>
        <p>In cosmology, distances range from the relatively "small" scale of planets to the vast expanse between galaxies. To handle these massive numbers without writing endless zeros, astronomers use three primary units of measurement.</p>

        <h4>1. The Planetary Scale: Kilometers (km)</h4>
        <p>Used for measuring diameters of planets or distances of moons. However, on a solar system scale, kilometers become unwieldy.</p>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7; opacity: 0.3;">

        <h4>2. The Solar System Scale: Astronomical Unit (AU)</h4>
        <p>The <b>Astronomical Unit</b> is defined as the average distance between the Earth and the Sun. It is the standard "ruler" for measuring distances within our Solar System.</p>

        <ul>
            <li><b>Definition:</b> <span class="math">\( 1 \, \mathrm{AU} = 149,597,871 \, \mathrm{km} \)</span></li>
            <li><b>Approximation:</b> <span class="math">\( \approx 1.5 \times 10^8 \, \mathrm{km} \)</span></li>
            <li><b>Usage:</b> Jupiter is about <span class="math">\( 5.2 \, \mathrm{AU} \)</span> from the Sun; the Voyager 1 probe is over <span class="math">\( 160 \, \mathrm{AU} \)</span> away.</li>
        </ul>
        <h4>3. The Interstellar Scale: Light-year (ly)</h4>
        <p>The <b>Light-year</b> is the distance that light travels in a vacuum in one Julian year (365.25 days). Despite the name, it is a unit of distance, not time.</p>

        <ul>
            <li><b>Definition:</b> \( 1 \, \mathrm{ly} = c \times 1 \text{ year} \)</li>
            <li><b>Conversion to km:</b> \( \approx 9.461 \times 10^{12} \, \mathrm{km} \)</li>
            <li><b>Conversion to AU:</b> \( \approx 63,241 \, \mathrm{AU} \)</li>
            <li><b>Conversion to pc:</b> \( \approx 0.3066 \, \mathrm{pc} \)</li>
        </ul>
        <hr style="margin: 15px 0; border-top: 1px solid #0284c7; opacity: 0.3;">

        <h4>3. The Galactic Scale: Parsec (pc)</h4>
        <p>The <b>Parsec</b> (<i>parallax second</i>) is the fundamental unit for interstellar distances. It is defined geometrically using the parallax effect: the distance at which <span class="math">\( 1 \, \mathrm{AU} \)</span> subtends an angle of one arcsecond (<span class="math">\( 1'' \)</span>).</p>

        <ul>
            <li><b>Definition:</b> <span class="math">\( 1 \, \mathrm{pc} \approx 206,265 \, \mathrm{AU} \)</span></li>
            <li><b>Conversion to km:</b> <span class="math">\( \approx 3.086 \times 10^{13} \, \mathrm{km} \)</span></li>
            <li><b>Light Years:</b> <span class="math">\( 1 \, \mathrm{pc} \approx 3.26 \, \mathrm{light\text{-}years} \)</span></li>
            <li><b>Usage:</b> Proxima Centauri (nearest star) is <span class="math">\( 1.3 \, \mathrm{pc} \)</span> away. The center of the Milky Way is <span class="math">\( \approx 8,000 \, \mathrm{pc} \)</span> (<span class="math">\( 8 \, \mathrm{kpc} \)</span>) away.</li>
        </ul>

        <div style="margin-top: 20px; padding: 10px; background-color: rgba(255,255,255,0.6); border-radius: 6px; border: 1px dashed #0284c7;">
            <p style="text-align:center; font-weight:bold; margin:0; color:#0369a1;">
                Summary Conversion Chain:<br>
                <span class="math">\( 1 \, \mathrm{pc} \approx 3.26 \, \mathrm{ly} \approx 206,265 \, \mathrm{AU} \approx 3.1 \times 10^{13} \, \mathrm{km} \)</span>
            </p>
        </div>
    """)
                ext_units_dialog.open()
        def introduction_page(container):
            ui.run_javascript('window.gameRunning = false; window.gameScore = 0;')
            timer_state = {'time': 0, 'running': False, 'started': False}
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            with container:

                ui.label("Embark on a journey through time and space to understand the scales and fundamental structures of the universe.").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                
                with ui.dialog() as units_curiosity_d, ui.card().classes("p-6 w-full max-w-[600px]").props('role=dialog aria-label=Curiosity'):
                    html_info_box(r"""
                <h2 class="text-2xl font-bold mb-4 text-purple-800">💡 Did you know?</h2>
                <p class="text-gray-700 leading-relaxed mb-4">
                    The concept of measuring distances using parallax, which forms the basis for the parsec, was first successfully used by Friedrich Bessel in 1838 to measure the distance to the star 61 Cygni. This was a monumental achievement, proving that stars are incredibly far away and cementing our modern understanding of the universe's vast scale.
                </p>
                                    """)
                    aria_button("Close", "close curiosity popup", on_click=units_curiosity_d.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-4")
                

               

             
   
                  
                def update_math():
                    ui.run_javascript(r'''
        if (window.MathJax) {
            // Se MathJax è pronto
            if (window.MathJax.typesetPromise) {
                // Resetta lo stato interno (evita errori su elementi rimossi)
                try { MathJax.typesetClear(); } catch(e) {}

                // Ricalcola tutto
                MathJax.typesetPromise().then(() => {
                    console.log("MathJax Updated.");
                }).catch((err) => {
                    console.log("MathJax Error: " + err.message);
                });
            }
        }
    ''')
                ranges_data = [
            (r'$\mathbf{0 \;\text{--}\; 10^3 \, km}$', 
             r'$\mathit{\approx 0 \, AU \;\mid\; \approx 0 \, pc}$', 
             0, 1e3, "0 - 1000 km"),

            (r'$\mathbf{10^3 \;\text{--}\; 10^6 \, km}$', 
             r'$\mathit{10^{-6} \;\text{--}\; 10^{-3} \, AU \;\mid\; 10^{-11} \;\text{--}\; 10^{-8} \, pc}$', 
             1e3, 1e6, "10^3 - 10^6 km"),

            (r'$\mathbf{10^6 \;\text{--}\; 10^9 \, km}$', 
             r'$\mathit{10^{-3} \;\text{--}\; 1 \, AU \;\mid\; 10^{-8} \;\text{--}\; 10^{-5} \, pc}$', 
             1e6, 1e9, "10^6 - 10^9 km"),

            (r'$\mathbf{10^9 \;\text{--}\; 10^{18} \, km}$', 
             r'$\mathit{1 \;\text{--}\; 10^9 \, AU \;\mid\; 10^{-5} \;\text{--}\; 10^4 \, pc}$', 
             1e9, 1e18, "10^9 - 10^18 km"),

            (r'$\mathbf{> 10^{18} \, km}$', 
             r'$\mathit{> 10^9 \, AU \;\mid\; > 10^4 \, pc}$', 
             1e18, 1e26, "> 10^18 km"),
        ]

               
                @ui.refreshable
                def render_drop_column():
                    with ui.column().classes("flex-[1] gap-2 w-full"): 
                        for title_html, sub_html, min_s, max_s, simple_name in ranges_data:
                            with ui.card().classes('p-1.5 w-full h-auto min-h-[130px] relative overflow-hidden cursor-pointer flex flex-col'):
                                
                              
                                ui.html(title_html).classes('font-bold text-lg text-white-500 leading-none text-center').props('role=heading aria-level=3')
                                ui.html(sub_html).classes('text-md text-white-500 mt-1 leading-none text-center')
                                
                                drop_zone = ui.column().classes(
                    "gap-1 mt-1 flex-wrap items-center justify-center w-full flex-grow h-full min-h-[60px] rounded bg-gray-50/50 border-2 border-dashed border-gray-300"
                )
                                
                                js_logic = (
                                    f'ondragover="event.preventDefault();" '
                                    f'ondrop="event.preventDefault(); '
                                    f'if(!window.gameRunning) {{ '
                                    f'  if(window.Quasar && window.Quasar.Notify) {{ window.Quasar.Notify.create({{ message: \'Premi START prima di giocare!\', color: \'warning\', position: \'top\' }}); }} '
                                    f'  return; '
                                    f'}} '
                                    f'const data=JSON.parse(event.dataTransfer.getData(\'application/json\')); '
                                    f'const size=parseFloat(data.size); '
                                    f'if(size < {min_s} || size > {max_s}) {{ '
                                    f'  window.gameScore -= 0.25; document.getElementById(\'score_display\').innerText = \'Score: \' + window.gameScore; '
                                    f'  if(window.Quasar && window.Quasar.Notify) {{ '
                                    f'    window.Quasar.Notify.create({{ message: data.title + \' does not belong here!\', color: \'negative\', position: \'top\', icon: \'warning\' }}); '
                                    f'  }} return; '
                                    f'}} '
                                    f'window.gameScore += 1; document.getElementById(\'score_display\').innerText = \'Score: \' + window.gameScore; '
                                    f'const scale=Math.max(55, Math.min(110, size/{max_s}*100)); ' 
                                    f'const col=document.createElement(\'div\'); '
                                    f'col.style.display=\'flex\'; col.style.flexDirection=\'column\'; col.style.alignItems=\'center\'; col.style.margin=\'3px\'; '
                                    f'const img=document.createElement(\'img\'); '
                                    f'img.src=data.file; img.width=scale; img.height=scale; img.style.objectFit=\'cover\'; img.className=\'rounded shadow-md\'; '
                                    f'const lbl=document.createElement(\'div\'); lbl.innerText=data.title; lbl.style.fontSize=\'10px\'; lbl.style.fontWeight=\'bold\'; lbl.style.textAlign=\'center\'; '
                                    f'col.appendChild(img); col.appendChild(lbl); '
                                    f'this.appendChild(col); '
                                    f'const sourceImg = document.getElementById(\'img-\' + data.safe_id); '
                                    f'if(sourceImg) {{ sourceImg.style.opacity = \'0.3\'; sourceImg.style.filter = \'grayscale(1) blur(1px)\'; sourceImg.style.pointerEvents = \'none\'; }}"'
                                )
                                # ----------------------------------------------------
                                drop_zone.props(js_logic)
                    
                    ui.timer(0.1, update_math, once=True)
                    #update_math()
             
               

            
                with ui.dialog() as intro, ui.card().classes('p-4 w-full text-lg max-w-[1200px] overflow-x-auto').props('role=dialog aria-label="Game Instructions"'):
                    html_info_box(r"""
<div style="font-family: sans-serif; line-height: 1.6;">
    
    <h3 style="margin-top: 0; color: #059669; border-bottom: 2px solid #10b981; padding-bottom: 10px; margin-bottom: 15px;">1. Universe Structure Challenge</h3>
    <p>Put your cosmic perspective to the test by organizing structures from the smallest to the largest scales!</p>
    <ul style="margin-bottom: 25px;">
        <li>Press <b>Start Mission</b> to activate the timer and start the timer.</li>
        <li><b>How to Play:</b> Drag the images of various cosmic objects (planets, stars, galaxies, etc.) and drop them into the correct <b>Distance Scale</b> category.</li>
        <li><b>Scoring:</b> You earn <b>1 point</b> for every correctly categorized structure. An incorrect placement results in a <b>-0.25 point</b> penalty.</li>
        <li><b>Reset:</b> Use the <b>Reset</b> button if you want to clear the boxes and shuffle the structures to try again.</li>
        <li>Organize all elements correctly to complete the mission and stop the clock!</li>
    </ul>

    <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Distance Scales and Universe Elements</h3>
    <p>Explore the vast scales of the universe and its fundamental components through interactive tools:</p>
    <ul>
        <li><strong>Interactive Simulators:</strong> Use the provided simulations to visualize the massive differences in size between atoms, planets, and galaxies.</li>
        <li><strong>Distance Units:</strong> Use the <b>Units</b> button to review cosmic distance measurements like kilometers (km), astronomical units (AU), parsecs (pc), and light-years (ly).</li>
    </ul>
    <p>Move the images of various cosmic structures into their corresponding distance scale categories by dragging and dropping them into the boxes on the main panel.</p>
    
    <div style="border-left: 4px solid #3b82f6; padding-left: 15px; margin-top: 20px; background-color: #f0f9ff; padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
        <p style="margin: 0; color: #1e40af; font-size: 0.95em;">
            <strong>Pro Tip:</strong> If you are unsure about an object's scale, check the <b>Distance Simulator</b> first to see where it fits in the scheme of the cosmos!
        </p>
    </div>
</div>
""").props('tabindex=0 role=document aria-label="Distance scales and game instructions"')
                    aria_button("Close", "close the box",on_click=lambda:intro.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.row().classes('w-full justify-center gap-6 items-center'):
                
                    
                   
                    aria_button('Instructions', 'Open Instructions', 
                        on_click=lambda: (intro.open(), update_math())
                    ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")

                 
                    aria_button(
            'Scientific Info', 
            'Open Resources and Cosmic Scales Dialog', 
            on_click=lambda: [open_external_and_units_dialog(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")

                    aria_button(
            'Curiosity', 
            'Open Curiosity Dialog', 
            on_click=units_curiosity_d.open
        ).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                   
                    aria_button(
                        '♿ Accessibility Mode', 
                        'Jump to accessibility exercise',
                        on_click=lambda: (
                            acc_structures.open(), 
                            ui.run_javascript("setTimeout(() => document.getElementById('acc_structures').scrollIntoView({behavior: 'smooth', block: 'start'}), 100);")
                        )
                    ).classes("!bg-red-600 hover:!bg-red-700 text-white font-bold py-2 px-4 rounded shadow-md")
                  
                   
                 
                

             
                with ui.column().classes('w-full p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-600 flex flex-col items-center mt-8 mb-12'):
                    ui.label("🎮 AstroScales: Cosmic Structures").classes("text-3xl md:text-4xl font-bold text-yellow-400 mb-4 text-center drop-shadow-md uppercase tracking-wider").props('role=heading aria-level=2 tabindex=0')
                    
                    with ui.row().classes('w-full max-w-[800px] justify-between bg-slate-800 p-4 rounded-lg border border-slate-500 mb-6 shadow-inner items-center'):
                        time_label = ui.label('Time: 0s').classes('text-xl text-white font-bold w-24')
                        ui.html('<div id="score_display" class="text-xl font-bold w-32 text-center text-green-400">Score: 0</div>')

                        def update_timer():
                            if timer_state['running']:
                                timer_state['time'] += 1
                                time_label.set_text(f"Time: {timer_state['time']}s")
                                
                        ui.timer(1.0, update_timer)

                        def start_game():
                            timer_state['time'] = 0
                            timer_state['running'] = True
                            timer_state['started'] = True
                            time_label.set_text("Time: 0s")
                            ui.run_javascript("window.gameRunning = true; window.gameScore = 0; document.getElementById('score_display').innerText = 'Score: 0';")
                            render_astroscales_board.refresh()

                        def stop_game():
                            timer_state['running'] = False
                            ui.run_javascript("window.gameRunning = false;")
                            
                        with ui.row().classes('gap-4'):
                            aria_button("Start", "start game", on_click=start_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105 animate-pulse")
                            aria_button("Stop", "stop game", on_click=stop_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                            aria_button("Reset", "reset", on_click=lambda: (
                                timer_state.update({'time': 0, 'started': False}),
                                stop_game(),
                                time_label.set_text('Time: 0s'),
                                ui.run_javascript("window.gameScore = 0; document.getElementById('score_display').innerText = 'Score: 0';"),
                                ui.run_javascript('document.querySelectorAll("[id^=\'img-\']").forEach(img => {img.style.opacity = "1"; img.style.filter = "none"; img.style.pointerEvents = "auto"})'),
                                render_astroscales_board.refresh(),
                                render_drop_column.refresh(),
                                ui.timer(0.2, update_math, once=True)
                            )).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")

                    @ui.refreshable
                    def render_astroscales_board():
                        if not timer_state.get('started', False):
                            ui.label("Press Start to reveal the cosmic structures!").classes("text-slate-400 italic text-center text-2xl w-full py-12")
                            return

                        with ui.row().classes("w-full items-start justify-center gap-6"):
                            with ui.column().classes("flex-[3]"):
                                with ui.grid(columns=6).classes('gap-4 p-4'):
                                    for obj in IMAGES_MACRO:
                                        create_interactive_tile(obj['file'], obj['title'], obj['description'], obj['size'])
                            render_drop_column()
                    
                    render_astroscales_board()
                    
                 

                    acc_structures = ui.expansion('Accessibility Mode: Structures Classification', icon='accessible').classes('w-full mt-8 bg-slate-800 text-white rounded-lg').props('id="acc_structures" aria-label="Keyboard accessible alternative for structures classification"')
                    with acc_structures:
                        ui.label('Classify the objects by selecting their correct cosmic scale based on their description.').classes('mb-4 text-lg font-bold text-blue-200').props('tabindex=0')
                        
                        
                       
                        scale_options = [r[4] for r in ranges_data]
                        
                 
                        user_choices = {}
                        
                        with ui.column().classes('w-full gap-4 p-2'):
                           
                            for obj in IMAGES_MACRO:
                                with ui.row().classes('w-full items-center justify-between border-b border-slate-700 pb-2 flex-wrap'):
                                    
                                  
                                    with ui.column().classes('flex-[2] min-w-[300px]'):
                                        ui.label(obj['title']).classes('text-xl font-bold text-blue-300').props('tabindex=0')
                                       
                                        ui.markdown(obj['description']).classes('text-sm text-gray-300').props('tabindex=0')
                                    
                                    
                                    user_choices[obj['title']] = ui.select(
                                        options=scale_options, 
                                        label='Select Scale'
                                    ).classes('w-48 flex-[1]').props(f'dark aria-label="Select scale for {obj["title"]}"')
                        
                       
                        def check_structures_accessible():
                            score = 0
                            errors = 0
                            
                          
                            scale_limits = {r[4]: (r[2], r[3]) for r in ranges_data}
                            
                            for obj in IMAGES_MACRO:
                                selected_scale = user_choices[obj['title']].value
                                
                                if not selected_scale:
                                    continue 
                                
                                min_s, max_s = scale_limits[selected_scale]
                                
                            
                                if min_s <= obj['size'] <= max_s:
                                    score += 1
                                else:
                                    errors += 1
                            
                            tot_items = len(IMAGES_MACRO)
                            
                           
                            if score == tot_items:
                                accessible_notify(f"✅ Mission Complete! Perfect score: {score}/{tot_items}", type_='success')
                            elif score > 0 and errors == 0:
                                accessible_notify(f"Good start! Correct: {score}. Keep classifying the rest.", type_='info')
                            elif errors > 0:
                                accessible_notify(f"⚠️ Some errors found (Correct: {score}, Wrong: {errors}). Check the sizes and try again!", type_='warning')
                            else:
                                accessible_notify("Please select a scale for the objects before checking.", type_='warning')

                       
                        aria_button(
                            'Verify Classification', 
                            'Submit accessible structure classification', 
                            on_click=check_structures_accessible
                        ).classes('mt-4 !bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded')
                  


                




                        
                        
        
                

        
    #panel particles        

        leptons_nodes = [
    {"id":"leptons_root","label":"LEPTONS","desc":"Fermions (half-integer spin) - Leptons: interact via weak and electromagnetic forces, with integer charges","color":"#7b61ff","shape":"diamond"},
    {"id":"e","label":"Electron (e⁻)","mass":"0.511 MeV","charge":"-1","spin":"1/2","desc":"Lightest charged lepton; stable; forms atoms","color":"#c8b8ff","shape":"circle"},
    {"id":"mu","label":"Muon (μ⁻)","mass":"105.7 MeV","charge":"-1","spin":"1/2","desc":"Heavier; decays to electron + neutrinos","color":"#bda0ff","shape":"circle"},
    {"id":"tau","label":"Tau (τ⁻)","mass":"1776.86 MeV","charge":"-1","spin":"1/2","desc":"Heaviest charged lepton; short-lived","color":"#a583ff","shape":"circle"},
    {"id":"nu_e","label":"Electron neutrino (νₑ)","mass":"<1 eV","charge":"0","spin":"1/2","desc":"Neutral lepton; participates in weak interactions","color":"#e6e0ff","shape":"circle"},
    {"id":"nu_mu","label":"Muon neutrino (ν_μ)","mass":"<0.8 eV","charge":"0","spin":"1/2","desc":"Associated with muon decays","color":"#efeaff","shape":"circle"},
    {"id":"nu_tau","label":"Tau neutrino (ν_τ)","mass":"<0.8 eV","charge":"0","spin":"1/2","desc":"Associated with tau decays","color":"#f6f2ff","shape":"circle"},
    ]

        leptons_edges = [
            ("leptons_root","e"),
            ("leptons_root","mu"),
            ("leptons_root","tau"),
            ("leptons_root","nu_e"),
            ("leptons_root","nu_mu"),
            ("leptons_root","nu_tau"),
        
        ]

        # ---------- 2) Quarks ----------
        quarks_nodes = [
            {"id":"quarks_root","label":"QUARKS","desc":"Fermions (half-integer spin) - Quarks: feel strong force (QCD), have fractioned charge","color":"#ff6b6b","shape":"diamond"},
            {"id":"u","label":"Up (u)","mass":"2.2 MeV","charge":"+2/3","spin":"1/2","desc":"Light up quark, component of protons/neutrons","color":"#ffb4b4","shape":"circle"},
            {"id":"d","label":"Down (d)","mass":"4.7 MeV","charge":"-1/3","spin":"1/2","desc":"Light down quark, component of protons/neutrons","color":"#ffbcbc","shape":"circle"},
            {"id":"s","label":"Strange (s)","mass":"95 MeV","charge":"-1/3","spin":"1/2","desc":"Strange quark, appears in kaons and hyperons","color":"#ff9a9a","shape":"circle"},
            {"id":"c","label":"Charm (c)","mass":"1.28 GeV","charge":"+2/3","spin":"1/2","desc":"Charm quark, heavier generation","color":"#ff8b8b","shape":"circle"},
            {"id":"b","label":"Bottom (b)","mass":"4.18 GeV","charge":"-1/3","spin":"1/2","desc":"Bottom quark","color":"#ff7c7c","shape":"circle"},
            {"id":"t","label":"Top (t)","mass":"173 GeV","charge":"+2/3","spin":"1/2","desc":"Top quark; very heavy; decays quickly","color":"#ff5f5f","shape":"circle"},
        ]

        quarks_edges = [
            ("quarks_root","u"),
            ("quarks_root","d"),
            ("quarks_root","s"),
            ("quarks_root","c"),
            ("quarks_root","b"),
            ("quarks_root","t"),
        
        ]

        # ---------- 3) Bosons ----------
        bosons_nodes = [
            {"id":"bosons_root","label":"BOSONS","desc":"Force carriers and interactions medium with integer spin","color":"#ffd54f","shape":"diamond"},
            {"id":"photon","label":"Photon (γ)","mass":"0","charge":"0","spin":"1","desc":"Electromagnetic force carrier; mediates light and EM interactions","color":"#fff1c2","shape":"circle"},
            {"id":"gluon","label":"Gluon (g)","mass":"0","charge":"0","spin":"1","desc":"Carrier of the strong force (QCD)","color":"#ffe59a","shape":"circle"},
            {"id":"W","label":"W±","mass":"80.4 GeV","charge":"±1","spin":"1","desc":"Charged weak boson; mediates weak charged current","color":"#ffd27a","shape":"circle"},
            {"id":"Z","label":"Z0","mass":"91.2 GeV","charge":"0","spin":"1","desc":"Neutral weak boson; mediates weak neutral current","color":"#ffc94f","shape":"circle"},
            {"id":"H","label":"Higgs (H)","mass":"125 GeV","charge":"0","spin":"0","desc":"Gives mass to particles via Higgs mechanism","color":"#ffe082","shape":"circle"},
        ]

        bosons_edges = [
            ("bosons_root","photon"),
            ("bosons_root","gluon"),
            ("bosons_root","W"),
            ("bosons_root","Z"),
            ("bosons_root","H"),
        ]

        # ---------- 4) Hadrons (Baryons & Mesons) ----------
        hadrons_nodes = [
            {"id":"hadrons_root","label":"HADRONS","desc":"Composite particles made of quarks (baryons, mesons)","color":"#4da6ff","shape":"diamond"},
            {"id":"p","label":"Proton (p)","mass":"938.3 MeV","charge":"+1","spin":"1/2","desc":"Baryon with 3 quarks:uud (two up, one down)","color":"#cfe9ff","shape":"circle"},
            {"id":"n","label":"Neutron (n)","mass":"939.6 MeV","charge":"0","spin":"1/2","desc":"Baryon with 3 quarks:udd (one up, two down)","color":"#dff2ff","shape":"circle"},
            {"id":"pi","label":"Pion (π)","mass":"139.6 MeV","charge":"±/0","spin":"0","desc":"Lightest Meson (quark+anti-quark); π⁺, π⁰, π⁻","color":"#bfe0ff","shape":"circle"},
            {"id":"K","label":"Kaon (K)","mass":"493.7 MeV","charge":"±/0","spin":"0","desc":"Meson (quark+anti-quark),contains a strange quark","color":"#aeddff","shape":"circle"},
        
        ]

        hadrons_edges = [
            ("hadrons_root","p"),
            ("hadrons_root","n"),
            ("hadrons_root","pi"),
            ("hadrons_root","K"),
        
        ]

    
        processes_nodes = [
        # Processes
        {"id":"bb","label":"Big Bang","shape":"square","color":"#dfffdc",
        "desc":"The initial singularity from which the universe expanded."},
        {"id":"bbn","label":"BBN","shape":"square","color":"#d6ffce",
        "desc":"Primordial nucleosynthesis forming Deuterium, Helium-4, and Lithium-7."},
        {"id":"rec","label":"Recombination","shape":"square","color":"#caffc4",
        "desc":"Epoch when electrons combined with protons to form neutral hydrogen."},
        {"id":"reion","label":"Reionization","shape":"square","color":"#bdffc0",
        "desc":"Period when the first stars and AGN reionized the neutral hydrogen."},

        # Particles


        {"id":"p_proc_bbn","label":"Proton (p)","shape":"circle","color":"#cfe9ff",
        "mass":"1.007 u","charge":"+1 e","spin":"1/2","desc":"Stable baryon, constituent of atomic nuclei."},
        {"id":"p_proc_rec","label":"Proton (p)","shape":"circle","color":"#cfe9ff",
        "mass":"1.007 u","charge":"+1 e","spin":"1/2","desc":"Stable baryon, constituent of atomic nuclei."},
        {"id":"n_proc","label":"Neutron (n)","shape":"circle","color":"#dff2ff",
        "mass":"1.009 u","charge":"0 e","spin":"1/2","desc":"Neutral baryon, constituent of atomic nuclei."},
        {"id":"e_proc","label":"Electron (e⁻)","shape":"circle","color":"#c8b8ff",
        "mass":"5.49e-4 u","charge":"-1 e","spin":"1/2","desc":"Light stable lepton, orbits atomic nuclei."},
        {"id":"star_form","label":"Stellar/AGN radiation","shape":"circle","color":"#ffe9b8",
        "desc":"Photons emitted by stars and active galactic nuclei."},
        {"id":"photon","label":"Photon (γ)","shape":"circle","color":"#fff1c2","mass":"0","charge":"0","spin":"1","desc":"Electromagnetic force carrier; mediates light and EM interactions"},

        # Nuclei / atoms
        {"id":"D","label":"Deuterium (D)","shape":"hexagon","color":"#e8ffe8",
        "mass":"2.014 u","charge":"+1 e","spin":"1","desc":"Heavy isotope of hydrogen formed during BBN."},
        {"id":"He4","label":"Helium-4 (⁴He)","shape":"hexagon","color":"#d8ffd8",
        "mass":"4.0026 u","charge":"+2 e","spin":"0","desc":"Stable helium isotope from BBN."},
        {"id":"Li7","label":"Lithium-7 (⁷Li)","shape":"hexagon","color":"#c8ffc8",
        "mass":"7.016 u","charge":"+3 e","spin":"3/2","desc":"Rare lithium isotope formed in BBN."},
        {"id":"H_atom_O","label":"Neutral Hydrogen (H)","shape":"hexagon","color":"#f0fff0",
        "mass":"1.008 u","charge":"0 e","spin":"1/2","desc":"Neutral hydrogen formed after recombination."},
        {"id":"H_atom_I","label":"Neutral Hydrogen (H)","shape":"circle","color":"#f0fff0",
        "mass":"1.008 u","charge":"0 e","spin":"1/2","desc":"Neutral hydrogen, input for reionization."},
        {"id":"H_ion","label":"Ionized Hydrogen (H)","shape":"hexagon","color":"#f0fff0",
        "mass":"1.008 u","charge":"+1 e","spin":"1/2","desc":"Ionized hydrogen, output after reionization."},
    ]



        processes_edges = [
        # Big Bang → BBN
        ("bb","bbn", {"interaction":""}),

        # BBN inputs → output nuclei
        ("p_proc_bbn","bbn", {"interaction":"p+n→D"}),
        ("n_proc","bbn", {"interaction":"p+n→D"}),
        ("bbn","D", {"interaction":""}),
        ("bbn","He4", {"interaction":""}),
        ("bbn","Li7", {"interaction":""}),

        # Recombination
        ("bbn","rec", {"interaction":""}),
        ("p_proc_rec","rec", {"interaction":"p+e⁻→H"}),
        ("e_proc","rec", {"interaction":"p+e⁻→H"}),
        ("rec","H_atom_O", {"interaction":""}),

        # Reionization
        ("rec","reion", {"interaction":""}),
        ("H_atom_I","reion", {"interaction":""}),
        ("star_form","reion", {"interaction":""}),
        ("photon","reion", {"interaction":""}),
        ("reion","H_ion", {"interaction":""}),
    ]


        def create_mass_conversion_dialog():
        

            with ui.dialog() as mass_conversion_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto').props('role=dialog aria-label="Mass to Energy Conversion Information"'):
                html_info_box(r"""
        <h3>Mass-Energy Conversion: kg to eV</h3>
        <p>In particle physics, mass (<span class="math">\(m\)</span>) is often expressed as equivalent energy (<span class="math">\(E\)</span>) using <b>Einstein's mass-energy equivalence</b> principle:</p>
        
        <p style="text-align:center; font-size: 1.2em; margin: 15px 0;">
            <span class="math">\( E = m c^2 \)</span>
        </p>

        <p>Where <span class="math">\(c\)</span> is the speed of light in a vacuum.</p>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Conversion Steps</h4>
        <ol>
            <li>
                <b>Mass (kg) to Energy (Joule):</b>
                <p style="text-align:center;"><span class="math">\( E \, (\text{J}) = m \, (\text{kg}) \times (c)^2 \)</span></p>
                <p>($c \approx 2.9979 \times 10^8 \text{ m/s}$)</p>
            </li>
            <li style="margin-top: 10px;">
                <b>Energy (Joule) to Electronvolt (eV):</b>
                <p>The energy in Joules must be divided by the elementary charge ($e$) to get Electronvolts.</p>
                <p style="text-align:center;"><span class="math">\( E \, (\text{eV}) = \frac{E \, (\text{J})}{e} \)</span></p>
                <p>($e \approx 1.60217 \times 10^{-19} \text{ J/eV}$)</p>
            </li>
        </ol>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Key Conversion Factors</h4>
        <p>The standard convention uses prefixes to denote magnitude:</p>
        
        <table style="width:100%; border-collapse: collapse; margin-top: 10px; text-align:center;">
            <tr style="background-color: #bae6fd; color: #0c4a6e;">
                <th style="border: 1px solid #0284c7; padding: 8px;">Unit</th>
                <th style="border: 1px solid #0284c7; padding: 8px;">Value in eV</th>
                <th style="border: 1px solid #0284c7; padding: 8px;">Example</th>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">keV (kilo)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">$10^3 \, \text{eV}$</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Binding energies of inner-shell electrons.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">MeV (mega)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">$10^6 \, \text{eV}$</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Electron rest mass ($\approx 0.511 \, \text{MeV}$)</td>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">GeV (giga)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">$10^9 \, \text{eV}$</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Proton rest mass ($\approx 0.938 \, \text{GeV}$)</td>
            </tr>
        </table>
        
        <p style="margin-top: 15px; font-weight: bold;">
            The factor for 1 kg mass is approximately <span class="math">\( \mathbf{5.61 \times 10^{35} \text{ eV}} \)</span>.
        </p>
    """).props('tabindex=0 role=document aria-label="Mass and Energy Conversion information"')
            
                aria_button("Close", 'close',on_click=lambda: mass_conversion_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")


            return aria_button(            "Conversion Units ",'Conversion Unit ',
                on_click=lambda: [
                    mass_conversion_dialog.open(),
                
                    ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")
                ]
            ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded self-center") 
        def plot_particle_graph(title, nodes, edges,is_process_graph=False, height=600, width=None , seed=42):
        
            G = nx.DiGraph()

        
            for n in nodes:
                G.add_node(n['id'], **n)

            for e in edges:
                if len(e) == 3:
                    src, tgt, data = e
                    G.add_edge(src, tgt, **data)
                else:
                    src, tgt = e
                    G.add_edge(src, tgt)

            pos = {}

            if is_process_graph:
                pos = {}
                y_top = 1.0
                row_spacing = 0.2

            
                process_nodes = [n['id'] for n in nodes if n['shape']=='square']

        
                input_nodes = [n['id'] for n in nodes if n['shape'] in ('circle','hexagon') and any(n['id'] in G.predecessors(p) for p in process_nodes)]

        
                output_nodes = [n['id'] for n in nodes if n['shape'] in ('circle','hexagon') and any(n['id'] in G.successors(p) for p in process_nodes)]

            
                for i, proc in enumerate(process_nodes):
                    y_center = y_top - i * row_spacing

                
                    inputs = [n for n in input_nodes if proc in G.successors(n)]
                    n_in = len(inputs)
                    for j, inp in enumerate(inputs):
                        y = y_center if n_in==1 else y_center + 0.05*(n_in-1)/2 - j*0.05
                        pos[(inp, i, 'in')] = (0.0, y)

                
                    pos[(proc, i, 'proc')] = (0.5, y_center)

                
                    outputs = [n for n in output_nodes if n in G.successors(proc)]

                    n_out = len(outputs)
                    for j, out in enumerate(outputs):
                        y = y_center if n_out==1 else y_center + 0.05*(n_out-1)/2 - j*0.05
                        pos[(out, i, 'out')] = (1.0, y)

                
                node_to_pos_keys = {}
                for key in pos.keys():
                    node_id = key[0] if isinstance(key, tuple) else key
                    node_to_pos_keys.setdefault(node_id, []).append(key)

            
                edges_to_plot = [(s, t, d) for s, t, d in G.edges(data=True) if s in node_to_pos_keys and t in node_to_pos_keys]
            
                node_x = []
                node_y = []
                node_text = []
                labels = []
                colors = []
                symbols = []
                sizes = []

                for key, (x, y) in pos.items():
                    node_id = key[0] if isinstance(key, tuple) else key
                    info = G.nodes[node_id]

                    node_x.append(x)
                    node_y.append(y)
                    labels.append(info.get('label', node_id))
                    raw_desc = info.get('desc', '')
           
                    clean_desc = raw_desc.replace('\n', ' ')
                    wrapped_desc = "<br>".join(textwrap.wrap(clean_desc, width=40))
                
                    if info.get('shape') == 'square': 
                        text = f"<b>{info.get('label','')}</b><br>{info.get('desc','')}"
                    else:  
                        text = f"<b>{info.get('label','')}</b><br>" \
                            f"Mass: {info.get('mass','N/A')}<br>" \
                            f"Charge: {info.get('charge','N/A')}<br>" \
                            f"Spin: {info.get('spin','N/A')}<br>" \
                            f"{wrapped_desc}"
                    node_text.append(text)

                    colors.append(info.get('color', '#2b7de9'))
                    symbols.append(info.get('shape', 'circle'))
                    sizes.append(44 if info.get('shape') == 'diamond' else 30)


            
                edge_x = []
                edge_y = []
                edge_hovertext = []
                annotations = []

                for source, target, data in edges_to_plot:
                    source_key = node_to_pos_keys[source][0]
                    target_key = node_to_pos_keys[target][0]

                    x0, y0 = pos[source_key]
                    x1, y1 = pos[target_key]

                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    interaction = data.get("interaction", "")
                  
                    raw_edge_desc = data.get("desc", "")
                    clean_edge_desc = raw_edge_desc.replace('\n', ' ')
                    wrapped_edge_desc = "<br>".join(textwrap.wrap(clean_edge_desc, width=40))
                    hover = f"<b>{interaction}</b><br>{G.nodes[source].get('label', source)} → {G.nodes[target].get('label', target)}<br>{wrapped_edge_desc}"
                    edge_hovertext.append(hover)

                    annotations.append(
                        dict(
                            x=x1, y=y1, ax=x0, ay=y0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,        # Stile freccia più nitido
                            arrowsize=1.2,      # Freccia leggermente più grande
                            arrowwidth=2,       # Linea più spessa
                            arrowcolor="#546e7a", # Colore Blue-Grey professionale
                            opacity=1,
                            standoff=8,         # Distanza dal nodo per non toccarlo
                            startstandoff=8,    # Distanza dall'inizio
                            text=interaction if interaction else "",
                            font=dict(size=11, color="#455a64", family="Helvetica"),
                            bgcolor="rgba(255,255,255,0.7)", # Sfondo testo semitrasparente per leggibilità
                            hovertext=wrapped_edge_desc if wrapped_edge_desc else "",
                            hoverlabel=dict(      
            bgcolor="white", 
            bordercolor="black",
            font=dict(size=12)
           
                        )
                    ))


            else:
                
                category_nodes = list(G.nodes)

                if category_nodes:
                    sub_pos = nx.spring_layout(G.subgraph(category_nodes), seed=seed, k=0.7)

            
                    roots = [n for n in category_nodes if G.nodes[n].get("shape") in ("diamond", "square")]

                    if roots:
                    
                        ys = [p[1] for p in sub_pos.values()]
                        ymin, ymax = min(ys), max(ys)

                    
                        for n, p in sub_pos.items():
                            y_norm = (p[1] - ymin) / (ymax - ymin + 1e-9)
                            sub_pos[n] = (p[0], y_norm)

                    
                        for r in roots:
                            sub_pos[r] = (sub_pos[r][0], 1.0)

                    pos.update(sub_pos)



                edge_x = []
                edge_y = []
                edge_hovertext = []  
                annotations = []

            
                node_to_pos_keys = {}
                for key in pos.keys():
                    if isinstance(key, tuple):
                        node_id = key[0]
                    else:
                        node_id = key
                    node_to_pos_keys.setdefault(node_id, []).append(key)

                for source, target, data in G.edges(data=True):
                
                    source_key = node_to_pos_keys[source][0]
                    target_key = node_to_pos_keys[target][0]

                    x0, y0 = pos[source_key]
                    x1, y1 = pos[target_key]

                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                    interaction = data.get("interaction", "")
                    raw_edge_desc = data.get("desc", "")
                    clean_edge_desc = raw_edge_desc.replace('\n', ' ')
                    wrapped_edge_desc = "<br>".join(textwrap.wrap(clean_edge_desc, width=40))
                    
                    hover = f"<b>{interaction}</b><br>{G.nodes[source].get('label', source)} → {G.nodes[target].get('label', target)}<br>{wrapped_edge_desc}"
                    edge_hovertext.append(hover)

                    annotations.append(
                        dict(
                            x=x1, y=y1, ax=x0, ay=y0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,        # Stile freccia più nitido
                            arrowsize=1.2,      # Freccia leggermente più grande
                            arrowwidth=2,       # Linea più spessa
                            arrowcolor="#546e7a", # Colore Blue-Grey professionale
                            opacity=1,
                            standoff=8,         # Distanza dal nodo per non toccarlo
                            startstandoff=8,    # Distanza dall'inizio
                            text=interaction if interaction else "",
                            font=dict(size=11, color="#455a64", family="Helvetica"),
                            bgcolor="rgba(255,255,255,0.7)", # Sfondo testo semitrasparente per leggibilità
                            hovertext=wrapped_edge_desc if wrapped_edge_desc else "",
                            hoverlabel=dict(     # Testo allineato a sinistra
            bgcolor="white", bordercolor="black",
            font=dict(size=12))
                        )
                    )


        
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color="#546e7a"), # Più spesso e stesso colore delle frecce
                mode='lines',
                hoverinfo='text',
                hovertext=[h for h in edge_hovertext for _ in (0,1,2)],
                showlegend=False,
                opacity=0.8
            )

        
            node_x = []
            node_y = []
            node_text = []
            labels = []
            colors = []
            symbols = []
            sizes = []

            for key, (x, y) in pos.items():
            
                if isinstance(key, tuple):
                    node_id = key[0]
                else:
                    node_id = key

                info = G.nodes[node_id]

                node_x.append(x)
                node_y.append(y)
                labels.append(info.get('label', node_id))
            
                text = f"<b>{info.get('label','')}</b><br>" \
                    f"Mass: {info.get('mass','N/A')}<br>" \
                    f"Charge: {info.get('charge','N/A')}<br>" \
                    f"Spin: {info.get('spin','N/A')}<br>" \
                    f"{info.get('desc','')}"
                node_text.append(text)
                
                colors.append(info.get('color', '#2b7de9'))
                symbols.append(info.get('shape', 'circle'))
            
                sizes.append(44 if info.get('shape') == 'diamond' else 30)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=labels,
                textposition="bottom center",
                textfont=dict(size=13, family="Arial", color="black"), # Font più grande e leggibile
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    showscale=False,
                    color=colors,
                    size=sizes,
                    line=dict(width=2, color='#37474f'), # Bordo scuro "Dark Slate" per contrasto netto
                    symbol=symbols,
                    opacity=1,
                    gradient=dict(
                        type='radial',     
                        color='white'       
                    )
                ),
                
                showlegend=False
            )

        
            fig = go.Figure(data=[edge_trace, node_trace])

    
            fig.update_layout(
            title={
                    'text': f"<b>{title}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
            title_x=0.5,
            width=width,  
            height=height,
            autosize=True if width is None else False,
            hovermode='closest',
           margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            hoverlabel=dict(
            align="left", 
            bgcolor="white", 
            bordercolor="black",
            font=dict(size=12),
            namelength=-1
        )
        )

            #plot_width = "100%" if is_process_graph else "600px"
            #ui.plotly(fig).classes('w-full h-full').style(f"height:{height}px; width: {plot_width}")
            if width is None:
        
                ui.plotly(fig).classes('w-full h-full').style(f"height:{height}px;").props('tabindex=0 aria-label="Interactive graph visualization of particles and processes"')
            else:
              
                ui.plotly(fig).style(f"height:{height}px; width: {width}px;").props('tabindex=0 aria-label="Interactive graph visualization of particles and processes"')

        def _superscript(n: int):
            sup = str(n).replace('-', '⁻')
            sup = sup.translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
            return f"10{sup}"
        def _parse_mass_to_eV(mass_str):
        
            s = str(mass_str).strip()
            if s == "" or s.lower() == "n/a":
                return None, s
        
            s = s.replace("\u2009", "") 
            m = re.match(r'^\s*<\s*([0-9.eE+-]+)\s*([a-zA-Z]*)', s)
            if m:
                val = float(m.group(1))
                unit = m.group(2) or "eV"
            
                if unit.lower() in ["ev", "eV"]:
                    return val, s
                if unit.lower() in ["kev"]:
                    return val * 1e3, s
                if unit.lower() in ["mev"]:
                    return val * 1e6, s
                if unit.lower() in ["gev"]:
                    return val * 1e9, s
            
                return val, s

        
            if re.match(r'^(0(\.0*)?)$', s) or s.strip() == "0":
                return 0.0, s

        
            m = re.match(r'^\s*([0-9.eE+-]+)\s*([a-zA-Zμμμ^0-9]*)', s)
            if not m:
                return None, s
            val = float(m.group(1))
            unit = m.group(2).strip()
        
            unit = unit.replace(" ", "").replace("MeV/c2", "MeV").replace("GeV/c2", "GeV")
            unit = unit.replace("u", "u")  
            if unit.lower().startswith("ev") or unit == "":
                return val, s
            if unit.lower().startswith("kev"):
                return val * 1e3, s
            if unit.lower().startswith("mev"):
                return val * 1e6, s
            if unit.lower().startswith("gev"):
                return val * 1e9, s
        
            return val, s

        def plot_mass_strip(container, height=280, width=None, title="Particles masses"):

        
            categories = [
                ("Leptons", leptons_nodes, "#7b61ff"),
                ("Quarks", quarks_nodes, "#ff6b6b"),
                ("Bosons", bosons_nodes, "#ffd54f"),
                ("Hadrons", hadrons_nodes, "#4da6ff"),
            ]

            fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.1, 0.9],  
        shared_yaxes=True,
        horizontal_spacing=0.02,  
    )
            #fig = go.Figure()
            y_positions = {cat[0]: i for i, cat in enumerate(reversed(categories))}  
            
            ymin = -0.5
            ymax = len(categories) - 0.5
            for cat_name, _, color in categories:
                y = y_positions[cat_name]
                fig.add_shape(
            type="rect",
            xref="paper", yref="y",  
            x0=0, x1=1,              
            y0=y-0.45, y1=y+0.45,
            fillcolor=color,
            opacity=0.08,            
            line_width=0,
            layer="below",         
        )

        
            xs = []
            traces = []
            all_massive_values = []

            
            for cat_name, node_list, root_color in categories:
                y = y_positions[cat_name]
                
              
                massive_x = []
                massive_hover = []
                
              
                massless_x = []
                massless_hover = []

                for n in node_list:
                    
                    if n.get("shape") in ("diamond", "square") and (n.get("label", "").upper() == cat_name.upper()):
                        continue
                    
                    mass_str = n.get("mass", "N/A")
                    mass_eV, mass_label = _parse_mass_to_eV(mass_str)
                
                    hover_txt = f"<b>{n.get('label','')}</b><br>Category: {cat_name}<br>Mass: {mass_label}<br>Charge: {n.get('charge','N/A')}"

                    if mass_eV == 0.0:
                        
                        massless_x.append(0) 
                        massless_hover.append(hover_txt)
                    
                    elif mass_eV is not None:
                        
                        massive_x.append(mass_eV)
                        massive_hover.append(hover_txt)
                        all_massive_values.append(mass_eV)

            
                if massive_x:
                    fig.add_trace(
                        go.Scatter(
                            x=massive_x,
                            y=[y + np.random.normal(scale=0.06) for _ in massive_x],
                            mode="markers",
                           
                            marker=dict(
                                size=14, 
                                color=root_color, 
                                line=dict(width=2, color="white"), 
                                symbol="circle",
                                opacity=1
                            ),
                            hoverinfo="text",
                            hovertext=massive_hover,
                            showlegend=False
                        ),
                        row=1, col=2
                    )

               
                if massless_x:
                    fig.add_trace(
                        go.Scatter(
                            x=massless_x,
                            y=[y + np.random.normal(scale=0.06) for _ in massless_x],
                            mode="markers",
                      
                            marker=dict(
                                size=14, 
                                color=root_color, 
                                line=dict(width=2, color="white"), 
                                symbol="circle", 
                                opacity=1
                            ),
                            hoverinfo="text",
                            hovertext=massless_hover,
                            showlegend=False
                        ),
                        row=1, col=1
                    )

            
            if not all_massive_values:
                x_min_log, x_max_log = 1e-3, 1e12
            else:
                x_min_log = min(all_massive_values) * 0.1
                x_max_log = max(all_massive_values) * 10.0

           
            log_tickvals = [10**i for i in range(-9, 13)] 
            log_ticktext = [_superscript(i) for i in range(-9, 13)]

          
            ytickvals = [y_positions[c[0]] for c in categories]
            yticktext = [c[0] for c in categories]

            fig.update_layout(
                title={
                    'text': f"<b>{title}</b>",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': 'black'}
                },
                width=width,
                height=height,
                autosize=True if width is None else False,
                margin=dict(l=20, r=20, t=60, b=40),
                hovermode="closest",
                plot_bgcolor='rgba(0,0,0,0)', 
            )

            

            
            fig.update_xaxes(
                title_text="Massless", 
                range=[-0.5, 0.5],    
                showgrid=False,
                zeroline=True,         
                zerolinecolor="gray",
                tickvals=[0],          
                ticktext=["0"],
                row=1, col=1
            )

            fig.update_xaxes(
                type="log",
                title_text="Mass (eV)",
                range=[np.log10(x_min_log), np.log10(x_max_log)],
                tickmode="array",
                tickvals=log_tickvals,
                ticktext=log_ticktext,
                showgrid=True,
                zeroline=False,
                row=1, col=2
            )

           
            fig.update_yaxes(
                tickmode="array",
                tickvals=ytickvals,
                ticktext=yticktext,
                range=[-0.5, len(categories)-0.5],
                showgrid=False,
                zeroline=False,
         
                showticklabels=True, 
                row=1, col=1
            )
            
            
            fig.update_yaxes(showticklabels=False, row=1, col=2)

       
            fig.add_shape(
                type="line",
                xref="paper", yref="paper",
                x0=0.105, x1=0.105, 
                y0=0, y1=1,
                line=dict(color="gray", width=1, dash="dot"),
            )

            if width is None:
                ui.plotly(fig).classes('w-full').style(f"height:{height}px; width: 100%;").props('tabindex=0 aria-label="Mass distribution of fundamental particles"')
            else:
                ui.plotly(fig).style(f"height:{height}px; width: {width}px;").props('tabindex=0 aria-label="Mass distribution of fundamental particles"')
            #ui.plotly(fig).classes('w-full').style(f"height:{height}px; width: 100%;")
            #ui.plotly(fig).style(f"height:{height}px; width:100%")
        
      
        
    
 


        def open_comprehensive_particles_dialog():
            with ui.dialog() as comp_particles_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="main-title"'):
                
                with ui.column().classes('w-full h-full bg-white'):
                    
                 
                    with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="main-title"'):
                        ui.label('Particles & Cosmology').classes('text-xl font-bold').props('id="main-title" role=heading aria-level=2 tabindex=0')
                        aria_button('Close', 'close', on_click=comp_particles_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    
                    with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                        t_info = ui.tab('Particles Classification ')
                        t_mass_conv = ui.tab('Mass Conversion')
                        t_leptons = ui.tab('Leptons')
                        t_quarks = ui.tab('Quarks')
                        t_bosons = ui.tab('Bosons')
                        t_hadrons = ui.tab('Hadrons')
                        t_processes = ui.tab('Cosmic Processes')
                        t_mass_plot = ui.tab('Mass Plot')
                        #t_materials = ui.tab('Supplementary Materials')

                        tabs.on_value_change(lambda: ui.run_javascript("setTimeout(() => { if(typeof MathJax !== 'undefined') MathJax.typesetPromise(); }, 100)"))
                    with ui.tab_panels(tabs, value=t_info).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                        
                      
                        with ui.tab_panel(t_info):
                            html_info_box(r"""
            <h3>1. The Particles Zoo in the universe(Micro-Cosmos)</h3>
            
            <p>The Standard Model of Particle Physics classifies all known fundamental particles into several categories based on their properties and interactions. Here's an overview of the main categories:</p>

            <h4>A. Leptons</h4>
            <p><b>Fermions (Spin 1/2) that do not interact via the strong force.</b></p>
            <ul>
                <li><b>Electron (e⁻):</b> The lightest charged lepton (0.511 MeV). Stable and fundamental to atomic structure.</li>
                <li><b>Muon (<span class="math">\(\mu^-\)</span>) & Tau (<span class="math">\(\tau^-\)</span>):</b> Heavier, unstable cousins of the electron. They decay quickly into lighter particles.</li>
                <li><b>Neutrinos (<span class="math">\(\nu_e, \nu_\mu, \nu_\tau\)</span>):</b> Neutral "ghost" particles corresponding to each charged lepton. They have tiny masses (<1 eV) and interact only via the weak force.</li>
            </ul>

            <h4>B. Quarks</h4>
            <p><b>Fermions (Spin 1/2) that interact via the strong force (QCD). They possess fractional electric charges.</b></p>
            <ul>
                <li><b>Up (u) & Down (d):</b> The light quarks that make up everyday matter (protons and neutrons).</li>
                <li><b>Charm (c) & Strange (s):</b> Heavier, second-generation quarks. Found in exotic particles like Kaons.</li>
                <li><b>Top (t) & Bottom (b):</b> The heaviest generation. The Top quark is the most massive elementary particle known (173 GeV).</li>
            </ul>

            <h4>C. Bosons</h4>
            <p><b>The Force Carriers (Integer Spin).</b></p>
            <ul>
                <li><b>Photon (<span class="math">\(\gamma\)</span>):</b> Massless carrier of the Electromagnetic force (light).</li>
                <li><b>Gluon (g):</b> Massless carrier of the Strong force; it "glues" quarks together.</li>
                <li><b>W± & Z Bosons:</b> Massive carriers of the Weak force, responsible for radioactive decay.</li>
                <li><b>Higgs Boson (H):</b> The particle associated with the Higgs field, giving mass to other elementary particles.</li>
            </ul>

            <h4>D. Hadrons</h4>
            <p><b>Composite particles made of Quarks held together by Gluons.</b></p>
            <ul>
                <li><b>Baryons (3 Quarks):</b>
                    <ul>
                        <li><b>Proton (p):</b> Stable (uud). The nucleus of Hydrogen.</li>
                        <li><b>Neutron (n):</b> Neutral (udd). Stable inside nuclei, but decays when free.</li>
                    </ul>
                </li>
                <li><b>Mesons (Quark + Anti-Quark):</b>
                    <ul>
                        <li><b>Pion (<span class="math">\(\pi\)</span>):</b> The lightest meson, mediates forces between nucleons.</li>
                        <li><b>Kaon (K):</b> Contains a strange quark.</li>
                    </ul>
                </li>
            </ul>

            <hr style="margin: 20px 0; border-top: 1px solid #0284c7;">

            <h3>2. Cosmological Timeline (Macro-Cosmos)</h3>
            
            <h4>Epoch 1: The Big Bang</h4>
            <p>The initial singularity from which space, time, and energy expanded. The universe began in an extremely hot and dense state.</p>

            <h4>Epoch 2: Big Bang Nucleosynthesis (BBN)</h4>
            <p><b>"The First Nuclei"</b><br>
            Shortly after the Big Bang, the universe cooled enough for protons (p) and neutrons (n) to fuse.</p>
            <ul>
                <li><b>Process:</b> <span class="math">\(p + n \rightarrow D\)</span> (Deuterium) <span class="math">\(\rightarrow\)</span> Helium-4 (⁴He) & Lithium-7 (⁷Li).</li>
                <li><b>Outcome:</b> Created the primordial abundance of light elements (mostly Hydrogen and Helium).</li>
            </ul>

            <h4>Epoch 3: Recombination</h4>
            <p><b>"The First Atoms"</b></p>
            <ul>
                <li><b>Process:</b> The universe cooled to ~3000K. Electrons (e⁻) combined with Protons (p) to form <b>Neutral Hydrogen (H)</b>.</li>
                <li><b>Significance:</b> Photons could finally travel freely, creating the Cosmic Microwave Background (CMB).</li>
            </ul>

            <h4>Epoch 4: Reionization</h4>
            <p><b>"The Cosmic Dawn"</b></p>
            <ul>
                <li><b>Process:</b> The first stars and Active Galactic Nuclei (AGN) formed. Their intense radiation (photons) hit the neutral hydrogen gas.</li>
                <li><b>Outcome:</b> The neutral hydrogen was ionized back into protons and electrons (<span class="math">\(H \rightarrow p + e^-\)</span>), ending the "Dark Ages."</li>
            </ul>
            """).props('aria-label="Particles information"')
                            
                            aria_image('/images/Standard_Model_of_Elementary_Particles.png', "Image of standard model for elementary particles").classes('w-full max-w-[800px] h-auto rounded-lg shadow-lg border border-gray-300 mt-6 mx-auto block')

                      
                        with ui.tab_panel(t_mass_conv):
                            html_info_box(r"""
        <h3>Mass-Energy Conversion: kg to eV</h3>
        <p>In particle physics, mass (\(m\)) is often expressed as equivalent energy (\(E\)) using <b>Einstein's mass-energy equivalence</b> principle:</p>
        
        <div style="text-align:center; font-size: 1.2em; margin: 15px 0;">
            \[ E = m c^2 \]
        </div>

        <p>Where \(c\) is the speed of light in a vacuum.</p>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Conversion Steps</h4>
        <ol>
            <li>
                <b>Mass (kg) to Energy (Joule):</b>
                <div style="text-align:center; margin: 10px 0;">\[ E \, (\mathrm{J}) = m \, (\mathrm{kg}) \times (c)^2 \]</div>
                <p>(\(c \approx 2.9979 \times 10^8 \, \mathrm{m/s}\))</p>
            </li>
            <li style="margin-top: 10px;">
                <b>Energy (Joule) to Electronvolt (eV):</b>
                <p>The energy in Joules must be divided by the elementary charge (\(e\)) to get Electronvolts.</p>
                <div style="text-align:center; margin: 10px 0;">\[ E \, (\mathrm{eV}) = \frac{E \, (\mathrm{J})}{e} \]</div>
                <p>(\(e \approx 1.60217 \times 10^{-19} \, \mathrm{J/eV}\))</p>
            </li>
        </ol>

        <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

        <h4>Key Conversion Factors</h4>
        <p>The standard convention uses prefixes to denote magnitude:</p>
        
        <table style="width:100%; border-collapse: collapse; margin-top: 10px; text-align:center;">
            <tr style="background-color: #bae6fd; color: #0c4a6e;">
                <th style="border: 1px solid #0284c7; padding: 8px;">Unit</th>
                <th style="border: 1px solid #0284c7; padding: 8px;">Value in eV</th>
                <th style="border: 1px solid #0284c7; padding: 8px;">Example</th>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">keV (kilo)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">\(10^3 \, \mathrm{eV}\)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Binding energies of inner-shell electrons.</td>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">MeV (mega)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">\(10^6 \, \mathrm{eV}\)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Electron rest mass (\(\approx 0.511 \, \mathrm{MeV}\))</td>
            </tr>
            <tr>
                <td style="border: 1px solid #0284c7; padding: 8px;">GeV (giga)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">\(10^9 \, \mathrm{eV}\)</td>
                <td style="border: 1px solid #0284c7; padding: 8px;">Proton rest mass (\(\approx 0.938 \, \mathrm{GeV}\))</td>
            </tr>
        </table>
        
        <p style="margin-top: 15px; font-weight: bold;">
            The factor for 1 kg mass is approximately \( 5.61 \times 10^{35} \, \mathrm{eV} \).
        </p>
        """)

                        with ui.tab_panel(t_leptons).classes('flex flex-col items-center'):
                            plot_particle_graph("Leptons (Fermions)", leptons_nodes, leptons_edges, height=600, width=800)
                            
                        with ui.tab_panel(t_quarks).classes('flex flex-col items-center'):
                            plot_particle_graph("Quarks (Fermions)", quarks_nodes, quarks_edges, height=600, width=800)
                            
                        with ui.tab_panel(t_bosons).classes('flex flex-col items-center'):
                            plot_particle_graph("Bosons (Force Carriers & Higgs)", bosons_nodes, bosons_edges, height=600, width=800)
                            
                        with ui.tab_panel(t_hadrons).classes('flex flex-col items-center'):
                            plot_particle_graph("Hadrons (Baryons & Mesons)", hadrons_nodes, hadrons_edges, height=600, width=800)

                       
                        with ui.tab_panel(t_processes).classes('flex flex-col items-center'):
                            plot_particle_graph("Cosmological Processes ", processes_nodes, processes_edges, is_process_graph=True, height=700 , width=500)

                        with ui.tab_panel(t_mass_plot).classes('flex flex-col items-center overflow-x-auto'):
                            
                            mass_container = ui.column().classes('w-full items-center justify-center')
                            plot_mass_strip(mass_container, height=400, width=1000)

                      
                       # with ui.tab_panel(t_materials).classes('flex flex-col items-center justify-center h-full'):
                        #    ui.label("Introductory Slides").classes("text-3xl font-bold text-slate-800 mb-4")
                         #   ui.label("Click the button below to view the supplementary PDF presentation about particles.").classes("text-lg text-gray-600 mb-8")
                          #  aria_button(
                           #     'Open PDF: Particles', 
                            #    'Open Introductory Slides about Particles',
                             #   on_click=lambda: ui.run_javascript('window.open("/slides/Particles.pdf", "_blank")')
                           # ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-3 px-8 text-lg rounded shadow-lg")

            comp_particles_dialog.open()
        def particle_page(container):
            timer_state = {'time': 0, 'running': False, 'started': False}
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            with container:
                ui.label(
    "Dive into the subatomic world to identify the fundamental particles and physical forces that weave the composition of the universe.").classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                with ui.dialog() as instru , ui.card().classes("p-4 w-full max-w-[800px]").props('role=dialog aria-label=Instructions'):
                    html_info_box(r"""
<div style="font-family: sans-serif; line-height: 1.6;">
    
    <h3 style="margin-top: 0; color: #059669; border-bottom: 2px solid #10b981; padding-bottom: 10px; margin-bottom: 15px;">1. Particle Physics Challenge</h3>
    <p>Put your knowledge of the subatomic world to the test!</p>
    <ul style="margin-bottom: 25px;">
        <li>Press <b>Start Game</b> to begin the identification sequence. A timer will track your speed in analyzing the quantum world.</li>
        <li>For each round, you will be presented with specific properties (like Mass, Charge, or Spin) or a conversion task. Identify the <b>correct particle name</b> or <b>energy value</b>.</li>
        <li><b>Scoring:</b> You earn <b>1 point</b> for every correct identification. A wrong choice results in a <b>-0.25 point</b> penalty.</li>
        <li><b>Interactive Tools:</b> Use the information you've gathered from the interactive graphs to solve the most difficult questions.</li>
        <li>Complete the full sequence of particles to stop the clock and achieve "Quantum Mastery".</li>
    </ul>

    <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Particle Activity Instructions</h3>
    <ol style="padding-left: 20px;">
        <li style="margin-bottom: 10px;"><strong>Explore the Particle Zoo:</strong> Familiarize yourself with the various fundamental particles such as leptons, quarks, bosons, and hadrons. Use the interactive graphs to visualize their properties and relationships.</li>
        <li style="margin-bottom: 10px;"><strong>Understand Mass-Energy Conversion:</strong> Learn how to convert mass in kilograms to energy in electronvolts (eV) using Einstein's mass-energy equivalence principle ($E=mc^2$). Refer to the information boxes for detailed steps and key conversion factors.</li>
        <li style="margin-bottom: 10px;"><strong>Analyze Particle Interactions:</strong> Study the interaction graphs to see how particles interact through fundamental forces. Pay attention to the processes and outcomes of these interactions.</li>
    </ol>
    
    <div style="border-left: 4px solid #10b981; padding-left: 15px; margin-top: 20px; background-color: #ecfdf5; padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
        <p style="margin: 0; color: #064e3b; font-size: 0.95em;">
            <strong>Tip:</strong> Keep the <b>Mass-Energy Calculator</b> ready! Some game questions will ask you to calculate the energy equivalent of a particle's rest mass.
        </p>
    </div>
</div>
""")
                    aria_button("Close", "close popup", on_click=lambda:instru.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                
                with ui.dialog() as cur_part, ui.card().classes('p-4 w-full max-w-[600px]').props('role=dialog aria-label="Curiosity: The Ghost Particles"'):
                    html_info_box(r"""
                    <h3>The Ghost Particles</h3>
                    <p>Did you know that you are currently being bombarded by "ghosts"?</p>
                    <p><b>Neutrinos</b> are elusive fundamental particles that rarely interact with matter. Right now, about <b>100 trillion</b> neutrinos from the Sun are passing through your body every second!</p>
                    <p>They are so non-interactive that a neutrino could pass through a block of lead one light-year thick without hitting a single atom.</p>
                    """)
                    reference_box("**Source:** [Di Valentino (2024)](https://arxiv.org/html/2404.19322v1)")
                    aria_button("Close", "close", on_click=lambda:cur_part.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

              
                
                with ui.row().classes('w-full justify-center gap-2'):
                    aria_button("Instructions", "Open instructions", on_click=lambda:instru.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button(
            'Scientific Info', 
            'Open Comprehensive Particles Dialog', 
            on_click=lambda: [open_comprehensive_particles_dialog(), ui.run_javascript("setTimeout(() => { if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); } }, 250);")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                    aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_part.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                    aria_button(
                        '♿ Accessibility Mode', 
                        'Jump to accessibility exercise',
                        on_click=lambda: (
                            acc_particles.open(), 
                            ui.run_javascript("setTimeout(() => document.getElementById('acc_particles').scrollIntoView({behavior: 'smooth', block: 'start'}), 100);")
                        )
                    ).classes("!bg-red-600 hover:!bg-red-700 text-white font-bold py-2 px-4 rounded shadow-md")
                    
               
                    
                ui.run_javascript('window.particleGameRunning = false; window.particleScore = 0;')
                timer_state = {'time': 0, 'running': False}

                
                GAME_PARTICLES = [
                    {"symbol": "e⁻", "color": "#c8b8ff", "targets": ["Fermions", "Leptons"]},
                    {"symbol": "μ⁻", "color": "#bda0ff", "targets": ["Fermions", "Leptons"]},
                    {"symbol": "νₑ", "color": "#e6e0ff", "targets": ["Fermions", "Leptons"]},
                    {"symbol": "u", "color": "#ffb4b4", "targets": ["Fermions", "Quarks"]},
                    {"symbol": "d", "color": "#ffbcbc", "targets": ["Fermions", "Quarks"]},
                    {"symbol": "s", "color": "#ff9a9a", "targets": ["Fermions", "Quarks"]},
                    {"symbol": "γ", "color": "#fff1c2", "targets": ["Bosons"]},
                    {"symbol": "g", "color": "#ffe59a", "targets": ["Bosons"]},
                    {"symbol": "W±", "color": "#ffd27a", "targets": ["Bosons"]},
                    {"symbol": "H", "color": "#ffe082", "targets": ["Bosons"]},
                    {"symbol": "p", "color": "#cfe9ff", "targets": ["Fermions", "Hadrons", "Baryons"]},
                    {"symbol": "n", "color": "#dff2ff", "targets": ["Fermions", "Hadrons", "Baryons"]},
                    {"symbol": "π", "color": "#bfe0ff", "targets": ["Bosons", "Hadrons", "Mesons"]}
                ]
                CATEGORIES = ["Fermions", "Leptons", "Quarks", "Bosons", "Hadrons", "Mesons", "Baryons"]

            
                pool_items = []
                for p in GAME_PARTICLES:
                    for i in range(len(p['targets'])):
                        pool_items.append({
                            "uid": f"ball_{p['symbol'].replace('⁻', '_').replace('±', '_')}_{i}",
                            "symbol": p["symbol"],
                            "color": p["color"],
                            "targets": p["targets"]
                        })
                
               
                random.shuffle(pool_items)

                with ui.column().classes('w-full p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-600 flex flex-col items-center mt-8 mb-12'):
                    ui.label("🎮 AstroParticles: Challenge").classes("text-3xl md:text-4xl font-bold text-yellow-400 mb-4 text-center drop-shadow-md uppercase tracking-wider").props('role=heading aria-level=2 tabindex=0')

                    with ui.row().classes('w-full max-w-[800px] justify-between bg-slate-800 p-4 rounded-lg border border-slate-500 mb-6 shadow-inner items-center'):
                        time_label = ui.label('Time: 0s').classes('text-xl text-white font-bold w-24')
                        ui.html('<div id="particle_score" class="text-xl font-bold w-32 text-center text-green-400">Score: 0</div>')

                        def update_timer():
                            if timer_state['running']:
                                timer_state['time'] += 1
                                time_label.set_text(f"Time: {timer_state['time']}s")
                        ui.timer(1.0, update_timer)

                        def start_game():
                            timer_state['time'] = 0
                            timer_state['running'] = True
                            timer_state['started'] = True
                            time_label.set_text("Time: 0s")
                            ui.run_javascript("window.particleGameRunning = true; window.particleScore = 0; document.getElementById('particle_score').innerText = 'Score: 0';")
                            render_game.refresh()

                        def stop_game():
                            timer_state['running'] = False
                            ui.run_javascript("window.particleGameRunning = false;")

                        with ui.row().classes('gap-4'):
                            aria_button("Start", "start", on_click=start_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105 animate-pulse")
                            aria_button("Stop", "stop", on_click=stop_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                            aria_button("Reset", "reset", on_click=lambda: (
                                stop_game(),
                                timer_state.update({'time': 0, 'started': False}),
                                time_label.set_text("Time: 0s"),
                                ui.run_javascript("window.particleScore = 0; document.getElementById('particle_score').innerText = 'Score: 0';"),
                                render_game.refresh()
                            )).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")

                    @ui.refreshable
                    def render_game():
                        if not timer_state.get('started', False):
                            ui.label("Press Start to reveal the quantum realm!").classes("text-slate-400 italic text-center text-2xl w-full py-12")
                            return

                        with ui.row().classes('w-full items-start justify-between gap-6 flex-wrap md:flex-nowrap'):
                            with ui.column().classes('w-full md:w-1/3 min-h-[500px] bg-slate-900 rounded-lg p-4 border border-slate-600 shadow-inner'):
                                ui.label('Particle Pool').classes('text-lg font-bold text-gray-300 mb-2 text-center w-full border-b border-slate-600 pb-2')
                                with ui.row().classes('w-full flex-wrap gap-3 justify-center mt-2'):
                                    for item in pool_items:
                                        ball = ui.label(item['symbol']).classes(
                                            'w-10 h-10 rounded-full flex items-center justify-center font-bold text-black shadow-md cursor-grab active:cursor-grabbing hover:scale-110 transition-transform'
                                        ).style(f"background-color: {item['color']};")
                                        ball.props('draggable="true"')
                                        ball.props(f'id="{item["uid"]}"')
                                        target_str = ",".join(item['targets']) 
                                        ball.props(f'ondragstart="event.dataTransfer.setData(\'application/json\', JSON.stringify({{uid:\'{item["uid"]}\', symbol:\'{item["symbol"]}\', color:\'{item["color"]}\', targets:\'{target_str}\'}}))"')

                            with ui.column().classes('w-full md:w-2/3 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4'):
                                for box_name in CATEGORIES:
                                    with ui.card().classes('bg-slate-700 border-2 border-dashed border-slate-400 rounded-lg p-2 min-h-[140px] flex flex-col items-center shadow-lg'):
                                        ui.label(box_name).classes('text-sm font-bold text-white mb-2 uppercase tracking-wide border-b border-slate-500 w-full text-center pb-1')
                                        drop_zone = ui.row().classes('w-full h-full flex-wrap gap-1 justify-center items-start content-start min-h-[100px]')
                                        js_logic = (
                                            f'ondragover="event.preventDefault();" '
                                            f'ondrop="event.preventDefault(); '
                                            f'if(!window.particleGameRunning) {{ '
                                            f'  if(window.Quasar && window.Quasar.Notify) {{ window.Quasar.Notify.create({{ message: \'Press START to play!\', color: \'warning\', position: \'top\' }}); }} '
                                            f'  return; '
                                            f'}} '
                                            f'const rawData = event.dataTransfer.getData(\'application/json\'); '
                                            f'if(!rawData) return; ' 
                                            f'const data = JSON.parse(rawData); '
                                            f'const targetCat = \'{box_name}\'; '
                                            f'const targetArray = data.targets.split(\',\'); ' 
                                            f'if(!targetArray.includes(targetCat)) {{ '
                                            f'  window.particleScore -= 0.25; document.getElementById(\'particle_score\').innerText = \'Score: \' + window.particleScore; '
                                            f'  if(window.Quasar && window.Quasar.Notify) {{ window.Quasar.Notify.create({{ message: data.symbol + \' is not a \' + targetCat + \'!\', color: \'negative\', position: \'top\', icon: \'cancel\' }}); }} '
                                            f'  return; '
                                            f'}} '
                                            f'window.particleScore += 1; document.getElementById(\'particle_score\').innerText = \'Score: \' + window.particleScore; '
                                            f'const ball = document.createElement(\'div\'); '
                                            f'ball.className = \'w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shadow-md m-0.5 text-black\'; '
                                            f'ball.style.backgroundColor = data.color; '
                                            f'ball.innerText = data.symbol; '
                                            f'this.appendChild(ball); '
                                            f'const sourceBall = document.getElementById(data.uid); '
                                            f'if(sourceBall) {{ sourceBall.style.visibility = \'hidden\'; sourceBall.style.pointerEvents = \'none\'; }}"'
                                        )
                                        drop_zone.props(js_logic)

                    render_game()
                    acc_particles = ui.expansion('Accessibility Mode: Particle Classification', icon='accessible').classes('w-full mt-8 bg-slate-800 text-white rounded-lg border border-slate-600').props('id="acc_particles" aria-label="Keyboard accessible alternative for particle classification"')
                    with acc_particles:
                        ui.label('Classify each particle by selecting one of its correct categories.').classes('mb-4 text-lg font-bold text-blue-200').props('tabindex=0')
                        
                        user_choices = {}
                        
                        with ui.column().classes('w-full gap-4 p-2'):
                           
                            for p in GAME_PARTICLES:
                                with ui.row().classes('w-full items-center justify-between border-b border-slate-700 pb-2 flex-wrap'):
                                    
                                 
                                    ui.label(p['symbol']).classes('text-2xl font-bold text-blue-300').props('tabindex=0')
                                    
                                
                                    user_choices[p['symbol']] = ui.select(
                                        options=CATEGORIES, 
                                        label='Select Category'
                                    ).classes('w-48').props(f'dark aria-label="Select category for particle {p["symbol"]}"')
                        
                     
                        def check_particles_accessible():
                            score = 0
                            errors = 0
                            
                            for p in GAME_PARTICLES:
                                selected = user_choices[p['symbol']].value
                                
                                if not selected:
                                    continue
                                    
                               
                                if selected in p['targets']:
                                    score += 1
                                else:
                                    errors += 1
                            
                            tot_items = len(GAME_PARTICLES)
                            
                            if score == tot_items:
                                accessible_notify(f"✅  Mastery Achieved! Score: {score}/{tot_items}", type_='success')
                            elif score > 0 and errors == 0:
                                accessible_notify(f"Good progress! Correct: {score}. Keep going.", type_='info')
                            elif errors > 0:
                                accessible_notify(f"⚠️ Anomalies detected (Correct: {score}, Wrong: {errors}). Review the Standard Model and try again!", type_='warning')
                            else:
                                accessible_notify("Please select a category before checking.", type_='warning')

                        aria_button(
                            'Verify Particles', 
                            'Submit accessible particle classification', 
                            on_click=check_particles_accessible
                        ).classes('mt-4 !bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded shadow-md')
                
        
                    
                    



                
                
                






        
#panel discovery timeline   
                
        DISCOVERY_EVENTS = {

       "Prehistory and Ancient": [
        {'year': -38000, 'title': 'Earliest Lunar Tracking',
        'image': 'paleolithic_lunar_bone.jpg',
        'desc': 'Upper Paleolithic humans record lunar cycles on engraved bones,possibly the earliest astronomical observations. [Wikimedia](https://commons.wikimedia.org/wiki/File:Lunar_calendar_Blanchard_MAN_ArtsEtPrehistoire.jpg)'},

        {'year': -5000, 'title': 'Goseck Solar Observatory',
        'image': 'goseck_circle.jpg',
        'desc': 'The Goseck Circle in Germany is built, aligned with winter solstice sunrise and sunset. [Wikipedia](https://en.wikipedia.org/wiki/Goseck_Circle)'},

        {'year': -2500, 'title': 'Stonehenge Alignments',
        'image': 'stonehenge.jpg',
        'desc': 'Stonehenge constructed with precise solstice alignments for tracking seasons. [Wikipedia](https://en.wikipedia.org/wiki/Stonehenge)'},

     
        {'year': -280, 'title': 'Aristarchus Proposes Heliocentrism',
        'image': 'aristarchus.jpg',
        'desc': 'Aristarchus of Samos proposes a Sun-centered model of the Solar System. [Astronomia](https://www.astronomia.com/progetti/locchio-infinito-di-galileo/prima-di-galileo/aristarco-di-samo-e-la-teoria-eliocentrica/),[AstronomyForChange](https://astronomyforchange.org/greek-scholars-development-constellations/)'},
       ],
       "Classical and Renaissance": [
       
        {'year': 150, 'title': 'Ptolemy’s Almagest',
        'image': 'ptolemy_almagest.jpg',
        'desc': 'Ptolemy publishes the Almagest, defining 48 constellations and a detailed geocentric model. [OldMaa](https://old.maa.org/press/periodicals/convergence/mathematical-treasure-erasmuss-greek-edition-of-ptolemys-almagest),[Wikipedia](https://en.wikipedia.org/wiki/Ptolemy)'},

      
        {'year': 1543, 'title': 'Copernicus – Heliocentric Model',
        'image': 'copernicus.jpg',
        'desc': 'Copernicus publishes *De Revolutionibus*, reviving the heliocentric theory. [Wikipedia](https://it.wikipedia.org/wiki/De_revolutionibus_orbium_coelestium),[AIF](https://www.aif.it/fisico/biografia-nicolo-copernico/)'},

        {'year': 1609, 'title': 'Galileo’s Telescope',
        'image': 'galileo_telescope.jpg',
        'desc': 'Galileo builds a telescope and observes Moon craters, Jupiter’s moons, and star fields. [Britannica](https://www.britannica.com/science/Galilean-telescope), [Wikipedia](https://en.wikipedia.org/wiki/Galileo_Galilei)'},

        {'year': 1609, 'title': 'Kepler’s Laws (I & II)',
        'image': 'kepler_laws.png',
        'desc': 'Johannes Kepler formulates his first two laws of planetary motion. [ScienceNotes](https://sciencenotes.org/keplers-laws-of-planetary-motion/)'},

        {'year': 1619, 'title': 'Kepler’s Third Law',
        'image': 'kepler_third_law.jpg',
        'desc': 'Kepler publishes *Harmonices Mundi*, including the third law of planetary motion. [ResearchGate](https://www.researchgate.net/publication/346510651_The_Timeline_Of_Gravity/figures?lo=1&utm_source=google&utm_medium=organic), [Wikipedia](https://it.wikipedia.org/wiki/Giovanni_Keplero)'},

      
        {'year': 1687, 'title': 'Newton’s Principia',
        'image': 'newton_principia.jpg',
        'desc': 'Isaac Newton publishes the *Principia Mathematica*, founding modern gravitational physics. [UniversityCollection](https://university-collections.wp.st-andrews.ac.uk/2017/07/05/celebrating-330-years-of-isaac-newtons-principia/)'},
       ],
       
       "Eighteenth and Early Twentieth Century": [
        {'year': 1781, 'title': 'Discovery of Uranus',
        'image': 'uranus_discovery.jpg',
        'desc': 'William Herschel discovers Uranus, expanding the Solar System for the first time in history. [BBC](https://www.skyatnightmagazine.com/space-science/how-was-uranus-discovered),[Wikipedia](https://en.wikipedia.org/wiki/William_Herschel)'},

        {'year': 1838, 'title': 'First Stellar Parallax',
        'image': 'stellar_parallax.jpg',
        'desc': 'Friedrich Bessel measures the distance to 61 Cygni using parallax — the first direct stellar distance. [Mlahanas](http://www.mlahanas.de/Greeks/Astronomy2.htm),[Wikipedia](https://en.wikipedia.org/wiki/Friedrich_Wilhelm_Bessel)'},

        {'year': 1868, 'title': 'Helium Discovered in the Sun',
        'image': 'helium_sun.jpg',
        'desc': 'Helium is identified in the solar spectrum before its discovery on Earth by Pierre Janssen. [ArmaghPlanet](https://armaghplanet.com/spectroscopy-and-the-discovery-of-helium.html),[Wikimedia](https://commons.wikimedia.org/wiki/File:Jules_Janssen_2.jpg)'},

       
        {'year': 1912, 'title': 'Leavitt’s Cepheid Law',
        'image': 'henrietta_leavitt.jpg',
        'desc': 'Henrietta Leavitt discovers the period–luminosity relation for Cepheids, enabling cosmic distances. [OgleAstrouw](https://ogle.astrouw.edu.pl/cont/4_main/var/ogleiv/cepheid_collection_update/)'},

        {'year': 1916, 'title': 'Schwarzschild Black Hole Solution',
        'image': 'schwarzschild.jpg',
        'desc': 'Karl Schwarzschild derives the first exact black hole solution to Einstein’s equations. [DarkCosmos](https://darkcosmos.com/home/f/the-metric-of-a-black-hole),[Wikipedia](https://en.wikipedia.org/wiki/Karl_Schwarzschild)'},

        {'year': 1923, 'title': 'Hubble Discovers Galaxies',
        'image': 'hubble_andromeda.jpg',
        'desc': 'Edwin Hubble proves that Andromeda is an external galaxy beyond the Milky Way. [Astronomy](https://www.astronomy.com/science/how-edwin-hubble-won-the-great-debate/),[Wikipedia](https://en.wikipedia.org/wiki/Edwin_Hubble)'},

        {'year': 1929, 'title': 'Cosmic Expansion',
        'image': 'hubble_expansion.jpg',
        'desc': 'Hubble and Lemaitre discover the expansion of the Universe — the Hubble–Lemaître Law. [AstronomyNow-NASA](https://astronomynow.com/2018/10/29/iau-recommends-renaming-hubbles-law-to-include-lemaitre/),[Wikipedia](https://en.wikipedia.org/wiki/Georges_Lema%C3%AEtre)'},

        {'year': 1933, 'title': 'Zwicky Proposes Dark Matter',
        'image': 'zwicky_dark_matter.jpg',
        'desc': 'Fritz Zwicky finds missing mass in galaxy clusters → first dark matter evidence. [Nature](https://www.nature.com/articles/d41586-019-02603-7),[Wikipedia](https://it.wikipedia.org/wiki/Fritz_Zwicky)'},

       ],
       "Space Age": [
        {'year': 1957, 'title': 'Sputnik ',
        'image': 'sputnik.jpg',
        'desc': 'After Serjei Korolev studies about human spaceflights, the first artificial satellite was launched, marking the beginning of the space age. [Wikipedia](https://it.wikipedia.org/wiki/Sputnik_1),[spsAviation](https://www.sps-aviation.com/story/?id=3240&h=Sergei-Korolev-1907-1966)'},

        {'year': 1965, 'title': 'Discovery of the CMB',
        'image': 'cmb_penzias_wilson.jpg',
        'desc': 'Penzias and Wilson detect the cosmic microwave background, confirming the Big Bang. [ResearchGate](researchgate.net/publication/271140622_Physics_of_the_cosmic_microwave_background_anisotropy/figures?lo=1&utm_source=google&utm_medium=organic),[ScienzaPerTutti]([Penzias and Wilson](https://scienzapertutti.infn.it/rubriche/biografie/2709-wilson-robert-woodrow))'},

        {'year': 1969, 'title': 'Apollo Moon Landing',
        'image': 'apollo_11.jpg',
        'desc': 'Neil Armstrong and Buzz Aldrin become the first humans to walk on the Moon. [BBC](https://www.bbc.co.uk/newsround/48789792),[Wikipedia](https://it.wikipedia.org/wiki/Apollo_11)'},

        {'year': 1974, 'title': 'Discovery of the First Pulsar Binary',
        'image': 'binary_pulsar.jpg',
        'desc': 'Hulse and Taylor discover the first binary pulsar — indirect evidence for gravitational waves. [NASA](https://asd.gsfc.nasa.gov/blueshift/index.php/2016/03/17/we-knew-that-already/),[CredoLibrary](https://credo.library.umass.edu/view/full/murg171-sl94-r730-i012)'},

        {'year': 1977, 'title': 'Voyager Launch',
        'image': 'voyager.jpg',
        'desc': 'Voyager 1 and 2 begin their Grand Tour of the outer planets and interstellar space. [NASA](https://www.nasa.gov/image-article/voyager-1s-mission-outer-planet-begins/),[ApNews](https://apnews.com/article/nasa-voyager-spacecraft-contact-19e16b945869623cd94778795e62001b)'},

      ],
       "Modern Epoch": [
        {'year': 1990, 'title': 'Hubble Space Telescope Launch',
        'image': 'hst.jpg',
        'desc': 'The Hubble Space Telescope begins operating, revolutionizing astronomy. [ESA-Hubble](https://esahubble.org/images/hst_launch_hi/)'},

        {'year': 1992, 'title': 'First Exoplanets Detected',
        'image': 'first_exoplanets.jpg',
        'desc': 'The first confirmed exoplanets are found around a pulsar by Wolszczan & Frail. [ScientificAmerican](https://www.scientificamerican.com/blog/observations/who-really-discovered-the-first-exoplanet/),[BigThink](https://bigthink.com/starts-with-a-bang/the-planets-that-never-were/)'},

        {'year': 1998, 'title': 'Accelerating Universe',
        'image': 'supernova_acceleration.jpg',
        'desc': 'Type Ia supernova studies reveal accelerating expansion → dark energy. [ESA](https://www.esa.int/ESA_Multimedia/Images/2019/01/Investigating_the_expansion_of_the_Universe_with_type-Ia_supernovas_and_quasars)'},

       
        {'year': 2015, 'title': 'Gravitational Waves Detected',
        'image': 'ligo_gw150914.jpg',
        'desc': 'LIGO makes the first direct detection of gravitational waves from merging black holes and Rainer Weiss, Kip Thorne and Barry Barish obtained the nobel prize. [LIGO-Caltech](https://www.ligo.caltech.edu/image/ligo20160211a),[BBC](https://www.bbc.com/news/science-environment-41476648)'},

        {'year': 2019, 'title': 'First Black Hole Image (EHT)',
        'image': 'm87_eht.jpg',
        'desc': 'Event Horizon Telescope produces the first image of a black hole (M87*). [EHT](https://eventhorizontelescope.org/blog/astronomers-reveal-first-image-black-hole-heart-our-galaxy),[LotusLeafLive](https://lotusleaflive.com/archives/4839)'},
        
        {'year': 2021, 'title': 'JWST Launch',
        'image': 'jwst.jpg',
        'desc': 'The James Webb Space Telescope launches to explore the early Universe. [NASA](https://science.nasa.gov/mission/webb/launch/),[NationalWorld](https://www.nationalworld.com/news/world/who-is-james-webb-nasa-telescope-named-after-space-agency-boss-what-did-he-do-and-who-named-the-telescope-3508502)'},
        
       ], "Present Time": [
        {'year': 2024, 'title': 'Gaia Black Hole Discovery ',
        'image': 'gaia_bh3.jpg',
        'desc': 'Gaia discovers a dormant stellar-mass black hole (BH3) only ~2,000 light-years from Earth, with a mass ≈ 33 M☉, revealing a hidden population of quiescent black holes. [ESA](https://www.esa.int/Science_Exploration/Space_Science/Gaia/Sleeping_giant_surprises_Gaia_scientists)'},

        {'year': 2024, 'title': 'Fastest-Growing Distant Black Hole',
        'image': 'fast_growing_quasar.jpg',
        'desc': 'Astronomers identify a distant quasar powered by a black hole swallowing the mass of a Sun per day — one of the most luminous and rapidly feeding black holes ever observed [Space](https://www.space.com/40596-fastest-growing-black-hole-found.html). '},

        {'year': 2024, 'title': 'First Exoplanet Imaged by JWST Coronagraph',
        'image': 'jwst_coronagraph_exoplanet.jpg',
        'desc': 'Using JWST’s MIRI coronagraph (mid-infrared), a previously unknown exoplanet is directly imaged in a debris disk [NASA](https://science.nasa.gov/blogs/webb/2022/09/01/nasas-webb-takes-its-first-ever-direct-image-of-distant-world/). '},
        {'year': 2025, 'title': 'Most Massive Black Hole Merger Detected (GW250114)',
        'image': 'gw2501124.png',
        'desc': 'LIGO/Virgo/KAGRA detect GW250114, a gravitational-wave event from a merger producing a black hole ~225 × solar mass. [Wikipedia](https://en.wikipedia.org/wiki/GW250114) '},

        {'year': 2024, 'title': 'Unusual Spin in Sagittarius A*',
        'image': 'sgrA_spin.jpg',
        'desc': 'Event Horizon Telescope observations show the Milky Way’s central black hole (Sgr A*) is spinning in a way that suggests a violent past, possibly from a major merger. [LiveScience](https://www.livescience.com/space/black-holes/the-milky-way-s-supermassive-black-hole-is-spinning-incredibly-fast-and-at-the-wrong-angle-scientists-may-finally-know-why) '},

        {'year': 2025, 'title': 'Jet Launch Near a Black Hole Edge',
        'image': 'blackhole_jet.jpg',
        'desc': 'Scientists observe a plasma jet moving at ~1/3 the speed of light and rapid X-ray fluctuations very close to the event horizon of an active black hole, revealing new physics at the brink. [Guardian](https://www.theguardian.com/science/2023/apr/26/astronomers-capture-first-image-of-jet-being-launched-from-edge-of-black-hole) '},

        {'year': 2025, 'title': 'Gaia: First Confirmed Exoplanet ',
        'image': 'gaia4b.jpg',
        'desc': 'Gaia-4 b, a massive exoplanet (~11.8 Mₙ), is confirmed via astrometric wobble + radial velocities. [Phys]’(https://phys.org/news/2025-02-high-precision-spectrograph-massive-exoplanet.html) '},

        {'year': 2025, 'title': 'The New Moons Around Saturn',
        'image': 'saturn_moons.jpg',
        'desc': 'Astronomers announce the discovery of 128 previously unknown moons of Saturn, bringing its total to 274 and giving information about its formation and ring evolution. [ZmeScience](https://www.zmescience.com/space/saturn-moon-king/) '}
       ]

        }

                 

        ERA_IMAGES = {
        "Prehistory and Ancient": {
            "file": "goseck_circle.jpg",
            "source": "AncientWisdom",
            "url": "http://www.ancient-wisdom.com/astronomy.htm"
        },
        "Classical and Renaissance": {
            "file": "Eliocentric.jpg",
            "source": "Wikipedia",
            "url": "https://it.wikipedia.org/wiki/Rivoluzione_astronomica"
        },
        "Eighteenth and Early Twentieth Century": {
            "file": "hubble_star.jpg",
            "source": "NASA",
            "url": "https://science.nasa.gov/missions/hubble/hubble-views-the-star-that-changed-the-universe/"
        },
        "Space Age": {
            "file": "space.jpg", 
            "url": "https://www.kpbs.org/news/arts-culture/2011/01/19/space-age-nasas-story-ground"
        },
        "Modern Epoch": {
            "file": "multiverse.jpg",
            "source": "SpaceShipEarth",
            "url": "https://spaceshipearth1.wordpress.com/modern-astronomy/"
        },
        "Present Time": {
            "file": "exo.jpg",
            "source": "NASA ",
            "url": "https://science.nasa.gov/exoplanets/"
        }
    }
        def format_desc_with_new_tab(text):
       
            # Pattern: [Gruppo 1](Gruppo 2) -> <a href="Gruppo 2" target="_blank">Gruppo 1</a>
            return re.sub(
                r'\[([^\]]+)\]\(([^\)]+)\)', 
                r'<a href="\2" target="_blank" style="text-decoration: underline; color: #60a5fa;">\1</a>', 
                text
            )
        def show_event_dialog(event):
            with ui.dialog() as dialog, ui.card().classes('w-96').props('role=dialog aria-modal=true aria-labelledby=event_title'):
                ui.label(f"**{event['year']} – {event['title']}**").classes('text-sm text-center w-40 text-white').props('tabindex=0 role=heading aria-level=3')
                aria_image(f"/discovery_images/{event['image']}", f"Image of {event['title']}").classes('w-full rounded-lg shadow')
                formatted_desc = format_desc_with_new_tab(event['desc'])
                info_box(formatted_desc).props('tabindex=0 role=document aria-label=Event description')
                aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            dialog.open()
        def start_history_quiz():
            
            questions = [{'era': k, 'data': v} for k, items in DISCOVERY_EVENTS.items() for v in items]
            random.shuffle(questions)
            
           
            q_state = {'idx': 0, 'score': 0, 'time': 0, 'correct': False}
            
            with ui.dialog() as q_dialog, ui.card().style('width: 600px; max-width: 95vw;').classes('p-6 items-center bg-slate-50').props('role=dialog aria-modal=true aria-labelledby=quiz_title'):
                aria_button('X', 'close', on_click=q_dialog.close).classes('absolute top-2 right-2 !bg-orange-500 hover:!bg-orange-700 text-white font-bold py-1 px-3 rounded z-50')
                
                ui.label('History of Astronomy Quiz').classes('text-2xl font-bold text-blue-600 mb-4 text-center').props('role=heading aria-level=2 tabindex=0')
                
                with ui.row().classes('w-full justify-between mb-4 bg-gray-200 p-3 rounded-lg border border-gray-300 shadow-inner'):
                    time_lbl = ui.label('Time: 0s').classes('text-lg text-black font-bold')
                    score_lbl = ui.label('Score: 0').classes('text-lg text-blue-700 font-bold')
                
                q_text = ui.label('').classes('text-xl font-semibold text-black text-center mb-6 h-16 flex items-center justify-center')
                
               
                with ui.grid(columns=2).classes('w-full gap-4') as btn_container:
                    for era in DISCOVERY_EVENTS.keys():
                        aria_button(era, f"Select {era}", on_click=lambda e=era: check_ans(e)).classes('!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-3 rounded shadow-md')
                
               
                with ui.row().classes('w-full justify-center gap-4 mt-6') as action_row:
                    explore_btn = aria_button("Explore Topic", "Read more", on_click=lambda: show_event_dialog(questions[q_state['idx']]['data'])).classes('!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded shadow-md')
                    next_btn = aria_button("Next Question ➔", "Next", on_click=lambda: next_question()).classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded shadow-md')
                
                def update_ui():
                    if q_state['idx'] < len(questions):
                        q_state['correct'] = False
                        evt_title = questions[q_state['idx']]['data']['title']
                        q_text.set_text(f"In which era did this occur: {evt_title}?")
                        explore_btn.set_visibility(False)
                        next_btn.set_visibility(False)
                    else:
                        q_text.set_text("Quiz Completed! Awesome job. 🎉")
                        btn_container.set_visibility(False)
                        explore_btn.set_visibility(False)
                        next_btn.set_visibility(False)
                        q_timer.deactivate()
                        accessible_notify('End Game! You completed the quiz.', color='positive', position='top', icon='emoji_events')
                
                def next_question():
                    q_state['idx'] += 1
                    update_ui()

                def check_ans(selected_era):
                    if q_state['correct']: return 
                    correct_era = questions[q_state['idx']]['era']
                    
                    if selected_era == correct_era:
                        q_state['score'] += 1
                        q_state['correct'] = True
                        score_lbl.set_text(f"Score: {q_state['score']}")
                        accessible_notify('Correct! Great job.', color='positive', position='top', icon='check_circle')
                        explore_btn.set_visibility(True)
                        next_btn.set_visibility(True)
                    else:
                        q_state['score'] -= 0.25
                        score_lbl.set_text(f"Score: {q_state['score']}")
                        accessible_notify('Wrong era! Try again.', color='negative', position='top', icon='cancel')
                
                def tick():
                    q_state['time'] += 1
                    time_lbl.set_text(f"Time: {q_state['time']}s")
                
                q_timer = ui.timer(1.0, tick)
                q_dialog.on('hide', q_timer.deactivate) 
                
                update_ui()
                q_dialog.open()
        def astronomy_timeline(container):
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            with container:
                with ui.dialog() as introd, ui.card().classes('p-4 w-full text-lg max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-labelledby=intro_title'):
                    html_info_box(r"""
<div style="font-family: sans-serif; line-height: 1.6;">
    
    <h3 style="margin-top: 0; color: #16a34a; border-bottom: 2px solid #22c55e; padding-bottom: 10px; margin-bottom: 15px;">1. Astronomical Milestones Quiz</h3>
    <p>Test your knowledge of the greatest discoveries in human history!</p>
    <ul style="margin-bottom: 25px;">
        <li>Press <b>Start Quiz</b> to begin the challenge. A timer will track how fast you can navigate through history.</li>
        <li>For each question, identify which <b>historical event or milestone</b> corresponds to the description provided.</li>
        <li><b>Scoring:</b> You earn <b>1 point</b> for every correct answer. A wrong choice costs a <b>-0.25 point</b> penalty.</li>
        <li><b>Interactive Learning:</b> When you answer correctly, you can use the <b>Explore Topic</b> button to see images and detailed scientific references about that discovery, then press <b>Next Question ➔</b> to continue.</li>
        <li>Complete all milestones to reach the "End Game" and stop the clock.</li>
    </ul>

    <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Timeline of Astronomical Discoveries</h3>
    <p>The history of astronomy is a journey through time, divided into different key eras:</p>
    <ul>
        <li><b>Prehistory and Ancient:</b> The earliest observations of the stars and planets.</li>
        <li><b>Classical and Renaissance:</b> The birth of modern physics and the heliocentric model.</li>
        <li><b>Eighteenth and Early Twentieth Century:</b> The discovery of new planets and the scale of the Milky Way.</li>
        <li><b>Space Age:</b> The era of satellites and the first steps beyond Earth.</li>
        <li><b>Modern Epoch and Present Time:</b> From the discovery of exoplanets to the latest deep-space missions.</li>
    </ul>
    <p>Select an <b>Era</b> on the main panel to explore specific events, analyze discoveries, and understand how our view of the cosmos has evolved.</p>
    
    <div style="border-left: 4px solid #1d4ed8; padding-left: 15px; margin-top: 20px; background-color: #f0f9ff; padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
        <p style="margin: 0; color: #1e40af; font-size: 0.95em;">
            <strong>Tip:</strong> If you find a question difficult, try exploring the eras in the timeline first to find the answer!
        </p>
    </div>
</div>
""")
                    aria_button("Close", "close the box",on_click=lambda:introd.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.row().classes('w-full gap-1 mb-4 items-center justify-center'):
                    
                    ui.label(
    "Trace the evolution of human discoveries of the cosmos, from ancient observations to the breakthroughs of modern astrophysics."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
                with ui.row().classes('w-full gap-1 mb-4 items-center justify-center'):
                    aria_button("Instructions", "Open introduction", on_click=lambda:introd.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button("Start Quiz", "Start interactive quiz", on_click=start_history_quiz).classes("!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded shadow-lg animate-pulse")
                    
                def load_era(era_name):
                    
                    selected_events = DISCOVERY_EVENTS[era_name]
                    
                 
                    with ui.dialog() as timeline_dialog, ui.card().style('width: fit-content; max-width: 98vw;').classes('p-6 flex flex-col items-center bg-slate-50 overflow-hidden shadow-none').props('role=dialog aria-modal=true aria-labelledby=timeline_title'):
                        
                        
                        aria_button('X', 'Close timeline', on_click=timeline_dialog.close).classes("absolute top-4 right-4 !bg-orange-500 hover:!bg-orange-700 text-white font-bold py-1 px-3 rounded z-50")
                        
                        ui.label(f"Era: {era_name}").classes('text-2xl font-bold text-blue-600 mb-4')
                        
                      
                        with ui.row().classes('w-full h-full items-start justify-center overflow-x-auto no-wrap p-4 gap-4 bg-gray-100 rounded-lg shadow-inner'):
                            for event in selected_events:
                                with ui.column().classes('items-center min-w-[100px] relative'): 
                                    ui.element('div').classes('absolute top-5 w-full h-1 bg-gray-600 -z-10')
                                    aria_button(str(event['year']), f"Open event: {event['title']}",
                                                on_click=lambda e=event: show_event_dialog(e)
                                                ).classes('!bg-blue-600 text-white hover:!bg-blue-800 transition-transform hover:scale-110 z-10')
                                    
                                    ui.label(event['title']).classes('text-sm text-center w-40 text-black leading-tight mt-2').props('tabindex=0 ')
                    
                    timeline_dialog.open()
              
                with ui.grid(columns=3).classes('w-full gap-6 mb-8 justify-center'):
                    for era_key in DISCOVERY_EVENTS.keys():
                        data =       ERA_IMAGES.get(era_key, {
                    "file": "goseck_circle.jpg", 
                    "source": "Unknown", 
                    "url": ""
                })
                       
                        bg_image = data.get("file", "goseck_circle.jpg")
                  
                        
                    
                        with ui.card().classes('w-full h-40 p-0 relative cursor-pointer overflow-hidden group hover:ring-4 ring-blue-500 transition-all shadow-lg rounded-xl').props(f'role=button tabindex=0 aria-label="Explore {era_key} era"').on('click', lambda k=era_key: load_era(k)).on('keydown.enter', lambda k=era_key: load_era(k)):
                            
                           
                            aria_image(f"/discovery_images/{bg_image}", f"Image of {era_key}").classes('w-full h-full object-cover absolute top-0 left-0 opacity-60 group-hover:opacity-40 group-hover:scale-110 transition-all duration-500')
                            
                           
                            ui.element('div').classes('absolute inset-0 bg-black/40 group-hover:bg-black/20 transition-colors')

                        
                            with ui.column().classes('absolute inset-0 items-center justify-center p-4'):
                                ui.label(era_key).classes('text-xl md:text-2xl font-bold text-white text-center drop-shadow-md group-hover:text-blue-200 transition-colors uppercase tracking-wide')

                                target_url = data.get("url", "")
                                source_text = data.get("source", "Source")

                                if target_url:
       
                                    with ui.link(target=target_url, new_tab=True).classes('absolute bottom-2 right-2 z-20 no-underline').on('click.stop', lambda: None):
                            
                                        ui.label(f"Ref: {source_text}").classes('text-[10px] font-mono text-gray-300 bg-black/60 px-2 py-1 rounded border border-gray-600 hover:bg-blue-600 hover:text-white hover:border-blue-400 transition-colors')

#panel cosmological timeline
        EPOCHS_EXTENDED = {"Before Big Bang":[

        {'name': 'Inflation', 'z_ref': 1e30, 'image': 'inflation.jpg',
        'desc': 'A rapid exponential expansion occurring ~10⁻³⁶ seconds after the Big Bang. [Wikipedia](https://en.wikipedia.org/wiki/Cosmic_inflation)'},

        {'name': 'Planck Era', 'z_ref': 1e32, 'image': 'planck_era.jpg',
        'desc': 'Quantum gravity dominates; no classical spacetime yet defined. [Astronomy](https://www.astronomy.com/science/the-planck-era-imagining-our-infant-universe/)'},

        {'name': 'Grand Unification Epoch', 'z_ref': 1e28, 'image': 'gut_epoch.jpg',
        'desc': 'Strong, weak, and electromagnetic forces may have been unified. [SoulWanderers](https://thesoulwanderers.blogspot.com/2017/12/quotes-of-wisdom-grand-unified-theories.html)'},

        {'name': 'Electroweak Symmetry Breaking', 'z_ref': 1e15, 'image': 'electroweak.png',
        'desc': 'The Higgs field activates; weak and electromagnetic forces separate. [QuantumDiaries](https://www.quantumdiaries.org/2011/11/21/why-do-we-expect-a-higgs-boson-part-i-electroweak-symmetry-breaking/)'},

        {'name': 'Quark–Gluon Plasma', 'z_ref': 1e12, 'image': 'qgp.jpg',
        'desc': 'The Universe is a hot dense soup of quarks and gluons. [NcatLab](https://ncatlab.org/nlab/show/quark-gluon+plasma)'},

        {'name': 'Hadron Epoch', 'z_ref': 1e10, 'image': 'hadron_epoch.png',
        'desc': 'Quarks combine into protons and neutrons [KevinBinz](https://kevinbinz.com/2016/06/26/cosmogenesis/).'},
        ],
        "Early Universe":[


        {'name': 'Big Bang Nucleosynthesis', 'z_ref': 1e9, 'image': 'bbn.jpg',
        'desc': 'Formation of light elements (H, He, D, Li). ~3 minutes after the Big Bang. [EinsteinOnline](https://www.einstein-online.info/en/spotlight/bbn/)'},

        {'name': 'Photon Epoch', 'z_ref': 1e7, 'image': 'photon_epoch.jpg',
        'desc': 'Universe dominated by radiation; matter remains ionized. [UniOregon](https://pages.uoregon.edu/jimbrau/astr123/Notes/Chapter27.html)'},

        {'name': 'Recombination ', 'z_ref': 1100, 'image': 'recomb.jpeg',
        'desc': 'Electrons bind with nuclei to form atoms [Telescoper](https://telescoper.blog/tag/recombination/)'},   
        
        {'name':  'CMB', 'z_ref': 1100, 'image': 'cmb.jpg',
        'desc': 'The CMB is released (380,000 years) [DarkMatter](https://darkmattercrisis.wordpress.com/2013/04/22/the-planck-results-on-the-cosmic-microwave-background/).'},  
        

        {'name': 'Dark Ages', 'z_ref': 200, 'image': 'dark_ages.jpg',
        'desc': 'No stars yet exist; only dark matter and cooling gas fill the Universe [EarthSky](https://earthsky.org/space/cosmic-dark-ages-lyman-alpha-galaxies-lager/).'},
        ],
         "Structures Formation":[

        {'name': 'First Stars (Pop III)', 'z_ref': 30, 'image': 'first_stars.jpg',
        'desc': 'The first generation of massive, metal-poor stars ignite. [Astronomy](https://www.astronomy.com/science/first-stars-in-the-universe-werent-lonely/)'},

        {'name': 'First Galaxies', 'z_ref': 15, 'image': 'first_galaxies.jpg',
        'desc': 'Minihalos collapse to form the earliest protogalaxies [NASA](https://science.nasa.gov/mission/webb/early-universe/).'},

        {'name': 'Reionization (Midpoint)', 'z_ref': 8, 'image': 'reionization.jpg',
        'desc': 'Radiation from early galaxies begins reionizing hydrogen [Caltech](https://ned.ipac.caltech.edu/level5/March19/Wise/Wise2.html).'},

        {'name': 'End of Reionization', 'z_ref': 6, 'image': 'end_reionization.png',
        'desc': 'The intergalactic medium becomes fully ionized. [Wikipedia](https://en.wikipedia.org/wiki/Reionization)'},

        {'name': 'Cosmic Noon (Peak SFR)', 'z_ref': 2, 'image': 'cosmic_noon.jpg',
        'desc': 'Peak star formation across the Universe (~10 billion years ago) [SpaceAustralia](https://spaceaustralia.com/news/puffy-galaxies-more-productive-after-cosmic-noon).'},
         ],
         "Cosmic Web":[
        {'name': 'Peak AGN Activity', 'z_ref': 2.5, 'image': 'agn_peak.png',
        'desc': 'Supermassive black holes undergo their most intense growth. [Acronomy](https://acronomy.lw1.at/acronym/agn)'},

        {'name': 'Galaxy Clusters Form', 'z_ref': 1, 'image': 'cluster_formation.png',
        'desc': 'Large-scale structures like the Local Group assemble. [Harvard](https://www.cfa.harvard.edu/research/topic/galaxy-clusters)'},

        {'name': 'Modern Universe', 'z_ref': 0.5, 'image': 'modern_universe.jpg',
        'desc': 'Galaxy groups, clusters, and filaments take their present form. [AmericanScientidt](https://www.americanscientist.org/article/modern-cosmology-science-or-folktale)'},

        {'name': 'Today', 'z_ref': 0, 'image': 'today.png',
        'desc': 'The present Universe: age ≈ 13.8 billion years. [ESA](https://esahubble.org/images/heic2003b/)'},
         ]
        }

        COSMIC_METADATA = {
    "Before Big Bang": {
        "file": "bigbang.jpg",
        "source": "News",
        "url": "https://www.9news.com.au/world/more-secrets-of-moments-after-big-bang-being-understood-through-recreation-and-new-data/f98c2b0b-6efe-414e-9713-05b9bc18578d"
    },
    "Early Universe": {
        "file": "early.jpg",
        "source": "AstronomySwin",
        "url": "https://astronomy.swin.edu.au/cosmos/e/epoch+of+recombination"
    },
    "Structures Formation": {
        "file": "first.jpg", 
        "source": "LiveScience",
        "url": "https://www.livescience.com/space/cosmology/the-early-universe-is-nothing-like-we-expected-james-webb-telescope-reveals-new-understanding-of-how-galaxies-formed-at-cosmic-dawn"
    },
   
    "Cosmic Web": {
        "file": "now.jpg",
        "source": "Harvard",
        "url": "https://www.cfa.harvard.edu/news/cosmic-evolution-galaxies"
    }
}
        def z_to_age_gyr(z):
            if z <= 0:
                return cosmo.age(0).value
            return cosmo.age(z).value
        def z_to_time_seconds(z):
        
            if z <= 1100:
                # età in Gyr → secondi
                t_gyr = z_to_age_gyr(z)
                return t_gyr * 3.1536e16  
            else:
            
                return float('nan')

        def show_epoch_dialog(epoch):
            z = min(epoch['z_ref'], 1100)
            age = z_to_age_gyr(z)
            

            
            with ui.dialog() as dialog, ui.card().classes('w-96').props('role=dialog aria-modal=true aria-labelledby=epoch_title'):
                ui.label(f"**{epoch['name']} (z={epoch['z_ref']:.2e})**").classes('text-sm text-center text-white').props('tabindex=0 role=heading aria-level=3')
                time_sec = z_to_time_seconds(z)
                if not math.isnan(time_sec):
                    ui.label(f"Universe age: {age:.2e} Gyr ({time_sec:.2e} s)").classes('text-sm text-center text-white').props('tabindex=0 ')
                else:
                    ui.label(f"Universe age: {age:.2e} Gyr (time in seconds not defined for theoretical epochs)").props('tabindex=0 ')

                aria_image(f"/cosmic_epochs/{epoch['image']}", f"Image of {epoch['name']} epoch").classes('w-full rounded-lg shadow')
                formatted_desc = format_desc_with_new_tab(epoch['desc'])
                info_box(formatted_desc).props('tabindex=0 aria-label=Epoch description')
                aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            dialog.open()
        

        def start_cosmic_quiz():
    
            questions = [{'era': k, 'data': v} for k, items in EPOCHS_EXTENDED.items() for v in items]
            random.shuffle(questions)
            
            q_state = {'idx': 0, 'score': 0, 'time': 0, 'correct': False}
            
            with ui.dialog() as q_dialog, ui.card().style('width: 600px; max-width: 95vw;').classes('p-6 items-center bg-gray-900 border border-gray-700').props('role=dialog aria-modal=true aria-labelledby=quiz_title'):
                aria_button('X', 'close', on_click=q_dialog.close).classes('absolute top-2 right-2 !bg-orange-500 hover:!bg-orange-700 text-white font-bold py-1 px-3 rounded z-50')
                
                ui.label('Cosmological Epochs Quiz').classes('text-2xl font-bold text-green-400 mb-4 text-center drop-shadow-md').props('role=heading aria-level=2 tabindex=0')
                
                with ui.row().classes('w-full justify-between mb-4 bg-gray-800 p-3 rounded-lg border border-gray-600 shadow-inner'):
                    time_lbl = ui.label('Time: 0s').classes('text-lg text-white font-bold')
                    score_lbl = ui.label('Score: 0').classes('text-lg text-blue-400 font-bold').props('aria-live=polite')
                
                q_text = ui.label('').classes('text-xl font-semibold text-gray-100 text-center mb-6 h-16 flex items-center justify-center')
                
                with ui.grid(columns=2).classes('w-full gap-4') as btn_container:
                    for era in EPOCHS_EXTENDED.keys():
                        aria_button(era, f"Select {era}", on_click=lambda e=era: check_ans(e)).classes('!bg-gray-700 hover:!bg-blue-600 border border-gray-500 text-white font-bold py-3 rounded shadow-lg transition-colors')
                
                with ui.row().classes('w-full justify-center gap-4 mt-6') as action_row:
                    explore_btn = aria_button("Explore Topic", "Read more", on_click=lambda: show_epoch_dialog(questions[q_state['idx']]['data'])).classes('!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded shadow-md')
                    next_btn = aria_button("Next Question ➔", "Next", on_click=lambda: next_question()).classes('!bg-green-600 hover:!bg-green-800 text-white font-bold py-2 px-4 rounded shadow-md')
                
                def update_ui():
                    if q_state['idx'] < len(questions):
                        q_state['correct'] = False
                        evt_name = questions[q_state['idx']]['data']['name']
                        q_text.set_text(f"Which phase does this belong to: {evt_name}?")
                        explore_btn.set_visibility(False)
                        next_btn.set_visibility(False)
                    else:
                        q_text.set_text("Quiz Completed! The Universe has no secrets for you. 🌌")
                        btn_container.set_visibility(False)
                        explore_btn.set_visibility(False)
                        next_btn.set_visibility(False)
                        q_timer.deactivate()
                        accessible_notify('End Game! You completed the quiz.', color='positive', position='top', icon='emoji_events')

                def next_question():
                    q_state['idx'] += 1
                    update_ui()

                def check_ans(selected_era):
                    if q_state['correct']: return
                    correct_era = questions[q_state['idx']]['era']
                    
                    if selected_era == correct_era:
                        q_state['score'] += 1
                        q_state['correct'] = True
                        score_lbl.set_text(f"Score: {q_state['score']}")
                        accessible_notify('Correct! Great job.', color='positive', position='top', icon='check_circle')
                        explore_btn.set_visibility(True)
                        next_btn.set_visibility(True)
                    else:
                        q_state['score'] -= 0.25
                        score_lbl.set_text(f"Score: {q_state['score']}")
                        accessible_notify('Wrong phase! Try again.', color='negative', position='top', icon='cancel')
                
                def tick():
                    q_state['time'] += 1
                    time_lbl.set_text(f"Time: {q_state['time']}s")
                
                q_timer = ui.timer(1.0, tick)
                q_dialog.on('hide', q_timer.deactivate)
                
                update_ui()
                q_dialog.open()
        def cosmic_timeline(container):
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            def get_button_color(epoch_name):
          
                theoretical_list = [
                    'Inflation', 'Planck Era', 'Grand Unification Epoch', 
                    'Electroweak Symmetry Breaking', 'Quark–Gluon Plasma', 
                    'Hadron Epoch'
                ]
                if epoch_name in theoretical_list:
                    return '!bg-green-600 hover:!bg-green-800'
                return '!bg-blue-600 hover:!bg-blue-800'
            with container:
                with ui.dialog() as introc, ui.card().classes('p-4 w-full text-lg max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-labelledby:intro_title'):
                    html_info_box(r"""
<div style="font-family: sans-serif; line-height: 1.6;">
    
    <h3 style="margin-top: 0; color: #16a34a; border-bottom: 2px solid #22c55e; padding-bottom: 10px; margin-bottom: 15px;">1. Cosmological Epochs Quiz</h3>
    <p>Test your knowledge of the history of the Universe with this interactive challenge!</p>
    <ul style="margin-bottom: 25px;">
        <li>Press <b>Start Quiz</b> to begin the journey from the Big Bang to the present day. A timer will track your performance.</li>
        <li>For each question, identifying the correct cosmological phase related to the cosmic event described.</li>
        <li><b>Scoring:</b> You earn <b>1 point</b> for every correct answer. A wrong choice results in a <b>-0.25 point</b> penalty.</li>
        <li><b>Interactive Learning:</b> When you answer correctly, two buttons will appear:
            <ul>
                <li><b>Explore Topic:</b> Opens a detailed window with images and scientific references about that specific event.</li>
                <li><b>Next Question ➔:</b> Proceeds to the next stage of cosmic history.</li>
            </ul>
        </li>
        <li>Finish all events to stop the timer and complete the "End Game".</li>
    </ul>

    <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Timeline of Universe Evolution</h3>
    <p>The history of the universe can be divided into different cosmological epochs: Before Big Bang, Early Universe, Structures Formation, and Cosmic Web.</p>
    <p>Theoretical epochs are phases predicted by cosmological models but not directly observable, while observational epochs are supported by empirical evidence.</p>
    <p>The redshift (z) indicates how much the universe has expanded since that epoch; higher z values correspond to earlier times.</p>
    <p>The age of the universe at each epoch is given in gigayears (1Gyr=10^9 years) and seconds (s) where applicable.</p>
    <p>The Big Bang marks the beginning of the observable universe approximately 13.8 billion years ago (z → ∞).</p>
    <p>Select an epoch on the main panel to explore the universe evolution and analyze each phase.</p>
    
    <hr style="margin: 20px 0; border: 0; border-top: 1px solid #ccc;">

    <div style="font-size: 0.9em; background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;">
        <strong style="display: block; margin-bottom: 8px; color: #475569;">Map Legend</strong>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <span style="display: flex; align-items: center;">
                <span style="margin-right: 6px; font-size: 1.2em;">🟩</span> Theoretical epochs (not directly observable)
            </span>
            <span style="display: flex; align-items: center;">
                <span style="margin-right: 6px; font-size: 1.2em;">🟦</span> Observational epochs (supported by evidence)
            </span>
        </div>
    </div>
</div>
""")
                    aria_button("Close", "close the box",on_click=lambda:introc.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.row().classes('w-full items-center justify-center '):
                    ui.label(
    "Explore the chronology of the Universe from the Big Bang to the present day, witnessing the major epochs of cosmic evolution."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
              
                with ui.row().classes('w-full gap-1 mb-4 items-center justify-center'):
                    aria_button("Instructions", "Open introduction", on_click=lambda:introc.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button("Start Quiz", "Start interactive quiz", on_click=start_cosmic_quiz).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded shadow-lg animate-pulse")
                    
                        
             
                        



                def load_cosmic_era(era_name):
                    selected_epochs = EPOCHS_EXTENDED.get(era_name, [])
                    
                   
                    with ui.dialog() as timeline_dialog2, ui.card().style('width: fit-content; max-width: 98vw;').classes('p-6 flex flex-col items-center bg-gray-900 overflow-hidden shadow-none').props('role=dialog aria-modal=true aria-labelledby=timeline_title'):
                        
                     
                        aria_button('X', 'Close timeline', on_click=timeline_dialog2.close).classes("absolute top-4 right-4 !bg-orange-500 hover:!bg-orange-700 text-white font-bold py-1 px-3 rounded z-50")
                        
                        ui.label(f"Phase: {era_name}").classes('text-2xl font-bold text-blue-400 mb-4 animate-pulse')
                        
                        with ui.row().classes('w-full h-full items-start justify-center overflow-x-auto no-wrap p-6 gap-6 bg-gray-800 rounded-xl shadow-2xl border border-gray-700'):
                            for epoch in selected_epochs:
                                with ui.column().classes('items-center min-w-[140px] relative group'):
                                    btn_color = get_button_color(epoch['name'])
                            
                                    aria_button(epoch['name'], f"Open epoch: {epoch['name']}",
                                        on_click=lambda e=epoch: show_epoch_dialog(e)
                                        ).classes(f'{btn_color} text-white text-xs font-bold w-24 h-24 rounded-full shadow-lg border-4 border-gray-700 hover:scale-110 transition-transform z-10 text-center break-words p-1 flex items-center justify-center')

                                    ui.label(f"z ≈ {epoch['z_ref']:.2e}").classes('text-md text-center text-gray-300 mt-2').props('tabindex=0 ')

                    timeline_dialog2.open()
                with ui.grid(columns=2).classes('w-full items-center justify-center gap-6 mb-8'):
            
                    for era_key in EPOCHS_EXTENDED.keys():
                        
                     
                        data = COSMIC_METADATA.get(era_key, {
                            "file": "now.jpg", 
                            "source": "Unknown", 
                            "url": ""
                        })
                        bg_image = data["file"]
                        
                       
                        with ui.card().classes('w-full h-40 p-0 relative cursor-pointer overflow-hidden group hover:ring-4 ring-green-500 transition-all shadow-lg rounded-xl').props(f'role=button tabindex=0 aria-label="Explore {era_key} phase"').on('click', lambda k=era_key: load_cosmic_era(k)).on('keydown.enter', lambda k=era_key: load_cosmic_era(k)):
                            
                         
                            aria_image(f"/cosmic_epochs/{bg_image}", f"Image of {era_key}").classes('w-full h-full object-cover absolute top-0 left-0 opacity-60 group-hover:opacity-40 group-hover:scale-110 transition-all duration-700')
                            
                           
                            ui.element('div').classes('absolute inset-0 bg-gradient-to-t from-black/90 to-transparent')

                        
                            with ui.column().classes('absolute inset-0 items-center justify-center p-4'):
                                ui.label(era_key).classes('text-xl md:text-2xl font-bold text-white text-center drop-shadow-lg uppercase tracking-wider group-hover:text-green-300 transition-colors')

                          
                            if data["url"]:
                             
                                with ui.link(target=data["url"], new_tab=True).classes('absolute bottom-2 right-2 z-20 no-underline').on('click.stop', lambda: None):
                                    
                                    ui.label(f"Ref: {data['source']}").classes('text-[10px] font-mono text-gray-300 bg-black/60 px-2 py-1 rounded border border-gray-600 hover:bg-green-600 hover:text-white hover:border-green-400 transition-colors')
                
                
            
        
    
#panel stellar evolution
        stellar_nodes = [
            # Root
            {
                "id": "stellar_nebula",
                "label": "Stellar Nebula",
                "desc": "A vast, cold cloud of gas (primarily **hydrogen** and **helium**) and cosmic dust. "
                "\n\nNew stars form when regions of this cloud collapse under their own **gravity**,"
                "\n\ninitiating the process of star formation. This collapse creates a dense protostar.",
                "color": "#6A0DAD",
                "shape": "diamond"
            },

            # --- Average Star Branch (Stars up to ~8 Solar Masses) ---
            
            {
                "id": "average_star",
                "label": "Avg-Mass Star(MS)",
                "desc": "A medium-mass star (like the Sun). It achieves hydrostatic equilibrium and "
                "\n\nremains stable by fusing **hydrogen into helium** in its core via the proton-proton chain. "
                "\n\nThis is the longest phase of a star's life.",
                "color": "#FFD700",
                "shape": "circle"
            },
            {
                "id": "red_giant",
                "label": "Red Giant",
                "desc": "After exhausting core hydrogen, the core contracts while **hydrogen shell burning** begins,"
                "\n\ncausing the outer layers to expand and cool. The star increases in size and luminosity, "
                "\n\nmoving toward the upper-right of the H-R diagram.",
                "color": "#FF4500",
                "shape": "circle"
            },
            {
                "id": "planetary_nebula",
                "label": "Planetary Nebula",
                "desc": "As the star's outer layers are expelled due to thermal pulses,"
                "\n\nthey form a **glowing shell of ionized gas** expanding away from the core. "
                "\n\nThis phase is relatively brief and enriches the interstellar medium with heavier elements.",
                "color": "#00CED1",
                "shape": "circle"
            },
            {
                "id": "white_dwarf",
                "label": "White Dwarf",
                "desc": "The exposed, hot, inert core of the former star, composed mainly of carbon and oxygen."
                "\n\nIt is supported against collapse by **electron degeneracy pressure** and"
                "\n\nslowly cools over billions of years, eventually becoming a black dwarf.",
                "color": "#F0F8FF",
                "shape": "circle"
            },

            # --- Massive Star Branch (Stars > ~8 Solar Masses) ---

            {
                "id": "massive_star",
                "label": "Massive Star(MS)",
                "desc": "A star with significantly greater mass than the Sun."
                "\n\nIt burns fuel much faster and hotter due to higher core pressure, primarily fusing hydrogen via the **CNO cycle**. "
                "\n\nThese stars have short, brilliant lifetimes.",
                "color": "#1E90FF",
                "shape": "circle"
            },
            {
                "id": "red_supergiant",
                "label": "Red Supergiant",
                "desc": "An enormous, highly luminous star that has exhausted its core hydrogen. "
                "\n\nFusion proceeds in a series of shells (e.g., carbon, neon, oxygen, silicon) until the core is predominantly **iron**. "
                "\n\nThis is the stage immediately preceding core collapse.",
                "color": "#8B0000",
                "shape": "circle"
            },
            {
                "id": "supernova",
                "label": "Supernova(II)",
                "desc": "The cataclysmic, violent explosion that occurs when the iron core collapses, leading to a massive rebound shockwave. "
                "\n\nThis process is responsible for creating and dispersing elements heavier than **iron** (e.g., gold and uranium) into space.",
                "color":"#FFD700",
                "shape": "star"
            },
            {
                "id": "neutron_star",
                "label": "Neutron Star",
                "desc": "The ultra-compact remnant of a supernova (if the remnant mass is below ~3 solar masses). "
                "\n\nIt is supported by **neutron degeneracy pressure**, with extreme density. Many are observed as pulsars.",
                "color":"#708090",
                "shape": "circle"
            },
            {
                "id": "black_hole",
                "label": "Black Hole",
                "desc": "BH has only mass,angular momentum and eventually charge, "
                "\n\nit is formed when the core remnant is so massive (exceeding the Tolman–Oppenheimer–Volkoff limit) "
                "\n\nthat its gravity overcomes all internal pressure, leading to infinite density at the singularity. "
                "\n\nIts gravitational pull is so immense that nothing, not even light, can escape the **event horizon**.",
                "color": "#000000",
                "shape": "circle"
            },
        ]



        stellar_edges = [
            # Branch 1: Average star
            ("stellar_nebula", "average_star"),
            ("average_star", "red_giant"),
            ("red_giant", "planetary_nebula"),
            ("planetary_nebula", "white_dwarf"),

            # Branch 2: Massive star
            ("stellar_nebula", "massive_star"),
            ("massive_star", "red_supergiant"),
            ("red_supergiant", "supernova"),
            ("supernova", "neutron_star"),
            ("supernova", "black_hole"),
        ]

        def plot_star_graph(title, nodes, edges, height=600, width=600):
        

            G = nx.DiGraph()

        
            for n in nodes:
                G.add_node(n['id'], **n)

        
            for e in edges:
                if len(e) == 3:
                    src, tgt, data = e
                    G.add_edge(src, tgt, **data)
                else:
                    src, tgt = e
                    G.add_edge(src, tgt)

        
            roots = [n for n, d in G.nodes(data=True) if d.get("shape") in ("diamond", "square")]
            if len(roots) != 1:
                raise ValueError("Serve un singolo nodo root (shape diamond o square).")
            root = roots[0]

        
            root_children = list(G.successors(root))
        
            if not root_children:
                pos = {root: (0.5, 1.0)}
            else:
            
                n_children = len(root_children)
        
                split = (n_children + 1) // 2

                for i, child in enumerate(root_children):
                    if 'side' not in G.nodes[child]:
                        G.nodes[child]['side'] = 'left' if i < split else 'right'

            
                queue = [root]
                while queue:
                    cur = queue.pop(0)
                    cur_side = G.nodes[cur].get('side', None)
                    for ch in G.successors(cur):
                    
                        if 'side' not in G.nodes[ch] and cur_side is not None:
                            G.nodes[ch]['side'] = cur_side
                        queue.append(ch)

                depth = {root: 0}
                queue = [root]
                while queue:
                    cur = queue.pop(0)
                    for ch in G.successors(cur):
                    
                        if ch not in depth:
                            depth[ch] = depth[cur] + 1
                            queue.append(ch)

            
                from collections import defaultdict
                nodes_by_side_depth = defaultdict(lambda: defaultdict(list))
                nodes_by_depth = defaultdict(list)

                for nnode, d in depth.items():
                    side = G.nodes[nnode].get('side', 'left')  
                    nodes_by_side_depth[side][d].append(nnode)
                    nodes_by_depth[d].append(nnode)

            
                x_root = 0.5
                y_root = 1.0
                x_left = 0.18
                x_right = 0.82
                y_level_gap = 0.18   
                y_within_gap = 0.08  

        
                max_depth = max(depth.values()) if depth else 0

                pos = {}
                pos[root] = (x_root, y_root)

            
                for d in range(1, max_depth + 1):
                
                    #left_nodes = nodes_by_side_depth['left'].get(d, [])
                    #right_nodes = nodes_by_side_depth['right'].get(d, [])
                    current_nodes = nodes_by_depth[d]

                    left_nodes = []
                    right_nodes = []

                    for parent in nodes_by_depth[d-1]:    
                        children = list(G.successors(parent))

                        if len(children) == 2:
                            left_nodes.append(children[0])
                            right_nodes.append(children[1])  

                        else:
                        
                            for c in children:
                                side = G.nodes[c].get("side", "left")
                                if side == "left":
                                    left_nodes.append(c)
                                else:
                                    right_nodes.append(c)


                    y_base = y_root - d * y_level_gap

            
                    def assign_positions_for_side(node_list, x_pos):
                        n = len(node_list)
                        if n == 0:
                            return
                        
                        total_span = (n - 1) * y_within_gap
                        start = y_base + total_span / 2.0
                        for i, nodeid in enumerate(node_list):
                            ynode = start - i * y_within_gap
                            pos[nodeid] = (x_pos, ynode)

                    assign_positions_for_side(left_nodes, x_left)
                    assign_positions_for_side(right_nodes, x_right)

                for nnode in G.nodes():
                    if nnode not in pos:
                
                        sd = depth.get(nnode, 1)
                        side = G.nodes[nnode].get('side', 'left')
                        x = x_left if side == 'left' else x_right
                        y = y_root - sd * y_level_gap
                    
                        while any(abs(px - x) < 1e-6 and abs(py - y) < 1e-6 for px, py in pos.values()):
                            y -= 0.02
                        pos[nnode] = (x, y)

                if 'supernova' in pos and 'neutron_star' in pos and 'black_hole' in pos:
                    x_sn, y_sn = pos['supernova']
                    
                    
                    new_y = y_sn - 0.12 
                    
                  
                    pos['neutron_star'] = (x_sn - 0.12, new_y)
                    pos['black_hole'] = (x_sn + 0.12, new_y)
            edge_x = []
            edge_y = []
            edge_hovertext = []
            annotations = []

            for s, t, data in G.edges(data=True):
                if s not in pos or t not in pos:
                
                    continue

                x0, y0 = pos[s]
                x1, y1 = pos[t]

                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

                interaction = data.get("interaction", "")
                desc = data.get("desc", "")
                hover = f"<b>{interaction}</b><br>{G.nodes[s].get('label', s)} → {G.nodes[t].get('label', t)}<br>{desc}"
                edge_hovertext.append(hover)

                annotations.append(
                    dict(
                        x=x1, y=y1, ax=x0, ay=y0,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, 
                        arrowhead=2,      
                        arrowsize=1.5,    
                        arrowwidth=2,   
                        arrowcolor="#555",
                        standoff=8,       
                        startstandoff=8, 
                        text=interaction, 
                        opacity=1.0
                    )
                )

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color="#555"),
                mode="lines",
                hoverinfo="text",
                hovertext=[h for h in edge_hovertext for _ in (0,1,2)],
                showlegend=False
            )

    
            node_x = []
            node_y = []
            node_text = []
            labels = []
            colors = []
            symbols = []
            sizes = []

        
            sorted_nodes = sorted(pos.items(), key=lambda kv: (-kv[1][1], kv[1][0]))

            for node, (x, y) in sorted_nodes:
                info = G.nodes[node]

                node_x.append(x)
                node_y.append(y)
                labels.append(info.get('label', node))
                #desc_html = info.get('desc', '').replace('\n\n', '<br><br>').replace('\n', '<br>')
                #node_text.append(f"<b>{info.get('label','')}</b><br>{desc_html}")
                raw_desc = info.get('desc', '')
                
             
                clean_desc = raw_desc.replace('\n', ' ')
                
         
                wrapped_lines = textwrap.wrap(clean_desc, width=40) 
                desc_html = "<br>".join(wrapped_lines)
                
                node_text.append(f"<b>{info.get('label','')}</b><br><br>{desc_html}")
                
                colors.append(info.get('color', '#2b7de9'))
                symbols.append(info.get('shape', 'circle'))
                sizes.append(44 if info.get('shape') == 'diamond' else 30)
               

            line_colors = ['#333' if c != '#000000' else '#666' for c in colors] 

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=labels,
                textposition="bottom center", 
                textfont=dict(size=14, color='black', family="Arial Black"),
                hoverinfo="text",
                hovertext=node_text,
            
                cliponaxis=False, 
                marker=dict(
                    color=colors, 
                    size=sizes, 
                    symbol=symbols, 
                    line=dict(width=2, color=line_colors),
                    opacity=1.0,
                    gradient=dict(type='radial', color='white')
                ),
                showlegend=False
            )
            fig = go.Figure(data=[edge_trace, node_trace])

            fig.update_layout(
                title={
                    'text': f"<b>{title}</b>",   
                    'y': 0.95,                  
                    'x': 0.5,                    
                    'xanchor': 'center',      
                    'yanchor': 'top',
                    'font': {
                        'size': 24,              
                        'color': 'black'         
                    }
                },
                title_x=0.5,
                width=width, height=height,
                hovermode="closest",
                margin=dict(b=30, l=30, r=30, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=annotations,
                hoverlabel=dict(
    align="left", 
    bgcolor="white", 
    bordercolor="black",
namelength=-1,
    font=dict(size=12) 
    )
            )

            #ui.plotly(fig).style(f"height:{height}px; width:600px")
            ui.plotly(fig).classes('w-full').style(f"height:{height}px").props('tabindex=0 aria-label="Stellar evolution graph with nodes and edges"')




        def show_star_dialog(df, i):
            row = df.iloc[i]
            with ui.dialog() as dialog, ui.card().classes('w-96').props('role=dialog aria-modal=true aria-labelledby=star_title'):
                ui.label(f"Star: {row['source_id']}").classes('text-sm text-center').props('tabindex=0 aria-label=Star ID')
                ui.label(f"RA: {row['ra']:.5f}, DEC: {row['dec']:.5f}").props('tabindex=0 aria-label=Star coordinates')
                #aria_image(row['image_url'], f"Image of star {row['source_id']}").classes('w-full rounded shadow')
            
                info_box(
                    f"**Phase:** {row['stellar_phase']}  \n"
                    f"**Bp−Rp:** {row['bp_rp']:.3f}  \n"
                    f"**Absolute G Magnitude:** {row['abs_mag']:.2f}"
                ).props('tabindex=0 aria-label=Star details')
                aria_button("Close",'Close dialog', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            dialog.open()
            
        def assign_phase(df):
            
                df = df.sort_values(['star_mass','log_Teff'], ascending=[True, False])
                df['phase'] = 'MS'  # default Main Sequence

                # White Dwarfs
                df.loc[
        (df['log_Teff'] > 4.25) &   #
        (df['log_L'] < 0.0),       
        'phase'
    ] = 'WD'

                # RGB
                df.loc[(df['log_L'] >= 1.0) & (df['log_Teff'] < 3.9), 'phase'] = 'RGB'

                # Helium Burning (HeB)
                df.loc[(df['log_L'] >= 1.0) & (df['log_L'] <= 3.5) & (df['log_Teff'] >= 3.7) & (df['log_Teff'] <= 4.1), 'phase'] = 'HeB'

                # AGB
                df.loc[(df['log_L'] > 3.0) & (df['log_Teff'] < 3.8), 'phase'] = 'AGB'
                
                df.loc[(df['log_L'] > 0.2) & (df['log_L'] < 1.0) & (df['log_Teff'] < 3.9), 'phase'] = 'SGB'

                return df
            
        def refine_phase_with_gaia(df):
        
            refined = df['stellar_phase'].copy()
            
            bp_rp = df['bp_rp']
            abs_g = df['abs_mag']

            
            is_wd = (abs_g > 10) & (abs_g > 2.5 * bp_rp + 8.5)
            refined[is_wd] = 'WD'

        
            is_sgb_region = (abs_g >= 3.0) & (abs_g <= 4.5) & (bp_rp > 0.6) & (bp_rp < 1.4)
            
            mask_force_sgb = is_sgb_region & refined.isin(['MS', 'SGB'])
            refined[mask_force_sgb] = 'SGB' 

        
            is_bright_red = (abs_g < 3.0) & (bp_rp > 0.7)
            
            mask_fix_giants = is_bright_red & (refined.isin(['MS', 'HeB', 'AGB']))
            refined[mask_fix_giants] = 'RGB' 


            is_faint = (abs_g > 5.0) & (~is_wd)
            
        
            mask_fix_ms = is_faint & (refined.isin(['RGB', 'AGB', 'HeB', 'SGB']))
            refined[mask_fix_ms] = 'MS'

            return refined
        @lru_cache(maxsize=1)
        def load_isochrones_cached(mist_iso_folder):
            iso_list = []
            for file in os.listdir(mist_iso_folder):
                if file.endswith(".txt"):
                    iso = pd.read_csv(os.path.join(mist_iso_folder, file), sep=r"\s+")

                    if {'Gaia_BP_MAWb','Gaia_RP_MAW','Gaia_G_MAW'}.issubset(iso.columns):
                        iso = assign_phase(iso)
                        iso['color'] = iso['Gaia_BP_MAWb'] - iso['Gaia_RP_MAW']
                        iso['abs_mag'] = iso['Gaia_G_MAW']
                        iso = iso[np.isfinite(iso['color']) & np.isfinite(iso['abs_mag'])]
                        iso_list.append(iso[['color','abs_mag','phase','star_mass','log_L','log_Teff']])

            iso_df = pd.concat(iso_list, ignore_index=True)
            tree = cKDTree(iso_df[['color','abs_mag']].values)


            return iso_df, tree

    
       
        
        
        @lru_cache(maxsize=1)
        def load_gaia_cached(csv_path):
            df = pd.read_csv(csv_path, dtype={'source_id': str})
            df = df.dropna(subset=['bp_rp', 'phot_g_mean_mag', 'parallax', 'ra', 'dec'])
            df = df[df['parallax'] > 0]
            df['abs_mag'] = df['phot_g_mean_mag'] + 5*np.log10(df['parallax']/1000) + 5
            M_sun_G = 4.67  
            df['log_L'] = 0.4*(M_sun_G - df['abs_mag'])
            df['log_Teff'] = 3.7 - 0.25*(df['bp_rp'] - 0.5)  
            return df

        

        def hr_diagram_page(container, gaia_csv_path: str,sample_size=20000):
        
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            with container:
                ui.label(
    "Analyze the life cycle of stars using the Hertzsprung-Russell diagram to understand stellar evolution and classification."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
               
                
                with ui.dialog() as cur_intro, ui.card().classes('p-4 w-full max-w-[600px]').props('role=dialog aria-modal=true aria-labelledby=curiosities_title'):
                    html_info_box(r"""
                    <h3>Stars vs. Sand</h3>
                    <p>Did you know that there are more stars in the observable Universe than there are grains of sand on all the beaches on Earth?</p>
                    <p>Astronomers estimate there are about <b>10<sup>24</sup> stars</b>. That is a 1 followed by 24 zeros! Yet, the Universe is mostly empty space.</p>
                    """)
                    
                    reference_box("**Source:** [ESA - How many stars are there?](https://www.esa.int/Science_Exploration/Space_Science/Herschel/How_many_stars_are_there_in_the_Universe)")
                    html_info_box(r"""
                    <h3>Stars Made of Diamond</h3>
                    <p>When a low-mass star like our Sun dies, it becomes a <b>White Dwarf</b>. As it cools over billions of years, the carbon in its core crystallizes.</p>
                    <p>Astronomers have discovered a star named <b>BPM 37093</b> (nicknamed "Lucy") which is essentially a diamond the size of the Moon, weighing 10 billion trillion trillion carats!</p>
                    """)
                    reference_box("**Source:** [Wikipedia](https://en.wikipedia.org/wiki/BPM_37093)")
                    aria_button("Close", "close", on_click=lambda:cur_intro.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                with ui.dialog() as introd, ui.card().classes('p-4 w-full max-w-[600px]').props('role=dialog aria-modal=true aria-labelledby=introd_title'):
                    html_info_box(r"""
                    <div style="font-family: sans-serif; line-height: 1.6;">
                        
                        <h3 style="margin-top: 0; color: #b45309; border-bottom: 2px solid #f59e0b; padding-bottom: 10px; margin-bottom: 15px;">1. Stellar Roleplay: You Are a Star!</h3>
                        <p>Embark on a cosmic journey and experience the life cycle of a star firsthand!</p>
                        <ul style="margin-bottom: 25px;">
                            <li>Press <b>Start Evolution</b> to be "born" in a stellar nursery. You will be assigned a random <b>Solar Mass (M☉)</b>.</li>
                            <li>Your destiny depends on your mass: low-mass stars, high-mass stars, and extremely massive stars follow very different paths.</li>
                            <li>At each step, you must select the <b>correct next evolutionary phase</b> from the options provided.</li>
                            <li><b>Scoring:</b> You earn <b>1 point</b> for every correct stage reached. A wrong choice destabilizes your core, resulting in a <b>-0.25 point</b> penalty.</li>
                            <li>Complete the sequence to reach your final remnant state (White Dwarf, Neutron Star, or Black Hole) and finish the game.</li>
                        </ul>

                        <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Interactive Visualizations</h3>
                        <p>Use these tools to help you understand the physics of stars or to find clues for the game:</p>
                        <ul>
                            <li><b> H-R Animation:</b> Opens an animation of the <b>Hertzsprung-Russell diagram</b>. It illustrates star classification based on temperature and luminosity. Stars appear ordered by their age: first the main sequence, then giants and supergiants, and finally white dwarfs.</li>
                            <li><b> Evolution Graph:</b> Opens a structural plot illustrating the <b>Life Cycle of a Star</b> based on its mass. It shows how stars start from stellar nebulae and evolve into different remnants depending on their initial size.</li>
                        </ul>

                        <h3 style="margin-top: 20px; color: #1f2937;">3. Star Characteristics and Evolution</h3>
                        <p>Stars are massive celestial bodies composed primarily of hydrogen and helium undergoing nuclear fusion in their cores. This fusion process releases energy, producing light and heat that radiate into space. The life cycle of a star is determined mainly by its mass, influencing its temperature, luminosity, size, and lifespan.</p>
                        <p>Stars form from clouds of gas and dust in space, known as nebulae. When regions within these clouds collapse under gravity, they form protostars that eventually ignite nuclear fusion. Depending on their initial mass, stars follow different evolutionary paths: smaller stars evolve into red giants and white dwarfs, while bigger stars evolve into red supergiants and eventually explode as supernovae.</p>

                        <div style="border-left: 4px solid #f59e0b; padding-left: 15px; margin: 15px 0; background-color: #fffbeb; padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
                            <p style="margin: 0; color: #92400e;">
                                <strong>Tip:</strong> If you are stuck in the game, open the <b>View Evolution Graph</b> to see the branches of the stellar family tree!
                            </p>
                        </div>
                    </div>
                    """).props('tabindex=0 role=document aria-label="Stellar evolution instructions"')
                    
               
                    aria_button("Close", "close", on_click=lambda:introd.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")
               
                def gaia_plot():
                    ISO_DF, ISO_TREE = load_isochrones_cached(MIST_PATH)
                    GAIA_DF = load_gaia_cached(STAR_GAIA_PATH)
                    df = GAIA_DF.copy()
                    if len(df) > sample_size:
                        df = df.sample(sample_size, random_state=1).reset_index(drop=True)

                    distances, idxs = ISO_TREE.query(df[['bp_rp','abs_mag']].values)
                    df['stellar_phase'] = ISO_DF['phase'].values[idxs]
                    df['mass_est'] = ISO_DF['star_mass'].values[idxs]
                
                
                    df['stellar_phase'] = refine_phase_with_gaia(df)
                

                    phase_colors = {
                        'MS':'blue', 'SGB':'green', 'RGB':'red', 
                        'HeB':'orange', 'AGB':'magenta', 'WD':'grey', 'CHeB':'purple'
                    }
                    df['stellar_phase_color'] = df['stellar_phase'].map(phase_colors).fillna('blue')
                    counts = df['stellar_phase'].value_counts()
                    total_stars = len(df)

                
                    label_map = {}      
                    final_colors = {}   

                    for phase in counts.index:
                        count = counts[phase]
                        perc = (count / total_stars) * 100
                        
                    
                        new_label = f"{phase} ({perc:.1f}%)"
                        label_map[phase] = new_label
                        
                    
                        color = phase_colors.get(phase, 'black')
                        final_colors[new_label] = color

                
                    df['stellar_phase_label'] = df['stellar_phase'].map(label_map)


                    phase_order = [label_map[p] for p in counts.index]
                    fig = px.scatter(
                        df,
                        x='bp_rp',
                        y='abs_mag',
                        color='stellar_phase_label',
                        hover_data=['source_id', 'bp_rp', 'abs_mag','mass_est'],
                        title=f"Gaia H–R Diagram (N={total_stars})",
                        color_discrete_map=final_colors,
                        category_orders={'stellar_phase_label': phase_order},
                        labels={'bp_rp':'BP−RP Color', 'abs_mag':'Absolute G Magnitude'},
                        opacity=0.6
                    )
                    fig.update_yaxes(autorange='reversed')

                
                    fig.add_trace(
                        go.Scatter(
                            x=df['bp_rp'],
                            y=df['mass_est'],
                            mode='markers',
                            marker=dict(opacity=0),
                            showlegend=False,
                            hoverinfo='skip',
                            yaxis='y2'
                        )
                    )
                    fig.update_layout(
                        title={
                        'text': f"<b>{title}</b>",   
                        'y': 0.95,                  
                        'x': 0.5,                    
                        'xanchor': 'center',      
                        'yanchor': 'top',
                        'font': {
                            'size': 24,              
                            'color': 'black'         
                        }
                    },
                        yaxis2=dict(
                            title="Mass (M☉)",
                            overlaying='y',
                            side='right',
                            type='log',
                            exponentformat='power'
                        ),
                        margin=dict(l=60, r=80, t=60, b=60),
                        height=500
                    

                    )
                    plot = ui.plotly(fig).props('tabindex=0 aria-label="Interactive Hertzsprung-Russell diagram of Gaia star dataset"')


                    plot.on('plotly_click', lambda e: show_star_dialog(df, e.args['points'][0]['pointIndex']))
                
             
                

                 

              
                @ui.refreshable
                def render_gif():
                    ui.html(f'<img src="/images/hr_diagram_evolution.gif?t={time.time()}" alt="Star evolution HR diagram" class="w-full max-w-[800px] h-auto rounded-lg shadow-lg border border-gray-300 mx-auto block">')

                def open_stellar_info_dialog():
                    render_gif.refresh()
                    
                    with ui.dialog() as comp_stellar_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="stellar-title"'):
                        
                        with ui.column().classes('w-full h-full bg-white'):
                            
                            with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="stellar-title"'):
                                ui.label('Stellar Evolution').classes('text-xl font-bold').props('id="stellar-title" role=heading aria-level=2 tabindex=0')
                                aria_button('Close', 'close', on_click=comp_stellar_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                            with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                t_hr = ui.tab('H-R Diagram Info')
                                t_graph = ui.tab('Evolution Graph')
                                t_gif = ui.tab('H-R Animation')
                              

                            with ui.tab_panels(tabs, value=t_hr).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                
                                with ui.tab_panel(t_hr):
                                    html_info_box(r"""
                                        <h3>Hertzsprung–Russell (H–R) Diagram</h3>
                                        <p>The <b>Hertzsprung–Russell (H–R)</b> diagram shows stars according to their color or temperature and luminosity (brightness) or absolute magnitude. The diagram reveals distinct groups of stars from GAIA dataset providing information about stellar evolution and properties.</p>
                                        <ul>
                                             <li><b>Main Sequence (MS):</b> The longest and most stable phase in a star's lifetime, where it resides in hydrostatic equilibrium. Stars generate energy by fusing hydrogen into helium in their cores. The position of an MS star along the diagonal band is determined by its mass, with high-mass stars residing at the luminous, hot upper-left and low-mass stars at the cooler, dim lower-right.</li>
            
            <li><b>Subgiant Branch (SGB):</b> Marks the transition off the Main Sequence after the core hydrogen is exhausted. The star's core begins to contract, heating the surrounding shell, where hydrogen starts burning, while the outer layers start to expand and cool causing it to move horizontally rightward and slightly upward on the H-R diagram.</li>
            
            <li><b>Red Giants Branch (RGB):</b> The outer envelope expands, driven by hydrogen shell burning around the helium core and leading to an increase in luminosity and a sharp drop in surface temperature, placing these stars in the upper-right region of the H-R diagram.</li>
            
            <li><b>Helium Burning (HeB):</b> When the contracting helium core reaches a critical temperature, helium fusion (the triple-alpha process) ignites, converting helium into carbon and oxygen. This new energy source brings the star to a temporary, stable phase (Red Clump for solar-mass stars) and they settle at a lower luminosity and slightly hotter temperature than the tip of the RGB.</li>
            
            <li><b>Asymptotic Giant Branch (AGB):</b> Late evolutionary stage characterized by fusion occurring in two shells: an outer hydrogen shell and an inner helium shell, both surrounding an inert carbon/oxygen core. Stars are highly luminous and the coolest giants, occupying a region above the RGB on the H-R diagram. They undergo episodic mass loss, leading to the formation of a planetary nebula.</li>
            
            <li><b>White Dwarfs (WD):</b> The remnants of low to intermediate-mass stars that have shed their outer layers, extremely dense, hot, and small. They have no active fusion and are supported against gravitational collapse solely by electron degeneracy pressure. They appear in the bottom-left corner of the H-R diagram and cool down over cosmic time, eventually becoming black dwarfs.</li>
                                        </ul>
                                    """)
                                    with ui.row().classes('w-full justify-center gap-6 mt-6'):
                                        aria_image('/images/H-R_ESO.png', "Image of Hertzsprung–Russell diagram").classes('w-full max-w-[500px] h-auto rounded-lg shadow-lg border border-gray-300')
                                        aria_image('/images/Life-Cycle-of-a-Star.png', "Image of life-cycle of a star").classes('w-full max-w-[500px] h-auto rounded-lg shadow-lg border border-gray-300')

                                with ui.tab_panel(t_graph).classes('flex flex-col items-center justify-start'):
                                   
                                    plot_star_graph(title="Life Cycle of a Star", nodes=stellar_nodes, edges=stellar_edges, height=600, width=800)

                                with ui.tab_panel(t_gif).classes('flex flex-col items-center justify-center'):
                                   
                                    render_gif()
                                    info_box("Dataset Gaia DR3: RA (right ascension), DEC(declination), z (redshift), mag_ur(magnitude in u-band filter, r-band filter). Dataset MIST: logTeff (effective temperature), logL (luminosity), stellar_mass")
                                    reference_box(""" **Dataset reference**: [GAIA SDR3](https://gea.esac.esa.int/archive/), Sysoliatina Kseniia 2022 JJ-model isochrone set: PARSEC MIST and BaSTI stellar evolution, [MIST](https://doi.org/10.11588/DATA/ZCXHOE) """)
                              
                                        

                    comp_stellar_dialog.open()
                
                with ui.row().classes('w-full justify-center items-center gap-4'):
                    aria_button('Instructions','Introduction to the stars activity', on_click=lambda:[introd.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button(
            'Scientific Info', 
            'Open Comprehensive Stellar Information Dialog', 
            on_click=lambda: [open_stellar_info_dialog(), ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                    aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_intro.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                

                game_state = {'running': False, 'mass': 0, 'step': 0, 'score': 0, 'time': 0, 'path': []}
                node_names = {n['id']: n['label'] for n in stellar_nodes}

              
                with ui.column().classes('w-full p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-600 flex flex-col items-center mt-8 mb-12'):
                    ui.label("🎮 AstroStars: Roleplay").classes("text-3xl md:text-4xl font-bold text-yellow-400 mb-4 text-center drop-shadow-md uppercase tracking-wider").props('role=heading aria-level=2 tabindex=0')
                    
                    with ui.row().classes('w-full max-w-[800px] justify-between bg-slate-800 p-4 rounded-lg border border-slate-500 mb-6 shadow-inner items-center'):
                        lbl_time = ui.label('Time: 0s').classes('text-xl text-white font-bold w-24')
                        lbl_score = ui.label('Score: 0').classes('text-xl text-green-400 font-bold w-32 text-center').props('aria-live=polite')
                        
                        def game_tick():
                            if game_state['running']:
                                game_state['time'] += 1
                                lbl_time.set_text(f"Time: {game_state['time']}s")
                        ui.timer(1.0, game_tick)
                        
                    def game_tick():
                        if game_state['running']:
                            game_state['time'] += 1
                            lbl_time.set_text(f"Time: {game_state['time']}s")
                    ui.timer(1.0, game_tick)
                    
                    def start_game():
                        import random
                        game_state['running'] = True
                        game_state['time'] = 0
                        game_state['score'] = 0
                        game_state['step'] = 1 
                        lbl_time.set_text("Time: 0s")
                        lbl_score.set_text("Score: 0")
                        
                       
                        mass = random.randint(1, 40)
                        game_state['mass'] = mass
                        if mass <= 8:
                            game_state['path'] = ["stellar_nebula", "average_star", "red_giant", "planetary_nebula", "white_dwarf"]
                        elif mass <= 20:
                            game_state['path'] = ["stellar_nebula", "massive_star", "red_supergiant", "supernova", "neutron_star"]
                        else:
                            game_state['path'] = ["stellar_nebula", "massive_star", "red_supergiant", "supernova", "black_hole"]
                            
                        render_game_ui.refresh()

                    def stop_game():
                          
                            game_state['running'] = False

                    def reset_game():
              
                        game_state['running'] = False
                        game_state['time'] = 0
                        game_state['score'] = 0
                        game_state['step'] = 0
                        game_state['path'] = []
                        lbl_time.set_text("Time: 0s")
                        lbl_score.set_text("Score: 0")
                        render_game_ui.refresh()

                    with ui.row().classes('gap-4'):
                        aria_button("Start ", "start", on_click=start_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105 animate-pulse")
                        aria_button("Stop", "stop", on_click=stop_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                        aria_button("Reset", "reset", on_click=reset_game).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                
                    @ui.refreshable
                    def render_game_ui():
                        import random
                        if not game_state['running']:
                            ui.label("Awaiting birth in the stellar nursery... Press Start!").classes("text-slate-400 italic text-center mt-4 text-xl")
                            return
                        
                        if game_state['step'] >= len(game_state['path']):
                            ui.label("Evolution Complete!").classes("text-4xl text-green-400 font-bold text-center mt-6").props('role=alert aria-live=assertive')
                            ui.label("You have reached your final destiny in the cosmos. 🌌").classes("text-white text-center mt-3 text-2xl")
                            return
                    
                        current_phase_id = game_state['path'][game_state['step'] - 1]
                        current_phase_name = node_names[current_phase_id]
                        
                        ui.label(f"Your Mass: {game_state['mass']} M☉").classes("text-3xl text-orange-400 font-bold text-center mb-4 bg-slate-800 px-8 py-3 rounded-lg border-2 border-orange-500 shadow-lg")
                        ui.label(f"Current Phase: {current_phase_name}").classes("text-2xl text-cyan-300 font-bold text-center mb-6").props('aria-live=polite')
                        ui.label("Select your NEXT evolutionary phase:").classes("text-gray-200 mb-6 text-center text-xl")
                        
                      
                        correct_id = game_state['path'][game_state['step']]
                        all_ids = list(node_names.keys())
                        options = [correct_id]
                        
                        while len(options) < 4:
                            c = random.choice(all_ids)
                          
                            if c not in options and c != "stellar_nebula":
                                options.append(c)
                                
                        random.shuffle(options)
                        
                        def check_ans(choice):
                            if choice == game_state['path'][game_state['step']]:
                                game_state['score'] += 1
                                game_state['step'] += 1
                                lbl_score.set_text(f"Score: {game_state['score']}")
                                accessible_notify("Correct evolution!", color="positive", position="top", icon="check_circle")
                                if game_state['step'] >= len(game_state['path']):
                                    game_state['running'] = False
                                    accessible_notify("End Game! Destiny reached.", color="purple", position="top", icon="emoji_events")
                            else:
                                game_state['score'] -= 0.25
                                lbl_score.set_text(f"Score: {game_state['score']}")
                                accessible_notify("Wrong phase! Core destabilized...", color="negative", position="top", icon="warning")
                            render_game_ui.refresh()

                        with ui.grid(columns=2).classes('w-full max-w-[800px] gap-6'):
                            for opt in options:
                                aria_button(node_names[opt], "choose phase", on_click=lambda o=opt: check_ans(o)).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-5 px-4 rounded-lg w-full shadow-lg text-xl transition-transform hover:scale-105")

                    render_game_ui()
               
                    

    



            
#panel planets
        def planets_page(container):
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            with container:
                ui.label(
                    "Explore our cosmic neighborhood, from the planets of our Solar System to the vast spiral arms of the Milky Way."
                ).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')

           
                
                with ui.dialog() as instru, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
                    html_info_box(r"""
                    <h3>How to play AstroLocation</h3>
                    <p>Your mission is to find the exact location of our Solar System within the Milky Way!</p>
                    <ul>
                        <li>Click directly on the interactive map of the galaxy.</li>
                        <li>A marker will appear where you click, and the system will tell you if you are too close to the center, too far, or in the right orbit but the wrong spiral arm.</li>
                        <li>Read the <b>Info</b> section to find the mathematical clues (distance in kpc) needed to solve the puzzle.</li>
                    </ul>
                    """)
                    aria_button("Close", "close", on_click=instru.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-2")

                
                with ui.dialog() as curio_dialog, ui.card().classes('p-4 w-full max-w-[600px]'):
                    html_info_box(r"""
                    <h3>Cosmic Curiosities</h3>
                    <p><b>Venus's weird rotation:</b> A day on Venus lasts longer than its year! It takes 243 Earth days to rotate on its axis, but only 225 Earth days to orbit the Sun.</p>
                    <p><b>Rogue Planets:</b> Astronomers estimate there could be billions of "rogue planets" wandering through the Milky Way without a star, floating in perpetual darkness.</p>
                    <p><b>Voyager 1:</b> Launched in 1977, Voyager 1 is the farthest human-made object. It has officially left the solar system and entered interstellar space.</p>
                    """)
                    aria_button("Close", "close", on_click=curio_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-2")

                def open_cosmic_neighborhood_dialog():
                    with ui.dialog() as cosmic_dialog, ui.card().classes('p-0 w-full min-w-[1000px] max-w-[95vw] h-[80vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="cosmic-title"'):
                        
                        with ui.column().classes('w-full h-full bg-white'):
                            
                            # Header
                            with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="cosmic-title"'):
                                ui.label('Our Cosmic Neighborhood').classes('text-xl font-bold').props('id="cosmic-title" role=heading aria-level=2 tabindex=0')
                                aria_button('Close', 'close', on_click=cosmic_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                            # Tabs
                            with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                t_mw_context = ui.tab('Milky Way Context')
                                t_solar = ui.tab('Solar System')
                                t_exo = ui.tab('Exoplanets')
                                t_milky = ui.tab('Milky Way Galaxy')

                            # Tab Panels
                            with ui.tab_panels(tabs, value=t_mw_context).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                
                                with ui.tab_panel(t_mw_context):
                                    html_info_box(r"""
                                <h3>The Milky Way Context</h3>
                                <p>The Milky Way is a barred spiral galaxy. Its mass is approx. <b>1.5 trillion solar masses (M☉)</b>.</p>
                                <p>By analyzing rotation data, astronomers determined the Sun's orbital velocity is approx. <b>220-250 km/s</b>.</p>
                                <p>Using the keplerian law, we relate velocity (<i>v</i>) to distance (<i>r</i>):</p>
                                <div class="my-2 py-2 bg-gray-100 text-center rounded">
                                    $$ v = \sqrt{\frac{G \cdot M}{r}} $$
                                </div>
                                <p>According to this relation, the velocity corresponds to a distance of roughly <b>8.2 kpc</b> from the center.</p>
                                <p>Why 8.2 kpc? The Sun is located in the <b>Orion Arm</b>, a minor spiral arm between the Sagittarius and Perseus arms.</p>
                                    """)

                                with ui.tab_panel(t_solar):
                                    html_info_box(r"""
                                <h3>Our Solar System</h3>
                                <p>Our planetary system is located in an outer spiral arm of the Milky Way. It consists of our star, the Sun, and everything bound to it by gravity.</p>
                                <ul>
                                    <li><b>Inner Terrestrial Planets:</b> Mercury, Venus, Earth, and Mars. These are relatively small, dense, and rocky.</li>
                                    <li><b>Outer Gas/Ice Giants:</b> Jupiter, Saturn, Uranus, and Neptune. Massive planets composed mostly of hydrogen, helium, water, ammonia, and methane.</li>
                                    <li><b>Asteroid Belt & Kuiper Belt:</b> Regions filled with millions of rocky and icy remnants from the system's formation, including dwarf planets like Pluto.</li>
                                </ul>
                                    """)

                                with ui.tab_panel(t_exo):
                                    html_info_box(r"""
                                <h3>Exoplanets</h3>
                                <p>An exoplanet is any planet beyond our solar system. Most orbit other stars, but free-floating "rogue planets" orbit the galactic center directly.</p>
                                <ul>
                                    <li><b>Detection Methods:</b> We usually find them via the <i>Transit Method</i> (a tiny dip in starlight as the planet passes in front) or the <i>Radial Velocity Method</i> (the star "wobbling" due to the planet's gravity).</li>
                                    <li><b>Habitable Zone:</b> Also known as the "Goldilocks Zone", it's the distance from a star where conditions might be just right (not too hot, not too cold) for liquid water to exist on the surface.</li>
                                </ul>
                                    """)

                                with ui.tab_panel(t_milky):
                                    html_info_box(r"""
                                <h3>The Milky Way Galaxy</h3>
                                <p>The Milky Way is our home galaxy, a vast rotating disk of stars, gas, and dust.</p>
                                <ul>
                                    <li><b>Structure:</b> It is a "barred spiral galaxy", meaning it has a central bar-shaped structure composed of stars, with multiple spiral arms extending outward.</li>
                                    <li><b>Sagittarius A*:</b> At the very heart of our galaxy lies a supermassive black hole, about 4 million times the mass of our Sun.</li>
                                    <li><b>Dark Matter Halo:</b> The visible disk of the Milky Way is embedded in a massive, invisible spherical halo of dark matter that holds the galaxy together.</li>
                                </ul>
                                    """)
                    
                    cosmic_dialog.open()
                with ui.row().classes('w-full justify-center gap-4 mt-2'):
                    aria_button("Instructions", "Open instructions", on_click=instru.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button(
            'Scientific Info', 
            'Open Comprehensive Information Dialog', 
            on_click=lambda: [open_cosmic_neighborhood_dialog(), ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                    aria_button("Curiosity", "Open curiosity", on_click=curio_dialog.open).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
                    aria_button('♿ Accessibility Mode', 'Jump to accessibility', on_click=lambda: (
                        acc_solar.open(), 
                        ui.run_javascript("setTimeout(() => document.getElementById('acc_solar').scrollIntoView({behavior: 'smooth', block: 'start'}), 100);")
                    )).classes("!bg-red-600 hover:!bg-red-700 text-white font-bold py-2 px-4 rounded shadow-md")

             
                IMG_PIXEL_SIZE = 800  
                REAL_WIDTH_KPC = 40.0 
                SUN_DIST_KPC = 8.2  
                KPC_TO_PX = IMG_PIXEL_SIZE / REAL_WIDTH_KPC 
                CENTER_PX = IMG_PIXEL_SIZE / 2.0

                with ui.column().classes('w-full max-w-[1000px] mx-auto p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-600 flex flex-col items-center mt-8 mb-6'):
                    ui.label("🎮 AstroEarth: Locate the Solar System").classes("text-3xl md:text-4xl font-bold text-yellow-400 mb-6 text-center drop-shadow-md uppercase tracking-wider").props('role=heading aria-level=2 tabindex=0')
                    
                    img_container = ui.element('div').classes('relative rounded-xl shadow-2xl border-4 border-slate-800 cursor-crosshair shrink bg-black')
                    img_container.style(f'width: {IMG_PIXEL_SIZE}px; max-width: 100%; aspect-ratio: 1/1;')
                    
                    with img_container:
                        src_img = '/images/MKW.jpg' 
                        feedback_label = ui.label('Click on the map to find Earth!').classes('absolute top-4 left-4 z-10 bg-black/80 text-white px-4 py-2 rounded font-mono text-lg border-l-4 border-slate-500')
                        
                        def on_click(e: MouseEventArguments):
                                 
                            click_x, click_y = e.image_x, e.image_y
                            
                        
                            dx = click_x - CENTER_PX
                            dy = click_y - CENTER_PX
                            r_px = math.sqrt(dx**2 + dy**2)
                            r_kpc = r_px / KPC_TO_PX
                            dist_error = abs(r_kpc - SUN_DIST_KPC)
                            
                            
                            # Atan2 restituisce radianti tra -pi e pi.
                            # 0° = Destra, 90° = Giù (Sotto), 180° = Sinistra, -90° = Su
                            angle_rad = math.atan2(dy, dx)
                            angle_deg = math.degrees(angle_rad)
                            
                            
                            TARGET_ANGLE_MIN = 80   
                            TARGET_ANGLE_MAX = 125  
                            
                            angle_correct = TARGET_ANGLE_MIN <= angle_deg <= TARGET_ANGLE_MAX
                            
                            
                            if dist_error <= 1.5 and angle_correct:
                                status = "✅ ORBIT FOUND!"
                                color = "#4ade80" # Verde
                                msg = f"Distance: {r_kpc:.1f} kpc. Correct Location!"
                            elif dist_error <= 1.5 and not angle_correct:
                                status = "⚠️ RIGHT ORBIT, WRONG ARM"
                                color = "#facc15" # Giallo
                                msg = f"Distance correct ({r_kpc:.1f} kpc), but wrong position. We are in the Orion Arm!"
                            elif r_kpc < SUN_DIST_KPC:
                                status = "🔥 TOO CLOSE"
                                color = "#f87171" # Rosso
                                msg = f"Distance: {r_kpc:.1f} kpc. Too hot!"
                            else:
                                status = "❄️ TOO FAR"
                                color = "#60a5fa" # Blu
                                msg = f"Distance: {r_kpc:.1f} kpc. Too cold!"

                            feedback_label.text = status
                            feedback_label.style(f'color: {color}; border-left: 5px solid {color}')
                            
                            
                            svg_content = f'''
                                <line x1="{click_x-10}" y1="{click_y-10}" x2="{click_x+10}" y2="{click_y+10}" stroke="{color}" stroke-width="4" />
                                <line x1="{click_x+10}" y1="{click_y-10}" x2="{click_x-10}" y2="{click_y+10}" stroke="{color}" stroke-width="4" />
                                <text x="{click_x+20}" y="{click_y}" fill="white" font-weight="bold" font-size="24" style="text-shadow: 2px 2px 4px black;">YOU</text>
                                
                                <circle cx="{CENTER_PX}" cy="{CENTER_PX}" r="{SUN_DIST_KPC * KPC_TO_PX}" fill="none" stroke="#fbbf24" stroke-width="2" stroke-dasharray="10,10" opacity="0.5" />
                            '''
                            
                            
                            if dist_error <= 1.5:
                                
                                r_target_px = SUN_DIST_KPC * KPC_TO_PX
                                x1 = CENTER_PX + r_target_px * math.cos(math.radians(TARGET_ANGLE_MIN))
                                y1 = CENTER_PX + r_target_px * math.sin(math.radians(TARGET_ANGLE_MIN))
                                x2 = CENTER_PX + r_target_px * math.cos(math.radians(TARGET_ANGLE_MAX))
                                y2 = CENTER_PX + r_target_px * math.sin(math.radians(TARGET_ANGLE_MAX))
                                
                                svg_content += f'''
                                    <path d="M {CENTER_PX} {CENTER_PX} L {x1} {y1} A {r_target_px} {r_target_px} 0 0 1 {x2} {y2} Z" fill="#4ade80" fill-opacity="0.2" stroke="none" />
                                    <text x="{CENTER_PX}" y="{CENTER_PX + r_target_px + 30}" text-anchor="middle" fill="#4ade80" font-weight="bold" font-size="18">Correct Zone</text>
                                '''

                            interactive_img.content = svg_content
                            accessible_notify(msg, type='positive' if "✅" in status else 'warning')

                        interactive_img = ui.interactive_image(src_img, on_mouse=on_click, cross=True).classes('w-full h-full object-contain')

              
                acc_solar = ui.expansion('Accessibility Mode: Locate Solar System Quiz', icon='accessible').classes('w-full max-w-[1000px] mx-auto mt-2 mb-12 bg-slate-800 text-white rounded-lg border border-slate-600').props('id="acc_solar" aria-label="Keyboard accessible alternative for finding the solar system"')
                with acc_solar:
                    ui.label("Identify the Solar System's position based on the provided context in the 'Info' dialog:").classes('text-blue-200 font-bold mb-4').props('tabindex=0')
                    
                    opts = [
                        'At the exact center of the Galactic Bulge',
                        'In the Orion Arm, about 8.2 kpc from the center',
                        'In the Perseus Arm, about 15 kpc away',
                        'In the extreme outer edge of the Stellar Halo'
                    ]
                    loc_choice = ui.radio(opts).classes('text-white gap-2 p-2').props('dark aria-label="Select the correct location"')

                    def check_loc_alt():
                        if not loc_choice.value:
                            accessible_notify('Please select an option first.', type='warning')
                            return
                            
                        if 'Orion Arm' in loc_choice.value:
                            accessible_notify('✅ Correct! The Solar System is located approx. 8.2 kpc from the center in the Orion Arm.', type_='success')
                        else:
                            accessible_notify('❌ Incorrect location. Re-read the Info clues and try again!', type_='error')

                    aria_button('Confirm Location', 'Submit accessible location answer', on_click=check_loc_alt).classes('mt-4 !bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded shadow-md')


#panel galaxy

        def show_galaxy_dialog(df, i):
            with ui.dialog() as dialog, ui.card().classes('w-96').props('role=dialog aria-modal=true aria-labelledby=galaxy_title'):
                ui.label(f"{df.loc[i,'galaxy_name']}").classes('text-sm text-center').props('tabindex=0 aria-label=Galaxy name')
                aria_image(df.loc[i,'image_url'], f"Image of galaxy {df.loc[i,'galaxy_name']}").classes('w-full rounded shadow')
                info_box(
                    f"""
                    **SpecObj ID:** {df.loc[i,'specobj_id']}  
                    **Redshift:** {df.loc[i,'z']}  
                    **Distance (Mpc):** {df.loc[i,'dist_comoving_mpc']:.1f}  
                    **Class:** {df.loc[i,'class']}  
                    **Age (Gyr):** {df.loc[i,'age_gyr']:.2f}  
                    **Type:** {df.loc[i,'subclass']}
                    **RA:** {df.loc[i,'ra_deg']}  
                    **DEC:** {df.loc[i,'dec_deg']}
                    
                    """
                ).props('tabindex=0 aria-label=Galaxy details including SpecObj ID, Redshift, Distance, and Class')
                aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            dialog.open()

        
        
        @lru_cache(maxsize=1)
        def load_sdss_cached(csv_path):
            df = pd.read_csv(csv_path, quotechar="'", skipinitialspace=True)
            df.columns = df.columns.str.strip()
            return df

        SDSS_GALAXY_DF = load_sdss_cached(GAL_SDSS_PATH)
        
        GALAXY_IMAGES = [
        {'file': 'images/spiral_galaxy.jpg', 'title': 'Spiral Galaxy', 'description': '...'},
        {'file': 'images/elliptical_galaxy.jpg', 'title': 'Elliptical Galaxy', 'description': '...'},
        {'file': 'images/irregular_galaxy.jpg', 'title': 'Irregular Galaxy', 'description': '...'},
        {'file': 'images/lenticular_galaxy.jpg', 'title': 'Lenticular Galaxy', 'description': '...'}
    ]
        def galaxy_image_grid(galaxies):
                
            with ui.grid(columns=2):
                for galaxy in galaxies:
                    
                    with ui.card().classes('w-full'):
                        
                    
                        aria_image(galaxy['file'], galaxy['title']).classes('rounded-md')
                        ui.label(galaxy['title']).classes('text-lg font-bold').props('tabindex=0 aria-label=Galaxy title')
                        
        def galaxy_map( csv_path: str,on_galaxy_select=None,sample_size=100):
                    
                    
                    
            #df = sample_csv_reservoir(csv_path, n=1000, quotechar="'", skipinitialspace=True)
            #df = pd.read_csv(csv_path, quotechar="'", skipinitialspace=True)
            #df.columns = df.columns.str.strip()
            df = SDSS_GALAXY_DF.copy()

            df = df.dropna(subset=['ra','dec','z'])

            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=1).reset_index(drop=True)
            df['subclass'] = df['subclass'].replace('', 'Unknown')
        
            coords = SkyCoord(df['ra'].values, df['dec'].values,
                            unit=(u.hourangle, u.deg), frame='icrs')
            df['ra_deg'] = coords.ra.degree
            df['dec_deg'] = coords.dec.degree

            df = df.dropna(subset=['ra_deg','dec_deg','z'])

        

            df['age_gyr'] = df['z'].apply(lambda z: cosmo.age(z).value)
            df['dist_comoving_mpc'] = df['z'].apply(lambda z: cosmo.comoving_distance(z).value)

            
            df['image_url'] = (
                "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?"
                "ra=" + df['ra_deg'].astype(str) +
                "&dec=" + df['dec_deg'].astype(str) +
                "&scale=0.2&width=100&height=100"
            )

        
            df['galaxy_name'] = ' ' 
            for i, row in df.iterrows():
                try:
                    coord = SkyCoord(row['ra_deg']*u.deg, row['dec_deg']*u.deg, frame='icrs')
                    result = Simbad.query_region(coord, radius='5s')
                    if result is not None and len(result) > 0:
                        df.at[i, 'galaxy_name'] = result['MAIN_ID'][0].decode() if isinstance(result['MAIN_ID'][0], bytes) else result['MAIN_ID'][0]
                except Exception as e:
                    pass  

            fig = px.scatter(
                df,
                x='ra_deg',
                y='dec_deg',
                color='subclass',
                hover_data={'galaxy_name': True,
                    'specobj_id': True,
    'z': True,
    'age_gyr': ':.2f',
    'dist_comoving_mpc': ':.1f',
    'class': True, 'image_url': False 
                },
                title='Galaxy Distribution from SDSS Sample',
                labels={'ra_deg':'Right Ascension (deg)', 'dec_deg':'Declination (deg)'}
            )

            
                #plot=ui.plotly(fig)
                #plot.on('plotly_click', lambda e: show_galaxy_dialog(df, e.args['points'][0]['pointIndex']))



            
            #ui.label("Selext a galaxy:").classes('text-lg mt-4').props('tabindex=0 aria-label=Select a galaxy instruction')

            options = [
                (f"{row['specobj_id']} ", i)
                for i, row in df.iterrows()
            ]

            #selected = ui.select(            options=options,                label='Galaxy ID',                with_input=True            ).classes('w-80').props('aria-label=Galaxy selection dropdown')

            def on_select(e):
                data = e.args

            
                if not data:
                    return  

        
                if isinstance(data, dict):
                    i = data.get('value')
                    if i is not None and i in df.index:
                        show_galaxy_dialog(df, i)
                        if on_galaxy_select:
                            row = df.loc[i]
                            
                            on_galaxy_select(row['ra_deg'], row['dec_deg'])

            #selected.on('update:model-value', on_select)    
        def show_morphology_dialog(df, i):
            row = df.iloc[i]
          
            #gal_name = row['galaxy_name'] if row['galaxy_name'] != 'Unknown' else ''

            with ui.dialog() as dialog, ui.card().classes('w-96 p-0 overflow-hidden').props('role=dialog aria-modal=true aria-labelledby=galaxy_title'):
                
                html_info_box(f"""
                <div style="font-family: sans-serif; padding: 20px;">
                    
                    <h3 style="margin-top: 0; color: #3b82f6; margin-bottom: 10px;">Galaxy Details</h3>
                    
                    <div style="text-align: center; margin-bottom: 15px;">
                        <img src="{row['image_url']}" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);" alt="Galaxy Image">
                        
                    </div>

                    <div style="background-color: #f3f4f6; padding: 10px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #e5e7eb;">
                        <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase; font-weight: bold;">SpecObj ID</div>
                        <div style="font-family: monospace; font-size: 0.95em; color: #111; word-break: break-all;">{row['specobj_id']}</div>
                    </div>

                    <div style="display: grid; grid-template-columns: auto auto; gap: 5px 15px; align-items: baseline; font-size: 0.95em;">
                        
                        <strong style="color: #4b5563;">Class:</strong>
                        <span style="color: #d97706; font-weight: bold;">{row['class']}</span>
                        
                        <strong style="color: #4b5563;">Type:</strong>
                        <span>{row['subclass']}</span>

                        <strong style="color: #4b5563;">Redshift:</strong>
                        <span>{row['z']}</span>
                        
                        <strong style="color: #4b5563;">Age:</strong>
                        <span>{row['age_gyr']:.2f} Gyr</span>

                        <strong style="color: #4b5563;">Distance:</strong>
                        <span>{row['dist_comoving_mpc']:.1f} Mpc</span>
                        
                        <strong style="color: #4b5563;">RA / DEC:</strong>
                        <span style="font-size: 0.85em;">{row['ra_deg']:.2f}, {row['dec_deg']:.2f}</span>

                    </div>
                </div>
                """).props('tabindex=0 aria-label="Selected galaxy details"')
                
                with ui.row().classes('w-full justify-center pb-4'):
                    aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded shadow-md")
            dialog.open()
        
            
        @lru_cache(maxsize=1)
        def load_morphology_cached(path):
            colnames = [
            "OBJID","COL2","COL3",
            "umag","gmag","rmag","imag","zmag",
            "Extu","Extg","Extr","Exti","Extz",
            "R50","R90","R50_R90","z","DL",
            "uMAG","gMAG","rMAG","iMAG","zMAG",
            "u_r","g_i","r_z",
            "Kcorg","Kcori","Kcorr","Kcoru","Kcorz",
            "SVM","RF"
        ]
            df = pd.read_csv(path, sep=r"\s+", names=colnames)
            return df

        
        
        

        
            
        def galaxy_morphology_page(gz_path: str, sample_size=50000):
        
                    
            SDSS_MORPHO_DF = load_morphology_cached(SDSS_MORPHO_PATH)
            #df = pd.read_csv(gz_path, sep=r"\s+", names=colnames)

            
            

            df = SDSS_MORPHO_DF.copy()


            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=1).reset_index(drop=True)
            df = df.dropna(subset=["uMAG", "rMAG", "R50", "R90","R50_R90","z","DL","SVM"])


        
            df["color_ur"] = df["uMAG"] - df["rMAG"]
            df["color_gi"] = df["gMAG"] - df["iMAG"]
            df["concentration"] = df["R50"] / df["R90"]


            
            def classify(row):
                if row['SVM'] == 0:
                    return "Elliptical"
                else:
                    
                    if row['color_ur'] < 1.8 and row['concentration'] < 2.2:
                        return "Irregular"
                    elif row['color_ur'] < 2.2 and row['concentration'] < 2.6:
                        return "Spiral"
                    else:
                        return "Lenticular"


            df["morphology"] = df.apply(classify, axis=1)
        
            counts = df["morphology"].value_counts()
            total = len(df)
            labels_with_pct = {
                morph: f"{morph} ({counts[morph]/total*100:.1f}%)"
                for morph in counts.index
            }
            df["morphology_pct"] = df["morphology"].map(labels_with_pct)


            #coords = SkyCoord(df['COL2'].values, df['COL3'].values, unit=(u.deg, u.deg), frame='icrs')
            #df['ra_deg'] = coords.ra.degree
            #df['dec_deg'] = coords.dec.degree

            #df['image_url'] = (            "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?"      "ra=" + df['ra_deg'].astype(str) +       "&dec=" + df['dec_deg'].astype(str) + "&scale=0.2&width=100&height=100" )


        
            color_map = {
                "Elliptical": "red",
                "Lenticular": "orange",
                "Spiral": "blue",
                "Irregular": "green"
            }

            

            color_discrete_map = {labels_with_pct[k]: v for k, v in color_map.items() if k in labels_with_pct}


            fig1 = px.scatter(
                df,
                x="color_ur",
                y="concentration",
                color="morphology_pct",
                color_discrete_map=color_discrete_map,
                hover_data={
                    "OBJID": True,
                    "morphology": True,
                    "color_ur": ':.2f',
                    "R50_R90": ':.2f',
                    "z": True,
                    "DL": ':.1f',
                    "rMAG": True
                    
                    
                },
                title="SDSS Galaxy Morphology — Color–Concentration "
            )
            fig2 = px.scatter(
    df,
    x="color_ur",
    y="rMAG",  
    color="morphology",
    color_discrete_map=color_map,
    hover_data={
                    "OBJID": True,
                    "morphology": True,
                    "color_ur": ':.2f',
                    "R50_R90": ':.2f',
                    "z": True,
                    "DL": ':.1f',
                    "rMAG": True
                    
                    
                },title="SDSS Galaxy Morphology — Color–Magnitude"
    )
            fig2.update_yaxes(autorange='reversed')
            
            
            plot1 = ui.plotly(fig1).classes("w-full").props('tabindex=0 aria-label="Color vs Concentration scatter plot of SDSS galaxy morphology classification"')
            
            #plot2 = ui.plotly(fig2).classes("w-full")  


            plot1.on(    'plotly_click',    lambda e: show_morphology_dialog(df, e.args["points"][0]["pointIndex"]))
            #plot2.on(    'plotly_click',    lambda e: show_morphology_dialog(df, e.args["points"][0]["pointIndex"]))



                    
        def galaxy_map_page( container):
            container.classes('w-full flex flex-col items-center justify-center text-center mx-auto gap-6')
            full_df = SDSS_GALAXY_DF.copy() 
            mem_state = {
                'running': False, 'time': 0, 'score': 0,
                'cards': [], 'flipped': [], 'matched': set(), 'lock': False
            }
           
            coords = SkyCoord(full_df['ra'].values, full_df['dec'].values, unit=(u.hourangle, u.deg), frame='icrs')
            full_df['ra_deg'] = coords.ra.degree
            full_df['dec_deg'] = coords.dec.degree
            RA_MIN = full_df['ra_deg'].min() - 5
            RA_MAX = full_df['ra_deg'].max() + 5
            DEC_MIN = full_df['dec_deg'].min() - 5
            DEC_MAX = full_df['dec_deg'].max() + 5
            
            RA_SPAN = RA_MAX - RA_MIN
            DEC_SPAN = DEC_MAX - DEC_MIN
            with container:
                ui.label(
    "Visualize the large-scale structures of the Universe by mapping galaxies in the space to understand their distribution and morphological classification."
).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')

                  
                            
                with ui.dialog() as info, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-labelledby=morphology_info_title'):
                    html_info_box(r"""
                    <div style="font-family: sans-serif; line-height: 1.6;">
                        
                        <h3 style="margin-top: 0; color: #047857; border-bottom: 2px solid #10b981; padding-bottom: 10px; margin-bottom: 15px;">1. Galaxy Morphology Memory Game</h3>
                        <p>Test your pattern recognition skills by matching galaxies based on their morphological types!</p>
                        <ul style="margin-bottom: 25px;">
                            <li>Press <b>Start</b> to deal 24 hidden galaxy images face down and start the timer.</li>
                            <li>Click on two cards to reveal them. Your goal is not to find two identical images, but to match <b>two different galaxies of the SAME classification</b> (e.g., two Spirals, or two Ellipticals).</li>
                            <li>If they match, you earn <b>1 point</b> and the cards stay face up.</li>
                            <li>If they are a wrong pair, you lose <b>0.25 points</b> and the cards will flip back over after 1 second. Try to memorize them!</li>
                            <li>Clear the entire cosmic board to win. Use the <b>Reset</b> button if you want to restart the timer and shuffle again.</li>
                        </ul>

                        <h3 style="margin-top: 0; color: #1d4ed8; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 15px;">2. Galaxies on the Map</h3>
                        <p>Use the buttons on the main panel to open the interactive plots:</p>
                        <ul>
                            <li><b> Map Animation:</b> Opens the gif plot that represents galaxies located in two clusters by their celestial coordinates (Right Ascension / Declination).</li>
                            <li><b> Morpho Animation:</b> Opens the animation representing galaxy morphology classification: elliptical, lenticular, spiral, and irregular.</li>
                        </ul>
                        <p>The galaxies in the plots appear ordered by their redshift (z), which is a measure of how much the wavelength of light has stretched due to the expansion of the Universe. Higher redshift values indicate objects that are further away and observed further back in time.</p>
                        <ul>
                            <li>Explore the galaxy distribution to understand large-scale structures in the Universe.</li>
                            <li>Observe how galaxy morphology varies with redshift, providing insights into galaxy evolution over cosmic time.</li>
                        </ul>
                        
                        <div style="border-left: 4px solid #3b82f6; padding-left: 15px; margin: 15px 0; background-color: #f9fafb; padding-top: 10px; padding-bottom: 10px; border-radius: 4px;">
                            <p style="margin: 0; color: #1f2937;">
                                <strong>Interact:</strong> Inside the <b>Map Animation</b> dialog, use the dropdown menu to select a <b>Galaxy ID</b>. This will place a marker on the exact coordinates of the map and open a separate window with further details about that galaxy's morphology, redshift, and distance from Earth.
                            </p>
                        </div>

                        <br>

                        <h3 style="margin-top: 0; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 20px;">3. Data Fields Glossary</h3>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: start;">
                            
                            <div>
                                <h4 style="color: #1d4ed8; margin-top: 0; margin-bottom: 15px;">Physics & Properties</h4>
                                
                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">Redshift (z)</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                        A measure of how much the wavelength of light has stretched due to the expansion of the Universe. Higher values indicate objects that are further away and observed further back in time.
                                    </span>
                                </div>

                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">Distance (Mpc)</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                        The estimated luminosity distance to the object in Megaparsecs. <br>
                                        <i>(1 Mpc ≈ 3.26 million light-years)</i>.
                                    </span>
                                </div>

                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">Age (Gyr)</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                            The lookback time: how long the light has traveled to reach us (in Billions of years).
                                    </span>
                                </div>
                            </div>

                            <div>
                                <h4 style="color: #1d4ed8; margin-top: 0; margin-bottom: 15px;">Classification & Location</h4>
                                
                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">Class & Type</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                        <b>Class:</b> The broad category (GALAXY, STAR, or QSO/Quasar).<br>
                                        <b>Type:</b> The specific morphology (e.g., Spiral, Elliptical). <br>
                                        <i style="color: #d97706;">Note: If "nan" appears, it means the morphological classification is missing or the object was too faint to classify automatically.</i>
                                    </span>
                                </div>

                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">RA & DEC</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                        <b>Right Ascension (RA)</b> and <b>Declination (DEC)</b> are the celestial coordinates, acting like Longitude and Latitude for the sky map.
                                    </span>
                                </div>

                                <div style="margin-bottom: 15px;">
                                    <strong style="color: #111827;">SpecObj ID</strong><br>
                                    <span style="font-size: 0.95em; color: #4b5563;">
                                        A unique 19-digit identifier used in the SDSS database to catalogue the specific spectral observation.
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """).props('tabindex=0 role=document aria-label="Galaxy map instructions"')
                    
                    aria_button('Close', 'close button', on_click=lambda: info.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded mt-4")
              
                                
                with ui.dialog() as cur_gal, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('role=dialog aria-modal=true aria-labelledby=cur_gal_title'):
                    html_info_box(r"""
                    <h3>The "Green Valley"</h3>
                    <p>Galaxies usually fall into two color categories: the <b>"Red Sequence"</b> (old, dead elliptical galaxies) and the <b>"Blue Cloud"</b> (active, star-forming spiral galaxies).</p>
                    <p>However, there is a rare intermediate state called the <b>"Green Valley"</b>. These galaxies are slowly dying, transitioning from blue to red as they run out of gas to form new stars.</p>
                    """)
                    reference_box("**Source:** [Schawinski (2014)](https://academic.oup.com/mnras/article/440/1/889/1749989)")
                    aria_button("Close", "close", on_click=lambda:cur_gal.close()).classes("!bg-orange-500 text-white font-bold py-2 px-4 rounded")

                    
                marker_state = {'element': None}
    
                def move_marker_on_gif(ra, dec):
                    if marker_state['element'] is None: return
                    margin_left_pct = 12.5   
                    margin_bottom_pct = 11.0 
                    plot_width_pct = 77.5    
                    plot_height_pct = 77.0   
                    norm_ra = (ra - RA_MIN) / RA_SPAN
                    norm_dec = (dec - DEC_MIN) / DEC_SPAN
                    left_pos = margin_left_pct + (norm_ra * plot_width_pct)
                    margin_top_pct = 100 - margin_bottom_pct - plot_height_pct
                    top_pos = margin_top_pct + ((1.0 - norm_dec) * plot_height_pct)
                
                    marker_state['element'].classes(remove='hidden')
                    marker_state['element'].style(f'top: {top_pos}%; left: {left_pos}%; display: block;')

                @ui.refreshable
                def render_map_gif():
                    ui.html(f'<img src="/images/sdss_distribution_z.gif?t={time.time()}" alt="Distribution of galaxies" class="w-full h-auto block m-0 p-0 rounded-lg">')

                @ui.refreshable
                def render_morpho_gif():
                    ui.html(f'<img src="/images/galaxy_evolution.gif?t={time.time()}" alt="Plot concentration vs color u-r" class="w-full max-w-[800px] h-auto block m-0 p-0 border-none mx-auto">')
                def open_galaxy_info_dialog():
                 
                    render_map_gif.refresh()
                    render_morpho_gif.refresh()
                    
                    if marker_state['element']:
                        marker_state['element'].classes(add='hidden')
                        
                    with ui.dialog() as comp_galaxy_dialog, ui.card().classes('p-0 w-full min-w-[1200px] max-w-[95vw] h-[90vh] overflow-hidden').props('role=dialog aria-modal="true" aria-labelledby="galaxy-title"'):
                        
                        with ui.column().classes('w-full h-full bg-white'):
                            
                            
                            with ui.row().classes('w-full justify-between items-center bg-slate-900 text-white p-4 shrink-0').props('role=dialog aria-modal="true" aria-labelledby="galaxy-title"'):
                                ui.label('Galaxy Morphology & Classification').classes('text-xl font-bold').props('id="galaxy-title" role=heading aria-level=2 tabindex=0')
                                aria_button('Close', 'close', on_click=comp_galaxy_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                            
                         
                            with ui.tabs().classes('w-full text-white bg-slate-700 shrink-0') as tabs:
                                t_morpho = ui.tab('Morphology Classes')
                                t_subclass = ui.tab('SDSS Subclasses')
                                t_images = ui.tab('Galaxy Images')
                                t_plot_z = ui.tab('Plot z')
                                t_plot_mag = ui.tab('Plot r-mag')
                                t_map_gif = ui.tab('Map Animation')
                                t_morpho_gif = ui.tab('Morpho Animation')
                             

                           
                            with ui.tab_panels(tabs, value=t_morpho).classes('w-full h-full p-6 overflow-y-auto bg-gray-50 text-slate-900'):
                                
                           
                                with ui.tab_panel(t_morpho):
                                    html_info_box(r"""
                    <h3>Morphology Classification</h3>
                    <p>This plot presents a classification of galaxy morphologies based on their color and concentration index using data from the Sloan Digital Sky Survey (SDSS). Galaxies are categorized into four main types: Elliptical, Lenticular, Spiral, and Irregular. The classification is based on the u−r color index and the concentration index (R90/R50), which are indicative of the stellar populations and structural properties of galaxies.</p>
                    
                    <ul>
                        <li><b>Elliptical:</b> Galaxies characterized by red colors and high concentration indices, indicating older stellar populations and a more compact structure.</li>
                        <li><b>Lenticular:</b> Galaxies exhibit intermediate colors and concentration indices, representing a transitional morphology between elliptical and spiral types.</li>
                        <li><b>Spiral:</b> Galaxies tend to have bluer colors and lower concentration indices, reflecting ongoing star formation and a disk-like structure.</li>
                        <li><b>Irregular:</b> Galaxies display very blue colors and low concentration indices, often due to recent or ongoing star formation triggered by interactions or mergers.</li>
                    </ul>
                    
                    <p><b>Galaxy classification is based on the u−r color index and the concentration index (R50/R90), following observational studies from the Sloan Digital Sky Survey (SDSS).</b></p>
                    
                    <ul>
                        <li><b>Color (u−r):</b> Measures the difference between the ultraviolet (u) and red (r) magnitudes.
                            <ul>
                                <li>Bluer galaxies (lower u−r) indicate younger stellar populations and ongoing star formation, typical of Spiral and Irregular galaxies.</li>
                                <li>Redder galaxies (higher u−r) indicate older stellar populations, characteristic of Elliptical and Lenticular galaxies.</li>
                            </ul>
                        </li>
                        <li><b>Concentration (R50/R90):</b> Ratio of the radius containing 50-percent of the galaxy’s light (R50) to the radius containing 90-percent (R90).
                            <ul>
                                <li>High concentration values correspond to more centrally concentrated light profiles, typical of Elliptical galaxies.</li>
                                <li>Lower concentration values indicate more extended disks, typical of Spiral and Irregular galaxies.</li>
                            </ul>
                        </li>
                        <li><b>Classification criteria:</b>
                            <ul>
                                <li><b>Elliptical:</b> red colors (u−r > 2.22) and high concentration (C > 2.6)</li>
                                <li><b>Lenticular:</b> red colors and intermediate concentration (2.4 ≤ C ≤ 2.6)</li>
                                <li><b>Spiral:</b> blue colors (u−r < 2.2) and moderate concentration (C < 2.6)</li>
                                <li><b>Irregular:</b> very blue colors (u−r < 1.8) and low concentration (C < 2.2)</li>
                            </ul>
                        </li>
                    </ul>
                                    """).props('tabindex=0 role=document aria-label="Galaxy classification instructions"')

                                
                                with ui.tab_panel(t_subclass):
                                    html_info_box(r"""
                    <h3>SDSS Galaxy Subclasses Explained</h3>
                    <p>Galaxies in the Sloan Digital Sky Survey (SDSS) are spectroscopically classified based on their observed spectral lines, which reveal the dominant energy sources within the galaxy.</p>
                    
                    <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

                    <h4>Key Subclasses and Activity</h4>
                    
                    <table style="width:100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
                        <tr style="background-color: #bae6fd; color: #0c4a6e;">
                            <th style="border: 1px solid #0284c7; padding: 8px;">Subclass</th>
                            <th style="border: 1px solid #0284c7; padding: 8px;">Dominant Activity</th>
                            <th style="border: 1px solid #0284c7; padding: 8px;">Spectral Characteristics</th>
                            <th style="border: 1px solid #0284c7; padding: 8px;">Description</th>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #0284c7; padding: 8px;"><b>STARFORMING</b></td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Normal Star Formation</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Narrow emission lines (e.g., H-alpha, [O II]), typical of H II regions.</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Galaxies where the primary source of light and energy is the birth of new stars.</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #0284c7; padding: 8px;"><b>STARBURST</b></td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Intense Star Formation</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Very strong emission lines indicating a high rate of star formation, often compressed into a small region.</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Undergoing a brief, rapid burst of star formation, using up gas quickly.</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #0284c7; padding: 8px;"><b>AGN</b></td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Active Galactic Nucleus (Seyfert)</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Narrow emission lines from gas excited by a central supermassive black hole (SMBH).</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">The core luminosity is dominated by the SMBH accretion disk, but the spectral lines are narrow.</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #0284c7; padding: 8px;"><b>BROADLINE</b></td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Active Galactic Nucleus (Quasar/Broadline Seyfert)</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Very wide (broad) emission lines, indicating high-velocity gas close to the central SMBH.</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Often classified as Quasars or the brightest AGNs. The broad lines indicate rapid motion.</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid #0284c7; padding: 8px;"><b>COMPOSITE</b></td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Mixed Activity</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">Shows spectral signatures from both Star Formation and an AGN.</td>
                            <td style="border: 1px solid #0284c7; padding: 8px;">A transition stage where both star formation and the central black hole contribute significantly to the total energy output.</td>
                        </tr>
                    </table>

                    <hr style="margin: 15px 0; border-top: 1px solid #0284c7;">

                    <h4>The 'BROADLINE' Distinction</h4>
                    <p>The <b>BROADLINE</b> label (e.g., AGN BROADLINE) signifies that the galaxy's spectrum contains very wide emission lines. This indicates that the gas is moving at high speeds, usually because it is orbiting very close to the supermassive black hole at the galaxy's center. It is generally a classification subset of the most energetic <b>AGNs</b> (often Quasars).</p>
                                    """).props('tabindex=0 role=document aria-label="SDSS Galaxy Classification information"')

                               
                                with ui.tab_panel(t_images).classes('flex flex-col items-center justify-start'):
                                    ui.label("Galaxy Image Grid").classes("text-2xl font-bold mb-4 text-slate-800")
                                    galaxy_image_grid(GALAXY_IMAGES)

                             
                                with ui.tab_panel(t_plot_z).classes('flex flex-col items-center justify-start gap-4'):
                                   
                                    ui.label("Galaxy Redshift (z) Distribution").classes("text-2xl font-bold mb-2 text-slate-800")
                                    
                                   
                                    with ui.row().classes('w-full justify-center gap-6'):
                                        with ui.column().classes('flex-1 items-center'):
                                            aria_image('/images/sdss_gal_z.png', "Image of the distribution of galaxies from SDSS dataset with redshift").classes('w-full max-w-[600px] h-auto rounded-lg shadow-lg border border-gray-300')
                                        with ui.column().classes('flex-1 items-center'):
                                            aria_image('/images/sdss_gal_z3D.png', "Image of the distribution of galaxies from SDSS dataset with redshift").classes('w-full max-w-[600px] h-auto rounded-lg shadow-lg border border-gray-300')
                                            
                                with ui.tab_panel(t_plot_mag).classes('flex flex-col items-center justify-start gap-4'):
                       
                                    ui.label("Magnitude vs Color").classes("text-2xl font-bold mb-2 text-slate-800")
                                    
                                    
                                    with ui.row().classes('w-full justify-center gap-6'):
                                        with ui.column().classes('flex-1 items-center'):
                                            aria_image('/images/mag_gal_redshift.png', "Image of the galaxy morphology magnitude vs color").classes('w-full max-w-[600px] h-auto rounded-lg shadow-lg border border-gray-300')
                                        with ui.column().classes('flex-1 items-center'):
                                            aria_image('/images/mag_gal_z_color.png', "Image of the galaxy morphology magnitude vs color").classes('w-full max-w-[600px] h-auto rounded-lg shadow-lg border border-gray-300')

                               
                                with ui.tab_panel(t_map_gif).classes('flex flex-col items-center justify-start gap-4'):
                                   
                                    ui.label("Galaxy Distribution Map").classes('text-2xl font-bold mb-2 text-slate-800').props('id="map_gif_title" tabindex=0')
                                    
                                   
                                    with ui.row().classes('w-full justify-center items-center mb-2 z-50 gap-4'):
                                        ui.label("Select a galaxy:").classes('text-lg font-bold text-slate-800')
                                        
                                        local_df = full_df.dropna(subset=['ra_deg','dec_deg','z']).copy()
                                        if len(local_df) > 100:
                                            local_df = local_df.sample(100, random_state=1).reset_index(drop=True)
                                        
                                        local_df['age_gyr'] = local_df['z'].apply(lambda z: cosmo.age(z).value)
                                        local_df['dist_comoving_mpc'] = local_df['z'].apply(lambda z: cosmo.comoving_distance(z).value)
                                        local_df['image_url'] = (
                                            "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?"
                                            "ra=" + local_df['ra_deg'].astype(str) +
                                            "&dec=" + local_df['dec_deg'].astype(str) +
                                            "&scale=0.2&width=100&height=100"
                                        )
                                        local_df['galaxy_name'] = 'Unknown'
                                        options = {i: str(row['specobj_id']) for i, row in local_df.iterrows()}

                                        def on_select(e):
                                            if e.value is not None and e.value in local_df.index:
                                                i = e.value
                                                show_galaxy_dialog(local_df, i)
                                                row = local_df.loc[i]
                                                move_marker_on_gif(row['ra_deg'], row['dec_deg'])
                                        
                                        ui.select(
                                            options=options,
                                            label='Galaxy ID',
                                            with_input=True,
                                            on_change=on_select
                                        ).classes('w-80 bg-white').props('behavior="menu" aria-label="Galaxy selection dropdown"')

                                  
                                    with ui.element('div').classes('w-full max-w-[500px] relative rounded-lg shadow-lg border border-gray-300 mx-auto'):
                                        render_map_gif()
                                        marker_state['element'] = ui.element('div').classes(
                                            'absolute w-6 h-6 bg-cyan-400 rounded-full border-2 border-white shadow-[0_0_10px_rgba(0,255,255,1)] z-50 hidden'
                                        ).style('transform: translate(-50%, -50%); transition: top 0.5s ease-out, left 0.5s ease-out;')
                               
                                with ui.tab_panel(t_morpho_gif).classes('flex flex-col items-center justify-center'):
                                    ui.label("Galaxy Morphology Evolution").classes("text-2xl font-bold mb-4 text-slate-800")
                                    render_morpho_gif()
                                    info_box('Dataset: RA (right ascension), DEC (declination), z (redshift), z_error, subclass').props('tabindex=0 role=document aria-label=Galaxy data')
                                    reference_box(""" **Dataset reference**: [SDSS DR16](https://www.sdss.org/dr16/), [SDSS](https://cdsarc.cds.unistra.fr/ftp/J/A+A/648/A122/), VizieR Online Data Catalog: SDSS galaxies morphological classification (Vavilova+, 2021)""")


                         
                               
                                        
                    comp_galaxy_dialog.open()
                with ui.row().classes('w-full items-center justify-center ' ):
                    galaxy_map(GAL_SDSS_PATH,on_galaxy_select=move_marker_on_gif)
                    aria_button("Instructions",label="Read instruction",on_click=lambda:     [info.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                    aria_button(
            'Scientific Info ', 
            'Open Galaxy Information Dialog', 
            on_click=lambda: [open_galaxy_info_dialog(), ui.run_javascript("if (typeof MathJax !== 'undefined') { MathJax.typesetPromise(); }")]
        ).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-6 rounded shadow-md")
                    aria_button("Curiosity", "Open curiosity", on_click=lambda:[cur_gal.open(),ui.run_javascript("MathJax.typesetPromise()")]).classes("!bg-purple-600 hover:!bg-purple-800 text-white font-bold py-2 px-4 rounded")
               
               

               
                with ui.column().classes('w-full p-8 bg-slate-900 rounded-xl shadow-2xl border border-slate-600 flex flex-col items-center mt-8 mb-12'):
                    ui.label("🎮 AstroMemory: Galaxy Morphology").classes("text-3xl md:text-4xl font-bold text-yellow-400 mb-4 text-center drop-shadow-md uppercase tracking-wider").props('role=heading aria-level=2 tabindex=0')

                    with ui.row().classes('w-full max-w-[800px] justify-between bg-slate-800 p-4 rounded-lg border border-slate-500 mb-6 shadow-inner items-center'):
                        lbl_time = ui.label('Time: 0s').classes('text-xl text-white font-bold w-24')
                        lbl_score = ui.label('Score: 0').classes('text-xl text-green-400 font-bold w-32 text-center').props('aria-live=polite')
                        def game_tick():
                            if mem_state['running']:
                                mem_state['time'] += 1
                                lbl_time.set_text(f"Time: {mem_state['time']}s")
                        ui.timer(1.0, game_tick)
                        
                        def start_memory():
                            import random
                            mem_state['running'] = True
                            mem_state['time'] = 0
                            mem_state['score'] = 0
                            mem_state['flipped'].clear()
                            mem_state['matched'].clear()
                            mem_state['lock'] = False
                            lbl_time.set_text("Time: 0s")
                            lbl_score.set_text("Score: 0")
                            
                            memory_deck = [
                                # --- IRREGULAR (6) ---
                                {'type': 'Irregular', 'img': 'images/IC559_irregular.jpg'},
                                {'type': 'Irregular', 'img': 'images/IC4710_irregular.jpg'},
                                {'type': 'Irregular', 'img': 'images/NGC1427_irregular.jpg'},
                                {'type': 'Irregular', 'img': 'images/NGC7292_irregular.jpg'},
                                {'type': 'Irregular', 'img': 'images/NGC55_irregular.jpg'},
                                {'type': 'Irregular', 'img': 'images/NGC5264_irregular.jpeg'}, 

                                # --- LENTICULAR (6) ---
                                {'type': 'Lenticular', 'img': 'images/NGC4921_lenticular.jpg'},
                                {'type': 'Lenticular', 'img': 'images/NGC1387_lenticular.png'},
                                {'type': 'Lenticular', 'img': 'images/NGC2787_lenticular.jpg'},
                                {'type': 'Lenticular', 'img': 'images/NGC5866_lenticular.png'},
                                {'type': 'Lenticular', 'img': 'images/NGC5010_lenticular.jpg'},
                                {'type': 'Lenticular', 'img': 'images/NGC4886_lenticular.jpeg'},

                                # --- ELLIPTICAL (6) ---
                                {'type': 'Elliptical', 'img': 'images/NGC221_elliptical.jpg'},
                                {'type': 'Elliptical', 'img': 'images/NGC4365_elliptical.png'},
                                {'type': 'Elliptical', 'img': 'images/IC2006_elliptical.jpg'},
                                {'type': 'Elliptical', 'img': 'images/NGC1316_elliptical.jpg'},
                                {'type': 'Elliptical', 'img': 'images/NGC4150_elliptical.jpg'},
                                {'type': 'Elliptical', 'img': 'images/NGC2865_elliptical.jpeg'},

                                # --- SPIRAL (6) ---
                                {'type': 'Spiral', 'img': 'images/NGC1068_spiral.jpg'},
                                {'type': 'Spiral', 'img': 'images/NGC1232_spiral.jpg'},
                                {'type': 'Spiral', 'img': 'images/NGC_6217_spiral.jpg'},
                                {'type': 'Spiral', 'img': 'images/NGC5335_spiral.jpg'},
                                {'type': 'Spiral', 'img': 'images/NGC1365_spiral.jpg'},
                                {'type': 'Spiral', 'img': 'images/NGC1300_spiral.jpg'}
                            ]
                            
                            random.shuffle(memory_deck)
                            mem_state['cards'] = [{'id': i, 'type': c['type'], 'img': c['img']} for i, c in enumerate(memory_deck)]
                            render_memory_board.refresh()
                            
                        def stop_memory():
                            mem_state['running'] = False

                        def reset_memory():
                            stop_memory()
                            mem_state['time'] = 0
                            mem_state['score'] = 0
                            mem_state['flipped'].clear()
                            mem_state['matched'].clear()
                            mem_state['lock'] = False
                            lbl_time.set_text("Time: 0s")
                            lbl_score.set_text("Score: 0")
                          
                            mem_state['cards'] = []
                            render_memory_board.refresh()

                        with ui.row().classes('gap-4'):
                            aria_button("Start", "start", on_click=start_memory).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105 animate-pulse")
                            aria_button("Stop", "stop", on_click=stop_memory).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                            aria_button("Reset", "reset", on_click=reset_memory).classes("!bg-green-500 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-full shadow-lg transition-transform hover:scale-105")
                    @ui.refreshable
                    def render_memory_board():
                        if not mem_state['running'] and not mem_state['cards']:
                            ui.label("Press Start to reveal the cosmos!").classes("text-slate-400 italic text-center text-xl")
                            return

                        if len(mem_state['matched']) == len(mem_state['cards']) and len(mem_state['cards']) > 0:
                            ui.label("Universe Unlocked! All pairs found. 🌌").classes("text-3xl text-green-400 font-bold text-center mt-6").props('role=alert aria-live=assertive')
                            mem_state['running'] = False
                            return

                        current_client = ui.context.client

                        async def handle_match():
                            import asyncio
                          
                            await asyncio.sleep(1.0)
                            
                          
                            with current_client:
                              
                                if len(mem_state['flipped']) != 2:
                                    return
                                    
                                id1, id2 = mem_state['flipped'][0], mem_state['flipped'][1]
                                card1 = next(c for c in mem_state['cards'] if c['id'] == id1)
                                card2 = next(c for c in mem_state['cards'] if c['id'] == id2)
                                
                            
                                if card1['type'] == card2['type']:
                                    mem_state['matched'].update([id1, id2])
                                    mem_state['score'] += 1
                                    accessible_notify(f"Correct! Found two {card1['type']}.", color='positive', position='top', icon='check_circle')
                                else:
                                    mem_state['score'] -= 0.25
                                    accessible_notify("Wrong pair! Memorize and try again...", color='negative', position='top', icon='cancel')
                                    
                                lbl_score.set_text(f"Score: {mem_state['score']}")
                                
                              
                                mem_state['flipped'].clear()
                                mem_state['lock'] = False
                                render_memory_board.refresh()

                        def flip_card(card_id):
                           
                            if not mem_state['running'] or mem_state['lock'] or card_id in mem_state['flipped'] or card_id in mem_state['matched']:
                                return
                            
                           
                            mem_state['flipped'].append(card_id)
                            render_memory_board.refresh()
                            
                          
                            if len(mem_state['flipped']) == 2:
                                mem_state['lock'] = True
                                from nicegui import background_tasks
                                background_tasks.create(handle_match())

                      
                        with ui.grid(columns=8).classes('w-full gap-4'):
                            for card in mem_state['cards']:
                                is_revealed = card['id'] in mem_state['flipped'] or card['id'] in mem_state['matched']
                                label_stato = f"{card['type']} revealed" if is_revealed else "Hidden card"
                                with ui.card().style('aspect-ratio: 1').classes('p-0 cursor-pointer w-full flex items-center justify-center overflow-hidden transition-transform hover:scale-105 shadow-md border-2 border-slate-600').props(f'role=button tabindex=0 aria-label="Memory card: {label_stato}"').on('click', lambda c_id=card['id']: flip_card(c_id)).on('keydown.enter', lambda c_id=card['id']: flip_card(c_id)):
                                    if is_revealed:
                                        
                                        ui.image(card['img']).classes('w-full h-full object-cover').props('aria-hidden="true"')
                                    else:
                                       
                                        ui.element('div').classes('w-full h-full bg-blue-900 flex items-center justify-center').style('background-image: radial-gradient(circle, #1e3a8a, #0f172a);')
                                        ui.label('?').classes('text-white text-4xl font-bold absolute')

                    render_memory_board()
                        

            
            

        
        with ui.tab_panels(tabs, value='intro').classes('w-full !bg-transparent'):
            with ui.tab_panel('intro') as intro_panel:
                introduction_page(intro_panel)


            with ui.tab_panel('discovery') as discovery_panel:
                astronomy_timeline(discovery_panel)
            with ui.tab_panel('universe') as history_panel:
                cosmic_timeline(history_panel)
            with ui.tab_panel('instrument') as instrument_panel:
                instrument_page(instrument_panel)
            with ui.tab_panel('planets') as planets_panel:
                planets_page(planets_panel)
            with ui.tab_panel('galaxy') as galaxy_panel:
                galaxy_map_page(galaxy_panel)
            with ui.tab_panel('stars') as star_panel:
                hr_diagram_page(star_panel,STAR_GAIA_PATH)
            with ui.tab_panel('particles') as particle_panel:
                particle_page(particle_panel)

                
                
                
                




    