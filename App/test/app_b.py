
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
#inject_layout_tool()

_rho_rs_cache = {}
_cluster_calc_cache = {}

load_dotenv()


    
@app.get('/.well-known/appspecific/com.chrome.devtools.json')
async def chrome_devtools_stub():
    return {}

logging.getLogger("nicegui").setLevel(logging.ERROR)
client.Client.DEFAULT_JS_TIMEOUT = 3600
async def keep_alive():
    while True:
        await asyncio.sleep(60)
        try:
            ui.run_javascript('void(0);')  
        except Exception:
            pass  


app.on_startup(lambda: asyncio.create_task(keep_alive()))
def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) 
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


HF_API_TOKEN = os.getenv("HF_API_TOKEN")





_local_generator = None
_local_lock = threading.Lock()
#LOCAL_MODEL_NAME = "google/flan-t5-large"

#HUGGINGFACE_MODEL = "tiiuae/falcon-7b-instruct"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FILE = 'app_data.json'

app_data = {'users': {'admin': 'admin'}, 'reflection_log': []}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GALAXY_DATA_PATH = os.path.join(BASE_DIR, "galaxy_data")
GALAXY_IMG_PATH = os.path.join(BASE_DIR, "galaxy_img")
GALAXY_TABLES_PATH = os.path.join(BASE_DIR, "galaxy_tables")
CLUSTER_DATA_PATH = os.path.join(BASE_DIR, "cluster_data")
CLUSTER_IMG_PATH = os.path.join(BASE_DIR, "cluster_img")
CLUSTER_TABLES_PATH = os.path.join(BASE_DIR, "cluster_tables")
GALAXY_SPECTRA_PATH = os.path.join(BASE_DIR, "galaxy_spectra/csv_converted")
GALAXY_LINE_PATH = os.path.join(BASE_DIR, "galaxy_spectra/lambda_obs")
GALAXY_FITS_PATH = os.path.join(BASE_DIR, "galaxy_fits")
PLANETS_IMG_PATH = os.path.join(BASE_DIR, "planet_image")
GAL_SDSS_PATH = os.path.join(DATA_DIR, "SDSS_100.csv")
STAR_GAIA_PATH = os.path.join(DATA_DIR, "Gaia_20000.csv")
GALAXY_ZOO_PATH = os.path.join(DATA_DIR, "galaxy_zoo_classifications.csv")
MIST_PATH = os.path.join(BASE_DIR, "iso_fe0.01")
SDSS_MORPHO_PATH = os.path.join(DATA_DIR, "sdss_gal_morfo.txt")
app.add_static_files('/images', os.path.join(BASE_DIR, 'images'))
app.add_static_files('/galaxy_img', os.path.join(BASE_DIR, 'galaxy_img'))
app.add_static_files('/cluster_img', os.path.join(BASE_DIR, 'cluster_img'))
app.add_static_files('/slides', os.path.join(BASE_DIR, 'slides'))
app.add_static_files('/discovery_images', os.path.join(BASE_DIR, 'discovery_images'))
app.add_static_files('/cosmic_epochs', os.path.join(BASE_DIR, 'cosmic_epochs'))

def aria_button(text: str, label: str, **kwargs):
   
    return ui.button(text, **kwargs).props(f'role=button tabindex=0 aria-label={label}')

def aria_button2(text, label, **kwargs):
   
    tooltip_text = kwargs.pop('tooltip', None) 
    
   
    btn = ui.button(text, **kwargs).props(f'role=button tabindex=0 aria-label={label}')
    
   
    if tooltip_text:
        with btn:
            ui.tooltip(tooltip_text)
            
    return btn

def aria_image(src: str, alt_text: str, **kwargs):
    
    return ui.image(src, **kwargs).props(f'alt={alt_text}')


def aria_chart_label(description: str):

    return ui.label(description).props('class=sr-only role=note aria-hidden=false')


def aria_table(columns, rows, label: str, **kwargs):
    
    return ui.table(columns=columns, rows=rows, **kwargs).props(f'role=table tabindex=0 aria-label={label}')


def aria_input(label: str, aria_label: str, **kwargs):
   
    return ui.input(label, **kwargs).props(f'aria-label={aria_label} role=textbox tabindex=0')

def aria_textarea(label: str, aria_label: str, **kwargs):
   
    return ui.textarea(label, **kwargs).props(f'aria-label={aria_label} role=textbox tabindex=0')

def aria_slider(*, min_value=None, max_value=None, min=None, max=None, value=0.0, step=0.01, aria_label='', **kwargs):
    min_v = min_value if min_value is not None else min
    max_v = max_value if max_value is not None else max
    return ui.slider(min=min_v, max=max_v, value=value, step=step, **kwargs).props(
        f'role=slider aria-valuemin={min_v} aria-valuemax={max_v} aria-valuenow={value} aria-label={aria_label} tabindex=0'
    )



def aria_formula_input(**kwargs):
 
    return ui.input(label="", **kwargs).props('aria-label="Input the missing term in the formula" role=textbox tabindex=0')

def aria_navigate(path: str, message: str):
    ui.run_javascript(f"""
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('role','status');
        liveRegion.setAttribute('aria-live','polite');
        liveRegion.style.position='absolute';
        liveRegion.style.left='-9999px';
        liveRegion.innerText='{message}';
        document.body.appendChild(liveRegion);
    """)
    ui.navigate.to(path)
    ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")


def safe_click(*actions):
   
    def _handler():
        for act in actions:
            try:
                if callable(act):
                    act()
            except Exception as e:
                print(f"[safe_click] Errore durante l'esecuzione di {act}: {e}")
    return _handler




def load_data():
    """Loads application data from a JSON file."""
    global app_data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            try:
                app_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Data file is corrupted. Resetting data.")
                app_data = {'users': {'admin': 'admin'}, 'reflection_log': []}
 
    if 'users' not in app_data:
        app_data['users'] = {'admin': 'admin'}
    if 'reflection_log' not in app_data:
        app_data['reflection_log'] = []

def save_data():
    """Saves application data to a JSON file."""
    with open(DATA_FILE, 'w') as f:
        json.dump(app_data, f, indent=4)

load_data()



async def ask_ai(question: str, response_container: ui.column):

    response_container.clear()

    if not GROQ_API_KEY:
        with response_container:
            ui.markdown("⚠️ Nessuna API key trovata. Aggiungi `GROQ_API_KEY` nel file .env.")
        return

    with response_container:
        with ui.row().classes('items-center'):
            ui.spinner(size='lg')
            ui.label("AI Tutor is thinking...").props('role=status aria-live=polite')

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an AI tutor. Respond in the same language "
                "as the user’s question (English, Italian, etc.)."
            )
        },
        {"role": "user", "content": question},
    ],
    #max_tokens=300,
)


        answer = completion.choices[0].message.content.strip()
        response_container.clear()
        with response_container:
            ui.markdown(f"**AI Tutor (Groq llama-3.1-8b):**\n\n{answer}")

    except Exception as e:
        response_container.clear()
        with response_container:
            ui.markdown(f"❌ Errore Groq API: {e}")




def submit_reflection(module_name: str, text: str):
    """Saves user reflections to the application data."""
    if text:
        username = app.storage.user.get('name', 'Guest')
        entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | {username} | {module_name}: {text}"
        app_data['reflection_log'].append(entry)
        save_data()
        accessible_notify('Reflection saved!', type_='success')
    else:
        accessible_notify('Reflection cannot be empty.', type_='warning')




def main_layout(title: str):
    if not hasattr(ui, "_aria_landmarks_added"):
        ui.element('header').props('role=banner aria-label=Main application header')
        ui.element('nav').props('role=navigation aria-label=Main navigation menu')
        ui.element('main').props('role=main aria-label=Primary educational content area')
        ui.element('footer').props('role=contentinfo aria-label=Footer with credits and additional resources')
        ui._aria_landmarks_added = True
    with ui.dialog() as ai_dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('🤖 AI Tutor').classes('text-xl font-bold').props('role=heading aria-level=2')
        ui.markdown(f"Ask a question about: **{title}**")
        
        ai_q_input = aria_input('Your question...', "Ask a question to the AI tutor").classes('w-full')
        ai_response_container = ui.column().classes('w-full min-h-[50px] p-2 !bg-black-50 rounded mt-2')
        
        async def handle_ask_ai_popup():
            try:
                await ask_ai(ai_q_input.value, ai_response_container)
            except Exception as e:
                ai_response_container.clear()
                with ai_response_container:
                    ui.markdown(f"❌ Error: {e}")

        with ui.row().classes('w-full justify-end mt-2'):
            aria_button('Close', 'Close AI dialog', on_click=ai_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            aria_button('Ask', 'Send question to AI', on_click=handle_ask_ai_popup)
    with ui.dialog() as refl_dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('📝 Your Reflections').classes('text-xl font-bold').props('role=heading aria-level=2')
        ui.markdown(f"Write your thoughts on: **{title}**")
        
        refl_input = aria_textarea('Write here...', "Insert your reflection").classes('w-full')
        
        def handle_save_reflection():
            submit_reflection(title, refl_input.value)
            refl_input.value = "" # Pulisce il campo dopo il salvataggio
            refl_dialog.close()

        with ui.row().classes('w-full justify-end mt-2'):
            aria_button('Close', 'Close reflection dialog', on_click=refl_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            aria_button('Save', 'Save reflection', on_click=handle_save_reflection)
            
    drawer = ui.left_drawer(value=False).classes('!bg-gray-100 shadow-lg')
    with drawer:
        ui.label('🔭 Cosmo-Edu Navigation').classes('text-lg p-4 text-blue-600 font-bold').props('role=heading aria-level=2 tabindex=0')
        aria_button('🏠 Home', "Go to the homepage",
    on_click=safe_click(lambda: aria_navigate('/main', 'Navigating to main menu.'))).classes('w-full')


        aria_button('📚 Cosmology Modules', "Go to the cosmology modules",
    on_click=safe_click(lambda: aria_navigate('/main', 'Navigating to cosmology modules.'))
).classes('w-full')


        aria_button('🧭 Physics → Cosmology', "Analyze Physics and Cosmology topics",
    on_click=safe_click(lambda: aria_navigate('/physics-links', 'Navigating to Physics and Cosmology connections.'))
).classes('w-full')


        aria_button('🧪 Physics Curriculum', "Open Physics curriculum page",
    on_click=safe_click(lambda: aria_navigate('/physics-program', 'Navigating to Physics curriculum page.'))
).classes('w-full')


        
        ui.separator().classes('my-4')
        ui.label('🛠️ Tools').classes('px-4 text-gray-600 font-bold').props("aria-label=Tools section")

     
        aria_button('🤖 AI Tutor', "Open AI Tutor ",
            on_click=ai_dialog.open
        ).classes('w-full !bg-indigo-600 text-white')

        aria_button('✍️ Write Reflection', "Write Reflection ",
            on_click=refl_dialog.open
        ).classes('w-full !bg-green-600 text-white')
        aria_button('📝 Read Reflections', "Read user reflections",
    on_click=safe_click(lambda: aria_navigate('/reflections', 'Navigating to reflections page.'))
).classes('w-full')
        
        
    with ui.header(elevated=True).classes('!bg-primary text-white items-center justify-between'):
        with ui.row().classes('items-center'):
         
          
            aria_button2('Menu', 'Menu', 
                on_click=lambda: drawer.toggle(),
                tooltip='Toggle navigation menu'
            ).classes('mr-4').props('aria-expanded=false aria-controls=left-drawer')
            
        ui.label(title).classes('text-lg font-semibold')
        with ui.row().classes('items-center'):
            aria_button('🏠 Home', "Go to the homepage",
    on_click=safe_click(lambda: aria_navigate('/main', 'Navigating to home'))
).props('icon=home')

            aria_button('🔙 Back', "Go back to previous page",
    on_click=safe_click(lambda: ui.run_javascript('window.history.back()'))
).props('icon=arrow_back')
           
            if app.storage.user.get('name'):
                aria_button(f"Logout ({app.storage.user.get('name')})", "Logout and return to login page",
    on_click=safe_click(lambda: (app.storage.user.clear(), aria_navigate('/login', 'Login to the App')))).props('color=negative icon=logout')



   





@ui.page('/login')
def login_page():
    ui.run_javascript("setTimeout(() => document.querySelector('h1, .title, .text-2xl')?.focus(), 3600)")
    """Login page for users."""
    def try_login():
        if app_data['users'].get(username.value) == password.value:
            app.storage.user['name'] = username.value
            ui.navigate.to('/main')
        else:
            accessible_notify('Invalid username or password', type_='warning')

    with ui.card().classes('absolute-center'):
        ui.label('Login').classes('text-2xl font-bold').props('role=heading aria-level=2 aria-label=Login form area tabindex=0')
        username = aria_input('Username',"Insert the username").on('keydown.enter', lambda: password.focus())
        password = aria_input('Password',"Insert the password", password=True).on('keydown.enter', try_login)
        aria_button('Login', "Log in", on_click=try_login)
        ui.link("Don't have an account? Register", '/register').classes('mt-2')

@ui.page('/register')
def register_page():
    """Registration page for new users."""
    def do_register():
        if not new_user.value or not new_pass.value:
            accessible_notify('Username and password are required.', type_='warning')
            return
        if new_user.value in app_data['users']:
            accessible_notify('This username is already taken.', type_='warning')
            return
        app_data['users'][new_user.value] = new_pass.value
        save_data()
        accessible_notify('User registered! You can now log in.', type_='success')
        ui.navigate.to('/login')

    with ui.card().classes('absolute-center'):
        ui.label('Register New Account').classes('text-2xl font-bold').props('role=heading aria-level=2 tabindex=0')
        new_user = aria_input('New Username', "Insert the new username")
        new_pass = aria_input('New Password',"Insert the new password", password=True)
        aria_button('Register', "Create a new account", on_click=do_register)

@ui.page('/')
def index_page():
    """Redirects to login or main menu based on authentication status."""
    if not app.storage.user.get('name'):
        ui.navigate.to('/login')
    else:
        ui.navigate.to('/main')


@ui.page('/main')
def main_menu():
    #ui.run_javascript('window.__nicegui__.requestTimeout = 5000')
    main_layout(" COSMO-EDU-LAB")

    with ui.column().classes('w-full items-center p-8'):
        ui.label(f"Welcome, {app.storage.user.get('name', 'Explorer')}!").classes('text-3xl m-4').props('role=heading aria-level=2 tabindex=0')


       
        
        ui.label('\"The cosmos is within us. We are made of star-stuff.\" — Carl Sagan').classes('italic text-lg mt-2').props('role=heading aria-level=2 tabindex=0')

        ui.markdown('Select a module to begin your journey into cosmology.')
        with ui.grid(columns=4).classes('gap-4 m-4 w-full'):
            module_titles = [
                "Introduction","Redshift & Universe Expansion","Universe History & CMB" ,"Dark Matter"
            ]
            for i, title in enumerate(module_titles, 1):
                with ui.card().classes('w-full hover:shadow-xl transition'):
                    ui.label(f'Module {i}').classes('text-xl font-bold').props('role=heading aria-level=2 tabindex=0')
                    ui.label(title).classes('text-sm text-gray-600')
                    aria_button(f'Explore Module {i}', "Go to module",
    on_click=safe_click(lambda i=i: (ui.navigate.to(f'/module{i}'),  ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")))).classes('mt-4 w-full')

       
        aria_button('🧭 Physics → Cosmology Connections', "Open classical physics topics",
    on_click=safe_click(lambda: (ui.navigate.to('/physics-links'), ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")))).classes('mt-4')

        aria_button('🧪 Physics Curriculum', "Open the physics curriculum",
    on_click=safe_click(lambda: (ui.navigate.to('/physics-program'), ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")))).classes('mt-2')

        aria_button('📝 View Reflections', "View the user's reflections",
    on_click=safe_click(lambda: (ui.navigate.to('/reflections'),  ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")))).classes('mt-2')



@ui.page('/physics-links')
def physics_links():
    main_layout("Classical Physics → Cosmology Connections")

    table_data = [
       
         ("1. Redshift & Universe Expansion", "Doppler effect,optics, velocity-distance relationships, waves, frequency,flux, wavelenght,relativistic velocity ","Redshift, standard candles,supernovae Ia, luminosity, Hubble’s law, cosmic acceleration,universe expansion, Dark energy, lambda-CDM model", "/module2"),
        ("2. Universe History & CMB", "Thermodynamics, radiation, blackbody spectrum, adiabatic transformation, adiabatic index, Stephen-Boltzmann law", "Thermal history, CMB, Plank spectrum, universe evolution, radiation and matter density, equivalence,photons", "/module3"),
        
       
         ("3. Dark Matter ", "Circular motion, gravity, energy conservation, centripetal force, gravitational force, Newton, Kepler laws, kinetic energy,gravitational potential energy", "Rotation curves, missing mass, virial theorem, dark matter, universe composition", "/module4")
        
       
    ]
    rows = [
        {'classical': title, 'topics': classical} for title, classical, cosmo, link in table_data
    ]
    with aria_table(
        columns=[
            {'name': 'classical', 'label': 'Classical Physics', 'field': 'classical', 'sortable': True},
            {'name': 'topics', 'label': 'Cosmology Topics', 'field': 'topics', 'sortable': False},
        ],
        rows=rows,
        label="Connections between classical physics and cosmology topics"
    ).classes('w-full'):
        pass

    with ui.row():
        for title, classical, cosmo, link in table_data:
            aria_button(f"Go to {title}","Got to cosmology module", on_click=safe_click(lambda l=link: (ui.navigate.to(l),  ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")))).classes('mr-4')


   


@ui.page('/physics-program')
def physics_program():
    main_layout("Physics Curriculum → Cosmology Modules")

    physics_curriculum = [
        ("1st–2nd Year", [
            ("Scalar and vectorial measures and units", "/module4"),
            ("Geometry optics: reflection and refraction", "/module2"),
            ("Thermal phenomena: temperature, heat, thermal balance", "/module3"),
            ("Mechanics: motion, forces, Newton’s laws", "/module4"),
        ]),
        ("3rd–4th Year", [
            ("Relativity: reference systems, principles", "/module2"),
            ("Conservation laws: energy, fluids", "/module4"),
            ("Gravitation: Kepler, Newton, cosmological systems", "/module4"),
            ("Thermodynamics: gas laws, kinetic theory", "/module3"),
            ("Waves: mechanical, interference, diffraction", "/module2"),
            ("Electromagnetism: fields, energy, potential", "/module2"),
        ]),
        ("5th Year", [
            ("Electromagnetism: induction, Maxwell, EM waves", "/module2"),
            ("Micro & macro cosmos: space-time, energy", "/module2"),
            ("Relativity: Einstein’s principles, dilation, contraction", "/module2"),
            ("Quantum light: Planck, photoelectric, De Broglie", "/module3"),
        ]),
    ]

    for year, topics in physics_curriculum:
        with ui.expansion(year, icon='school').classes('w-full').props(f'role=region aria-label="{year} physics topics" aria-expanded=false tabindex=0') as exp:
            ui.run_javascript(f"""
    const exp = document.querySelector('[aria-label="{year} physics topics"]');
    if (exp) exp.addEventListener('click', () => {{
        exp.setAttribute('aria-expanded', exp.open);
    }});
""")

            for topic, link in topics:
                if link:
                    aria_button(
                    topic,
                    "Go to cosmology topic linked to physics topic",
                    on_click=safe_click(lambda l=link: (
                        ui.run_javascript("""
                            const liveRegion = document.createElement('div');
                            liveRegion.setAttribute('role','status');
                            liveRegion.setAttribute('aria-live','polite');
                            liveRegion.style.position='absolute';
                            liveRegion.style.left='-9999px';
                            liveRegion.innerText='Navigating to cosmology module.';
                            document.body.appendChild(liveRegion);
                        """),
                        ui.navigate.to(l),
                        ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")
                    )
                )).classes('w-full text-left')
                else:
                    ui.label(topic).classes('text-gray-600 ml-4')



@ui.page('/reflections')
def all_reflections():
    main_layout("Student Reflections")

    user = app.storage.user.get('name')
    if user == 'admin':
        ui.label('📒 All reflections (admin view)').classes('text-xl').props('role=heading aria-level=2 tabindex=0')
        for entry in app_data.get('reflection_log', []):
            ui.markdown(entry)
    else:
        ui.label('🗒️ Your reflections').classes('text-xl').props('role=heading aria-level=2 tabindex=0')
        for entry in app_data.get('reflection_log', []):
            if f"| {user} |" in entry:
                ui.markdown(entry)



def title(title: str):
    """Titolo principale (blu scuro, grande, bold)"""
    el= ui.element('h2').classes("title").props('tabindex=0 role=heading aria-level=2')
    with el:    
        ui.label(title)
    return el
def subtitle(text: str):
    """Sottotitolo (blu acceso, medio-grande)"""
    el= ui.element('h3').classes("subtitle").props('tabindex=0 role=heading aria-level=3')
    with el:
        ui.label(text)  
    return el

def section(text: str):
    """Titolo di sezione (grigio scuro, medio)"""
    el= ui.element('h3').classes("section").props(
        'tabindex=0 role=heading aria-level=3'
    )
    with el:
        ui.label(text)
    return el



def info_box(text: str):
    """Box informativo (sfondo chiaro, bordo blu, testo scuro)"""
    return ui.markdown(text).classes("info-box").props(
        'role=note aria-label=Information box tabindex=0'
    )

def warning_box(text: str):
    """Box avviso (arancione)"""
    return ui.markdown(text).classes("warning-box").props(
        'role=alert aria-live=assertive tabindex=0')

def success_box(text: str):
    """Box positivo / conferma (verde)"""
    return ui.markdown(text).classes("success-box").props(
        'role=status aria-live=polite tabindex=0'
    )
def reference_box(text: str):
    """Box per reference e bibliografia (grigio elegante, italico)"""
    return ui.markdown(text).classes("reference-box").props(
        'role=doc-biblioentry tabindex=0 aria-label=Reference section'
    )
def plot_info_box(info: dict, title: str = "📊 Results"):
    with ui.card().classes(
        "p-4 !bg-slate-50 border border-slate-300 rounded-lg shadow-md w-full max-w-2xl mt-2"
    ).props('role=region aria-label=Plot results summary tabindex=0'):
        with ui.element('h4').classes("text-lg font-bold text-slate-700 mb-2").props('tabindex=0 role=heading aria-level=4'):
            ui.label(title)
        for label, value in info.items():
            ui.label(f"{label}: {value}").classes("text-base font-medium text-slate-800").props('role=heading aria-level=2 tabindex=0')


def title_on_dark(title: str):
    card = ui.card().classes(
        "w-fit mx-auto px-8 py-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg mb-1"
    )
    with card:
        el= ui.element('h2').classes("text-4xl font-bold text-center mb-1 title title-on-dark").props(
        'tabindex=0 role=heading aria-level=2'
    )
    with el:
        ui.label(title)
    return el


def plot_info_box_compact2(info: dict, title: str = None, compact: bool = True):
    # Aggiungi le classi di posizionamento assoluto e lo z-index
    base_classes = "plot-info-box absolute top-2 right-2 z-10" 
    
    if compact:
        base_classes += " compact2" # Per agganciare lo stile CSS personalizzato

    # Il contenitore avrà: posizionamento, z-index, e le classi personalizzate
    with ui.element("div").classes(base_classes).props(
        'role=region aria-label=Compact plot results summary tabindex=0'
    ):
        # Stile per il titolo (opzionale)
        if title:
            with ui.element('h3').classes(
                "text-sm font-semibold text-slate-700"
            ).style("margin-bottom:6px;").props('tabindex=0 role=heading aria-level=3'):
                ui.label(title)

        with ui.element("div").props('role=list'):
            for lab, val in info.items():
                with ui.element("div").classes("info-row").props('role=listitem'):
                    # Stile per l'etichetta (label)
                    ui.label(str(lab)).classes("label text-slate-600") 
                    # Stile per il valore (value)
                    ui.label(str(val)).classes("value font-mono font-semibold text-slate-800")

def plot_info_box_compact(info: dict, title: str = None, compact: bool = True):
  
    base_classes = "plot-info-box"
    if compact:
        base_classes += " compact"

    with ui.element("div").classes(base_classes).props(
        'role=region aria-label=Compact plot results summary tabindex=0'
    ):
        if title:
            with ui.element('h3').classes(
                "text-sm font-semibold text-slate-700"
            ).style("margin-bottom:6px;").props('tabindex=0 role=heading aria-level=3'):
                ui.label(title)

        with ui.element("div").props('role=list'):
            for lab, val in info.items():
                with ui.element("div").classes("info-row").props('role=listitem'):
                    ui.label(str(lab)).classes("label")
                    ui.label(str(val)).classes("value")
                    
last_notification_card = None 
def accessible_notify(text: str, type_: str = "info"):

    global last_notification_card
    if last_notification_card is not None:
        try:
          
            last_notification_card.delete()
        except Exception as e:
     
            pass 
    color = {
        "info": "!bg-blue-50 text-blue-800 border-blue-400",
        "success": "!bg-green-50 text-green-800 border-green-400",
        "warning": "!bg-yellow-50 text-yellow-800 border-yellow-400",
        "error": "!bg-red-50 text-red-800 border-red-400",
    }[type_]

    with ui.card().classes(f"p-3 border rounded-lg shadow-sm mt-2 {color}").props(
        f'role={"alert" if type_ in ["error","warning"] else "status"} aria-live=polite tabindex=0'
    ) as notification_card: 
        ui.label(text)

    last_notification_card = notification_card
    
   
    def close_and_reset():
        global last_notification_card
       
        try:
            notification_card.delete() 
        except Exception as e:
           
            pass 
     
        if last_notification_card == notification_card:
            last_notification_card = None
            
    ui.timer(4.0, close_and_reset, once=True)

    

@ui.page('/module1')
def introduction():
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
}


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

    ui.add_body_html("""
<script>
if (!window.MathJaxLoaded) {
window.MathJaxLoaded = true;
window.MathJax = {
    tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]},
    svg: {fontCache: 'global'}
};
var s = document.createElement('script');
s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
s.async = true;
document.head.appendChild(s);
}
</script>
""")
    main_layout("Module 1: Introduction to Cosmo-Edu Lab")
    with ui.tabs().classes('w-full justify-center') as tabs:
        one= ui.tab(' Introduction').props('aria-label="Introduction"')
        two= ui.tab('Discovery Timeline').props('aria-label="Discovery Timeline"')
        three = ui.tab('Universe Timeline').props('aria-label="Universe Timeline"')
        four = ui.tab(' Galaxy Map').props('aria-label="Galaxy Map"')
        five = ui.tab(' Stars Map').props('aria-label="Stars Map"')
        six=ui.tab(' Fundamental Particles').props('aria-label="Fundamental Particles"')
   
    
    IMAGES_MACRO = [
        {'file': 'images/spiral_galaxy.jpg', 'title': 'Spiral Galaxy',
        'description': 'A spiral galaxy rich in gas, dust, and billions of stars (Messier 77) [NASA](https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/messier-77/).', 'size': 9.5e17},

        {'file': 'images/elliptical_galaxy.jpg', 'title': 'Elliptical Galaxy',
        'description': 'A smooth, featureless galaxy composed mostly of old stars (Messier 32) [UniverseToday](https://www.universetoday.com/articles/messier-32).', 'size': 3e17},

        {'file': 'images/irregular_galaxy.jpg', 'title': 'Irregular Galaxy',
        'description': 'A chaotic galaxy without a defined structure (NGC 55) [ESO](https://www.eso.org/public/italy/images/eso0914a/).', 'size': 6e17},
        {'file': 'images/lenticular_galaxy.jpg', 'title': 'Lenticular Galaxy',
        'description': 'A galaxy with a central bulge and disk but no spiral arms (NGC 6861) [SciNews](https://www.sci.news/astronomy/science-hubble-space-telescope-lenticular-galaxy-ngc6861-02416.html).', 'size': 5e17},

        {'file': 'images/nebula_emission.jpg', 'title': 'Emission Nebula',
        'description': 'Clouds of ionized gas glowing due to nearby young stars (N44) [Wikipedia-ESA](https://en.wikipedia.org/wiki/N44_%28emission_nebula%29).', 'size': 9e16},

        {'file': 'images/nebula_dark.jpg', 'title': 'Dark Nebula',
        'description': 'Dense, cold clouds of dust blocking background starlight (Bernard 3) [AstroCarballada](https://astro.carballada.com/barnard_3/).', 'size': 3e16},

        {'file': 'images/protoplanetary_disk.jpg', 'title': 'Protoplanetary Disk',
        'description': 'A disk of gas and dust where new planets are forming (HL Tauri) [Wikipedia-ALMA](https://en.wikipedia.org/wiki/Protoplanetary_disk).', 'size': 3e12},

        {'file': 'images/star.jpg', 'title': 'Main Sequence Star',
        'description': 'A stable star fusing hydrogen into helium in its core (Sun) [BBC](https://www.skyatnightmagazine.com/space-science/main-sequence-stars).', 'size': 1.39e6},

        {'file': 'images/red_giant.jpg', 'title': 'Red Giant',
        'description': 'A dying star that has swollen to enormous size (CW Leonin) [SciNews](https://www.sci.news/astronomy/hubble-cw-leonis-10219.html).', 'size': 1e8},

        {'file': 'images/supernova.jpg', 'title': 'Supernova Explosion',
        'description': 'A catastrophic stellar explosion releasing heavy elements (Cassiopea A) [NASA-JLP-Caltech](https://www.schoolsobservatory.org/discover/projects/supernovae/examples).', 'size': 1e17},

        {'file': 'images/black_hole.jpg', 'title': 'Black Hole',
        'description': 'A region where gravity is so strong that nothing can escape (Messier 87) [Wikipedia](https://en.wikipedia.org/wiki/Black_hole).', 'size': 3e10},

        {'file': 'images/quasar.jpg', 'title': 'Quasar',
        'description': 'An extremely luminous active galactic nucleus powered by a supermassive black hole (J0749 2255) [NASA](https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-quasars/).', 'size': 1e14},

        {'file': 'images/exoplanet.jpg', 'title': 'Exoplanet',
        'description': 'A planet orbiting a star outside our solar system (2M1207b) [NASA](https://science.nasa.gov/resource/2m1207-b-first-image-of-an-exoplanet/).', 'size': 1e5},

        {'file': 'images/planet.jpg', 'title': 'Planet',
        'description': 'A world orbiting a star, possibly with an atmosphere and moons (Jupiter) [ESA](https://www.esa.int/Science_Exploration/Space_Science/Juice/Hello_Jupiter!_How_to_observe_a_gas_giant).', 'size': 1.4e5},

        {'file': 'images/moon.jpeg', 'title': 'Moon',
        'description': 'A natural satellite orbiting a planet (Moon) [NASA](https://science.nasa.gov/moon/viewing-guide/).', 'size': 3474},

        {'file': 'images/comet.jpg', 'title': 'Comet',
        'description': 'Ice-rich bodies that form glowing tails when near a star (Halley) [Wikipedia](https://en.wikipedia.org/wiki/Halley%27s_Comet).', 'size': 20},

        {'file': 'images/asteroid.jpg', 'title': 'Asteroid',
        'description': 'A small rocky body left over from the formation of the solar system (Psyche) [NASA](https://science.nasa.gov/solar-system/asteroids/16-psyche/).', 'size': 226},

        {'file': 'images/meteor.jpg', 'title': 'Meteor',
        'description': 'A piece of debris burning in a planet’s atmosphere. [BBC](https://www.skyatnightmagazine.com/astrophotography/observe-photograph-meteor-train)', 'size': 0.01},

        {'file': 'images/globular_cluster.jpg', 'title': 'Globular Cluster',
        'description': 'A dense spherical group of very old stars (NGC 2298) [NASA](https://science.nasa.gov/image-detail/hubble-ngc2298-acs-wfc3-v3-5fcont-final/).', 'size': 3e15},

        {'file': 'images/galaxy_cluster.jpg', 'title': 'Galaxy Cluster',
        'description': 'A massive structure containing hundreds or thousands of galaxies (Abell 370) [NASA-ESA-Harvard](https://www.cfa.harvard.edu/research/topic/galaxy-clusters).', 'size': 3e22},
        {'file': 'images/galaxy_group.jpg', 'title': 'Galaxy Group',
         'description': 'A small collection of galaxies bound by gravity (Local Group) [BBC](https://www.skyatnightmagazine.com/space-science/local-group-guide-galaxy-neighbourhood).', 'size': 3e19},

        {'file': 'images/filaments.jpg', 'title': 'Cosmic Filament (Simulation)',
        'description': 'One of the largest structures in the universe, forming the cosmic web [SciTech](https://scitechdaily.com/first-direct-image-of-the-cosmic-web-reveals-the-universes-hidden-highways/).', 'size': 3e23},
        {'file': 'images/voids.jpg', 'title': 'Cosmic Void (Simulation)',
        'description': 'Vast, empty regions of space with very few galaxies [LiveScience](https://www.livescience.com/65928-stare-into-the-fuzzy-dark-void.html).', 'size': 3e24},
        {'file': 'images/cosmic_web.jpg', 'title': 'Cosmic Web (Simulation)',
        'description': 'The large-scale structure of the universe, composed of filaments and voids [UniverseToday](https://www.universetoday.com/articles/astronomy-without-a-telescope-the-edge-of-greatness).', 'size': 1e25}
    ]
    def markdown_to_html(md_text):
        import re
        # Trasforma [testo](url) in <a href="url" target="_blank">testo</a>
        return re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', md_text)

    def create_interactive_tile(image_file: str, title: str, description: str, size: int):
       
        with ui.dialog() as popup, ui.card():
            ui.label(title).classes('text-h5').props('role=heading aria-level=3 tabindex=0')
            aria_image(image_file, title).classes('w-96 h-auto')
            ui.markdown(markdown_to_html(description)).classes('mt-2').props('tabindex=0 aria-label="Description" role=document')
            aria_button('Close','Close image popup', on_click=lambda: popup.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
        
        with ui.card().classes('p-2 hover:!bg-gray-200 transition-all cursor-pointer'):
            thumb = aria_image(image_file, title).classes('w-40 h-40 object-cover rounded-md cursor-pointer')
            thumb.on('click', lambda: popup.open())
           
            thumb.props('draggable="true"')
            thumb.props(f'ondragstart="event.dataTransfer.setData(\'application/json\', JSON.stringify({{file:\'{image_file}\', title:\'{title}\', size:{size}}}))"')

    


    def introduction_page(container):
        with container:

            title_on_dark("Welcome to Cosmo-Edu Lab: Getting Started")
            
            def open_slides_cosmo():
                ui.run_javascript('window.open("/slides/Cosmology.pdf", "_blank")')
            def open_activities():
                ui.run_javascript('window.open("/slides/Cosmo-Edu-Lab Activities.pdf", "_blank")')
            
            with ui.dialog() as intro, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                info_box(
                "**Cosmo-Edu Lab** is an interactive educational platform designed to explore the world of cosmology. "
                "Through a series of modules, users can learn about key cosmology concepts and their connections to classical physics."
                "\n\nTo begin your journey into cosmology, navigate to the main menu and select a module that interests you. "
                "You can ask questions to our AI Tutor for additional explanations and save your reflections on each topic."
                "\n\n**Tip:** Use the navigation drawer on the left to easily access different sections of the platform. Make sure to log in or register an account to save your progress and reflections."
            
            )
                aria_button("Close", "close the box",on_click=intro.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            DROP_BOXES = []

            def handle_macro_drop(box, data):
                size = float(data['size'])
                title = data['title']
                file = data['file']

                scale = max(40, min(120, size / 1e9 * 120))  
                with ui.column().classes('items-center p-1') as tile:
                    ui.image(file).style(f"width:{scale}px; height:{scale}px; object-fit:cover;")
                    ui.label(title).classes("text-xs")
                box._content.add(tile)

            def reset_drop_boxes():
                for box in DROP_BOXES:
                    box._content.refresh() 
            with ui.row().classes('w-full justify-between item-center'):
                with ui.element('div').classes("text-xl font-bold flex items-center h-10 px-4"):
                    ui.markdown("**MACRO-COSMOS**: Structures of the Universe").props('role=heading aria-level=2 tabindex=0')
                aria_button(
            'Instruction', 'Open Instruction',
            on_click=intro.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                aria_button(
            'Open Introductory Slides:Cosmology', 'Open Introductory Slides:Cosmology',
            on_click=open_slides_cosmo).props('target=_blank').classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                aria_button(
            'Open Cosmo-Edu Lab Activities', 'Open Cosmo-Edu Lab Activities',
            on_click=open_activities).props('target=_blank').classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                aria_button("Reset", "reset", on_click=reset_drop_boxes).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
            with ui.row().classes('w-full flex justify-center'):
                aria_button('Explore the Solar System',"Explore the Solar System",
                on_click=safe_click(lambda: ui.run_javascript("window.open('https://www.solarsystemscope.com/', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button('Explore Universe Scales',"Explore Universe Scales",
                on_click=safe_click(lambda: ui.run_javascript("window.open('https://scaleofuniverse.com/en', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button('Explore the Universe',"Explore the Universe",
                on_click=safe_click(lambda: ui.run_javascript("window.open('https://theskylive.com/3dsolarsystem', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button('Explore the Stars',"Explore the Stars",
                on_click=safe_click(lambda: ui.run_javascript("window.open('https://stellarium-web.org/', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button('NASA simulators',"NASA simulators",
                on_click=safe_click(lambda: ui.run_javascript("window.open('https://science.nasa.gov/eyes/', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
        
                
            

            

            with ui.row().classes("w-full gap-6"):


                with ui.column().classes("flex-[3]"):
  
                    with ui.grid(columns=6).classes('gap-4 p-4'):
                        for obj in IMAGES_MACRO:
                            create_interactive_tile(obj['file'], obj['title'], obj['description'], obj['size'])

                with ui.column().classes("flex-[1] gap-4"):
                    for label, min_s, max_s in [
                        ('0-1000 km', 0, 1e3),
                        ('1000-10^6 km', 1e3, 1e6),
                        ('10^6-10^9 km', 1e6, 1e9),
                        ('10^9-10^18 km', 1e9, 1e18),
                        ('>10^18 km', 1e18, 1e26),
                    ]:
                        with ui.card().classes('p-3 h-35 w-full relative overflow-auto') as box:
                            DROP_BOXES.append(box)
                            ui.label(label).classes('font-bold').props('tabindex=0 role=heading aria-level=3')
                            
                    
                      
                            @ui.refreshable
                            def box_content(m_label=label, m_min=min_s, m_max=max_s):
                                
                                drop_zone = ui.column().classes("gap-2 mt-2 flex-wrap items-center w-full h-full min-h-[50px]")
                       
                                drop_zone.props(f'ondragover="event.preventDefault();" '
                                      f'ondrop="event.preventDefault(); '
                                      f'const data=JSON.parse(event.dataTransfer.getData(\'application/json\')); '
                                      f'const size=parseFloat(data.size); '
                                
                                      f'if(size<{m_min}||size>{m_max}){{alert(data.title+\' does not belong to {m_label}\');return;}} '
                                      f'const scale=Math.max(40,Math.min(120,size/{m_max}*120)); '
                                      f'const col=document.createElement(\'div\'); '
                                      f'col.style.display=\'flex\';col.style.flexDirection=\'column\';col.style.alignItems=\'center\';col.style.margin=\'2px\'; '
                                      f'const img=document.createElement(\'img\'); '
                                      f'img.src=data.file;img.width=scale;img.height=scale;img.style.objectFit=\'cover\';img.className=\'rounded\'; '
                                      f'const lbl=document.createElement(\'div\'); lbl.innerText=data.title; lbl.style.fontSize=\'10px\'; '
                                      f'col.appendChild(img); col.appendChild(lbl); '
                                      f'this.appendChild(col);"')

                            box_content()
                            box._content = box_content




            




                       
                    
      
            

    
        

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
    ("bb","bbn", {"interaction":"Outcome"}),

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
    

        with ui.dialog() as mass_conversion_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
            
        
            ui.html(r"""
                <div style="font-family: Arial, sans-serif;">
                    <h4 class="text-xl font-bold">Mass-Energy Conversion: kg to eV</h4>
                    <p>In particle physics, mass ($m$) is often expressed as equivalent energy ($E$) using <b>Einstein's mass-energy equivalence</b> principle:</p>
                    
                    <p style="text-align:center;"><span class="math">\( E = m c^2 \)</span></p>

                    <p>Where $c$ is the speed of light in a vacuum.</p>

                    <hr class="my-3">

                    <h5 class="text-lg font-semibold">Conversion Steps</h5>
                    <ol>
                        <li>
                            <b>Mass (kg) to Energy (Joule):</b>
                            <p style="text-align:center;"><span class="math">\( E \, (\text{J}) = m \, (\text{kg}) \times (c)^2 \)</span></p>
                            <p>($c \approx 2.9979 \times 10^8 \text{ m/s}$)</p>
                        </li>
                        <li>
                            <b>Energy (Joule) to Electronvolt (eV):</b>
                            <p>The energy in Joules must be divided by the elementary charge ($e$) to get Electronvolts.</p>
                            <p style="text-align:center;"><span class="math">\( E \, (\text{eV}) = \frac{E \, (\text{J})}{e} \)</span></p>
                            <p>($e \approx 1.60217 \times 10^{-19} \text{ J/eV}$)</p>
                        </li>
                    </ol>

                    <hr class="my-3">

                    <h5 class="text-lg font-semibold">Key Conversion Factors</h5>
                    <p>The standard convention uses prefixes to denote magnitude:</p>
                    <table class="w-full text-sm text-left text-gray-700 mt-2" border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin:auto; text-align:center;">
                    <tr class="!bg-gray-200">
                        <th>Unit</th>
                        <th>Value in eV</th>
                        <th>Example</th>
                    </tr>
                    <tr><td>$\text{keV}$ (kilo)</td><td>$10^3 \, \text{eV}$</td><td>Binding energies of inner-shell electrons.</td></tr>
                    <tr><td>$\text{MeV}$ (mega)</td><td>$10^6 \, \text{eV}$</td><td>Electron rest mass ($\approx 0.511 \, \text{MeV}$)</td></tr>
                    <tr><td>$\text{GeV}$ (giga)</td><td>$10^9 \, \text{eV}$</td><td>Proton rest mass ($\approx 0.938 \, \text{GeV}$)</td></tr>
                    </table>
                    <p class="mt-2 text-xs text-gray-500">
                    The factor for $\mathbf{1 \text{ kg}}$ mass is approximately $\mathbf{5.61 \times 10^{35} \text{ eV}}$.
                    </p>
                </div>
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

              
                if info.get('shape') == 'square': 
                    text = f"<b>{info.get('label','')}</b><br>{info.get('desc','')}"
                else:  
                    text = f"<b>{info.get('label','')}</b><br>" \
                        f"Mass: {info.get('mass','N/A')}<br>" \
                        f"Charge: {info.get('charge','N/A')}<br>" \
                        f"Spin: {info.get('spin','N/A')}<br>" \
                        f"{info.get('desc','')}"
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
                desc = data.get("desc", "")
                hover = f"<b>{interaction}</b><br>{G.nodes[source].get('label', source)} → {G.nodes[target].get('label', target)}<br>{desc}"
                edge_hovertext.append(hover)

                annotations.append(
                    dict(
                        x=x1, y=y1, ax=x0, ay=y0,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1.4,
                        arrowcolor="#444",
                        opacity=0.9,
                        standoff=6,
                        text=interaction if interaction else "",
                        font=dict(size=10),
                        hovertext=desc if desc else "",
                        hoverlabel=dict(bgcolor="white", font=dict(color="black"))
                    )
                )


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
                desc = data.get("desc", "")
                hover = f"<b>{interaction}</b><br>{G.nodes[source].get('label', source)} → {G.nodes[target].get('label', target)}<br>{desc}"
                edge_hovertext.append(hover)

                annotations.append(
                    dict(
                        x=x1, y=y1, ax=x0, ay=y0,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1,
                        arrowwidth=1.4,
                        arrowcolor="#444",
                        opacity=0.9,
                        standoff=6,
                        text=interaction if interaction else "",
                        font=dict(size=10),
                        hovertext=desc if desc else "",
                        hoverlabel=dict(bgcolor="white", font=dict(color="black"))
                    )
                )


      
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#888"),
            mode='lines',
            hoverinfo='text',
            hovertext=[h for h in edge_hovertext for _ in (0,1,2)],
            showlegend=False,
            opacity=0.9
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
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=colors,
                size=sizes,
                line_width=2,
                symbol=symbols
            ),
            showlegend=False
        )

      
        fig = go.Figure(data=[edge_trace, node_trace])

   
        fig.update_layout(
        title=title,
        title_x=0.5,
        width=width,  
        height=height,
        autosize=True, 
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations
    )

        plot_width = "100%" if is_process_graph else "600px"
        ui.plotly(fig).classes('w-full h-full').style(f"height:{height}px; width: {plot_width}")

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

    def plot_mass_strip(container, height=280, width=None, title="Particle masses"):

     
        categories = [
            ("Leptons", leptons_nodes, "#7b61ff"),
            ("Quarks", quarks_nodes, "#ff6b6b"),
            ("Bosons", bosons_nodes, "#ffd54f"),
            ("Hadrons", hadrons_nodes, "#4da6ff"),
        ]

      
        fig = go.Figure()
        y_positions = {cat[0]: i for i, cat in enumerate(reversed(categories))}  
        
        ymin = -0.5
        ymax = len(categories) - 0.5
        for cat_name, _, color in categories:
            y = y_positions[cat_name]
            fig.add_shape(
                type="rect",
                x0=1e-9, x1=1e18,  
                y0=y-0.45, y1=y+0.45,
                fillcolor=color,
                opacity=0.06,
                line_width=0,
                layer="below",
            )

      
        xs = []
        traces = []
        for cat_name, node_list, root_color in categories:
            y = y_positions[cat_name]
            px = []
            ph = []
            labels = []
            masses_eV = []
            for n in node_list:
              
                if n.get("shape") in ("diamond", "square") and (n.get("label", "").upper() == cat_name.upper()):
                    continue
                mass_str = n.get("mass", "N/A")
                mass_eV, mass_label = _parse_mass_to_eV(mass_str)
              
                if mass_eV == 0.0:
                    plot_x = 1e-12 
                    mass_display = f"{mass_label} (massless)"
                elif mass_eV is None:
                    plot_x = None
                    mass_display = f"{mass_label}"
                else:
                    plot_x = mass_eV
                    mass_display = f"{mass_label}"
                if plot_x is not None:
                    xs.append(plot_x)
                    masses_eV.append(plot_x)
                    px.append(plot_x)
                    labels.append(n.get("label", n.get("id")))
                    ph.append(f"<b>{n.get('label','')}</b><br>Category: {cat_name}<br>Mass: {mass_label}<br>Charge: {n.get('charge','N/A')}")
            if px:
                traces.append(
                    go.Scatter(
                        x=px,
                        y=[y + np.random.normal(scale=0.06) for _ in px],  
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=root_color,
                            line=dict(width=1, color="black"),
                            symbol="circle"
                        ),
                        hoverinfo="text",
                        hovertext=ph,
                        name=cat_name,
                        showlegend=False
                    )
                )

    
        for tr in traces:
            fig.add_trace(tr)

       

        if len(xs) == 0:
            x_min = 1e-12
            x_max = 1e3
        else:
            x_min = min(xs) * 0.1
            x_max = max(xs) * 10.0

      
        ytickvals = [y_positions[c[0]] for c in categories]
        yticktext = [c[0] for c in categories]

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            autosize=True ,
            xaxis=dict(
    type="log",
    title="Mass (eV)",
    range=[np.log10(x_min), np.log10(x_max)],
    tickmode="array",
    tickvals = [10**i for i in range(-12, 13)],
ticktext = [_superscript(i) for i in range(-12, 13)],
    showgrid=True,
    zeroline=False,
),

            yaxis=dict(
                tickmode="array",
                tickvals=ytickvals,
                ticktext=yticktext,
                range=[-0.5, len(categories)-0.5],
                showgrid=False,
                zeroline=False,
            ),
            margin=dict(l=80, r=30, t=40, b=60),
            hovermode="closest",
        )

    
        fig.add_annotation(
            x=1e-11,
            y=len(categories)-0.5,
            xref="x", yref="y",
          
            showarrow=False,
            align="left",
            font=dict(size=10),
            yshift=-40
        )

        ui.plotly(fig).classes('w-full').style(f"height:{height}px; width: 100%;")
        #ui.plotly(fig).style(f"height:{height}px; width:100%")
    

    def particle_page(container):
        with container:
            title_on_dark("Fundamental Particles & Interactions")
            with ui.dialog() as mass, ui.card().classes('w-full  h-auto '):
                plot_mass_strip(container, height=400, width=600)
                aria_button("Close", 'close',on_click=lambda: mass.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.dialog() as relation, ui.card().classes('w-full  h-auto '):
                plot_particle_graph(
                        "Cosmological Processes (Reactions)",    processes_nodes,  processes_edges,is_process_graph=True,            height=700 , width=400                     )
                aria_button("Close", 'close',on_click=lambda: relation.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.row().classes('w-full justify-center gap-2'):
                with ui.element('div').classes("text-xl font-bold flex items-center h-10 px-4"):
                    ui.markdown("**MICRO-COSMOS**: Particles & Processes").props('role=heading aria-level=2 tabindex=0')
                create_mass_conversion_dialog()
                aria_button("Mass Particle plot","Mass Particle plot",on_click=lambda: mass.open(),).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Cosmological Processes","Cosmological Processes",on_click=lambda: relation.open(),).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                

            with ui.row().classes("w-full flex-nowrap gap-1 items-stretch justify-center "):
                with ui.column().classes("flex-1 min-w-0 p-0 border rounded overflow-hidden"):
                    plot_particle_graph("Leptons (Fermions)", leptons_nodes, leptons_edges, height=520,width=600)
                with ui.column().classes("flex-1 min-w-0 p-0 border rounded overflow-hidden"):
                    plot_particle_graph("Quarks (Fermions)", quarks_nodes, quarks_edges, height=520,width=600)

     
                with ui.column().classes("flex-1 min-w-0 p-0 border rounded overflow-hidden"):
                    plot_particle_graph("Bosons (Force Carriers & Higgs)", bosons_nodes, bosons_edges, height=520,width=600)
                with ui.column().classes("flex-1 min-w-0 p-0 border rounded overflow-hidden"):
                    plot_particle_graph("Hadrons (Baryons & Mesons)", hadrons_nodes, hadrons_edges, height=520,width=600)

    
                
                



            
            
            






       
        


            
    DISCOVERY_EVENTS = [

        # --- Prehistory & Ancient Astronomy ---
        {'year': -38000, 'title': 'Earliest Lunar Tracking',
        'image': 'paleolithic_lunar_bone.jpg',
        'desc': 'Upper Paleolithic humans record lunar cycles on engraved bones,possibly the earliest astronomical observations. [Wikimedia](https://commons.wikimedia.org/wiki/File:Lunar_calendar_Blanchard_MAN_ArtsEtPrehistoire.jpg)'},

        {'year': -5000, 'title': 'Goseck Solar Observatory',
        'image': 'goseck_circle.jpg',
        'desc': 'The Goseck Circle in Germany is built, aligned with winter solstice sunrise and sunset. [Wikipedia](https://en.wikipedia.org/wiki/Goseck_Circle)'},

        {'year': -2500, 'title': 'Stonehenge Alignments',
        'image': 'stonehenge.jpg',
        'desc': 'Stonehenge constructed with precise solstice alignments for tracking seasons. [Wikipedia](https://en.wikipedia.org/wiki/Stonehenge)'},

        # --- Classical Astronomy ---
        {'year': -280, 'title': 'Aristarchus Proposes Heliocentrism',
        'image': 'aristarchus.jpg',
        'desc': 'Aristarchus of Samos proposes a Sun-centered model of the Solar System. [Astronomia](https://www.astronomia.com/progetti/locchio-infinito-di-galileo/prima-di-galileo/aristarco-di-samo-e-la-teoria-eliocentrica/),[AstronomyForChange](https://astronomyforchange.org/greek-scholars-development-constellations/aristarchus/)'},

        {'year': 150, 'title': 'Ptolemy’s Almagest',
        'image': 'ptolemy_almagest.jpg',
        'desc': 'Ptolemy publishes the Almagest, defining 48 constellations and a detailed geocentric model. [OldMaa](https://old.maa.org/press/periodicals/convergence/mathematical-treasure-erasmuss-greek-edition-of-ptolemys-almagest),[Wikipedia](https://en.wikipedia.org/wiki/Ptolemy)'},

        # --- Renaissance ---
        {'year': 1543, 'title': 'Copernicus – Heliocentric Model',
        'image': 'copernicus.jpg',
        'desc': 'Copernicus publishes *De Revolutionibus*, reviving the heliocentric theory. [Wikipedia](https://it.wikipedia.org/wiki/De_revolutionibus_orbium_coelestium),[SciencePhoto](https://www.sciencephoto.com/media/407566/view/nicolaus-copernicus-1473-1543-)'},

        {'year': 1609, 'title': 'Galileo’s Telescope',
        'image': 'galileo_telescope.jpg',
        'desc': 'Galileo builds a telescope and observes Moon craters, Jupiter’s moons, and star fields. [Britannica](https://www.britannica.com/science/Galilean-telescope), [Wikipedia](https://en.wikipedia.org/wiki/Galileo_Galilei)'},

        {'year': 1609, 'title': 'Kepler’s Laws (I & II)',
        'image': 'kepler_laws.png',
        'desc': 'Johannes Kepler formulates his first two laws of planetary motion. [ScienceNotes](https://sciencenotes.org/keplers-laws-of-planetary-motion/)'},

        {'year': 1619, 'title': 'Kepler’s Third Law',
        'image': 'kepler_third_law.jpg',
        'desc': 'Kepler publishes *Harmonices Mundi*, including the third law of planetary motion. [ResearchGate](https://www.researchgate.net/publication/346510651_The_Timeline_Of_Gravity/figures?lo=1&utm_source=google&utm_medium=organic), [Wikipedia](https://it.wikipedia.org/wiki/Giovanni_Keplero)'},

        # --- Scientific Revolution ---
        {'year': 1687, 'title': 'Newton’s Principia',
        'image': 'newton_principia.jpg',
        'desc': 'Isaac Newton publishes the *Principia Mathematica*, founding modern gravitational physics. [UniversityCollection](https://university-collections.wp.st-andrews.ac.uk/2017/07/05/celebrating-330-years-of-isaac-newtons-principia/)'},

        # --- 18th–19th Century ---
        {'year': 1781, 'title': 'Discovery of Uranus',
        'image': 'uranus_discovery.jpg',
        'desc': 'William Herschel discovers Uranus, expanding the Solar System for the first time in history. [BBC](https://www.skyatnightmagazine.com/space-science/how-was-uranus-discovered),[Wikipedia](https://en.wikipedia.org/wiki/William_Herschel)'},

        {'year': 1838, 'title': 'First Stellar Parallax',
        'image': 'stellar_parallax.jpg',
        'desc': 'Friedrich Bessel measures the distance to 61 Cygni using parallax — the first direct stellar distance. [Mlahanas](http://www.mlahanas.de/Greeks/Astronomy2.htm),[Wikipedia](https://en.wikipedia.org/wiki/Friedrich_Wilhelm_Bessel)'},

        {'year': 1868, 'title': 'Helium Discovered in the Sun',
        'image': 'helium_sun.jpg',
        'desc': 'Helium is identified in the solar spectrum before its discovery on Earth by Pierre Janssen. [ArmaghPlanet](https://armaghplanet.com/spectroscopy-and-the-discovery-of-helium.html),[Wikimedia](https://commons.wikimedia.org/wiki/File:Jules_Janssen_2.jpg)'},

        # --- Early 20th Century ---
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

        # --- Space Age ---
        {'year': 1957, 'title': 'Sputnik 1',
        'image': 'sputnik.jpg',
        'desc': 'After Serjei Korolev studies about human spaceflights, the first artificial satellite was launched, marking the beginning of the space age. [Wikipedia](https://it.wikipedia.org/wiki/Sputnik_1),[spsAviation](https://www.sps-aviation.com/story/?id=3240&h=Sergei-Korolev-1907-1966)'},

        {'year': 1965, 'title': 'Discovery of the CMB',
        'image': 'cmb_penzias_wilson.jpg',
        'desc': 'Penzias and Wilson detect the cosmic microwave background, confirming the Big Bang. [ResearchGate](researchgate.net/publication/271140622_Physics_of_the_cosmic_microwave_background_anisotropy/figures?lo=1&utm_source=google&utm_medium=organic),[ScienzaPerTutti]([Penzias and Wilson](https://scienzapertutti.infn.it/rubriche/biografie/2709-wilson-robert-woodrow))'},

        {'year': 1969, 'title': 'Apollo 11 Moon Landing',
        'image': 'apollo_11.jpg',
        'desc': 'Neil Armstrong and Buzz Aldrin become the first humans to walk on the Moon. [BBC](https://www.bbc.co.uk/newsround/48789792),[Wikipedia](https://it.wikipedia.org/wiki/Apollo_11)'},

        {'year': 1974, 'title': 'Discovery of the First Pulsar Binary',
        'image': 'binary_pulsar.jpg',
        'desc': 'Hulse and Taylor discover the first binary pulsar — indirect evidence for gravitational waves. [NASA](https://asd.gsfc.nasa.gov/blueshift/index.php/2016/03/17/we-knew-that-already/),[CredoLibrary](https://credo.library.umass.edu/view/full/murg171-sl94-r730-i012)'},

        {'year': 1977, 'title': 'Voyager Launch',
        'image': 'voyager.jpg',
        'desc': 'Voyager 1 and 2 begin their Grand Tour of the outer planets and interstellar space. [NASA](https://www.nasa.gov/image-article/voyager-1s-mission-outer-planet-begins/),[ApNews](https://apnews.com/article/nasa-voyager-spacecraft-contact-19e16b945869623cd94778795e62001b)'},

        # --- Modern Cosmology ---
        {'year': 1990, 'title': 'Hubble Space Telescope Launch',
        'image': 'hst.jpg',
        'desc': 'The Hubble Space Telescope begins operating, revolutionizing astronomy. [ESA-Hubble](https://esahubble.org/images/hst_launch_hi/)'},

        {'year': 1992, 'title': 'First Exoplanets Detected',
        'image': 'first_exoplanets.jpg',
        'desc': 'The first confirmed exoplanets are found around a pulsar by Wolszczan & Frail. [ScientificAmerican](https://www.scientificamerican.com/blog/observations/who-really-discovered-the-first-exoplanet/),[BigThink](https://bigthink.com/starts-with-a-bang/the-planets-that-never-were/)'},

        {'year': 1998, 'title': 'Accelerating Universe',
        'image': 'supernova_acceleration.jpg',
        'desc': 'Type Ia supernova studies reveal accelerating expansion → dark energy. [ESA](https://www.esa.int/ESA_Multimedia/Images/2019/01/Investigating_the_expansion_of_the_Universe_with_type-Ia_supernovas_and_quasars)'},

        # --- 21st Century ---
        {'year': 2015, 'title': 'Gravitational Waves Detected',
        'image': 'ligo_gw150914.jpg',
        'desc': 'LIGO makes the first direct detection of gravitational waves from merging black holes and Rainer Weiss, Kip Thorne and Barry Barish obtained the nobel prize. [LIGO-Caltech](https://www.ligo.caltech.edu/image/ligo20160211a),[BBC](https://www.bbc.com/news/science-environment-41476648)'},

        {'year': 2019, 'title': 'First Black Hole Image (EHT)',
        'image': 'm87_eht.jpg',
        'desc': 'Event Horizon Telescope produces the first image of a black hole (M87*). [EHT](https://eventhorizontelescope.org/blog/astronomers-reveal-first-image-black-hole-heart-our-galaxy),[LotusLeafLive](https://lotusleaflive.com/archives/4839)'},
        
        {'year': 2021, 'title': 'JWST Launch',
        'image': 'jwst.jpg',
        'desc': 'The James Webb Space Telescope launches to explore the early Universe. [NASA](https://science.nasa.gov/mission/webb/launch/),[NationalWorld](https://www.nationalworld.com/news/world/who-is-james-webb-nasa-telescope-named-after-space-agency-boss-what-did-he-do-and-who-named-the-telescope-3508502)'},
        {'year': 2024, 'title': 'Gaia Black Hole Discovery (BH3)',
        'image': 'gaia_bh3.jpg',
        'desc': 'Gaia discovers a dormant stellar-mass black hole (BH3) only ~2,000 light-years from Earth, with a mass ≈ 33 M☉, revealing a hidden population of quiescent black holes. [ESA](https://www.esa.int/Science_Exploration/Space_Science/Gaia/Sleeping_giant_surprises_Gaia_scientists)'},

        {'year': 2024, 'title': 'Fastest-Growing Distant Black Hole',
        'image': 'fast_growing_quasar.jpg',
        'desc': 'Astronomers identify a distant quasar powered by a black hole swallowing the mass of a Sun per day — one of the most luminous and rapidly feeding black holes ever observed [Space](https://www.space.com/40596-fastest-growing-black-hole-found.html). '},

        {'year': 2024, 'title': 'First Exoplanet Imaged by JWST Coronagraph',
        'image': 'jwst_coronagraph_exoplanet.jpg',
        'desc': 'Using JWST’s MIRI coronagraph (mid-infrared), a previously unknown exoplanet is directly imaged in a debris disk [NASA](https://science.nasa.gov/blogs/webb/2022/09/01/nasas-webb-takes-its-first-ever-direct-image-of-distant-world/). '},
        {'year': 2025, 'title': 'Most Massive Black Hole Merger Detected (GW250114)',
        'image': 'gw250114.png',
        'desc': 'LIGO/Virgo/KAGRA detect GW250114, a gravitational-wave event from a merger producing a black hole ~225 × solar mass. [Wikipedia](https://en.wikipedia.org/wiki/GW250114) '},

        {'year': 2024, 'title': 'Unusual Spin in Sagittarius A*',
        'image': 'sgrA_spin.jpg',
        'desc': 'Event Horizon Telescope observations show the Milky Way’s central black hole (Sgr A*) is spinning in a way that suggests a violent past, possibly from a major merger. [LiveScience](https://www.livescience.com/space/black-holes/the-milky-way-s-supermassive-black-hole-is-spinning-incredibly-fast-and-at-the-wrong-angle-scientists-may-finally-know-why) '},

        {'year': 2025, 'title': 'Jet Launch Near a Black Hole Edge',
        'image': 'blackhole_jet.jpg',
        'desc': 'Scientists observe a plasma jet moving at ~1/3 the speed of light and rapid X-ray fluctuations very close to the event horizon of an active black hole, revealing new physics at the brink. [Guardian](https://www.theguardian.com/science/2023/apr/26/astronomers-capture-first-image-of-jet-being-launched-from-edge-of-black-hole) '},

        {'year': 2025, 'title': 'Gaia-4 b: First Confirmed Exoplanet by Gaia',
        'image': 'gaia4b.jpg',
        'desc': 'Gaia-4 b, a massive exoplanet (~11.8 Mₙ), is confirmed via astrometric wobble + radial velocities. [Phys]’(https://phys.org/news/2025-02-high-precision-spectrograph-massive-exoplanet.html) '},

        {'year': 2025, 'title': 'The 128 New Moons Around Saturn',
        'image': 'saturn_moons.jpg',
        'desc': 'Astronomers announce the discovery of 128 previously unknown moons of Saturn, bringing its total to 274 and giving information about its formation and ring evolution. [ZmeScience](https://www.zmescience.com/space/saturn-moon-king/) '}


    ]

    def show_event_dialog(event):
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label(f"**{event['year']} – {event['title']}**").classes('text-sm text-center w-40 text-black').props('tabindex=0 role=heading aria-level=3 aria-label=Event title and year')
            aria_image(f"/discovery_images/{event['image']}", f"Image of {event['title']}").classes('w-full rounded-lg shadow')

            ui.markdown(event['desc']).props('tabindex=0 role=document aria-label=Event description')
            aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
        dialog.open()
    
    def astronomy_timeline(container):
        with container:
            with ui.row().classes('w-1/4 gap-1 '):
                title_on_dark('Timeline of Astronomical Discoveries')
                info_box('**Explore the main events and discoveries of Astronomy history** ')

            with ui.row().classes('overflow-x-auto no-wrap p-4 gap-8 !bg-gray-100 rounded-lg'):
                for event in DISCOVERY_EVENTS:
                    with ui.column().classes('items-center'):
                        aria_button(str(event['year']),'event year button',
                                        on_click=lambda e=event: show_event_dialog(e)
                                        ).classes('!bg-blue-600 text-white')

                        ui.label(event['title']).classes('text-sm text-center w-40 text-black').props('tabindex=0 aria-label=Event title')



    EPOCHS_EXTENDED = [

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
    
    {'name': 'Peak AGN Activity', 'z_ref': 2.5, 'image': 'agn_peak.png',
     'desc': 'Supermassive black holes undergo their most intense growth. [Acronomy](https://acronomy.lw1.at/acronym/agn)'},

    {'name': 'Galaxy Clusters Form', 'z_ref': 1, 'image': 'cluster_formation.png',
     'desc': 'Large-scale structures like the Local Group assemble. [Harvard](https://www.cfa.harvard.edu/research/topic/galaxy-clusters)'},

    {'name': 'Modern Universe', 'z_ref': 0.5, 'image': 'modern_universe.jpg',
     'desc': 'Galaxy groups, clusters, and filaments take their present form. [AmericanScientidt](https://www.americanscientist.org/article/modern-cosmology-science-or-folktale)'},

    {'name': 'Today', 'z_ref': 0, 'image': 'today.png',
     'desc': 'The present Universe: age ≈ 13.8 billion years. [ESA](https://esahubble.org/images/heic2003b/)'},
]

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
        


        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label(f"**{epoch['name']} (z={epoch['z_ref']})**").classes('text-sm text-center text-white').props('tabindex=0 aria-label=Epoch name and redshift')
            time_sec = z_to_time_seconds(z)
            if not math.isnan(time_sec):
                ui.label(f"Universe age: {age:.3f} Gyr ({time_sec:.2e} s)").classes('text-sm text-center text-white').props('tabindex=0 aria-label=Universe age in gigayears')
            else:
                ui.label(f"Universe age: {age:.3f} Gyr (time in seconds not defined for theoretical epochs)").props('tabindex=0 aria-label=Universe age in gigayears')

            aria_image(f"/cosmic_epochs/{epoch['image']}", f"Image of {epoch['name']} epoch").classes('w-full rounded-lg shadow')

            ui.markdown(epoch['desc']).props('tabindex=0 aria-label=Epoch description')
            aria_button('Close','close button', on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
        dialog.open()
    

    NUCLEOSYNTHESIS_INDEX = next(i for i, e in enumerate(EPOCHS_EXTENDED)   if e['name'] == 'Big Bang Nucleosynthesis')
    def cosmic_timeline(container):
        with container:
            
            with ui.row().classes('w-1/3 gap-1'):
                title_on_dark('Cosmic Timeline — From the Big Bang to Today')
                info_box('**Explore the universe evolution and analyze each phase**')
            with ui.row().classes(' w-full  '):
               
                    
                with ui.card().classes('p-4 !bg-white shadow-md border border-gray-300 max-w-md'):
                    ui.label("Legend").classes('text-lg font-bold text-black mb-2').props('tabindex=0 aria-label=Legend')

                    with ui.row().classes('gap-1 flex-nowrap justify-start'):
                        ui.label("🟩 Theoretical epochs (not directly observable)").classes('text-black').props('tabindex=0 aria-label=Theoretical epochs')
                        ui.label("🟦 Observational epochs (supported by evidence)").classes('text-black').props('tabindex=0 aria-label=Observational epochs')



        
            with ui.row().classes('overflow-x-auto no-wrap p-4 gap-10 !bg-gray-100 rounded-lg'):
                for epoch in EPOCHS_EXTENDED:
                    aria_button(epoch['name'], "Universe epochs timeline",
                                    on_click=lambda e=epoch: show_epoch_dialog(e)).classes('text-white ' + ('!bg-green-600' if EPOCHS_EXTENDED.index(epoch) < NUCLEOSYNTHESIS_INDEX else '!bg-blue-600'))

                    ui.label(f"z ≈ {epoch['z_ref']}").classes('text-sm text-center text-black').props('tabindex=0 aria-label=Redshift value')


    
    
        
    
  

    stellar_nodes = [
        # Root
        {
            "id": "stellar_nebula",
            "label": "Stellar Nebula",
            "desc": "A vast, cold cloud of gas (primarily **hydrogen** and **helium**) and cosmic dust. "
            "\n\nNew stars form when regions of this cloud collapse under their own **gravity**,"
            "\n\ninitiating the process of star formation. This collapse creates a dense protostar.",
            "color": "#bb9bff",
            "shape": "diamond"
        },

        # --- Average Star Branch (Stars up to ~8 Solar Masses) ---
        
        {
            "id": "average_star",
            "label": "Avg-Mass Star(MS)",
            "desc": "A medium-mass star (like the Sun). It achieves hydrostatic equilibrium and "
            "\n\nremains stable by fusing **hydrogen into helium** in its core via the proton-proton chain. "
            "\n\nThis is the longest phase of a star's life.",
            "color": "#ffe58a",
            "shape": "circle"
        },
        {
            "id": "red_giant",
            "label": "Red Giant",
            "desc": "After exhausting core hydrogen, the core contracts while **hydrogen shell burning** begins,"
            "\n\ncausing the outer layers to expand and cool. The star increases in size and luminosity, "
            "\n\nmoving toward the upper-right of the H-R diagram.",
            "color": "#ff9b6b",
            "shape": "circle"
        },
        {
            "id": "planetary_nebula",
            "label": "Planetary Nebula",
            "desc": "As the star's outer layers are expelled due to thermal pulses,"
            "\n\nthey form a **glowing shell of ionized gas** expanding away from the core. "
            "\n\nThis phase is relatively brief and enriches the interstellar medium with heavier elements.",
            "color": "#a4d1ff",
            "shape": "circle"
        },
        {
            "id": "white_dwarf",
            "label": "White Dwarf",
            "desc": "The exposed, hot, inert core of the former star, composed mainly of carbon and oxygen."
            "\n\nIt is supported against collapse by **electron degeneracy pressure** and"
            "\n\nslowly cools over billions of years, eventually becoming a black dwarf.",
            "color": "#e6f3ff",
            "shape": "circle"
        },

        # --- Massive Star Branch (Stars > ~8 Solar Masses) ---

        {
            "id": "massive_star",
            "label": "Massive Star(MS)",
            "desc": "A star with significantly greater mass than the Sun."
            "\n\nIt burns fuel much faster and hotter due to higher core pressure, primarily fusing hydrogen via the **CNO cycle**. "
            "\n\nThese stars have short, brilliant lifetimes.",
            "color": "#a8cfff",
            "shape": "circle"
        },
        {
            "id": "red_supergiant",
            "label": "Red Supergiant",
            "desc": "An enormous, highly luminous star that has exhausted its core hydrogen. "
            "\n\nFusion proceeds in a series of shells (e.g., carbon, neon, oxygen, silicon) until the core is predominantly **iron**. "
            "\n\nThis is the stage immediately preceding core collapse.",
            "color": "#ff7b52",
            "shape": "circle"
        },
        {
            "id": "supernova",
            "label": "Supernova(II)",
            "desc": "The cataclysmic, violent explosion that occurs when the iron core collapses, leading to a massive rebound shockwave. "
            "\n\nThis process is responsible for creating and dispersing elements heavier than **iron** (e.g., gold and uranium) into space.",
            "color": "#fff1a6",
            "shape": "circle"
        },
        {
            "id": "neutron_star",
            "label": "Neutron Star",
            "desc": "The ultra-compact remnant of a supernova (if the remnant mass is below ~3 solar masses). "
            "\n\nIt is supported by **neutron degeneracy pressure**, with extreme density. Many are observed as pulsars.",
            "color": "#c7f0ff",
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
                    showarrow=True, arrowhead=3,
                    arrowsize=1, arrowwidth=1.2,
                    arrowcolor="#444", text=interaction, opacity=0.9
                )
            )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#888"),
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
            desc_html = info.get('desc', '').replace('\n\n', '<br><br>').replace('\n', '<br>')
            node_text.append(f"<b>{info.get('label','')}</b><br>{desc_html}")

            colors.append(info.get('color', '#2b7de9'))
            symbols.append(info.get('shape', 'circle'))
            sizes.append(44 if info.get('shape') == 'diamond' else 30)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(color=colors, size=sizes, symbol=symbols, line_width=2),
            showlegend=False
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=title,
            title_x=0.5,
            width=width, height=height,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            hoverlabel=dict(
 align="left", 
 bgcolor="white", 
 bordercolor="black",

 font=dict(size=12) 
 )
        )

        ui.plotly(fig).style(f"height:{height}px; width:600px")




    def show_star_dialog(df, i):
        row = df.iloc[i]
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label(f"Star: {row['source_id']}").classes('text-sm text-center').props('tabindex=0 aria-label=Star ID')
            ui.label(f"RA: {row['ra']:.5f}, DEC: {row['dec']:.5f}").props('tabindex=0 aria-label=Star coordinates')
            #aria_image(row['image_url'], f"Image of star {row['source_id']}").classes('w-full rounded shadow')
          
            ui.markdown(
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

   
    ISO_DF, ISO_TREE = load_isochrones_cached(MIST_PATH)
    
    
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

    GAIA_DF = load_gaia_cached(STAR_GAIA_PATH)

    def hr_diagram_page(container, gaia_csv_path: str,sample_size=20000):
     
 
        with container:
            title_on_dark("H–R Diagram for Stars ")
            with ui.dialog() as hr_info, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                info_box(
                " The **Hertzsprung–Russell (H–R)** diagram shows stars according to their color or temperature and luminosity (brightness) or absolute magnitude. "
                "The diagram reveals distinct groups of stars from GAIA dataset providing information about stellar evolution and properties."
                "\n\n- **Main Sequence (MS)**: the longest and most stable phase in a star's lifetime, where it resides in hydrostatic equilibrium. Stars generate energy by fusing hydrogen into helium in their cores. The position of an MS star along the diagonal band is determined by its mass, with high-mass stars residing at the luminous, hot upper-left and low-mass stars at the cooler, dim lower-right."
                "\n\n-**Subgiant Branch (SGB)**: marks the transition off the Main Sequence after the core hydrogen is exhausted. The stars core begins to contract, heating the surrounding shell, where hydrogen starts burning,while the outer layers start to expand and cool causing it to move horizontally rightward and slightly upward on the H-R diagram."
                "\n\n-**Red Giants Branch (RGB)**: the outer envelope expands, driven by hydrogen shell burning around the helium core and leading to an increase in luminosity and a sharp drop in surface temperature, placing these stars in the upper-right region of the H-R diagram."
                    "\n\n-**Helium Burning (HeB)**: when the contracting helium core reaches a critical temperature, helium fusion (the triple-alpha process) ignites, converting helium into carbon and oxygen. This new energy source brings the star to a temporary, stable phase (Red Clump for solar-mass stars) and they settle at a lower luminosity and slightly hotter temperature than the tip of the RGB."
                    "\n\n-**Asymptotic Giant Branch (AGB)**: late evolutionary stage characterized by fusion occurring in two shells: an outer hydrogen shell and an inner helium shell, both surrounding an inert carbon/oxygen core. Stars are highly luminous and the coolest giants, occupying a region above the RGB on the H-R diagram. They undergo episodic mass loss, leading to the formation of a planetary nebula."
                    "\n\n-**White Dwarfs (WD)**: the remnants of low to intermediate-mass stars that have shed their outer layers, extremely dense, hot, and small. They have no active fusion and are supported against gravitational collapse solely by electron degeneracy pressure. They appear in the bottom-left corner of the H-R diagram and cool down over cosmic time, eventually becoming black dwarfs."
                    
                )
                aria_button('Close','close button', on_click=hr_info.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.dialog() as data_info, ui.card().classes('w-96'):
                info_box("Dataset Gaia DR3: RA (right ascension),DEC(declination),z (redshift),mag_ur(magnitude in u-band filter, r-band filter). Dataset MIST:logTeff (effective temperature),logL (luminosity),stellar_mass")
                reference_box(""" **Dataset reference**: [GAIA SDR3](https://gea.esac.esa.int/archive/), Sysoliatina Kseniia 2022 JJ-model isochrone set: PARSEC MIST and BaSTI stellar evolution, [MIST](https://doi.org/10.11588/DATA/ZCXHOE) """)
                aria_button('Close','close button', on_click=data_info.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.row().classes('w-full justify-center gap-4'):
                aria_button('Info H-R stars','Information about H-R diagram stars', on_click=hr_info.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                aria_button('Dataset','Datasets info and references', on_click=data_info.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
            #df = sample_csv_reservoir(gaia_csv_path, n=sample_size, dtype={'source_id': str})
            #df = pd.read_csv(gaia_csv_path, dtype={'source_id': str})
            #df = df.dropna(subset=['bp_rp', 'phot_g_mean_mag', 'parallax', 'ra', 'dec'])
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
           
            with ui.row().classes('w-full justify-center gap-4'):
                with ui.column().classes('flex-1'):
            
                    plot = ui.plotly(fig)


                    plot.on('plotly_click', lambda e: show_star_dialog(df, e.args['points'][0]['pointIndex']))
                with ui.column().classes('flex-1'):
                    plot_star_graph(
        title="Life Cycle of a Star",
        nodes=stellar_nodes,
        edges=stellar_edges,
    
        height=600
      
       
        
    )

  



        






    def show_galaxy_dialog(df, i):
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label(f"{df.loc[i,'galaxy_name']}").classes('text-sm text-center').props('tabindex=0 aria-label=Galaxy name')
            aria_image(df.loc[i,'image_url'], f"Image of galaxy {df.loc[i,'galaxy_name']}").classes('w-full rounded shadow')
            ui.markdown(
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
                     
    def galaxy_map( csv_path: str,sample_size=100):
                
                
                
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

    
        df['galaxy_name'] = 'Unknown' 
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



        
        ui.label("Selext a galaxy:").classes('text-lg mt-4').props('tabindex=0 aria-label=Select a galaxy instruction')

        options = [
            (f"{row['specobj_id']} ", i)
            for i, row in df.iterrows()
        ]

        selected = ui.select(
            options=options,
            label='Galaxy ID',
            with_input=True
        ).classes('w-80').props('aria-label=Galaxy selection dropdown')

        def on_select(e):
            data = e.args

        
            if not data:
                return  

    
            if isinstance(data, dict):
                i = data.get('value')
                if i is not None and i in df.index:
                    show_galaxy_dialog(df, i)


        selected.on('update:model-value', on_select)    
    def show_morphology_dialog(df, i):
        row = df.iloc[i]

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Galaxy OBJID: {row['OBJID']}") \
                .classes("text-sm text-center") \
                .props('tabindex=0 aria-label=Galaxy OBJID')

            ui.label(f"Morphology: {row['morphology']}") \
                .classes("text-xs text-center") \
                .props('tabindex=0 aria-label=Galaxy morphology class')
            #aria_image(row["image_url"], f"SDSS image of galaxy {row['OBJID']}").classes("w-full rounded shadow")

            ui.markdown(
                f"""
                **u−r color:** {row['color_ur']:.2f}  
                **Concentration (R90/R50):** {row['concentration']:.2f}  
        
                **Redshift:** {row['z']}  
                **Luminosity Distance (Mpc):** {row['DL']:.1f}
                **Morphology:** {row['morphology']}
                
                """
            ).props('tabindex=0 aria-label=Galaxy color and concentration values')

            aria_button("Close", "close dialog", on_click=dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

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

    SDSS_MORPHO_DF = load_morphology_cached(SDSS_MORPHO_PATH)
    
    

    
        
    def galaxy_morphology_page(gz_path: str, sample_size=50000):
       
                
            
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
        
          
        plot1 = ui.plotly(fig1).classes("w-full")
           
        #plot2 = ui.plotly(fig2).classes("w-full")  


        plot1.on(    'plotly_click',    lambda e: show_morphology_dialog(df, e.args["points"][0]["pointIndex"]))
        #plot2.on(    'plotly_click',    lambda e: show_morphology_dialog(df, e.args["points"][0]["pointIndex"]))


  
                 
    def galaxy_map_page( container):
        with container:
            title_on_dark('Galaxy Map & Morphology Classification')
                  
            with ui.dialog() as info_morpho, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                info_box(
                            "This plot presents a classification of galaxy morphologies based on their color and concentration index using data from the Sloan Digital Sky Survey (SDSS). "
                            "Galaxies are categorized into four main types: Elliptical, Lenticular, Spiral, and Irregular. "
                            "The classification is based on the u−r color index and the concentration index (R90/R50), which are indicative of the stellar populations and structural properties of galaxies."
                            "\n\n- **Elliptical**: galaxies are characterized by red colors and high concentration indices, indicating older stellar populations and a more compact structure."
                            "\n\n- **Lenticular**: galaxies exhibit intermediate colors and concentration indices, representing a transitional morphology between elliptical and spiral types."
                            "\n\n- **Spiral**: galaxies tend to have bluer colors and lower concentration indices, reflecting ongoing star formation and a disk-like structure."
                            "\n\n- **Irregular**: galaxies display very blue colors and low concentration indices, often due to recent or ongoing star formation triggered by interactions or mergers."
                        ).props('tabindex=0 role=document aria-label=Galaxy classification instructions')
                aria_button('Close','close button', on_click=lambda: info_morpho.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                        
                        
            with ui.dialog() as info, ui.card().classes('w-96'):
                info_box('Galaxies are located by their coordinates (RA/DEC) in the plot. Select a galaxy to see details: type, redshift, age, distance from Earth.').props('tabindex=0 role=document aria-label=Galaxy map instructions')
            
                aria_button('Close','close button', on_click=lambda: info.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            
            with ui.dialog() as data_info, ui.card().classes('w-96'):
                info_box('Dataset:RA (right ascension),DEC(declination),z(redshift),z_error,subclass').props('tabindex=0 role=document aria-label=Galaxy data')
                reference_box(""" **Dataset reference**: [SDSS DR16](https://www.sdss.org/dr16/), [SDSS](https://cdsarc.cds.unistra.fr/ftp/J/A+A/648/A122/), VizieR Online Data Catalog: SDSS galaxies morphological classification (Vavilova+, 2021)""")
            
                 
                aria_button('Close','close button', on_click=lambda: data_info.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
           
                            
                 
            with ui.dialog() as dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                ui.label("Galaxy Classification Criteria").classes("text-lg font-bold text-center").props('aria-label=Galaxy classification explanation title')
                ui.markdown(
                    """
        **Galaxy classification is based on the u−r color index and the concentration index (R50/R90), following observational studies from the Sloan Digital Sky Survey (SDSS).**

        - **Color (u−r)**: Measures the difference between the ultraviolet (u) and red (r) magnitudes. 
        Bluer galaxies (lower u−r) indicate younger stellar populations and ongoing star formation, typical of Spiral and Irregular galaxies. 
        Redder galaxies (higher u−r) indicate older stellar populations, characteristic of Elliptical and Lenticular galaxies.

        - **Concentration (R50/R90)**: Ratio of the radius containing 50-percent of the galaxy’s light (R50) to the radius containing 90-percent (R90). 
        High concentration values correspond to more centrally concentrated light profiles, typical of Elliptical galaxies. 
        Lower concentration values indicate more extended disks, typical of Spiral and Irregular galaxies.

        - **Classification criteria:**
        • **Elliptical:** red colors (u−r > 2.22) and high concentration (C > 2.6)
        • **Lenticular:** red colors and intermediate concentration (2.4 ≤ C ≤ 2.6)
        • **Spiral:** blue colors (u−r < 2.2) and moderate concentration (C < 2.6)
        • **Irregular:** very blue colors (u−r < 1.8) and low concentration (C < 2.2)

        
                    """
                ).props('aria-label=Galaxy classification explanation')
                aria_button("Close","close", on_click=lambda: dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                      


            with ui.dialog() as galaxy_class_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
    
                ui.html(r"""
                    <div style="font-family: Arial, sans-serif;">
                        <h4 class="text-xl font-bold">SDSS Galaxy Subclasses Explained</h4>
                        <p>Galaxies in the Sloan Digital Sky Survey (SDSS) are spectroscopically classified based on their observed spectral lines, which reveal the dominant energy sources within the galaxy.</p>
                                    <hr class="my-3">

                        <h5 class="text-lg font-semibold">Key Subclasses and Activity</h5>
                        
                        <table class="w-full text-sm text-left text-gray-700 mt-2" border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin:auto; text-align:center;">
                            <tr class="!bg-gray-200">
                                <th>Subclass</th>
                                <th>Dominant Activity</th>
                                <th>Spectral Characteristics</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td><b>STARFORMING</b></td>
                                <td>Normal Star Formation</td>
                                <td>Narrow emission lines (e.g., H$\alpha$, [O II]), typical of H II regions.</td>
                                <td>Galaxies where the primary source of light and energy is the birth of new stars.</td>
                            </tr>
                            <tr>
                                <td><b>STARBURST</b></td>
                                <td>Intense Star Formation</td>
                                <td>Very strong emission lines indicating a high rate of star formation, often compressed into a small region.</td>
                                <td>Undergoing a brief, rapid burst of star formation, using up gas quickly.</td>
                            </tr>
                            <tr>
                                <td><b>AGN</b></td>
                                <td>Active Galactic Nucleus (Seyfert)</td>
                                <td>Narrow emission lines from gas excited by a central supermassive black hole (SMBH).</td>
                                <td>The core luminosity is dominated by the SMBH accretion disk, but the spectral lines are narrow.</td>
                            </tr>
                            <tr>
                                <td><b>BROADLINE</b></td>
                                <td>Active Galactic Nucleus (Quasar/Broadline Seyfert)</td>
                                <td>Very wide (broad) emission lines, indicating high-velocity gas close to the central SMBH.</td>
                                <td>Often classified as Quasars or the brightest AGNs. The broad lines indicate rapid motion.</td>
                            </tr>
                            <tr>
                                <td><b>COMPOSITE</b></td>
                                <td>Mixed Activity</td>
                                <td>Shows spectral signatures from both Star Formation and an AGN (not explicitly listed in your legend, but often present in SDSS).</td>
                                <td>A transition stage where both star formation and the central black hole contribute significantly to the total energy output.</td>
                            </tr>
                        </table>
                        
                        <hr class="my-3">

                        <h5 class="text-lg font-semibold">The 'BROADLINE' Distinction</h5>
                        <p class="text-xs text-gray-500">
                        The **BROADLINE** label (e.g., AGN BROADLINE) signifies that the galaxy's spectrum contains very wide emission lines. This indicates that the gas is moving at high speeds, usually because it is orbiting very close to the supermassive black hole at the galaxy's center. It is generally a classification subset of the most energetic **AGNs** (often Quasars).
                        </p>
                    </div>
                    """).props('tabindex=0 role=document aria-label="SDSS Galaxy Classification information"')
                
                
                aria_button("Close", 'close', on_click=lambda: galaxy_class_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")


            
                       
            with ui.dialog() as img, ui.card().classes('w-96'):
                galaxy_image_grid(GALAXY_IMAGES)
                aria_button('Close','close button', on_click=lambda: img.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.dialog() as img2, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                aria_image('/images/sdss_gal_z.png', "Image of the distribution of galaxies from SDSS dataset with redshift").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                aria_button('Close','close button', on_click=lambda: img2.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.dialog() as img3, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                aria_image('/images/mag_gal.png', "Image of the galaxy morphology magnitude vs color").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                aria_button('Close','close button', on_click=lambda: img3.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
            with ui.row().classes('w-full items-center justify-between gap-2 py-2' ):
                galaxy_map(GAL_SDSS_PATH)
                aria_button("Info",label="Read instruction",on_click=lambda:     info.open(),).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Dataset",label="Read dataset info",on_click=lambda:    data_info.open(),).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button( "Galaxy Subclasses Explained", 'Galaxy Classification Info',             on_click=lambda: galaxy_class_dialog.open()            ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded self-center") 
    
                aria_button("Galaxy Classification",label="Read galaxy classification",on_click=lambda:     info_morpho.open(),).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                
                aria_button("Classification Criteria", "Show Classification Info",on_click=lambda: dialog.open(),).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Galaxy Class Images","Show Galaxy Images",on_click=lambda: img.open(),).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Galaxy Plot with z","Show Galaxy plot",on_click=lambda: img2.open(),).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Galaxy Plot r-mag vs color","Plot r-mag vs u-r",on_click=lambda: img3.open(),).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
            with ui.row().classes("w-full no-wrap items-start gap-4"):
                with ui.card().classes('w-1/2 p-0'):
                    aria_image('/images/sdss_gal.png', "Image of the distribution of galaxies from SDSS dataset").classes('w-3/4 mx-auto h-auto rounded-lg shadow-lg border border-gray-300')
                    
                with ui.card().classes('w-1/2 p-0 overflow-hidden'):
                    galaxy_morphology_page(SDSS_MORPHO_PATH)
                    #aria_image('/images/col_gal.png', "Plot concentration vs color u-r").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                
            
                    

        
        

    
    with ui.tab_panels(tabs, value=one):
        with ui.tab_panel(one) as intro_panel:
            introduction_page(intro_panel)


        with ui.tab_panel(two) as discovery_panel:
            astronomy_timeline(discovery_panel)
        with ui.tab_panel(three) as history_panel:
            cosmic_timeline(history_panel)
        with ui.tab_panel(four) as galaxy_panel:
            galaxy_map_page(galaxy_panel)
        with ui.tab_panel(five) as star_panel:
            hr_diagram_page(star_panel,STAR_GAIA_PATH)
        with ui.tab_panel(six) as particle_panel:
            particle_page(particle_panel)

            
            
            
            




   
        

@ui.page('/module2')
def module2():
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

  
  .info_box {
    font-size: 18px !important;
    line-height: 1.7;
    text-align: justify;
    margin-bottom: 10px;
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
  }

  
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
   
    ui.add_body_html(r"""
<script>
if (!window.MathJaxLoaded) {
    window.MathJaxLoaded = true;
    window.MathJax = {
        tex: {inlineMath: [['$', '$'], ['\\(', '\\)']]},
        svg: {fontCache: 'global'}
    };
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
    s.async = true;
    document.head.appendChild(s);
}
</script>
""")
    
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
    
    ui.add_head_html("""
<script>
window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]}
};
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
""")
    ui.add_body_html(r"""
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
""")


    main_layout("Module 2: Redshift & Universe Expansion")
    ui.label("Cosmological Redshift & Hubble Law").classes("text-lg font-bold mt-2").props('id=tabs_desc role=heading aria-level=2 tabindex=0')
    with ui.tabs().classes('w-full justify-center') as tabs:
        one = ui.tab(' Cosmological Redshift', icon='auto_awesome').props('aria-label="Activity 1: Cosmological Redshift"')
        two = ui.tab('Hubble law', icon='science').props('aria-label="Activity 2: Hubble law"')
        
    spectra_csv_dir = GALAXY_SPECTRA_PATH
    fits_parent_dir =GALAXY_FITS_PATH
  

   
    state = {"zs_points": []} 
    def add_redshift_activity(container):

        ui.add_body_html(r"""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  
""")
        
        with container:
            title_on_dark("Cosmological Redshift (SDSS spectra)")
            with ui.dialog() as datalam,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                reference_box("""**Dataset reference**: [SDSS-Galaxy spectra](https://www.sdss4.org/dr16/ , https://dr17.sdss.org/ , https://classic.sdss.org/dr6/algorithms/linestable.php), [NIST Atomic Spectra Database](https://www.nist.gov/pml/atomic-spectra-database)""")
                info_box("**Dataset**: wavelength (Å), flux (W·m⁻²·Å⁻¹).")
                aria_button("close","close",on_click=lambda: datalam.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

   
            
            with ui.dialog() as info_red,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                info_box("Explore how galaxy spectra reveal the cosmic redshift.\n\n"
                          "1. Plot flux vs wavelength for a galaxy spectrum.\n\n"
                          "2. Identify emission peaks lambda_obs for each galaxy and compare observed peaks with standard lines (Hα, Hβ, O III, etc.).\n\n"    
                        "3. Verify that galaxy lines are shifted relative to standard lines and compute redshift comparing with galaxy catalog .\n\n"
                           "4. Compute and plot non-relativistic and relativistic velocity vs redshift.\n\n"
                           "5. Observe that for small z the velocities are similar, while for large z relativistic effects become important.")
                aria_button("close","close",on_click=lambda: info_red.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
               
            with ui.dialog() as redshift_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                ui.html(r"""
<h6><b>Redshift and Flux Relations</b></h6>

<p>When a galaxy moves away due to the expansion of the Universe, its spectral lines shift toward longer wavelengths (<b>redshift</b>):</p>
<p style="text-align:center;"><span class="math">\( z = \frac{\lambda_{\rm obs} - \lambda_0}{\lambda_0} \)</span></p>

<p>For small redshifts:</p>
<p style="text-align:center;"><span class="math">\( v = c \, z \)</span></p>

<p>Relativistically:</p>
<p style="text-align:center;"><span class="math">\( v = c \, \frac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span></p>

<hr>

<h6><b>Flux–Luminosity Relation</b></h6>
<p>The flux <span class="math">\( F(\lambda) \)</span> is the energy received per unit area, time, and wavelength (W·m⁻²·Å⁻¹).  
For isotropic emission from a source with intrinsic luminosity <span class="math">\( L \)</span> at luminosity distance <span class="math">\( d_L \)</span>:</p>
<p style="text-align:center;"><span class="math">\( F = \frac{L}{4 \pi d_L^2} \)</span></p>
<p>Including the effect of redshift:</p>
<p style="text-align:center;"><span class="math">\( F(\lambda_{\rm obs}) = \frac{L(\lambda_{\rm emit})}{4 \pi d_L^2 (1+z)} \)</span></p>

<hr>

<h6><b>Relativistic Doppler Effect</b></h6>
<p>For light, the observed frequency changes if the source and/or observer are moving relative to each other.</p>

<p><b>Source moving away:</b></p>
<p style="text-align:center;"><span class="math">\( f_{\rm obs} = f_{\rm emit} \sqrt{\frac{1 - v/c}{1 + v/c}} \)</span></p>

<p><b>Source approaching:</b></p>
<p style="text-align:center;"><span class="math">\( f_{\rm obs} = f_{\rm emit} \sqrt{\frac{1 + v/c}{1 - v/c}} \)</span></p>

<p>Redshift corresponds to a wavelength increase (<span class="math">\( \lambda_{\rm obs} > \lambda_{\rm emit} \)</span>),  
blueshift corresponds to a wavelength decrease (<span class="math">\( \lambda_{\rm obs} < \lambda_{\rm emit} \)</span>).</p>

<hr>

<h6><b>Common Spectral Lines (SDSS)</b></h6>
<p>SDSS provides galaxy spectra (<span class="math">\( \lambda_{\rm obs}, F \)</span>) with key emission and absorption lines:</p>

<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin:auto; text-align:center;">
<tr><th>Element / Transition</th><th>Rest Wavelength (Å)</th></tr>
<tr><td>Hα (Hydrogen Balmer)</td><td>6562.8</td></tr>
<tr><td>Hβ (Hydrogen Balmer)</td><td>4861.3</td></tr>
<tr><td>[O III] (Forbidden Oxygen)</td><td>5006.8</td></tr>
<tr><td>[O II] Doublet</td><td>3727</td></tr>
<tr><td>Ca II H &amp; K</td><td>3968.5 / 3933.7</td></tr>
<tr><td>Na I D</td><td>5890 / 5896</td></tr>
</table>

<p style="text-align:center; margin-top:10px;">NIST Atomic Spectra Database</p> """).props('tabindex=0 role=document aria-label="Redshift and Flux Relations information"')

                
                aria_button("Close", label="Close the box", on_click=lambda: redshift_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

            with ui.row().classes("w-full items-center justify-center"):
                csv_files = sorted([f for f in os.listdir(spectra_csv_dir) if f.lower().endswith(".csv")])
                default_file = csv_files[0] if csv_files else None
                if not csv_files:
                    ui.label("No CSV spectra found in folder: " + spectra_csv_dir).classes("text-red-600")

            
                file_select = ui.select(csv_files, label="Choose spectrum CSV",value=default_file).classes("w-80").props(
                        'aria-describedby=tabs_desc aria-label=choose spectrum file'
                    )
                plot_btn = aria_button("Plot Spectrum", "plot galaxy spectrum").classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Instruction","Instruction",on_click=lambda: info_red.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Dataset","Data",on_click=lambda: datalam.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Info",label="Read detailed information about Redshift and Flux Relations",on_click=lambda: redshift_dialog.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded")
                
        
           

           
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
                    plate = int(str(plate_s).strip().strip("'\""))
                    mjd = int(str(mjd_s).strip().strip("'\""))
                    fiberid = int(str(fiberid_s).strip().strip("'\""))

                    gal_file = os.path.join(DATA_DIR, "gal_spectra_data.csv")
                    cols = [
                        "plate", "mjd", "fiberid", "run2d", "specobj_id", "ra", "dec",
                        "sn_median_r", "z", "zerr", "zwarning", "class", "subclass"
                    ]
                    df = pd.read_csv(gal_file, comment='#', sep=',', names=cols, engine='python', dtype=str)

                    for c in ["plate", "mjd", "fiberid"]:
                        if c in df.columns:
                            df[c] = df[c].astype(str).str.strip().str.strip("'\"")
                    
                            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

                    if "z" in df.columns:
                        df["z"] = pd.to_numeric(df["z"].astype(str).str.strip().str.strip("'\""), errors="coerce")
                    if "zerr" in df.columns:
                        df["zerr"] = pd.to_numeric(df["zerr"].astype(str).str.strip().str.strip("'\""), errors="coerce")

                    df = df.dropna(subset=["plate", "mjd", "fiberid"])

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
                    base_dir = GALAXY_LINE_PATH
                    base_name = os.path.splitext(os.path.basename(file_select_value))[0]
                    expected_name = f"{base_name}_lines.csv"
                    expected_path = os.path.join(base_dir, expected_name)

                    if os.path.exists(expected_path):
                        return pd.read_csv(expected_path), expected_path

                    prefix = base_name.split("_")[0]
                    candidates = [f for f in os.listdir(base_dir) if f.startswith(prefix) and f.endswith("_lines.csv")]
                    if candidates:
                        chosen = os.path.join(base_dir, candidates[0])
                        print(f"✅ Trovato file corrispondente: {candidates[0]}")
                        return pd.read_csv(chosen), chosen
                    else:
                        raise FileNotFoundError(f"Nessun file *_lines.csv trovato per {base_name}")
                def plot_spectrum():
                    spec_plot_container.clear()
                    if not file_select.value:
                        accessible_notify("Select a spectrum CSV first.", type_="warning")
                        return

                    df = pd.read_csv(os.path.join(spectra_csv_dir, file_select.value))
                    wl = np.array(df["wavelength_A"], dtype=float)
                    flux = np.array(df["flux"], dtype=float)
                    mask = np.isfinite(wl) & np.isfinite(flux)
                    wl, flux = wl[mask], flux[mask]
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
                            plt.title(f"Spectrum: {file_select.value}")
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

                

                

                with ui.dialog() as tab_spectra,ui.card():
                    

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

                    with ui.dialog() as result_dialog, ui.card().classes("w-2/3 max-w-3xl"):
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
                
                
                with ui.dialog() as velocity,ui.card():
                    with ui.row().classes('w-full items-center'):
                        info_box(
                            "Insert a value of redshift to compute the velocities:\n\n"
                            "**Non relativistic velocity**:v=cz\n\n"
                            "**Relativistic velocity**: v=c((1+z)²−1)/((1+z)²+1)"
                        )
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

                            gal_file = os.path.join(DATA_DIR, "gal_spectra_data.csv")
                            cols = [
                                "plate", "mjd", "fiberid", "run2d", "specobj_id", "ra", "dec",
                                "sn_median_r", "z", "zerr", "zwarning", "class", "subclass"
                            ]
                            df = pd.read_csv(gal_file, comment='#', sep=',', names=cols, engine='python')

                            df["z"] = pd.to_numeric(df["z"], errors="coerce")
                            df["zerr"] = pd.to_numeric(df["zerr"], errors="coerce")
                            df = df.dropna(subset=["z", "zerr"])

                            c_km_s = 299792.458
                            df["v_nr"] = c_km_s * df["z"]
                            df["v_nr_err"] = c_km_s * df["zerr"]
                            df["v_rel"] = c_km_s * (((1 + df["z"]) ** 2 - 1) / ((1 + df["z"]) ** 2 + 1))
                            df["v_rel_err"] = c_km_s * (4 * (1 + df["z"]) * df["zerr"]) / (((1 + df["z"]) ** 2 + 1) ** 2)

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
                                    plt.title("Galaxy velocities from SDSS dataset")
                                    plt.legend()
                                    plt.grid(alpha=0.4)
                                    plt.tight_layout()
                                    aria_chart_label("Galaxy velocities plot from redshift dataset with non-relativistic and relativistic velocities")

                    
                        plot_vel_btn.on("click", plot_velocity_dataset)
                    aria_button("Close","close",on_click=lambda:velocity.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as velocity,ui.card():
                    with ui.row().classes('w-full items-center'):
                        info_box(
                            "Insert a value of redshift to compute the velocities:\n\n"
                            "**Non relativistic velocity**:v=cz\n\n"
                            "**Relativistic velocity**: v=c((1+z)²−1)/((1+z)²+1)"
                        )
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

                            gal_file = os.path.join(DATA_DIR, "gal_spectra_data.csv")
                            cols = [
                                "plate", "mjd", "fiberid", "run2d", "specobj_id", "ra", "dec",
                                "sn_median_r", "z", "zerr", "zwarning", "class", "subclass"
                            ]
                            df = pd.read_csv(gal_file, comment='#', sep=',', names=cols, engine='python')

                            df["z"] = pd.to_numeric(df["z"], errors="coerce")
                            df["zerr"] = pd.to_numeric(df["zerr"], errors="coerce")
                            df = df.dropna(subset=["z", "zerr"])

                            c_km_s = 299792.458
                            df["v_nr"] = c_km_s * df["z"]
                            df["v_nr_err"] = c_km_s * df["zerr"]
                            df["v_rel"] = c_km_s * (((1 + df["z"]) ** 2 - 1) / ((1 + df["z"]) ** 2 + 1))
                            df["v_rel_err"] = c_km_s * (4 * (1 + df["z"]) * df["zerr"]) / (((1 + df["z"]) ** 2 + 1) ** 2)

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
                                    plt.title("Galaxy velocities from SDSS dataset")
                                    plt.legend()
                                    plt.grid(alpha=0.4)
                                    plt.tight_layout()
                                    aria_chart_label("Galaxy velocities plot from redshift dataset with non-relativistic and relativistic velocities")

                    
                        plot_vel_btn.on("click", plot_velocity_dataset)
                    aria_button("Close","close",on_click=lambda:velocity.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Velocity exercise","velocity exercise",on_click=lambda: velocity.open()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                
                spec_plot_container = ui.column().classes('w-full items-center justify-center')
                plot_btn.on("click", plot_spectrum)
                file_select.on("update:model-value", plot_spectrum)
                if default_file:
                    ui.timer(0.1, plot_spectrum, once=True)

                    

                




    sn_file = os.path.join(DATA_DIR, "supernovae.txt")

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
    def add_hubble_activity(container):
    

        with container:
            title_on_dark("Hubble Law with Type Ia Supernovae")
            
            
            with ui.dialog() as datas_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto'):
                reference_box("""**Dataset reference**: [Supernovae data Pantheon-SH0ES](https://github.com/dscolnic/Pantheon , https://pantheonplussh0es.github.io/)""")
      
                info_box("**Dataset**: zcmb (CMB frame redshift), mb (apparent B magnitude), err_mag.")
                aria_button("Close", label="Close", on_click=lambda: datas_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

            with ui.dialog() as cosmology_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto'):
                    ui.html(r"""
<h6><b>Fundamental Cosmological Relations</b></h6>

<p><b>Hubble Law:</b><br>
<span class="math">\( v = H_0 \cdot d_L \)</span><br>
For small redshifts (<span class="math">\( z \ll 1 \)</span>), <span class="math">\( v \approx c \cdot z \)</span>.</p>

<p><b>Distance Modulus:</b><br>
<span class="math">\( \mu = m_B - M_B = 5 \log_{10}(d_L [\rm pc]) - 5 \)</span><br>
→ connects apparent and absolute magnitude.</p>

<p><b>Luminosity Distance:</b><br>
<span class="math">\( d_L = (1+z)\frac{c}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m (1+z')^3 + \Omega_\Lambda}} \)</span><br>
→ accounts for cosmological expansion (ΛCDM model).</p>

<p><b>Scale Factor Relation:</b><br>
<span class="math">\( 1 + z = \frac{1}{a(t)} \)</span> → expansion of the Universe.</p>
""")
                    aria_button("Close", label="Close", on_click=lambda: cosmology_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

      
                

           
            with ui.dialog() as lcdm_dialog, ui.card().classes('p-4 w-full max-w-[900px] max-h-[600px] overflow-y-auto'):
                ui.html(r"""
<h6><b>ΛCDM Explanation</b></h6>

<ul style="line-height:1.6; font-size:0.95rem;">
    <li><b>Velocity (non-relativistic):</b> <span class="math">\( v = c \cdot z \)</span></li>
    <li><b>Relativistic velocity:</b> <span class="math">\( v = c \frac{(1+z)^2 - 1}{(1+z)^2 + 1} \)</span></li>
    <li><b>Distance modulus:</b> <span class="math">\( \mu = m_B - M_B \)</span></li>
    <li><b>Luminosity distance:</b> <span class="math">\( d_L = 10^{(\mu+5)/5} [\rm pc] \)</span></li>
    <li><b>Hubble constant estimate:</b> <span class="math">\( H_0 = \frac{\sum_i v_i d_{L,i}}{\sum_i d_{L,i}^2} \)</span></li>
    <li><b>Residuals:</b> <span class="math">r = v - H_0 \cdot d_L</span></li>
    <li><b>Scale factor relation:</b> <span class="math">\( 1 + z = \frac{1}{a(t)} \)</span></li>
    <li><b>ΛCDM distance:</b> <span class="math">\( d_L(z) = \frac{c (1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}, \quad E(z) = \sqrt{\Omega_m (1+z)^3 + \Omega_\Lambda} \)</span></li>
    <li><b>Typical parameters:</b> Ωₘ = 0.3, Ω_Λ = 0.7, Ωₖ = 0 (flat Universe)</li>
</ul>
""")
                aria_button("Close", label="Close", on_click=lambda: lcdm_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
            with ui.dialog() as info_dlg, ui.card().props(   'aria-modal="true" role="dialog" aria-labelledby="info-title" aria-describedby="info-content"'):
                    ui.label("Information").classes("text-lg font-semibold text-blue-400").props("id=info-title")
                                        
                    ui.html(r"""
                <div style="padding:10px; font-size:14px;">
        
                <div>
                    <strong>Redshift and scale factor</strong><br/>
                    \(1+z = \dfrac{a(t_0)}{a(t_{\rm emit})}\) with \(a(t_0)=1 \Rightarrow 1+z = 1/a(t)\).
                </div>
                <div style="margin-top:6px;">
                    <strong>Flux and luminosity</strong><br/>
                    For isotropic emission: \(F = \dfrac{L}{4\pi d_L^2}\).<br/>
                    In wavelengths: \(F(\lambda_{\rm obs}) = \dfrac{1}{1+z}\dfrac{L(\lambda_{\rm emit})}{4\pi d_L^2}\).
                </div>
                <div style="margin-top:6px;">
                    <strong>Hubble relation & Hubble time</strong><br/>
                    \(v = \dot a\, r = H_0 r \Rightarrow H_0 = \dot a / a\).<br/>
                    Hubble time \(t_0 \approx 1/H_0\) (in appropriate units; e.g. H0=70 km/s/Mpc → t0~14 Gyr).
                </div>
                <div style="margin-top:6px;">
                    <strong>ΛCDM luminosity distance (flat)</strong><br/>
                    \(d_L(z) = \dfrac{c(1+z)}{H_0}\int_0^z \dfrac{dz'}{E(z')}\), with \(E(z)=\sqrt{\Omega_m(1+z)^3 + \Omega_k(1+z)^2 + \Omega_\Lambda}\).
                    Typical values: Ω_m=0.3, Ω_k=0, Ω_Λ=0.7.
                </div>
                <div style="margin-top:6px;">
                    <strong>Velocities</strong><br/>
                    Non-relativistic: \(v = c z\).<br/>
                    Relativistic: \(v = c\,\dfrac{(1+z)^2 - 1}{(1+z)^2 + 1}\).
                </div>
                </div>
                """).props("id=planck-info-content role=document aria-live=polite aria-describedby=info-content aria-label='Detailed information about redshift, flux, Hubble relation, Lambda CDM luminosity distance, and velocities.'")
                    
                    aria_button("Close", label="Close", on_click=lambda: info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

            with ui.dialog() as leg_dlg, ui.card():
                ui.label("Legend: Symbols and Units").classes("text-lg font-semibold").props("id=info-title")
                ui.html(r"""
        <ul style="line-height:1.5;">
            <li>c = 299,792.458 km/s — speed of light</li>
            <li>H₀ — Hubble constant [km/s/Mpc]</li>
            <li>z — redshift (dimensionless)</li>
            <li>v<sub>nr</sub>, v<sub>rel</sub> — velocities [km/s]</li>
            <li>d<sub>L</sub> — luminosity distance [Mpc]</li>
            <li>μ — distance modulus (mag)</li>
            <li>m<sub>B</sub>, M<sub>B</sub> — apparent and absolute magnitude</li>
            <li>Ωₘ, Ω_Λ — density parameters (matter, dark energy)</li>
            <li>a(t) — cosmological scale factor</li>
        </ul>
    """).props("role=document aria-live=polite aria-describedby=info-content aria-label='Legend explaining symbols and units used in the Hubble Law with Type Ia Supernovae activity.'")
                aria_button("Close", label="Close", on_click=lambda: leg_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
        

            with ui.row().classes("items-center gap-2 justify-center"):
                aria_button("Dataset","data info",on_click=lambda: datas_dialog.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                aria_button("Info ", "Information",on_click=lambda: info_dlg.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                aria_button("Legend ", "Variable and constants legend",on_click=lambda: leg_dlg.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
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

                
                H0_box = ui.number(label="H₀ [km/s/Mpc]", format="%.3f").classes("w-44").props('aria-label="Hubble constant H0 in km per second per Megaparsec"')
             
                M_B_input = ui.number(label="Absolute Magnitude M_B (SNe Ia)", value=-19.3, format="%.3f").classes("w-52").props('aria-label="Absolute Magnitude of Type Ia Supernovae M sub B"')
                with ui.column().classes("items-start gap-2"):
                    ui.label("Compute H₀ from data (linear fit v = H₀·dₗ)").classes("text-sm text-blue-600")
                    result_label = ui.label("").classes("text-lg font-semibold text-green-600")
                    aria_button("Compute H₀", "compute H0",on_click=lambda: compute_H0()).props("color=primary outline").classes(" w-32 !bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                
                if not os.path.exists(sn_file):
                    ui.label(f"File not found: {sn_file}").classes("text-red-600")
                    return
                df = pd.read_csv(sn_file, sep=r"\s+", comment="#", engine="python")
                if 'zcmb' not in df.columns and 'z' in df.columns:
                    df['zcmb'] = df['z']
                df['zcmb'] = pd.to_numeric(df['zcmb'], errors='coerce')
                df['mb'] = pd.to_numeric(df['mb'], errors='coerce')
                df = df.dropna(subset=['zcmb', 'mb']).reset_index(drop=True)
                c = 299792.458

                df['v_nonrel'] = c * df['zcmb']
                df['v_rel'] = c * (((1+df['zcmb'])**2 - 1)/((1+df['zcmb'])**2 + 1))
                df['mu'] = df['mb'] - M_B_input.value
                df['d_L_Mpc'] = 10**((df['mu']+5)/5)/1e6

                state = {'show_vrel': False, 'show_hubble_line': False, 'show_mag_hubble': False,'show_lcdm': False, 'show_noLambda': False}

        
                plots_row = ui.row().classes("gap-4 mt-4")
                with plots_row:
                
                    vdL_col = ui.column().style("flex: 1")
                    vz_col = ui.column().style("flex: 1")
            
                    mB_col = ui.column().style("flex: 1")
                    mu_col = ui.column().style("flex: 1")
                    
                with ui.dialog() as tablex,ui.card().props('aria-label=table exercise'):
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

                    
                        with ui.dialog() as dlg, ui.card():
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

                    with ui.dialog() as dlg, ui.card():
                        ui.label(f"Linear fit result: H₀ = {H0_est:.2f} km/s/Mpc").classes("text-lg font-semibold text-blue-700")
                        with ui.pyplot():
                            plt.subplot(2, 1, 1)
                            plt.scatter(x, y, s=10, color="blue", label="Data (z < 0.1)")
                            plt.plot(x, H0_est * x, 'r--', label=f'Fit H₀={H0_est:.2f}')
                            plt.xlabel("d_L [Mpc]"); plt.ylabel("v [km/s]")
                            plt.legend(); plt.title("Hubble fit (low-z)")

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
                    vdL_col.clear(); vz_col.clear(); mB_col.clear(); mu_col.clear()
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
                        plt.title("v vs d_L")
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
                        plt.title("v vs z")
                        plt.legend(); plt.tight_layout()
                        aria_chart_label("Plot of velocity v versus redshift z")
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
                        plt.title("m_B vs z")
                        #plt.xscale('log'); plt.yscale('log')
                        plt.legend(); plt.tight_layout()
                        aria_chart_label("Plot of apparent magnitude m sub B versus redshift z")

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
                        plt.title("μ vs z")
                        #plt.xscale('log'); plt.yscale('log')
                        plt.legend(); plt.tight_layout()
                        aria_chart_label("Plot of distance modulus μ versus redshift z")
                aria_button("Add v_rel","add relativistic velocity" ,on_click=lambda: (state.update(show_vrel=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Add Hubble line", "add Hubble line",on_click=lambda: (state.update(show_hubble_line=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Add Hubble mag","add Hubble magnitude", on_click=lambda: (state.update(show_mag_hubble=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Add CDM", "add only matter model",on_click=lambda: (state.update(show_noLambda=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                aria_button("Add ΛCDM", "add cosmological model",on_click=lambda: (state.update(show_lcdm=True), plot_all())).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

            plot_all()


        
    with ui.tab_panels(tabs, value=one):
        with ui.tab_panel(one) as redshift_panel:
            add_redshift_activity(redshift_panel)


        with ui.tab_panel(two) as hubble_panel:
            add_hubble_activity(hubble_panel)
  


@ui.page('/module3')
def module3():
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

  
  .info_box {
    font-size: 18px !important;
    line-height: 1.7;
    text-align: justify;
    margin-bottom: 10px;
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
  }

  
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
   
    ui.add_body_html("""
<script>
if (!window.MathJaxLoaded) {
  window.MathJaxLoaded = true;
  window.MathJax = {
    tex: {inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]},
    svg: {fontCache: 'global'}
  };
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
  s.async = true;
  document.head.appendChild(s);
}
</script>
""")



    main_layout("Module 3: Universe History & CMB")
  
    with ui.tabs().classes('w-full justify-center') as tabs:
        one = ui.tab(' Planck spectrum', icon='auto_awesome').props('aria-label="Activity 1: Planck Spectrum"')
        two = ui.tab('CMB Adiabatic Index', icon='science').props('aria-label="Activity 2: Adiabatic Index"')
        three = ui.tab('Radiation & Matter', icon='public').props('aria-label="Activity 3: Radiation & Matter"')
      

       
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


            cobefile = os.path.join(DATA_DIR, "cobe_firas_monopole.txt")

         
            try:
                df_cobe = pd.read_csv(cobefile, sep=r"\s+", header=0, engine="python")

              
                freq_cm1 = pd.to_numeric(df_cobe["frequency(cm^-1)"], errors="coerce").values
                I_MJy_sr = pd.to_numeric(df_cobe["monopole_spectrum(MJy/sr)"], errors="coerce").values
                err_kJy_sr = pd.to_numeric(df_cobe["uncertainty(kJy/sr)"], errors="coerce").values

            
                mask = np.isfinite(freq_cm1) & np.isfinite(I_MJy_sr)
                freq_cm1, I_MJy_sr, err_kJy_sr = freq_cm1[mask], I_MJy_sr[mask], err_kJy_sr[mask]

              
                cobewl = 1.0 / (freq_cm1 * 100.0)          # [m]
                I_W_m2_Hz_sr = I_MJy_sr * 1e-20           # MJy/sr -> W·m^-2·Hz^-1·sr^-1
                err_W_m2_Hz_sr = err_kJy_sr * 1e-23       # kJy/sr -> W·m^-2·Hz^-1·sr^-1

                # Convert B_nu -> B_lambda: B_lambda = B_nu * c / lambda^2
                cobeI = I_W_m2_Hz_sr * (c / (cobewl ** 2))
                cobeErr = err_W_m2_Hz_sr * (c / (cobewl ** 2))

            
                sort_idx = np.argsort(cobewl)
                cobewl, cobeI, cobeErr = cobewl[sort_idx], cobeI[sort_idx], cobeErr[sort_idx]


                cobedata_loaded = True
            except Exception as e:
                cobewl, cobeI, cobeErr = np.array([]), np.array([]), np.array([])
                cobedata_loaded = False
                coberead_error = str(e)


            user_rows = []         
            generated_rows = []    
            current_temp = None    
            auto_fill = False      


          
            with container:
                title_on_dark("Planck Spectrum — CMB and T₀")
                
                with ui.dialog() as plank,ui.card():
                
                    reference_box("""**Dataset reference**: Fixsen et al. (1996), The Astrophysical Journal. [COBE-FIRAS monopole spetrum](https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_spect.html)""")
                    info_box("**Dataset**:frequency(cm^-1) monopole_spectrum(MJy/sr) uncertainty(kJy/sr)")
                    aria_button("Close","close",on_click=lambda:plank.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    
            
                    

                
                with ui.dialog().props(
                    'aria-modal="true" role="dialog" aria-labelledby="planck-info-title" aria-describedby="planck-info-content"'
                ) as info_dlg, ui.card():
                    ui.label("Information").classes("text-lg font-semibold text-blue-400").props("id=planck-info-title")
                    
                    ui.html(r"""
<div class="prose text-gray-100 text-sm" id="planck-info-content" role="document" aria-live="polite">
<h5>Planck’s Law and Blackbody Radiation</h5>
<p>Planck’s law describes the spectral radiance of an ideal blackbody:</p>
<p>$$B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5} \frac{1}{e^{hc/(\lambda k_B T)} - 1}$$</p>

<p><strong>where:</strong><br>
\(B_\lambda\) = spectral radiance [W·m⁻²·sr⁻¹·m⁻¹]<br>
\(h\) = Planck constant = 6.626×10⁻³⁴ J·s<br>
\(c\) = speed of light = 2.998×10⁸ m/s<br>
\(k_B\) = Boltzmann constant = 1.381×10⁻²³ J/K</p>

<p>The distribution peaks at a wavelength inversely proportional to T:</p>
<p>$$\lambda_{max} T = b$$</p>
<p>with \(b = 2.897771955\times10^{-3}\) m·K.</p>

<p>The COBE/FIRAS measurement confirmed the CMB blackbody at T ≈ 2.725 K.</p>
</div>
""").props("id=planck-info-content role=document aria-live=polite")

                    aria_button("Close","close",on_click=lambda:info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")



                
                with ui.dialog().props(
                    'aria-modal="true" role="dialog" aria-labelledby="planck-legend-title" aria-describedby="planck-legend-content"'
                ) as legend_dlg, ui.card():
                    ui.label("Legend").classes("text-lg font-semibold text-blue-400").props("id=planck-legend-title")
                    ui.markdown(
                        "**Symbols and Units**  \n"
                        "- **λ** = wavelength [m]  \n"
                        "- **ν** = frequency [Hz]  \n"
                        "- **Bλ** = spectral radiance [W·m⁻²·sr⁻¹·m⁻¹]  \n"
                        "- **T** = absolute temperature [K]  \n"
                        "- **h** = Planck constant = 6.626×10⁻³⁴ J·s  \n"
                        "- **c** = speed of light = 2.998×10⁸ m/s  \n"
                        "- **kB** = Boltzmann constant = 1.381×10⁻²³ J/K  \n"
                        "- **b** = Wien constant = 2.898×10⁻³ m·K  \n"
                        "- **I** = intensity [W·m⁻²·sr⁻¹·m⁻¹]  \n"
                        "- **z** = redshift (dimensionless)"
                    ).classes("prose text-gray-100").props("id=planck-legend-content role=document aria-live=polite")

                    aria_button("Close","close",on_click=lambda:legend_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")



            
                with ui.dialog() as temp ,ui.card():
                    info_box(
                "Compare theoretical blackbody curves at different temperatures with the COBE/FIRAS CMB spectrum. "
                "\nSelect or enter a temperature, then build a table of wavelength–intensity pairs. "
                "\nMake sure to use wavelengths in meters and intensities in compatible units."
                "\nClick 'Generate Plot' to overlay your theoretical curve on the COBE data."
                "\nUse one of these interactive blackbody calculators:\n\n"
                "[OpticsTheWebsite](https://www.opticsthewebsite.com/OpticsCalculators)\n\n"
                "[NOAA Planck Calculator](https://ncc.nesdis.noaa.gov/Planck.php)\n\n"
                "[Bodkin Design Blackbody Calculator](https://www.bodkindesign.com/reference-library/blackbody-spectral-radiance-calculator/)\n\n"
                "[SpectralCalc Blackbody Tool](https://spectralcalc.com/blackbody_calculator/blackbody.php)\n\n"
                
            )
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
                        plt.title("COBE/FIRAS CMB Spectrum vs Planck Curve")
                        plt.grid(True, which='both', ls='--', alpha=0.4)
                        plt.legend()
                        plt.tight_layout()

                        aria_chart_label("Plot of COBE/FIRAS CMB spectrum and Planck blackbody curve")
                with ui.dialog() as wien,ui.card():
                        
                    info_box("Wien’s Law Calculator: T=λₘₐₓ/b"
                    
                    "\n\nInsert the observed peak wavelength 'λₘₐₓ' to compute the temperature")

                    lambda_input = aria_input("λₘₐₓ [m]", "Enter peak wavelength in meters").classes("w-full")
                    T_output = ui.label("").classes("text-md font-medium text-blue-600 mt-2")

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

                with ui.dialog() as gen_data_dlg, ui.card().classes('w-full h-full'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label("Generated spectrum from input peak temperature ").classes('text-lg font-bold')
 
                    gen_table = ui.table(
                        columns=[
                            {'name':'wavelength_m','label':'λ [m]','field':'wavelength_m'},
                            {'name':'intensity','label':'Intensity','field':'intensity'}
                        ], 
                        rows=[]
                    ).classes('w-full')
                    aria_button("Close","close",on_click=lambda:gen_data_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.row().classes("w-full justify-center gap-2"):
                    aria_button("Dataset","dataset",on_click=lambda:plank.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button("Info ", "Information",on_click=lambda:info_dlg.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button("Legend ", "Variable and constants legend",on_click=lambda:legend_dlg.open()).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button("CMB spectrum exercise","CMB spectrum exercise",on_click=lambda:temp.open()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    aria_button("Wien law","Wien law",on_click=lambda:wien.open()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                 

                    

                     
                
        
                
                with ui.row().classes('w-full justify-center gap-8'):
                    with ui.column().classes('flex-1'):
                        draw_plot()
                        lambda_peak_box = ui.label("").classes("text-md font-semibold text-blue-400 ")
                    
                    
                
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


        def add_cmb_activity(container):

            cmb_path = os.path.join(DATA_DIR, "dataset_CMB.txt")

            try:
                df = pd.read_csv(cmb_path, comment="#", sep=r"\s+", header=None, engine="python")
                z_obs = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
                T_obs = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
                T_err = pd.to_numeric(df.iloc[:, 3], errors="coerce").values

                valid = np.isfinite(z_obs) & np.isfinite(T_obs) & np.isfinite(T_err)
                z_obs, T_obs, T_err = z_obs[valid], T_obs[valid], T_err[valid]
            except Exception as e:
                with container:
                    ui.label("Error loading CMB dataset (dataset_CMB.txt).").classes("text-red-600")
                    ui.label(str(e))
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
                title_on_dark("CMB Thermodynamics — Discover the Adiabatic Index γ")
                with ui.dialog() as info_dlg, ui.card().classes('p-4 w-full overflow-x-auto'):
                    info_box(
                    "Explore how the temperature of the cosmic microwave background (CMB) changes with redshift.  \n"
                    "Use the slider to adjust the adiabatic index γ and compare with the observed data points."
                   
                )
                    aria_button("Close","close",on_click=lambda:info_dlg.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as ref_dialog, ui.card().classes('p-4 w-full overflow-x-auto'):
                    info_box(
                    
                    "**Dataset**: Redshift (z), CMB Temperature (K), T/z+1, T_error"
                )
                    reference_box(
                    """**Dataset reference**: Riechers et al. 2022, *Nature*, 'Microwave background temperature at a redshift of 6.34 from H2O absorption'."""
                )
                    aria_button("Close","close",on_click=lambda:ref_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as thermo_dialog, ui.card().classes('p-4 w-full overflow-x-auto'):
                    ui.html(r"""
                        <h6><b>Thermodynamics and Adiabatic Transformations</b></h6>

                        <p>Imagine the large-scale Universe as an isolated, closed system — it does not exchange heat or energy with any 'outside'. 
                        Its total internal energy can be approximated as conserved over time. In this context, the early Universe was hot, dense, 
                        and dominated by photon radiation. The cosmic microwave background (CMB) discovered by Penzias and Wilson is the remnant 
                        of that radiation, now cooled to about 2.725 K. Its spectrum is an almost perfect blackbody, and its temperature has decreased 
                        with the expansion of the Universe.</p>

                        <p>The expansion behaves as an <b>adiabatic process</b>: total energy is conserved, the system expands, and the temperature drops 
                        as the volume increases.</p>

                        <p><b>First Law of Thermodynamics:</b> <span class="math">\( \Delta U = Q - L \)</span> → <span class="math">\( dQ = dU + P dV \)</span>  
                        → for adiabatic processes <span class="math">\( dQ = 0 \Rightarrow dU + P dV = 0 \)</span></p>

                        <p>For an ideal gas: <span class="math">\( P V^\gamma = \text{const} \)</span> → <span class="math">\( T V^{\gamma-1} = \text{const} \)</span> → <span class="math">\( T \propto V^{1-\gamma} \)</span></p>
                        <p>The adiabatic index <span class="math">\( \gamma = C_p / C_v \)</span> depends on the degrees of freedom of the gas (equipartition of energy).</p>

                        <ul>
                        <li>Monatomic gas: <span class="math">\( \gamma = 5/3 \)</span></li>
                        <li>Diatomic gas: <span class="math">\( \gamma = 7/5 \)</span></li>
                        <li>Polyatomic (photon gas): <span class="math">\( \gamma = 4/3 \)</span></li>
                        </ul>

                        <p><b>In cosmology:</b> Using the Stefan–Boltzmann law for radiation energy density, <span class="math">\( u = \alpha_{\rm rad} T^4 \)</span> with <span class="math">\( \alpha_{\rm rad} = 4\sigma/c \approx 7.56\times10^{-16} \, \rm J \, m^{-3} K^{-4} \)</span>, and <span class="math">\( P = u/3 \)</span>,</p>
                        <p>we obtain <span class="math">\( \frac{1}{T} \frac{dT}{dt} = -\frac{1}{3V} \frac{dV}{dt} \)</span>. With the scale factor <span class="math">\( a(t) \)</span>, since <span class="math">\( V \propto a^3 \)</span>, we get <span class="math">\( T \propto a^{-1} \)</span>.</p>
                        <p>Because redshift relates to the scale factor as <span class="math">\( 1+z = 1/a \)</span>, we have <span class="math">\( T(z) \propto (1+z) \)</span>.</p>
                        <p>In general form, for any <span class="math">\( \gamma \)</span>: <span class="math">\( T(z) = T_0 (1+z)^{3(\gamma-1)} \)</span>.</p>
                        """).props('tabindex=0 role=document aria-label="Thermodynamics and Adiabatic Transformations information"')

                    aria_button("Close", label="Close the box", on_click=lambda: thermo_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                        


                      
                with ui.dialog() as legend_dialog, ui.card().classes('p-4 w-full overflow-x-auto'):
                    ui.html(r"""
                        <h6><b>Legend: Symbols and Units</b></h6>

                        <ul style="line-height:1.6; font-size:0.95rem;">
                        <li><b>T</b> = Temperature [K]</li>
                        <li><b>V</b> = Volume [m³]</li>
                        <li><b>P</b> = Pressure [Pa]</li>
                        <li><b>γ</b> = Adiabatic index (Cp/Cv)</li>
                        <li><b>Cp</b>, <b>Cv</b> = Specific heats [J·mol⁻¹·K⁻¹]</li>
                        <li><b>kB</b> = Boltzmann constant = 1.380649×10⁻²³ J·K⁻¹</li>
                        <li><b>R</b> = Gas constant = 8.314 J·mol⁻¹·K⁻¹</li>
                        <li><b>a_rad</b> = Radiation density const. = 7.56×10⁻¹⁶ J·m⁻³·K⁻⁴</li>
                        <li><b>σ</b> = Stefan–Boltzmann const. = 5.67×10⁻⁸ W·m⁻²·K⁻⁴</li>
                        <li><b>z</b> = Cosmological redshift (dimensionless)</li>
                        <li><b>λ_obs</b>, <b>λ_em</b> = Observed / Emitted wavelength [m]</li>
                        <li><b>a(t)</b> = Scale factor (dimensionless)</li>
                        </ul>
                        """).props('tabindex=0 role=document aria-label="Legend of variables and constants used in the exercise"')

                    aria_button("Close", label="Close the box", on_click=lambda: legend_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.row().classes("w-full justify-center gap-1 "):

               
                    with ui.column().classes("flex-1"):

                        with ui.row().classes("justify-center gap-1 w-full"):
                            aria_button("Instruction","instruction",on_click=lambda:info_dlg.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                            aria_button("Dataset","dataset",on_click=lambda:ref_dialog.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                            aria_button(                            "Info Thermodynamics",
                            "Read detailed information about thermodynamics and adiabatic transformations",
                            on_click=lambda:                                 thermo_dialog.open()                        ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                            aria_button(                            "Legend","Open legend of symbols and units",
                            on_click=lambda: 
                                legend_dialog.open()                           
                        ).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
                        
                        with ui.row().classes("justify-center gap-1 w-full"):
                            T0_input = ui.number(value=T0_default, label="T₀ [K] (present-day)", format="%.6g") \
                            .props('aria-label="Present-day CMB temperature input in Kelvin"')
                            gamma_box = ui.markdown("").classes("p-2 border rounded ").props("aria-live=polite")
                            chi2_box = ui.markdown("").classes("p-2 border rounded").props("aria-live=polite")
                        with ui.row().classes("justify-center gap-1 w-full"):
                            gamma_slider = aria_slider(min=0.0, max=2.0, value=gamma_default, step=0.0001,aria_label="Gamma adiabatic index slider").props('aria-describedby=gamma_slider_label label-always color="light-blue"')
                            

                     
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


                    with ui.column().classes("flex-1"):
                        with ui.card().classes("p-6 w-full max-w-4xl min-w-[600px] !bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown(" CMB Thermodynamics Exercise: fill in the missing variables").classes("text-xl font-bold text-blue-300").props("aria-label='CMB Thermodynamics Exercise Title'")
                           

                            answer_du, answer_dq, answer_pdv, answer_pv, answer_gamma, answer_cp, answer_cv, answer_meno1, answer_zero, answer_zrel, answer_gamma_uno, answer_p, answer_v, answer_32, answer_T, answer_52, answer_75, answer_53, answer_four, answer_3, answer_1, answer_lambda_obs, answer_lambda_emit, answer_1z = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

                            @ui.refreshable
                            def show_adiabatic_exercise():
                                with ui.row().classes('w-full items-start justify-between gap-1'):
                                    with ui.column().classes('flex-1 gap-1'):

                                        ui.label("1) First Law Thermodynamics").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            answer_dq["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("= ")
                                            answer_du["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("+")
                                            answer_pdv["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")

                                        ui.label("2) Adiabatic condition").classes("step text-blue-200 ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            answer_dq["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(" = ")
                                            answer_zero["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            answer_du["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("+")
                                            answer_pdv["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("= 0")

                                        ui.label("3) Adiabatic Transformation").classes("step text-blue-200 ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            answer_p["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("·")
                                            answer_v["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("^")
                                            answer_gamma["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(" = const")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("T ∝ ")
                                            answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("^(")
                                            answer_gamma_uno["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(")")
                                    
                                        ui.label("4) Adiabatic Index").classes("step text-blue-200 ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("γ = ")
                                            answer_cp["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("/")
                                            answer_cv["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                    with ui.column().classes('flex-1 gap-1'):
                                        ui.label("5) Energy Equipartition").classes("step text-blue-200 ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("Monoatomic gas: K = ")
                                            answer_32["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("kB·")
                                            answer_T["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(" → γ = ")
                                            answer_53["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("Biatomic gas: U = ")
                                            answer_52["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("nR·")
                                            answer_T["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(" → γ = ")
                                            answer_75["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")

                                        ui.label("6) Stefan-Boltzmann law for radiation").classes("step text-blue-200 ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("U = a_rad·")
                                            answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("^")
                                            answer_four["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("·")
                                            answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("P = ")
                                            answer_1["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("/")
                                            answer_3["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("·")
                                            answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("^")
                                            answer_four["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("·")
                                            answer_v["el4"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                    with ui.column().classes('flex-1 gap-1'):
                                        ui.label("7) Doppler–Redshift relation").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("1 + z = ")
                                            answer_lambda_obs["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("/")
                                            answer_lambda_emit["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("λ_emitted ∝ a(t) → universe expansion factor  ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("a(t)=")
                                            answer_1["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("/")
                                            answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            
                                        ui.label("8) Cosmological Application").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("The relation temperature-redshift: T(z) = T₀·(1+z)")
                                    
                                            
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("T ∝ a(t)^(")
                                            answer_meno1["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(")")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("Considering a section of the expanding universe V ∝ a(t)³")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("The adiabatic expansion:T ∝ a(t)^(3(")
                                            answer_gamma_uno["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("))")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("T ∝ ")
                                            answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label("^(")
                                            answer_zrel["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md !bg-gray-800 transition-colors duration-200")
                                            ui.label(")")

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
                title_on_dark("Radiation–Matter Equivalence Epoch")
                with ui.dialog() as i_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    info_box(
                    "Explore the epoch where the energy density of matter and radiation were equal. "
                    "\nFill in the missing formulas to derive the physical laws governing both components. "
                    "\nOnce all inputs are correct, the two analytical curves will appear on the graph. "
                    "\nThen, estimate the redshift and temperature of the equivalence epoch."
                )
                    aria_button("Close", label="Close the box", on_click=lambda: i_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                with ui.dialog() as eq_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    ui.html(r"""
                <h6><b>Equivalence between Matter and Radiation Densities</b></h6>

                <p>After the Big Bang, the Universe passed through three main eras:</p>
                <ul>
                <li><b>Radiation-dominated epoch:</b> <span class="math">\( \rho_r > \rho_m \)</span></li>
                <li><b>Matter–radiation equality:</b> <span class="math">\( \rho_r = \rho_m \)</span></li>
                <li><b>Matter-dominated epoch:</b> <span class="math">\( \rho_m > \rho_r \)</span></li>
                </ul>

                <p>During the cosmic expansion, the density of matter decreases as <span class="math">\( a^{-3} \)</span>, 
                while radiation decreases faster <span class="math">\( a^{-4} \)</span> since photons lose energy by redshifting 
                (<span class="math">\( \lambda \propto a \Rightarrow E \propto 1/a \)</span>).</p>

                <p>From classical physics:</p>
                <ul>
                <li>Non-relativistic matter: <span class="math">\( E \approx mc^2 \Rightarrow \rho_m \propto a^{-3} \)</span></li>
                <li>Radiation: <span class="math">\( E_\gamma = h\nu = hc/\lambda \Rightarrow \rho_r \propto a^{-4} \)</span></li>
                <li>Using Stefan–Boltzmann law <span class="math">\( \rho_r = \alpha_{\rm rad} T^4 \)</span> and adiabatic expansion <span class="math">\( T \propto 1/a \)</span>, we again get <span class="math">\( \rho_r \propto a^{-4} \)</span></li>
                </ul>

                <p>The equality redshift <span class="math">\( z_{\rm eq} \)</span> satisfies <span class="math">\( \rho_m(z_{\rm eq}) = \rho_r(z_{\rm eq}) \)</span>.</p>
                """).props('tabindex=0 role=document aria-label="Equivalence between matter and radiation densities information"')

                    aria_button("Close", label="Close the box", on_click=lambda: eq_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                


         
                with ui.dialog() as eq_legend_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    ui.html(r"""
                <h6><b>Legend: Symbols and Constants</b></h6>

                <ul style="line-height:1.6; font-size:0.95rem;">
                <li><b>ρ_m, ρ_r</b> = matter / radiation energy density [kg·m⁻³]</li>
                <li><b>Ω_m₀, Ω_r₀</b> = present-day density parameters (dimensionless)</li>
                <li><b>ρ_c</b> = critical density = <span class="math">\( 3 H_0^2 / 8 \pi G \)</span></li>
                <li><b>H₀</b> = Hubble constant = 2.27×10⁻¹⁸ s⁻¹</li>
                <li><b>G</b> = gravitational constant = 6.67×10⁻¹¹ m³·kg⁻¹·s⁻²</li>
                <li><b>T₀</b> = present-day CMB temperature = 2.725 K</li>
                <li><b>z</b> = redshift (dimensionless)</li>
                <li><b>a(t)</b> = scale factor = 1/(1+z)</li>
                <li><b>α_rad</b> = radiation density constant = 7.56×10⁻¹⁶ J·m⁻³·K⁻⁴</li>
                <li><b>h</b> = Planck constant = 6.626×10⁻³⁴ J·s</li>
                <li><b>c</b> = speed of light = 2.998×10⁸ m/s</li>
                <li><b>λ</b> = photon wavelength [m]</li>
                <li><b>ν</b> = photon frequency [Hz]</li>
                </ul>
                """).props('tabindex=0 role=document aria-label="Legend of symbols and constants used in the exercise"')

                    aria_button("Close", label="Close the box", on_click=lambda: eq_legend_dialog.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                


                with ui.row().classes("w-full justify-center items-start gap-6 "):
                    with ui.column().classes("flex-1"):
                        with ui.card().classes("p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown(" Radiation & Matter Density Evolution").classes("text-2xl font-bold mb-4 text-blue-300").props("aria-label='Radiation and Matter Density Evolution Exercise '")
                            
                            @ui.refreshable
                            def show_formulas():
                                with ui.row().classes('w-full  justify-center gap-1'):
                                  
                                  
                                    with ui.column().classes("flex-1 gap-1"):
                                        ui.label("1) Energy density definition").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("ρ = ")
                                            answer_u["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")                         
                                            ui.label("/")
                                            answer_v["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                        
                                        ui.label("2) Matter particles energy ").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("The energy of non relativistic matter particles:")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("Eₘ =(")   
                                            answer_m["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·")
                                            answer_c["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("^2)")
                                            ui.label("+(1/2·")
                                            answer_m["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·")
                                            answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("^2)")
                                            
                                        ui.label("3) Matter energy density").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"): 
                                            ui.label("ρₘ = ") 
                                            answer_E["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("/")
                                            answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("Considering a portion of universe: V ∝ a(t)³")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("ρₘ=Ωm ·ρc with ρc=(3·H0²)/(8·π·G)")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("ρₘ(a) =ρₘ0· a^(")
                                            answer_m3["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")= ρₘ0·(")
                                            answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")^")
                                            answer_3["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("with ρₘ0=Ωm0 ·ρc ")
                                    with ui.column().classes("flex-1 gap-1"):
                                    

                                        ui.label("4) Radiation energy density").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("The energy for photons: Eᵣ =")
                                            answer_h["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·")
                                            answer_nu["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("=")
                                            answer_h["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·")
                                            answer_c["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("/")
                                            answer_lambda["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("The wavelength stretches with universe expansion: λ ∝ a(t)")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("The number of photons per volume decreases with expansion: n ∝ a(t)^(-3)")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("Every photon energy decreases with expansion: Eᵣ ∝ a(t)^(-1)")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("From the energy equation: ρᵣ= ")
                                            answer_n["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·")
                                            answer_E["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("∝ a(t)^(")
                                            answer_m4["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")")
                                    with ui.column().classes("flex-1 gap-1"):
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("From the Stefan-Boltzmann law: ρᵣ= a_rad ·")
                                            answer_T["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("^")
                                            answer_4["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("with T ∝ a(t)^(-1)")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("ρᵣ=Ωr ·ρc ")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("ρᵣ(a) =ρᵣ0· a^(")
                                            answer_m4["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")= ρᵣ0·(")
                                            answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")^")
                                            answer_4["el2"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("with ρᵣ0=Ωr0 ·ρc ")
                                            

                                    
                                    
                                        ui.label("5) Equivalence condition").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            answer_rhom["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("(z_eq) = ")
                                            answer_rhor["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("(z_eq)  ")

                                        ui.label("6) Redshift of equivalence").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("1 + z_eq = ")
                                            answer_rhom0["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("/")
                                            answer_rhor0["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")

                                        ui.label("7) Temperature–redshift relation").classes("step text-blue-200")
                                        with ui.row().classes("items-center text-sm flex-wrap"):
                                            ui.label("T(z) = ")
                                            answer_T0["el"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label("·(")
                                            answer_1z["el3"] = aria_formula_input().props("dense filled").classes("w-10 border border-gray-600 rounded-md")
                                            ui.label(")")
                                        

                            

                            show_formulas()
                            formula_button_slot = ui.row().classes(" w-full items-center justify-center")

                    with ui.column().classes("flex-1 justify-start !bg-gray-900 p-4 rounded-xl shadow-lg"):

              
                        with ui.row().classes("justify-center w-full gap-2 flex-wrap"):
                            aria_button("Instruction","instruction",on_click=lambda:i_dialog.open()).classes("!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded ")
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

                           
                                plt.title("Matter and Radiation Density Evolution")
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

            
                

          
                

               
                     



        
    with ui.tab_panels(tabs, value=one).classes('w-full p-4'):

        with ui.tab_panel(one).props('role=tabpanel aria-label="Planck spectrum "'):
            planck_container = ui.column().classes('w-full p-4')
            add_planck_activity(planck_container)
        with ui.tab_panel(two).props('role=tabpanel aria-label="CMB Adiabatic Index "'):
            cmb_container = ui.column().classes('w-full p-4')
            add_cmb_activity(cmb_container)
            
        with ui.tab_panel(three).props('role=tabpanel aria-label="Radiation & Matter"'):
            ui.label("Activity 2: Radiation & Matter").classes('text-lg text-gray-600')
            radmat_container = ui.column().classes('w-full p-4')
            add_radiation_matter_activity(radmat_container)
    



 


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

  
  .info_box {
    font-size: 18px !important;
    line-height: 1.7;
    text-align: justify;
    margin-bottom: 10px;
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
  }

  
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
    




    main_layout("Module 4: Dark Matter ")
    

    with ui.column().classes('w-full p-4 gap-4'):
        with ui.tabs().classes('w-full') as tabs:
            zero,one, two, three, four,five = ui.tab('Kepler laws planets').props('role=tab aria-selected=true'), ui.tab('Galaxy rotation curve').props('role=tab aria-selected=false'), ui.tab('Galaxy mass & DM').props('role=tab aria-selected=false'), ui.tab('Cluster velocity distribution').props('role=tab aria-selected=false'), ui.tab('Cluster mass & DM').props('role=tab aria-selected=false'),ui.tab('CMB').props('role=tab aria-selected=false')
      

        with ui.tab_panels(tabs, value=zero).classes('w-full'):
            
            
            with ui.tab_panel(zero).props('role=tabpanel'):
                #with ui.card().classes("p-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                title_on_dark("Kepler's Laws and Planetary Data")
                with ui.dialog() as info_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    info_box( "Explore the orbital characteristics of planets in our solar system using Kepler's Laws."
                    "\n\n The first plot shows the orbital velocity vs semi-major axis for each planet, illustrating Kepler's Second Law."
                    "\n\n The second plot displays the orbital period versus semi-major axis, demonstrating Kepler's Third Law."
                    "\n\n The third plot presents the mass of each planet vs radius.").props('role=dialog aria-modal=true aria-label=Descriptive text about Kepler Laws activity')

                    aria_button("Close", "close the box",on_click=info_kepler.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                with ui.dialog() as data_kepler, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    info_box( "**Dataset variables**: Celestial_Body (name of the planet), SemiMajorAxis(km) (orbital radius in km), Velocity(km/s) (orbital velocity in km/s), Period(days) (orbital period in days), Mass(kg) (mass of the planet in kg).")
                    reference_box(
    """**Dataset reference**: [NASA-SSD](https://ssd.jpl.nasa.gov/planets/) ;[Orbital Mechanics](https://orbital-mechanics.space/reference/planetary-parameters) ;Ryan S. Park, William M. Folkner, James G. Williams, and Dale H. Boggs. The JPL Planetary and Lunar Ephemerides DE440 and DE441. The Astronomical Journal, 161(3):105, February 2021.; Brandon Rhodes. Skyfield: high precision research-grade positions for planets and Earth satellites generator. July 2019""").classes('text-base italic')
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
                        plt.title("Kepler III: Period vs Semi-major axis")
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
                        plt.title("Mass vs Semi-major axis")
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
                    
                with ui.row().classes('w-full mb-6 justify-center'):

                    
                    with ui.grid(columns=6).classes('w-full'):
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
                    
                with ui.row().classes('w-full items-start justify-center gap-8'): 
                    
                  
                    with ui.column().classes('w-full md:w-1/3 lg:w-1/3 min-w-[280px]'):
                        with ui.row().classes('flex-1'):
                        
                            ui.label("Select a Planet").style("font-size:20px; font-weight:bold; margin-bottom:6px;").props('id=planet_selector_label role=heading aria-level=3 tabindex=0')
                       

                            selector = ui.select(
                        planets,
                        value="Earth" if "Earth" in planets else planets[0],
                        label="Planet"
                    ).classes('w-48').props('id=planet_selector aria-labelledby=planet_selector_label role=listbox tabindex=0')
                            
                        with ui.row().classes('flex-1'):
                            info_panel(selector.value)
                        
                  
                    with ui.column().classes('flex-1'):
                       
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
                
                    info_box(
                    "Explore the rotation curve of the NGC3198 galaxy."
                    "\n\n The **red curve** shows the **baryonic prediction** with a Keplerian-like trend (initial rise followed by a decrease). "
                    "\n\n The **green curve** simulates the addition of a **dark matter halo** to flatten the curve and match observations."
                    "\n\n**Slider**: 0 = total baryonic curve, 1 = total curve with full dark matter halo contribution.")
                    aria_button("close",'close',on_click=info_galaxy.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.dialog() as data_galaxy,ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                
                    info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                
                    reference_box(
    """**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')
                    aria_button("close",'close',on_click=data_galaxy.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
   

                
                
                with ui.row().classes('w-full '):
                    with ui.dialog() as velocity_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        ui.html(r"""
       <h3 class="text-xl font-semibold mb-2">Note</h3>

<p><b>Step 1:</b> Baryonic velocity (from data):  
<span class="math">\( v_{\mathrm{bar}}^2(r) = v_{\mathrm{gas}}^2(r) + v_{\mathrm{disk}}^2(r) + v_{\mathrm{bulge}}^2(r) \)</span></p>

<p><b>Step 2:</b> Baryonic mass:  
<span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2(r)}{G} \)</span></p>

<p><b>Step 3:</b> Total observed mass from velocity data:  
<span class="math">\( M_{\mathrm{tot}}(r) = \frac{r \, v_{\mathrm{obs}}^2(r)}{G} \)</span></p>

<p><b>Step 4:</b> Observed dark matter density (from data):  
<span class="math">\( \rho_{\mathrm{DM}}(r) = \frac{1}{4 \pi G r^2} \, \frac{d}{dr} \Bigg[ r^2 \Big( \frac{v_{\mathrm{obs}}^2(r)}{r} - \frac{v_{\mathrm{gas}}^2(r)}{r} - \frac{v_{\mathrm{disk}}^2(r)}{r} - \frac{v_{\mathrm{bulge}}^2(r)}{r} \Big) \Bigg] \)</span></p>

<p><b>Step 5:</b> NFW dark matter profile:  
<span class="math">\( \rho_{\mathrm{NFW}}(r) = \frac{\rho_s}{\left(\tfrac{r}{r_s}\right)\left(1 + \tfrac{r}{r_s}\right)^2} \)</span>  
with <span class="math">\( \alpha=1, \beta=3, \gamma=1 \)</span>,  
<span class="math">\( R_{200} = \left[\tfrac{3 M_{200}}{4\pi \cdot 200 \rho_{\mathrm{crit}}}\right]^{1/3}, \;\; r_s = \tfrac{R_{200}}{c} \)</span>,  
<span class="math">\( c = 0.3 \times 11.7 \left(\tfrac{M_{200}}{10^{11} M_\odot}\right)^{-0.075} \)</span></p>

<p><b>Step 6:</b> Match observed density with NFW:  
<span class="math">\( \rho_{\mathrm{NFW}}(r_{\mathrm{match}}) = \rho_{\mathrm{DM}}(r_{\mathrm{match}}) \)</span>  
→ Obtain <span class="math">\( M_{200}, \rho_s, r_s \)</span></p>

<p><b>Step 7:</b> Enclosed dark matter mass (NFW):  
<span class="math">\( M_{\mathrm{DM}}(r) = 4 \pi \rho_s r_s^3 \left[\ln(1+x) - \frac{x}{1+x}\right], \;\; x = \tfrac{r}{r_s} \)</span></p>

<p><b>Step 8:</b> Dark matter rotational velocity:  
<span class="math">\( v_{\mathrm{DM}}(r) = \sqrt{\frac{G M_{\mathrm{DM}}(r)}{r}} \)</span></p>

<p><b>Step 9:</b> Total simulated velocity curve (linked to DM slider):  
<span class="math">\( v_{\mathrm{tot,sim}}(r) = \sqrt{\tfrac{G \, (M_{\mathrm{bar}}(r) + f\,M_{\mathrm{DM}}(r))}{r}}, \;\; f \in [0,1] \)</span></p>

<p><b>Step 10:</b> Fit quality (χ²/d.o.f):  
<span class="math">\( \chi-squared = \sum_i \left(\tfrac{v_{\mathrm{obs}}(r_i) - v_{\mathrm{tot,sim}}(r_i)}{\sigma_i}\right)^2, \;\; \chi-squared_{\mathrm{dof}} = \tfrac{\chi-squared}{N_{\mathrm{obs}} - N_{\mathrm{params}}} \)</span></p>

<p><b>Plot:</b>  
<ul>
<li>X-axis: radius (data)</li>
<li>Y-axis: <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue with grey error bars), <span class="math">\( v_{\mathrm{tot,sim}} \)</span> (green)</li>
</ul>
</p>

        """).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the computational steps of the galaxy rotation curve activity')

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
                        
                with ui.dialog() as image_dialog, ui.card().classes('p-4 w-full max-w-[600px]'):
                    ui.html("<h5>Galaxy Image</h5>")
                    update_image()
                    aria_button("close",'close',on_click=image_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.dialog() as table_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
                    update_table()
                    aria_button("close",'close',on_click=table_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
               
                
                
                with ui.dialog() as chi2_info_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    subtitle("Parabolic interpolation and compute chi^2 minimization ")
                    info_box("Use the formula below to compute the χ² minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)). "
                        "\n\n 1a) Choose 3 values of dark matter fraction (f) with the slider above to compute the χ² at that point."
                        "\n\n 1b) Press 'Add χ² point' button to add the point to the χ² plot."
                        "\n\n 2a) Input 3 values for dark matter mass points (x-coordinates),"
                        "\n\n 2b) Press 'compute χ² and minimum' button to calculate the χ² at those points and check the minimum on the plot.")
                        
                    ui.html(r"""
        <p style="font-family:monospace; background:#e0f0ff; padding:10px; border:1px solid #88c; border-radius:8px; color:#111;">

        <b>Formula:</b><br>
        $$
        x_{min} = \frac{1}{2} \cdot \frac{ (x_1^2-x_2^2) \chi^2(x_3) + (x_2^2-x_3^2) \chi^2(x_1) + (x_3^2-x_1^2) \chi^2(x_2) }{ (x_1-x_2) \chi^2(x_3) + (x_2-x_3) \chi^2(x_1) + (x_3-x_1) \chi^2(x_2) }
        $$
        </p>""").classes('w-full').props(' aria-label=Parabolic interpolation formula for chi-squared minimization')
                    aria_button("close",'close',on_click=chi2_info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.row().classes('w-full items-center gap-4'):
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
                    with ui.row().classes('gap-4 justify-center'):
                        aria_button("Galaxy panel Instruction","Instruction for galaxy panel",on_click=safe_click(lambda: [info_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Dataset info", "Info galaxy dataset",on_click=safe_click(lambda: [data_galaxy.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [velocity_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded"
            )
                        aria_button("Image", "Show galaxy image",on_click=lambda: image_dialog.open()).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Table", "Show galaxy data table",on_click=lambda: table_dialog.open()).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        aria_button("Info chi2", "Read detailed information about chi2 ", on_click=chi2_info_dialog.open).classes(
                "!bg-blue-500 hover:!bg-blue-700 text-white font-bold py-2 px-4 rounded" )
                        
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
                

              

                chi2_result_label = None

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
              
                def plot_chi2_user_curve():
                    global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED
                    

                    
                    chi2_plot_container.clear()
                    plt.close()
                    with chi2_plot_container:
                        with ui.pyplot(figsize=(6,4), close=False,clear=True):
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
                            ax.set_xlabel("DM mass ($M_\\odot$)")
                            ax.set_ylabel(r"χ²/dof")
                            ax.grid(True)
                            handles, labels = ax.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            if by_label: ax.legend(by_label.values(), by_label.keys())
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
                        
                        
                
                    
                    
                chi2_dialog = ui.dialog().props('role="dialog" aria-modal="false" no-dim position="right" aria-label="Chi-squared minimization and parabolic interpolation"')

                with chi2_dialog, ui.card().classes('w-full max-w-xl mx-auto h-screen flex flex-col'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label('χ² Minimization and Parabolic Interpolation').classes('text-2xl font-bold').props('aria-label="Chi-squared minimization and parabolic interpolation"')
                        aria_button('close','close', on_click=chi2_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
               
                    
                    with ui.column().classes('w-full flex-1 overflow-y-auto space-y-4'):
               
                     
                            
                   

                        chi2_plot_container = ui.column().classes('w-full')
                        plot_chi2_user_curve()
                        
                        with ui.row().classes("gap-2 justify-center"):
                            
                            aria_button("Add chi2 point", "Add a new chi-squared point", 
                                        on_click=lambda: add_chi2_point()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                            aria_button("Refresh chi2 plot", "Refresh the chi-squared plot", 
                                        on_click=lambda: refresh_chi2_plot()).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                            
                        slider_result_label = ui.label("Results: ---").classes("text-green-600 font-bold mt-2").props('role=status aria-live=polite aria-atomic=true')
                        
                   
                        

                        with ui.grid(columns=2).classes("w-full gap-3 "):
                            input_a = ui.number(label='Mass Point 1 (x₁)', format='%.2e').classes('flex-1 font-bold').props('aria-label="First mass point for chi-squared computation"')
                            fa_in = ui.number(label="f(a) = χ²(x₁)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point a"')
                            input_b = ui.number(label='Mass Point 2 (x₂)', format='%.2e').classes('flex-1 font-bold').props('aria-label="Second mass point for chi-squared computation"')
                            fb_in = ui.number(label="f(b) = χ²(x₂)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point b"')
                            input_c = ui.number(label='Mass Point 3 (x₃)', format='%.2e').classes('flex-1 font-bold').props('aria-label="Third mass point for chi-squared computation"')
                            fc_in = ui.number(label="f(c) = χ²(x₃)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point c"')
                        fa_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        fb_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        fc_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        input_a.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        input_b.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        input_c.on('update:model-value', lambda e: refresh_formula_from_inputs())
                        refresh_formula_from_inputs()
                        result_label = ui.label("Results: ---").classes("text-green-600 font-bold mt-2 font-bold").props('role=status aria-live=polite aria-atomic=true')
                        
                        with ui.row().classes("w-full justify-center"):
                            aria_button("Compute χ² and minimum", "Compute chi-squared and minimum from three input dark matter masses values",
                                        on_click=lambda: initialize_parabolic_points()).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                #slider_result_label = ui.label("").classes("text-green-600 font-bold mt-2").props('role=status aria-live=polite aria-atomic=true')
                points_display = ui.column()
                history_display = ui.column()

                with ui.row().classes('w-full no-wrap justify-center gap-x-8'):
                    with ui.column().classes('flex-1 min-w-0 items-center gap-x-4 '):
                    
                   
                        plot_container = ui.column().classes("flex-1 ") 
                        aria_button("Compute chi2 ", "Open Chi-Squared Minimization Tool", on_click=chi2_dialog.open).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")
                    
             
                    mass_plot_container = ui.column().classes("flex-1 ")
                    
                
                    morph_plot_container = ui.column().classes("flex-1 ")
                

                
                with ui.row().classes("gap-4 mt-4"):
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
                             
                                
                                plt.plot(r_ngc, v_baryonic, color='red', linewidth=2, label=f'Keplerian velocity (Baryonic mass: {M_vis_tot:.2e} M☉)')
                                plt.plot(r_ngc, v_total_curve, linewidth=2, color='green', label=f"Simulated velocity: {v_total_curve.max():.1f} km/s "
                    f"DM mass: {M_dm_tot:.2e} M☉, {dm_fraction:.1f}% of visible - "
                    f"Fit $\\chi^2$/dof: {chi2_dof:.2f}")
                                plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', markersize=4, color='blue', ecolor='gray', capsize=2, label=f'Observed velocity:{v_obs_ngc.max():.1f} km/s ', zorder=5)
                                #plt.text(0.05, 0.95, f"Fit quality (χ²/dof): {chi2_dof:.2f}", transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
                                plt.xlabel('Radius (kpc)', fontsize=14)
                                plt.ylabel('Rotation Speed (km/s)', fontsize=14)
                                galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                plt.title(f'Galaxy Rotation Curve: {galaxy_name_for_title}', fontsize=16,fontweight='bold')


                                plt.ylim(0, max(300, 1.1 * np.max(v_obs_ngc + v_err_ngc)))
                                plt.xlim(0, r_ngc.max() * 1.1)
                                plt.grid(True)
                                plt.legend(loc='upper right', fontsize=14)
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
                           
                               
                                plt.plot(r_ngc, m_baryonic, "r-", lw=2, label=f"Baryonic mass: {np.max(m_baryonic):.2e} M☉")
                                plt.plot(r_ngc, m_total_obs, "b-", lw=2, label=f"Total mass: {np.max(m_total_obs):.2e} M☉")
                                plt.plot(r_ngc, m_total_model, "g-", lw=2, label=f"Simulated mass DM: {np.max(m_total_model):.2e} M☉")
                             
                                plt.xlim(x_min_mass, x_max_mass)
                                plt.ylim(y_min_mass, y_max_mass)

                                plt.xlabel("Radius (kpc)", fontsize=14)
                                plt.ylabel("Mass ($M_\\odot$)", fontsize=14)
                                galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                plt.title(f"Mass vs radius: {galaxy_name_for_title}", fontsize=16,fontweight='bold')

                                plt.legend(loc='upper right', fontsize=14)
                                plt.grid(True)
                                plt.tight_layout()

                                ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Total mass of the galaxy (from data), baryonic mass (computed from data) and simulated mass updated with new dark matter value'
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
                                with ui.pyplot(figsize=(4,4), close=False,clear=True):
                                 
                                    
                                    ax = plt.gca()

                                    #maxR = max(R_halo, R_disk, R_bulge) * 1.2
                                    maxR = float(np.max(r_ngc)) * 1.1
                                    bulge = plt.Circle((0,0), R_bulge, color='gold', alpha=0.6, label='Bulge')
                                    ax.add_artist(bulge)

           
                                    disk = plt.Circle((0,0), R_disk, color='red', alpha=0.3, label=f'Disk (M={M_disk:.1e})')
                                    ax.add_artist(disk)

                                    R_halo = min(R_halo, maxR) 
                                    if f > 0 and R_halo > 0:
                                        halo = plt.Circle((0,0), R_halo, color='green', alpha=0.15,label=f'Halo DM (M={M_dm_tot:.1e}, f={f:.2f})')
                                        ax.add_artist(halo)
                                    bh = plt.Circle((0,0), 0.1, color='black', label='Black Hole')
                                    ax.add_artist(bh)


                                    
                                    ax.set_xlim(-maxR, maxR)
                                    ax.set_ylim(-maxR, maxR)
                                    ax.set_aspect('equal', 'box')
                                    ax.set_xlabel("x [kpc]",fontsize=8)
                                    ax.set_ylabel("y [kpc]",fontsize=8)
                                    galaxy_name_for_title = current_galaxy_name.removesuffix('.txt')
                                    ax.set_title(f"Galaxy structure from above: {galaxy_name_for_title}",fontsize=10,fontweight='bold')

                                    ax.legend(loc='upper right',fontsize=8)
                                    plt.tight_layout()
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Galaxy morphology representation (from above), showing disk with baryonic component (stars,gas) and halo appears adding dark matter (slider)'
)

                                    #plot_info_box_compact2({    "Disk and Bulge Mass": f"{M_bary_tot:.2e} M☉",    "DM Mass": f"{M_dm_tot:.2e} M☉"})
                 
                    
                    
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
                            slider_result_label.set_text(f"DM min ≈ {xmin:.3e}, χ² min ≈ {ymin:.4f}")
                            #else:
                            #    slider_result_label.set_text("Points are collinear.")
                        else:
                            slider_result_label.set_text("Add more points to find the minimum.")

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
                       
                        slider_result_label.set_text("")
                        
                       
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
                        slider_result_label.set_text("Results: ---")
                        update_all_plots.refresh()

                update_all_plots()
                  

                    


                
            '''
plot_info_box_compact2({
    "Visible (luminous) mass": f"{M_vis_tot:.2e} M☉",
    "Dark matter mass": f"{M_dm_tot:.2e} M☉ ({dm_fraction:.1f}% of visible)",
    "Component contribution": f"Gas {v_gas_frac:.1f}%, Disk {v_disk_frac:.1f}%, Bulge {v_bul_frac:.1f}%",
    "Fit quality (χ²/dof)": f"{chi2_dof:.2f}",
    "V_max observed": f"{v_obs_ngc.max():.1f} km/s at r ≈ {r_ngc[np.argmax(v_obs_ngc)]:.1f} kpc",
    "V_max simulated": f"{v_total_curve.max():.1f} km/s at r ≈ {r_ngc[np.argmax(v_total_curve)]:.1f} kpc",
    f"DM fraction within {r_ref} kpc": f"{dm_frac_r:.1f}%",
    "Residuals": f"Mean = {np.mean(residui):.2f} σ, Max = {np.max(np.abs(residui)):.2f} σ"
})
'''
                

     
            with ui.tab_panel(two).props('role=tabpanel'):
               # with ui.card().classes("p-4 !bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg"):
                title_on_dark("Galaxy Mass ")
                
                with ui.dialog() as info_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Descriptive text about galaxy velocity,mass activities'):
                
                    info_box(
    "You are an astrophysicist investigating the **presence of dark matter** in a galaxy. "
    "\n\n Choose a dataset and complete the missing fields with the correct formulas. "
    "\n\n Click **Run Analysis** to compare the baryonic mass with the total mass and the observed velocity with the Keplerian-like velocity.")
                    aria_button("Close", "close the box",on_click=info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as data_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Dataset description and references'):
                    info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                    aria_button("Close", "close the box",on_click=info_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
               
                    reference_box(
    """**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.""").classes('text-base italic')
                
                with ui.dialog() as baryonic_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                    ui.html(r"""
<h3 class="text-xl font-semibold mb-2">Note</h3>

<p><b>Step 1:</b> Compute the baryonic velocity as the sum of each component from data:  
<span class="math">\( v_{\mathrm{bar}}^2 = v_{\mathrm{gas}}^2 + v_{\mathrm{disk}}^2 + v_{\mathrm{bulge}}^2 \)</span></p>

<p><b>Step 2:</b> Derive the baryonic mass:  
<span class="math">\( M_{\mathrm{bar}}(r) = \frac{r \, v_{\mathrm{bar}}^2}{G} \)</span></p>

<p><b>Step 3:</b> Compute the total mass from observations:  
<span class="math">\( M(r) = \frac{r \, v_{\mathrm{obs}}^2}{G} \)</span></p>

<p><b>Step 5:</b> <b>Plot:</b>  
<ul>
<li>X-axis: radius (data)</li>
<li>Y-axis: <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue)</li>
<li>Also plot masses: <span class="math">\( M_{\mathrm{bar}} \)</span> (red), <span class="math">\( M_{\mathrm{tot}} \)</span> (blue)</li>
</ul>
</p>
""").props('role=dialog aria-modal=true aria-label=Mathematical explanation of the steps to compute baryonic velocity and mass from data')
                    aria_button("Close", "close the box",on_click=baryonic_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"')  as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl !bg-gray-900 rounded-xl shadow-xl"):
                    with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg !bg-gray-800 border border-blue-400 shadow-lg'):
                        ui.markdown("**Legend of Symbols and Units**").classes('!text-xl !text-blue-300 !-mt-2').props('id="legend-title"')
                        ui.markdown("""
- `Rad` = Radius [kpc]  
- `V_obs` = Observed rotation velocity [km/s]  
- `V_gas, V_disk, V_bul` = velocity contributions from gas, stellar disk, bulge [km/s]
- `G` = 4.30091×10⁻⁶ kpc·(km/s)²·M☉⁻¹  
- `M_baryonic` = Baryonic mass [M☉]
- `M_total` = Total mass (from observations)  [M☉]
- `V_baryonic` = Baryonic velocity (Keplerian-like prediction) [km/s]
""").classes('text-lg').props('id="legend-desc"')
                    aria_button("Close", "close the box",on_click=legend_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

            
                with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog,ui.card().classes("p-6 w-full max-w-2xl !bg-gray-900 rounded-xl shadow-xl"):
                    with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg !bg-slate-500 border border-green-500 shadow-lg'):
                        ui.markdown("**Units conversion**").classes('!text-xl !text-green-300 !-mt-2').props('id="units-title"')
                        ui.markdown("""
- `1 kpc` = 3.086 × 10¹⁶ m = 3.26 × 10³ light-years  
- `1 pc` = 3.086 × 10¹³ km = 3.26 light-years  
- `1 Mpc` = 10⁶ pc = 3.086 × 10¹⁹ km  
- `1 km/s` = 3.6 × 10³ km/h = 10³ m/s  
- `1 M☉` = 1.989 × 10³⁰ kg (solar mass)  
- `1 L☉` = 3.828 × 10²⁶ W (solar luminosity)  
""").classes('text-lg').props('id="units-desc"')
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


              

                
                        
                 
                answer_vb, answer_vo,answer_rad,answer_g,answer_Mtot,answer_Mbar,answer_vgas,answer_vdisk,answer_vbulge = {}, {},{},{},{},{},{},{},{}
                @ui.refreshable
                def show_galaxy_pseudocode():
                    with ui.card().classes("p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                        ui.markdown("Galaxy Mass Exercise:fill the missing parts in the pseudo-code:").classes("text-2xl font-bold text-blue-300")
                        with ui.row().classes('w-full gap-4 no-wrap'):
                            with ui.column().classes('pseudo-code flex-1'):
                                ui.label("1) Load galaxy dataset").classes('step')
                                ui.label(f'data = load_data("{galaxy_select.value }")')

                                ui.label("\n2) Compute the baryonic velocity as the sum of velocity contributions from data").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("V_baryonic = sqrt( ")
                                    answer_vgas['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("^2 + ")
                                    answer_vdisk['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("^2 + ")
                                    answer_vbulge['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("^2 )")
                                
                                ui.label("\n3) Compute luminous mass from baryonic velocity").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_baryonic = ( ")
                                    answer_vb['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("^2 * ")
                                    answer_rad['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(" ) / ")
                                    answer_g['el'] = aria_formula_input().props('dense filled').classes('w-48')


                            with ui.column().classes('pseudo-code flex-1'):
                                ui.label("\n4) Compute total mass from observations (data)").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_total    = ( ")
                                    answer_vo['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("^2 * ")
                                    answer_rad['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(" ) / ")
                                    answer_g['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("\n5) Compute the dark matter mass").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_DM    =  ")
                                    answer_Mtot['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("-")
                                    answer_Mbar['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                ui.label("\n6) Plot the results").classes('step mt-2')
                                ui.label('plot(Rad, M_baryonic, label="Baryonic Mass")')
                                ui.label('plot(Rad, M_total,    label="Total Mass")')
                                ui.label('plot(Rad, V_baryonic, label="Baryonic Velocity")')
                                ui.label('plot(Rad, V_obs,    label=" Velocity (Observations)")')

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
                                            plt.xlabel('Radius (kpc)'); plt.ylabel('Velocity (km/s)'); plt.title(f'{os.path.splitext(selected_file)[0]} Rotation Curve')
                                            plt.ylim(0, max(np.nanmax(v_obs_ngc+v_err_ngc), np.nanmax(v_baryonic))*1.1); plt.xlim(0, np.nanmax(r_ngc)*1.1); plt.grid(True); plt.legend()
                                            ui.element('div').props(
'role=img tabindex=0 aria-label=Rotational velocity curve of the selected galaxy (from data) compared with the baryonic velocity'
)

                                    with ui.column().classes('flex-1 items-center'):
                                        with ui.pyplot(figsize=(8, 6)):
                                        
                                            plt.plot(r_ngc, m_baryonic/1e9, color='red', lw=2, label='Baryonic Mass')
                                            plt.plot(r_ngc, m_total/1e9,    color='blue', lw=2, label='Total Mass')
                                            plt.xlabel('Radius (kpc)'); plt.ylabel(' Mass (10^9 M☉)'); plt.title(f'{os.path.splitext(selected_file)[0]} Enclosed Mass')
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
                            plt.title("Rotation Curve")
                            plt.grid(True); plt.legend()

                        
                        with ui.pyplot(figsize=(8, 6)):
                            plt.plot(r, mbar/1e9, color='red', lw=2, label='Baryonic Mass')
                            plt.plot(r, mtot/1e9, color='blue', lw=2, label='Total Mass')
                            plt.xlabel("Radius (kpc)"); plt.ylabel("Mass (10^9 M☉)")
                            plt.title("Enclosed Mass")
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

                with ui.row().classes('w-full justify-center '):
                    with ui.column().classes('flex-1 items-center'):
                        aria_button("Run Analysis", "Run the analysis to reproduce the plots",on_click=check_and_run_galaxy).classes("!bg-red-600 hover:!bg-red-800 text-white font-bold py-2 px-4 rounded")
                    with ui.column().classes('flex-1 items-center'):
                
                        aria_button("Open Plots", "Open the two galaxy plots in a popup window",            on_click=lambda: [update_plots_popup(), plots_popup.open()]).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")



            with ui.tab_panel(three).props('role=tabpanel'):
                #with ui.card().classes("p-4 !bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg shadow-lg"):
                title_on_dark("Cluster Velocity Distribution")
                
                with ui.dialog() as instruction_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Descriptive text about galaxy cluster activity'):
             
                    info_box("The blue histogram shows the observed velocities of galaxies in the Coma cluster; this plot is fixed. "
                            "\n\n Use the slider to **add Dark Matter (DM)** to your simulation. "
                            "\n\n With **DM = 0**, the simulated galaxies are gravitationally unbound. "
                            "\n\n As you increase the **DM**, the gravitational pull keeps the galaxies bound and shapes their velocity distribution to match the observed histogram.")
                    aria_button("Close","Close the box", on_click=instruction_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.dialog() as dataset_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto aria-label=Dataset info'):
             
             
                    info_box("**Dataset variables**: objid (galaxy ID),ra (right ascension),dec (declination),modelmag_r (apparent magnitude in r band),modelmagerr_r (magnitude error),extinction_r,redshift (z),zErr (redshift error)")
                                                  
                    reference_box("""**Dataset reference**: [Kaggle:Coma cluster](https://www.kaggle.com/datasets/mertalkan98/coma-cluster) ; [SDSS](https://www.sdss.org/science/data-release-publications)""").classes('text-base italic')
                    aria_button("Close","Close the box", on_click=dataset_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                             
                with ui.row().classes('w-full justify-center '):
                    with ui.dialog() as cluster_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                        ui.html(r"""
        <h3 class="text-xl font-semibold mb-2">Note</h3>

    <p><b>Step 1:</b> Velocity of each galaxy from redshift data:  
    <span class="math">\( v_i = c \cdot z_i \)</span></p>

    <p><b>Step 2:</b> Number of observed galaxies:  
    <span class="math">\( N \)</span></p>

    <p><b>Step 3:</b> Compute the mean velocity:  
    <span class="math">\( \bar{v} = \frac{1}{N} \sum_i v_i \)</span></p>

    <p><b>Step 4:</b> Observed velocity dispersion:  
    <span class="math">\( \sigma_{\mathrm{obs}} = \sqrt{ \frac{1}{N} \sum_i (v_i - \bar{v})^2 } \)</span></p>

    <p><b>Step 5:</b> Comoving distance:  
    <span class="math">\( \chi(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{\sqrt{\Omega_m (1+z')^3 + (1-\Omega_m)}} \)</span></p>

    <p><b>Step 6:</b> Angular diameter distance:  
    <span class="math">\( D_A(z) = \frac{\chi(z)}{1+z} \)</span></p>

    <p><b>Step 7:</b> Define cluster center from BCG (brightest galaxy) or from median RA/DEC:  
    <span class="math">\( (\mathrm{center\_ra}, \mathrm{center\_dec}) = (ra[idx_{bcg}], dec[idx_{bcg}]) \;\; \mathrm{or} \;\; (\mathrm{median}(ra), \mathrm{median}(dec)) \)</span></p>

    <p><b>Step 8:</b> Angular separation between galaxies:  
    <span class="math">\( \theta = \arccos(\sin \delta_1 \sin \delta_2 + \cos \delta_1 \cos \delta_2 \cos(\alpha_1 - \alpha_2)) \)</span></p>

    <p><b>Step 9:</b> Critical density:  
    <span class="math">\( \rho_{\mathrm{crit}} = \frac{3 H_0^2}{8 \pi G} \)</span></p>

    <p><b>Step 10:</b> Projected radius:  
    <span class="math">\( r_{\mathrm{proj},i} = \max(\theta_i \cdot D_A) \)</span></p>

    <p><b>Step 11:</b> Virial theorem and <span class="math">\( M_{200} \)</span>:  
    <span class="math">\( \sigma_{\mathrm{obs}}^2 = \frac{G M_{200}}{3 R_{200}}, \;\; M_{200} = \frac{4}{3} \pi 200 \rho_{\mathrm{crit}} R_{200}^3 \)</span></p>

    <p><b>Step 12:</b> Concentration parameter:  
    <span class="math">\( c = A \left(\frac{M_{200}}{M_{\mathrm{pivot}}}\right)^B (1+z)^C \)</span></p>

    <p><b>Step 13:</b> NFW density factor:  
    <span class="math">\( \delta_c(c) = \frac{200}{3} \frac{c^3}{\ln(1+c) - c/(1+c)} \)</span></p>

    <p><b>Step 14:</b> Characteristic radius and density:  
    <span class="math">\( r_{200} = \left(\frac{3M_{200}}{4 \pi 200 \rho_{\mathrm{crit}}}\right)^{1/3}, \;\; r_s = \frac{r_{200}}{c}, \;\; \rho_s = \delta_c(c)\,\rho_{\mathrm{crit}} \)</span></p>

    <p><b>Step 15:</b> Dark matter mass (NFW):  
    <span class="math">\( M_{\mathrm{NFW}}(r_{\mathrm{proj},i}) = M_{200} \cdot \frac{\ln(1+x_i) - x_i/(1+x_i)}{\ln(1+c) - c/(1+c)}, \;\; x_i = \frac{r_{\mathrm{proj},i}}{r_s} \)</span></p>

    <p><b>Step 16:</b> Luminosity distance and distance modulus:  
    <span class="math">\( D_L \approx \frac{c \cdot z_{\mathrm{cluster}}}{H_0}, \;\; \mathrm{distmod} = 5 \log_{10}(D_L) - 5 \)</span></p>

    <p><b>Step 17:</b> Magnitude and luminosity:  
    <span class="math">\( M_r = m_r - A_r - (5 \log_{10}(D_L/10\,pc)), \;\; L_r = 10^{0.4(M_{r,\odot} - M_r)} \)</span></p>

    <p><b>Step 18:</b> Stellar/baryonic mass:  
    <span class="math">\( M_{\mathrm{bar}} = (M/L)\,L_r, \;\; (M/L = 2) \)</span></p>

    <p><b>Step 19:</b> Total mass (dark matter + baryonic, linked to slider):  
    <span class="math">\( M_{\mathrm{tot}}(i) = M_{\mathrm{bar}}(i) + f \cdot M_{\mathrm{NFW}}(r_{\mathrm{proj},i}) \)</span></p>

    <p><b>Step 20:</b> Velocity dispersions:  
    <ul>
    <li>Baryonic: <span class="math">\( \sigma_{\mathrm{bar}}(i) = \sqrt{ \frac{G M_{\mathrm{bar}}(i)}{3 r_{\mathrm{proj},i}} } \)</span></li>
    <li>Total (simulated): <span class="math">\( \sigma_{\mathrm{sim}}(i) = \sqrt{ \frac{G M_{\mathrm{tot}}(i)}{3 r_{\mathrm{proj},i}} } \)</span></li>
    </ul>
    </p>

    <p><b>Step 21:</b> Plot histograms:  
    <ul>
    <li>Observed histogram: <span class="math">\( \mathrm{plt.hist}(v_{\mathrm{obs}}, bins) \)</span> (blue)</li>
    <li>Simulated histogram: <span class="math">\( \mathrm{plt.hist}(\sigma_{\mathrm{tot}}, bins) \)</span> (green)</li>
    <li>X-axis: velocity, Y-axis: number of galaxies</li>
    </ul>
    </p>

            """).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the steps to compute cluster properties from data')
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
                    with ui.dialog() as image_dialog, ui.card().classes('p-4 w-full max-w-[600px]'):
                        ui.html("<h5>Cluster Image</h5>")
                        update_cluster_image()
                        aria_button("close",'close',on_click=image_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                    
                    with ui.dialog() as table_dialog, ui.card().classes('p-4 w-full max-w-[800px] overflow-x-auto'):
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
                            plot_container_histo = ui.column()
                          
   
                        with ui.column().classes('flex-1 items-center'):
                            plot_container_scatter = ui.column()
                           
                
                  

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
                            
                                plt.hist(observed_vel, bins=bins, alpha=1.0,color='blue', 
             label=f'Observed Velocities: v_mean={v_mean_val:.1f} km/s' + 
                   f' σ_obs = {sigma_obs:.1f} km/s' + 
                   f' N_gal={N_obs}', 
             rasterized=True)
    
                                if len(v_dm) > 0:
                                    # Correzione: usare '+' per concatenare stringhe
                                    plt.hist(v_dm, bins=bins, alpha=0.7, color='green', 
                                            label=f'Simulated Velocities: v_mean={v_model_mean:.1f} km/s' + 
                                                f' σ_sim = {v_model_sigma:.1f} km/s' + 
                                                f' N_sim_gal={len(v_dm)}' + 
                                                f' Chi2={chi2_hist:.1f}' + 
                                                f' Chi2/Nobs={chi2_norm:.2f}', 
                                            rasterized=True)
                                
                                if len(v_bar) > 0:
                                    # Correzione: usare '+' per concatenare stringhe
                                    plt.hist(v_bar, bins=bins, alpha=0.6, color='red', 
                                            label=f'Baryonic Component: σ_bar = {sigma_bar_val:.1f} km/s', 
                                            rasterized=True)
                                plt.xlim(0, x_max + padding)
                                #plt.yscale('log')
                                plt.ylim(0, y_max)
                                plt.xlabel('Velocity [km/s]',fontsize=12)
                                plt.ylabel('N° of Galaxies',fontsize=12)
                                #if cluster_name is not None:
                                #    plt.title(f'Galaxy Velocity Distribution ({cluster_name})')
                                #else:
                                #    plt.title('Galaxy Velocity Distribution (Cluster)')
                                plt.title('Galaxy Velocity Distribution (Cluster)',fontsize=14,fontweight='bold')
                                plt.legend(fontsize=8, loc='upper center')
                                plt.tight_layout()
                                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                                ui.element('div').props(
'role=status aria-live=polite tabindex=0 aria-label=Histogram showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
)

                      

                            
                            


                
                    

                        with plot_container_scatter:
                            plot_container_scatter.clear()
                            #plt.close()
                            with ui.pyplot(figsize=(8, 4)):
                        
                            

                                plt.scatter(r_proj_kpc, observed_vel, s=10, color='blue', alpha=0.6, label=f"Observed Galaxies: Chi2={chi2_scatter:.1f}"+ 
                      f" Chi2/Ngal={chi2_scatter_norm:.2f}", rasterized=True)

                                plt.scatter(r_proj_kpc, v_bar, s=10, color='red', alpha=0.6, 
                label=f'Baryonic Component: σ_bar ={sigma_bar_val:.1f} km/s' + 
                      f' N_gal_bar={N_bar}', 
                rasterized=True)
    

                                plt.scatter(r_proj_kpc, v_dm, s=10, color='green', alpha=0.6, 
                label=f"Simulated Galaxies: σ_total= {sigma_dm_val:.1f} km/s" + 
                      f" DM_mean = {M_dm_mean:.2e} M☉" + 
                      f" N_gal_sim={N_sim}" + 
                      f" Mass ratio={mass_ratio:.4e}" , 
                rasterized=True)

                            #r_sorted = np.sort(r_proj_kpc)
                            #sigma_los_sorted = np.sqrt(G_grav * (m_bary_at_gal + f * M_dm_at_gal) / (3.0 * r_sorted))
                            #v_upper = v_mean_val + 3 * sigma_los_sorted
                            #v_lower = v_mean_val - 3 * sigma_los_sorted
                            #plt.plot(r_sorted, v_upper, 'r--', lw=1.5, label='Caustic envelope')
                            #plt.plot(r_sorted, v_lower, 'r--', lw=1.5)


                                plt.xlim(0, r_max + r_padding)
                                plt.ylim(0, ylim_max)


                                plt.xlabel("Radius [kpc]",fontsize=12)
                                plt.ylabel("Velocity [km/s]",fontsize=12)
                                #if cluster_name is not None:
                                #    plt.title(f"Phase-space diagram of cluster galaxies ({cluster_name})")
                                #else:
                                #    plt.title("Phase-space diagram of cluster galaxies")
                                plt.title("Scatter plot of cluster galaxies velocities",fontsize=14,fontweight='bold')
                                plt.legend(fontsize=8, loc='upper center')
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
                                plt.title("Observed Galaxies")
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
                                plt.title("Simulated Galaxies")
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
                with ui.dialog() as inst_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label=Descriptive text about galaxy cluster mass and density activities'):
               
                    info_box(
                   "Imagine analyzing a **galaxy cluster** to reveal dark matter. "
"\n\n Choose a dataset and complete the missing fields."
"\n\n Click **Run Analysis** to compare **luminous mass** vs **total (virial) mass**.")
                    aria_button("Close", "Close the box",on_click=inst_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                with ui.dialog() as datac_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto').props('aria-label=Info about cluster dataset'):
                    info_box("**Dataset variables**: Cluster (name), ID (galaxy ID), RAdeg (right ascension in degrees), DEdeg (declination in degrees), RV (radial velocity in km/s), e_RV (velocity error in km/s), q_RV (quality flag),Nint (n.emission/absorbed lines), bmag (apparent magnitude in B band).")
                                        
                    reference_box("""**Dataset reference**: Way M.J. et al., *Redshifts in the Southern Abell Redshift Survey Clusters*. I. The Data'""").classes('text-base italic')
                    aria_button("Close", "Close the box",on_click=datac_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
                
                with ui.dialog() as formula_dialog, ui.card().classes('p-4 w-full max-w-[1200px] overflow-x-auto'):
                            
                    ui.html(r""" <h3 class="text-xl font-semibold mb-2">Note</h3>

<p><b>Step 1:</b> Compute the angular separation from right ascension and declination (converted to radians):  
<span class="math">\( \Delta \theta \)</span></p>

<p><b>Step 2:</b> Compute the radius from angular separation, assuming a cluster distance <span class="math">\( D = 100 \,\mathrm{Mpc} \)</span>:  
<span class="math">\( r = D \cdot \Delta \theta \)</span> (converted to kpc)</p>

<p><b>Step 3:</b> Compute observed velocity from redshift data for each galaxy:  
<span class="math">\( v_i = c \, z_i \)</span></p>

<p><b>Step 4:</b> Convert distance into parsecs and compute distance modulus:  
<span class="math">\( \text{distmod} = 5 \log_{10}(D_{\mathrm{pc}}) - 5 \)</span></p>

<p><b>Step 5:</b> Compute the absolute magnitude using apparent magnitude <span class="math">\( b_{\mathrm{mag}} \)</span>:  
<span class="math">\( M_{\mathrm{abs}} = b_{\mathrm{mag}} - \text{distmod} \)</span></p>

<p><b>Step 6:</b> Derive luminosity of each galaxy from absolute magnitude (<span class="math">\( M_{\odot,B} = 5.48 \)</span>):  
<span class="math">\( L_B = 10^{-0.4 (M_{\mathrm{abs}} - M_{\odot,B})} \)</span></p>

<p><b>Step 7:</b> Compute the total luminosity of the cluster:  
<span class="math">\( L_r = \sum_i L_{B,i} \)</span></p>

<p><b>Step 8:</b> Compute luminous/baryonic mass from luminosity (<span class="math">\( M/L = 5 \)</span>):  
<span class="math">\( M_{\mathrm{lum}} = (M/L) \cdot L_r \)</span></p>

<h4 class="text-lg font-semibold mt-4">Virial Theorem Setup</h4>

<p><b>Step 9:</b> Gravitational potential energy:  
<span class="math">\( U(r) = - \frac{G M m}{r} \)</span></p>

<p><b>Step 10:</b> Gravitational force:  
<span class="math">\( F_{\mathrm{grav}} = \frac{G M m}{r^2} \)</span></p>

<p><b>Step 11:</b> Centripetal force:  
<span class="math">\( F_{\mathrm{cent}} = \frac{m v^2}{r} \)</span></p>

<p><b>Step 12:</b> Equating forces <span class="math">\( F_{\mathrm{grav}} = F_{\mathrm{cent}} \)</span>:  
<span class="math">\( v^2(r) = \frac{G M}{r} \)</span></p>

<p><b>Step 13:</b> Kinetic energy:  
<span class="math">\( K = \tfrac{1}{2} m v^2 = \tfrac{1}{2} \frac{G M m}{r} \)</span></p>

<p><b>Step 14:</b> Relation between kinetic and potential energy:  
<span class="math">\( K = -\tfrac{1}{2} U \)</span></p>

<p><b>Step 15:</b> Virial theorem:  
<span class="math">\( 2  K  +  U  = 0 \)</span></p>

<p><b>Step 16:</b> In a cluster with <span class="math">\( N \)</span> galaxies, the system is treated as <span class="math">\( N \)</span> light particles orbiting a heavy center of mass.  
Total potential energy = sum of all pairwise interactions.  
Total kinetic energy = sum of all galaxy kinetic energies.</p>

<h4 class="text-lg font-semibold mt-4">Cluster Mass & Density</h4>

<p><b>Step 17:</b> Compute the 1-D velocity dispersion from observations:  
<span class="math">\( \sigma_v = \sqrt{\tfrac{1}{N} \sum_{i=0}^N (v_i - \bar{v})^2} \)</span></p>

<p><b>Step 18:</b> For an isotropic, stable system, the 3-D velocity dispersion is:  
<span class="math">\( \sigma_{3D} = \sqrt{3} \, \sigma_v \)</span></p>

<p><b>Step 19:</b> Compute total cluster mass:  
<span class="math">\( M_{\mathrm{tot}} = \frac{3 \, \sigma_v^2 \, r}{G} \)</span></p>

<p><b>Step 20:</b> Compute the cluster volume:  
<span class="math">\( V = \tfrac{4}{3} \pi r^3 \)</span></p>

<p><b>Step 21:</b> Compute luminous density:  
<span class="math">\( \rho_{\mathrm{lum}} = \frac{M_{\mathrm{lum}}}{V} \)</span></p>

<p><b>Step 22:</b> Compute total density:  
<span class="math">\( \rho_{\mathrm{tot}} = \frac{M_{\mathrm{tot}}}{V} \)</span></p>

<p><b>Step 23:</b> <b>Plot:</b>  
<ul>
<li>X-axis: radius</li>
<li>Y-axis: <span class="math">\( M_{\mathrm{lum}} \)</span> (red), <span class="math">\( M_{\mathrm{tot}} \)</span> (blue), <span class="math">\( \rho_{\mathrm{lum}} \)</span> (red), <span class="math">\( \rho_{\mathrm{tot}} \)</span> (blue)</li>
</ul>
</p>
""").props('role=dialog aria-modal=true aria-label=Mathematical explanation of the formulas used in the cluster mass and virial theorem activity')
                    aria_button("Close", "Close the box",on_click=formula_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

                with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"') as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl !bg-gray-900 rounded-xl shadow-xl"):
                    with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg !bg-gray-800 border border-blue-400 shadow-lg'):
            
                        ui.markdown("**Legend of Symbols and Units**").classes('!text-xl !text-blue-300 !-mt-2').props('id="legend-title"')
                        ui.markdown("""
- `RV` = Radial velocity of cluster galaxies [km/s] (dataset)
- `sigma_v` = Velocity dispersion of cluster galaxies[km/s]
- `N` = Number of galaxies in the cluster 
- `D` = Distance to the cluster [Mpc] (fixed or from redshift D=z*c/H0)
- `z` = redshift 
- `r` = Cluster radius [kpc] (computed from angular separation in dataset and distance) 
- `M_total(r)` = Total mass (Virial theorem) [M☉]
- `M_lum(r)` = Luminous mass (B-band luminosity × M/L) [M☉]
- `M/L` = 5 (Mass-to-light ratio in B-band) [M☉/L☉]
- `L(r)` = Cumulative B-band luminosity within radius r [L☉]
- `M_sun_B` = 5.48 (absolute magnitude of the Sun in B-band)  
- `H0` = 70 km/s/Mpc (Hubble constant)
- `G` = 4.302×10⁻⁶ kpc·(km/s)²·M☉⁻¹  
- `c' = 299792 km/s (Speed of light)
- `U` = Gravitational potential energy of a particle [km²/s²]  
- `F_grav` = Gravitational force [M☉·kpc/s²]  
- `F_cent` = Centripetal force [M☉·kpc/s²]  
- `v^2(R)` = Squared orbital velocity for circular orbit [km²/s²]  
- `K` = Kinetic energy of a particle [km²/s²]  
- `2K + U = 0` = Virial theorem for bound system  
- `M_circ` = Mass corresponding to circular orbit of one particle [M☉]  
- `M_tot(r)` = Total virial mass of the cluster [M☉]  
- `L_B` = Luminosity of a single galaxy in B-band [L☉]  
- `L(r)` = Cumulative luminosity up to radius r [L☉]  
- `M_lum(r)` = Luminous mass within radius r [M☉]  
- `D_pc` = Cluster distance in parsecs [pc]  
- `dist_mod` = Distance modulus  
- `M_abs` = Absolute magnitude of a galaxy [mag]  
- `m` = Mass of a light single particle or galaxy [M☉]
- `M` = Mass inside radius r of heavy particle or total enclosed mass [M☉]
- `M_circ` = Mass of a particle in circular orbit [M☉]
- `v^2` = Squared orbital velocity [km²/s²]
- `L_B_i` = Luminosity of a single galaxy [L☉]
- `bmag` = Apparent magnitude in B band [mag] (dataset)



""").classes('text-lg').props('id="legend-desc"')
                    aria_button("Close","Close the box", on_click=legend_dialog.close).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")

            
                with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog, ui.card().classes("p-6 w-full max-w-2xl !bg-gray-900 rounded-xl shadow-xl"):
                    with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg !bg-slate-500 border border-green-500 shadow-lg'):
                        ui.markdown("**Units conversion**").classes('!text-xl !text-green-300 !-mt-2').props('id="units-title"')
                        ui.markdown("""
- `1 kpc` = 3.086 × 10¹⁶ m = 3.26 × 10³ light-years  
- `1 pc` = 3.086 × 10¹³ km = 3.26 light-years  
- `1 Mpc` = 10⁶ pc = 3.086 × 10¹⁹ km  
- `1 km/s` = 3.6 × 10³ km/h = 10³ m/s  
- `1 M☉` = 1.989 × 10³⁰ kg (solar mass)  
- `1 L☉` = 3.828 × 10²⁶ W (solar luminosity)  
- Distance modulus: `m - M = 5 log10(d/10pc)`
- 1 arcsec at distance D → `D * tan(1 arcsec) ≈ D * 1/206265`  """).classes('text-lg').props('id="units-desc"')
            
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
            
             

                

                    answer_sigma,answer_r,answer_g2,answer_vel,answer_vel_mean = {}, {},{},{},{}
                    ans_U, ans_Fg, ans_Fc, ans_K, ans_v2 = {}, {}, {}, {}, {}
                    ans_virialLHS, ans_virialRHS, ans_Mcirc, ans_Mvir,answer_LB = {}, {}, {}, {}, {}
                    answer_D, answer_Dmod, answer_Mabs, answer_Lsum,answer_Mlum = {}, {}, {}, {},{}
                    @ui.refreshable
                    def show_cluster_pseudocode():
                        with ui.card().classes("p-6 w-full !bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown("Cluster Mass Exercise:fill in the missing parts in the pseudo-code below").classes("text-xl font-bold text-blue-300 item-center").props('aria-label=cluster exercises fill the formulas in the boxes')
                          
                            with ui.row().classes('w-full gap-2 no-wrap justify-between'):
                                with ui.column().classes('pseudo-code flex-1'):

                                    ui.label("0) Load cluster dataset").classes('step text-sm')
                                    ui.label(f'data = load_data("{cluster_select.value}")').classes('text-sm')

                                    ui.label("1) Orbital dynamics (single light particle around heavy one)").classes('step text-sm')
                                    ui.label("2) Compute the gravitational potential energy").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("U(r) = ")
                                        ans_U['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("3) Compute the gravitational force").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("F_grav = ").classes('text-sm')
                                        ans_Fg['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                    ui.label("4) Compute the centripetal force").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("F_cent = ").classes('text-sm')
                                        ans_Fc['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("5) Equate forces: F_grav = F_cent → derive v²(R)").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("v²(R) = ").classes('text-sm')
                                        ans_v2['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                    ui.label("6) Kinetic energy of the light particle").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("K = ").classes('text-sm')
                                        ans_K['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("7) Virial theorem (time-averaged, bound system)").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("2⟨").classes('text-sm')
                                        ans_virialLHS['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label("⟩ + ⟨")
                                        ans_virialRHS['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label("⟩ = 0").classes('text-sm')

                                    ui.label("8) Compute circular orbit mass of one particle").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("M_circ(r) = ").classes('text-sm')
                                        ans_Mcirc['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("9) Compute the total mass for a system of N particles:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("M_tot(r) ≈ sum ").classes('text-sm')
                                        ans_Mvir['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                    ui.label("10) Compute velocity dispersion").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("σ_v² = 1/N * Σ |").classes('text-sm')
                                        answer_vel['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(" - mean(").classes('text-sm')
                                        answer_vel_mean['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(")|²").classes('text-sm')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("11) Compute virial/total mass").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("M_total(r) = (3 * ").classes('text-sm')
                                        answer_sigma['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label("² * ").classes('text-sm')
                                        answer_r['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(") / ").classes('text-sm')
                                        answer_g2['el'] = aria_formula_input().props('dense filled').classes('w-12')

                                    ui.label("12) Compute cluster distance in pc:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("D_pc = ").classes('text-sm')
                                        answer_D['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("13) Compute distance modulus:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("dist_mod = 5 * log10(").classes('text-sm')
                                        answer_Dmod['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(") - 5").classes('text-sm')

                                    ui.label("14) Compute absolute magnitude from apparent magnitude (dataset) and distance:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("M_abs = ").classes('text-sm')
                                        answer_Mabs['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(" - dist_mod").classes('text-sm')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("15) Compute luminosity for each galaxy:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("L_B = 10 ** (-0.4 * (").classes('text-sm')
                                        answer_LB['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(" - M_sun_B))").classes('text-sm')

                                    ui.label("16) Compute total luminosity:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("L(r) = Σ(").classes('text-sm')
                                        answer_Lsum['el'] = aria_formula_input().props('dense filled').classes('w-12')
                                        ui.label(") for galaxies with r_i ≤ r").classes('text-sm')
                                with ui.column().classes('pseudo-code flex-1'):
                                    ui.label("17) Compute luminous mass:").classes('step text-sm')
                                    with ui.row().classes('items-center'):
                                        ui.label("M_lum(r) = (M/L) * ").classes('text-sm')
                                        answer_Mlum['el'] = aria_formula_input().props('dense filled').classes('w-12 ')

                                    ui.label("18) Compare mass and density profiles").classes('step text-sm')
                                    ui.label('plot(r, M_lum and M_total, label="Luminous and Total Mass")').classes('text-sm')
                                
                                    ui.label('plot(r, D_lum=M_lum/V and D_total=M_tot/V, label="Luminous and Total Density")').classes('text-sm')
                             

                            
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
                                        plt.xlabel('Radius (kpc)'); plt.ylabel('Mass (M☉)')
                                        plt.title(f'Mass Profile {os.path.splitext(selected_file)[0]}')
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
                                        plt.ylabel('Density (M☉ / kpc³)')
                                        plt.title(f'Density Profile {os.path.splitext(selected_file)[0]}')
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
                                plt.title(f"Mass Profile — {selected_file}")
                                plt.grid(True, which="both", ls="--")
                                plt.legend()

                            with ui.pyplot(figsize=(8, 6)):
                                plt.plot(R_cum, rho_lum_cum, label="Luminous Density", color="red")
                                plt.plot(R_cum, rho_tot_cum, label="Total Density", color="blue")
                                plt.xscale("log"); plt.yscale("log")
                                plt.xlabel("Radius (kpc)"); plt.ylabel("Density (M☉/kpc³)")
                                plt.title(f"Density Profile — {selected_file}")
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


                with ui.row().classes('w-full gap-4 justify-between'):
                   
                    aria_button('Additional Information:Numerical Explanation',"Additional Information:Numerical Explanation", on_click=open_slides_DM).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    aria_button('References',"References", on_click=ref_dialog.open).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
               
                     
                        
                    aria_button('Explore the full Planck CMB Simulator',"Explore the interactive Plank CMB simulator",
          on_click=safe_click(lambda: ui.run_javascript("window.open('https://chrisnorth.github.io/planckapps/Simulator', '_blank')"))).classes("!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-4 rounded")

                        
                    aria_button('ESA: The Universe According to Planck', "ESA reference website",on_click=safe_click(lambda: ui.run_javascript("window.open('https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck','_blank')"))).classes("!bg-blue-600 hover:!bg-blue-800 text-white font-bold py-2 px-4 rounded")
                    
                with ui.row().classes('w-full mx-auto place-items-center justify-between'):
                    
                    aria_image('/images/univ_composition.jpg', "Image of the universe composition with dark energy,dark matter and ordinary matter").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/power_spectrum.jpg', "Image of the power spectrum of cosmic microwave background").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/CMB_asymmetry.jpg', "Image of CMB asymmetry in the universe").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/Planck_freq.jpg', "Image of the plank frequency spectrum").classes('w-64 h-auto rounded-lg shadow-lg border border-gray-300')
        
        
        






   


#inject_layout_tool()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0",
        port=7860,title='Cosmo-Edu Lab', storage_secret='a-very-secret-and-secure-key-for-sessions', dark=True)