
import os
import io
import json
import uuid
import base64
import datetime
import threading
import random
from io import BytesIO

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
from scipy.optimize import minimize_scalar
# Librerie Web/App
import requests
from dotenv import load_dotenv
from fastapi import Request
from groq import Groq
from ipywidgets import interact, FloatSlider
from nicegui import ui, app, run , client
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
import asyncio           
import logging

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

app.add_static_files('/images', os.path.join(BASE_DIR, 'images'))
app.add_static_files('/galaxy_img', os.path.join(BASE_DIR, 'galaxy_img'))
app.add_static_files('/cluster_img', os.path.join(BASE_DIR, 'cluster_img'))


def aria_button(text: str, label: str, **kwargs):
   
    return ui.button(text, **kwargs).props(f'role=button tabindex=0 aria-label={label}')


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

    with ui.header(elevated=True).classes('bg-primary text-white items-center justify-between'):
        ui.label('Cosmo-Edu Lab').classes('text-2xl').props('role=heading aria-level=2 tabindex=0')
        ui.label(title).classes('text-lg')
        with ui.row():
            aria_button('🏠 Home', "Go to the homepage",
    on_click=safe_click(lambda: aria_navigate('/main', 'Navigating to home'))
).classes('mt-4').props('icon=home')

            aria_button('🔙 Back', "Go back to previous page",
    on_click=safe_click(lambda: ui.run_javascript('window.history.back()'))
).props('icon=arrow_back')

            if app.storage.user.get('name'):
                aria_button(f"Logout ({app.storage.user.get('name')})", "Logout and return to login page",
    on_click=safe_click(lambda: (app.storage.user.clear(), aria_navigate('/login', 'Login to the App')))).props('color=negative icon=logout')



    with ui.left_drawer().classes('bg-gray-100 shadow-lg'):
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


        aria_button('📝 Reflections', "Read user reflections",
    on_click=safe_click(lambda: aria_navigate('/reflections', 'Navigating to reflections page.'))
).classes('w-full')


def standard_module_ui(module_name: str):
    """
    Provides standard UI elements for modules, including AI tutor interaction
    and reflection submission.
    """
    with ui.card().classes('w-full mt-4'):
        ui.label('Engage with the AI Tutor').classes('text-lg font-bold').props('role=heading aria-level=2 tabindex=0')
        q_input = aria_input(f'Ask a question about {module_name}...', "Ask a question to the AI tutor about the module").classes('w-full')
        ai_response_container = ui.column().classes('w-full min-h-[50px]')
        async def handle_ask_ai():
            """Funzione asincrona per gestire la chiamata all'AI in modo sicuro."""
            try:
             
                await ask_ai(q_input.value, ai_response_container)
            except Exception as e:
                print(f"[handle_ask_ai] Errore durante l'esecuzione di ask_ai: {e}")
                ai_response_container.clear()
                with ai_response_container:
                    ui.markdown(f"❌ Si è verificato un errore interno: {e}")

        aria_button('Ask Question', "Ask a question to the AI Tutor",
            on_click=handle_ask_ai  
        ).classes('mt-2')

    with ui.card().classes('w-full mt-4'):
        ui.label('Your Reflections').classes('text-lg font-bold').props('role=heading aria-level=2 tabindex=0')
        reflection_input = aria_textarea(f'Write your reflections on {module_name} here...', "Insert your reflection about the module").classes('w-full')
        aria_button('Save Reflection', "Save your reflection",
    on_click=safe_click(lambda: submit_reflection(module_name, reflection_input.value))
).classes('mt-2')



@ui.page('/login')
def login_page():
    ui.run_javascript("setTimeout(() => document.querySelector('h1, .title, .text-2xl')?.focus(), 3600)")
    """Login page for users."""
    def try_login():
        if app_data['users'].get(username.value) == password.value:
            app.storage.user['name'] = username.value
            ui.navigate.to('/main')
        else:
            accessible_notify('Invalid username or password', type='warning')

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
            accessible_notify('Username and password are required.', type='warning')
            return
        if new_user.value in app_data['users']:
            accessible_notify('This username is already taken.', type='warning')
            return
        app_data['users'][new_user.value] = new_pass.value
        save_data()
        accessible_notify('User registered! You can now log in.', type='success')
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
    main_layout("Welcome to Cosmo-Edu")

    with ui.column().classes('w-full items-center p-8'):
        ui.label(f"Welcome, {app.storage.user.get('name', 'Explorer')}!").classes('text-3xl m-4').props('role=heading aria-level=2 tabindex=0')


       
        
        ui.label('\"The cosmos is within us. We are made of star-stuff.\" — Carl Sagan').classes('italic text-lg mt-2').props('role=heading aria-level=2 tabindex=0')

        ui.markdown('Select a module to begin your journey into cosmology.')
        with ui.grid(columns=4).classes('gap-4 m-4 w-full'):
            module_titles = [
                "Redshift & Universe Expansion","Universe History & CMB" ,"Dark Matter"
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
       
         ("1. Redshift & Universe Expansion", "Doppler effect,optics, velocity-distance relationships, waves, frequency,flux, wavelenght,relativistic velocity ","Redshift, standard candles,supernovae Ia, luminosity, Hubble’s law, cosmic acceleration,universe expansion, Dark energy, lambda-CDM model", "/module1"),
        ("2. Universe History & CMB", "Thermodynamics, radiation, blackbody spectrum, adiabatic transformation, adiabatic index, Stephen-Boltzmann law", "Thermal history, CMB, Plank spectrum, universe evolution, radiation and matter density, equivalence,photons", "/module2"),
        
       
         ("3. Dark Matter ", "Circular motion, gravity, energy conservation, centripetal force, gravitational force, Newton, Kepler laws, kinetic energy,gravitational potential energy", "Rotation curves, missing mass, virial theorem, dark matter, universe composition", "/module3")
        
       
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
            ("Scalar and vectorial measures and units", "/module3"),
            ("Geometry optics: reflection and refraction", "/module1"),
            ("Thermal phenomena: temperature, heat, thermal balance", "/module2"),
            ("Mechanics: motion, forces, Newton’s laws", "/module3"),
        ]),
        ("3rd–4th Year", [
            ("Relativity: reference systems, principles", "/module1"),
            ("Conservation laws: energy, fluids", "/module3"),
            ("Gravitation: Kepler, Newton, cosmological systems", "/module3"),
            ("Thermodynamics: gas laws, kinetic theory", "/module2"),
            ("Waves: mechanical, interference, diffraction", "/module1"),
            ("Electromagnetism: fields, energy, potential", "/module1"),
        ]),
        ("5th Year", [
            ("Electromagnetism: induction, Maxwell, EM waves", "/module1"),
            ("Micro & macro cosmos: space-time, energy", "/module1"),
            ("Relativity: Einstein’s principles, dilation, contraction", "/module1"),
            ("Quantum light: Planck, photoelectric, De Broglie", "/module2"),
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

def paragraph(text: str):
    """Paragrafo standard (18px, giustificato, line spacing 1.7)"""
    return ui.markdown(text).classes("paragraph").props(
        'tabindex=0 role=document')

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
        "p-4 bg-slate-50 border border-slate-300 rounded-lg shadow-md w-full max-w-2xl mt-2"
    ).props('role=region aria-label=Plot results summary tabindex=0'):
        with ui.element('h4').classes("text-lg font-bold text-slate-700 mb-2").props('tabindex=0 role=heading aria-level=4'):
            ui.label(title)
        for label, value in info.items():
            ui.label(f"{label}: {value}").classes("text-base font-medium text-slate-800").props('role=heading aria-level=2 tabindex=0')


def title_on_dark(title: str):
    """Titolo accessibile su sfondo scuro"""
    el= ui.element('h2').classes("text-4xl font-bold text-center mb-4 title-on-dark").props(
        'tabindex=0 role=heading aria-level=2'
    )
    with el:
        ui.label(title)
    return el




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
        "info": "bg-blue-50 text-blue-800 border-blue-400",
        "success": "bg-green-50 text-green-800 border-green-400",
        "warning": "bg-yellow-50 text-yellow-800 border-yellow-400",
        "error": "bg-red-50 text-red-800 border-red-400",
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
def module1():
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

  
  .paragraph {
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
    
    
    main_layout("Module 1: Redshift & Universe Expansion")
    ui.label("Cosmological Redshift & Hubble Law").classes("text-lg font-bold mt-2").props('id=tabs_desc role=heading aria-level=2 tabindex=0')
    with ui.tabs().classes('w-full justify-center') as tabs:
        one = ui.tab(' Cosmological Redshift', icon='auto_awesome').props('aria-label="Activity 1: Cosmological Redshift"')
        two = ui.tab('Hubble law', icon='science').props('aria-label="Activity 2: Hubble law"')
        
    spectra_csv_dir = GALAXY_SPECTRA_PATH
    fits_parent_dir =GALAXY_FITS_PATH
  

    #measured_auto = []
    state = {"zs_points": []} 
    def add_redshift_activity(container):

        ui.add_body_html(r"""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  
""")
        
        with container:
            title_on_dark("Cosmological Redshift (SDSS spectra)")
            reference_box("Dataset reference: [SDSS-Galaxy spectra] (https://www.sdss4.org/dr16/ , https://dr17.sdss.org/)")
            info_box("Dataset: wavelength (Å), flux (W·m⁻²·Å⁻¹).")

   
            with ui.row().classes("items-center"):
                paragraph("Explore how galaxy spectra reveal the cosmic redshift.\n\n"
                          "1. Plot flux vs wavelength for a galaxy spectrum.\n\n"
                          "2. Identify emission peaks lambda_obs for each galaxy and compare observed peaks with standard lines (Hα, Hβ, O III, etc.).\n\n"    
                        "3. Verify that galaxy lines are shifted relative to standard lines and compute redshift comparing with galaxy catalog .\n\n"
                           "4. Compute and plot non-relativistic and relativistic velocity vs redshift.\n\n"
                           "5. Observe that for small z the velocities are similar, while for large z relativistic effects become important.")
                def open_info():
                    with ui.dialog() as d, ui.card():
                        ui.html(r"""
<h6><b>Redshift and Flux Relations</b></h6>

<p>
When a galaxy moves away due to the expansion of the Universe, its spectral lines shift toward longer wavelengths (<b>redshift</b>):
</p>

<p style="text-align:center;">
  $$ z = \frac{\lambda_{\rm obs} - \lambda_0}{\lambda_0} $$
</p>

<p>For small redshifts:</p>
<p style="text-align:center;">
  $$ v = c \, z $$
</p>

<p>Relativistically:</p>
<p style="text-align:center;">
  $$ v = c \, \frac{(1+z)^2 - 1}{(1+z)^2 + 1} $$
</p>

<hr>

<h6><b>Flux–Luminosity Relation</b></h6>

<p>
The flux \( F(\lambda) \) is the energy received per unit area, time, and wavelength (W·m⁻²·Å⁻¹).  
For isotropic emission from a source with intrinsic luminosity \( L \) at luminosity distance \( d_L \):
</p>

<p style="text-align:center;">
  $$ F = \frac{L}{4 \pi d_L^2} $$
</p>

<p>Including the effect of redshift:</p>

<p style="text-align:center;">
  $$ F(\lambda_{\rm obs}) = \frac{L(\lambda_{\rm emit})}{4 \pi d_L^2 (1+z)} $$
</p>

<hr>

<h6><b>Relativistic Doppler Effect</b></h6>

<p>
For light, the observed frequency changes if the source and/or observer are moving relative to each other.
</p>

<ul>
<li><b>Source moving away:</b></li>
</ul>

<p style="text-align:center;">
  $$ f_{\rm obs} = f_{\rm emit} \, \sqrt{\frac{1 - v/c}{1 + v/c}} $$
</p>

<ul>
<li><b>Source approaching:</b></li>
</ul>

<p style="text-align:center;">
  $$ f_{\rm obs} = f_{\rm emit} \, \sqrt{\frac{1 + v/c}{1 - v/c}} $$
</p>

<p>
Redshift corresponds to a wavelength increase (\( \lambda_{\rm obs} > \lambda_{\rm emit} \)),  
blueshift corresponds to a wavelength decrease (\( \lambda_{\rm obs} < \lambda_{\rm emit} \)).
</p>

<hr>

<h6><b>Common Spectral Lines (SDSS)</b></h6>

<p>
SDSS provides galaxy spectra (\( \lambda_{\rm obs}, F \)) with key emission and absorption lines:
</p>

<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse; margin:auto; text-align:center;">
  <tr><th>Element / Transition</th><th>Rest Wavelength (Å)</th></tr>
  <tr><td>Hα (Hydrogen Balmer)</td><td>6562.8</td></tr>
  <tr><td>Hβ (Hydrogen Balmer)</td><td>4861.3</td></tr>
  <tr><td>[O III] (Forbidden Oxygen)</td><td>5006.8</td></tr>
  <tr><td>[O II] Doublet</td><td>3727</td></tr>
  <tr><td>Ca II H &amp; K</td><td>3968.5 / 3933.7</td></tr>
  <tr><td>Na I D</td><td>5890 / 5896</td></tr>
</table>

<p style="text-align:center; margin-top:10px;">
NIST Atomic Spectra Database
</p>
""", sanitize=False).props('tabindex=0' 'role=document' 'aria-label="Redshift and Flux Relations information"')


                        aria_button("Close",label="close the box", on_click=lambda: d.close())
                    d.open()
                aria_button("Info", label="information", on_click=open_info)

           

           
            known_lines = {
    "Hα": 6562.8, "Hβ": 4861.3, "[O III]": 5006.8,
    "[O II]": 3727.0, "N II": 6583.0,
    "Ca II K": 3933.7, "Ca II H": 3968.5,
    "Na I D1": 5890.0, "Na I D2": 5896.0
}
            known_to_csv = {
    "Hα": "H_alpha",
    "Hβ": "H_beta",
    "[O III]": "[O_III] 5007",
    "[O II]": "[O_II] 3727",
    "N II": "[N_II] 6583",
    "Ca II K": "Ca_II_K",
    "Ca II H": "Ca_II_H",
    "Na I D1": "Na_I_D1",
    "Na I D2": "Na_I_D2"
}
            def load_observed_lines(file_select_value):
                base_dir = GALAXY_LINE_PATH
                base_name = os.path.splitext(os.path.basename(file_select_value))[0]
                expected_name = f"{base_name}_lines.csv"
                expected_path = os.path.join(base_dir, expected_name)

                if os.path.exists(expected_path):
                    return pd.read_csv(expected_path), expected_path

                prefix = base_name.split("_")[0]
                candidates = [f for f in os.listdir(base_dir)
                            if f.startswith(prefix) and f.endswith("_lines.csv")]

                if candidates:
                    chosen = os.path.join(base_dir, candidates[0])
                    print(f"✅ Trovato file corrispondente: {candidates[0]}")
                    return pd.read_csv(chosen), chosen
                else:
                    raise FileNotFoundError(f"Nessun file *_lines.csv trovato per {base_name}")


            csv_files = sorted([f for f in os.listdir(spectra_csv_dir) if f.lower().endswith(".csv")])
            if not csv_files:
                ui.label("No CSV spectra found in folder: " + spectra_csv_dir).classes("text-red-600")

            with ui.row().classes("items-center gap-4 mt-3"):
                file_select = ui.select(csv_files, label="Choose spectrum CSV").classes("w-80").props('aria-describedby=tabs_desc aria-label=choose spectrum file' )
                plot_btn = aria_button("Plot Spectrum", "plot galaxy spectrum")

            with ui.row().classes("w-full mt-4 gap-6 items-start"):
                
                spec_plot_container = ui.column()

                try:
                    if file_select.value:
                        df_obs, obs_path = load_observed_lines(file_select.value)
                   
                        lambda_obs_dict = dict(zip(df_obs["line"], df_obs["lambda_obs"]))
                    else:
                        lambda_obs_dict = {}
                except FileNotFoundError:
                    accessible_notify("No matching *_lines.csv file found for the selected spectrum.", type_="warning")
                    lambda_obs_dict = {}


                
                student_rows = []
                for line, lambda0 in known_lines.items():
                    lambda_obs_val = lambda_obs_dict.get(line, "")
                    student_rows.append({
                        "line": line,
                        "lambda0": lambda0,
                        "lambda_obs": f"{lambda_obs_val:.2f}" if lambda_obs_val else "",
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
                    label="Student measured spectral lines",
                    row_key="line"
                ).classes("w-full")

              
                with ui.row().classes("mt-2 items-center gap-3"):
                    selected_line = ui.select([r["line"] for r in student_rows], label="Select line").classes("w-44")
                    z_input = aria_input(label="Enter redshift z", aria_label="Enter redshift z", placeholder="e.g. 0.02").classes("w-32")
                    update_z_btn = aria_button("Add z value", "add z redshift value",color="blue-6")
                    check_btn = aria_button("Show λ_obs & z", "check table results")

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
                    result_dialog_card_label = ui.label("Measured Spectral Lines").classes("text-lg font-bold mb-2").props('tabindex=0 role=heading aria-level=2')
    
                    result_table = aria_table(
                        columns=[
                            {"name":"line","label":"Line","field":"line"},
                            {"name":"lambda0","label":"λ₀ [Å]","field":"lambda0"},
                            {"name":"lambda_obs","label":"λ_obs [Å]","field":"lambda_obs"},
                            {"name":"z","label":"z (auto)","field":"z"},
                        ],
                        rows=[],
                        label="Measured spectral lines results table"
                    ).classes("w-full")
                    with ui.row().classes("justify-end mt-3"):
                        aria_button("Close", "close dialog",on_click=result_dialog.close)

              
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

                    try:
                        df_obs, obs_path = load_observed_lines(file_select.value)
                        if "line_name" in df_obs.columns:
                            df_obs = df_obs.rename(columns={"line_name": "line"})
                        measured = []
                        for kname, csv_name in known_to_csv.items():
                            match = df_obs[df_obs["line"].str.strip() == csv_name]
                            if not match.empty:
                                r = match.iloc[0]
                                measured.append({"line": kname, "lambda_obs": r["lambda_obs"]})
                    except FileNotFoundError:
                        measured = []

                    with spec_plot_container:
                        plt.close()
                        with ui.pyplot(figsize=(15, 5)):
                            
                            plt.plot(wl, flux, lw=1, color="blue", label="Flux")
                            for m in measured:
                                λ = m["lambda_obs"]
                                label = m["line"]
                                plt.axvline(λ, color="black", ls="--", alpha=0.6)
                                plt.text(λ, np.nanmax(flux)*0.95, f"{label}\n({λ:.1f} Å)",
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
                                    r = -(nm - 440)/(440-380); g = 0.0; b = 1.0
                                elif 440 <= nm < 490:
                                    r = 0.0; g = (nm-440)/(490-440); b = 1.0
                                elif 490 <= nm < 510:
                                    r = 0.0; g = 1.0; b = -(nm-510)/(510-490)
                                elif 510 <= nm < 580:
                                    r = (nm-510)/(580-510); g = 1.0; b = 0.0
                                elif 580 <= nm < 645:
                                    r = 1.0; g = -(nm-645)/(645-580); b = 0.0
                                else:
                                    r = 1.0; g = 0.0; b = 0.0
                                return (r, g, b)

                            for j, xv in enumerate(xs):
                                grad[0, j, :] = wavelength_to_rgb(xv)

                            plt.imshow(grad, aspect='auto', extent=(xmin, xmax, 0, 1))
                            plt.axis('off')

                plot_btn.on("click", plot_spectrum)

           
                def show_solution():
                    if not file_select.value:
                        accessible_notify("Select a spectrum CSV first.", type_="warning")
                        return

                    try:
                       
                        df_obs, obs_path = load_observed_lines(file_select.value)
                       
                    except FileNotFoundError as e:
                        accessible_notify(str(e), type_="error")
                        return

                    result_dialog_card_label.set_text(f"Measured Spectral Lines for: {file_select.value}")

                  
                    if "line_name" in df_obs.columns:
                        df_obs = df_obs.rename(columns={"line_name": "line"})

                
                    df_obs["line_norm"] = df_obs["line"].str.replace(" ", "").str.lower()
                    csv_to_known = {v.replace(" ", "").lower(): k for k, v in known_to_csv.items()}

                
                    result_rows = []
                    for _, r in df_obs.iterrows():
                        line_norm = r["line_norm"]
                        if line_norm in csv_to_known:
                            known_key = csv_to_known[line_norm]
                            lambda0 = known_lines[known_key]
                            z_val = (r["lambda_obs"] - lambda0) / lambda0
                            result_rows.append({
                                "line": known_key,
                                "lambda0": f"{lambda0:.1f}",
                                "lambda_obs": f"{r['lambda_obs']:.3f}",
                                "z": f"{z_val:.6f}"
                            })

                    result_table.rows = result_rows
                    result_table.update()
                    result_dialog.open()



                check_btn.on("click", show_solution)
                
                
            with ui.row().classes("w-full mt-4 "):
               
              
                info_box("Insert a value of redshift to compute the velocities:\n\n"
                            "**Non relativistic velocity**:v=cz\n\n"
                            "**Relativistic velocity**: v=c((1+z)²−1)/((1+z)²+1)")
                
                z_input = aria_input(label="Enter redshift z", aria_label="Enter redshift z",placeholder="e.g. 0.02").classes("w-48")
                calc_vel_btn = aria_button("Compute velocity", "compute velocity for single z")
            with ui.row().classes("w-full mt-6 "):
                vel_table = aria_table(
                        columns=[
                            {"name":"z","label":"z","field":"z"},
                            {"name":"v_nr","label":"v_non-rel [km/s]","field":"v_nr"},
                            {"name":"v_rel","label":"v_rel [km/s]","field":"v_rel"},
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
                        v_rel = c_km_s * (((1+z)**2 - 1) / ((1+z)**2 + 1))
                        vel_table.rows = [{"z": f"{z:.6f}", "v_nr": f"{v_nr:.1f}", "v_rel": f"{v_rel:.1f}"}]
                        vel_table.update()
                calc_vel_btn.on("click", compute_single_velocity)

               
                plot_vel_btn = aria_button("Plot velocities (dataset)", "plot velocity dataset automatically")
                vel_plot_container = ui.column()
                def plot_velocity_dataset():
                      

                        vel_plot_container.clear()

                        gal_file = os.path.join(DATA_DIR, "optical_search_884989.csv")

                        cols = ["plate", "mjd", "fiberid", "run2d", "specobj_id", "ra", "dec",
                                "sn_median_r", "z", "zerr", "zwarning", "class", "subclass"]
                        df = pd.read_csv(gal_file, comment='#', sep=',', names=cols, engine='python')


                        df["z"] = pd.to_numeric(df["z"], errors="coerce")
                        df["zerr"] = pd.to_numeric(df["zerr"], errors="coerce")
                        df = df.dropna(subset=["z", "zerr"])

                        c_km_s = 299792.458
                        df["v_nr"] = c_km_s * df["z"]
                        df["v_nr_err"] = c_km_s * df["zerr"]
                        df["v_rel"] = c_km_s * (((1 + df["z"])**2 - 1) / ((1 + df["z"])**2 + 1))
                        df["v_rel_err"] = c_km_s * (4 * (1 + df["z"]) * df["zerr"]) / (((1 + df["z"])**2 + 1)**2)

                        with vel_plot_container:
                            plt.close()
                            with ui.pyplot(figsize=(7, 5)):
                                plt.errorbar(df["z"], df["v_nr"], yerr=df["v_nr_err"], fmt='o', capsize=4,
                                            color='tab:blue', label='v_non-rel')
                                plt.errorbar(df["z"], df["v_rel"], yerr=df["v_rel_err"], fmt='o', capsize=4,
                                            color='tab:green', label='v_relativistic')

                                zs_plot = np.linspace(0, max(df["z"].max()*1.2, 0.2), 300)
                                v_nr_curve = c_km_s * zs_plot
                                v_rel_curve = c_km_s * (((1+zs_plot)**2 - 1) / ((1+zs_plot)**2 + 1))
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
    def dL_Mpc_LCDM(z, H0=70.0, Omega_m=0.3, Omega_Lambda=0.7):
      
        #d_L = (1 + z) * (c / H0) * integral_0^z [1 / sqrt(Ω_m(1+z')^3 + Ω_Λ)] dz'
        
        c = 3e5  # km/s
        def integrand(zp):
            return 1.0 / np.sqrt(Omega_m * (1 + zp)**3 + Omega_Lambda)
        integral, _ = quad(integrand, 0.0, float(z))
        return (1.0 + z) * (c / float(H0)) * integral
    def dL_Mpc_noLambda(z, H0=70.0, Omega_m=1.0):
       
        # (ΩΛ = 0)
        #d_L = (1+z) * (c/H0) * ∫_0^z dz' / sqrt(Ω_m(1+z')^3)
       
        c = 3e5  # km/s
        def integrand(zp):
            return 1.0 / np.sqrt(Omega_m * (1 + zp)**3)
        integral, _ = quad(integrand, 0.0, float(z))
        return (1.0 + z) * (c / float(H0)) * integral



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
            reference_box("Dataset reference: [Supernovae data Pantheon-SH0ES] (https://github.com/dscolnic/Pantheon , https://pantheonplussh0es.github.io/)")
            info_box("Dataset: zcmb (CMB frame redshift), mb (apparent B magnitude), err_mag.")

          
            with ui.row().classes("items-center gap-6 mt-3"):
                H0_box = ui.number(label="H₀ [km/s/Mpc]", format="%.3f").classes("w-44").props('aria-label="Hubble constant H0 in km per second per Megaparsec"')
             
                M_B_input = ui.number(label="Absolute Magnitude M_B (SNe Ia)", value=-19.3, format="%.3f").classes("w-52").props('aria-label="Absolute Magnitude of Type Ia Supernovae M sub B"')
                with ui.column().classes("items-start gap-2"):
                    ui.label("Compute H₀ from data (linear fit v = H₀·dₗ)").classes("text-sm text-blue-600")
                    result_label = ui.label("").classes("text-lg font-semibold text-green-600")
                    aria_button("Compute H₀", "compute H0",on_click=lambda: compute_H0()).props("color=primary outline").classes("w-32")

                with ui.expansion('Fundamental Cosmological Relations', icon='school', value=False):
                    ui.html("""
                    <div style="font-size:1rem; line-height:1.5; padding:8px;">
                    <p><b>Hubble Law:</b><br>
                    v = H₀ · d<sub>L</sub><br>
                    For small redshifts (z ≪ 1), v ≈ c·z.</p>

                    <p><b>Distance Modulus:</b><br>
                    μ = m<sub>B</sub> − M<sub>B</sub> = 5·log₁₀(d<sub>L</sub>[pc]) − 5<br>
                    → connects apparent and absolute magnitude.</p>

                    <p><b>Luminosity Distance:</b><br>
                    d<sub>L</sub> = (1+z)·(c/H₀)·∫₀ᶻ [dz'/√(Ωₘ(1+z')³ + Ω_Λ)]<br>
                    → accounts for cosmological expansion (ΛCDM model).</p>

                    <p><b>Scale Factor Relation:</b><br>
                    1 + z = 1 / a(t) → expansion of the Universe.</p>
                    </div>
                    """, sanitize=False).props('aria-label="Fundamental Cosmological Relations"')
                with ui.expansion("ΛCDM Explanation", icon="functions"):
                    ui.html(r"""
                    <ul style="line-height:1.6; font-size:0.95rem;">
                        <li><b>Velocity (non-relativistic):</b> v = c·z</li>
                        <li><b>Relativistic velocity:</b> \(v = c\frac{(1+z)^2 - 1}{(1+z)^2 + 1}\)</li>
                        <li><b>Distance modulus:</b> μ = m<sub>B</sub> − M<sub>B</sub></li>
                        <li><b>Luminosity distance:</b> \(d_L = 10^{(\mu+5)/5}\,[pc]\)</li>
                        <li><b>Hubble constant estimate:</b> \(H_0 = \frac{\sum_i v_i d_{L,i}}{\sum_i d_{L,i}^2}\)</li>
                        <li><b>Residuals:</b> r = v − H₀·d<sub>L</sub></li>
                        <li><b>Scale factor relation:</b> 1 + z = 1 / a(t)</li>
                        <li><b>ΛCDM distance:</b> \(d_L(z) = \frac{c(1+z)}{H_0}\int_0^z \frac{dz'}{E(z')}, 
                        \quad E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}\)</li>
                        <li><b>Typical parameters:</b> Ωₘ = 0.3, Ω_Λ = 0.7, Ωₖ = 0 (flat Universe)</li>
                    </ul>
                    """, sanitize=False).props("aria-live=polite" 'role=document' 'aria-label="Lambda CDM Explanation"')


    


    
                with ui.row().classes("items-center gap-3 mt-4"):
                    info_button = aria_button("Info ▶", "Information").props(
                                            'aria-label="Open detailed information dialog "'
                                        )
                    legend_button = aria_button("Legend ▶", "Variable and constants legend").props(
                                            'aria-label="Open legend dialog "'
                                )
        
            
                with ui.dialog().props(   'aria-modal="true" role="dialog" aria-labelledby="info-title" aria-describedby="info-content"') as info_dlg, ui.card():
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
                """, sanitize=False).props("id=planck-info-content role=document aria-live=polite aria-describedby=info-content aria-label='Detailed information about redshift, flux, Hubble relation, Lambda CDM luminosity distance, and velocities.'")
                    
            
                with ui.dialog() as leg_dlg, ui.card():
                    ui.label("Legend: Symbols and Units").classes("text-lg font-semibold").props("id=info-title")
                    ui.html("""
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
        """,sanitize=False).props("role=document aria-live=polite aria-describedby=info-content aria-label='Legend explaining symbols and units used in the Hubble Law with Type Ia Supernovae activity.'")
                
            
                def open_info(_=None):
                    info_dlg.open()
                    ui.run_javascript("""
                        if (window.MathJax && window.MathJax.typesetPromise) {
                            MathJax.typesetPromise();
                        }
                    """)

                def open_legend(_=None):
                    leg_dlg.open()
                    ui.run_javascript("""
                        if (window.MathJax && window.MathJax.typesetPromise) {
                            MathJax.typesetPromise();
                        }
                    """)

                info_button.on("click", open_info)
                legend_button.on("click", open_legend)


            

         
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
                coeffs = np.polyfit(x, y, 1)
                H0_est = coeffs[0]

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
                    aria_button("Close", "close dialog box",on_click=lambda: dlg.close())
                dlg.open()



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
                        z_hubble = np.linspace(0, df['zcmb'].max(), 300)
                        #z_hubble = np.linspace(1e-4, df['zcmb'].max()*1.05, 300)
                        dL_h = (c / H0v) * z_hubble
                        mu_h = 5 * np.log10(dL_h * 1e6) - 5
                        
                        plt.plot(z_hubble, mu_h + M_B, '--', color="red", label=f'm_B (Hubble law, H₀={H0v:.1f})')

                    if state['show_lcdm']:
                        H0v = get_H0_value()
                        z = np.linspace(0, df['zcmb'].max(), 300)
                        #z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                        dL = np.array([dL_Mpc_LCDM(zv, H0v) for zv in z])
                        mu = 5*np.log10(dL*1e6)-5
                        plt.plot(z, mu+M_B,color="green", label='m_B(ΛCDM)')
                        
                    if state['show_noLambda']:
                        H0v = get_H0_value()
                        #z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                        z = np.linspace(0, df['zcmb'].max(), 300)
                        dL_noL = np.array([dL_Mpc_noLambda(zv, H0v) for zv in z])
                        mu_noL = 5 * np.log10(dL_noL * 1e6) - 5
                        plt.plot(z, mu_noL + M_B, color="orange", lw=2,
                                label='m_B (ΩΛ=0, matter-only)')

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
                        z_hubble = np.linspace(0, df['zcmb'].max(), 300)
                        #z_hubble = np.linspace(1e-4, df['zcmb'].max()*1.05, 300)
                        dL_h = (c / H0v) * z_hubble
                        mu_h = 5 * np.log10(dL_h * 1e6) - 5
                        plt.plot(z_hubble, mu_h, '--', color='red', lw=2, label=f'μ (Hubble law, H₀={H0v:.1f})')

                    if state['show_lcdm']:
                        H0v = get_H0_value()
                        z = np.linspace(0, df['zcmb'].max(), 300)
                        #z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                        dL = np.array([dL_Mpc_LCDM(zv, H0v) for zv in z])
                        mu = 5*np.log10(dL*1e6)-5
                        plt.plot(z, mu,color="green", label='μ(ΛCDM)')
                        
                    if state['show_noLambda']:
                        H0v = get_H0_value()
                        z = np.linspace(0, df['zcmb'].max(), 300)
                        #z = np.linspace(1e-4, df['zcmb'].max()*1.05, 250)
                        dL_noL = np.array([dL_Mpc_noLambda(zv, H0v) for zv in z])
                        mu_noL = 5 * np.log10(dL_noL * 1e6) - 5
                        plt.plot(z, mu_noL, color="orange", lw=2,
                                label='μ (ΩΛ=0, matter-only)')

                    plt.xlabel("z"); plt.ylabel("μ [mag]")
                    plt.title("μ vs z")
                    #plt.xscale('log'); plt.yscale('log')
                    plt.legend(); plt.tight_layout()
                    aria_chart_label("Plot of distance modulus μ versus redshift z")

            plot_all()

       
            with ui.row().classes("gap-3 mt-2"):
                aria_button("Add v_rel","add relativistic velocity" ,on_click=lambda: (state.update(show_vrel=True), plot_all()))
                aria_button("Add Hubble line", "add Hubble line",on_click=lambda: (state.update(show_hubble_line=True), plot_all()))
                aria_button("Add Hubble mag","add Hubble magnitude", on_click=lambda: (state.update(show_mag_hubble=True), plot_all()))
                aria_button("Add CDM", "add only matter model",on_click=lambda: (state.update(show_noLambda=True), plot_all()))
                aria_button("Add ΛCDM", "add cosmological model",on_click=lambda: (state.update(show_lcdm=True), plot_all()))

         
            info_box(" Exercise Table: Fill in values and compute missing columns ")
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
                    aria_button("Close", "close dialog",on_click=lambda: dlg.close())
                dlg.open()
                
            with ui.row().classes("gap-3"):
                aria_button("Add / Update row","add row in table", on_click=update_row).classes("mt-2")
                aria_button("Show results","show table with results", on_click=show_results).classes("mt-3")
            

        
    with ui.tab_panels(tabs, value=one):
        with ui.tab_panel(one) as redshift_panel:
            add_redshift_activity(redshift_panel)

    with ui.tab_panels(tabs, value=two):
        with ui.tab_panel(two) as hubble_panel:
            add_hubble_activity(hubble_panel)
  
    standard_module_ui("Redshift & Universe Expansion")

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

  
  .paragraph {
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



    main_layout("Module 2: Universe History & CMB")
    ui.label("Select a section to explore the thermodynamics activities").props('id=tabs_desc role=heading aria-level=2 tabindex=0')
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
                

                
                reference_box("Dataset reference: Fixsen et al. (1996), The Astrophysical Journal. "
                              "\n[COBE-FIRAS monopole spetrum] (https://lambda.gsfc.nasa.gov/product/cobe/firas_monopole_spect.html)")
                info_box("Dataset:frequency(cm^-1) monopole_spectrum(MJy/sr) uncertainty(kJy/sr)")
                with ui.row().classes("w-full items-start justify-between gap-8 mt-4").style("flex-wrap: nowrap; align-items: flex-start;"):


                    with ui.column().classes("w-1/2 min-w-[480px] max-w-[640px] flex-shrink-0"):

                    

                     
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
                
                        draw_plot()
                        lambda_peak_box = ui.label("").classes("text-md font-semibold text-blue-400 mt-3")


                    
                        if cobedata_loaded:
                            accessible_notify("✅ COBE/FIRAS dataset successfully loaded (Fixsen et al. 1996).",type_='success')
                        else:
                            accessible_notify(f"⚠️ Could not load COBE data: {coberead_error}",type_='warning')

                     
                        ui.separator()
                        
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
                            aria_button("Calculate", "Compute temperature", on_click=compute_T_from_lambda)
                            aria_button("Reset", "Clear input and result", on_click=reset_wien_calc).classes("negative")

                      
                        with ui.row().classes("items-center gap-3 mt-4"):
                            info_button = aria_button("Info ▶", "Information").props(
                                'aria-label="Open detailed information dialog about Planck law and Wien law"'
                            )
                            legend_button = aria_button("Legend ▶", "Variable and constants legend").props(
                                'aria-label="Open legend dialog about Planck law variables"'
                            )

                     
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
""", sanitize=False).props("id=planck-info-content role=document aria-live=polite")






                        def open_info(_=None):
                            info_dlg.open()
                            ui.run_javascript("""
                            if (window.MathJax && window.MathJax.typesetPromise) {
                                MathJax.typesetPromise();
                            }
                            """)


                        info_button.on("click", open_info)

                       
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

                        def open_legend(_=None):
                            legend_dlg.open()

                        legend_button.on("click", open_legend)


                  
                    with ui.column().classes("w-1/2 min-w-[400px] max-w-[520px] flex-shrink"):
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
                  
                        with ui.row().classes('gap-2 mt-4'):
                      
                            with ui.column().classes('items-center gap-4'):
                                temp_select = ui.select(
                                    [1,2,3,4,5,6,7,8,9,10,100,200,300,400,500,600,700,800,900,1000, 2000, 3000, 4000, 5000, 6000, 10000],
                                    label="Peak temperature (K)"
                                ).props('dense outlined aria-label="Select peak temperature"').classes('w-64')

                                temp_custom = ui.number(
                                    label="Or enter custom temperature [K]",
                                    format="%.6g"
                                ).props('dense outlined aria-label="Custom temperature in Kelvin"').classes('w-64')

                        
                            

                            with ui.column().classes('items-center gap-4 mt-2'):
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


                        auto_fill_checkbox = ui.checkbox(
                                "Auto-fill table from theoretical curve", 
                                value=False
                            ).props('aria-label="Auto fill table from theoretical curve"').classes('mt-2')
                        with ui.row().classes("items-center gap-3 mt-2"):
                            aria_button("Add Row", "Add new row to table", on_click=lambda: add_row_wl_I()).classes('mr-2')
                            aria_button("Generate Plot", "Overlay Planck curve for selected T", on_click=lambda: generate_and_plot()).classes('mr-2')
                            aria_button("Reset", "Clear table and curve", on_click=lambda: reset_all()).classes("negative")

                   
                        table_container = ui.column().classes("overflow-auto")


                        wl_input = aria_input("Wavelength [m]", "Enter wavelength in meters").classes("w-full")
                        I_input = aria_input("Intensity", "Enter corresponding intensity value").classes("w-full")
                      
                        def refresh_table():
                            nonlocal table_container
                            table_container.clear()  
                            rows = []
                            for r in user_rows:
                                rows.append({'wavelength_m': f"{r['wavelength_m']:.6e}", 'intensity': f"{r['intensity']:.6e}", 'source': 'manual'})
                            for r in generated_rows:
                                rows.append({'wavelength_m': f"{r['wavelength_m']:.6e}", 'intensity': f"{r['intensity']:.6e}", 'source': 'generated'})

                            if len(rows) == 0:
                                with table_container:
                                    ui.label("Table is empty. Add new rows with 'Add Row' or enable Auto-fill.").props('role=status aria-live=polite').classes('text-gray-400')
                            else:
                                with table_container:
                                    aria_table(
                                        columns=[
                                            {'name':'wavelength_m','label':'λ [m]','field':'wavelength_m'},
                                            {'name':'intensity','label':'Intensity','field':'intensity'},
                                            {'name':'source','label':'Source','field':'source'}
                                        ],
                                        rows=rows,
                                        label="Wavelengths and intensities (manual / generated)"
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

                          
                            if auto_fill_checkbox.value:
                             
                                accessible_notify("Table auto-filled with theoretical intensities.", type_='success')
                            else:
                                accessible_notify("Theoretical curve generated (table not auto-filled). Use Auto-fill to copy points into the table.", type_='info')

                           
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
              
                reference_box(
                    "Dataset reference: Riechers et al. 2022, *Nature*, 'Microwave background temperature at a redshift of 6.34 from H2O absorption'."
                )

              
                with ui.row().classes("w-full items-start justify-between gap-8 mt-4"):

               
                    with ui.column().classes("w-1/2"):

                       
                        info_box(
                    "Explore how the temperature of the cosmic microwave background (CMB) changes with redshift.  \n"
                    "Use the slider to adjust the adiabatic index γ and compare with the observed data points.\n"
                    "Dataset: Redshift (z), CMB Temperature (K), T/z+1, T_error"
                )
                        gamma_slider = ui.slider(min=0.0, max=2.0, value=gamma_default, step=0.0001) \
                            .props('role="slider" aria-label="Gamma adiabatic index slider" aria-valuemin="0.0" aria-valuemax="2.0" aria-valuenow="0.0"')

                        T0_input = ui.number(value=T0_default, label="T₀ [K] (present-day)", format="%.6g") \
                            .props('aria-label="Present-day CMB temperature input in Kelvin"')
                        with ui.row().classes("items-center gap-3"):
                            info_button = aria_button("Info ▶", "Information").props('aria-label="Open detailed information dialog"')
                            legend_button = aria_button("Legend ▶", "Variable and constants legend").props('aria-label="Open legend dialog"')

                        with ui.row().classes("items-center gap-3"):
                            gamma_box = ui.markdown("").classes("p-2 border rounded mt-3").props("aria-live=polite")
                            chi2_box = ui.markdown("").classes("p-2 border rounded").props("aria-live=polite")

                     
                        @ui.refreshable
                        def plot_area():
                            g = float(gamma_slider.value)
                            T0_val = float(T0_input.value)
                            T_grid = T_model(z_grid, T0_val, g)

                            y_min = min(np.nanmin(T_obs), np.nanmin(T_model(z_grid, T0_default, gamma_default)))
                            y_max = max(np.nanmax(T_obs), np.nanmax(T_model(z_grid, T0_default, gamma_default)))
                            y_pad = 0.12 * (y_max - y_min) if (y_max - y_min) > 0 else 1.0

                            frac_str = float_to_fraction_str(g, max_denominator=12)

                            with ui.pyplot():
                                plt.errorbar(z_obs, T_obs, yerr=T_err, fmt="o", markersize=4, color="blue",
                                            ecolor="gray", capsize=2, label="Observed data", zorder=5)
                                plt.plot(z_grid, T_grid, lw=2.2, color="green",
                                        label=f"Model: γ={g:.4f} ({frac_str})", zorder=4)
                                plt.xlabel("Redshift z")
                                plt.ylabel("CMB Temperature [K]")
                                plt.title("CMB Temperature vs Redshift")
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

                        plot_area()

                        def update_plot(*_):
                            plot_area.refresh()

                        gamma_slider.on("update:model-value", update_plot)
                        T0_input.on("update:model-value", update_plot)

                      
                        with ui.dialog().props(
                            'aria-modal="true" role="dialog" aria-labelledby="info-title" aria-describedby="info-content"'
                        ) as info_dlg, ui.card():
                            ui.label("Information").classes("text-lg font-semibold").props("id=info-title")
                            ui.markdown(
                                "**Thermodynamics and adiabatic transformations**  \n\n"
                                "Imagine the large-scale Universe as an isolated, closed system — it does not exchange heat or energy with any 'outside'. "
                                "Its total internal energy can be approximated as conserved over time. In this context, the early Universe was hot, dense, "
                                "and dominated by photon radiation. The cosmic microwave background (CMB) discovered by Penzias and Wilson is the remnant "
                                "of that radiation, now cooled to about 2.725 K. Its spectrum is an almost perfect blackbody, and its temperature has decreased "
                                "with the expansion of the Universe.  \n\n"
                                "The expansion behaves as an **adiabatic process**: total energy is conserved, the system expands, and the temperature drops "
                                "as the volume increases.  \n\n"
                                "**First Law of Thermodynamics:** ΔU = Q − L → dQ = dU + P dV → for adiabatic processes dQ = 0 ⇒ dU + P dV = 0.  \n\n"
                                "For an ideal gas: PV^γ = const → T V^(γ−1) = const → T ∝ V^(1−γ).  \n"
                                "The adiabatic index γ = Cₚ / Cᵥ depends on the degrees of freedom of the gas (equipartition of energy).  \n\n"
                                "- Monatomic gas: γ = 5/3  \n\n"
                                "- Diatomic gas: γ = 7/5  \n\n"
                                "In the adiabatic universe the main component is radiation, represented by photons that are no-massive particles. \n\n"
                                "- Polyatomic (photon gas): γ = 4/3  \n\n"
                                "**In cosmology:**  \n"
                                "Using the Stefan–Boltzmann law for radiation energy density, u = α_rad T⁴ with α_rad = 4σ/c ≈ 7.56×10⁻¹⁶ J·m⁻³·K⁻⁴, and P = u/3,  \n\n"
                                "we obtain (1/T) dT/dt = −(1/3V) dV/dt. With the scale factor a(t), since V ∝ a³, we get T ∝ a⁻¹.  \n\n"
                                "Because redshift relates to the scale factor as 1+z = 1/a, we have T(z) ∝ (1+z).  \n\n"
                                "In general form, for any γ: T(z) = T₀ (1 + z)^[3(γ - 1)]."
                            ).classes("prose").props("id=info-content role=document aria-live=polite aria-label='Detailed information about thermodynamics and adiabatic transformations in cosmology'")

                        def open_info(_=None):
                            info_dlg.open()

                        info_button.on("click", open_info).props(
                            'aria-controls="info-title" aria-expanded="false" role="button" aria-label="Open detailed information dialog"'
                        )
                        with ui.dialog().props(
    'aria-modal="true" role="dialog" aria-labelledby="legend-title" aria-describedby="legend-content"'
) as legend_dlg, ui.card():
                            ui.label("Legend").classes("text-lg font-semibold").props("id=legend-title")
                            ui.markdown(
                                "**Symbols and Units**  \n"
                                "- **T** = Temperature [K]  \n"
                                "- **V** = Volume [m³]  \n"
                                "- **P** = Pressure [Pa]  \n"
                                "- **γ** = Adiabatic index (Cp/Cv)  \n"
                                "- **Cp**, **Cv** = Specific heats [J·mol⁻¹·K⁻¹]  \n"
                                "- **kB** = Boltzmann constant = 1.380649×10⁻²³ J·K⁻¹  \n"
                                "- **R** = Gas constant = 8.314 J·mol⁻¹·K⁻¹  \n"
                                "- **a_rad** = Radiation density const. = 7.56×10⁻¹⁶ J·m⁻³·K⁻⁴  \n"
                                "- **σ** = Stefan–Boltzmann const. = 5.67×10⁻⁸ W·m⁻²·K⁻⁴  \n"
                                "- **z** = Cosmological redshift (dimensionless)  \n"
                                "- **λ_obs**, **λ_em** = Observed / Emitted wavelength [m]  \n"
                                "- **a(t)** = Scale factor (dimensionless)"
                            ).classes("prose").props("id=legend-content role=document aria-live=polite aria-label='Legend of variables and constants used in the exercise'")

                        def open_legend(_=None):
                            legend_dlg.open()

                        legend_button.on("click", open_legend).props(
                            'aria-controls="legend-title" aria-expanded="false" role="button" aria-label="Open legend dialog"'
                        )

                 
                    with ui.card().classes("p-6 w-full max-w-2xl min-w-[600px] bg-gray-900 text-white rounded-xl shadow-xl"):
                        ui.markdown(" CMB Thermodynamics Exercise").classes("text-2xl font-bold mb-4 text-blue-300").props("aria-label='CMB Thermodynamics Exercise Title'")
                        ui.markdown("Fill in the missing variables to connect thermodynamics and cosmology:").classes("text-sm text-gray-300").props("aria-label='CMB Thermodynamics Exercise Instructions'")

                        answer_du, answer_dq, answer_pdv, answer_pv, answer_gamma, answer_cp, answer_cv, answer_meno1, answer_zero, answer_zrel, answer_gamma_uno, answer_p, answer_v, answer_32, answer_T, answer_52, answer_75, answer_53, answer_four, answer_3, answer_1, answer_lambda_obs, answer_lambda_emit, answer_1z = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

                        @ui.refreshable
                        def show_adiabatic_exercise():
                            

                            ui.label("1) First Law Thermodynamics").classes("step mt-2 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                answer_dq["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("= ")
                                answer_du["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("+")
                                answer_pdv["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")

                            ui.label("2) Adiabatic condition").classes("step mt-2 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                answer_dq["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(" = ")
                                answer_zero["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                            with ui.row().classes("items-center text-lg"):
                                answer_du["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("+")
                                answer_pdv["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("= 0")

                            ui.label("3) Adiabatic Transformation").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                answer_p["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("·")
                                answer_v["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("^")
                                answer_gamma["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(" = const")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("T ∝ ")
                                answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("^(")
                                answer_gamma_uno["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(")")

                            ui.label("4) Adiabatic Index").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("γ = ")
                                answer_cp["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("/")
                                answer_cv["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")

                            ui.label("5) Energy Equipartition").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("Monoatomic gas: K = ")
                                answer_32["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("kB·")
                                answer_T["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(" → γ = ")
                                answer_53["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("Biatomic gas: U = ")
                                answer_52["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("nR·")
                                answer_T["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(" → γ = ")
                                answer_75["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")

                            ui.label("6) Stefan-Boltzmann law for radiation").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("U = a_rad·")
                                answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("^")
                                answer_four["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("·")
                                answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("P = ")
                                answer_1["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("/")
                                answer_3["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("·")
                                answer_T["el3"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("^")
                                answer_four["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("·")
                                answer_v["el4"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")

                            ui.label("7) Doppler–Redshift relation").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("1 + z = ")
                                answer_lambda_obs["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("/")
                                answer_lambda_emit["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("λ_emitted ∝ a(t) → universe expansion factor  ")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("a(t)=")
                                answer_1["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("/")
                                answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                
                            ui.label("8) Cosmological Application").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("The relation temperature-redshift: T(z) = T₀·(1+z)")
                           
                                
                            with ui.row().classes("items-center text-lg"):
                                ui.label("T ∝ a(t)^(")
                                answer_meno1["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label(")")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("Considering a section of the expanding universe V ∝ a(t)³")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("The adiabatic expansion:T ∝ a(t)^(3(")
                                answer_gamma_uno["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("))")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("T ∝ ")
                                answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
                                ui.label("^(")
                                answer_zrel["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md bg-gray-800 transition-colors duration-200")
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
                                           
                                            el.classes(remove='border-red-500 border-green-500 bg-red-900/10 bg-green-900/10')
                                        
                                            el.classes(add='border border-gray-600')
                                            el.update()

                            def mark(ans, ok):
                                for el in ans.values():
                                    if el:
                                       
                                        el.classes(remove='border-red-500 border-green-500 bg-red-900/10 bg-green-900/10')
                                     
                                        if ok:
                                            el.classes(add='border-green-500 bg-green-900/10')
                                        else:
                                            el.classes(add='border-red-500 bg-red-900/10')
                                     
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

           
                        aria_button("Check Exercise", "Check all formulas for correctness", on_click=check_and_run_adiabatic)
                      

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
                info_box(
                    "Explore the epoch where the energy density of matter and radiation were equal. "
                    "\nFill in the missing formulas to derive the physical laws governing both components. "
                    "\nOnce all inputs are correct, the two analytical curves will appear on the graph. "
                    "\nThen, estimate the redshift and temperature of the equivalence epoch."
                )

                with ui.row().classes("w-full items-start justify-between gap-8 mt-4"):

                
                    with ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 text-white rounded-xl shadow-xl"):
                        ui.markdown(" Radiation & Matter Density Evolution").classes("text-2xl font-bold mb-4 text-blue-300").props("aria-label='Radiation and Matter Density Evolution Exercise '")
                        
                        @ui.refreshable
                        def show_formulas():
                          

                            ui.label("1) Energy density definition").classes("step mt-2 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("ρ = ")
                                answer_u["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")                         
                                ui.label("/")
                                answer_v["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                             
                            ui.label("2) Matter particles energy ").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("The energy of non relativistic matter particles:")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("Eₘ =(")   
                                answer_m["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·")
                                answer_c["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("^2)")
                                ui.label("+(1/2·")
                                answer_m["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·")
                                answer_v["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("^2)")
                                
                            ui.label("3) Matter energy density").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"): 
                                ui.label("ρₘ = ") 
                                answer_E["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("/")
                                answer_v["el3"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("Considering a portion of universe: V ∝ a(t)³")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("ρₘ=Ωm ·ρc with ρc=(3·H0²)/(8·π·G)")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("ρₘ(a) =ρₘ0· a^(")
                                answer_m3["el"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label(")= ρₘ0·(")
                                answer_1z["el"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label(")^")
                                answer_3["el"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label("with ρₘ0=Ωm0 ·ρc ")

                            ui.label("4) Radiation energy density").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("The energy for photons: Eᵣ =")
                                answer_h["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·")
                                answer_nu["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("=")
                                answer_h["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·")
                                answer_c["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("/")
                                answer_lambda["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("The wavelength stretches with universe expansion: λ ∝ a(t)")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("The number of photons per volume decreases with expansion: n ∝ a(t)^(-3)")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("Every photon energy decreases with expansion: Eᵣ ∝ a(t)^(-1)")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("From the energy equation: ρᵣ= ")
                                answer_n["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·")
                                answer_E["el2"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("∝ a(t)^(")
                                answer_m4["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label(")")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("From the Stefan-Boltzmann law: ρᵣ= a_rad ·")
                                answer_T["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("^")
                                answer_4["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("with T ∝ a(t)^(-1)")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("ρᵣ=Ωr ·ρc ")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("ρᵣ(a) =ρᵣ0· a^(")
                                answer_m4["el2"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label(")= ρᵣ0·(")
                                answer_1z["el2"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label(")^")
                                answer_4["el2"] = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")
                                ui.label("with ρᵣ0=Ωr0 ·ρc ")
                                

                

                            ui.label("5) Equivalence condition").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                answer_rhom["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("(z_eq) = ")
                                answer_rhor["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("(z_eq)  ")

                            ui.label("6) Redshift of equivalence").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("1 + z_eq = ")
                                answer_rhom0["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("/")
                                answer_rhor0["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")

                            ui.label("7) Temperature–redshift relation").classes("step mt-4 text-blue-200")
                            with ui.row().classes("items-center text-lg"):
                                ui.label("T(z) = ")
                                answer_T0["el"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label("·(")
                                answer_1z["el3"] = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                                ui.label(")")
                               

                          

                        show_formulas()
                        formula_button_slot = ui.row().classes("mt-4 justify-center")

                    with ui.column().classes("w-1/2 items-center bg-gray-900 p-4 rounded-xl shadow-lg"):

                        ui.label("Energy density evolution").classes("text-lg font-semibold text-blue-300 mb-2").props("aria-label='Energy density evolution plot title'")

              
                        with ui.row().classes("gap-4 mb-4"):
                            unit_select = ui.select(
                                {
                                    "kg/m³": "kg/m³",
                                    "J/m³": "J/m³",
                                    "erg/cm³": "erg/cm³",
                                    "GeV/cm³": "GeV/cm³",
                                    "M⊙/Mpc³": "M⊙/Mpc³",
                                },
                                value="kg/m³",
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

                     
                        with ui.row().classes("items-center gap-2 mb-2"):
                            ui.label("ρₘ = ρₘ₀ · (")
                            rho_m_z_input = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                            ui.label(")^")
                            rho_m_power_input = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")

                        with ui.row().classes("items-center gap-2 mb-4"):
                            ui.label("ρᵣ = ρᵣ₀ · (")
                            rho_r_z_input = aria_formula_input().props("dense filled").classes("w-24 border border-gray-600 rounded-md")
                            ui.label(")^")
                            rho_r_power_input = aria_formula_input().props("dense filled").classes("w-16 border border-gray-600 rounded-md")

                        generate_plot_btn = aria_button("Generate Plot", "Plot densities").classes("mt-2")

                     
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
                           
                            with ui.row().classes("gap-4 mt-4"):
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

                                aria_button("Check Results", "Verify z_eq and T_eq", on_click=check_results)

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
                        )

            
                with ui.row().classes("mt-6 gap-3"):
                    info_button = aria_button("Info ▶", "Detailed physical explanation")
                    legend_button = aria_button("Legend ▶", "Variable and constants legend")

                with ui.dialog().props('aria-modal="true" role="dialog" aria-labelledby="info-title" aria-describedby="info-content"') as info_dlg, ui.card():
                    ui.label("Information").classes("text-lg font-semibold").props("id=info-title aria-label='Information dialog title'")
                    ui.markdown(
                        "**Equivalence between matter and radiation densities**  \n\n"
                        "After the Big Bang, the Universe passed through three main eras:  \n"
                        "- **Radiation-dominated epoch** (ρᵣ > ρₘ)  \n"
                        "- **Matter–radiation equality** (ρᵣ = ρₘ)  \n"
                        "- **Matter-dominated epoch** (ρₘ > ρᵣ)  \n\n"
                        "During the cosmic expansion, the density of matter decreases as a⁻³, while radiation decreases faster (a⁻⁴) "
                        "since photons lose energy by redshifting (λ ∝ a ⇒ E ∝ 1/a).  \n\n"
                        "From classical physics: for non-relativistic matter, E ≈ mc² → ρₘ ∝ 1/a³.  \n"
                        "For radiation: E_γ = hν = hc/λ ⇒ ρᵣ ∝ 1/a⁴.  \n\n"
                        "Using the Stefan–Boltzmann law (ρᵣ = α_rad T⁴) and the adiabatic expansion (T ∝ 1/a), we also get ρᵣ ∝ a⁻⁴.  \n\n"
                        "The equality redshift z_eq satisfies ρₘ(z_eq) = ρᵣ(z_eq)."
                    ).classes("prose text-gray-200").props("id=info-content aria-label='Information dialog content'")

                with ui.dialog().props('aria-modal="true" role="dialog" aria-labelledby="legend-title" aria-describedby="legend-content"') as legend_dlg, ui.card():
                    ui.label("Legend").classes("text-lg font-semibold").props("id=legend-title aria-label='Legend dialog title'")
                    ui.markdown(
                        "**Symbols and Constants**  \n"
                        "- **ρₘ, ρᵣ** = matter / radiation energy density [kg·m⁻³]  \n"
                        "- **Ωₘ₀, Ωᵣ₀** = present-day density parameters (dimensionless)  \n"
                        "- **ρ_c** = critical density = 3H₀² / 8πG  \n"
                        "- **H₀** = Hubble constant = 2.27×10⁻¹⁸ s⁻¹  \n"
                        "- **G** = gravitational constant = 6.67×10⁻¹¹ m³·kg⁻¹·s⁻²  \n"
                        "- **T₀** = present-day CMB temperature = 2.725 K  \n"
                        "- **z** = redshift (dimensionless)  \n"
                        "- **a(t)** = scale factor = 1/(1+z)  \n"
                        "- **α_rad** = radiation density constant = 7.56×10⁻¹⁶ J·m⁻³·K⁻⁴"
                        "- **h** = Planck constant = 6.626×10⁻³⁴ J·s  \n"
"- **c** = speed of light = 2.998×10⁸ m/s  \n"
"- **λ** = photon wavelength [m]   \n"
"- **ν** = photon frequency [Hz] \n"

                    ).classes("prose text-gray-200").props("id=legend-content aria-label='Legend dialog content'")

                def open_info(_=None): info_dlg.open()
                def open_legend(_=None): legend_dlg.open()

                info_button.on("click", open_info)
                legend_button.on("click", open_legend)




        
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
    

    standard_module_ui("Universe Evolution & CMB")


 


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


@ui.page('/module3')
def module3():
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

  
  .paragraph {
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
    




    main_layout("Module 3: Dark Matter ")
    ui.label("Select a section to explore dark matter activities").props('id=tabs_desc role=heading aria-level=2 tabindex=0')

    with ui.column().classes('w-full p-4 gap-4'):
        with ui.tabs().classes('w-full') as tabs:
            one, two, three, four,five = ui.tab('Galaxy rotation curve').props('role=tab aria-selected=true'), ui.tab('Galaxy mass & DM').props('role=tab aria-selected=false'), ui.tab('Cluster velocity distribution').props('role=tab aria-selected=false'), ui.tab('Cluster mass & DM').props('role=tab aria-selected=false'),ui.tab('CMB').props('role=tab aria-selected=false')
      

        with ui.tab_panels(tabs, value=one).classes('w-full'):
            with ui.tab_panel(one).props('role=tabpanel'):
                with ui.card().classes("p-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Rotation Curve")
         
                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about galaxy rotation curve activity').classes('text-lg'):
                    paragraph(
                    "Explore the rotation curve of the NGC3198 galaxy."
                    "\n\n The **red curve** shows the **baryonic prediction** with a Keplerian-like trend (initial rise followed by a decrease). "
                    "\n\n The **green curve** simulates the addition of a **dark matter halo** to flatten the curve and match observations."
                    "\n\n**Slider**: 0 = total baryonic curve, 1 = total curve with full dark matter halo contribution.")
                with ui.element('div').props('role=region tabindex=0 aria-label=Dataset variables').classes('text-lg'):
                    info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                
               
                    reference_box(
    "**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.").classes('text-base italic')
   

                
                
                with ui.row().classes('w-full mt-4'):
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

        """, sanitize=False).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the computational steps of the galaxy rotation curve activity')

                        aria_button("Close", "close the box",on_click=velocity_dialog.close)

   
                aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [velocity_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                ui.html('''
<div class="flex space-x-2">
  <button id="audio-button" role="button" aria-label=" or deactivate audio" aria-pressed="false" onclick="initAudio()" 
            class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Activate Audio
    </button>
  <button role="button" aria-label="Play observed velocity curve mean" onclick="playObservedVelMean()" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-1 rounded">▶ Observed Vel (mean)</button>
  <button role="button" aria-label="Play observed velocity curve" onclick="playObservedVelCurve()" class="bg-blue-300 hover:bg-blue-500 text-white px-3 py-1 rounded">▶ Observed Vel (curve)</button>
  <button role="button" aria-label="Play baryonic velocity curve mean" onclick="playBaryonicVelMean()" class="bg-red-500 hover:bg-red-700 text-white px-3 py-1 rounded">▶ Baryonic Vel (mean)</button>
  <button role="button" aria-label="Play baryonic velocity curve" onclick="playBaryonicVelCurve()" class="bg-red-300 hover:bg-red-500 text-white px-3 py-1 rounded">▶ Baryonic Vel (curve)</button>
  <button role="button" aria-label="Play simulated velocity curve mean" onclick="playSimVelMean()" class="bg-green-500 hover:bg-green-700 text-white px-3 py-1 rounded">▶ Simulated Vel (mean)</button>
  <button role="button" aria-label="Play simulated velocity curve" onclick="playSimVelCurve()" class="bg-green-300 hover:bg-green-500 text-white px-3 py-1 rounded">▶ Simulated Vel (curve)</button>
  <button role="button" aria-label="Play difference velocity curves mean" onclick="playDifferenceVelMean()" class="bg-yellow-500 hover:bg-yellow-700 text-black px-3 py-1 rounded">▶ Difference (mean)</button>
  <button role="button" aria-label="Play difference velocity curves" onclick="playDifferenceVelCurve()" class="bg-yellow-300 hover:bg-yellow-500 text-black px-3 py-1 rounded">▶ Difference (curve)</button>
</div>
''', sanitize=False)
                ui.label("").props(
    'id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0'
)
                DATA_LOADED=False
              
                G_grav = 4.30091e-6   # kpc (km/s)^2 / M_sun
            
                r_ngc = np.array([])
                v_obs_ngc = np.array([])
                v_gas_ngc = np.array([])
                v_disk_ngc = np.array([])
                v_bul_ngc = np.array([])
                v_err_ngc = np.array([])
                r_match = None
                current_galaxy_name = None
                

           

                ui.label("Move the slider to add the dark matter to the simulated velocity curve").props('id=alpha_slider_label')
                alpha_slider = aria_slider(min=0.0, max=2.0, value=0.0, step=0.01,
                                        aria_label="Dark matter fraction control slider").props('aria-describedby=alpha_slider_label label-always')

                def load_galaxy_data(filename):
                    global DATA_LOADED

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
            
                with ui.row().classes('w-full justify-center mt-4'):
                    galaxy_files = [f for f in os.listdir(GALAXY_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]

                    if not galaxy_files:
                        warning_box("Nessun file di galassia trovato in GALAXY_DATA_PATH").classes("text-red-500")
                        galaxy_files = []

                    default_galaxy = "NGC3198.txt" if "NGC3198.txt" in galaxy_files else (galaxy_files[0] if galaxy_files else None)
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

                    galaxy_select = ui.select(
                        galaxy_files,
                        value=default_galaxy,
                        label='Select a Galaxy Dataset'
                    ).classes('w-1/2')
                
                    selected_file=galaxy_select.value
                    galaxy_file_map = get_data_and_images(GALAXY_DATA_PATH, GALAXY_IMG_PATH)
                    image_filepath=galaxy_file_map.get(selected_file)

             
                def reload_galaxy(new_value):
                   
                    global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED,current_galaxy_name 

                    if not new_value:
                        #print("reload_galaxy: filename is None, abort")
                        return

                    filename = new_value
                    #print("reload_galaxy: filename scelto ->", filename)

                    loaded = load_galaxy_data(filename)
                    if loaded is None:
                        #print("reload_galaxy: load fallito")
                        DATA_LOADED = False
                        return

                    r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match = loaded
                    DATA_LOADED = True
                    current_galaxy_name = filename
                    #print(f"reload_galaxy: current_galaxy_name impostato -> {current_galaxy_name!r}")
                    alpha_slider.value = 0.0
                    update_all_plots.refresh()
               




               
                galaxy_select.on_value_change(lambda e: reload_galaxy(e.value))

                alpha_slider.on('update:model-value', update_all_plots.refresh)
                
                
                if default_galaxy:
                    reload_galaxy(default_galaxy)
                    current_galaxy_name = default_galaxy
                    #print("startup: current_galaxy_name impostato a default ->", current_galaxy_name)

                

               
                



                points_display = ui.column()
                history_display = ui.column()
              
                with ui.column().classes('w-full items-center'):
                    with ui.row().classes('w-full justify-center gap-4'):
                        plot_container = ui.column().classes("flex-1 items-center")
                        mass_plot_container = ui.column().classes("flex-1 items-center")
                    with ui.row().classes('w-full justify-center gap-4'):
                        morph_plot_container = ui.column().classes("flex-1 items-center")
                    
                        with ui.column().classes("flex-1 items-center"):
                            chi2_plot_container = ui.column()
                          
                            with ui.row().classes("gap-2 mt-2 justify-center"):
                                aria_button("Add χ² point", "Add a new chi-squared point",on_click=lambda: add_chi2_point())
                                aria_button("Refresh χ² plot", "Refresh the chi-squared plot",on_click=lambda: refresh_chi2_plot())

                            slider_result_label = ui.label("").classes("text-green-600 font-bold mt-2").props('role=status aria-live=polite aria-atomic=true')

                        
                with ui.card().classes("w-11/12 xl:w-5/6 mx-auto mt-6 p-8 border-2 border-dashed border-blue-300").props('role="region" aria-label="Parabolic interpolation instructions"'):
                    with ui.element('div').classes("grid grid-cols-12 gap-12 items-start w-full"):
                        with ui.column().classes("col-span-12 lg:col-span-7 space-y-4"):
                            subtitle("Parabolic interpolation and compute χ² minimization")
                            paragraph("Use the formula below to compute the χ² minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)). "
                              "\n\n 1a) Choose 3 values of dark matter fraction (f) with the slider above to compute the χ² at that point."
                              "\n\n 1b) Press 'Add χ² point' button to add the point to the χ² plot."
                              "\n\n 2a) Input 3 values for dark matter mass points (x-coordinates),"
                              "\n\n 2b) Press 'compute χ² and minimum' button to calculate the χ² at those points and check the minimum on the plot.")
                             
                            ui.html(r"""
<p style="font-family:monospace; background:#e0f0ff; padding:10px; border:1px solid #88c; border-radius:8px; color:#111;">

<b>Formula:</b><br>
$$
x_{min} = \frac{1}{2} \cdot \frac{ (a^2-b^2) f(c) + (b^2-c^2) f(a) + (c^2-a^2) f(b) }{ (a-b) f(c) + (b-c) f(a) + (c-a) f(b) }
$$
</p>""",sanitize=False).props(' aria-label=Parabolic interpolation formula for chi-squared minimization')
                                    
                        with ui.column().classes("col-span-12 lg:col-span-5 flex flex-col items-center space-y-4 mt-8"):
        
        
                            with ui.grid(columns=2).classes("w-full max-w-md gap-3"):
                            
                      
                                input_a = ui.number(label='Mass Point 1 (x₁)', format='%.2e').classes('flex-1 font-bold').props('aria-label="First mass point for chi-squared computation"')
                                fa_in = ui.number(label="f(a) = χ²(a)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point a"')
                                input_b = ui.number(label='Mass Point 2 (x₂)', format='%.2e').classes('flex-1 font-bold').props('aria-label="Second mass point for chi-squared computation"')
                                fb_in = ui.number(label="f(b) = χ²(b)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point b"')
                                input_c = ui.number(label='Mass Point 3 (x₃)', format='%.2e').classes('flex-1 font-bold').props('aria-label="Third mass point for chi-squared computation"')

                                fc_in = ui.number(label="f(c) = χ²(c)", format="%.6g").classes("flex-1 font-bold").props('aria-label="Chi-squared value at point c"')

                            result_label = ui.label("Results: ---").classes("text-green-600 font-bold mt-2 font-bold").props('role=status aria-live=polite aria-atomic=true')
                            
                            with ui.row().classes("w-full justify-center gap-6 mt-10"):
                                aria_button("Compute χ² and minimum", "Compute chi-squared and minimum from three input dark matter masses values",on_click=lambda: initialize_parabolic_points())
                       
                
                
                with ui.row().classes("gap-4 mt-4"):
                    points_display
                    history_display
                    f = float(alpha_slider.value)
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
                            with ui.pyplot(figsize=(8, 5), close=False, clear=True):
                             
                                
                                plt.plot(r_ngc, v_baryonic, color='red', linewidth=2, label='Keplerian velocity')
                                plt.plot(r_ngc, v_total_curve, linewidth=2, color='green', label=f"Simulated velocity (DM mass: {M_dm_tot:.2e} M☉, {dm_fraction:.1f}% of visible mass)")
                                plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', markersize=4, color='blue', ecolor='gray', capsize=2, label='Observed velocity', zorder=5)
                            
                                plt.xlabel('Radius (kpc)')
                                plt.ylabel('Rotation Speed (km/s)')
                                
                                plt.title(f'Galaxy Rotation Curve: {current_galaxy_name}')


                                plt.ylim(0, max(300, 1.1 * np.max(v_obs_ngc + v_err_ngc)))
                                plt.xlim(0, r_ngc.max() * 1.1)
                                plt.grid(True)
                                plt.legend(loc='upper right', fontsize='small')
                                ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Rotational velocity curve of the galaxy (from data),baryonic velocity curve (computed from data) and simulated velocity curve updated with new dark matter value (slider moved)'
)

                            plot_info_box_compact({
    "Visible (luminous) mass": f"{M_vis_tot:.2e} M☉",
    "Dark matter mass": f"{M_dm_tot:.2e} M☉ ({dm_fraction:.1f}% of visible)",
    "Component contribution": f"Gas {v_gas_frac:.1f}%, Disk {v_disk_frac:.1f}%, Bulge {v_bul_frac:.1f}%",
    "Fit quality (χ²/dof)": f"{chi2_dof:.2f}",
    "V_max observed": f"{v_obs_ngc.max():.1f} km/s at r ≈ {r_ngc[np.argmax(v_obs_ngc)]:.1f} kpc",
    "V_max simulated": f"{v_total_curve.max():.1f} km/s at r ≈ {r_ngc[np.argmax(v_total_curve)]:.1f} kpc",
    f"DM fraction within {r_ref} kpc": f"{dm_frac_r:.1f}%",
    "Residuals": f"Mean = {np.mean(residui):.2f} σ, Max = {np.max(np.abs(residui)):.2f} σ"
})
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
                            with ui.pyplot(figsize=(8, 5), close=False,clear=True):
                           
                               
                                plt.plot(r_ngc, m_baryonic, "r-", lw=2, label="Baryonic mass")
                                plt.plot(r_ngc, m_total_obs, "b-", lw=2, label="Total mass")
                                plt.plot(r_ngc, m_total_model, "g-", lw=2, label=f"Simulated mass DM: {np.max(m_total_model):.2e} M☉")
                             
                                plt.xlim(x_min_mass, x_max_mass)
                                plt.ylim(y_min_mass, y_max_mass)

                                plt.xlabel("Radius (kpc)")
                                plt.ylabel("Mass ($M_\\odot$)")
                                plt.title(f"Mass vs radius: {current_galaxy_name}")

                                plt.legend()
                                plt.grid(True)
                                ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Total mass of the galaxy (from data), baryonic mass (computed from data) and simulated mass updated with new dark matter value'
)

                            plot_info_box_compact({"Visible baryonic mass": f"{np.max(m_baryonic):.2e} M☉",
        "Total mass": f"{np.max(m_total_obs):.2e} M☉",
        f"Simulated mass DM (α={f:.2f})": f"{np.max(m_total_model):.2e} M☉"
    })
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

   
                            R_bulge = np.average(r_ngc, weights=np.maximum(0, v_bul_ngc**2)) if np.any(v_bul_ngc > 0) else 1
                            R_disk  = np.average(r_ngc, weights=np.maximum(0, v_disk_ngc**2)) if np.any(v_disk_ngc > 0) else 5
                            R_halo  = np.sqrt(M_dm_tot / (M_vis_tot+1e-12)) * np.max(r_ngc) if M_dm_tot > 0 else 0
                        

                            with morph_plot_container:
                                morph_plot_container.clear()
                                plt.close()
                                with ui.pyplot(figsize=(5,5), close=False,clear=True):
                                 
                                    
                                    ax = plt.gca()

        
                                    #bulge = plt.Circle((0,0), R_bulge, color='gold', alpha=0.6, label=f'Bulge (M={M_bulge:.1e})')
                                    #ax.add_artist(bulge)

           
                                    disk = plt.Circle((0,0), R_disk, color='red', alpha=0.3, label=f'Disk (M={M_disk:.1e})')
                                    ax.add_artist(disk)

          
                                    if f > 0 and R_halo > 0:
                                        halo = plt.Circle((0,0), R_halo, color='green', alpha=0.15,label=f'Halo DM (M={M_dm_tot:.1e}, f={f:.2f})')
                                        ax.add_artist(halo)

                                    maxR = max(R_halo, R_disk, R_bulge) * 1.2
                                    ax.set_xlim(-maxR, maxR)
                                    ax.set_ylim(-maxR, maxR)
                                    ax.set_aspect('equal', 'box')
                                    ax.set_xlabel("x [kpc]")
                                    ax.set_ylabel("y [kpc]")
                                    ax.set_title(f"Galaxy structure from above: {current_galaxy_name}")

                                    ax.legend()
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Galaxy morphology representation (from above), showing disk with baryonic component (stars,gas) and halo appears adding dark matter (slider)'
)

                                plot_info_box_compact({
    "Disk Mass": f"{M_disk:.2e} M☉",
    "DM Mass": f"{M_dm_tot:.2e} M☉"
})
                 
                    chi2_points = []
                    manual_points = []
                   
                    formula_pts = {"a": None, "b": None, "c": None, "fa": None, "fb": None, "fc": None}
                 

                
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


                    
                    parabolic_state = {
                        "points": [],
                        "history": [],
                        "plot_points": [],
                        "iteration": 0
                    }
                  
                 

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
                        update_all_plots.refresh()
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

              
                    def plot_chi2_user_curve():
                        global r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, v_err_ngc, r_match, DATA_LOADED
                       
                        with chi2_plot_container:
                            chi2_plot_container.clear()
                            plt.close()
                            with ui.pyplot(figsize=(8,5), close=False,clear=True):
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
  
                    fa_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    fb_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    fc_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_a.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_b.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_c.on('update:model-value', lambda e: refresh_formula_from_inputs())

               
                    refresh_formula_from_inputs()
                    plot_chi2_user_curve()
                    
                    def refresh_chi2_plot():
                        chi2_points.clear()
                        parabolic_state["points"].clear()
                        parabolic_state["plot_points"].clear()
                        parabolic_state["history"].clear()
                        parabolic_state["iteration"] = 0
                        manual_points.clear()
                   

                        formula_pts["auto_update"] = False 
                        update_all_plots.refresh()
                    
          
                    update_all_plots()
          
                    
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
                                                table_filepath = os.path.join(
                    GALAXY_TABLES_PATH,
                    os.path.splitext(selected_file)[0] + ".csv"
                )
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

                    


                
               
                

     
            with ui.tab_panel(two).props('role=tabpanel'):
                with ui.card().classes("p-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Mass ")
                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about galaxy velocity and mass activities').classes('text-lg'):
                    paragraph(
    "You are an astrophysicist investigating the **presence of dark matter** in a galaxy. "
    "\n\n Choose a dataset and complete the missing fields with the correct formulas. "
    "\n\n Click **Run Analysis** to compare the baryonic mass with the total mass and the observed velocity with the Keplerian-like velocity.")
                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about dataset variables').classes('text-lg'):
                    info_box( "**Dataset variables**: Rad (radius), Vobs (observed velocity),  errV (velocity error), Vgas (gas velocity), Vdisk (disk velocity), Vbul (bulge velocity),SBdisk (surface brightness disk),SBbul (surface brightness bulge)")
                
               
                reference_box(
    "**Dataset reference**: Lelli F. et al., *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*.").classes('text-base italic')
                with ui.row().classes('w-full mt-4'):
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
        """, sanitize=False).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the steps to compute baryonic velocity and mass from data')
                            aria_button("Close", "close the box",on_click=baryonic_dialog.close)
                        with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"')  as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-gray-800 border border-blue-400 shadow-lg'):
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
                            aria_button("Close", "close the box",on_click=legend_dialog.close).classes("mt-4 bg-blue-600 text-white rounded-lg px-4 py-2")

                    
                        with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog,ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-slate-500 border border-green-500 shadow-lg'):
                                ui.markdown("**Units conversion**").classes('!text-xl !text-green-300 !-mt-2').props('id="units-title"')
                                ui.markdown("""
- `1 kpc` = 3.086 × 10¹⁶ m = 3.26 × 10³ light-years  
- `1 pc` = 3.086 × 10¹³ km = 3.26 light-years  
- `1 Mpc` = 10⁶ pc = 3.086 × 10¹⁹ km  
- `1 km/s` = 3.6 × 10³ km/h = 10³ m/s  
- `1 M☉` = 1.989 × 10³⁰ kg (solar mass)  
- `1 L☉` = 3.828 × 10²⁶ W (solar luminosity)  
 """).classes('text-lg').props('id="units-desc"')
                            aria_button("Close","close the box", on_click=units_dialog.close).classes("mt-4 bg-green-600 text-white rounded-lg px-4 py-2")

                    
   
                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [baryonic_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📘 Legend", "Read the legend of symbols and units",on_click=legend_dialog.open).classes(
    "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
)
                        aria_button("📐 Units","Read the units conversion", on_click=units_dialog.open).classes(
    "bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
)



              

                galaxy_files = [f for f in os.listdir(GALAXY_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                galaxy_file_map = get_data_and_images(GALAXY_DATA_PATH, GALAXY_IMG_PATH)
                
                if not galaxy_file_map:
                    ui.label("No galaxy data files found in 'App/galaxy_data/' directory.").classes('text-red-500')
                else:
                    with ui.row().classes('w-full items-center'):
                        galaxy_select = ui.select(galaxy_files, label='Select a Galaxy Dataset').classes('flex-1 max-w-md items-start text-lg')
                        
                 
                    answer_vb, answer_vo,answer_rad,answer_g,answer_Mtot,answer_Mbar,answer_vgas,answer_vdisk,answer_vbulge = {}, {},{},{},{},{},{},{},{}
                    @ui.refreshable
                    def show_galaxy_pseudocode():
                        with ui.card().classes("p-6 w-full max-w-4xl bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown("### 🧮 Galaxy Mass Exercise").classes("text-2xl font-bold mb-4 text-blue-300")
                            ui.markdown("#### Fill the missing parts in the pseudo-code:")
                            with ui.column().classes('pseudo-code w-full'):
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
                                                table_filepath = os.path.join(
                    GALAXY_TABLES_PATH,
                    os.path.splitext(selected_file)[0] + ".csv"
                )
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



                    aria_button("Run Analysis", "Run the analysis to reproduce the plots",on_click=check_and_run_galaxy)


            with ui.tab_panel(three).props('role=tabpanel'):
                with ui.card().classes("p-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Velocity Distribution")

                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about galaxy cluster activity').classes('text-lg'):
                    paragraph("The **blue histogram shows the observed velocities** of galaxies in the Coma cluster; this plot is **fixed**. "
                            "\n\n Use the slider to **add Dark Matter (DM)** to your simulation. "
                            "\n\n With **DM = 0**, the simulated galaxies are gravitationally unbound. "
                            "\n\n As you increase the **DM**, the gravitational pull keeps the galaxies bound and shapes their velocity distribution to match the observed histogram.")
                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about Coma cluster dataset').classes('text-lg '):
                    info_box("**Dataset variables**: objid (galaxy ID),ra (right ascension),dec (declination),modelmag_r (apparent magnitude in r band),modelmagerr_r (magnitude error),extinction_r,redshift (z),zErr (redshift error)")
                                                  
                    reference_box("**Dataset reference**: [Kaggle:Coma cluster](https://www.kaggle.com/datasets/mertalkan98/coma-cluster) ; [SDSS](https://www.sdss.org/science/data-release-publications)").classes('text-base italic')
                             
                             

                
                with ui.row().classes('w-full justify-center mt-4'):
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

        """, sanitize=False).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the steps to compute cluster properties from data')
                        aria_button("Close","Close the box", on_click=cluster_dialog.close)
                        
                        
                        
                        
                aria_button("Info", "Read the detailed information about computational steps from data to plots",on_click=safe_click(lambda: [cluster_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded")      
               
              
                ui.html('''
<div class="flex space-x-2">
  <button id="audio-button" role="button" aria-label="Activate or deactivate audio" aria-pressed="false"
            onclick="initAudio()" 
            class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Activate Audio
    </button>
  <button role="button" aria-label="Play observed dispersion velocity mean" onclick="playObservedSigmaMean()" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-1 rounded">▶ Observed σ (mean)</button>
  <button role="button" aria-label="Play observed dispersion velocity" onclick="playObservedSigmaCurve()" class="bg-blue-300 hover:bg-blue-500 text-white px-3 py-1 rounded">▶ Observed σ (curve)</button>
  <button role="button" aria-label="Play baryonic dispersion velocity mean" onclick="playBaryonicSigmaMean()" class="bg-red-500 hover:bg-red-700 text-white px-3 py-1 rounded">▶ Baryonic σ (mean)</button>
  <button role="button" aria-label="Play baryonic dispersion velocity" onclick="playBaryonicSigmaCurve()" class="bg-red-300 hover:bg-red-500 text-white px-3 py-1 rounded">▶ Baryonic σ (curve)</button>
  <button role="button" aria-label="Play simulated dispersion velocity mean" onclick="playSimSigmaMean()" class="bg-green-500 hover:bg-green-700 text-white px-3 py-1 rounded">▶ Simulated σ (mean)</button>
  <button role="button" aria-label="Play simulated dispersion velocity" onclick="playSimSigmaCurve()" class="bg-green-300 hover:bg-green-500 text-white px-3 py-1 rounded">▶ Simulated σ (curve)</button>
  <button role="button" aria-label="Play differences dispersion velocities mean" onclick="playDifferenceSigmaMean()" class="bg-purple-700 hover:bg-purple-900 text-white px-3 py-1 rounded">▶ Difference σ (mean)</button>
  <button role="button" aria-label="Play differences dispersion velocities" onclick="playDifferenceSigmaCurve()" class="bg-purple-500 hover:bg-purple-700 text-white px-3 py-1 rounded">▶ Difference σ (curve)</button>

</div>
''', sanitize=False)
                ui.label("").props(
    'id=audio_status role=status aria-live=polite aria-atomic=true tabindex=0'
)
                DEFAULT_CLUSTER = "coma_data.csv" 
                #DEFAULT_CLUSTER = "Abell0080.txt"
                cluster_name = None


                dataset_files = [
                    f for f in os.listdir(CLUSTER_DATA_PATH)
                    if f.lower().endswith(('.txt', '.dat', '.csv'))
                ]
                if DEFAULT_CLUSTER not in dataset_files:
                    dataset_files.insert(0, DEFAULT_CLUSTER)
                ui.label("Select Cluster Dataset")
                dataset_selector = ui.select(
                    options=dataset_files,
                    value=DEFAULT_CLUSTER,
                    label="Dataset"
                ).classes("w-64")
                select=dataset_selector.value
              
                cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)
                image_filepath = cluster_file_map.get(select)
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
                            stats_container_histo = ui.column()
   
                        with ui.column().classes('flex-1 items-center'):
                            plot_container_scatter = ui.column()
                            stats_container_scatter = ui.column()
                
                  

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
                else:
                    initialize_cluster_from_df(df)
                recompute_axis_limits()
                cluster_name = DEFAULT_CLUSTER
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
                    if new_value.lower() == "coma_data.csv":
                        df = load_coma_dataset(new_value)
                      
                    else:
                        df = load_cluster_dataset(new_value)
                        initialize_cluster_from_df(df)
                    recompute_axis_limits()
                    cluster_name = filename
                    #plt.close()
                    #plot_container_histo.clear()
                    #stats_container_histo.clear()
                    #plot_container_scatter.clear()
                    #stats_container_scatter.clear()
                    dm_slider.value = 0.0
                    refresh_cluster_plots.refresh()
                    #ui.timer(0.1, refresh_cluster_plots.refresh, once=True)
                    global state, N_gal, vx_obs, vy_obs
                    N_gal = min(80, len(observed_vel))
                    rng = np.random.default_rng(123)
                    vx_obs = rng.normal(0.0, sigma_obs/3000.0, N_gal)
                    vy_obs = rng.normal(0.0, sigma_obs/3000.0, N_gal)

                    state = {
                        "x_obs": rng.uniform(0.0, 1.0, N_gal),
                        "y_obs": rng.uniform(0.0, 1.0, N_gal),
                        "x_model": None,
                        "y_model": None,
                        "vx_obs": vx_obs.copy(),
                        "vy_obs": vy_obs.copy(),
                        "vx_model": vx_obs.copy(),
                        "vy_model": vy_obs.copy(),
                        "rng": rng,
                    }
                    state["x_model"] = state["x_obs"].copy()
                    state["y_model"] = state["y_obs"].copy()

                    
                dm_slider.on('update:model-value', refresh_cluster_plots.refresh)
                #dataset_selector.on('update:model-value',  on_dataset_change)
                dataset_selector.on_value_change(lambda e: on_dataset_change(e.value))


                
                with ui.column().classes('w-full items-center '):
                    #dm_slider_min, dm_slider_max = 0.0, 50.0
                    #ui.label("Move the slider to add dark matter to the simulated velocity distribution in the cluster").props('id=dm_slider_label')
                    #dm_slider = aria_slider(min=dm_slider_min, max=dm_slider_max,value=0.0, step=0.01,  aria_label="Dark matter fraction control slider").props('aria-describedby=dm_slider_label label-always')

             
                    
                    with ui.row().classes('w-full justify-center gap-4'):
                        #with ui.column().classes('flex-1 items-center'):
                        #    plot_container_histo = ui.column()
                        #    stats_container_histo = ui.column()
   
                        #with ui.column().classes('flex-1 items-center'):
                        #    plot_container_scatter = ui.column()
                        #    stats_container_scatter = ui.column()
                    
                        
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
                                    with ui.pyplot(figsize=(8, 5)):
                                    
                                        plt.hist(observed_vel, bins=bins, alpha=1.0,color='blue', label='Observed Velocities', rasterized=True)
                                        
                                        if len(v_dm) > 0:
                                            #weights_sim = np.full(len(v_dm), len(observed_vel) / len(v_dm))
                                            plt.hist(v_dm, bins=bins, alpha=0.7, color='green', label=f"Simulated Velocities (v_mean={v_model_mean:.1f} km/s)", rasterized=True)
                                        if len(v_bar) > 0:
                                            #weights_bar = np.full(len(v_bar), len(observed_vel) / len(v_bar))
                                            plt.hist(v_bar, bins=bins, alpha=0.6, color='red', label='Baryonic Component', rasterized=True)

                                        plt.xlim(0, x_max + padding)
                                        #plt.yscale('log')
                                        plt.ylim(0, y_max)
                                        plt.xlabel('Velocity [km/s]')
                                        plt.ylabel('N° of Galaxies')
                                        if cluster_name is not None:
                                            plt.title(f'Galaxy Velocity Distribution ({cluster_name})')
                                        else:
                                            plt.title('Galaxy Velocity Distribution (Cluster)')
                                        plt.legend()
                                        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                                        ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Histogram showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
)

                                with stats_container_histo:
                                    stats_container_histo.clear()
                                    plot_info_box_compact({
    "Histogram discrepancy (χ²)": f"{chi2_hist:.1f}",
    "Normalized to obs χ²": f"{chi2_norm:.2f}",
    "Dark Matter factor (f)": f"{f:.2f}",
    "N simulated galaxies (green)": f"{len(v_dm)}",
    "Simulated mean velocity": f"{v_model_mean:.1f} km/s, σ_sim = {v_model_sigma:.1f} km/s",
    "Peak ratio (green/blue)": f"{peak_model/peak_obs:.2f}",
    "N observed galaxies": f"{N_obs}",
    "Observed mean velocity": f"{v_mean_val:.1f} km/s, σ_obs = {sigma_obs:.1f} km/s",
    "M200, R200, c": f"{M200:.2e} Msun, {R200:.1f} kpc, c={c:.2f}"
})

                                 
                                    


                        
                            

                                with plot_container_scatter:
                                    plot_container_scatter.clear()
                                    #plt.close()
                                    with ui.pyplot(figsize=(8, 6)):
                                
                                    
 
                                        plt.scatter(r_proj_kpc, observed_vel, s=10, color='blue', alpha=0.6, label='Observed Galaxies', rasterized=True)

                                        plt.scatter(r_proj_kpc, v_bar, s=10, color='red', alpha=0.6, label='Baryonic Component', rasterized=True)
                                        plt.scatter(r_proj_kpc, v_dm, s=10, color='green', alpha=0.6, label=f"Simulated Galaxies (DM_mean = {M_dm_mean:.2e} M☉)", rasterized=True)
                                    

  
                                    #r_sorted = np.sort(r_proj_kpc)
                                    #sigma_los_sorted = np.sqrt(G_grav * (m_bary_at_gal + f * M_dm_at_gal) / (3.0 * r_sorted))
                                    #v_upper = v_mean_val + 3 * sigma_los_sorted
                                    #v_lower = v_mean_val - 3 * sigma_los_sorted
                                    #plt.plot(r_sorted, v_upper, 'r--', lw=1.5, label='Caustic envelope')
                                    #plt.plot(r_sorted, v_lower, 'r--', lw=1.5)

   
                                        plt.xlim(0, r_max + r_padding)
                                        plt.ylim(0, ylim_max)


                                        plt.xlabel("Radius [kpc]")
                                        plt.ylabel("Velocity [km/s]")
                                        if cluster_name is not None:
                                            plt.title(f"Phase-space diagram of cluster galaxies ({cluster_name})")
                                        else:
                                            plt.title("Phase-space diagram of cluster galaxies")
                                        plt.legend()
                                        plt.grid(True, linestyle="--", alpha=0.7)
                                        ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the observed velocities of galaxies in the Coma cluster in blue,the baryonic component in red and the simulated velocities in green, which depend on the dark matter fraction set by the slider'
)
                                with stats_container_scatter:
                                    stats_container_scatter.clear()
                                    plot_info_box_compact({
    "Scatter discrepancy (χ²)": f"{chi2_scatter:.1f}",
    "Normalized to obs χ²": f"{chi2_scatter_norm:.2f}",
    "σ_bar (Baryons)": f"{sigma_bar_val:.1f} km/s",
    "σ_total (Baryons+DM)": f"{sigma_dm_val:.1f} km/s",
    "Simulated galaxies (green)": f"{N_sim}",
    "N. galaxies baryonic component (red)": f"{N_bar}",
    "Mean DM mass per galaxy": f"{M_dm_mean:.2e} Msun",
    "Mass ratio baryons/total": f"{mass_ratio:.4e}",
    "Fraction of DM mass": f"{dm_fraction:.2f}",
    "v_max_DM, v_min_DM": f"{v_dm.max():.1f} km/s, {v_dm.min():.1f} km/s",
    "v_max_bar, v_min_bar": f"{v_bar.max():.1f} km/s, {v_bar.min():.1f} km/s"
})



                            except Exception as e:
                                warning_box(f"An error occurred: {e}").classes('text-red-500')
                    
                    
                    
                 
                    
                
                    with ui.row().classes('w-full justify-center gap-4'):
                        with ui.column().classes('flex-1 items-center'):
                            sim_container_obs = ui.column()
                            stats_obs = ui.label()
                        with ui.column().classes('flex-1 items-center'):
                            sim_container_model = ui.column()
                            stats_model = ui.label()


                    N_gal = min(80, len(observed_vel))
                    box_size = 1.0
                    dt = 0.05
                    rng = np.random.default_rng(123)


                    v_mean_obs = float(np.mean(observed_vel))
                    sigma_obs_val = float(np.std(observed_vel))
                    vx_obs = rng.normal(0.0, sigma_obs_val/3000.0, N_gal)
                    vy_obs = rng.normal(0.0, sigma_obs_val/3000.0, N_gal)


                    state = {
    "x_obs": rng.uniform(0.0, box_size, N_gal),
    "y_obs": rng.uniform(0.0, box_size, N_gal),
    "x_model": None,
    "y_model": None,
    "vx_obs": vx_obs.copy(),
    "vy_obs": vy_obs.copy(),
    "vx_model": vx_obs.copy(),
    "vy_model": vy_obs.copy(),
    "rng": rng,
}
                    state["x_model"] = state["x_obs"].copy()
                    state["y_model"] = state["y_obs"].copy()

                    def _reflect_axis(pos, vel, lo=0.0, hi=box_size):
   
                        mask_lo = pos < lo
                        if mask_lo.any():
                            pos[mask_lo] = lo + (lo - pos[mask_lo])
                            vel[mask_lo] *= -1.0
                        mask_hi = pos > hi
                        if mask_hi.any():
                            pos[mask_hi] = hi - (pos[mask_hi] - hi)
                            vel[mask_hi] *= -1.0
                        return pos, vel

                   
                    def update_cluster_points():
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
                        #global cluster_name
                        try:
                            f = float(dm_slider.value)

    
                            r_safe = np.maximum(r_proj_kpc, 1.0)
                            M_dm_at_gal = Mc_nfw_enclosed(r_safe, M200, c, rho_crit)
                            M_total_bary = np.maximum(m_bary_at_gal, 1e-6)
                            M_total_withDM = np.maximum(m_bary_at_gal + f*M_dm_at_gal, 1e-6)

                            sigma_bary = np.sqrt(np.maximum(0.0, G_grav * M_total_bary / (3.0 * r_safe))).mean()
                            sigma_withDM = np.sqrt(np.maximum(0.0, G_grav * M_total_withDM / (3.0 * r_safe))).mean()

                            if f == 0:
                                scale = sigma_bary / sigma_obs_val if sigma_obs_val > 0 else 1.0
                            else:
                                scale = sigma_withDM / sigma_obs_val if sigma_obs_val > 0 else 1.0

                            state["vx_model"][:] = state["vx_obs"] * scale
                            state["vy_model"][:] = state["vy_obs"] * scale

   
                            state["x_obs"] += state["vx_obs"] * dt
                            state["y_obs"] += state["vy_obs"] * dt
                            state["x_model"] += state["vx_model"] * dt
                            state["y_model"] += state["vy_model"] * dt

       
                            state["x_obs"], state["vx_obs"] = _reflect_axis(state["x_obs"], state["vx_obs"])
                            state["y_obs"], state["vy_obs"] = _reflect_axis(state["y_obs"], state["vy_obs"])
                            state["x_model"], state["vx_model"] = _reflect_axis(state["x_model"], state["vx_model"])
                            state["y_model"], state["vy_model"] = _reflect_axis(state["y_model"], state["vy_model"])

       
                            with sim_container_obs:
                            
                                sim_container_obs.clear()
                                plt.close()
                                with ui.pyplot(figsize=(4, 4)):
                                  
                                    plt.scatter(state["x_obs"], state["y_obs"], c="blue", s=20, alpha=0.8, label="Observed")
                                    plt.xlim(0, box_size); plt.ylim(0, box_size)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Observed Galaxies")
                                    plt.legend(loc="upper right")
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the positions and movements of observed galaxies in the Coma cluster '
)
                                plot_info_box_compact({
    "Observed ⟨v⟩": f"{v_mean_obs:.0f} km/s",
    "σ_obs": f"{sigma_obs_val:.0f} km/s"
})

       
                            with sim_container_model:
                                sim_container_model.clear()
                                
                                plt.close()
                                with ui.pyplot(figsize=(4, 4)):
                                    
                                    plt.scatter(state["x_model"], state["y_model"], c="green", s=20, alpha=0.8, label="Simulated")
                                    plt.xlim(0, box_size); plt.ylim(0, box_size)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Simulated Galaxies")
                                    plt.legend(loc="upper right")
                                    ui.element('div').props(
    'role=status aria-live=polite tabindex=0 aria-label=Scatter plot showing the positions and movements of simulated galaxies in the Coma cluster which depend on the dark matter fraction set by the slider'
)
                                plot_info_box_compact({
    "Simulated σ": f"{sigma_withDM:.0f} km/s",
    "DM factor": f"{f:.2f}"
})


                        except Exception as e:
                            accessible_notify(f"Simulator error: {e}", color="red")

                    
                       
                    
                      

                   

                   
                    refresh_cluster_plots()
                    #ui.timer(0.05, refresh_cluster_plots.refresh, once=True)
               


                    ui.timer(dt, update_cluster_points)



                
                    with ui.row().classes('w-full gap-4 justify-start'):

   
                        with ui.column().classes('flex-1 items-center '):
                                        if image_filepath and os.path.exists(image_filepath):
                                            file_name = os.path.basename(image_filepath)
                                            web_path = f'/cluster_img/{file_name}'
                                            aria_image(web_path, f"Image of the selected galaxy cluster {select} ").classes('w-full max-w-xl')
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
                os.path.splitext(select)[0] + ".csv"
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
                  

       
            with ui.tab_panel(four).props('role=tabpanel'):
                with ui.card().classes("p-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Mass & Virial Theorem")

                with ui.element('div').props('role=region tabindex=0 aria-label=Descriptive text about galaxy cluster mass and density activities').classes('text-lg'):
                    paragraph(
                   "Imagine analyzing a **galaxy cluster** to reveal dark matter. "
"\n\n Choose a dataset and complete the missing fields."
"\n\n Click **Run Analysis** to compare **luminous mass** vs **total (virial) mass**.")
                with ui.element('div').props(' ìrole=region tabindex=0 aria-label=Descriptive text about cluster datasets').classes('text-lg'):
                    info_box("**Dataset variables**: Cluster (name), ID (galaxy ID), RAdeg (right ascension in degrees), DEdeg (declination in degrees), RV (radial velocity in km/s), e_RV (velocity error in km/s), q_RV (quality flag),Nint (n.emission/absorbed lines), bmag (apparent magnitude in B band).")
                                        
                    reference_box("**Dataset reference**: Way M.J. et al., *Redshifts in the Southern Abell Redshift Survey Clusters*. I. The Data'").classes('text-base italic')

                with ui.row().classes('w-full mt-4'):
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
""", sanitize=False).props('role=dialog aria-modal=true aria-label=Mathematical explanation of the formulas used in the cluster mass and virial theorem activity')
                            aria_button("Close", "Close the box",on_click=formula_dialog.close)

                        with ui.dialog().props('role="dialog" aria-label="legend-title" aria-describedby="legend-desc"') as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-gray-800 border border-blue-400 shadow-lg'):
                   
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
                            aria_button("Close","Close the box", on_click=legend_dialog.close).classes("mt-4 bg-blue-600 text-white rounded-lg px-4 py-2")

                    
                        with ui.dialog().props('role="dialog" aria-label="units-title" aria-describedby="units-desc"') as units_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-slate-500 border border-green-500 shadow-lg'):
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
                    
                            aria_button("Close", "Close the box",on_click=units_dialog.close).classes("mt-4 bg-green-600 text-white rounded-lg px-4 py-2")

                    
                        aria_button("Info", "Read detailed information about computational steps from data to plots",on_click=safe_click(lambda: [formula_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")])).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        aria_button("📘 Legend", "Read the legend of symbols and units",on_click=legend_dialog.open).classes(
    "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
)
                        aria_button("📐 Units","Read units conversion", on_click=units_dialog.open).classes(
    "bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
)
                ui.label("").classes('h-2')  # Spacer       
             

                cluster_files = [f for f in os.listdir(CLUSTER_DATA_PATH) if f.endswith(('.txt', '.dat', '.csv'))]
                cluster_file_map = get_data_and_images(CLUSTER_DATA_PATH, CLUSTER_IMG_PATH)

                if not cluster_file_map:
                    ui.label("No cluster data files found.").classes('text-red-500')
                else:
                    with ui.row().classes('w-full items-center'):
                        cluster_select = ui.select(cluster_files, label='Select a Cluster Dataset').classes('flex-1 max-w-md items-start text-lg')

                    answer_sigma,answer_r,answer_g2,answer_vel,answer_vel_mean = {}, {},{},{},{}
                    ans_U, ans_Fg, ans_Fc, ans_K, ans_v2 = {}, {}, {}, {}, {}
                    ans_virialLHS, ans_virialRHS, ans_Mcirc, ans_Mvir,answer_LB = {}, {}, {}, {}, {}
                    answer_D, answer_Dmod, answer_Mabs, answer_Lsum,answer_Mlum = {}, {}, {}, {},{}
                    @ui.refreshable
                    def show_cluster_pseudocode():
                        with ui.card().classes("p-6 w-full max-w-4xl bg-gray-900 text-white rounded-xl shadow-xl"):
                            ui.markdown("### 🧮 Cluster Mass Exercise").classes("text-2xl font-bold mb-4 text-blue-300")
                            ui.markdown("Fill in the missing parts in the pseudo-code below:").classes("text-lg mb-4")

                            with ui.column().classes("pseudo-code w-full"):

                                ui.label("0) Load cluster dataset").classes('step')
                                ui.label(f'data = load_data("{cluster_select.value}")')

                                ui.label("1) Orbital dynamics (single light particle around heavy one)").classes('step')
                                ui.label("2) Compute the gravitational potential energy").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("U(r) = ")
                                    ans_U['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("3) Compute the gravitational force").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("F_grav = ")
                                    ans_Fg['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("4) Compute the centripetal force").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("F_cent = ")
                                    ans_Fc['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("5) Equate forces: F_grav = F_cent → derive v²(R)").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("v²(R) = ")
                                    ans_v2['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("6) Kinetic energy of the light particle").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("K = ")
                                    ans_K['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("7) Virial theorem (time-averaged, bound system)").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("2⟨")
                                    ans_virialLHS['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("⟩ + ⟨")
                                    ans_virialRHS['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("⟩ = 0")

                                ui.label("8) Compute circular orbit mass of one particle").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_circ(r) = ")
                                    ans_Mcirc['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("9) Compute the total mass for a system of N particles:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_tot(r) ≈ sum ")
                                    ans_Mvir['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("10) Compute velocity dispersion").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("σ_v² = 1/N * Σ |")
                                    answer_vel['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(" - mean(")
                                    answer_vel_mean['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(")|²")

                                ui.label("11) Compute virial/total mass").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_total(r) = (3 * ")
                                    answer_sigma['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label("² * ")
                                    answer_r['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(") / ")
                                    answer_g2['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("12) Compute cluster distance in pc:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("D_pc = ")
                                    answer_D['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("13) Compute distance modulus:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("dist_mod = 5 * log10(")
                                    answer_Dmod['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(") - 5")

                                ui.label("14) Compute absolute magnitude from apparent magnitude (dataset) and distance:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_abs = ")
                                    answer_Mabs['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(" - dist_mod")

                                ui.label("15) Compute luminosity for each galaxy:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("L_B = 10 ** (-0.4 * (")
                                    answer_LB['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(" - M_sun_B))")

                                ui.label("16) Compute total luminosity:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("L(r) = Σ(")
                                    answer_Lsum['el'] = aria_formula_input().props('dense filled').classes('w-48')
                                    ui.label(") for galaxies with r_i ≤ r")

                                ui.label("17) Compute luminous mass:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_lum(r) = (M/L) * ")
                                    answer_Mlum['el'] = aria_formula_input().props('dense filled').classes('w-48')

                                ui.label("18) Compare mass profiles").classes('step mt-2')
                                ui.label('plot(r, M_lum, label="Luminous Mass")')
                                ui.label('plot(r, M_total, label="Total Mass")')

                            
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
                                
                                df = pd.read_csv(data_filepath, delim_whitespace=True, header=None)
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

                             
                        
                    aria_button("Run Analysis", "Ruon the analysis to make the plots",on_click=check_and_run_cluster)

            with ui.tab_panel(five).props('role=tabpanel'):
                with ui.card().classes("p-4 bg-gradient-to-r from-pink-500 to-red-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cosmic Microwave Background (CMB)")

                with ui.row().classes('w-full gap-4 justify-start'):
                    
                    with ui.column().classes('flex-1 max-w-md items-start '):
                     
                        
                        aria_button('Explore the full Planck CMB Simulator',"Explore the interactive Plank CMB simulator",
          on_click=safe_click(lambda: ui.run_javascript("window.open('https://chrisnorth.github.io/planckapps/Simulator', '_blank')"))).props('outline color=primary')

                        reference_box("""
**Planck Simulator References:**  
- [Chris North Planck Simulator](https://chrisnorth.github.io/planckapps/Simulator)  
- [UCSB Planck Project](https://www.deepspace.ucsb.edu/projects/planck)  
  
""")
                    with ui.column().classes('flex-1 max-w-md items-start'):
                        aria_button('ESA: The Universe According to Planck', "ESA reference website",on_click=safe_click(lambda: ui.run_javascript("window.open('https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck','_blank')"))).props('outline color=accent')
                        reference_box("""

**Images References:**  
- [ESA Planck Overview](https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck)  
- [Planck Image Archive](https://planck.ipac.caltech.edu/page/contacts)  
""")
                with ui.grid(columns=2).classes('w-full'):
                    
                    aria_image('/images/univ_composition.jpg', "Image of the universe composition with dark energy,dark matter and ordinary matter").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/power_spectrum.jpg', "Image of the power spectrum of cosmic microwave background").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/CMB_asymmetry.jpg', "Image of CMB asymmetry in the universe").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    aria_image('/images/Planck_freq.jpg', "Image of the plank frequency spectrum").classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
        
        
        
        
    standard_module_ui("Dark Matter ")





   




if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0",
        port=7860,title='Cosmo-Edu Lab', storage_secret='a-very-secret-and-secure-key-for-sessions', dark=True)