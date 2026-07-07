
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

import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import curve_fit, fsolve, brentq, minimize_scalar
from scipy.signal import savgol_filter
from scipy.special import i0, i1, k0, k1
# Librerie Web/App
import requests
from dotenv import load_dotenv
from fastapi import Request
from groq import Groq
from ipywidgets import interact, FloatSlider
from nicegui import ui, app, run 
# Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
#import asyncio           

_rho_rs_cache = {}
_cluster_calc_cache = {}

load_dotenv()





def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) 
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


HF_API_TOKEN = os.getenv("HF_API_TOKEN")




import threading
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
            ui.label("AI Tutor is thinking...")

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
        ui.notify('Reflection saved!', type='positive')
    else:
        ui.notify('Reflection cannot be empty.', type='warning')




def main_layout(title: str):
    with ui.header(elevated=True).classes('bg-primary text-white items-center justify-between'):
        ui.label('Cosmo-Edu Lab').classes('text-2xl')
        ui.label(title).classes('text-lg')
        with ui.row():
            ui.button('🏠 Home', on_click=lambda: ui.navigate.to('/main'), icon='home')
            ui.button('🔙 Back', on_click=lambda: ui.run_javascript('window.history.back()'), icon='arrow_back')
            if app.storage.user.get('name'):
                ui.button(f"Logout ({app.storage.user.get('name')})",
                    on_click=lambda: (app.storage.user.clear(), ui.navigate.to('/login')),
                    color='negative', icon='logout')


    with ui.left_drawer().classes('bg-gray-100 shadow-lg'):
        ui.label('🔭 Cosmo-Edu Navigation').classes('text-lg p-4 text-blue-600 font-bold')
        ui.button('🏠 Home', on_click=lambda: ui.navigate.to('/main')).classes('w-full')
        ui.button('📚 Cosmology Modules', on_click=lambda: ui.navigate.to('/main')).classes('w-full')
        ui.button('🧭 Physics → Cosmology', on_click=lambda: ui.navigate.to('/physics-links')).classes('w-full')
        ui.button('🧪 Physics Curriculum', on_click=lambda: ui.navigate.to('/physics-program')).classes('w-full')
        ui.button('📝 Reflections', on_click=lambda: ui.navigate.to('/reflections')).classes('w-full')


def standard_module_ui(module_name: str):
    """
    Provides standard UI elements for modules, including AI tutor interaction
    and reflection submission.
    """
    with ui.card().classes('w-full mt-4'):
        ui.label('Engage with the AI Tutor').classes('text-lg font-bold')
        q_input = ui.input(f'Ask a question about {module_name}...').classes('w-full')
        ai_response_container = ui.column().classes('w-full min-h-[50px]')
        ui.button('Ask Question', on_click=lambda: ask_ai(q_input.value, ai_response_container)).classes('mt-2')

    with ui.card().classes('w-full mt-4'):
        ui.label('Your Reflections').classes('text-lg font-bold')
        reflection_input = ui.textarea(f'Write your reflections on {module_name} here...').classes('w-full')
        ui.button('Save Reflection', on_click=lambda: submit_reflection(module_name, reflection_input.value)).classes('mt-2')




@ui.page('/login')
def login_page():
    """Login page for users."""
    def try_login():
        if app_data['users'].get(username.value) == password.value:
            app.storage.user['name'] = username.value
            ui.navigate.to('/main')
        else:
            ui.notify('Invalid username or password', type='negative')

    with ui.card().classes('absolute-center'):
        ui.label('Login').classes('text-2xl font-bold')
        username = ui.input('Username').on('keydown.enter', lambda: password.focus())
        password = ui.input('Password', password=True).on('keydown.enter', try_login)
        ui.button('Login', on_click=try_login)
        ui.link("Don't have an account? Register", '/register').classes('mt-2')

@ui.page('/register')
def register_page():
    """Registration page for new users."""
    def do_register():
        if not new_user.value or not new_pass.value:
            ui.notify('Username and password are required.', type='warning')
            return
        if new_user.value in app_data['users']:
            ui.notify('This username is already taken.', type='negative')
            return
        app_data['users'][new_user.value] = new_pass.value
        save_data()
        ui.notify('User registered! You can now log in.', type='positive')
        ui.navigate.to('/login')

    with ui.card().classes('absolute-center'):
        ui.label('Register New Account').classes('text-2xl font-bold')
        new_user = ui.input('New Username')
        new_pass = ui.input('New Password', password=True)
        ui.button('Register', on_click=do_register)

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
        ui.label(f"Welcome, {app.storage.user.get('name', 'Explorer')}!").classes('text-3xl m-4')


       
        
        ui.label('\"The cosmos is within us. We are made of star-stuff.\" — Carl Sagan').classes('italic text-lg mt-2')

        ui.markdown('Select a module to begin your journey into cosmology.')
        with ui.grid(columns=4).classes('gap-4 m-4 w-full'):
            module_titles = [
                "Fundational Principles", "Measuring the Cosmos", "The Expanding Universe",
                "The Big Bang Model & The CMB", "The Growth of Cosmic Structures", "The Dark Matter and Energy",
                "Inflation & Open Questions", "Cosmology & the Scientific Method"
            ]
            for i, title in enumerate(module_titles, 1):
                with ui.card().classes('w-full hover:shadow-xl transition'):
                    ui.label(f'Module {i}').classes('text-xl font-bold')
                    ui.label(title).classes('text-sm text-gray-600')
                    ui.button(f'Explore Module {i}', on_click=lambda i=i: ui.navigate.to(f'/module{i}')).classes('mt-4 w-full')

       
        ui.button('🧭 Physics → Cosmology Connections', on_click=lambda: ui.navigate.to('/physics-links')).classes('mt-4')
        ui.button('🧪 Physics Curriculum', on_click=lambda: ui.navigate.to('/physics-program')).classes('mt-2')
        ui.button('📝 View Reflections', on_click=lambda: ui.navigate.to('/reflections')).classes('mt-2')



@ui.page('/physics-links')
def physics_links():
    main_layout("Classical Physics → Cosmology Connections")

    table_data = [
        ("1. Foundational principles", "Propagation of light, light intensity, energy conservation", "Light in infinite space, Olbers' paradox, energy density", "/module1"),
        ("2. Measuring the universe", "Waves, frequency, luminosity, inverse-square law, error propagation", "Standard candles, Doppler redshift", "/module2"),
        ("3. The expanding universe", "Doppler effect, velocity-distance relationships, kinetic energy", "Hubble’s law, cosmic acceleration, Dark energy", "/module3"),
        ("4. Big Bang and the CMB", "Thermodynamics, radiation, blackbody spectrum", "Thermal history, CMB, blackbody curves", "/module4"),
        ("5. Growth of cosmic structure", "Newtonian gravity, kinetic theory, pressure, density perturbations", "Jeans instability, collapse, structure growth", "/module5"),
        ("6. Dark Matter and energy", "Circular motion, gravity, energy conservation", "Rotation curves, missing mass, virial theorem", "/module6"),
        ("7. Inflation and open questions", "Geometry, density, exponentials", "Horizon, flatness problems, inflation", "/module7"),
        ("8. Cosmology and scientific method", "Scientific method, model validation, indirect evidence", "Simulation, reasoning, epistemology", "/module8"),
    ]
    rows = [
        {'classical': title, 'topics': classical} for title, classical, cosmo, link in table_data
    ]
    with ui.table(
        columns=[
            {'name': 'classical', 'label': 'Classical Physics', 'field': 'classical', 'sortable': True},
            {'name': 'topics', 'label': 'Cosmology Topics', 'field': 'topics', 'sortable': False},
        ],
        rows=rows
    ).classes('w-full'):
        pass

    with ui.row():
        for title, classical, cosmo, link in table_data:
            ui.button(f"Go to {title}", on_click=lambda l=link: ui.navigate.to(l)).classes('mr-4')


   


@ui.page('/physics-program')
def physics_program():
    main_layout("Physics Curriculum → Cosmology Modules")

    physics_curriculum = [
        ("1st–2nd Year", [
            ("Scalar and vectorial measures and units", "/module2"),
            ("Geometry optics: reflection and refraction", "/module1"),
            ("Thermal phenomena: temperature, heat, thermal balance", "/module4"),
            ("Mechanics: motion, forces, Newton’s laws", "/module5"),
        ]),
        ("3rd–4th Year", [
            ("Relativity: reference systems, principles", "/module7"),
            ("Conservation laws: energy, fluids", "/module6"),
            ("Gravitation: Kepler, Newton, cosmological systems", "/module5"),
            ("Thermodynamics: gas laws, kinetic theory", "/module4"),
            ("Waves: mechanical, interference, diffraction", "/module2"),
            ("Electromagnetism: fields, energy, potential", "/module4"),
        ]),
        ("5th Year", [
            ("Electromagnetism: induction, Maxwell, EM waves", "/module4"),
            ("Micro & macro cosmos: space-time, energy", "/module7"),
            ("Relativity: Einstein’s principles, dilation, contraction", "/module7"),
            ("Quantum light: Planck, photoelectric, De Broglie", "/module4"),
        ]),
    ]

    for year, topics in physics_curriculum:
        with ui.expansion(year, icon='school').classes('w-full'):
            for topic, link in topics:
                if link:
                    ui.button(topic, on_click=lambda l=link: ui.navigate.to(l)).classes('w-full text-left')
                else:
                    ui.label(topic).classes('text-gray-600 ml-4')


@ui.page('/reflections')
def all_reflections():
    main_layout("Student Reflections")

    user = app.storage.user.get('name')
    if user == 'admin':
        ui.label('📒 All reflections (admin view)').classes('text-xl')
        for entry in app_data.get('reflection_log', []):
            ui.markdown(entry)
    else:
        ui.label('🗒️ Your reflections').classes('text-xl')
        for entry in app_data.get('reflection_log', []):
            if f"| {user} |" in entry:
                ui.markdown(entry)



def title(text: str):
    """Titolo principale (blu scuro, grande, bold)"""
    return ui.label(text).classes("title")

def subtitle(text: str):
    """Sottotitolo (blu acceso, medio-grande)"""
    return ui.label(text).classes("subtitle")

def section(text: str):
    """Titolo di sezione (grigio scuro, medio)"""
    return ui.label(text).classes("section")

def paragraph(text: str):
    """Paragrafo standard (18px, giustificato, line spacing 1.7)"""
    return ui.markdown(text).classes("paragraph")

def info_box(text: str):
    """Box informativo (sfondo chiaro, bordo blu, testo scuro)"""
    return ui.markdown(text).classes("info-box")

def warning_box(text: str):
    """Box avviso (arancione)"""
    return ui.markdown(text).classes("warning-box")

def success_box(text: str):
    """Box positivo / conferma (verde)"""
    return ui.markdown(text).classes("success-box")
def reference_box(text: str):
    """Box per reference e bibliografia (grigio elegante, italico)"""
    return ui.markdown(text).classes("reference-box")
def plot_info_box(info: dict, title: str = "📊 Results"):
    """
    Crea un box con le informazioni sotto un grafico.
    - info: dict {etichetta: valore}
    - title: titolo opzionale sopra al box
    """
    with ui.card().classes(
        "p-4 bg-slate-50 border border-slate-300 rounded-lg shadow-md w-full max-w-2xl mt-2"
    ):
        ui.label(title).classes("text-lg font-bold text-slate-700 mb-2")
        for label, value in info.items():
            ui.label(f"{label}: {value}").classes("text-base font-medium text-slate-800")

def title_on_dark(text: str):
    return ui.label(text).classes("text-4xl font-bold text-center mb-4 title-on-dark")



def plot_info_box_compact(info: dict, title: str = None, compact: bool = True):
 
    base_classes = "plot-info-box"
    if compact:
        base_classes += " compact"

   
    with ui.element("div").classes(base_classes):
        if title:
            ui.label(title).classes("text-sm font-semibold text-slate-700").style("margin-bottom:6px;")
    
        for lab, val in info.items():
        
            with ui.element("div").classes("info-row"):
                ui.label(str(lab)).classes("label")
                ui.label(str(val)).classes("value")



@ui.page('/module1')
def module1():
    main_layout("Module 1: Fundational Principles")
   

@ui.page('/module2')
def module2():
    main_layout("Module 2: Measuring the Cosmos")
   
        

@ui.page('/module3')
def module3():
    main_layout("Module 3: The Expanding Universe")
   
        
        

@ui.page('/module4')
def module4():
    main_layout("Module 4: The Big Bang model & The CMB")
    

@ui.page('/module5')
def module5():
    main_layout("Module 5: The Growth of Cosmic Structures")
   





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


def estimate_M200_R200_from_sigma(sigma_obs, rho_crit_local, G=G_grav, R200_min=100.0, R200_max=5000.0):
    def func(R200):
        M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
        return np.sqrt(G * M200 / (3.0 * R200)) - sigma_obs
   
    if func(R200_min) * func(R200_max) > 0:
        raise RuntimeError("Intervals R200_min,R200_max without elements")
    R200 = brentq(func, R200_min, R200_max)
    M200 = (4.0/3.0) * np.pi * 200.0 * rho_crit_local * R200**3
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


@ui.page('/module6')
def module6():
    ui.add_head_html('<script src="https://cdn.tailwindcss.com"></script>')
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
    if (isAudioActive) return;
    await Tone.start();
    synth = new Tone.Synth().toDestination();
    isAudioActive = true;
    console.log("✅ Audio activated");
    document.getElementById('audio-button').innerText = 'Audio activated';
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


function playSimVelCurve() { playSeries(window.vSimSeries, 400); }
function playSimVelMean() {
    if (!isAudioActive || !synth) return;
    if (!window.vSimMean) return;
    const f = 200 + Math.log10(window.vSimMean + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}


function playObservedVelCurve() { playSeries(window.vObsSeries, 400); }
function playObservedVelMean() {
    if (!isAudioActive || !synth) return;
    if (!window.vObsMean) return;
    const f = 200 + Math.log10(window.vObsMean + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}


function playBaryonicVelCurve() { playSeries(window.vBarySeries, 400); }
function playBaryonicVelMean() {
    if (!isAudioActive || !synth) return;
    if (!window.vBarMean) return;
    const f = 200 + Math.log10(window.vBarMean + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}





function playSimSigmaCurve() { playSeries(window.sigmaSimSeries, 400); }
function playSimSigmaMean() {
    if (!isAudioActive || !synth) return;
    if (!window.sigmaSimVal) return;
    const f = 200 + Math.log10(window.sigmaSimVal + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}


function playObservedSigmaCurve() {
    playSeries(window.sigmaObsSeries, 400);
}
function playObservedSigmaMean() {
    if (!isAudioActive || !synth) return;
    if (!window.sigmaObsVal) return;
    const f = 200 + Math.log10(window.sigmaObsVal + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}


function playBaryonicSigmaCurve() {
    playSeries(window.sigmaBarSeries, 400);
}
function playBaryonicSigmaMean() {
    if (!isAudioActive || !synth) return;
    if (!window.sigmaBarVal) return;
    const f = 200 + Math.log10(window.sigmaBarVal + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}
function playDifferenceVelCurve() { playSeries(window.vDiffSeries, 400); }
function playDifferenceVelMean() {
    if (!isAudioActive || !synth) return;
    if (!window.vDiffMean && window.vDiffMean !== 0) return;
    const f = 200 + Math.log10(window.vDiffMean + 1) * 600;
    synth.triggerAttackRelease(f, "8n");
}

function playDifferenceSigmaCurve() {
 
    playSeries(window.sigmaDiffSeries, 400); 
}

function playDifferenceSigmaMean() {
    if (!isAudioActive || !synth) return;
    if (!window.sigmaDiffMean && window.sigmaDiffMean !== 0) return; 
   
    const f = 200 + Math.log10(window.sigmaDiffMean + 1) * 600; 
    synth.triggerAttackRelease(f, "8n");
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





    main_layout("Module 6: The Dark Matter and Energy")

    with ui.column().classes('w-full p-4 gap-4'):
        with ui.tabs().classes('w-full') as tabs:
            one, two, three, four,five = ui.tab('Galaxy rotation curve'), ui.tab('Galaxy mass & DM'), ui.tab('Cluster velocity distribution'), ui.tab('Cluster mass & DM'),ui.tab('CMB')

        with ui.tab_panels(tabs, value=one).classes('w-full'):

            with ui.tab_panel(one):
                with ui.card().classes("p-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Rotation Curve")
                #ui.markdown("Galaxy Rotation Curve").classes('text-2xl font-bold')
                with ui.element('div').classes('text-lg'):
                    paragraph(
                    "Explore the rotation curve of the NGC3198 galaxy."
                    "\n\n The **red curve** shows the **baryonic prediction** with a Keplerian-like trend (initial rise followed by a decrease). "
                    "\n\n The **green curve** simulates the addition of a **dark matter halo** to flatten the curve and match observations."
                    "\n\n**Slider**: 0 = total baryonic curve, 1 = total curve with full dark matter halo contribution.")
                with ui.element('div').classes('text-lg'):
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
<span class="math">\( \chi^2 = \sum_i \left(\tfrac{v_{\mathrm{obs}}(r_i) - v_{\mathrm{tot,sim}}(r_i)}{\sigma_i}\right)^2, \;\; \chi^2_{\mathrm{dof}} = \tfrac{\chi^2}{N_{\mathrm{obs}} - N_{\mathrm{params}}} \)</span></p>

<p><b>Plot:</b>  
<ul>
<li>X-axis: radius (data)</li>
<li>Y-axis: <span class="math">\( v_{\mathrm{bar}} \)</span> (red), <span class="math">\( v_{\mathrm{obs}} \)</span> (blue with grey error bars), <span class="math">\( v_{\mathrm{tot,sim}} \)</span> (green)</li>
</ul>
</p>

        """, sanitize=False)

                        ui.button("Close", on_click=velocity_dialog.close)

   
                ui.button("Info", on_click=lambda: [velocity_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                ui.html('''
<div class="flex space-x-2">
  <button id="audio-button" 
            onclick="initAudio()" 
            class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Activate Audio
    </button>
  <button onclick="playObservedVelMean()" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-1 rounded">▶ Observed Vel (mean)</button>
  <button onclick="playObservedVelCurve()" class="bg-blue-300 hover:bg-blue-500 text-white px-3 py-1 rounded">▶ Observed Vel (curve)</button>
  <button onclick="playBaryonicVelMean()" class="bg-red-500 hover:bg-red-700 text-white px-3 py-1 rounded">▶ Baryonic Vel (mean)</button>
  <button onclick="playBaryonicVelCurve()" class="bg-red-300 hover:bg-red-500 text-white px-3 py-1 rounded">▶ Baryonic Vel (curve)</button>
  <button onclick="playSimVelMean()" class="bg-green-500 hover:bg-green-700 text-white px-3 py-1 rounded">▶ Simulated Vel (mean)</button>
  <button onclick="playSimVelCurve()" class="bg-green-300 hover:bg-green-500 text-white px-3 py-1 rounded">▶ Simulated Vel (curve)</button>
  <button onclick="playDifferenceVelMean()" class="bg-yellow-500 hover:bg-yellow-700 text-black px-3 py-1 rounded">▶ Difference (mean)</button>
  <button onclick="playDifferenceVelCurve()" class="bg-yellow-300 hover:bg-yellow-500 text-black px-3 py-1 rounded">▶ Difference (curve)</button>
</div>
''', sanitize=False)

                

                #alpha_min,alpha_max=0.0,1.0
                alpha_slider = ui.slider(min=0.0, max=2.0, value=0.0, step=0.01).props('label-always')
               
            
               
                try:
                    data_obs = pd.read_csv(os.path.join(DATA_DIR, "NGC3198.txt"), comment='#', sep=r'\s+', header=0, engine='python')
                
                    r_ngc = pd.to_numeric(data_obs['Rad'], errors='coerce').values
                    v_obs_ngc = pd.to_numeric(data_obs['Vobs'], errors='coerce').values
                    v_gas_ngc = pd.to_numeric(data_obs['Vgas'], errors='coerce').values
                    v_disk_ngc = pd.to_numeric(data_obs['Vdisk'], errors='coerce').values
                    v_bul_ngc = pd.to_numeric(data_obs['Vbul'], errors='coerce').values
                    v_err_ngc = pd.to_numeric(data_obs['errV'], errors='coerce').values

                    r_match = r_ngc[-1]
                   
                    DATA_LOADED = True
                   
                except FileNotFoundError:
                    ui.label("Error: File not found.").classes('text-red-500')
                    DATA_LOADED = False
                    
              
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
                                ui.button("Add χ² point", on_click=lambda: add_chi2_point())
                                ui.button("Refresh χ² plot", on_click=lambda: refresh_chi2_plot())

                               

                        
                with ui.card().classes("w-full mt-4 p-4 border-2 border-dashed border-blue-300"):
                    subtitle("Parabolic interpolation and compute χ² minimization")
                    paragraph("Use the formula below to compute the χ² minimum of a parabola defined by three points (a, f(a)), (b, f(b)), (c, f(c)). "
                              "\n\n 1a) Input only 3 values for dark matter mass points (x-coordinates),"
                              "\n\n 1b) Press 'compute χ² and minimum' button to calculate the χ² at those points and check the minimum on the plot."
                              "\n\n 2a) Input the dark matter values (a, b, c) and their corresponding χ² values f(a), f(b), f(c) "
                              "\n\n 2b) Press 'compute minimum' button to calculate the minimum.")
                              
                    with ui.row().classes("items-start justify-center gap-12"):
        
        
                        with ui.column().classes("w-1/2 items-center"):
                            ui.html(r"""
<p style="font-family:monospace; background:#e0f0ff; padding:10px; border:1px solid #88c; border-radius:8px; color:#111;">

<b>Formula:</b><br>
$$
x_{min} = \frac{1}{2} \cdot \frac{ (a^2-b^2) f(c) + (b^2-c^2) f(a) + (c^2-a^2) f(b) }{ (a-b) f(c) + (b-c) f(a) + (c-a) f(b) }
$$
</p>""", sanitize=False)
                          
               
                        with ui.column().classes("w-1/3 gap-2"):
                            with ui.row().classes("gap-2"):
                                input_a = ui.number(label='Mass Point 1 (x₁)', format='%.2e').classes('flex-1')
                                input_b = ui.number(label='Mass Point 2 (x₂)', format='%.2e').classes('flex-1')
                                input_c = ui.number(label='Mass Point 3 (x₃)', format='%.2e').classes('flex-1')
                            with ui.row().classes("gap-2"):
                                fa_in = ui.number(label="f(a) = χ²(a)", format="%.6g").classes("flex-1")
                                fb_in = ui.number(label="f(b) = χ²(b)", format="%.6g").classes("flex-1")
                                fc_in = ui.number(label="f(c) = χ²(c)", format="%.6g").classes("flex-1")
                            result_label = ui.label("Result χ² minimum: ---").classes("text-green-600 font-bold mt-2")
                            
                    with ui.row().classes("justify-center gap-4 mt-6"):
                        ui.button("Compute χ² and minimum", on_click=lambda: initialize_parabolic_points())
                        #ui.button("Perform Parabolic Step", on_click=lambda: perform_parabolic_step())
                        ui.button("Compute minimum", on_click=lambda: compute_minimum_formula())

                 
                with ui.row().classes("gap-4 mt-4"):
                    points_display
                    history_display
  
                    def update_galaxy_rotation_plot():
                        if not DATA_LOADED:
                            return
                        
                           
                        f = float(alpha_slider.value)
                        
                        
                            
                            #rho_s_dynamic, rs_dynamic, alpha_dm, beta_dm, gamma_dm = dm_params_from_slider(f)
                        G_grav = 4.30091e-6
                        v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 + np.maximum(0, v_disk_ngc)**2 + np.maximum(0, v_bul_ngc)**2)
                        m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                        m_total    = (v_obs_ngc**2 * r_ngc) / G_grav
                    
                          
                        rho_s, r_s, M200 = get_rhos_rs_from_observed_matching(r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)

                        M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                         
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
                        idx_r = np.searchsorted(r_ngc, r_ref)
                        dm_frac_r = 100 * f * M_dm_grid[idx_r] / (m_baryonic[idx_r] + f * M_dm_grid[idx_r])
                        residui = (vobs_use - vmodel_use) / verr_use
                        v_gas_frac = 100 * np.mean(v_gas_ngc**2 / v_total_curve**2)
                        v_disk_frac = 100 * np.mean(v_disk_ngc**2 / v_total_curve**2)
                        v_bul_frac = 100 * np.mean(v_bul_ngc**2 / v_total_curve**2)





                        with plot_container:
                            plot_container.clear()
                            with ui.pyplot(figsize=(8, 5), close=False):
                                ('all')
                                
                                plt.plot(r_ngc, v_baryonic, color='red', linewidth=2, label='Keplerian velocity')
                                plt.plot(r_ngc, v_total_curve, linewidth=2, color='green', label=f"Simulated velocity (DM mass: {M_dm_tot:.2e} M☉, {dm_fraction:.1f}% of visible mass)")
                                plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', markersize=4, color='blue', ecolor='gray', capsize=2, label='Observed velocity', zorder=5)
                            
                                plt.xlabel('Radius (kpc)')
                                plt.ylabel('Rotation Speed (km/s)')
                                plt.title('Galaxy Rotation Curve: NGC3198')
                                plt.ylim(0, max(300, 1.1 * np.max(v_obs_ngc + v_err_ngc)))
                                plt.xlim(0, 50)
                                plt.grid(True)
                                plt.legend(loc='upper right', fontsize='small')
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
                        if not DATA_LOADED:
                            return

                        f = float(alpha_slider.value)
                        v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 +
                         np.maximum(0, v_disk_ngc)**2 +
                         np.maximum(0, v_bul_ngc)**2)
                        m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                        m_total_obs = (v_obs_ngc**2 * r_ngc) / G_grav

                        rho_s, r_s, _ = get_rhos_rs_from_observed_matching(
        r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                        M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                        m_total_model = m_baryonic + f * M_dm_grid
                        y_min_mass = 0
                        y_max_mass = max(np.max(m_total_obs), np.max(m_baryonic)*2) * 1.1
                        x_min_mass = r_ngc.min() * 0.9
                        x_max_mass = r_ngc.max() * 1.1
                        with mass_plot_container:
                            mass_plot_container.clear()
                            with ui.pyplot(figsize=(8, 5), close=False):
                           
                               
                                plt.plot(r_ngc, m_baryonic, "r-", lw=2, label="Baryonic mass")
                                plt.plot(r_ngc, m_total_obs, "b-", lw=2, label="Total mass")
                                plt.plot(r_ngc, m_total_model, "g-", lw=2, label=f"Simulated mass DM: {np.max(m_total_model):.2e} M☉")
                             
                                plt.xlim(x_min_mass, x_max_mass)
                                plt.ylim(y_min_mass, y_max_mass)

                                plt.xlabel("Radius (kpc)")
                                plt.ylabel("Mass ($M_\\odot$)")
                                plt.title("Mass vs radius")
                                plt.legend()
                                plt.grid(True)
                            plot_info_box_compact({"Visible baryonic mass": f"{np.max(m_baryonic):.2e} M☉",
        "Total mass": f"{np.max(m_total_obs):.2e} M☉",
        f"Simulated mass DM (α={f:.2f})": f"{np.max(m_total_model):.2e} M☉"
    })
                    def update_morphology_plot():
                            if not DATA_LOADED:
                                return

                            f = float(alpha_slider.value) 

                     
                            v_baryonic = np.sqrt(np.maximum(0, v_gas_ngc)**2 + np.maximum(0, v_disk_ngc)**2 + np.maximum(0, v_bul_ngc)**2)
                            m_baryonic = (v_baryonic**2 * r_ngc) / G_grav
                            M_vis_tot = np.max(m_baryonic)

                            rho_s, r_s, _ = get_rhos_rs_from_observed_matching(r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                            M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
                            M_dm_tot = np.max(M_dm_grid) * f

   
                            M_bulge = np.max((v_bul_ngc**2 * r_ngc) / G_grav)
                            M_disk  = np.max((v_disk_ngc**2 * r_ngc) / G_grav)

   
                            R_bulge = np.average(r_ngc, weights=np.maximum(0, v_bul_ngc**2)) if np.any(v_bul_ngc > 0) else 1
                            R_disk  = np.average(r_ngc, weights=np.maximum(0, v_disk_ngc**2)) if np.any(v_disk_ngc > 0) else 5
                            R_halo  = np.sqrt(M_dm_tot / (M_vis_tot+1e-12)) * np.max(r_ngc) if M_dm_tot > 0 else 0
                        

                            with morph_plot_container:
                                morph_plot_container.clear()
                                with ui.pyplot(figsize=(5,5), close=False):
                                    ('all')
                                    
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
                                    ax.set_title("Galaxy structure from above")
                                    ax.legend()

                                plot_info_box_compact({
    "Disk Mass": f"{M_disk:.2e} M☉",
    "DM Mass": f"{M_dm_tot:.2e} M☉"
})
                 
                    chi2_points = []
                 
                    formula_pts = {"a": None, "b": None, "c": None, "fa": None, "fb": None, "fc": None}
                 

                
                    rho_s_init, r_s_init, _ = get_rhos_rs_from_observed_matching(
                        r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                    M_dm_grid_init = M_nfw_enclosed(r_ngc, rho_s=rho_s_init, r_s=r_s_init)
                    dm_max_possible = np.max(M_dm_grid_init) * 2  
                    chi2_max_possible = 10  

                    x_min_chi = 0
                    x_max_chi = dm_max_possible * 1.05
                    y_min_chi = 0
                    y_max_chi = chi2_max_possible

                  
                    unscaled_M_dm_grid_cache = {}

                    def chi2_function_for_minimization(M_dm_tot):
                        galaxy_name = 'NGC3198' 
                        if galaxy_name not in unscaled_M_dm_grid_cache:
                            rho_s, r_s, _ = get_rhos_rs_from_observed_matching(
                                r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
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
                        f = float(alpha_slider.value)
                        rho_s, r_s, _ = get_rhos_rs_from_observed_matching(
                            r_ngc, v_obs_ngc, v_gas_ngc, v_disk_ngc, v_bul_ngc, r_match=r_match)
                        M_dm_grid = M_nfw_enclosed(r_ngc, rho_s=rho_s, r_s=r_s)
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
                            coeffs = np.polyfit(xs, ys, 2)
                            xmin = -coeffs[1] / (2*coeffs[0])  
                            ymin = np.polyval(coeffs, xmin)
                            result_label.set_text(f"x_min ≈ {xmin:.3e}, χ² ≈ {ymin:.4f}")

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
                            ui.notify("Insert 3 values for DM masses", type='negative')
                            return

                        sorted_m = sorted(masses)
                        if abs(sorted_m[2]-sorted_m[0])/(abs(sorted_m[0])+1e-12)<1e-6:
                            ui.notify("Values are too close", type='warning')
                            return

                        points = [(m, chi2_function_for_minimization(m)) for m in sorted(masses)]
                        parabolic_state["points"] = points
                        chi2_points.clear()
                        chi2_points.extend(points)

                        ui.notify(" Points initialized. Ready for χ² computation.", type='positive')
                        result_label.set_text("Results χ² minimum: ---")

                        update_displays.refresh()
                        update_all_plots.refresh()
                        plot_chi2_user_curve()

                    def perform_parabolic_step():
                        if len(parabolic_state["points"]) != 3:
                            ui.notify("Initialize first!", type='warning')
                            return

                        (x1,y1),(x2,y2),(x3,y3) = parabolic_state["points"]

                       
                        num = (x2-x1)**2*(y2-y3) - (x2-x3)**2*(y2-y1)
                        den = (x2-x1)*(y2-y3) - (x2-x3)*(y2-y1)
                        if abs(den)<1e-15:
                            ui.notify("Collinear points are not valid!", type='negative')
                            return

                        x_new = x2 - 0.5*num/den
                        y_new = chi2_function_for_minimization(x_new)
                        if not np.isfinite(y_new):
                            ui.notify(f"Value χ² not valid for x={x_new:.2e}.", type='negative')
                            return

                        parabolic_state["iteration"] += 1
                        parabolic_state["plot_points"].append((x_new, y_new))

                        all_four_points = parabolic_state["points"] + [(x_new,y_new)]
                        all_four_points.sort(key=lambda p: p[1])  # ordina per chi²
                        point_to_discard = all_four_points.pop()    # scarta il maggiore
                        parabolic_state["points"] = all_four_points

                        history_entry = (f"**Step {parabolic_state['iteration']}**: "
                                        f"New ({x_new:.3e}, χ²={y_new:.4f}), "
                                        f"Discarded ({point_to_discard[0]:.3e}, χ²={point_to_discard[1]:.4f})")
                        parabolic_state["history"].insert(0, history_entry)
                    
                        ui.notify("Brent minimization step performed.", type='positive')
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
                        with chi2_plot_container:
                            chi2_plot_container.clear()
                            with ui.pyplot(figsize=(8,5), close=False):
                                ax = plt.gca()

                              
                                if chi2_points:
                                    xs_slider, ys_slider = zip(*chi2_points)
                                    ax.scatter(xs_slider, ys_slider, s=40, c="blue", alpha=0.6, label="Points")

                                    if len(chi2_points) == 3:
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
                                        ax.scatter([xmin_slider], [ymin_slider], c='blue', s=140, marker='*', label="Min ")
                                       
                                        result_label.set_text(f"x_min ≈ {xmin_slider:.3e}, χ² ≈ {ymin_slider:.4f}")

                                if parabolic_state["plot_points"]:
                                    px_new, py_new = zip(*parabolic_state["plot_points"])
                                    ax.scatter(px_new, py_new, c="black", marker='X', s=120, label="New point", zorder=9, edgecolors='black')

                                    all_points = parabolic_state["points"] + list(zip(px_new, py_new))
                                    if len(all_points) >= 3:
                                        xs_alg, ys_alg = zip(*all_points[-3:])
                                        coeffs_alg = np.polyfit(xs_alg, ys_alg, 2)
                                        x_fit_alg = np.linspace(min(xs_alg), max(xs_alg), 400)
                                        y_fit_alg = np.polyval(coeffs_alg, x_fit_alg)
                                        ax.plot(x_fit_alg, y_fit_alg, "r--", lw=2, label="Algorithm Fit", zorder=8)
                           
                                        x_lo_alg, x_hi_alg = min(xs_alg), max(xs_alg)
                                        x_fit_alg = np.linspace(x_lo_alg - 0.1*(x_hi_alg-x_lo_alg), x_hi_alg + 0.1*(x_hi_alg-x_lo_alg), 400)
                                        y_fit_alg = np.polyval(coeffs_alg, x_fit_alg)
                                        ymin_index_alg = np.argmin(y_fit_alg)
                                        xmin_alg = x_fit_alg[ymin_index_alg]
                                        ymin_alg = y_fit_alg[ymin_index_alg]

                                        ax.scatter([xmin_alg], [ymin_alg], c='red', s=140, marker='*', label="New Min", zorder=11)
                                    
                                        result_label.set_text(f"x_min ≈ {xmin_alg:.3e}, χ² ≈ {ymin_alg:.4f}")

                               
                                all_ys = []
                                if chi2_points: all_ys += [p[1] for p in chi2_points]
                                if parabolic_state["points"]: all_ys += [p[1] for p in parabolic_state["points"]]
                                if parabolic_state["plot_points"]: all_ys += [p[1] for p in parabolic_state["plot_points"]]
                                y_max_plot = y_max_chi if not all_ys else max(y_max_chi, max(all_ys)*1.1)

                                ax.set_xlim(x_min_chi, x_max_chi)
                                ax.set_ylim(y_min_chi, y_max_plot)
                                ax.set_xlabel("DM mass ($M_\\odot$)")
                                ax.set_ylabel(r"$\chi^2$/dof")
                                ax.grid(True)
                                handles, labels = ax.get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                if by_label: ax.legend(by_label.values(), by_label.keys())

                  
                    def refresh_formula_from_inputs():
                        formula_pts["a"] = input_a.value
                        formula_pts["b"] = input_b.value
                        formula_pts["c"] = input_c.value
                        formula_pts["fa"] = fa_in.value
                        formula_pts["fb"] = fb_in.value
                        formula_pts["fc"] = fc_in.value
                        plot_chi2_user_curve()

                    def compute_minimum_formula():
                        a,b,c = formula_pts["a"], formula_pts["b"], formula_pts["c"]
                        fa,fb,fc = formula_pts["fa"], formula_pts["fb"], formula_pts["fc"]
                        if None in [a,b,c,fa,fb,fc]:
                            result_label.set_text("Insert all the values before computation!")
                            return
                        num = (a**2 - b**2)*fc + (b**2 - c**2)*fa + (c**2 - a**2)*fb
                        den = (a-b)*fc + (b-c)*fa + (c-a)*fb
                        if not np.isfinite(den) or abs(den)<1e-30:
                            result_label.set_text("Error:denominator is too small.")
                            return
                        xmin = 0.5*num/den
                        result_label.set_text(f"x_min ≈ {xmin:.3e}")

                        coeffs = np.polyfit([a,b,c], [fa,fb,fc], 2)
                        x_lo, x_hi = min(a,b,c), max(a,b,c)
                        if x_lo <= 0: x_fit = np.linspace(x_lo*0.8, x_hi*1.2, 400)
                        else: x_fit = np.logspace(np.log10(x_lo*0.8), np.log10(x_hi*1.2), 400)
                        y_fit = np.polyval(coeffs, x_fit)
                        ymin = np.polyval(coeffs, xmin)

                        with chi2_plot_container:
                            chi2_plot_container.clear()
                            with ui.pyplot(figsize=(8,5), close=False):
                                ax = plt.gca()
                                if chi2_points:
                                    xs, ys = zip(*chi2_points)
                                    ax.scatter(xs, ys, s=30, c="blue", alpha=0.3, label="Points")
                                ax.scatter([a],[fa], s=100, c="black", marker='o', label="f(a)")
                                ax.scatter([b],[fb], s=100, c="black", marker='o', label="f(b)")
                                ax.scatter([c],[fc], s=100, c="black", marker='o', label="f(c)")
                                ax.plot(x_fit, y_fit, 'r-', lw=2, label="Parabolic line")
                                ax.axvline(xmin, color='green', linestyle=':', lw=2, label=f"x_min ≈ {xmin:.3e}")
                                ax.scatter([xmin], [ymin], c='green', s=140, marker='*', zorder=10)
                                try: ax.set_xscale('log') if min([a,b,c])>0 else ax.set_xscale('linear')
                                except: ax.set_xscale('linear')
                                ax.set_xlabel("DM mass ($M_\\odot$)")
                                ax.set_ylabel(r"$\chi^2$/dof")
                                ax.grid(True)
                                handles, labels = ax.get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                if by_label: ax.legend(by_label.values(), by_label.keys())

              
                    fa_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    fb_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    fc_in.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_a.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_b.on('update:model-value', lambda e: refresh_formula_from_inputs())
                    input_c.on('update:model-value', lambda e: refresh_formula_from_inputs())

               
                    refresh_formula_from_inputs()
                    def refresh_chi2_plot():
                        chi2_points.clear()
                        parabolic_state["points"].clear()
                        parabolic_state["plot_points"].clear()
                        parabolic_state["history"].clear()
                        parabolic_state["iteration"] = 0
                        result_label.set_text("Results χ² minimum: ---")
                        update_all_plots.refresh()

                    @ui.refreshable
                    def update_all_plots():
                        plt.close('all')
                        update_galaxy_rotation_plot()
                        update_mass_plot()
                        plot_chi2_user_curve()
                        update_morphology_plot()
                    
                    alpha_slider.on('update:model-value', update_all_plots.refresh)
          
                    update_all_plots()
                  

               
                with ui.row().classes('w-full gap-4 justify-start'):
                    with ui.column().classes('flex-1 items-center '):
                        ui.image(os.path.join(GALAXY_IMG_PATH, "NGC3198.jpg")).classes("w-full max-w-xl")

                        reference_box("""
**Image reference:**  
- [ESA Hubble](https://esahubble.org/)
""").classes('w-full text-center text-base italic mt-2')

                    with ui.column().classes('flex-1 items-center '):
                        table_filepath = os.path.join(GALAXY_TABLES_PATH, "NGC3198.csv")

                        if os.path.exists(table_filepath):
                            df = pd.read_csv(table_filepath)
                            ui.table.from_pandas(df).classes('w-full max-w-xl')
                            reference_box("""
**Galaxy information reference:**  
- [Wikipedia: NGC3198 Galaxy](https://en.wikipedia.org/wiki/NGC_3198)
""").classes('w-full text-center text-base italic mt-2')
                        else:
                            ui.label("Table not found").classes('text-red-500')

     
            with ui.tab_panel(two):
                with ui.card().classes("p-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Galaxy Mass ")
                with ui.element('div').classes('text-lg'):
                    paragraph(
    "You are an astrophysicist investigating the **presence of dark matter** in a galaxy. "
    "\n\n Choose a dataset and complete the missing fields with the correct formulas. "
    "\n\n Click **Run Analysis** to compare the baryonic mass with the total mass and the observed velocity with the Keplerian-like velocity.")
                with ui.element('div').classes('text-lg'):
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
        """, sanitize=False)
                            ui.button("Close", on_click=baryonic_dialog.close)
                        with ui.dialog() as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-gray-800 border border-blue-400 shadow-lg'):
                                ui.markdown("**Legend of Symbols and Units**").classes('!text-xl !text-blue-300 !-mt-2')
                                ui.markdown("""
- `Rad` = Radius [kpc]  
- `V_obs` = Observed rotation velocity [km/s]  
- `V_gas, V_disk, V_bul` = velocity contributions from gas, stellar disk, bulge [km/s]
- `G` = 4.30091×10⁻⁶ kpc·(km/s)²·M☉⁻¹  
- `M_baryonic` = Baryonic mass [M☉]
- `M_total` = Total mass (from observations)  [M☉]
- `V_baryonic` = Baryonic velocity (Keplerian-like prediction) [km/s]
""").classes('text-lg')
                            ui.button("Close", on_click=legend_dialog.close).classes("mt-4 bg-blue-600 text-white rounded-lg px-4 py-2")

                    
                        with ui.dialog() as units_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-slate-500 border border-green-500 shadow-lg'):
                                ui.markdown("**Units conversion**").classes('!text-xl !text-green-300 !-mt-2')
                                ui.markdown("""
- `1 kpc` = 3.086 × 10¹⁶ m = 3.26 × 10³ light-years  
- `1 pc` = 3.086 × 10¹³ km = 3.26 light-years  
- `1 Mpc` = 10⁶ pc = 3.086 × 10¹⁹ km  
- `1 km/s` = 3.6 × 10³ km/h = 10³ m/s  
- `1 M☉` = 1.989 × 10³⁰ kg (solar mass)  
- `1 L☉` = 3.828 × 10²⁶ W (solar luminosity)  
 """).classes('text-lg')
                            ui.button("Close", on_click=units_dialog.close).classes("mt-4 bg-green-600 text-white rounded-lg px-4 py-2")

                    
   
                        ui.button("Info", on_click=lambda: [baryonic_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        ui.button("📘 Legend", on_click=legend_dialog.open).classes(
    "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
)
                        ui.button("📐 Units", on_click=units_dialog.open).classes(
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
                                    answer_vgas['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("^2 + ")
                                    answer_vdisk['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("^2 + ")
                                    answer_vbulge['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("^2 )")
                                
                                ui.label("\n3) Compute luminous mass from baryonic velocity").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_baryonic = ( ")
                                    answer_vb['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("^2 * ")
                                    answer_rad['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(" ) / ")
                                    answer_g['el'] = ui.input(label="").props('dense filled').classes('w-48')


                                
                                ui.label("\n4) Compute total mass from observations (data)").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_total    = ( ")
                                    answer_vo['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("^2 * ")
                                    answer_rad['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(" ) / ")
                                    answer_g['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("\n5) Compute the dark matter mass").classes('step')
                                with ui.row().classes('items-center text-lg'):
                                    ui.label("M_DM    =  ")
                                    answer_Mtot['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("-")
                                    answer_Mbar['el'] = ui.input(label="").props('dense filled').classes('w-48')
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
                                                ('all')
                                              
                                                plt.errorbar(r_ngc, v_obs_ngc, yerr=v_err_ngc, fmt='o', ms=4, color='blue', ecolor='lightblue', capsize=2, label=f'Observed ({selected_file})', zorder=5)
                                                plt.plot(r_ngc, v_baryonic, color='red', lw=2, label='Baryonic Velocity')
                                                plt.xlabel('Radius (kpc)'); plt.ylabel('Velocity (km/s)'); plt.title(f'{os.path.splitext(selected_file)[0]} Rotation Curve')
                                                plt.ylim(0, max(np.nanmax(v_obs_ngc+v_err_ngc), np.nanmax(v_baryonic))*1.1); plt.xlim(0, np.nanmax(r_ngc)*1.1); plt.grid(True); plt.legend()
                                        with ui.column().classes('flex-1 items-center'):
                                            with ui.pyplot(figsize=(8, 6)):
                                                ('all')
                                                plt.plot(r_ngc, m_baryonic/1e9, color='red', lw=2, label='Baryonic Mass')
                                                plt.plot(r_ngc, m_total/1e9,    color='blue', lw=2, label='Total Mass')
                                                plt.xlabel('Radius (kpc)'); plt.ylabel(' Mass (10^9 M☉)'); plt.title(f'{os.path.splitext(selected_file)[0]} Enclosed Mass')
                                                max_m = np.nanmax([np.nanmax(m_total/1e9), np.nanmax(m_baryonic/1e9)])
                                                plt.ylim(0, max_m*1.2 if not np.isnan(max_m) and max_m>0 else 500); plt.xlim(0, np.nanmax(r_ngc)*1.1 if not np.isnan(np.nanmax(r_ngc)) else 50); plt.grid(True); plt.legend()

                                
                                    if image_filepath and os.path.exists(image_filepath):
                                        with ui.row().classes('w-full gap-4 justify-start'):
                   
            
                                            with ui.column().classes('flex-1 items-center'):
                                                ui.image(image_filepath).classes('w-full max-w-xl')
                                                reference_box("""
**Image reference:**  
- [ESA Hubble](https://esahubble.org/)
""").classes('w-full text-center text-base italic mt-2')

           
                                            with ui.column().classes('flex-1 items-center '):
                                                table_filepath = os.path.join(
                    "/workspaces/Cosmo-Edu_Lab/App/galaxy_tables",
                    os.path.splitext(selected_file)[0] + ".csv"
                )
                                                if os.path.exists(table_filepath):
                                                    df_table = pd.read_csv(table_filepath)
                                                    ui.table.from_pandas(df_table).classes('w-full max-w-xl')
                                                    reference_box("""
**Galaxy information references:**  
- [The Sky Live](https://theskylive.com/sky)  
- [Wikipedia: List of NGC objects](https://en.wikipedia.org/wiki/List_of_NGC_objects)
""").classes('w-full text-center text-base italic mt-2')
                                                else:
                                                    ui.label("Table not found").classes('text-red-500')
                        except Exception as e:
                            with plots_and_image_container:
                                ui.label(f"Error processing {selected_file}: {e}").classes('text-red-500')

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
                                try:
                                    el.props(remove='error')
                                except KeyError:
                                    pass
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
                            ui.notify('All correct! Running analysis...', color='positive')
                            update_galaxy_mass_analysis()
                        else:
                            ui.notify('Wrong, try again!', color='negative')



                    ui.button("Run Analysis", on_click=check_and_run_galaxy)


            with ui.tab_panel(three):
                with ui.card().classes("p-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Velocity Distribution")

                with ui.element('div').classes('text-lg'):
                    paragraph("The **blue histogram shows the observed velocities** of galaxies in the Coma cluster; this plot is **fixed**. "
                            "\n\n Use the slider to **add Dark Matter (DM)** to your simulation. "
                            "\n\n With **DM = 0**, the simulated galaxies are gravitationally unbound. "
                            "\n\n As you increase the **DM**, the gravitational pull keeps the galaxies bound and shapes their velocity distribution to match the observed histogram.")
                with ui.element('div').classes('text-lg '):
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

        """, sanitize=False)
                        ui.button("Close", on_click=cluster_dialog.close)
                        
                        
                        
                        
                ui.button("Info", on_click=lambda: [cluster_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded")      
               
              
                ui.html('''
<div class="flex space-x-2">
  <button id="audio-button" 
            onclick="initAudio()" 
            class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
        Activate Audio
    </button>
  <button onclick="playObservedSigmaMean()" class="bg-blue-500 hover:bg-blue-700 text-white px-3 py-1 rounded">▶ Observed σ (mean)</button>
  <button onclick="playObservedSigmaCurve()" class="bg-blue-300 hover:bg-blue-500 text-white px-3 py-1 rounded">▶ Observed σ (curve)</button>
  <button onclick="playBaryonicSigmaMean()" class="bg-red-500 hover:bg-red-700 text-white px-3 py-1 rounded">▶ Baryonic σ (mean)</button>
  <button onclick="playBaryonicSigmaCurve()" class="bg-red-300 hover:bg-red-500 text-white px-3 py-1 rounded">▶ Baryonic σ (curve)</button>
  <button onclick="playSimSigmaMean()" class="bg-green-500 hover:bg-green-700 text-white px-3 py-1 rounded">▶ Simulated σ (mean)</button>
  <button onclick="playSimSigmaCurve()" class="bg-green-300 hover:bg-green-500 text-white px-3 py-1 rounded">▶ Simulated σ (curve)</button>
  <button onclick="playDifferenceSigmaMean()" class="bg-purple-700 hover:bg-purple-900 text-white px-3 py-1 rounded">▶ Difference σ (mean)</button>
  <button onclick="playDifferenceSigmaCurve()" class="bg-purple-500 hover:bg-purple-700 text-white px-3 py-1 rounded">▶ Difference σ (curve)</button>

</div>
''', sanitize=False)


                
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
                           
                m_bary_at_gal = stellar_mass_from_r_mag(df_coma['modelmag_r'].values,df_coma['extinction_r'].values,z_cluster )
                
               
     
           
                M200, R200 = estimate_M200_R200_from_sigma(sigma_obs, rho_crit)
       
                c = concentration_duffy2008(M200, z_cluster, h=0.7, relaxed=False)

           
           
                obs_mean = np.mean(observed_vel)
                observed_vel_pec = observed_vel - obs_mean
                x_min = observed_vel.min()
                x_max = observed_vel.max()
                padding = (x_max - x_min) * 1.10
                bins = np.linspace(0, x_max + padding, 50)

                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                counts_obs, _ = np.histogram(observed_vel, bins=bins)
                y_max = int(np.ceil(np.max(counts_obs) * 1.15))
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
                v_max_global = max(observed_vel.max(), v_max_expected * 1.2, v_max_bar * 1.5)


                v_padding = 0.1 * (v_max_global - v_min_global)


                ylim_min = v_min_global
                ylim_max = v_max_global + v_padding

             
                           
                
                
                
                with ui.column().classes('w-full items-center '):
                    dm_slider_min, dm_slider_max = 0.0, 50.0
                    dm_slider = ui.slider(min=dm_slider_min, max=dm_slider_max,
                      value=0.0, step=0.01).props('label-always')

             
                    
                    with ui.row().classes('w-full justify-center gap-4'):
                        with ui.column().classes('flex-1 items-center'):
                            plot_container_histo = ui.column()
                            stats_container_histo = ui.column()
   
                        with ui.column().classes('flex-1 items-center'):
                            plot_container_scatter = ui.column()
                            stats_container_scatter = ui.column()
                    
                        
                        def update_coma_histogram_v2():
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
                                    with ui.pyplot(figsize=(8, 5)):
                                        ('all')
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
                                        plt.title('Galaxy Velocity Distribution (Cluster)')
                                        plt.legend()
                                        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

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
                                    with ui.pyplot(figsize=(8, 6)):
                                        ('all')
 
                                        plt.scatter(r_proj_kpc, observed_vel, s=10, color='blue', alpha=0.6, label='Observed Galaxies', rasterized=True)

                                        plt.scatter(r_proj_kpc, v_bar, s=10, color='red', alpha=0.6, label='Baryonic Component', rasterized=True)
                                        plt.scatter(r_proj_kpc, v_dm, s=10, color='green', alpha=0.6, label=f"Simulated Galaxies (DM_mean = {M_dm_mean:.2e} M☉)", rasterized=True)
                                    

  
                                    #r_sorted = np.sort(r_proj_kpc)
                                    #sigma_los_sorted = np.sqrt(G_grav * (m_bary_at_gal + f * M_dm_at_gal) / (3.0 * r_sorted))
                                    #v_upper = v_mean_val + 3 * sigma_los_sorted
                                    #v_lower = v_mean_val - 3 * sigma_los_sorted
                                    #plt.plot(r_sorted, v_upper, 'r--', lw=1.5, label='Caustic envelope')
                                    #plt.plot(r_sorted, v_lower, 'r--', lw=1.5)

   
                                        plt.xlim(r_min, r_max + r_padding)
                                        plt.ylim(ylim_min, ylim_max)


                                        plt.xlabel("Radius [kpc]")
                                        plt.ylabel("Velocity [km/s]")
                                        plt.title("Phase-space diagram of cluster galaxies")
                                        plt.legend()
                                        plt.grid(True, linestyle="--", alpha=0.7)
                                        
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
                                ui.label(f"An error occurred: {e}").classes('text-red-500')
                    
                    
                    
                 
                    
                
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
                                with ui.pyplot(figsize=(4, 4)):
                                    ('all')
                                    plt.scatter(state["x_obs"], state["y_obs"], c="blue", s=20, alpha=0.8, label="Observed")
                                    plt.xlim(0, box_size); plt.ylim(0, box_size)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Observed Galaxies")
                                    plt.legend(loc="upper right")
                                plot_info_box_compact({
    "Observed ⟨v⟩": f"{v_mean_obs:.0f} km/s",
    "σ_obs": f"{sigma_obs_val:.0f} km/s"
})

       
                            with sim_container_model:
                                sim_container_model.clear()
                                with ui.pyplot(figsize=(4, 4)):
                                    ('all')
                                    plt.scatter(state["x_model"], state["y_model"], c="green", s=20, alpha=0.8, label="Simulated")
                                    plt.xlim(0, box_size); plt.ylim(0, box_size)
                                    plt.xticks([]); plt.yticks([])
                                    plt.title("Simulated Galaxies")
                                    plt.legend(loc="upper right")
                                plot_info_box_compact({
    "Simulated σ": f"{sigma_withDM:.0f} km/s",
    "DM factor": f"{f:.2f}"
})


                        except Exception as e:
                            ui.notify(f"Simulator error: {e}", color="red")

                    @ui.refreshable
                    def refresh_cluster_plots():
                        plt.close('all')
                        update_coma_histogram_v2()
                       
                    
                      

                    dm_slider.on('update:model-value', refresh_cluster_plots.refresh)

                   
                    refresh_cluster_plots()
               


                    ui.timer(dt, update_cluster_points)



                
                    with ui.row().classes(' w-full gap-4 justify-center'):

   
                        with ui.column().classes('flex-1 items-center'):
                            ui.image(os.path.join(CLUSTER_IMG_PATH, "coma_img.jpg"))\
            .classes('w-full max-w-xl')
                            reference_box("""
**Image reference:**  
- [ESA Hubble](https://esahubble.org/)
""").classes('w-full text-center text-base italic mt-2')

  
                        with ui.column().classes('flex-1 items-center'):
                            table_filepath = os.path.join(CLUSTER_TABLES_PATH, "coma_table.csv")

                            if os.path.exists(table_filepath):
                                df_table = pd.read_csv(table_filepath)
                                ui.table.from_pandas(df_table).classes('w-full max-w-xl')
                                reference_box("""
**Cluster information reference:**  
- [Wikipedia:Coma Cluster](https://en.wikipedia.org/wiki/Coma_Cluster)
""").classes('w-full text-center text-base italic mt-2')
                            else:
                                ui.label("Table not found").classes('text-red-500')

       
            with ui.tab_panel(four):
                with ui.card().classes("p-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cluster Mass & Virial Theorem")

                with ui.element('div').classes('text-lg'):
                    paragraph(
                   "Imagine analyzing a **galaxy cluster** to reveal dark matter. "
"\n\n Choose a dataset and complete the missing fields."
"\n\n Click **Run Analysis** to compare **luminous mass** vs **total (virial) mass**.")
                with ui.element('div').classes('text-lg'):
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
""", sanitize=False)
                            ui.button("Close", on_click=formula_dialog.close)

                        with ui.dialog() as legend_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-gray-800 border border-blue-400 shadow-lg'):
                   
                                ui.markdown("**Legend of Symbols and Units**").classes('!text-xl !text-blue-300 !-mt-2')
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



""").classes('text-lg')
                            ui.button("Close", on_click=legend_dialog.close).classes("mt-4 bg-blue-600 text-white rounded-lg px-4 py-2")

                    
                        with ui.dialog() as units_dialog, ui.card().classes("p-6 w-full max-w-2xl bg-gray-900 rounded-xl shadow-xl"):
                            with ui.column().classes('flex-1 max-w-md items-start p-4 rounded-lg bg-slate-500 border border-green-500 shadow-lg'):
                                ui.markdown("**Units conversion**").classes('!text-xl !text-green-300 !-mt-2')
                                ui.markdown("""
- `1 kpc` = 3.086 × 10¹⁶ m = 3.26 × 10³ light-years  
- `1 pc` = 3.086 × 10¹³ km = 3.26 light-years  
- `1 Mpc` = 10⁶ pc = 3.086 × 10¹⁹ km  
- `1 km/s` = 3.6 × 10³ km/h = 10³ m/s  
- `1 M☉` = 1.989 × 10³⁰ kg (solar mass)  
- `1 L☉` = 3.828 × 10²⁶ W (solar luminosity)  
- Distance modulus: `m - M = 5 log10(d/10pc)`
- 1 arcsec at distance D → `D * tan(1 arcsec) ≈ D * 1/206265`  """).classes('text-lg')
                    
                            ui.button("Close", on_click=units_dialog.close).classes("mt-4 bg-green-600 text-white rounded-lg px-4 py-2")

                    
                        ui.button("Info", on_click=lambda: [formula_dialog.open(), ui.run_javascript("MathJax.typesetPromise()")]).classes(
        "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    )
                        ui.button("📘 Legend", on_click=legend_dialog.open).classes(
    "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
)
                        ui.button("📐 Units", on_click=units_dialog.open).classes(
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

                    answer_sigma,answer_r,answer_g2,answer_vel = {}, {},{},{}
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
                                    ans_U['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("3) Compute the gravitational force").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("F_grav = ")
                                    ans_Fg['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("4) Compute the centripetal force").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("F_cent = ")
                                    ans_Fc['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("5) Equate forces: F_grav = F_cent → derive v²(R)").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("v²(R) = ")
                                    ans_v2['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("6) Kinetic energy of the light particle").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("K = ")
                                    ans_K['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("7) Virial theorem (time-averaged, bound system)").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("2⟨")
                                    ans_virialLHS['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("⟩ + ⟨")
                                    ans_virialRHS['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("⟩ = 0")

                                ui.label("8) Compute circular orbit mass of one particle").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_circ(r) = ")
                                    ans_Mcirc['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("9) Compute the total mass for a system of N particles:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_tot(r) ≈ sum ")
                                    ans_Mvir['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("10) Compute velocity dispersion").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("σ_v² = 1/N * Σ |")
                                    answer_vel['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(" - mean(")
                                    answer_vel['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(")|²")

                                ui.label("11) Compute virial/total mass").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_total(r) = (3 * ")
                                    answer_sigma['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label("² * ")
                                    answer_r['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(") / ")
                                    answer_g2['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("12) Compute cluster distance in pc:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("D_pc = ")
                                    answer_D['el'] = ui.input(label="").props('dense filled').classes('w-48')

                                ui.label("13) Compute distance modulus:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("dist_mod = 5 * log10(")
                                    answer_Dmod['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(") - 5")

                                ui.label("14) Compute absolute magnitude from apparent magnitude (dataset) and distance:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_abs = ")
                                    answer_Mabs['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(" - dist_mod")

                                ui.label("15) Compute luminosity for each galaxy:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("L_B = 10 ** (-0.4 * (")
                                    answer_LB['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(" - M_sun_B))")

                                ui.label("16) Compute total luminosity:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("L(r) = Σ(")
                                    answer_Lsum['el'] = ui.input(label="").props('dense filled').classes('w-48')
                                    ui.label(") for galaxies with r_i ≤ r")

                                ui.label("17) Compute luminous mass:").classes('step')
                                with ui.row().classes('items-center'):
                                    ui.label("M_lum(r) = (M/L) * ")
                                    answer_Mlum['el'] = ui.input(label="").props('dense filled').classes('w-48')

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
                                        ('all')
                                        mask_lum = M_lum_r > 0
                                        plt.plot(R_cum[mask_lum], M_lum_r[mask_lum], label='Luminous Mass', color='red')
                                        plt.plot(R_cum, M_tot_r, label='Total Mass (Virial)', color='blue')
                                        plt.xscale('log'); plt.yscale('log')
                                        plt.xlabel('Radius (kpc)'); plt.ylabel('Mass (M☉)')
                                        plt.title(f'Mass Profile {os.path.splitext(selected_file)[0]}')
                                        plt.grid(True, which="both", ls="--"); plt.legend()

                            
                                    rho_lum_cum = M_lum_r / ((4.0/3.0) * np.pi * R_cum**3)
                                    rho_tot_cum = M_tot_r / ((4.0/3.0) * np.pi * R_cum**3)
                                with ui.column().classes('flex-1 items-center'):
                                    with ui.pyplot(figsize=(8,6)):
                                        ('all')
                                        plt.plot(R_cum, rho_lum_cum, label=' Luminous Density', color='red')
                                        plt.plot(R_cum, rho_tot_cum, label=' Total Density', color='blue')
                                        plt.xscale('log'); plt.yscale('log')
                                        plt.xlabel('Radius (kpc)')
                                        plt.ylabel('Density (M☉ / kpc³)')
                                        plt.title(f'Density Profile {os.path.splitext(selected_file)[0]}')
                                        plt.grid(True, which="both", ls="--")
                                        plt.legend()



                                with ui.row().classes('w-full gap-4 justify-start'):

   
                                    with ui.column().classes('flex-1 items-center '):
                                        if image_filepath and os.path.exists(image_filepath):
                                            ui.image(image_filepath).classes('w-full max-w-xl')
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
                "/workspaces/Cosmo-Edu_Lab/App/cluster_tables",
                os.path.splitext(selected_file)[0] + ".csv"
            )
                                        if os.path.exists(table_filepath):
                                            df_table = pd.read_csv(table_filepath)
                                            ui.table.from_pandas(df_table).classes('w-full max-w-xl')
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
                                ui.label(f"Error processing {selected_file}: {e}").classes('text-red-500')
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

   
                        checks = {
        answer_sigma: s in {"sigma_v","sigmav","σv","σ_v"},
        answer_r:     r in {"r","radius","rad"},
        answer_g2:    g in {"g","gravitational_constant"},
        ans_v2:       v2 in {"g*m/r","g*m*m/r","gm/r"},
        ans_K:        K in {"0.5*m*v^2","0.5*m*v**2","1/2*m*v^2","1/2*m*v**2",
                            "g*m/(2*r)","g*m*m/(2*r)"},
        ans_U:        U in {"-g*m/r","-g*m*m/r"},
        ans_Fg:       Fg in {"g*m/r^2","g*m*m/r^2"},
        ans_Fc:       Fc in {"m*v^2/r","m*v**2/r"},
        ans_virialLHS: virLHS in {"k"},
        ans_virialRHS: virRHS in {"u"},
        ans_Mcirc:    Mcirc in {"v^2*r/g","v**2*r/g","(v**2)*r/g","r*v^2/g"},
        ans_Mvir:     Mvir in {"(v^2)*r/g","(rv^2)*r/g","(rv**2)*r/g","(rv^2)r/g","(rv**2)r/g"},
        answer_LB:    LB in {"m_abs","mabs","mag_abs"},
        answer_Lsum:  Lsum in {"l_b","lb","l_b(r)","lb(r)"},
        answer_Mlum:  Mlum in {"(l(r))","l(r)"},
        answer_D:     D in {"d*1e6","dist*1e6","1e6*d"},
        answer_Dmod:  Dmod in {"d_pc","dpc","dist_pc"},
        answer_Mabs:  Mabs in {"bmag"},
        answer_vel:   vel in {"rv","rv_i","rv(i)","v_obs"},
    }

  
                        for ans in checks:
                            el = ans.get('el')
                            if el:
                                el.classes(remove='border-red-500 border-green-500')

  
                        all_ok = True
                        for ans, ok in checks.items():
                            el = ans.get('el')
                            if el:
                                if ok:
                                    el.classes(add='border-green-500')
                                else:
                                    el.classes(add='border-red-500')
                                    all_ok = False

                        if all_ok:
                            ui.notify('Correct! Analysis running...', color='positive')
                            update_cluster_mass_profile()
                        else:
                            ui.notify('Wrong! Try again!', color='negative')

                             
                        
                    ui.button("Run Analysis", on_click=check_and_run_cluster)

            with ui.tab_panel(five):
                with ui.card().classes("p-4 bg-gradient-to-r from-pink-500 to-red-600 text-white rounded-lg shadow-lg"):
                    title_on_dark("Cosmic Microwave Background (CMB)")

                with ui.row().classes('w-full gap-4 justify-start'):
                    
                    with ui.column().classes('flex-1 max-w-md items-start '):
                     
                        
                        ui.button('Explore the full Planck CMB Simulator',
          on_click=lambda: ui.run_javascript("window.open('https://chrisnorth.github.io/planckapps/Simulator', '_blank')")).props('outline color=primary')

                        reference_box("""
**Planck Simulator References:**  
- [Chris North Planck Simulator](https://chrisnorth.github.io/planckapps/Simulator)  
- [UCSB Planck Project](https://www.deepspace.ucsb.edu/projects/planck)  
  
""")
                    with ui.column().classes('flex-1 max-w-md items-start'):
                        ui.button('ESA: The Universe According to Planck', on_click=lambda: ui.run_javascript("window.open('https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck','_blank')")).props('outline color=accent')
                        reference_box("""

**Images References:**  
- [ESA Planck Overview](https://sci.esa.int/web/planck/-/51551-simple-but-challenging-the-universe-according-to-planck)  
- [Planck Image Archive](https://planck.ipac.caltech.edu/page/contacts)  
""")
                with ui.grid(columns=2).classes('w-full'):
                    
                    ui.image('/workspaces/Cosmo-Edu_Lab/images/univ_composition.jpg').classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    ui.image('/workspaces/Cosmo-Edu_Lab/images/power_spectrum.jpg').classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    ui.image('/workspaces/Cosmo-Edu_Lab/images/CMB_asymmetry.jpg').classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
                    ui.image('/workspaces/Cosmo-Edu_Lab/images/Planck_freq.jpg').classes('w-full h-auto rounded-lg shadow-lg border border-gray-300')
       
        
        
        standard_module_ui("The Dark Matter and Energy")




@ui.page('/module7')
def module7():
    main_layout("Module 7: Inflation & Open Questions")
   
@ui.page('/module8')
def module8():
    main_layout("Module 8: Cosmology & the Scientific Method")
   




if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0",
        port=7860,title='Cosmo-Edu Lab', storage_secret='a-very-secret-and-secure-key-for-sessions', dark=True)