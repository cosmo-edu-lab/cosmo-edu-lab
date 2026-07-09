import PyInstaller.__main__
import os
import platform
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import sys


#APP_DIR = "App"
#sys.path.append(os.path.abspath(APP_DIR))

from astropy.utils import iers

os.environ['ASTROPY_SKIP_INTERNET_DOWNLOADS'] = '1'
os.environ['ASTROPY_ALLOW_INTERNET'] = 'False'
iers.conf.auto_download = False
APP_NAME = "Cosmo-Edu_Lab"


current_os = platform.system()
if current_os == 'Windows':
    ext = '.exe'
    os_suffix = '_Windows'
elif current_os == 'Darwin':
    ext = ''
    os_suffix = '_Mac'
elif current_os == 'Linux':
    ext = ''
    os_suffix = '_Linux'
else:
    ext = ''
    os_suffix = '_Other'

FINAL_OUTPUT_NAME = f"{APP_NAME}{os_suffix}"


print(f"Analisi dipendenze complesse per {current_os}...")

# Astroquery e Astropy
astroquery_datas = collect_data_files('astroquery')
astropy_datas = collect_data_files('astropy')
plotly_datas = collect_data_files('plotly')
toolkit_datas = collect_data_files('nicegui_toolkit')
latex_datas = collect_data_files('latex2mathml')
imageio_datas = collect_data_files('imageio')

my_datas = [
    ('static', 'static'),           
    ('images', 'images'),           
    ('data', 'data'),               
    ('dataset', 'dataset'), 
    ('galaxy_data', 'galaxy_data'), 
    ('cluster_data', 'cluster_data'),
    ('cluster_tables', 'cluster_tables'),
    ('iso_fe0.01', 'iso_fe0.01'),
    ('pages', 'pages'),
    ('galaxy_spectra', 'galaxy_spectra'),
    ('planet_image', 'planet_image'),
    ('galaxy_img', 'galaxy_img'),
    ('cluster_img', 'cluster_img'),
    ('galaxy_tables', 'galaxy_tables'),
    ('slides', 'slides'),
    ('discovery_images', 'discovery_images'),
    ('cosmic_epochs', 'cosmic_epochs'),
]

all_datas = my_datas + astroquery_datas + astropy_datas + plotly_datas + toolkit_datas + latex_datas + imageio_datas

hidden_imports = [
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'scipy.stats._stats',
    'astropy.coordinates',
    'astroquery.simbad',
    'astroquery.mast',
    'plotly.validators',
    'plotly.graph_objs',
    'networkx',
    'PIL',
    'matplotlib.backends.backend_svg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_ps',
    'pages', 
    'latex2mathml',
    'imageio',
    'nicegui_toolkit',
]

hidden_imports += collect_submodules('astroquery')
hidden_imports += collect_submodules('nicegui_toolkit')

add_data_args = []
separator = ';' if current_os == 'Windows' else ':' 

for source, dest in all_datas:
    if os.path.exists(source):
        add_data_args.append(f'--add-data={source}{separator}{dest}')
    else:
        print(f"WARNING: Data source not found: {source}")

hidden_import_args = [f'--hidden-import={mod}' for mod in hidden_imports]

args = [
   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py'),                
    f'--name={FINAL_OUTPUT_NAME}',       
    '--onefile',                
    '--clean',                  
    '--optimize=1',
    '--collect-all=nicegui',          
    '--collect-all=nicegui_toolkit', 
    '--collect-all=xyz_services',
    '--collect-all=astropy',
    '--collect-all=astroquery',
    '--collect-all=pyvo',
] + add_data_args + hidden_import_args


if current_os == 'Windows':
    args.append('--windowed')

print(f"\nAvvio build completo con supporto per {len(hidden_imports)} librerie su {current_os}...")

try:
    PyInstaller.__main__.run(args)
    print(f"\nSUCCESSO! Trovi il file in: dist/{FINAL_OUTPUT_NAME}{ext}")
except Exception as e:
    print(f"\nERRORE durante il build: {e}")