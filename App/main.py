
# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.
from nicegui import ui, app, client
import os
import asyncio
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir)
#from nicegui_toolkit import inject_layout_tool
from huggingface_hub import HfApi
if sys.stdout is None:
    class NullIO:
        def write(self, text):
            pass
        def flush(self):
            pass
        def isatty(self):
            return False
    
    sys.stdout = NullIO()
    sys.stderr = NullIO()

if not hasattr(ui, '_original_add_head'):
    ui._original_add_head = ui.add_head_html
    ui._original_add_css = ui.add_css
    ui._original_add_body = ui.add_body_html

def safe_add_head(html, **kwargs):
    kwargs['shared'] = True 
    ui._original_add_head(html, **kwargs)

def safe_add_css(css, **kwargs):
    kwargs['shared'] = True
    ui._original_add_css(css, **kwargs)

def safe_add_body(html, **kwargs):
    kwargs['shared'] = True
    ui._original_add_body(html, **kwargs)


ui.add_head_html = safe_add_head
ui.add_css = safe_add_css
ui.add_body_html = safe_add_body

import core

import layout
from layout import *
from core import *
import pages
from pages import auth, home, module1, module2, module3, module4

core.load_data()
core.download_submissions_from_cloud()
MODULES_LOCKED = os.getenv("MODULES_LOCKED", "True") == "True"
#app.add_static_files('/cluster_gif', os.path.join(BASE_DIR, 'cluster_gif'))
app.add_static_files('/images', os.path.join(BASE_DIR, 'images'))
app.add_static_files('/dataset', os.path.join(BASE_DIR, 'dataset'))
app.add_static_files('/galaxy_img', os.path.join(BASE_DIR, 'galaxy_img'))
app.add_static_files('/cluster_img', os.path.join(BASE_DIR, 'cluster_img'))
app.add_static_files('/slides', os.path.join(BASE_DIR, 'slides'))
app.add_static_files('/discovery_images', os.path.join(BASE_DIR, 'discovery_images'))
app.add_static_files('/cosmic_epochs', os.path.join(BASE_DIR, 'cosmic_epochs'))
app.add_static_files('/student_files', core.SUBMISSIONS_PATH)

app.add_static_files('/static', os.path.join(BASE_DIR, 'static'))

auth.create_auth_routes()  
home.create_routes()       
module1.create_page()    
module2.create_page()    
if not MODULES_LOCKED:  
    module3.create_page()     
    module4.create_page()     

async def keep_alive():
    while True:
        await asyncio.sleep(30)
        try:
            ui.run_javascript('void(0);')
        except:
            pass

app.on_startup(lambda: asyncio.create_task(keep_alive()))

#inject_layout_tool()



if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=7860,
        title='Cosmo-Edu Lab', 
        language='en-US',
        storage_secret='CosmoEduSecretKey2024Fixed', 
        dark=True,
        reconnect_timeout=30.0,
        binding_refresh_interval=0.1,
        reload=False, 
        show=True    
    )
'''
SPACE_ID = "CosmoEduLab/Cosmo-Edu" 
TIMEOUT_SECONDS = 15  

shutdown_timer = None

def pause_space():

    print("⏳ Nessun utente connesso. Metto in pausa lo Space...", flush=True)
    try:
       
        token = os.environ.get('HF_API_TOKEN')
        if not token:
            print("⚠️ ERRORE: Variabile HF_API_TOKEN non trovata nei Secrets!")
            return
            
        api = HfApi(token=token)
        api.pause_space(repo_id=SPACE_ID)
    except Exception as e:
        print(f"❌ Errore durante la pausa dello Space: {e}", flush=True)

def handle_connect():
  
    global shutdown_timer
    if shutdown_timer:
        shutdown_timer.cancel()
        shutdown_timer = None
        print("✅ Utente connesso. Spegnimento annullato.", flush=True)

def handle_disconnect():
  
    global shutdown_timer
  
    active_users = [c for c in client.Client.instances.values() if c.has_socket_connection]

 
    if len(active_users) == 0: 
        if shutdown_timer:
            return
        print(f"⚠️ Ultimo utente disconnesso. Spegnimento in {TIMEOUT_SECONDS} secondi...", flush=True)
        shutdown_timer = ui.timer(TIMEOUT_SECONDS, pause_space, once=True)


app.on_connect(handle_connect)
app.on_disconnect(handle_disconnect)
'''