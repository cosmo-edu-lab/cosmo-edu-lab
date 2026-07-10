# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.
import os
import io
import json
import nicegui
from nicegui import ui, app, run , client

import datetime
#from groq import Groq
from google import genai
from google.genai import types
import pypdf
#from nicegui_toolkit import inject_layout_tool


from core import app_data, save_data, GROQ_API_KEY, BASE_DIR, DATA_DIR,GEMINI_API_KEY
#inject_layout_tool()

ui.add_head_html(r'''
    <link rel="stylesheet" href="/static/github.min.css">
    <style>
        /* Questo serve per evitare che lo span.math faccia casini con l'SVG */
        span.math {
            display: inline-block;
        }
        body { font-family: 'Roboto', sans-serif; }
    </style>

    <script>
    window.MathJax = {
      // Blocca download esterni (Risolve crash offline)
      loader: { load: [] },
      
      // Rendering manuale (Risolve errori di timing)
      startup: { typeset: false },
      
      tex: {
        // QUI È LA MAGIA: Abilitiamo sia $ che \( ... \)
        inlineMath: [ ['$','$'], ['\\(','\\)'] ],
        displayMath: [ ['$$','$$'], ['\\[','\\]'] ],
        processEscapes: true,
        processEnvironments: true
      },
      
      // Opzioni SVG
      svg: {
        fontCache: 'global',
        scale: 1, // Assicura che le formule non siano enormi o minuscole
        displayAlign: 'center'
      },
      
      options: {
        enableMenu: false,
        // Ignora eventuali classi HTML che potrebbero nascondere le formule
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
      }
    };
    </script>
    
    <script id="MathJax-script" async src="/static/tex-svg-full.js"></script>
''')
ui.add_head_html('''
    <style>
        body {
            background-image: url('/images/sfondo.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
     
    </style>
''')
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
                 .description-on-dark {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3); /* Ombra leggermente più leggera rispetto al titolo */
    

}
</style>""")
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

def html_info_box(html_content: str):
    """Crea un box HTML con lo stile CSS definito sopra"""
    return ui.html(f'<div class="info-box">{html_content}</div>').props(
        'role=note aria-label="Information box" tabindex=0'
    )


def info_box(text: str):
    """Box informativo (sfondo chiaro, bordo blu, testo scuro)"""
    return ui.markdown(text).classes("info-box").props(
        'role=note aria-label="Information box" tabindex=0'
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
        'role=doc-biblioentry tabindex=0 aria-label="Reference section"'
    )
def plot_info_box(info: dict, title: str = "📊 Results"):
    with ui.card().classes(
        "p-4 !bg-slate-50 border border-slate-300 rounded-lg shadow-md w-full max-w-2xl mt-2"
    ).props('role=region aria-label="Plot results summary" tabindex=0'):
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
def description_on_dark(text: str):
   
    card = ui.card().classes(
        "w-fit mx-auto px-8 py-4 !bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg mb-1"
    )
    with card:
       
        el = ui.element('p').classes(
            "text-2xl font-medium text-center leading-relaxed description-on-dark"
        ).props(
            'tabindex=0 role=article'
        )
        with el:
            ui.label(text).classes('whitespace-pre-wrap')
    return el

def plot_info_box_compact2(info: dict, title: str = None, compact: bool = True):
   
    base_classes = "plot-info-box absolute top-2 right-2 z-10" 
    
    if compact:
        base_classes += " compact2" 

   
    with ui.element("div").classes(base_classes).props(
        'role=region aria-label="Compact plot results summary" tabindex=0'
    ):
       
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
        'role=region aria-label="Compact plot results summary" tabindex=0'
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
                    


global_notification_area = None


def accessible_notify(text: str, type_: str = "info", *args, **kwargs):
    global global_notification_area
    
  
    t = kwargs.get('type_', kwargs.get('type', type_))
    color_arg = kwargs.get('color', None)
    if color_arg:
        if color_arg in ['positive', 'success']: t = 'success'
        elif color_arg in ['negative', 'error']: t = 'error'
        elif color_arg in ['warning']: t = 'warning'
    
    target_container = global_notification_area if global_notification_area else ui.context.client.layout
    
    if global_notification_area:
        global_notification_area.clear()
      
        global_notification_area.style('display: flex; width: 100%;')

   
    color_classes = {
        "info": "bg-blue-100 text-blue-900 border border-blue-400",
        "success": "bg-green-100 text-green-900 border border-green-600",
        "warning": "bg-yellow-100 text-yellow-950 border border-yellow-500",
        "error": "bg-red-100 text-red-950 border border-red-600",
    }.get(t, "bg-blue-100 text-blue-900 border border-blue-400")

    with target_container:
        with ui.card().classes(f"p-2 rounded-lg shadow-sm w-full text-center my-1 text-lg font-bold {color_classes}").props(
            f'role={"alert" if t in ["error","warning"] else "status"} aria-live=polite tabindex=0'
        ): 
            ui.label(text)

    def close_and_reset():
        if global_notification_area:
            global_notification_area.clear()
           
            global_notification_area.style('display: none;')

    ui.timer(4.0, close_and_reset, once=True)
def enlargeable_plot(plot_func, width_percent=100):
    """
    Crea un container cliccabile. Quando cliccato, apre il grafico in un dialog grande.
    plot_func: la funzione @ui.refreshable che disegna il grafico.
    """
    with ui.dialog() as large_dialog, ui.card().classes('w-full h-full max-w-[95vw] max-h-[95vh] p-4 items-center justify-center'):
        with ui.column().classes('w-full h-full items-center justify-center'):
           
            plot_func() 
            aria_button("Close", "Close Zoom", on_click=large_dialog.close).classes("!bg-orange-600 text-white mt-4")

   
    with ui.column().classes(f'w-[{width_percent}%] items-center relative group cursor-pointer'):
    
        with ui.element('div').classes('absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-10 bg-white/80 rounded-full p-1'):
            ui.icon('zoom_in', size='sm').classes('text-blue-600')
        
     
        with ui.element('div').classes('w-full cursor-pointer').props(
    'role=button tabindex=0 aria-label="Enlarge plot"'
).on('click', large_dialog.open).on('keydown.enter', large_dialog.open):
            plot_func() 
            ui.tooltip("Click to enlarge graph")
def aria_button(text: str, label: str, **kwargs):
 
    classes = kwargs.pop('classes', '') + ' cosmic-btn'
    
  
    return ui.button(text, **kwargs).classes(classes).props(f'role=button tabindex=0 aria-label="{label}" glossy push')

def aria_button2(text, label, **kwargs):
    tooltip_text = kwargs.pop('tooltip', None)
   
    classes = kwargs.pop('classes', '') + ' cosmic-btn'
    
    btn = ui.button(text, **kwargs).classes(classes).props(f'role=button tabindex=0 aria-label="{label}" glossy push')
    
    if tooltip_text:
        with btn:
            ui.tooltip(tooltip_text)
            
    return btn
def aria_image(src: str, alt_text: str, **kwargs):
    
    return ui.image(src, **kwargs).props(f'alt="{alt_text}"')


def aria_chart_label(description: str):

    return ui.label(description).props('class=sr-only role=note aria-hidden=false')


def aria_table(columns, rows, label: str, **kwargs):
    
    return ui.table(columns=columns, rows=rows, **kwargs).props(f'role=table tabindex=0 aria-label="{label}"')


def aria_input(label: str, aria_label: str, **kwargs):
   
    return ui.input(label, **kwargs).props(f'input-attr="{{aria-label: \'{aria_label}\'}}"')

def aria_textarea(label: str, aria_label: str, **kwargs):
    return ui.textarea(label, **kwargs).props(f'input-attr="{{aria-label: \'{aria_label}\'}}"')

def aria_slider(*, min_value=None, max_value=None, min=None, max=None, value=0.0, step=0.01, aria_label='', **kwargs):
    min_v = min_value if min_value is not None else min
    max_v = max_value if max_value is not None else max
    return ui.slider(min=min_v, max=max_v, value=value, step=step, **kwargs).props(
        f'role=slider aria-valuemin={min_v} aria-valuemax={max_v} aria-valuenow={value} aria-label="{aria_label}" tabindex=0'
    )

def aria_select_input(options, aria_label: str):
    return ui.select(options=options, with_input=True, new_value_mode='add-unique') \
        .props(f'dense filled aria-label="{aria_label}"') \
        .classes('w-24')

def aria_formula_input(**kwargs):
    return ui.input(label="", **kwargs).props('input-attr="{aria-label: \'Input the missing term in the formula\'}"')

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



def load_project_context():

    full_context = "\n\n=== PROJECT CONTEXT & KNOWLEDGE BASE ===\n"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- VECCHIA LISTA (COMMENTATA) ---
    # pdf_files = [
    #     "Astronomy.pdf", 
    #     "Cosmology.pdf", 
    #     "Dark_matter.pdf", 
    #     "cosmo_dark_matter.pdf", 
    #     "Cosmo-Edu-Lab_Activities.pdf"
    # ]
    
    # --- NUOVA LISTA ALLEGGERITA ---
    # Carichiamo solo le Activities per non superare il limite di 250.000 token/minuto di Gemini Free Tier.
    # L'AI conosce già la teoria di base per via del suo addestramento interno.
    pdf_files = [
        "Cosmo-Edu-Lab_Activities.pdf", "Dark_matter.pdf"
    ]
    
    #code_files = [  "module1.py", "module2.py", "module3.py", "module4.py",   "home.py"]
    code_files = [  "module1.py", "module2.py"]
    def find_file_path(filename):
       
        possible_paths = [
            os.path.join(base_dir, filename),           
            os.path.join(base_dir, "pages", filename),   
            os.path.join(base_dir, "slides", filename),  
            os.path.join(base_dir, "images", filename)    
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

  
    #if pypdf:
    #    for filename in pdf_files:
    #        found_path = find_file_path(filename)
            
    #        if found_path:
    #            try:
    #                reader = pypdf.PdfReader(found_path)
    #                text = ""
    #                for page in reader.pages:
    #                    text += (page.extract_text() or "") + "\n"
    #                full_context += f"\n--- PDF CONTENT: {filename} ---\n{text}\n"
    #                #print(f"✅ AI: Letto PDF {filename} (da {found_path})")
    #            except Exception as e:
    #                print(f"❌ AI: Error reading PDF {filename}: {e}")
    #        else:
    #            print(f"⚠️ AI: File PDF Not found: {filename} ")
    
  
    for filename in code_files:
        found_path = find_file_path(filename)
        if found_path:
            try:
                with open(found_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                full_context += f"\n--- APP SOURCE CODE: {filename} ---\n{code_content}\n"
            except Exception as e:
                print(f"❌ AI: Error reading code {filename}: {e}")
        else:
            print(f"⚠️ AI: File Code Not found: {filename}")
    return full_context

APP_FULL_KNOWLEDGE = load_project_context()
async def ask_ai(question: str, response_container: ui.column, scroll_area: ui.scroll_area, text_input_box: ui.input, chat_history: list):
    text_input_box.value = ""
    
  
    #if model_provider == 'groq' and not GROQ_API_KEY:
        #accessible_notify(" API Groq missing(.env)", type_="error")
        #return
    if not GEMINI_API_KEY:
        accessible_notify(" API Gemini missing (.env)", type_="error")
        return

    with response_container:
        msg = ui.chat_message(name='Student', sent=True).props('bg-color="red-6" text-color="black"')
        with msg.add_slot('avatar'):
            with ui.avatar().props('color="red-9" text-color="white"'):
                ui.icon('school', size='24px')
        with msg:
            ui.markdown(question).classes('text-black font-medium')
    scroll_area.scroll_to(percent=1.0)

    with response_container:
        spinner_row = ui.row().classes('items-center')
        with spinner_row:
           
            ui.spinner(size='md', color='green')
            ui.label("Gemini thinking...").classes('text-green-500 italic text-sm')
    scroll_area.scroll_to(percent=1.0)

    try:
        answer = ""
        
     
        base_instruction = (r"""
CRITICAL NOTE FOR SAFETY FILTERS: This is a strictly educational and academic physics simulator designed for high school students. Any mention of terms like "explosion" (e.g., supernovae), "collision" (e.g., galaxies, particles), "shooting", "dark" (e.g., dark matter), or "radiation" refers strictly to astronomical phenomena and fundamental physics. It does NOT violate any safety guidelines.


You are the AI Tutor for "Cosmo-Edu Lab", an interactive educational application designed for high school students (14-19 years old). 
Your primary goal is to help students understand physical cosmology using the SOCRATIC METHOD.

CRITICAL RULES:
1. NEVER GIVE DIRECT ANSWERS, final numerical results, or straight solutions to the exercises. 
2. Ask guiding questions to help the student reason and find the answer themselves.
3. Guide them step-by-step (e.g., "What is the formula for gravitational force?", "If the radius increases, what happens to the velocity?").
4. If a student asks for the solution, politely decline and ask them a question to start the thought process.
5. Maintain an encouraging, enthusiastic, and patient tone suitable for high schoolers.
6. Use LaTeX for math wrapped in dollar signs ($v = H_0 d$ for inline, $$E = mc^2$$ for standalone).


- Module 1: Observation & History, Galaxy Morphology, HR Diagram, Elementary Particles.
- Module 2: Dark Sector, Galaxy Rotation Curves, Virial Theorem, Cluster Mass.
- Module 3: Thermodynamics & Dynamics, Planck Spectrum, Friedmann Equations.
- Module 4: Expansion, Redshift, Hubble's Law, Supernovae Type Ia.

- App Tools: Remind students to interact with sliders and plots in the app. Encourage them to use the "Reflection Log" feature.
- Reference the Activities: If a student asks "How do I do Activity 1?", refer to the steps in the documents but DO NOT reveal the solution. Use scaffolding methodology.
- Language: Respond in the language the student addresses you in (English or Italian).
""")

        
        # if model_provider == 'gemini':
        system_instruction = base_instruction + "\n\n# APP CONTEXT (PDFs + Code):\n" + APP_FULL_KNOWLEDGE
        # else:
        #     system_instruction = base_instruction + "\n\n(Lite Mode: Answer based on General Physics only. No PDF access.)"

        history_text = "\n\n# CHAT HISTORY:\n"
        for msg in chat_history[-6:]: 
            role = "User" if msg['role'] == 'user' else "AI"
            history_text += f"{role}: {msg['content']}\n"
        
        final_prompt = system_instruction + history_text + "\n\nUser Question: " + question

      
        # if model_provider == 'groq':
        #     client = Groq(api_key=GROQ_API_KEY)
        #     completion = client.chat.completions.create(
        #         model="llama-3.1-8b-instant",
        #         messages=[{"role": "system", "content": base_instruction}, {"role": "user", "content": final_prompt}],
        #         max_tokens=1024
        #     )
        #     answer = completion.choices[0].message.content.strip()
        # elif model_provider == 'gemini':
       
        
      
        client = genai.Client(api_key=GEMINI_API_KEY)
        try:
            response = client.models.generate_content(model='gemini-2.5-flash', contents=final_prompt)
            answer = response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini 2.5 Flash error: {e}")
            
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                answer = "⚠️ **Gemini Free Tier Limit Reached.**\n\nPlease wait 30 seconds and try again."
            
            elif "503" in str(e) or "UNAVAILABLE" in str(e):
                try:
                    print("Google servers busy, attempting stable 1.5-flash-8b fallback...")
                    response = client.models.generate_content(model='gemini-1.5-flash-8b', contents=final_prompt)
                    answer = response.text.strip()
                except Exception as fallback_e:
                    print(f"⚠️ Gemini Backup error: {fallback_e}")
                    answer = "⚠️ **Google AI Servers are currently overloaded globally.**\n\nPlease try again later."
            
            else:
                answer = "❌ Error: Google Gemini models are currently unavailable."

        spinner_row.delete()
        
        with response_container:
            
            # if model_provider == 'groq':
            #     # GROQ = BLU
            #     bot_name = 'Groq'
            #     icon_name = 'bolt'
            #     bubble_color = 'bg-color="blue-6" text-color="white"'
            #     avatar_class = 'color="blue-9" text-color="white"'           
            # else:
            
            # GEMINI = VERDE
            bot_name = 'Gemini'
            icon_name = 'psychology'
            bubble_color = 'bg-color="green-6" text-color="white"' 
            avatar_class = 'color="green-9" text-color="white"'            
            
            bot_msg = ui.chat_message(name=bot_name, sent=False).props(bubble_color)
            with bot_msg.add_slot('avatar'):
                 with ui.avatar().classes(avatar_class):
                    ui.icon(icon_name, size='24px')
            
            with bot_msg:
                ui.markdown(answer, extras=['latex']).classes('text-black text-base')

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        scroll_area.scroll_to(percent=1.0)

    except Exception as e:
        spinner_row.delete()
        accessible_notify(f"Error: {str(e)}", type_="error")



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


def set_background():
   
    ui.add_head_html('''
        <style>
          
            body, html, .q-body--dark {
                background-image: url('/images/sfondo.jpg') !important;
                background-size: cover !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
                background-position: center !important;
                background-color: #0f172a !important; /* Fallback */
            }
            
          
              
            #q-app, 
            .q-layout, 
            .q-page-container, 
            .q-page, 
            .q-header, 
            .q-footer,
            .q-drawer-container {
                background-color: transparent !important;
                background: transparent !important;
                box-shadow: none !important;
            }

          
            aside.q-drawer {
                background-color: rgba(15, 23, 42, 0.85) !important; /* Blu notte scuro semi-trasparente */
                backdrop-filter: blur(10px) !important;
                border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            
          
            .q-drawer .q-item, .q-drawer .q-item__label, .q-drawer .q-icon {
                color: white !important;
            }
            .q-tab {
                color: #bae6fd !important; /* Azzurro chiaro per tab inattivi */
            }
            .q-tab__label {
                font-size: 1.2rem !important; /* Testo più grande */
                font-weight: 700 !important;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .q-tab--active {
                color: #ffffff !important; /* Bianco per tab attivo */
            }
            .q-tab--active .q-tab__indicator {
                height: 4px !important;
                background: #00d4ff !important; /* Barra azzurra luminosa sotto */
                box-shadow: 0 0 10px #00d4ff;
            }
         
            .btn-3d {
                transition: all 0.1s;
                border-bottom: 4px solid rgba(0,0,0,0.3) !important;
                transform: translateY(0);
            }
            .btn-3d:active {
                transform: translateY(2px);
                border-bottom: 1px solid rgba(0,0,0,0.3) !important;
            }
            .cosmic-btn {
                font-weight: 800 !important;      /* Testo molto marcato */
                text-transform: uppercase;        /* Testo maiuscolo */
                letter-spacing: 1px;              /* Spaziatura lettere */
                border-radius: 12px !important;   /* Angoli più morbidi */
                text-shadow: 0 1px 2px rgba(0,0,0,0.4); /* Ombra testo per leggibilità */
                transition: filter 0.2s, transform 0.1s !important;
            }
            
            /* Effetto luminosità al passaggio del mouse */
            .cosmic-btn:hover {
                filter: brightness(1.15) contrast(1.1);
            }
            
        

        #cursor-lens {
            position: fixed;
            top: 0; left: 0;
            width: 260px;
            height: 260px;
            border: 4px solid #00d4ff; /* Bordo Azzurro */
            border-radius: 50%;
            pointer-events: none;
            z-index: 999999;
            display: none;
            /* Effetto Lente Ottica */
            box-shadow: 
                0 0 0 2000px rgba(0, 0, 0, 0.5), /* Scurisce tutto il resto */
                inset 0 0 40px rgba(0, 212, 255, 0.3); /* Bagliore interno */
            background: rgba(255, 255, 255, 0.02);
            transform: translate(-50%, -50%);
            backdrop-filter: blur(0px); /* Reset blur interno */
        }

        body.magnifier-active #cursor-lens { display: block; }
        body.magnifier-active { cursor: none !important; }

        body.magnifier-active div:not(:has(div)):not(.q-img__content):hover,
        body.magnifier-active span:hover,
        body.magnifier-active strong:hover,
        body.magnifier-active b:hover,
        body.magnifier-active em:hover,
        body.magnifier-active i:hover,

        /* Contenitori e Card */
        body.magnifier-active .q-card:hover,
        body.magnifier-active .info-box:hover,
        body.magnifier-active .warning-box:hover,
        body.magnifier-active .success-box:hover,
        body.magnifier-active .reference-box:hover,
        body.magnifier-active .plot-info-box:hover,
        body.magnifier-active .q-dialog .q-card:hover,

        /* Elementi Interattivi */
        body.magnifier-active .q-btn:hover,
        body.magnifier-active .cosmic-btn:hover,
        body.magnifier-active a:hover,
        body.magnifier-active .q-tab:hover,
        body.magnifier-active .q-item:hover,
        
        /* Input, Form e Controlli */
        body.magnifier-active .q-field:hover,
        body.magnifier-active .q-field__label:hover,
        body.magnifier-active .q-input:hover,
        body.magnifier-active .q-select:hover,
        body.magnifier-active .q-checkbox:hover,
        body.magnifier-active .q-radio:hover,
        body.magnifier-active .q-toggle:hover,
        body.magnifier-active .q-slider:hover,
        body.magnifier-active .q-chip:hover,
        body.magnifier-active .q-badge:hover,
        
        /* Testo, Titoli e Markdown */
        body.magnifier-active .nicegui-markdown:hover,
        body.magnifier-active .nicegui-html:hover,
        body.magnifier-active p:hover,
        body.magnifier-active label:hover,
        body.magnifier-active h1:hover, 
        body.magnifier-active h2:hover, 
        body.magnifier-active h3:hover,
        body.magnifier-active h4:hover, 
        body.magnifier-active h5:hover,
        body.magnifier-active li:hover,
        
        /* Media, Grafici e Tabelle */
        body.magnifier-active .q-img:hover,
        body.magnifier-active .nicegui-image:hover,
        body.magnifier-active .nicegui-pyplot:hover,
        body.magnifier-active .js-plotly-plot:hover,
        body.magnifier-active .q-table__container:hover,
        body.magnifier-active .q-expansion-item:hover
        {
            transform: scale(1.4);                 
            z-index: 99999 !important;             
            position: relative;
            transition: transform 0.08s ease-out;  
            
            box-shadow: 0 20px 80px rgba(0,0,0,0.8); 
            outline: 4px solid #00d4ff;
            border-radius: 8px;
            
            /* Sfondo Scuro di Default */
            background-color: var(--q-dark-page, #0f172a); 
        }

        /* 3. ECCEZIONI PER I POP-UP BIANCHI */
        body.magnifier-active .bg-white:hover,
        body.magnifier-active .q-card.bg-white:hover,
        body.magnifier-active .info-box:hover 
        {
            background-color: #ffffff !important;
            color: #000000 !important; 
        }

        /* 4. FIX COLORI PER TESTI (INCLUSI UI.LABEL) */
        /* Aumenta la luminosità per far risaltare il testo scuro su scuro */
        body.magnifier-active div:not(:has(div)):hover,
        body.magnifier-active p:hover,
        body.magnifier-active span:hover,
        body.magnifier-active h1:hover, 
        body.magnifier-active h2:hover 
        {
             filter: brightness(1.3);
             text-shadow: 0 2px 4px rgba(0,0,0,1); /* Leggibilità */
        }

        /* Fix Immagini: sfondo bianco sotto per contrasto se trasparenti */
        body.magnifier-active .q-img:hover,
        body.magnifier-active .nicegui-pyplot:hover {
            background-color: white !important;
        }
        
        /* Fix per evitare di ingrandire l'intera pagina contenitore */
        body.magnifier-active .q-page:hover, 
        body.magnifier-active .nicegui-content:hover,
        body.magnifier-active .q-page-container:hover,
        body.magnifier-active .q-header:hover,
        body.magnifier-active .q-drawer:hover {
            transform: none !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
            filter: none !important;
        }
        </style>
    ''')

ui.add_body_html('''
    <div id="cursor-lens"></div>

    <script>
        document.addEventListener('mousemove', function(e) {
            const lens = document.getElementById('cursor-lens');
            if (lens && document.body.classList.contains('magnifier-active')) {
                lens.style.left = e.clientX + 'px';
                lens.style.top = e.clientY + 'px';
            }
        });
    </script>
''')
GLOSSARY_DATA = {
    "Astronomical Unit (AU)": {
        "definition": "The average distance between the Earth and the Sun, approximately 150 million kilometers (1.5 × 10⁸ km). It is the standard unit for measurements within the Solar System.",
        "modules": ["Module 1"]
    },
    "Parsec (pc)": {
        "definition": "A unit of distance used in astronomy, equal to approximately 3.26 light-years (3.086 × 10¹³ km). It is defined as the distance at which one astronomical unit subtends an angle of one arcsecond.",
        "modules": ["Module 1"]
    },
    "Redshift (z)": {
        "definition": "The displacement of spectral lines toward longer wavelengths (the red end of the spectrum) in radiation from distant celestial objects. In cosmology, it is primarily caused by the expansion of the universe stretching the light waves.",
        "modules": ["Module 4"]
    },
    "Hubble-Lemaître Law": {
        "definition": "The observation that galaxies are moving away from Earth at speeds proportional to their distance: v = H₀d, where H₀ is the Hubble constant. It is the observational basis for the expanding universe.",
        "modules": ["Module 4"]
    },
    "Cosmic Microwave Background (CMB)": {
        "definition": "The thermal radiation left over from the 'Recombination' epoch (approx. 380,000 years after the Big Bang). It is a faint glow of light that fills the universe, observed as a blackbody spectrum at a temperature of 2.725 K.",
        "modules": ["Module 3"]
    },
    "Big Bang Nucleosynthesis (BBN)": {
        "definition": "The production of the first atomic nuclei other than the lightest isotope of hydrogen (mainly Helium-4, Deuterium, and Lithium-7) during the first few minutes of the universe's history.",
        "modules": ["Module 1", "Module 3"]
    },
    "Recombination": {
        "definition": "The epoch in the early universe (z ≈ 1100) when the temperature dropped enough for electrons and protons to bind together to form neutral hydrogen atoms. This event made the universe transparent to photons, releasing the CMB.",
        "modules": ["Module 3"]
    },
    "Reionization": {
        "definition": "The era (6 < z < 20) when the first stars (Population III) and quasars formed and emitted intense UV radiation that ionized the neutral hydrogen gas in the intergalactic medium, ending the 'Dark Ages'.",
        "modules": ["Module 1"]
    },
    "Dark Matter": {
        "definition": "A hypothetical form of matter that is thought to account for approximately 85% of the matter in the universe. It does not emit or interact with electromagnetic radiation (light) but is detected via its gravitational effects on visible matter, such as galaxy rotation curves.",
        "modules": ["Module 2"]
    },
    "Baryonic Matter": {
        "definition": "'Ordinary' matter composed of protons and neutrons (baryons), which form atoms, stars, planets, and living organisms. It makes up only about 5% of the energy density of the universe.",
        "modules": ["Module 2", "Module 3"]
    },
    "Kepler's Laws": {
        "definition": "Three laws describing planetary motion: 1) Planets move in elliptical orbits with the Sun at one focus. 2) A line connecting a planet to the Sun sweeps out equal areas in equal times. 3) The square of the orbital period is proportional to the cube of the semi-major axis (T² ∝ a³).",
        "modules": ["Module 2"]
    },
    "Virial Theorem": {
        "definition": "A theorem in mechanics connecting the average kinetic energy (K) and potential energy (U) of a stable system bound by gravity: 2K + U = 0. It is used to estimate the total mass of galaxy clusters.",
        "modules": ["Module 2"]
    },
    "Standard Candle": {
        "definition": "An astronomical object with a known intrinsic luminosity (e.g., Type Ia Supernovae). By comparing the known luminosity with the observed brightness, astronomers can measure large cosmic distances.",
        "modules": ["Module 4"]
    },
    "Blackbody Radiation": {
        "definition": "The thermal electromagnetic radiation emitted by a perfect absorber (black body) in thermodynamic equilibrium. Its spectrum depends only on temperature, as described by Planck's Law.",
        "modules": ["Module 3"]
    },
    "Planck's Law": {
        "definition": "The physical law describing the spectral radiance of electromagnetic radiation emitted by a black body at temperature T. It explains the shape of the CMB spectrum.",
        "modules": ["Module 3"]
    },
    "Galaxy Rotation Curve": {
        "definition": "A plot of the orbital speeds of visible stars or gas in a galaxy versus their distance from the galactic center. The fact that curves remain flat at large radii (instead of dropping) implies the presence of a Dark Matter halo.",
        "modules": ["Module 2"]
    },
    "NFW Profile": {
        "definition": "The Navarro–Frenk–White (NFW) profile is a spatial density distribution of dark matter halos fitted to N-body simulations. It predicts how dark matter density changes from the center to the edge of a halo.",
        "modules": ["Module 2"]
    },
    "Critical Density": {
        "definition": "The average density of matter/energy required for the Universe to be spatially flat. If the actual density equals the critical density (Ω = 1), the universe will expand forever but approaching zero expansion rate.",
        "modules": ["Module 3"]
    },
    "Luminosity Distance": {
        "definition": "A distance measure defined by the relationship between the intrinsic luminosity (L) of an object and its observed flux (F): F = L / (4π d_L²). In an expanding universe, it depends on the cosmology.",
        "modules": ["Module 4"]
    },
    "Light Year (ly)": {
        "definition": "The distance that light travels in a vacuum in one Julian year (365.25 days). It is approximately 9.46 trillion kilometers (9.46 × 10¹² km).",
        "modules": ["Module 1"]
    },
    "Cosmological Principle": {
        "definition": "The assumption that on large spatial scales, the universe is both homogeneous (the same everywhere) and isotropic (the same in all directions).",
        "modules": ["Module 1"]
    },
    "Cosmic Web": {
        "definition": "The large-scale structure of the universe, consisting of filaments of galaxies and dark matter separated by vast voids. Galaxy clusters form at the intersections of these filaments.",
        "modules": ["Module 1", "Module 2"]
    },
    "Lookback Time": {
        "definition": "The time elapsed between when a photon was emitted by a distant object and when it is detected. Because light travels at a finite speed, looking further into space means looking back in time.",
        "modules": ["Module 1"]
    },
    "Scale Factor a(t)": {
        "definition": "A dimensionless function of time that represents the relative expansion of the universe. It relates the proper distance between two objects at time t to their comoving distance.",
        "modules": ["Module 4"]
    },
    "Comoving Distance": {
        "definition": "A distance measure between two points in the universe that remains constant over time if the objects are moving solely due to the Hubble flow (expansion of the universe).",
        "modules": ["Module 4"]
    },
    "Friedmann Equations": {
        "definition": "A set of equations derived from General Relativity that describe the expansion of the universe in terms of its matter and energy density.",
        "modules": ["Module 4"]
    },
    "Dark Energy": {
        "definition": "An unknown form of energy that permeates all of space and tends to accelerate the expansion of the universe. It makes up about 68% of the total energy density of the universe today.",
        "modules": ["Module 4"]
    },
    "Wien's Displacement Law": {
        "definition": "A physical law stating that the wavelength of peak emission for a blackbody is inversely proportional to its absolute temperature (λ_max ∝ 1/T).",
        "modules": ["Module 3"]
    },
    "Stefan-Boltzmann Law": {
        "definition": "A law stating that the total energy radiated per unit surface area of a black body is proportional to the fourth power of its thermodynamic temperature (E ∝ T⁴).",
        "modules": ["Module 3"]
    },
    "Surface of Last Scattering": {
        "definition": "The spherical shell around the observer from which the CMB photons were emitted during the epoch of Recombination. It represents the limit of the observable universe in light.",
        "modules": ["Module 3"]
    },
    "Gravitational Lensing": {
        "definition": "The bending of light from a distant source as it passes near a massive object (like a galaxy cluster) due to the curvature of spacetime. It is a key tool for detecting Dark Matter.",
        "modules": ["Module 2"]
    },
    "Bullet Cluster": {
        "definition": "A system of two colliding galaxy clusters that provides one of the strongest pieces of observational evidence for the existence of Dark Matter, as the separation between X-ray gas and gravitational mass is observed.",
        "modules": ["Module 2"]
    },
    "WIMPs": {
        "definition": "Weakly Interacting Massive Particles. A leading theoretical candidate for Cold Dark Matter particles that interact only via gravity and the weak nuclear force.",
        "modules": ["Module 2"]
    },
    "Jeans Instability": {
        "definition": "A condition under which an interstellar gas cloud becomes unstable and collapses under its own gravity to form stars or structure, occurring when the internal gas pressure is not strong enough to counteract gravity.",
        "modules": ["Module 2"]
    },
   

    "Angular Size": {
        "definition": "The apparent size of an object as seen from an observer, measured as an angle (usually in degrees or arcseconds). In the simulation, it relates the physical size of an object to its distance.",
        "modules": ["Module 1"]
    },
    "Arcsecond": {
        "definition": "A unit of angular measurement equal to 1/3600 of a degree. It is the fundamental unit used to define the Parsec via the parallax method.",
        "modules": ["Module 1"]
    },
    "Distance Modulus (μ)": {
        "definition": "The difference between the apparent magnitude (m) and absolute magnitude (M) of an astronomical object (μ = m - M). It is a direct measure of distance used in the Supernova plots.",
        "modules": ["Module 4"]
    },
    "Density Parameter (Ω)": {
        "definition": "The ratio of the actual density of a substance (like Matter or Dark Energy) to the critical density of the universe. The sum of all Omega parameters determines the geometry of the universe.",
        "modules": ["Module 4", "Module 3"]
    },
    "Hubble Constant (H₀)": {
        "definition": "The unit describing how fast the universe is expanding at the current time. Its value (approx 70 km/s/Mpc) sets the scale and age of the universe.",
        "modules": ["Module 4"]
    },
    "Accelerating Universe": {
        "definition": "The observation that the expansion rate of the universe is increasing over time, rather than slowing down due to gravity. This discovery implies the existence of Dark Energy.",
        "modules": ["Module 4"]
    },
    "Isotropy": {
        "definition": "The quality of looking the same in all directions. The CMB is isotropic to one part in 100,000, indicating the universe was extremely uniform in its early stages.",
        "modules": ["Module 1", "Module 3"]
    },
    "Anisotropy": {
        "definition": "Tiny temperature fluctuations in the Cosmic Microwave Background (CMB). These variations represent the 'seeds' of density from which galaxies and large-scale structures eventually formed.",
        "modules": ["Module 3"]
    },
    "Deuterium": {
        "definition": "An isotope of hydrogen containing one proton and one neutron. Its abundance, determined during Big Bang Nucleosynthesis, is a sensitive probe of the density of ordinary matter in the early universe.",
        "modules": ["Module 3"]
    },
    "Velocity Dispersion (σ)": {
        "definition": "The statistical spread of velocities of galaxies within a cluster. Higher dispersion indicates a deeper gravitational potential well and thus more mass (including Dark Matter).",
        "modules": ["Module 2"]
    },
    "Galactic Halo": {
        "definition": "A large, spherical region extending far beyond the visible disk of a galaxy. It contains sparsely distributed stars, globular clusters, and the majority of the galaxy's Dark Matter.",
        "modules": ["Module 2"]
    },
    "Mass-to-Light Ratio": {
        "definition": "The ratio of the total mass of an object to its total luminosity. A high ratio (like in galaxy clusters) indicates the presence of significant amounts of non-luminous Dark Matter.",
        "modules": ["Module 2"]
    },


    "Parallax": {
        "definition": "The apparent displacement of an object when viewed from two different lines of sight. It is the fundamental method for measuring distances to nearby stars (Distance Ladder base).",
        "modules": ["Module 1"]
    },
    "Observable Universe": {
        "definition": "The spherical region of the universe comprising all matter that can be observed from Earth at the present time, because electromagnetic radiation from these objects has had time to reach the Solar System and the Earth since the beginning of the cosmological expansion.",
        "modules": ["Module 1"]
    },
    "Type Ia Supernova": {
        "definition": "A specific type of supernova that occurs in binary systems (white dwarf explosion). They serve as the primary 'Standard Candles' for measuring large cosmic distances because they explode with a consistent peak luminosity.",
        "modules": ["Module 4"]
    },
    "Flat Universe": {
        "definition": "A universe geometry where the density parameter Ω equals exactly 1. In such a universe, parallel lines never meet and the angles of a triangle sum to 180 degrees. It is the geometry supported by current CMB observations.",
        "modules": ["Module 4", "Module 3"]
    },
    "Spectral Radiance": {
        "definition": "The amount of energy emitted by a surface at a specific wavelength per unit area, per unit solid angle, per unit time. This is the quantity calculated by Planck's Law for the Blackbody spectrum.",
        "modules": ["Module 3"]
    },
    "Photon Decoupling": {
        "definition": "The specific physical event during the Recombination epoch when photons ceased interacting frequently with matter (scattering off free electrons) and began to travel freely through space, forming the CMB.",
        "modules": ["Module 3"]
    },
    "Intracluster Medium (ICM)": {
        "definition": "The superheated plasma (gas) that permeates a galaxy cluster. It emits X-rays and constitutes the majority of the baryonic mass in a cluster, but is still outweighed by Dark Matter.",
        "modules": ["Module 2"]
    },
    "Gravitational Potential": {
        "definition": "The work done per unit mass to bring a small object from infinity to a point in a gravitational field. In the NFW profile, the potential well created by Dark Matter dictates how stars and gas move.",
        "modules": ["Module 2"]
    },
    "Homogeneity": {
        "definition": "The property of being uniform in composition or character throughout. In cosmology, it means the universe looks roughly the same at every location on large scales (part of the Cosmological Principle).",
        "modules": ["Module 1"]
    }
    
}

def main_layout(title: str):
   
    set_background()

    if not hasattr(ui, "_aria_landmarks_added"):
        ui.element('header').props('role=banner aria-label=Header')
        ui.element('nav').props('role=navigation aria-label=Nav')
        ui.element('main').props('role=main aria-label=Content')
        ui.element('footer').props('role=contentinfo aria-label=Footer')
        ui._aria_landmarks_added = True

  
 
    user_chat_history = []

    with ui.dialog() as ai_dialog, ui.card().classes('w-full max-w-4xl h-[85vh] flex flex-col p-0 overflow-hidden shadow-2xl'):
        
      
        with ui.column().classes('w-full px-6 py-4 !bg-gradient-to-r from-indigo-900 to-purple-900 shadow-md z-10'):
            with ui.row().classes('w-full justify-between items-center'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('auto_awesome', size='md').classes('text-green-400')
                    ui.label('AI Tutor 🤖').classes('text-2xl font-extrabold text-white tracking-wide').props('role=heading aria-level=2')
                
              
                ui.button(icon='close', on_click=ai_dialog.close).props('flat round text-color="white"').classes('hover:scale-110 transition-transform')
            
            ui.markdown(f"*Current Topic:* **{title}**").classes('text-indigo-200 text-sm mt-1')

       
        ai_scroll = ui.scroll_area().classes('w-full flex-grow p-6 bg-slate-50 shadow-inner')
        with ai_scroll:
            ai_response = ui.column().classes('w-full gap-4 pb-4') 

      
        with ui.row().classes('w-full px-6 py-4 bg-white items-center gap-3 border-t border-slate-200 z-10'):
            
          
            ai_q_input = aria_input(
                label='Ask a question about physics or cosmology...', 
                aria_label="Ask AI"
            ).classes('flex-grow text-lg').props('outlined rounded bg-color="black" label-color="primary"')
            
            
            aria_button('Ask', 'Ask AI', on_click=lambda: ask_ai(
                question=ai_q_input.value, 
                response_container=ai_response, 
                scroll_area=ai_scroll,         
                text_input_box=ai_q_input,      
                chat_history=user_chat_history 
            )).classes("!bg-green-600 hover:!bg-green-500 text-black font-bold px-8 py-3 rounded-full shadow-md transition-all")
            
          
            ai_q_input.on('keydown.enter', lambda: ask_ai(
                ai_q_input.value, ai_response, ai_scroll, ai_q_input, user_chat_history
            ))

        #with ui.row().classes('w-full justify-end mt-2'):
            #aria_button('Close', 'Close', on_click=ai_dialog.close).classes("!bg-orange-500 text-white font-bold")

    with ui.dialog() as glossary_dialog, ui.card().classes('w-full max-w-4xl max-h-[85vh] flex flex-col p-0 overflow-hidden'):
        
        
        with ui.column().classes('w-full p-6 bg-slate-100 border-b border-slate-300'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('📖 Physics Glossary').classes('text-2xl font-bold text-teal-800').props('role=heading aria-level=2')
                ui.icon('close', size='sm').classes('cursor-pointer text-slate-500 hover:text-red-600').on('click', glossary_dialog.close)
            
            ui.label('Search for definitions and concepts used in the modules.').classes('text-slate-600 mb-2')
            
           
          
            search_input = aria_input(
                label='Search term...', 
                aria_label="Filter glossary terms"
            ).classes('w-full text-lg bg-white rounded-lg')
            
            search_input.props('input-class="text-black" input-style="color: black" label-color="black" outlined')
            
            with search_input.add_slot('prepend'):
                ui.icon('search').classes('text-slate-500')

          
            glossary_container = ui.scroll_area().classes('w-full flex-grow p-4 bg-slate-50')

       
            def update_glossary_list(filter_text=""):
                glossary_container.clear()
            
                filter_text = (filter_text or "").lower().strip()
                
                with glossary_container:
                    sorted_terms = sorted(GLOSSARY_DATA.keys())
                    count = 0
                    
                    for term in sorted_terms:
                        data = GLOSSARY_DATA[term]
                        definition = data["definition"]
                        modules = data["modules"]
                        
                        match_term = filter_text in term.lower()
                        match_def = filter_text in definition.lower()
                        match_mod = any(filter_text in m.lower() for m in modules)

                        if match_term or match_def or match_mod:
                            count += 1
                            with ui.expansion(term).classes(
                                'w-full mb-2 border border-teal-200 rounded-lg bg-white shadow-sm hover:shadow-md transition-all'
                            ).props('header-class="text-lg font-bold text-teal-900 bg-teal-50/50" expand-icon-class="text-teal-600"'):
                                
                                with ui.column().classes('p-4 text-slate-800'):
                                    ui.markdown(f"**Definition:** {definition}").classes('text-base leading-relaxed')
                                    
                                    if modules:
                                        with ui.row().classes('mt-2 gap-2'):
                                            for mod in modules:
                                                ui.badge(mod, color='teal').props('outline')

                    if count == 0:
                        with ui.column().classes('w-full items-center justify-center mt-10 text-slate-400'):
                            ui.icon('search_off', size='4em')
                            ui.label(f"No results found for '{filter_text}'").classes('text-lg')

            search_input.on_value_change(lambda e: update_glossary_list(e.value))

            update_glossary_list()

      

                    
    with ui.dialog() as refl_dialog, ui.card().style('width: 100%; max-width: 900px; height: 85vh; padding: 0; display: flex; flex-direction: column; background-color: white; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);'):
        
    
        with ui.row().style('width: 100%; background-color: #0f766e; padding: 16px 24px; align-items: center; justify-content: space-between; flex-shrink: 0; z-index: 10;'):
            with ui.row().style('align-items: center; gap: 12px;'):
                ui.icon('edit_note', size='md').style('color: #6ee7b7;')
                ui.label('Reflections Manager').style('font-size: 1.5rem; font-weight: bold; color: white; margin: 0;')
            ui.button(icon='close', on_click=refl_dialog.close).props('flat round text-color="white"')

        
        with ui.row().style('width: 100%; background-color: #f8fafc; padding: 12px 24px; border-bottom: 1px solid #cbd5e1; flex-shrink: 0;'):
            ui.label('📜 Previous Reflections').style('font-size: 1.1rem; font-weight: bold; color: #1e293b;')
            
       
        log_scroll = ui.scroll_area().style('width: 100%; flex-grow: 1; padding: 24px; background-color: #f1f5f9;')
        with log_scroll:
            log_container = ui.column().style('width: 100%; gap: 8px;')

        def refresh_logs():
            log_container.clear()
            with log_container:
                if not app_data['reflection_log']:
                    ui.label("No reflections yet.").style("color: #64748b; font-style: italic;")
                else:
                    for entry in reversed(app_data['reflection_log']):
                        ui.markdown(f"{entry}").style('color: #1e293b; padding-bottom: 8px; border-bottom: 1px solid #cbd5e1; width: 100%; margin-bottom: 8px;')

        refresh_logs() 

       
        with ui.column().style('width: 100%; padding: 20px 24px; background-color: #e2e8f0; border-top: 4px solid #0d9488; flex-shrink: 0; z-index: 10;'):
            ui.label('✍️ New Reflection').style('font-size: 1.1rem; font-weight: bold; color: #1e293b; margin-bottom: 8px;')
            
           
            refl_input = aria_textarea(
                'Write your thoughts, doubts, or discoveries here...', 
                "Reflection"
            ).style('width: 100%; background-color: white; border-radius: 4px;').props('outlined input-class="text-black"')
            
            def save_refl():
                submit_reflection(title, refl_input.value)
                refl_input.value = ""
                refresh_logs() 
                log_scroll.scroll_to(percent=0.0)

          
            with ui.row().style('width: 100%; justify-content: space-between; align-items: center; margin-top: 16px;'):
                aria_button('Close', 'Close', on_click=refl_dialog.close).style('background-color: #ef4444; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold;')
                aria_button('ADD REFLECTION', 'Save', on_click=save_refl).style('background-color: #0d9488; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold;')
    
    with ui.dialog() as credits_dialog, ui.card().classes('w-full max-w-md'):
        ui.label('🏆 Credits').classes('text-2xl font-bold text-purple-700').props('role=heading aria-level=2')
        
        with ui.column().classes('text-lg text-white-800 leading-relaxed gap-1 w-full').props('role=document aria-live=polite'):
            ui.html('<b>Developer:</b> Eleonora Panini')
            ui.html('<b>Supervisors:</b> Prof. Enrico Bertuzzo, Prof. Guido Goldoni')
            ui.html('<b>Affiliation:</b> University of Modena and Reggio Emilia, Department of Physics')
            ui.html('<b>Front-End:</b> NiceGUI')
            ui.html('<b>Programming Language:</b> Python')
            ui.html('<b>Platform Hosting:</b> Hugging Face Spaces')
            ui.html('<b>AI Engine:</b> Google Gemini ')
            ui.html('<b>Data Sources & Datasets:</b>').classes('mt-2')
            
            
            with ui.column().classes('pl-4 gap-2 text-base w-full'):
                ui.link('• NASA Solar System Dynamics (Planets)', 'https://ssd.jpl.nasa.gov/planets/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• SPARC Database (Rotation Curves)', 'http://astroweb.cwru.edu/SPARC/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• SDSS & Kaggle (Coma Cluster)', 'https://www.kaggle.com/datasets/mertalkan98/coma-cluster', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• Southern Abell Redshift Survey', 'https://vizier.u-strasbg.fr/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• ESA Gaia Archive (Stars & HR)', 'https://gea.esac.esa.int/archive/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• ESA Planck Legacy Archive (CMB)', 'https://pla.esac.esa.int/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• Supernova Cosmology Project', 'http://supernova.lbl.gov/Union/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• SIMBAD & Aladin Sky Atlas', 'https://simbad.u-strasbg.fr/', new_tab=True).classes('text-blue-600 hover:underline')
                ui.link('• NIST Atomic Spectra Database', 'https://physics.nist.gov/PhysRefData/ASD/lines_form.html', new_tab=True).classes('text-blue-600 hover:underline')

       
        with ui.row().classes('w-full justify-end mt-4'):
            aria_button('Close', 'Close', on_click=credits_dialog.close).classes("!bg-orange-600 text-white font-bold")
    with ui.dialog() as kahoot_dialog, ui.card().classes('p-6 w-full max-w-[500px] text-center'):
        ui.label('🏆 Final Challenge: Quiz!').classes('text-2xl font-bold mb-4 text-purple-600')
        
        with ui.column().classes('text-left w-full gap-2 mb-6'):
           ui.markdown('''
**Instructions:**

* **Enter the Kahoot website and insert the Game PIN** provided by your teacher.
* **Register** with a recognizable username.
* **Fast and Accurate**: You have **30 seconds** per question.
* **Scoring**: Points are awarded based on correct answers and your response speed.

*Good luck and have fun!* 🚀
''').classes('text-white-700').props('role=document aria-live=polite')
        
        with ui.row().classes('w-full justify-center gap-4'):
          
            aria_button('Go to Kahoot.it', "Open Kahoot website", 
                        on_click=lambda: ui.run_javascript('window.open("https://kahoot.it/", "_blank")')) \
                .classes('!bg-green-600 hover:!bg-green-700 text-white font-bold py-2 px-6 rounded-lg')
            
            aria_button('Close', "Close instructions", on_click=kahoot_dialog.close) \
                .classes('!bg-orange-400 hover:!bg-orange-500 text-white py-2 px-4 rounded-lg text-sm')

    drawer = ui.left_drawer(value=False).props('width=450 overlay behavior="mobile"') 
    with drawer:
      
        ui.label('🔭 School Alignment').classes('w-full text-left px-6 mt-4 text-xl font-bold text-gray-300 uppercase tracking-widest')
        
     
        btn_style = 'w-[85%] ml-6 px-6 text-lg font-bold mb-3 !bg-slate-700/80 hover:!bg-slate-600 border-l-4 border-cyan-400 text-white rounded-r-lg'
        
        aria_button('🧭 Physics Link', "Physics", on_click=safe_click(lambda: aria_navigate('/physics-links', 'Physics'))).classes(btn_style).props('align="left"')
        aria_button('🧪 Curriculum', "Curriculum", on_click=safe_click(lambda: aria_navigate('/physics-program', 'Curriculum'))).classes(btn_style).props('align="left"')

        ui.separator().classes('my-4 bg-gray-600')
    
        ui.label('🛠️ Tools').classes('w-full text-left px-6 text-xl font-bold text-gray-300 uppercase tracking-widest')

       
        tool_btn_style = 'w-[85%] ml-6 px-6 text-lg font-bold mb-3 !bg-green-600 hover:!bg-green-500 border-l-4 border-green-300 text-white rounded-r-lg'

        aria_button('🤖 AI Tutor', "AI", on_click=ai_dialog.open).classes(tool_btn_style).props('align="left"')
        aria_button('✍️ Reflections', "Reflections", on_click=refl_dialog.open).classes(tool_btn_style).props('align="left"')
        aria_button('📖 Glossary', "Open Physics Glossary", on_click=glossary_dialog.open).classes(tool_btn_style).props('align="left"')
       
        def toggle_magnifier():
            if mag_btn.text == 'Magnifier':
                mag_btn.text = 'Pointer'          
                mag_btn.props('icon=mouse align="left"')       
                ui.run_javascript("document.body.classList.add('magnifier-active')")
                accessible_notify("Magnifier Mode ON: Hover over elements to zoom", "info")
            else:
                mag_btn.text = 'Magnifier'     
                mag_btn.props('icon=zoom_in align="left"')    
                ui.run_javascript("document.body.classList.remove('magnifier-active')")
                accessible_notify("Pointer Mode Active",type_="warning")

        mag_btn = aria_button(
            text='Magnifier', 
            label="Toggle Magnifier Tool", 
            on_click=toggle_magnifier,
            icon='zoom_in' 
        ).classes(tool_btn_style).props('align="left"')
       
        aria_button('🎮 Kahoot Quiz', "Open Kahoot Instructions and Link", on_click=kahoot_dialog.open).classes(tool_btn_style).props('align="left"')
        
        ui.separator().classes('my-4 bg-gray-600')
        
        ui.label('📝 Teacher Materials').classes('w-full text-left px-6 text-xl font-bold text-gray-300 uppercase tracking-widest')
        def show_bibliography_dialog():
            with ui.dialog() as bib_dialog, ui.card().classes('w-full max-w-4xl p-6 bg-white').props('role=dialog aria-label="Open Educational Resources Bibliography"'):
                
                with ui.row().classes('w-full justify-between items-center mb-4 border-b pb-2'):
                    ui.label('Open Educational Resources ').classes('text-2xl font-bold text-gray-800').props('role=heading aria-level=2')
                    ui.button(icon='close', on_click=bib_dialog.close).props('flat round aria-label="Close dialog"').classes('text-gray-800')
                
                with ui.scroll_area().classes('w-full h-[65vh] pr-4'):
                    html_info_box('''
            <div class="text-base text-gray-800 leading-relaxed">
                
                <h3 class="text-xl font-bold text-indigo-700 mt-2 mb-3 border-b-2 border-indigo-200 pb-1">🌑 Dark Matter & Galaxy Dynamics</h3>
                <ul class="list-disc pl-5 space-y-4 mb-8">
                    <li>
                        <a href="https://resources.perimeterinstitute.ca/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">Perimeter Institute - The Mystery of Dark Matter</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Dark Matter, Gravity, Modern Cosmology.</div>
                        <div class="ml-2"><b>Description:</b> Excellent educational modules designed to guide students through the scientific inquiry process. Includes hands-on activities focusing on the indirect evidence of dark matter.</div>
                    </li>
                    <li>
                        <a href="https://rubinobservatory.org/explore/education" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">Rubin Observatory - Education Investigations</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Galaxy Rotation Curves, Dark Matter, Expansion of the Universe.</div>
                        <div class="ml-2"><b>Description:</b> Online interactive investigations allowing students to work with real astronomical datasets.</div>
                    </li>
                </ul>

                <h3 class="text-xl font-bold text-indigo-700 mt-6 mb-3 border-b-2 border-indigo-200 pb-1">🇮🇹 Italian & European Institutional Resources</h3>
                <ul class="list-disc pl-5 space-y-4 mb-8">
                    <li>
                        <a href="https://edu.inaf.it/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">EduINAF - National Institute for Astrophysics</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Astronomy, Cosmology, Space Exploration.</div>
                        <div class="ml-2"><b>Description:</b> The official educational portal of INAF. Offers didactic paths, interactive tools, and comprehensive lesson plans for physics teachers.</div>
                    </li>
                    <li>
                        <a href="https://www.aif.it/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">AIF - Associazione per l'Insegnamento della Fisica</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Physics Pedagogy, Classical and Modern Physics.</div>
                        <div class="ml-2"><b>Description:</b> Provides peer-reviewed articles, laboratory activities, and teaching methodologies to effectively translate complex concepts into accessible classroom modules.</div>
                    </li>
                    <li>
                        <a href="https://www.esa.int/Education/Teachers_Corner" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">ESA Education - Teach with Space</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Space Science, Satellite Data (Planck, Gaia).</div>
                        <div class="ml-2"><b>Description:</b> The European Space Agency provides high-quality teacher guides and student activities focusing on real space mission data.</div>
                    </li>
                </ul>

                <h3 class="text-xl font-bold text-indigo-700 mt-6 mb-3 border-b-2 border-indigo-200 pb-1">📚 Open Textbooks, Lecture Notes & Slides</h3>
                <ul class="list-disc pl-5 space-y-4 mb-8">
                    <li>
                        <a href="http://www.damtp.cam.ac.uk/user/tong/cosmo.html" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">David Tong: Lectures on Cosmology (Cambridge University)</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Expansion of the Universe, CMB, Dark Matter, Inflation.</div>
                        <div class="ml-2"><b>Description:</b> Lecture notes with mathematical derivations and clear explanations of dark matter and cosmological expansion.</div>
                    </li>
                    <li>
                        <a href="https://ocw.mit.edu/courses/8-286-the-early-universe-fall-2013/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">MIT OpenCourseWare - The Early Universe (Prof. Alan Guth)</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Big Bang, Dark Matter, Inflationary Cosmology.</div>
                        <div class="ml-2"><b>Description:</b> Lecture slides, comprehensive lecture notes, and assignments of cosmic inflation.</div>
                    </li>
                    <li>
                        <a href="https://openstax.org/details/books/astronomy-2e" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">OpenStax Astronomy 2e</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> General Astronomy, Astrophysics, Cosmology.</div>
                        <div class="ml-2"><b>Description:</b> A comprehensive open-source textbook with multiple accessible formats.</div>
                    </li>
                    <li>
                        <a href="https://www.teachastronomy.com/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">Teach Astronomy (Prof. Chris Impey)</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Stellar Evolution, Galaxies, Universe.</div>
                        <div class="ml-2"><b>Description:</b> An open-access textbook and video lecture repository.</div>
                    </li>
                </ul>

                <h3 class="text-xl font-bold text-indigo-700 mt-6 mb-3 border-b-2 border-indigo-200 pb-1">🔭 Interactive Simulations & Gamification</h3>
                <ul class="list-disc pl-5 space-y-4 mb-4">
                    <li>
                        <a href="https://phet.colorado.edu/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">PhET Interactive Simulations - Gravity & Orbits</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Orbital Mechanics, Gravitation, Mass Interactions.</div>
                        <div class="ml-2"><b>Description:</b> Highly accessible visual simulations, excellent for introducing Kepler's laws.</div>
                    </li>
                    <li>
                        <a href="https://voyages.sdss.org/" target="_blank" class="text-blue-600 hover:text-blue-800 hover:underline font-bold text-lg">SDSS Voyages</a>
                        <div class="ml-2 mt-1"><b>Topic:</b> Galaxy Classification, Spectroscopy.</div>
                        <div class="ml-2"><b>Description:</b> An educational platform leveraging the Sloan Digital Sky Survey database to create astronomical investigations.</div>
                    </li>
                </ul>
            </div>
            ''')
            bib_dialog.open()
        def open_slides_cosmo():
            ui.run_javascript('window.open("/slides/Cosmology_Notes.pdf", "_blank")')
        def open_activities():
            ui.run_javascript('window.open("/slides/activities.pdf", "_blank")')
        def open_astronomy():
            ui.run_javascript('window.open("/slides/Astrophysics_Notes.pdf", "_blank")')
        def DM_slides():
            ui.run_javascript('window.open("/slides/Dark_matter.pdf", "_blank")')
            
        
        mat_btn_style = 'w-[85%] ml-6 px-6 text-lg font-bold mb-3 !bg-blue-600 hover:!bg-blue-500 border-l-4 border-blue-300 text-white rounded-r-lg'
        aria_button(
    "📚OER & Bibliography ", 
    "View Open Educational Resources on Cosmology and Dark Matter", 
    on_click=show_bibliography_dialog
).classes(mat_btn_style).props('align="left"')
        aria_button('📄Cosmology Intro', 'Open Introductory material: Cosmology', on_click=open_slides_cosmo).classes(mat_btn_style).props('align="left"')
        aria_button('📄Astronomy Intro', 'Open Introductory material: Astronomy', on_click=open_astronomy).classes(mat_btn_style).props('align="left"')
        aria_button('🧪Dark Matter Module', 'Open Dark matter physics', on_click=open_activities).classes(mat_btn_style).props('align="left"')
        aria_button(' 🧪DM Slides', 'Open Dark Matter Slides', 
                                    on_click=DM_slides
                        ).classes(mat_btn_style).props('align="left"')
                                
    with ui.dialog() as intro, ui.card().classes('p-4 w-full text-lg max-w-[1200px] overflow-x-auto').style('background-color: #0f172a !important; color: white; border: 1px solid #334155;'):
            ui.html(r"""
    <div style="font-family: 'Roboto', sans-serif; color: #ffffff;">
        
        <h3 style="color: #60a5fa; font-weight: 800; margin-bottom: 12px; border-bottom: 2px solid #93c5fd; padding-bottom: 8px;">
            Welcome to Cosmo-Edu Lab!
        </h3>
        
        
        <h4 style="color: #38bdf8; font-weight: bold; margin-top: 15px; margin-bottom: 8px;">
            🚀 How to use the App
        </h4>
        <ul style="margin: 0; padding-left: 20px; line-height: 1.6; list-style-type: disc;">
            <li><b>Start your Journey</b> 
            <ul style="list-style-type: circle; margin-left: 15px; margin-top: 4px;">
            
            <li>Select a module from the main menu to begin.</li>
             <li>Toggle interactive tools with the top-left menu button</li>
       
              </ul>
               </li>
            <li><b>School Alignment</b> 
             <ul style="list-style-type: circle; margin-left: 15px; margin-top: 4px;">
            <li><b>Physics Links</b>: See how each module maps to key physics concepts.</li>
            <li><b>Curriculum</b>: Connect cosmology topics with your standard physics curriculum.</li>
                </ul>
                 </li>
            <li><b>Tools</b>
                <ul style="list-style-type: circle; margin-left: 15px; margin-top: 4px;">
                    <li>🔍 <b>Magnifier:</b> Hover over plots to zoom in on details.</li>
                    <li>🤖 <b>AI Tutor:</b> Ask questions to get real-time explanations.</li>
                    <li>📖 <b>Glossary:</b> Look up difficult astronomical terms.</li>
                    <li>📝 <b>Reflection Log:</b> Save your answers and thoughts.</li>
                </ul>
       
          </li>
            <li><b>Supporting Materials:</b> 
             <ul style="list-style-type: circle; margin-left: 15px; margin-top: 4px;">
            <li>📄 <b>Cosmology Intro:</b> Learn about the universe and its evolution.</li>
            <li>📄 <b>Astronomy Intro:</b> Explore the basics of stars, galaxies, and more.</li>
            <li>🧪 <b>Lab Activities:</b> Engage with hands-on exercises to deepen your understanding.</li>
        </ul>
         </li>
  <li><b>Modules:</b> 
  <li>Contain data analysis tools with predefined parameters which you can change with appropriate sliders.</li>
   <ul style="list-style-type: circle; margin-left: 15px; margin-top: 4px;">
   <li><b>Module 1:</b> Focuses on measuring distances in the universe and understanding its large-scale structure.</li>
   <li><b>Module 2:</b> Dives into the evidence for Dark Matter and how it shapes galaxies and clusters.</li>
   <li><b>Module 3:</b> Explores the Cosmic Microwave Background and what it tells us about the early universe.</li>
   <li><b>Module 4:</b> Examines the expansion of the universe and the Dark Energy driving it.</li>
   </ul>
  </li>
  
  
  
        <h4 style="color: #38bdf8; font-weight: bold; margin-top: 20px; margin-bottom: 8px;">
            🎨 Button Color Legend
        </h4>
        <div style="display: flex; flex-direction: column; gap: 14px; font-size: 0.95em;">
            
            <div style="display: flex; align-items: center;">
                <span style="background-color: #16a34a; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8em; min-width: 90px; text-align: center; margin-right: 12px; box-shadow: 0 1px 2px rgba(255,255,255,0.2);">
                    GREEN
                </span>
                <div style="line-height: 1.3;"><b>Action & Interactive:</b><br>Generate plots, Run exercises, External links.</div>
            </div>

            <div style="display: flex; align-items: center;">
                <span style="background-color: #2563eb; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8em; min-width: 90px; text-align: center; margin-right: 12px; box-shadow: 0 1px 2px rgba(255,255,255,0.2);">
                    BLUE
                </span>
                <div style="line-height: 1.3;"><b>Information:</b><br>Instructions, Datasets, Legends, Theory.</div>
            </div>

            <div style="display: flex; align-items: center;">
                <span style="background-color: #dc2626; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8em; min-width: 90px; text-align: center; margin-right: 12px; box-shadow: 0 1px 2px rgba(255,255,255,0.2);">
                    RED
                </span>
                <div style="line-height: 1.3;"><b>Accessibility:</b><br>Features for users with disabilities.</div>
            </div>

            <div style="display: flex; align-items: center;">
                <span style="background-color: #9333ea; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8em; min-width: 90px; text-align: center; margin-right: 12px; box-shadow: 0 1px 2px rgba(255,255,255,0.2);">
                    PURPLE
                </span>
                <div style="line-height: 1.3;"><b>Curiosities:</b><br>"Did you know?" facts and fun trivia.</div>
            </div>

            <div style="display: flex; align-items: center;">
                <span style="background-color: #f97316; color: black; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.8em; min-width: 90px; text-align: center; margin-right: 12px; box-shadow: 0 1px 2px rgba(255,255,255,0.2);">
                    ORANGE
                </span>
                <div style="line-height: 1.3;"><b>Navigation:</b><br>Close dialogs and interactive windows.</div>
            </div>

        </div>

        <p style="margin-top: 25px; font-style: italic; text-align: center; color: #cbd5e1;">
            Enjoy your exploration of the cosmos!
        </p>
    </div>
""").props('role=document aria-live=polite')
            aria_button("Close", "close the box",on_click=lambda:intro.close()).classes("!bg-orange-500 hover:!bg-orange-700 text-white font-bold py-2 px-4 rounded")
    
    def show_constants_dialog():
        with ui.dialog() as constants_dialog, ui.card().classes('p-6 w-full overflow-y-auto').props('role=dialog aria-label="Constants and Units"'):
            ui.label("Constants and Units").classes("text-2xl font-bold mb-4")
            
            with ui.tabs().classes('w-full') as tabs:
                t_const = ui.tab('Physical Constants')
                t_astro = ui.tab('Astronomical Units')
                t_cosmo = ui.tab('Cosmological Values')

            with ui.tab_panels(tabs, value=t_const).classes('w-full'):
                # TAB 1: PHYSICAL CONSTANTS
                with ui.tab_panel(t_const):
                    ui.html("""
                    <table class="w-full text-sm text-left border-collapse">
                    <tr class="bg-black text-white font-bold">
                        <th class="p-2">Constant</th><th class="p-2">Symbol</th><th class="p-2">Value</th><th class="p-2">Unit</th>
                    </tr>
                    <tr class="border-b"><td>Speed of Light</td><td>c</td><td>2.9979 × 10^8</td><td>m/s</td></tr>
                    <tr class="border-b"><td>Gravitational Constant (SI)</td><td>G</td><td>6.674 × 10^-11</td><td>m³/(kg·s²)</td></tr>
                    <tr class="border-b"><td>Gravitational Constant (Galactic)</td><td>G</td><td>4.302 × 10^-6</td><td>kpc·(km/s)²·M☉⁻¹</td></tr>
                    <tr class="border-b"><td>Planck Constant</td><td>h</td><td>6.626 × 10^-34</td><td>J·s</td></tr>
                    <tr class="border-b"><td>Boltzmann Constant</td><td>k<sub>B</sub></td><td>1.380 × 10^-23</td><td>J/K</td></tr>
                    <tr class="border-b"><td>Proton Mass</td><td>m<sub>p</sub></td><td>1.672 × 10^-27</td><td>kg</td></tr>
                    <tr class="border-b"><td>Electron Mass</td><td>m<sub>e</sub></td><td>9.109 × 10^-31</td><td>kg</td></tr>
                    <tr class="border-b"><td>Elementary Charge</td><td>e</td><td>1.602 × 10^-19</td><td>C</td></tr>
                </table>
                    """)
                
                # TAB 2: ASTRONOMICAL UNITS
                with ui.tab_panel(t_astro):
                    ui.html("""
                   <table class="w-full text-sm text-left border-collapse">
                    <tr class="bg-black text-white font-bold">
                        <th class="p-2">Unit</th><th class="p-2">Symbol</th><th class="p-2">Conversion</th>
                    </tr>
                    <tr class="border-b"><td>Solar Mass</td><td>M☉</td><td>1.989 × 10^30 kg</td></tr>
                    <tr class="border-b"><td>Solar Radius</td><td>R☉</td><td>6.957 × 10^5 km</td></tr>
                    <tr class="border-b"><td>Solar Luminosity</td><td>L☉</td><td>3.828 × 10^26 W</td></tr>
                    <tr class="border-b"><td>Earth Radius</td><td>R⊕</td><td>6,371 km</td></tr>
                    <tr class="border-b"><td>Jupiter Radius</td><td>R<sub>J</sub></td><td>71,492 km</td></tr>
                    <tr class="border-b"><td>Astronomical Unit</td><td>AU</td><td>1.496 × 10^8 km</td></tr>
                    <tr class="border-b"><td>Parsec</td><td>pc</td><td>3.086 × 10^13 km</td></tr>
                    <tr class="border-b"><td>Light-year</td><td>ly</td><td>9.461 × 10^12 km</td></tr>
                </table>
                    """)

                # TAB 3: COSMOLOGICAL VALUES
                with ui.tab_panel(t_cosmo):
                    ui.html("""
                    <table class="w-full text-sm text-left border-collapse">
                    <tr class="bg-black text-white font-bold">
                        <th class="p-2">Parameter</th><th class="p-2">Value (approx)</th><th class="p-2">Unit</th>
                    </tr>
                    <tr class="border-b"><td>Hubble Constant</td><td>H₀ ≈ 70</td><td>km/s/Mpc</td></tr>
                    <tr class="border-b"><td>Matter Density Parameter</td><td>Ω<sub>m</sub> ≈ 0.3</td><td>-</td></tr>
                    <tr class="border-b"><td>Dark Energy Density</td><td>Ω<sub>Λ</sub> ≈ 0.7</td><td>-</td></tr>
                    <tr class="border-b"><td>Critical Density</td><td>ρ<sub>crit</sub></td><td>9.2 × 10^-27 kg/m³</td></tr>
                    <tr class="border-b"><td>CMB Temperature</td><td>T<sub>CMB</sub></td><td>2.725 K</td></tr>
                    <tr class="border-b"><td>Age of the Universe</td><td>t₀</td><td>~13.8 Gyr</td></tr>
                    <tr class="border-b"><td>Mass-Energy Equiv.</td><td>1 kg</td><td>~5.61 × 10^35 eV</td></tr>
                </table>
                    """)
            
            aria_button("Close", "Close constants dialog", on_click=constants_dialog.close).classes("!bg-orange-500 mt-4")
        
        constants_dialog.open()
        
    with ui.header(elevated=True).classes('!bg-orange-900/30 text-white items-center justify-between q-py-sm backdrop-blur-md'):
        def show_ecosystem_map():
                
            with ui.dialog() as map_dialog, ui.card().classes('w-full max-w-5xl p-6 bg-white').props('role=dialog aria-label="Ecosystem Map"'):
                
                with ui.row().classes('w-full justify-between items-center mb-4 border-b pb-2'):
                    ui.label('Cosmo-Edu Lab Ecosystem Map').classes('text-2xl font-bold text-gray-800')
                    ui.button(icon='close', on_click=map_dialog.close).props('flat round').classes('text-gray-800')
                
            
                ui.mermaid('''
     graph TD
    Root("Cosmo-Edu Lab "):::rootClass
    
    Mod1("Introduction to Cosmology"):::modClass
    IC1("<b>IC.1</b> Structures of the Universe"):::panelClass
    IC2("<b>IC.2</b> Astronomical Discoveries"):::panelClass
    IC3("<b>IC.3</b> Universe Timelines"):::panelClass
    IC4("<b>IC.4</b> Observational Instruments"):::panelClass
    IC5("<b>IC.5</b> Galaxies"):::panelClass
    IC6("<b>IC.6</b> Milky Way"):::panelClass
    IC7("<b>IC.7</b> Stars"):::panelClass
    IC8("<b>IC.8</b> Fundamental Particles"):::panelClass
    
    Mod2("Dark Matter (DM)"):::modClass
    DM1("<b>DM.1</b> Kepler's Laws"):::panelClass
    DM2("<b>DM.2</b> Galaxy Rotation Curves"):::panelClass
    DM3("<b>DM.3</b> Galaxy Mass and DM"):::panelClass
    DM4("<b>DM.4</b> Cluster Velocity"):::panelClass
    DM5("<b>DM.5</b> Cluster Mass and DM"):::panelClass
    
    %% --- Connessioni principali ---
    Root --> Mod1
    Root --> Mod2
    
    Mod1 --> IC1 --> IC2 --> IC3 --> IC4 --> IC5 --> IC6 --> IC7 --> IC8
    Mod2 --> DM1 --> DM2 --> DM3 --> DM4 --> DM5
    
    %% --- Moduli Pianificati ---
    subgraph Planned [Planned Modules]
        PM1("Redshift and<br>Universe Expansion"):::futureClass
        PM2("Universe History<br>and CMB"):::futureClass
    end
    
    %% --- IL TRUCCO DEI COLLEGAMENTI INVISIBILI ---
    %% 1. Diciamo a PM1 di stare sotto DM5 (senza freccia)
    DM5 ~~~ PM1
    %% 2. Diciamo a PM2 di stare sotto PM1 (senza freccia)
    PM1 ~~~ PM2
    
    %% --- Stili ---
    classDef rootClass fill:#dbeafe,stroke:#1e40af,stroke-width:2px,color:#000,font-weight:bold,font-size:16px,rx:10px,ry:10px
    classDef modClass fill:#dcfce7,stroke:#166534,stroke-width:2px,color:#000,font-weight:bold,rx:8px,ry:8px
    classDef panelClass fill:#ffedd5,stroke:#c2410c,stroke-width:1px,color:#000,rx:5px,ry:5px
    classDef futureClass fill:#f3f4f6,stroke:#9ca3af,stroke-width:1px,stroke-dasharray: 5 5,color:#6b7280,font-weight:bold
    style Planned fill:transparent,stroke:#9ca3af,stroke-width:2px,stroke-dasharray: 5 5,color:#6b7280,font-weight:bold
    ''')

            map_dialog.open()
        with ui.row().classes('items-center gap-3'): 
            aria_button2('Menu', 'Menu', on_click=lambda: drawer.toggle(), tooltip='Toggle menu') \
                .classes('text-lg font-bold !bg-slate-800/80 hover:!bg-slate-700 px-4')
            aria_button(
    "App Map", 
    "View the ecosystem map of Cosmo-Edu Lab modules", 
    on_click=show_ecosystem_map
).classes('text-lg font-bold !bg-blue-600 hover:!bg-blue-500 text-white px-4')
            aria_button('General Instructions', 'Open Instruction', on_click=lambda:[intro.open(),ui.run_javascript("MathJax.typesetPromise()")]) \
                .classes('text-lg font-bold !bg-blue-600 hover:!bg-blue-500 text-white px-4') 
       
            aria_button(
        "Constants & Units", 
        "Open reference sheet for constants and units", 
        on_click=show_constants_dialog
    ).classes('text-lg font-bold !bg-blue-600 hover:!bg-blue-500 text-white px-4')
            aria_button('🏆 Credits', "View Credits", on_click=credits_dialog.open) \
                .classes('text-lg font-bold !bg-blue-600 hover:!bg-blue-500 text-white px-4')
        ui.label(title).classes('text-2xl font-black tracking-widest text-white drop-shadow-md')
        
        with ui.row().classes('items-center gap-2'):
            aria_button('Home', "Home", on_click=safe_click(lambda: aria_navigate('/main', 'Home'))).props('icon=home flat round')
            aria_button('Back', "Back", on_click=safe_click(lambda: ui.run_javascript('window.history.back()'))).props('icon=arrow_back flat round')
            if app.storage.user.get('name'):
                aria_button('Logout', "Logout", on_click=safe_click(lambda: (app.storage.user.clear(), aria_navigate('/login', 'Login')))).props('icon=logout flat round ')
            else:
                aria_button('Login', "Login", on_click=safe_click(lambda: aria_navigate('/login', 'Login'))).props('icon=login flat round ')

    global_notification_area = ui.row().classes('w-full min-h-[70px] justify-center items-start z-50 pt-2')