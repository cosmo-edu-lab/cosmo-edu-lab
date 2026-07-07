# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.
import os
import io
import json
import nicegui
from nicegui import ui, app, run , client

import datetime
from groq import Groq
import urllib.parse
#from nicegui_toolkit import inject_layout_tool


from core import *
from layout import *

MODULES_LOCKED = os.getenv("MODULES_LOCKED", "True") == "True"
#inject_layout_tool()
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
def create_routes():
    
    @ui.page('/main')
    def main_menu():
   
        #ui.run_javascript('window.__nicegui__.requestTimeout = 5000')
        main_layout(" COSMO-EDU-LAB")
        
                
        
        with ui.column().classes('w-full items-center p-8'):
            ui.label(f"Welcome, {app.storage.user.get('name', 'Explorer')}!").classes('text-5xl font-extrabold text-white drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)] tracking-wide').props('role=note tabindex=0')


        
            
            ui.label('\"The cosmos is within us. We are made of star-stuff.\" — Carl Sagan').classes('italic text-3xl text-blue-100 mt-4 font-serif drop-shadow-md text-center ').props(' tabindex=0')
            

            with ui.grid(columns=4).classes('w-full justify-center gap-10 mt-8 mb-8 flex-wrap'):
                module_titles = [
                    "Introduction to Cosmology","Dark Matter","Universe History & CMB" ,"Redshift & Universe Expansion"
                ]
                star_style = (
                        'w-72 h-72 rounded-full flex flex-col items-center justify-center text-center p-4 '
                        'cursor-pointer transition-all duration-500 ease-in-out '
                        'bg-gradient-to-br from-yellow-100 via-yellow-200 to-yellow-500 ' # Effetto rilievo dorato/chiaro
                        'border-4 border-white/50 ' # Bordo semitrasparente
                        'shadow-[0_0_40px_rgba(255,223,0,0.6)] ' # Effetto Glow esterno
                        'hover:scale-110 hover:shadow-[0_0_60px_rgba(255,255,255,0.9)] hover:z-10' # Effetto hover
                    )
                
                for i, title in enumerate(module_titles, 1):
               
                    is_locked = (i in [3, 4]) and MODULES_LOCKED
                    
                   
                    current_style = star_style
                    if is_locked:
                        current_style = current_style.replace('cursor-pointer', 'cursor-not-allowed') + ' opacity-75 grayscale'
                    
                
                    card = ui.card().classes(current_style).props(
                        f'role=button tabindex=0 aria-label="{"Locked" if is_locked else "Go to"} module {i}: {title}"'
                    )
                    
                   
                    if is_locked:
                       
                        def show_coming_soon(e):
                            accessible_notify('Module under development: coming soon', type_='warning')
                        
                        card.on('click', show_coming_soon)
                        card.on('keydown.enter', show_coming_soon)
                    else:
                   
                        card.on('click', lambda i=i: (
                            ui.navigate.to(f'/module{i}'), 
                            ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")
                        ))
                        card.on('keydown.enter', lambda i=i: (
                            ui.navigate.to(f'/module{i}'), 
                            ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")
                        ))
                    
                    with card: 
                       
                        if is_locked:
                            ui.label('🔒').classes('text-6xl absolute top-4 opacity-50')
                        else:
                            ui.label(f'{i}').classes('text-6xl font-black text-yellow-600/30 absolute top-4')
                        
                        ui.label(title).classes('text-2xl font-semibold text-slate-800 leading-tight')
                
      
            
        ui.label    (
                "Cosmo-Edu Lab is your interactive platform to explore the Universe.\n\n"
                "Through an immersive journey, bridge the gap \n\n"
                "between classical physics and modern cosmology."
                
            ).classes('font-bold text-3xl text-blue-100 mt-4 drop-shadow-md text-center whitespace-pre-wrap w-full').props('role=heading aria-level=2 tabindex=0')
            
          
    @ui.page('/physics-links')
    def physics_links():
      
        main_layout("Classical Physics → Cosmology Connections")
        def make_wiki_links(text_list_str):
            if not text_list_str: return ""
            items = text_list_str.split(',')
            links = []
            for item in items:
                item = item.strip()
                if not item: continue
            
             
                safe_slug = urllib.parse.quote(item.replace(' ', '_'))
              
                full_url = f"https://en.wikipedia.org/wiki/{safe_slug}"
            
             
                link_html = (
                    f'<a href="{full_url}" target="_blank" '
                    f'class="text-cyan-300 hover:text-cyan-100 underline decoration-cyan-500 '
                    f'hover:decoration-cyan-100 transition-all font-medium" '
                    f'onclick="event.stopPropagation();">{item}</a>'
                )
                links.append(link_html)
            return ", ".join(links)
       
        table_data = [
          ("Introduction to Cosmology", 
           "Observable universe,Cosmic distance ladder,Unit of measurement,Astronomy,Microcosm–macrocosm analogy",
           "Universe,Galaxy,Star,Hertzsprung–Russell diagram,Elementary particle,Standard Model", 
           "/module1"),
           ("Dark Matter", 
           "Circular motion,Gravity,Conservation of energy,Centripetal force,Newton's law of universal gravitation,Kepler's laws of planetary motion,Kinetic energy,Gravitational potential energy", 
           "Galaxy rotation curve,Missing mass problem,Virial theorem,Dark matter,Abundance of the chemical elements", 
           "/module2"),
        
           
          ("Universe History & CMB", 
           "Thermodynamics,Thermal radiation,Black-body radiation,Adiabatic process,Heat capacity ratio,Stefan–Boltzmann law", 
           "Chronology of the universe,Cosmic microwave background,Planck's law,Galaxy formation and evolution,Matter,Energy density,Mass–energy equivalence,Photon", 
           "/module3"),
             ("Redshift & Universe Expansion", 
           "Doppler effect,Optics,Velocity,Wave,Frequency,Radiant flux,Wavelength,Relativistic speed",
           "Redshift,Cosmic distance ladder,Type Ia supernova,Luminosity,Hubble's law,Accelerating expansion of the universe,Expansion of the universe,Dark energy,Lambda-CDM model", 
           "/module4")
          
        ]
        
       
        rows = [
            {
                'classical': title, 
                'topics': make_wiki_links(classical), 
                'cosmo': make_wiki_links(cosmo),    
                'link': link
            } 
            for title, classical, cosmo, link in table_data
        ]
            
        with ui.card().classes('w-full !bg-slate rounded-3xl shadow-2xl p-0 overflow-hidden border-4 border-blue-500/30 mx-auto'):
            table = ui.table(
                columns=[
                    {'name': 'classical', 'label': 'Module', 'field': 'classical', 'align': 'left', 'classes': 'w-1/4 font-bold text-white-700'},
                    {'name': 'topics', 'label': 'Core Physics', 'field': 'topics', 'align': 'left', 'classes': 'w-1/3'},
                    {'name': 'cosmo', 'label': 'Cosmology', 'field': 'cosmo', 'align': 'left', 'classes': 'w-1/3'},
                ],
                rows=rows,
                row_key='classical'
            ).classes('w-full') \
                .props('flat separator="cell" hide-bottom role=table tabindex=0 aria-label="Connections between Classical Physics and Cosmology"') 

         
            table.add_slot('header', r'''
                    <q-tr :props="props" class="bg-gradient-to-r from-blue-700 to-cyan-600 text-white">
                        <q-th v-for="col in props.cols" :key="col.name" :props="props" 
                              style="font-size: 24px !important; line-height: 1.2 !important; font-weight: 800; white-space: normal !important;"
                              class="uppercase tracking-wider p-6 text-left">
                            {{ col.label }}
                        </q-th>
                    </q-tr>
                ''')

          
            table.add_slot('body', r'''
                    <q-tr :props="props">
                        <q-td key="classical" :props="props" 
                              style="font-size: 20px !important; line-height: 1.5 !important; white-space: normal !important; min-width: 200px; vertical-align: top;"
                              class="!text-white-900 p-6 font-bold">
                            {{ props.row.classical }}
                        </q-td>
                        
                        <q-td key="topics" :props="props" 
                              style="font-size: 20px !important; line-height: 1.5 !important; white-space: normal !important; min-width: 200px; vertical-align: top;"
                              class="!text-white-900 p-6">
                            <div v-html="props.row.topics"></div>
                        </q-td>
                        
                        <q-td key="cosmo" :props="props" 
                              style="font-size: 20px !important; line-height: 1.5 !important; white-space: normal !important; min-width: 200px; vertical-align: top;"
                              class="!text-white-900 p-6">
                            <div v-html="props.row.cosmo"></div>
                        </q-td>
                    </q-tr>
                ''')

      
        with ui.row().classes('mt-12 gap-8 justify-center w-full flex-wrap'):
            for title, classical, cosmo, link in table_data:
                module_num = title.split('.')[0] if '.' in title else "Link"
                display_num = str(table_data.index((title, classical, cosmo, link)) + 1)
                
               
                is_locked = (link in ['/module3', '/module4']) and MODULES_LOCKED
                btn_text = f"🔒 Module {display_num}" if is_locked else f"Go to Module {display_num}"
                
              
                btn = aria_button(btn_text, f"Go to {title}", 
                    on_click=safe_click(lambda l=link: accessible_notify('Module under development: coming soon', type_='warning') if l in ['/module3', '/module4'] else ui.navigate.to(l))
                ).classes('!bg-cyan-600 hover:!bg-cyan-500 text-white text-xl font-bold py-4 px-10 rounded-full shadow-[0_4px_0_rgb(21,94,117)] transition-transform active:translate-y-1 active:shadow-none border-2 border-white/20')
                
                
                if is_locked:
                    btn.classes('opacity-60 grayscale cursor-not-allowed')
    @ui.page('/physics-program')
    def physics_program():
    
        main_layout("Physics Curriculum → Cosmology Modules")
        with ui.column().classes('w-full items-center p-6'):
            
            ui.label("Physics Curriculum Integration").classes('text-5xl font-black text-white drop-shadow-lg mb-10 text-center').props('role=heading aria-level=1 tabindex=0')

            physics_curriculum = [
            ("1st–2nd Year", [
                ("Scalar and vectorial measures and units", "/module1"),
                ("Geometry optics: reflection and refraction", "/module4"),
                ("Thermal phenomena: temperature, heat, thermal balance", "/module3"),
                ("Mechanics: motion, forces, Newton’s laws", "/module2"),
            ]),
            ("3rd–4th Year", [
                ("Relativity: reference systems, principles", "/module4"),
                ("Conservation laws: energy, fluids", "/module2"),
                ("Gravitation: Kepler, Newton, cosmological systems", "/module2"),
                ("Thermodynamics: gas laws, kinetic theory", "/module3"),
                ("Waves: mechanical, interference, diffraction", "/module4"),
                ("Electromagnetism: fields, energy, potential", "/module4"),
            ]),
            ("5th Year", [
                ("Electromagnetism: induction, Maxwell, EM waves", "/module4"),
                ("Micro & macro cosmos: space-time, energy", "/module1"),
                ("Relativity: Einstein’s principles, dilation, contraction", "/module4"),
                ("Quantum light: Planck, photoelectric, De Broglie", "/module3"),
            ]),
        ]

        

                       
            with ui.column().classes('w-full max-w-5xl gap-8'):
                for year, topics in physics_curriculum:
                    
                    
                    with ui.expansion(year, icon='school').classes(
                        'w-full bg-gradient-to-r from-blue-50 to-white rounded-2xl overflow-hidden shadow-xl border border-blue-200 transition-all duration-300'
                    ).props(
                        f'header-class="text-3xl font-bold text-blue-900 bg-blue-100/80 p-6" '
                        f'expand-icon-class="text-blue-900 text-4xl" '
                        f'role=region aria-label="{year}"'
                    ):
                        ui.run_javascript(f"""
                const exp = document.querySelector('[aria-label="{year} physics topics"]');
                if (exp) exp.addEventListener('click', () => {{
                    exp.setAttribute('aria-expanded', exp.open);
                }});
            """)
                        with ui.column().classes('p-6 w-full !bg-white'):
                            for topic, link in topics:
                                if link:
                                   
                                    is_locked = (link in ['/module3', '/module4']) and MODULES_LOCKED
                                    btn_text = f"🔒 {topic}" if is_locked else topic

                                   
                                    btn = aria_button(
                                        btn_text,
                                        f"Go to {topic}",
                                        on_click=safe_click(lambda l=link: accessible_notify('Module under development: coming soon', type_='warning') if l in ['/module3', '/module4'] else ui.navigate.to(l))
                                    ).classes(
                                        'w-full text-left text-xl font-semibold !text-slate-800 '
                                        'hover:!text-blue-600 hover:bg-blue-50 py-4 px-6 rounded-xl '
                                        'transition-colors border-b border-gray-100 last:border-0 '
                                        '!shadow-none !bg-transparent' 
                                    )
                                    
                                   
                                    if is_locked:
                                        btn.classes('opacity-60 cursor-not-allowed hover:!text-slate-800 hover:bg-transparent')
                                else:
                                    ui.label(topic).classes('text-gray-500 ml-6 py-3 italic text-xl')



    @ui.page('/reflections')
    def all_reflections():
     
        main_layout("Student Reflections")

        with ui.column().classes('w-full items-center p-6'):
            
          
            with ui.column().classes('w-full max-w-4xl !bg-white/95 !text-black p-8 rounded-2xl shadow-2xl border border-gray-200'):
                
                user = app.storage.user.get('name')
                
                if user == 'admin':
                    ui.label('📒 All Reflections (Admin View)').classes('text-3xl font-extrabold text-indigo-900 mb-6 border-b-2 border-indigo-200 pb-2 w-full').props('role=heading aria-level=2 tabindex=0')
                    
                    entries = app_data.get('reflection_log', [])
                    if not entries:
                        ui.label("No reflections recorded yet.").classes('text-lg text-gray-500 italic')
                    else:
                        for entry in entries:
                            ui.markdown(entry).classes('p-4 mb-4 bg-gray-50 rounded-lg border-l-4 border-indigo-500 text-lg text-slate-800 shadow-sm')
                else:
                    ui.label('🗒️ Your Reflections').classes('text-3xl font-extrabold text-indigo-900 mb-6 border-b-2 border-indigo-200 pb-2 w-full').props('role=heading aria-level=2 tabindex=0')
                    
                    found = False
                    entries = app_data.get('reflection_log', [])
                    for entry in entries:
                        if f"| {user} |" in entry:
                         
                            ui.markdown(entry).classes('p-4 mb-4 bg-indigo-50 rounded-lg border-l-4 border-indigo-400 text-lg text-slate-800 shadow-sm')
                            found = True
                    
                    if not found:
                        ui.label("You haven't written any reflections yet.").classes('text-xl text-gray-500 italic mt-4')
                        ui.label("Go to a module and use the 'Write Reflection' tool in the menu.").classes('text-lg text-gray-400')

'''
for i, title in enumerate(module_titles, 1):
                    with ui.card().classes(star_style).props(
                        f'role=button tabindex=0 aria-label="Go to module {i}: {title}"'
                    ).on('click', 
                        lambda i=i: (
                            ui.navigate.to(f'/module{i}'), 
                            ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")
                        )
                    ).on('keydown.enter', lambda i=i: (
                            ui.navigate.to(f'/module{i}'), 
                            ui.run_javascript("setTimeout(() => document.querySelector('h1, .title')?.focus(), 3600)")
                        )): 
                        ui.label(f'{i}').classes('text-6xl font-black text-yellow-600/30 absolute top-4')
                        ui.label(title).classes('text-2xl font-semibold text-slate-800 leading-tight')
                        '''