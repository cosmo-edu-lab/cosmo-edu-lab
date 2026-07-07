# Copyright (c) [2026] [Eleonora Panini - UniMoRe]. All rights reserved.
# This code is for portfolio purposes only and may not be copied, 
# distributed, or reused without written permission.
import os
import io
import json
import nicegui
from nicegui import ui, app, run , client

import datetime


#from nicegui_toolkit import inject_layout_tool


from core import app_data, save_data, BASE_DIR, DATA_DIR
from layout import aria_input, aria_button, accessible_notify


#ACCESS_CODE = os.getenv("APP_ACCESS_CODE", "Cosmology2026")

def create_auth_routes():
 
    

    @ui.page('/login')
    def login_page():
     
        #if not app.storage.user.get('gate_unlocked', False):
        #    ui.navigate.to('/access')
        #    return

        ui.run_javascript("setTimeout(() => document.querySelector('h1, .title, .text-2xl')?.focus(), 3600)")
        
        def try_login():
            if app_data['users'].get(username.value) == password.value:
                app.storage.user['name'] = username.value
                ui.navigate.to('/main')
            else:
                accessible_notify('Invalid username or password', type_='warning')
                password.value = ""
                password.run_method('focus')

    
        with ui.card().classes('absolute-center').props('role=region aria-label="Login Form"'):
            ui.label(' Login').classes('text-2xl font-bold').props('role=heading aria-level=2 tabindex=0')
            username = aria_input('Username',"Insert the username").on('keydown.enter', lambda: password.run_method('focus'))
            password = aria_input('Password',"Insert the password", password=True).on('keydown.enter', try_login)
            aria_button('Login', "Log in", on_click=try_login)
            
           
            with ui.row().classes('items-center gap-2 mt-2'):
                ui.label("Don't have an account?").classes('text-sm text-slate-300').props('tabindex=0')
                aria_button('Register', "Register", on_click=lambda: ui.navigate.to('/register')).classes('text-xs')


    @ui.page('/register')
    def register_page():
    
        #if not app.storage.user.get('gate_unlocked', False):
        #    ui.navigate.to('/access')
        #    return

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

        with ui.card().classes('absolute-center').props('role=region aria-label="Registration Form"'):
            ui.label('Register New Account').classes('text-2xl font-bold').props('role=heading aria-level=2 tabindex=0')
            new_user = aria_input('New Username', "Insert the new username")
            new_pass = aria_input('New Password',"Insert the new password", password=True)
            aria_button('Register', "Create a new account", on_click=do_register)
            ui.link("Back to Login", '/login').classes('mt-2 text-sm text-gray-500')

  
    @ui.page('/')
    def index_page():
     
        #if not app.storage.user.get('gate_unlocked', False):
        #    ui.navigate.to('/access')
        #    return
            
     
        if not app.storage.user.get('name'):
            ui.navigate.to('/login')
        

        else:
            ui.navigate.to('/main')
            
    '''
    @ui.page('/access')
    def access_page():
       
        if app.storage.user.get('gate_unlocked', False):
            ui.navigate.to('/login')
            return

        def check_code():
           
            if code_input.value == ACCESS_CODE:
               
                app.storage.user['gate_unlocked'] = True
                accessible_notify('Access Granted! Welcome.', type_='success')
                ui.navigate.to('/login')
            else:
              
                accessible_notify('Wrong Access Code. Ask your teacher.', type_='error')
                code_input.value = "" 
                code_input.run_method('focus')

       
        with ui.column().classes('w-full h-screen items-center justify-center bg-slate-900'):
          
            with ui.card().classes('p-8 items-center shadow-2xl !bg-slate-800 border border-slate-700').props('role=region aria-label="Classroom Access Form"'):
                ui.icon('lock', size='4em').classes('text-indigo-400 mb-4').props('aria-hidden=true')
                ui.label('Classroom Access').classes('text-2xl font-bold text-white mb-2').props('role=heading aria-level=1 tabindex=0') 
                ui.label('Enter the class code to proceed.').classes('text-sm text-slate-300 mb-6').props('tabindex=0')
                
             
                code_input = aria_input('Access Code', "Enter class code", password=True).classes('w-full text-lg')
                code_input.on('keydown.enter', check_code)
                
                aria_button('Unlock App', "Unlock", on_click=check_code).classes('w-full mt-4 !bg-indigo-600 text-white hover:!bg-indigo-500')

    '''
    