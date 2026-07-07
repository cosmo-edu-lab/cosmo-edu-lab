#!/bin/bash

# Termina immediatamente lo script se un comando fallisce
set -e

# --- 1. Aggiorna il repository 'Cosmo-Edu_Lab' e fai il push su 'origin' ---

echo "Inizio processo per /workspaces/Cosmo-Edu_Lab (push su origin)..."

# Assumendo che lo script venga eseguito da /workspaces/Cosmo-Edu_Lab
# Se così non fosse, decommenta la riga seguente:
# cd /workspaces/Cosmo-Edu_Lab

echo "Aggiungo App/app.py..."
git add App/app.py

echo "Eseguo il commit..."
# Usa il messaggio di commit "app" come da cronologia
git commit -m "app"

echo "Eseguo il push su origin main..."
git push origin main

echo "Push su 'origin' completato."
echo "---------------------------------"


# --- 2. Copia il file nel secondo repository ---

echo "Copio App/app.py in /workspaces/Cosmo-Edu_Lab_new/App/..."
cp /workspaces/Cosmo-Edu_Lab/App/app.py /workspaces/Cosmo-Edu_Lab_new/App/app.py
echo "Copia completata."
echo "---------------------------------"


# --- 3. Aggiorna il repository 'Cosmo-Edu_Lab_new' e fai il push su 'huggingface' ---

echo "Inizio processo per /workspaces/Cosmo-Edu_Lab_new (push su huggingface)..."

# Spostati nella directory del secondo repository
cd /workspaces/Cosmo-Edu_Lab_new

echo "Aggiungo App/app.py..."
git add App/app.py

echo "Eseguo il commit..."
# Usa lo stesso messaggio di commit "app"
git commit -m "app"

echo "Eseguo il push su huggingface main..."
git push huggingface main

echo "Push su 'huggingface' completato."
echo "---------------------------------"

echo "Tutte le operazioni sono state completate con successo."