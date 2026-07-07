@echo off
REM ===============================================================
REM 🌌 Cosmo-Edu_Lab — Sincronizzazione completa (GitHub + HuggingFace)
REM Funziona in Windows locale con Git Bash o VS Code terminal
REM ===============================================================

echo 🔍 Verifica connessione repository...
git remote -v

echo ⚙️ Configurazione Git LFS...
git lfs install
git lfs track "*.png" "*.jpg" "*.jpeg" "*.h5" "*.pkl" "*.zip" "*.npz" "*.npy" "*.js" "*.pyd" "*.pyc"
git add .gitattributes

echo 🧹 Aggiornamento .gitignore...
(
echo # File sensibili e locali
echo .env
echo *.log
echo __pycache__/
echo *.pyc
echo *.ipynb_checkpoints/
echo .DS_Store
echo .vscode/
echo .idea/
echo App/venv/
echo node_modules/
) > .gitignore

git add .gitignore

REM --- Rimozione forzata di file locali indesiderati dal tracking ---
echo 🧹 Rimozione di venv, node_modules e __pycache__ dal tracking...
git rm -r --cached App/venv >nul 2>&1
git rm -r --cached node_modules >nul 2>&1
git rm -r --cached __pycache__ >nul 2>&1

REM --- Pulizia definitiva di eventuali file residui prima del commit ---
for /f "delims=" %%F in ('git ls-files -o --exclude-standard') do (
    if "%%F"=="App/venv" (
        echo Ignoro %%F
    )
)

git add --all -- ':!App/venv' ':!Cosmo-Edu_Lab_test' ':!Cosmo-Edu_Lab_clean' ':!Cosmo-Edu-Lab'

echo ⬇️ Pull da GitHub e Hugging Face...
git fetch origin main
git merge origin/main --no-edit
git fetch huggingface main
git merge huggingface/main --no-edit

echo 💾 Commit modifiche locali...
git add -A
git commit -m "Sync automatico locale/remote %date% %time%" || echo Nessuna modifica da committare.

echo 🚀 Push su GitHub...
git lfs push origin main
git push origin main || git push origin main --force-with-lease

echo 🚀 Push su Hugging Face...
git lfs push huggingface main
git push huggingface main || git push huggingface main --force-with-lease

echo ✅ Sincronizzazione completata!
pause
