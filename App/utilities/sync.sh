#!/bin/bash
set -e

echo "🔍 Verifica connessione repository..."
git remote -v

echo "⚙️ Configurazione Git LFS..."
git lfs install
git lfs track "*.png" "*.jpg" "*.jpeg" "*.h5" "*.pkl" "*.zip" "*.npz" "*.npy" "*.js" "*.pyd" "*.pyc"
git add .gitattributes

echo "🧹 Aggiornamento .gitignore..."
cat > .gitignore <<EOF
# File sensibili e locali
.env
*.log
__pycache__/
*.pyc
*.ipynb_checkpoints/
.DS_Store
.vscode/
.idea/
EOF

git add .gitignore

echo "🧹 Rimozione di venv, node_modules e altri file locali dal tracking..."
git rm -r --cached App/venv 2>/dev/null || true
git rm -r --cached node_modules 2>/dev/null || true
git rm -r --cached __pycache__ 2>/dev/null || true

echo "🧹 Aggiunta degli altri file (escludendo venv e cartelle locali pesanti)..."
git add --all -- ':!App/venv' ':!Cosmo-Edu_Lab_test' ':!Cosmo-Edu_Lab_clean' ':!Cosmo-Edu-Lab'

echo "⬇️ Pull da GitHub e Hugging Face..."
git fetch origin main || true
git merge origin/main --no-edit || true
git fetch huggingface main || true
git merge huggingface/main --no-edit || true

echo "💾 Commit modifiche locali..."
git add -A
git commit -m "Sync automatico locale/remote $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nessuna modifica da committare."

echo "🚀 Push su GitHub..."
git lfs push origin main
git push origin main || { echo "⚠️ Push GitHub fallito, forzo..."; git push origin main --force-with-lease; }

echo "🚀 Push su Hugging Face..."
git lfs push huggingface main
git push huggingface main || { echo "⚠️ Push Hugging Face fallito, forzo..."; git push huggingface main --force-with-lease; }

echo "✅ Sincronizzazione completata con successo!"
