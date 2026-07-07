#!/bin/bash
set -e

FILE="App/app.py"

echo "💾 Commit di $FILE..."
git add "$FILE"
git commit -m "update app $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nessuna modifica da committare."

echo "🚀 Push su GitHub..."
git push origin main || { echo "⚠️ Forzo push GitHub..."; git push origin main --force-with-lease; }

echo "🚀 Push su Hugging Face (solo file aggiornato)..."
git push huggingface main --force-with-lease

echo "✅ Push completato solo per $FILE"
