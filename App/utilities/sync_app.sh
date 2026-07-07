#!/bin/bash
set -e

# === CONFIGURAZIONE ===
BACKUP_REPO="../Cosmo-Edu_Lab_backup"   # percorso della tua copia di backup
FILE_PATH="App/app.py"                  # file da sincronizzare

echo "🚀 Sincronizzazione $FILE_PATH verso repository di backup..."
echo "Backup repo: $BACKUP_REPO"
sleep 1

# === CONTROLLO ESISTENZA BACKUP ===
if [ ! -d "$BACKUP_REPO/.git" ]; then
    echo "❌ ERRORE: la cartella di backup '$BACKUP_REPO' non è un repository Git!"
    exit 1
fi

# === COPIA FILE ===
if [ -f "$FILE_PATH" ]; then
    mkdir -p "$(dirname "$BACKUP_REPO/$FILE_PATH")"
    cp "$FILE_PATH" "$BACKUP_REPO/$FILE_PATH"
    echo "✅ Copiato $FILE_PATH → $BACKUP_REPO/$FILE_PATH"
else
    echo "❌ ERRORE: il file $FILE_PATH non esiste!"
    exit 1
fi

# === GIT OPERATIONS NEL BACKUP ===
cd "$BACKUP_REPO"

echo "⚙️ Aggiunta e commit nel repository di backup..."
git add "$FILE_PATH"
git commit -m "🔄 Sync $FILE_PATH da repository originale ($(date '+%Y-%m-%d %H:%M:%S'))" || echo "Nessuna modifica da committare."

echo "🚀 Push su GitHub..."
git push origin main || { echo "⚠️ Push GitHub fallito, forzo..."; git push origin main --force-with-lease; }

echo "🚀 Push su Hugging Face..."
git push huggingface main || { echo "⚠️ Push Hugging Face fallito, forzo..."; git push huggingface main --force-with-lease; }

echo "✅ File $FILE_PATH sincronizzato e pubblicato con successo!"
