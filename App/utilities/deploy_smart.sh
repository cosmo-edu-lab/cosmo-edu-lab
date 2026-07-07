#!/bin/bash

# --- CONFIGURAZIONE PERCORSI ---
SOURCE_DIR="/workspaces/Cosmo-Edu_Lab"
TARGET_DIR="/workspaces/Cosmo-Edu_Lab_new"
COMMIT_MSG="$1"

# Verifica messaggio commit
if [ -z "$COMMIT_MSG" ]; then
  echo "❌ Errore: Inserisci un messaggio per il commit!"
  echo "Uso: ./deploy_smart.sh \"Messaggio del commit\""
  exit 1
fi

echo "=========================================="
echo "🕵️  RILEVAMENTO MODIFICHE IN CORSO..."
echo "=========================================="

# 1. Vai nella cartella Sorgente e prepara i file
cd "$SOURCE_DIR" || exit
git add .

# --- IL CUORE DELLO SCRIPT: TROVA SOLO I FILE MODIFICATI ---
# Questo comando lista i file che stanno per essere committati (staged)
# Esclude i file cancellati per evitare errori di copia
FILES_TO_COPY=$(git diff --name-only --cached --diff-filter=ACMR)

if [ -z "$FILES_TO_COPY" ]; then
    echo "⚠️ Nessun file modificato trovato. Nulla da copiare."
    exit 0
fi

echo "📄 File rilevati come modificati:"
echo "$FILES_TO_COPY"
echo "------------------------------------------"

# 2. Esegui il Commit su GitHub (Source)
echo "💾 1. Commit e Push su GitHub (Source)..."
git commit -m "$COMMIT_MSG"
git push origin main

# 3. Copia SOLO i file rilevati
echo "🔄 2. Copia selettiva su Hugging Face..."

# Legge la lista dei file riga per riga
while IFS= read -r file; do
    # cp --parents mantiene la struttura delle cartelle (es. crea App/images se serve)
    # Copia da SOURCE a TARGET
    cp --parents "$file" "$TARGET_DIR"
    echo "   ✅ Copiato: $file"
done <<< "$FILES_TO_COPY"

# 4. Deploy su Hugging Face (Target)
echo "🤗 3. Deploy su Hugging Face (Target)..."
cd "$TARGET_DIR" || exit
git add .
git commit -m "$COMMIT_MSG"
git push huggingface main

echo "=========================================="
echo "🚀 DEPLOY CHIRURGICO COMPLETATO!"
echo "=========================================="