"""
Sube dataset_sxf.jsonl al Hugging Face Hub como dataset privado.
Ejecutar DESPUÉS de revisar el dataset generado.

Uso:
    .venv/bin/python tools/upload_to_hf.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN  = os.environ["HF_TOKEN"]
HF_REPO   = "garana-osorio/waku-code-sxf"
JSONL_PATH = "data/dataset_sxf.jsonl"

from huggingface_hub import login
from datasets import load_dataset

if not os.path.exists(JSONL_PATH):
    print(f"❌  No se encontró {JSONL_PATH}")
    print("    Ejecuta primero: .venv/bin/python tools/generate_dataset.py")
    exit(1)

print(f"Cargando {JSONL_PATH} ...")
dataset = load_dataset("json", data_files=JSONL_PATH, split="train")
print(f"  {len(dataset)} ejemplos  |  columnas: {dataset.column_names}")

print(f"\nSubiendo a {HF_REPO} (privado) ...")
login(token=HF_TOKEN)
dataset.push_to_hub(HF_REPO, private=True)
print(f"\n✅  Dataset subido: https://huggingface.co/datasets/{HF_REPO}")
