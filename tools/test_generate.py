"""
Test rápido: 1 personaje × 1 tema × 2 ejemplos
Valida que la API, el parseo y el guardado funcionan antes del ciclo completo.
"""

import sys
from pathlib import Path

# Importar todo desde el script principal
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import (
    setup_gemini,
    generar_ejemplos,
    parsear_respuesta,
    validar_ejemplo,
    guardar_dataset,
    revisar_muestra,
    PERSONAJES,
    TEMAS,
)

PERSONAJE_TEST = "Anya Forger"
TEMA_TEST      = "variables_tipos"
CANTIDAD_TEST  = 2

print("=" * 60)
print("TEST — 1 personaje × 1 tema × 2 ejemplos")
print(f"  Personaje : {PERSONAJE_TEST}")
print(f"  Tema      : {TEMAS[TEMA_TEST]['titulo']}")
print("=" * 60)

# 1. Inicializar modelo
print("\n[1/4] Conectando con Gemini...")
model = setup_gemini()
print("      OK")

# 2. Generar
print("\n[2/4] Generando ejemplos...")
respuesta_cruda = generar_ejemplos(model, PERSONAJE_TEST, TEMA_TEST, CANTIDAD_TEST)
print(f"      Respuesta recibida ({len(respuesta_cruda)} chars)")

# 3. Parsear y validar
print("\n[3/4] Parseando y validando...")
ejemplos = parsear_respuesta(respuesta_cruda, PERSONAJE_TEST, TEMA_TEST)

dataset  = []
errores  = []

if ejemplos:
    for ej in ejemplos:
        valido, msg = validar_ejemplo(ej)
        if valido:
            dataset.append(ej)
        else:
            errores.append({"error": msg, "ejemplo": ej})
            print(f"      ⚠️  Ejemplo inválido: {msg}")
else:
    print("      ❌ No se pudo parsear la respuesta")
    print("\nRespuesta cruda:")
    print(respuesta_cruda)
    sys.exit(1)

# 4. Guardar en data/test_output.jsonl
print("\n[4/4] Guardando resultado...")
import json, os
os.makedirs("data", exist_ok=True)
with open("data/test_output.jsonl", "w", encoding="utf-8") as f:
    for ej in dataset:
        f.write(json.dumps(ej, ensure_ascii=False) + "\n")
print(f"      Guardado en data/test_output.jsonl ({len(dataset)} ejemplos)")

# Mostrar resultado
print("\n" + "=" * 60)
print("RESULTADO")
print(f"  Ejemplos válidos : {len(dataset)}")
print(f"  Errores          : {len(errores)}")
print("=" * 60)

revisar_muestra(dataset, n=len(dataset))
