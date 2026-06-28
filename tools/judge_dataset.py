"""
Waku Waku Code — LLM as Judge
Evalúa cada ejemplo de data/dataset_sxf.jsonl con Claude Haiku 4.5.

Uso:
    .venv/bin/python tools/judge_dataset.py

Resultados en data/judge/:
    revision_juez.json           — reporte completo
    rechazados.json              — ejemplos a corregir
    dataset_sxf_aprobado.jsonl   — dataset limpio para fine-tuning
    progress.json                — progreso (permite retomar si se interrumpe)
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
INPUT_JSONL       = "data/dataset_sxf.jsonl"
OUTPUT_DIR        = "data/judge"
SLEEP_ENTRE_CALLS = 0.5   # segundos entre requests

REGLAS_JUEZ = """
Eres un evaluador de calidad de datasets para fine-tuning de un modelo de IA.
El proyecto se llama "Waku Waku Code": un tutor de Python con personalidades de personajes de Spy x Family.

Tu trabajo es revisar cada ejemplo del dataset y verificar que cumpla TODAS las reglas.

### REGLAS DE CONTENIDO

1. PERSONALIDAD DEL PERSONAJE (OBLIGATORIO):
   - El assistant debe hablar con la personalidad del personaje indicado en el system prompt
   - Debe usar sus muletillas y forma de hablar características
   - Anya: tercera persona, "waku waku", infantil
   - Loid: metódico, preciso, referencias a misiones
   - Yor: amable, cálida, alentadora
   - Franky: informal, quejumbroso, humor
   - Damian: orgulloso, competitivo, retador

2. PROHIBIDO — ANALOGÍAS DEL ANIME PARA EXPLICAR CONCEPTOS:
   - NO debe usar elementos del universo de Spy x Family como analogía para explicar conceptos de Python
   - Nombres prohibidos en explicaciones de conceptos: Bond, Eden, WISE, Ostania, Berlint, Westalis,
     Operation Strix, Desmond, Donovan, Handler, Yuri, Nightfall, SSS
   - Ejemplo INCORRECTO: "Una variable es como las identidades encubiertas de papá"
   - Ejemplo INCORRECTO: "Un if es como cuando Bond presiente el peligro"
   - Las explicaciones deben usar ejemplos COTIDIANOS y NEUTROS (cajas, etiquetas, recetas, objetos del día a día)

3. PERMITIDO — REFERENCIAS AMBIENTALES:
   - SÍ puede hacer comentarios casuales sobre su vida que NO expliquen un concepto de Python
   - Ejemplo CORRECTO: "Anya tuvo un día largo en el colegio, pero esto es más divertido"
   - Ejemplo CORRECTO: "Esto requiere la concentración de una misión de alto nivel" (Loid)
   - La referencia ambiental debe estar SEPARADA de la explicación del concepto

4. ENFOQUE SOCRÁTICO:
   - NO debe dar la respuesta directa a un problema
   - Debe guiar al estudiante a descubrir la respuesta
   - Debe plantear un ejercicio o pregunta de seguimiento al final

5. PYTHON CORRECTO:
   - El código Python mostrado debe ser sintácticamente correcto
   - Las explicaciones técnicas deben ser precisas

### FORMATO DE RESPUESTA

Responde SOLO con un JSON válido con esta estructura exacta, sin backticks ni texto adicional:

{
  "aprobado": true/false,
  "puntuacion": 1-10,
  "problemas": ["problema 1", "problema 2"],
  "detalle": {
    "personalidad_correcta": true/false,
    "analogia_anime_en_explicacion": true/false,
    "referencia_ambiental_correcta": true/false,
    "enfoque_socratico": true/false,
    "python_correcto": true/false,
    "tiene_ejercicio_final": true/false
  },
  "sugerencia_correccion": "texto con sugerencia si hay problemas, o N/A si está aprobado"
}

Un ejemplo se considera APROBADO solo si:
- personalidad_correcta = true
- analogia_anime_en_explicacion = false (false = NO usó analogías, que es lo correcto)
- enfoque_socratico = true
- python_correcto = true
- tiene_ejercicio_final = true
"""

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_anthropic():
    try:
        import anthropic
    except ImportError:
        raise ImportError("Ejecuta: .venv/bin/pip install anthropic")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def cargar_dataset(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}\nEjecuta primero: .venv/bin/python tools/generate_dataset.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cargar_progreso(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def guardar_progreso(path: str, completados: set):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(completados), f)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_ejemplo(client, ejemplo: dict, indice: int) -> dict:
    import anthropic

    msgs = ejemplo["messages"]
    system_msg    = msgs[0]["content"]
    user_msg      = msgs[1]["content"]
    assistant_msg = msgs[2]["content"]

    prompt = f"""Evalúa el siguiente ejemplo de dataset:

SYSTEM PROMPT DEL EJEMPLO:
{system_msg}

PREGUNTA DEL ESTUDIANTE:
{user_msg}

RESPUESTA DEL TUTOR:
{assistant_msg}

Aplica las reglas y devuelve tu veredicto en el formato JSON especificado."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=REGLAS_JUEZ,
            messages=[{"role": "user", "content": prompt}],
        )

        texto = response.content[0].text.strip()

        # Limpiar backticks si el modelo los incluye
        if texto.startswith("```json"):
            texto = texto[7:]
        if texto.startswith("```"):
            texto = texto[3:]
        if texto.endswith("```"):
            texto = texto[:-3]
        texto = texto.strip()

        veredicto = json.loads(texto)
        veredicto["indice"]         = indice
        veredicto["tokens_input"]   = response.usage.input_tokens
        veredicto["tokens_output"]  = response.usage.output_tokens

        return veredicto

    except json.JSONDecodeError as e:
        return {"indice": indice, "aprobado": None,
                "error": f"JSON inválido: {e}", "respuesta_cruda": texto[:300]}
    except Exception as e:
        return {"indice": indice, "aprobado": None, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_revision(client, dataset: list) -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress_path  = os.path.join(OUTPUT_DIR, "progress.json")
    veredictos_path = os.path.join(OUTPUT_DIR, "revision_juez.json")

    completados = cargar_progreso(progress_path)

    # Cargar veredictos previos si existen
    veredictos = []
    if os.path.exists(veredictos_path):
        with open(veredictos_path, encoding="utf-8") as f:
            veredictos = json.load(f)

    total            = len(dataset)
    total_input_tok  = sum(v.get("tokens_input", 0) for v in veredictos)
    total_output_tok = sum(v.get("tokens_output", 0) for v in veredictos)

    print("=" * 60)
    print("WAKU WAKU CODE — REVISIÓN CON CLAUDE HAIKU 4.5")
    print(f"  Dataset     : {INPUT_JSONL}  ({total} ejemplos)")
    print(f"  Resultados  : {OUTPUT_DIR}/")
    if completados:
        print(f"  ↩️  Retomando: {len(completados)}/{total} ejemplos ya evaluados")
    print("=" * 60)

    for i, ejemplo in enumerate(dataset):
        clave = str(i)

        if clave in completados:
            print(f"  [{i+1:>3}/{total}] ⏭️  ya evaluado")
            continue

        print(f"  [{i+1:>3}/{total}] evaluando ...", end=" ", flush=True)

        veredicto = evaluar_ejemplo(client, ejemplo, i)
        veredictos.append(veredicto)

        # Actualizar tokens acumulados
        total_input_tok  += veredicto.get("tokens_input", 0)
        total_output_tok += veredicto.get("tokens_output", 0)

        # Mostrar resultado en tiempo real
        if veredicto.get("aprobado") is True:
            print(f"✅  {veredicto.get('puntuacion', '?')}/10")
        elif veredicto.get("aprobado") is False:
            print(f"❌  {veredicto.get('puntuacion', '?')}/10  — {', '.join(veredicto.get('problemas', []))}")
        else:
            print(f"⚠️   error: {veredicto.get('error', '')[:60]}")

        # Guardar progreso y veredictos incrementalmente
        completados.add(clave)
        guardar_progreso(progress_path, completados)
        with open(veredictos_path, "w", encoding="utf-8") as f:
            json.dump(veredictos, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_ENTRE_CALLS)

    print(f"\n  Tokens input : {total_input_tok:,}")
    print(f"  Tokens output: {total_output_tok:,}")
    costo = (total_input_tok / 1_000_000) * 1.00 + (total_output_tok / 1_000_000) * 5.00
    print(f"  Costo aprox  : ${costo:.4f} USD")

    return veredictos

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def exportar_resultados(dataset: list, veredictos: list):
    aprobados  = [v for v in veredictos if v.get("aprobado") is True]
    rechazados = [v for v in veredictos if v.get("aprobado") is False]
    errores    = [v for v in veredictos if v.get("aprobado") is None]

    # dataset_sxf_aprobado.jsonl
    aprobado_path = os.path.join(OUTPUT_DIR, "dataset_sxf_aprobado.jsonl")
    with open(aprobado_path, "w", encoding="utf-8") as f:
        for v in aprobados:
            f.write(json.dumps(dataset[v["indice"]], ensure_ascii=False) + "\n")

    # rechazados.json
    rechazados_path = os.path.join(OUTPUT_DIR, "rechazados.json")
    with open(rechazados_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"veredicto": v, "ejemplo_original": dataset[v["indice"]]} for v in rechazados],
            f, ensure_ascii=False, indent=2
        )

    # Análisis de tipos de fallo
    fallos = {"personalidad_incorrecta": 0, "analogia_anime": 0,
              "no_socratico": 0, "python_incorrecto": 0, "sin_ejercicio_final": 0}
    for v in rechazados:
        d = v.get("detalle", {})
        if not d.get("personalidad_correcta", True):     fallos["personalidad_incorrecta"] += 1
        if d.get("analogia_anime_en_explicacion", False): fallos["analogia_anime"] += 1
        if not d.get("enfoque_socratico", True):          fallos["no_socratico"] += 1
        if not d.get("python_correcto", True):            fallos["python_incorrecto"] += 1
        if not d.get("tiene_ejercicio_final", True):      fallos["sin_ejercicio_final"] += 1

    # Resumen en consola
    total = len(veredictos)
    pct   = len(aprobados) / total * 100 if total else 0
    prom  = sum(v.get("puntuacion", 0) for v in veredictos if "puntuacion" in v)
    n_pun = sum(1 for v in veredictos if "puntuacion" in v)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print(f"  Total evaluados  : {total}")
    print(f"  ✅ Aprobados     : {len(aprobados)}  ({pct:.1f}%)")
    print(f"  ❌ Rechazados    : {len(rechazados)}")
    print(f"  ⚠️  Errores       : {len(errores)}")
    if n_pun:
        print(f"  Puntuación media : {prom/n_pun:.1f}/10")
    print("=" * 60)

    if rechazados:
        print("\nTIPOS DE FALLO:")
        for fallo, cantidad in sorted(fallos.items(), key=lambda x: -x[1]):
            if cantidad:
                print(f"  {fallo}: {cantidad}  {'█' * cantidad}")

        peores = sorted(rechazados, key=lambda x: x.get("puntuacion", 10))[:5]
        print("\nTOP 5 PEORES:")
        for v in peores:
            print(f"  #{v['indice']}  {v.get('puntuacion','?')}/10 — {', '.join(v.get('problemas', []))}")
            print(f"       💡 {v.get('sugerencia_correccion', '')[:100]}")

    print(f"\nARCHIVOS GENERADOS en {OUTPUT_DIR}/:")
    print(f"  revision_juez.json          ({total} veredictos)")
    print(f"  dataset_sxf_aprobado.jsonl  ({len(aprobados)} ejemplos listos)")
    print(f"  rechazados.json             ({len(rechazados)} a corregir)")

    print("\nSIGUIENTE PASO:")
    if pct >= 90:
        print(f"  {pct:.0f}% aprobado ✅ — Dataset listo para fine-tuning.")
        print(f"  Usa: {aprobado_path}")
    elif pct >= 70:
        print(f"  {pct:.0f}% aprobado — Revisa rechazados.json y corrige los {len(rechazados)} ejemplos.")
    else:
        print(f"  {pct:.0f}% aprobado — Revisa el prompt de generación y regenera.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client  = setup_anthropic()
    dataset = cargar_dataset(INPUT_JSONL)

    veredictos = ejecutar_revision(client, dataset)
    exportar_resultados(dataset, veredictos)
