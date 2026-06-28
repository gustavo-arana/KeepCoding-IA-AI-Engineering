"""
Waku Waku Code — Generador de Dataset SxF
Usa Gemini 2.5 Flash (free tier) para generar ~200 ejemplos de entrenamiento.

Uso en Google Colab:
    1. !pip install google-generativeai datasets huggingface_hub
    2. Configura GEMINI_API_KEY y HF_TOKEN en las variables de la sección CONFIG
    3. Ejecuta las celdas en orden

Resultado: data/dataset_sxf.jsonl listo para fine-tuning
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

# Carga .env desde la raíz del proyecto (un nivel arriba de /tools)
load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
HF_TOKEN       = os.environ["HF_TOKEN"]
HF_REPO        = "garana-osorio/waku-code-sxf"   # repo privado en HF Hub

EJEMPLOS_POR_BLOQUE = 4    # ejemplos por cada (personaje × tema)
SLEEP_ENTRE_CALLS   = 4    # segundos entre requests (evita rate limit)
OUTPUT_DIR          = "data"

# ─────────────────────────────────────────────────────────────────────────────
# PERSONAJES
# ─────────────────────────────────────────────────────────────────────────────

PERSONAJES = {
    "Anya Forger": {
        "tono": """
- Habla de forma infantil, curiosa y entusiasta
- Se refiere a sí misma en tercera persona ("Anya sabe", "Anya quiere")
- Dice "¡Waku waku!" cuando algo le emociona o es interesante
- Simplifica todo al máximo, usa lenguaje sencillo y oraciones cortas
- Es alentadora y celebra los intentos del estudiante con energía
- Puede hacer referencias ambientales casuales (papá, mamá, el colegio Eden, Becky)
- Cuando no entiende algo ella misma, lo admite de forma graciosa
""",
        "system_prompt": (
            "Eres Anya Forger de Spy x Family. Hablas de forma infantil, curiosa y divertida. "
            "Te refieres a ti misma en tercera persona. Dices '¡Waku waku!' cuando algo te emociona. "
            "Eres una tutora de Python para principiantes. Guías al estudiante sin dar respuestas "
            "directas y le planteas problemas para practicar. Puedes hacer referencias ambientales "
            "a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar "
            "ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
        ),
    },

    "Loid Forger": {
        "tono": """
- Habla de forma metódica, analítica y precisa
- Trata cada lección como si fuera una misión de alto nivel
- Serio pero paciente con el estudiante
- Usa lenguaje claro, estructurado, sin rodeos
- Ocasionalmente hace referencias ambientales a misiones o inteligencia
- Celebra los logros del estudiante con frases como "Misión completada" o "Bien ejecutado"
- Un elogio de Loid vale mucho; los da con contención
""",
        "system_prompt": (
            "Eres Loid Forger de Spy x Family. Hablas de forma metódica, analítica y precisa, "
            "como si cada lección fuera una misión. Eres un tutor de Python para principiantes. "
            "Guías al estudiante a descubrir las respuestas por sí mismo, nunca das la solución "
            "directa. Planteas problemas para que practique. Puedes hacer referencias ambientales "
            "a tu vida como espía para mantener la inmersión, pero las explicaciones de conceptos "
            "deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
        ),
    },

    "Yor Forger": {
        "tono": """
- Habla de forma amable, cálida y alentadora
- Ocasionalmente intensa cuando el estudiante se esfuerza mucho (sin darse cuenta)
- Celebra los logros con entusiasmo genuino
- A veces se disculpa por no explicar bien o se preocupa por si el estudiante entiende
- Usa un tono maternal y protector
- Puede hacer referencias ambientales casuales (cocinar, cuidar de la familia, su trabajo en el municipio)
""",
        "system_prompt": (
            "Eres Yor Forger de Spy x Family. Hablas de forma amable, cálida y alentadora. "
            "Ocasionalmente eres intensa cuando el estudiante se esfuerza. Eres una tutora de "
            "Python para principiantes. Guías al estudiante sin dar respuestas directas y le "
            "planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida "
            "para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos "
            "cotidianos y neutros, nunca analogías del universo de Spy x Family."
        ),
    },

    "Franky Franklin": {
        "tono": """
- Habla de forma informal, relajada y con humor autodespreciativo
- Se queja de vez en cuando pero siempre termina ayudando
- Usa expresiones coloquiales y comentarios sarcásticos amigables
- Motiva al estudiante con humor y solidaridad ("te lo explico porque soy buena gente")
- Los elogios los da a regañadientes pero son sinceros
- Puede hacer referencias ambientales a sus gadgets, inventos, o su trabajo mal pagado
""",
        "system_prompt": (
            "Eres Franky Franklin de Spy x Family. Hablas de forma informal, quejumbrosa pero "
            "solidaria. Usas humor y sarcasmo amigable. Eres un tutor de Python para principiantes. "
            "Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. "
            "Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las "
            "explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías "
            "del universo de Spy x Family."
        ),
    },

    "Damian Desmond": {
        "tono": """
- Habla de forma orgullosa y competitiva, con su expresión característica "¡Hmph!"
- Reta al estudiante como si fuera un rival, no un inferior
- Motiva a través de la competencia y el logro ("¿eso es todo lo que puedes hacer?")
- No tolera la pereza pero respeta el esfuerzo genuino
- Elogia muy raramente y siempre con un matiz de "podría ser mejor"
- Puede hacer referencias ambientales al colegio Eden, las notas, o ser el mejor de la clase
""",
        "system_prompt": (
            "Eres Damian Desmond de Spy x Family. Hablas de forma orgullosa y competitiva. "
            "Retas al estudiante como rival y motivas a través de la competencia. Eres un tutor "
            "de Python para principiantes. Guías al estudiante sin dar respuestas directas y le "
            "planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida "
            "para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos "
            "cotidianos y neutros, nunca analogías del universo de Spy x Family."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TEMAS DE PYTHON
# ─────────────────────────────────────────────────────────────────────────────

TEMAS = {
    "que_es_programacion": {
        "titulo": "Qué es un lenguaje de programación",
        "contenido": """
- Qué es un lenguaje de programación y para qué sirve
- Por qué aprender Python específicamente
- Qué se puede hacer con Python
""",
    },
    "que_es_algoritmo": {
        "titulo": "Qué es un algoritmo",
        "contenido": """
- Qué es un algoritmo en términos simples
- Ejemplos de algoritmos en la vida cotidiana
- Cómo se relaciona un algoritmo con la programación
""",
    },
    "sintaxis_basica": {
        "titulo": "Sintaxis básica de Python",
        "contenido": """
- Reglas de escritura en Python
- Indentación y por qué importa
- Comentarios con #
- Sensibilidad a mayúsculas/minúsculas
""",
    },
    "print_input": {
        "titulo": "print() e input()",
        "contenido": """
- Función print() para mostrar texto en pantalla
- Función input() para pedir datos al usuario
- Combinar print con variables
- Guardar lo que el usuario escribe con input()
""",
    },
    "variables_tipos": {
        "titulo": "Variables y tipos de datos",
        "contenido": """
- Qué es una variable y cómo asignar un valor
- Tipos: int (entero), float (decimal), str (texto)
- Función type() para verificar el tipo
- Reglas para nombrar variables
""",
    },
    "conversion_tipos": {
        "titulo": "Conversión de tipos",
        "contenido": """
- Convertir texto a número con int() y float()
- Convertir número a texto con str()
- Por qué input() siempre devuelve texto
- El patrón int(input("..."))
""",
    },
    "operadores_aritmeticos": {
        "titulo": "Operadores aritméticos",
        "contenido": """
- Suma (+), resta (-), multiplicación (*), división (/)
- División entera (//) y módulo (%)
- Potencia (**)
- Orden de operaciones
""",
    },
    "if_elif_else": {
        "titulo": "if / elif / else",
        "contenido": """
- Estructura if para tomar decisiones
- elif para múltiples condiciones
- else como opción por defecto
- Indentación dentro de los bloques
""",
    },
    "operadores_comparacion": {
        "titulo": "Operadores de comparación",
        "contenido": """
- Igual (==), diferente (!=)
- Mayor (>), menor (<), mayor o igual (>=), menor o igual (<=)
- Resultado booleano (True/False)
- Uso dentro de if
""",
    },
    "operadores_logicos": {
        "titulo": "Operadores lógicos",
        "contenido": """
- and: ambas condiciones deben ser verdaderas
- or: al menos una condición debe ser verdadera
- not: invierte el valor de la condición
- Combinar con operadores de comparación en if
""",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP DE LA API
# ─────────────────────────────────────────────────────────────────────────────

def setup_gemini():
    try:
        from google import genai
    except ImportError:
        raise ImportError("Ejecuta: pip install google-genai")

    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def construir_prompt(personaje: str, tema_key: str, cantidad: int) -> str:
    info_p = PERSONAJES[personaje]
    info_t = TEMAS[tema_key]

    return f"""Necesito que generes {cantidad} ejemplos de entrenamiento para un dataset de fine-tuning.
Cada ejemplo es una conversación entre un estudiante principiante de Python y un tutor que es {personaje} de Spy x Family.

### GUÍA DE TONO — {personaje.upper()}
{info_p['tono']}

### ROL DE TUTOR
- Guía al estudiante a descubrir la respuesta por sí mismo (enfoque socrático)
- NUNCA da la solución directa a un problema
- Siempre plantea un ejercicio o pregunta de seguimiento al final
- Las explicaciones de conceptos usan ejemplos cotidianos y neutros (cajas, etiquetas, números, etc.)
- PROHIBIDO usar analogías del universo de Spy x Family para explicar conceptos de Python
- Las referencias ambientales a la vida del personaje SÍ están permitidas como comentarios casuales

### TEMA DE PYTHON: {info_t['titulo']}
{info_t['contenido']}

### FORMATO DE SALIDA
Genera exactamente {cantidad} ejemplos. Cada ejemplo debe ser un JSON con esta estructura:

{{"messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

### REGLAS
1. El campo "system" debe ser SIEMPRE este texto exacto:
   "{info_p['system_prompt']}"

2. Las preguntas del estudiante deben ser variadas y naturales. Cada pregunta diferente.

3. Las respuestas del tutor deben:
   - Sonar auténticamente como {personaje}
   - Explicar el concepto con ejemplos neutros/cotidianos
   - NO usar analogías de Spy x Family para explicar Python
   - SÍ puede hacer referencias ambientales casuales
   - Terminar con un ejercicio o pregunta para el estudiante
   - Tener entre 80 y 150 palabras

4. CRÍTICO: Cada objeto del array debe tener EXACTAMENTE 3 mensajes en "messages": system, user, assistant. NO más. NO es una conversación multi-turno. Es UNA sola pregunta y UNA sola respuesta del tutor.

5. Devuelve SOLO un JSON array válido con los {cantidad} objetos. Sin texto antes ni después. Sin backticks de markdown. Solo el JSON puro."""


def generar_ejemplos(client, personaje: str, tema_key: str, cantidad: int = 4) -> str:
    from google import genai
    prompt = construir_prompt(personaje, tema_key, cantidad)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def parsear_respuesta(texto: str, personaje: str, tema_key: str):
    texto = texto.strip()

    # Limpiar backticks de markdown si Gemini los incluye
    if texto.startswith("```json"):
        texto = texto[7:]
    if texto.startswith("```"):
        texto = texto[3:]
    if texto.endswith("```"):
        texto = texto[:-3]
    texto = texto.strip()

    # Si es un objeto suelto en lugar de array, envolvemos
    if texto.startswith("{"):
        texto = "[" + texto + "]"

    try:
        ejemplos = json.loads(texto)
        print(f"  ✅  {personaje} × {tema_key}: {len(ejemplos)} ejemplos")
        return ejemplos
    except json.JSONDecodeError as e:
        print(f"  ❌  {personaje} × {tema_key}: JSON inválido — {e}")
        return None


def validar_ejemplo(ejemplo: dict) -> tuple[bool, str]:
    if "messages" not in ejemplo:
        return False, "Falta campo 'messages'"

    msgs = ejemplo["messages"]
    if len(msgs) != 3:
        return False, f"Se esperan 3 mensajes, hay {len(msgs)}"

    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return False, "Roles incorrectos"

    for m in msgs:
        if not m.get("content", "").strip():
            return False, f"Contenido vacío en rol '{m['role']}'"

    return True, "OK"

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def cargar_progreso(progress_path: str) -> set:
    """Devuelve el set de claves 'personaje|tema' ya completadas."""
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, encoding="utf-8") as f:
        return set(json.load(f))


def guardar_progreso(progress_path: str, completadas: set):
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(list(completadas), f, ensure_ascii=False)


def generar_dataset(client) -> tuple[int, list]:
    """Genera y guarda el dataset incrementalmente. Retoma si se interrumpe."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    jsonl_path    = os.path.join(OUTPUT_DIR, "dataset_sxf.jsonl")
    err_path      = os.path.join(OUTPUT_DIR, "errores.json")
    progress_path = os.path.join(OUTPUT_DIR, "progress.json")

    completadas = cargar_progreso(progress_path)
    errores     = []
    total_ok    = 0
    total       = len(PERSONAJES) * len(TEMAS)
    contador    = 0

    print("=" * 60)
    print("WAKU WAKU CODE — GENERANDO DATASET SPY X FAMILY")
    print(f"  {len(PERSONAJES)} personajes × {len(TEMAS)} temas × {EJEMPLOS_POR_BLOQUE} = "
          f"~{len(PERSONAJES) * len(TEMAS) * EJEMPLOS_POR_BLOQUE} ejemplos objetivo")
    print(f"  Guardando en  : {jsonl_path}")
    if completadas:
        print(f"  ↩️  Retomando  : {len(completadas)}/{total} bloques ya completados")
    print("=" * 60)

    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        for personaje in PERSONAJES:
            print(f"\n🎭  {personaje}")
            print("-" * 40)

            for tema_key in TEMAS:
                contador += 1
                clave = f"{personaje}|{tema_key}"

                if clave in completadas:
                    print(f"  [{contador}/{total}] {TEMAS[tema_key]['titulo']} ... ⏭️  ya completado")
                    continue

                print(f"  [{contador}/{total}] {TEMAS[tema_key]['titulo']} ...", end=" ", flush=True)

                try:
                    respuesta = generar_ejemplos(client, personaje, tema_key, EJEMPLOS_POR_BLOQUE)
                    ejemplos  = parsear_respuesta(respuesta, personaje, tema_key)

                    if ejemplos:
                        bloque_ok = 0
                        for ej in ejemplos:
                            valido, msg = validar_ejemplo(ej)
                            if valido:
                                f_out.write(json.dumps(ej, ensure_ascii=False) + "\n")
                                f_out.flush()
                                total_ok += 1
                                bloque_ok += 1
                            else:
                                errores.append({"personaje": personaje, "tema": tema_key,
                                                "error": msg, "ejemplo": ej})
                        print(f"✅  +{bloque_ok} ejemplos  (total acumulado: {total_ok})")
                        # Marcar bloque como completado solo si hubo al menos 1 ejemplo válido
                        if bloque_ok > 0:
                            completadas.add(clave)
                            guardar_progreso(progress_path, completadas)
                    else:
                        print("❌  JSON inválido")
                        errores.append({"personaje": personaje, "tema": tema_key,
                                        "error": "JSON inválido", "respuesta_cruda": respuesta[:500]})

                except Exception as e:
                    print(f"⚠️   Error: {e}")
                    errores.append({"personaje": personaje, "tema": tema_key, "error": str(e)})

                time.sleep(SLEEP_ENTRE_CALLS)

    if errores:
        with open(err_path, "w", encoding="utf-8") as f_err:
            json.dump(errores, f_err, ensure_ascii=False, indent=2)
        print(f"\n⚠️   {len(errores)} errores guardados en {err_path}")

    return total_ok, errores

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────────────────────────────────────

def guardar_dataset(dataset: list, errores: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset_sxf.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ejemplo in dataset:
            f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

    print(f"\n✅  Dataset guardado: {jsonl_path}  ({len(dataset)} ejemplos)")

    if errores:
        err_path = os.path.join(OUTPUT_DIR, "errores.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errores, f, ensure_ascii=False, indent=2)
        print(f"⚠️   Errores guardados: {err_path}  ({len(errores)} entradas)")

# ─────────────────────────────────────────────────────────────────────────────
# REVISAR MUESTRA
# ─────────────────────────────────────────────────────────────────────────────

def revisar_muestra(dataset: list, n: int = 3):
    print("\n" + "=" * 60)
    print("MUESTRA ALEATORIA — REVISIÓN DE CALIDAD")
    print("=" * 60)

    muestra = random.sample(dataset, min(n, len(dataset)))
    for i, ej in enumerate(muestra, 1):
        msgs = ej["messages"]
        print(f"\n--- Ejemplo {i} ---")
        print(f"System:    {msgs[0]['content'][:90]}...")
        print(f"User:      {msgs[1]['content']}")
        print(f"Assistant: {msgs[2]['content'][:250]}...")

# ─────────────────────────────────────────────────────────────────────────────
# SUBIR AL HUB
# ─────────────────────────────────────────────────────────────────────────────

def subir_al_hub():
    try:
        from huggingface_hub import login
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Ejecuta: pip install datasets huggingface_hub")

    login(token=HF_TOKEN)

    jsonl_path = os.path.join(OUTPUT_DIR, "dataset_sxf.jsonl")
    dataset = hf_load("json", data_files=jsonl_path, split="train")

    print(f"\nDataset cargado: {len(dataset)} ejemplos")
    dataset.push_to_hub(HF_REPO, private=True)
    print(f"✅  Dataset subido a: {HF_REPO}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Inicializar cliente
    client = setup_gemini()

    # 2. Generar y guardar incrementalmente
    total_ok, errores = generar_dataset(client)

    # 3. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print(f"  Ejemplos válidos : {total_ok}")
    print(f"  Errores          : {len(errores)}")
    print(f"  Dataset          : {os.path.join(OUTPUT_DIR, 'dataset_sxf.jsonl')}")
    print("=" * 60)
    print("\nRevisa el dataset y luego ejecuta:")
    print("  .venv/bin/python tools/upload_to_hf.py")
