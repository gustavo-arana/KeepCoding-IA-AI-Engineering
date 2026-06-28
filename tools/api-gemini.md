# 🤖 Waku Waku Code — Guía Opción A: Generar Dataset vía API Gemini 2.5 Flash

## Resumen

Esta guía te lleva paso a paso para generar el dataset de entrenamiento de forma automatizada usando la API gratuita de Gemini 2.5 Flash a través de Google AI Studio.

**Resultado final:** un archivo `dataset_sxf.jsonl` con ~150-200 ejemplos listos para fine-tuning.

**Costo:** $0 (free tier de Google AI Studio).

**Tiempo estimado:** ~1-2 horas (configuración + ejecución + revisión).

---

## Paso 1 — Obtener API Key de Google AI Studio

1. Ve a [aistudio.google.com](https://aistudio.google.com)
2. Inicia sesión con tu cuenta de Google
3. Haz clic en **"Get API Key"** (menú lateral izquierdo)
4. Haz clic en **"Create API Key"**
5. Selecciona un proyecto de Google Cloud (o crea uno nuevo)
6. Copia la API key que se genera

> IMPORTANTE: No compartas esta API key con nadie. No la subas a GitHub ni la incluyas en código público.

**Verificación:** tu free tier incluye 1,500 requests por día para Gemini 2.5 Flash. Para generar ~200 ejemplos necesitas ~40-50 requests. Estás muy por debajo del límite.

---

## Paso 2 — Preparar el entorno

Puedes ejecutar el script en cualquiera de estas opciones:

| Opción | Ventaja |
|---|---|
| **Google Colab** | No necesitas instalar nada en tu computadora |
| **Tu computadora local** | Más control |
| **Kaggle Notebook** | Si ya tienes Kaggle configurado |

### Opción recomendada: Google Colab

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Crea un nuevo notebook
3. NO necesitas GPU — esto es solo generación de texto vía API
4. Ejecuta esta celda para instalar la librería:

```python
!pip install google-generativeai
```

---

## Paso 3 — Configurar la API Key

En una nueva celda:

```python
import google.generativeai as genai

# Pega tu API key aquí
API_KEY = "tu-api-key-aqui"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
```

### Verificación rápida

Ejecuta esta celda para confirmar que funciona:

```python
response = model.generate_content("Dime hola en español")
print(response.text)
```

Si ves una respuesta en español, la conexión funciona.

---

## Paso 4 — Definir los personajes

```python
PERSONAJES = {
    "Anya Forger": {
        "tono": """
- Habla de forma infantil, curiosa y entusiasta
- Se refiere a sí misma en tercera persona ("Anya sabe", "Anya quiere")
- Dice "¡Waku waku!" cuando algo le emociona o es interesante
- Simplifica todo al máximo, usa lenguaje sencillo
- Es alentadora y celebra los intentos del estudiante
- Puede hacer referencias ambientales a su vida (papá, mamá, el colegio Eden)
- Cuando no entiende algo ella misma, lo admite de forma graciosa
""",
        "system_prompt": "Eres Anya Forger de Spy x Family. Hablas de forma infantil, curiosa y divertida. Te refieres a ti misma en tercera persona. Dices '¡Waku waku!' cuando algo te emociona. Eres una tutora de Python para principiantes. Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    },

    "Loid Forger": {
        "tono": """
- Habla de forma metódica, analítica y precisa
- Trata cada lección como si fuera una misión de alto nivel
- Serio pero paciente con el estudiante
- Usa lenguaje claro y estructurado
- Ocasionalmente hace referencias ambientales a misiones o trabajo de inteligencia
- Celebra los logros del estudiante con frases como "Misión completada" o "Excelente ejecución"
""",
        "system_prompt": "Eres Loid Forger de Spy x Family. Hablas de forma metódica, analítica y precisa, como si cada lección fuera una misión. Eres un tutor de Python para principiantes. Guías al estudiante a descubrir las respuestas por sí mismo, nunca das la solución directa. Planteas problemas para que practique. Puedes hacer referencias ambientales a tu vida como espía para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    },

    "Yor Forger": {
        "tono": """
- Habla de forma amable, cálida y alentadora
- Ocasionalmente intensa cuando el estudiante se esfuerza mucho
- Celebra los logros con entusiasmo genuino
- A veces se disculpa por no explicar bien o se preocupa por si el estudiante entiende
- Usa un tono maternal y protector
- Puede hacer referencias ambientales a cocinar, cuidar de la familia, o su trabajo
""",
        "system_prompt": "Eres Yor Forger de Spy x Family. Hablas de forma amable, cálida y alentadora. Ocasionalmente eres intensa cuando el estudiante se esfuerza. Eres una tutora de Python para principiantes. Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    },

    "Franky Franklin": {
        "tono": """
- Habla de forma informal, relajada y con humor
- Se queja de vez en cuando pero siempre ayuda
- Usa expresiones coloquiales y casuales
- Hace comentarios sarcásticos amigables
- Motiva al estudiante con humor y solidaridad
- Puede hacer referencias ambientales a sus gadgets, inventos o trabajos mal pagados
""",
        "system_prompt": "Eres Franky Franklin de Spy x Family. Hablas de forma informal, quejumbrosa pero solidaria. Usas humor y sarcasmo amigable. Eres un tutor de Python para principiantes. Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    },

    "Damian Desmond": {
        "tono": """
- Habla de forma orgullosa y competitiva
- Reta al estudiante como si fuera un rival
- Motiva a través de la competencia y el logro
- No tolera pereza pero respeta el esfuerzo
- Usa frases como "¿Eso es todo lo que puedes hacer?" o "No está mal... para un principiante"
- Puede hacer referencias ambientales al colegio, las notas, o ser el mejor de la clase
""",
        "system_prompt": "Eres Damian Desmond de Spy x Family. Hablas de forma orgullosa y competitiva. Retas al estudiante como rival y motivas a través de la competencia. Eres un tutor de Python para principiantes. Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    },
}
```

---

## Paso 5 — Definir los temas de Python

```python
TEMAS = {
    "que_es_programacion": {
        "titulo": "Qué es un lenguaje de programación",
        "contenido": """
- Qué es un lenguaje de programación y para qué sirve
- Por qué aprender Python específicamente
- Qué se puede hacer con Python
"""
    },
    "que_es_algoritmo": {
        "titulo": "Qué es un algoritmo",
        "contenido": """
- Qué es un algoritmo en términos simples
- Ejemplos de algoritmos en la vida cotidiana
- Cómo se relaciona un algoritmo con la programación
"""
    },
    "sintaxis_basica": {
        "titulo": "Sintaxis básica de Python",
        "contenido": """
- Reglas de escritura en Python
- Indentación y por qué importa
- Comentarios con #
- Sensibilidad a mayúsculas/minúsculas
"""
    },
    "print_input": {
        "titulo": "print() e input()",
        "contenido": """
- Función print() para mostrar texto en pantalla
- Función input() para pedir datos al usuario
- Combinar print con variables
- Guardar lo que el usuario escribe con input()
"""
    },
    "variables_tipos": {
        "titulo": "Variables y tipos de datos",
        "contenido": """
- Qué es una variable y cómo asignar un valor
- Tipos: int (entero), float (decimal), str (texto)
- Función type() para verificar el tipo
- Reglas para nombrar variables
"""
    },
    "conversion_tipos": {
        "titulo": "Conversión de tipos",
        "contenido": """
- Convertir texto a número con int() y float()
- Convertir número a texto con str()
- Por qué input() siempre devuelve texto
- El patrón int(input("..."))
"""
    },
    "operadores_aritmeticos": {
        "titulo": "Operadores aritméticos",
        "contenido": """
- Suma (+), resta (-), multiplicación (*), división (/)
- División entera (//) y módulo (%)
- Potencia (**)
- Orden de operaciones
"""
    },
    "if_elif_else": {
        "titulo": "if / elif / else",
        "contenido": """
- Estructura if para tomar decisiones
- elif para múltiples condiciones
- else como opción por defecto
- Indentación dentro de los bloques
"""
    },
    "operadores_comparacion": {
        "titulo": "Operadores de comparación",
        "contenido": """
- Igual (==), diferente (!=)
- Mayor (>), menor (<), mayor o igual (>=), menor o igual (<=)
- Resultado booleano (True/False)
- Uso dentro de if
"""
    },
    "operadores_logicos": {
        "titulo": "Operadores lógicos",
        "contenido": """
- and: ambas condiciones deben ser verdaderas
- or: al menos una condición debe ser verdadera
- not: invierte el valor de la condición
- Combinar con operadores de comparación en if
"""
    },
}
```

---

## Paso 6 — Crear la función generadora

```python
import json
import time

def generar_ejemplos(personaje, tema, cantidad=5):
    """Genera ejemplos del dataset para un personaje y tema específicos."""

    info_personaje = PERSONAJES[personaje]
    info_tema = TEMAS[tema]

    prompt = f"""Necesito que generes {cantidad} ejemplos de entrenamiento para un dataset de fine-tuning.
Cada ejemplo es una conversación entre un estudiante principiante de Python y un tutor que es {personaje} de Spy x Family.

### GUÍA DE TONO — {personaje.upper()}
{info_personaje['tono']}

### ROL DE TUTOR
- Guía al estudiante a descubrir la respuesta por sí mismo (enfoque socrático)
- NUNCA da la solución directa a un problema
- Siempre plantea un ejercicio o pregunta de seguimiento al final
- Las explicaciones de conceptos deben usar ejemplos cotidianos y neutros (cajas, etiquetas, objetos del día a día)
- PROHIBIDO usar analogías del universo de Spy x Family para explicar conceptos de Python
- Las referencias ambientales a la vida del personaje SÍ están permitidas como comentarios casuales

### TEMA DE PYTHON: {info_tema['titulo']}
{info_tema['contenido']}

### FORMATO DE SALIDA
Genera exactamente {cantidad} ejemplos. Cada ejemplo debe ser un JSON válido con esta estructura exacta:

{{"messages": [{{"role": "system", "content": "[system prompt]"}}, {{"role": "user", "content": "[pregunta]"}}, {{"role": "assistant", "content": "[respuesta]"}}]}}

### REGLAS
1. El campo "system" debe ser SIEMPRE este texto exacto:
   "{info_personaje['system_prompt']}"

2. Las preguntas del estudiante deben ser variadas y naturales, como las haría un principiante real. Cada pregunta debe ser diferente.

3. Las respuestas del tutor deben:
   - Sonar auténticamente como {personaje}
   - Explicar el concepto con ejemplos neutros/cotidianos
   - NO usar analogías de Spy x Family para explicar Python
   - SÍ puede hacer referencias ambientales casuales a su vida
   - Terminar con un ejercicio o pregunta para el estudiante
   - Tener entre 80-150 palabras

4. Devuelve SOLO un JSON array válido con los {cantidad} objetos. Sin texto antes ni después. Sin backticks de markdown. Solo el JSON puro.
"""

    response = model.generate_content(prompt)
    return response.text
```

---

## Paso 7 — Función para parsear y validar

```python
def parsear_respuesta(texto_respuesta, personaje, tema):
    """Intenta parsear la respuesta de Gemini como JSON válido."""

    # Limpiar posibles backticks de markdown
    texto = texto_respuesta.strip()
    if texto.startswith("```json"):
        texto = texto[7:]
    if texto.startswith("```"):
        texto = texto[3:]
    if texto.endswith("```"):
        texto = texto[:-3]
    texto = texto.strip()

    # Si no empieza con [, intentar envolver
    if not texto.startswith("["):
        texto = "[" + texto + "]"

    try:
        ejemplos = json.loads(texto)
        print(f"  ✅ {personaje} × {tema}: {len(ejemplos)} ejemplos parseados")
        return ejemplos
    except json.JSONDecodeError as e:
        print(f"  ❌ {personaje} × {tema}: Error de JSON — {e}")
        print(f"     Guardando respuesta cruda para revisión manual")
        return None


def validar_ejemplo(ejemplo):
    """Verifica que un ejemplo tenga la estructura correcta."""

    if "messages" not in ejemplo:
        return False, "Falta campo 'messages'"

    messages = ejemplo["messages"]
    if len(messages) != 3:
        return False, f"Se esperan 3 mensajes, hay {len(messages)}"

    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        return False, f"Roles incorrectos: {roles}"

    for m in messages:
        if not m.get("content", "").strip():
            return False, f"Contenido vacío en rol '{m['role']}'"

    return True, "OK"
```

---

## Paso 8 — Ejecutar la generación completa

```python
import os

# Crear directorio de salida
os.makedirs("output", exist_ok=True)

# Almacenar todos los ejemplos válidos
dataset_completo = []
errores = []

print("=" * 60)
print("GENERANDO DATASET WAKU WAKU CODE — SPY X FAMILY")
print("=" * 60)

for personaje in PERSONAJES:
    print(f"\n🎭 Personaje: {personaje}")
    print("-" * 40)

    for tema_key, tema_info in TEMAS.items():
        # Generar ejemplos
        try:
            respuesta = generar_ejemplos(personaje, tema_key, cantidad=4)

            # Parsear
            ejemplos = parsear_respuesta(respuesta, personaje, tema_key)

            if ejemplos:
                for ej in ejemplos:
                    valido, msg = validar_ejemplo(ej)
                    if valido:
                        dataset_completo.append(ej)
                    else:
                        errores.append({
                            "personaje": personaje,
                            "tema": tema_key,
                            "error": msg,
                            "ejemplo": ej
                        })
            else:
                # Guardar respuesta cruda para debug
                errores.append({
                    "personaje": personaje,
                    "tema": tema_key,
                    "error": "JSON inválido",
                    "respuesta_cruda": respuesta[:500]
                })

        except Exception as e:
            print(f"  ⚠️  Error en API: {e}")
            errores.append({
                "personaje": personaje,
                "tema": tema_key,
                "error": str(e)
            })

        # Esperar entre requests para no exceder rate limits
        time.sleep(4)

print("\n" + "=" * 60)
print(f"RESUMEN")
print(f"  Ejemplos válidos: {len(dataset_completo)}")
print(f"  Errores: {len(errores)}")
print("=" * 60)
```

---

## Paso 9 — Guardar el dataset

```python
# Guardar como JSONL (un JSON por línea)
with open("output/dataset_sxf.jsonl", "w", encoding="utf-8") as f:
    for ejemplo in dataset_completo:
        f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

print(f"✅ Dataset guardado: output/dataset_sxf.jsonl")
print(f"   Total de ejemplos: {len(dataset_completo)}")

# Guardar errores para revisión
if errores:
    with open("output/errores.json", "w", encoding="utf-8") as f:
        json.dump(errores, f, ensure_ascii=False, indent=2)
    print(f"⚠️  Errores guardados: output/errores.json")
```

---

## Paso 10 — Revisar una muestra

```python
# Ver 3 ejemplos aleatorios para verificar calidad
import random

print("=" * 60)
print("MUESTRA ALEATORIA — REVISIÓN DE CALIDAD")
print("=" * 60)

muestra = random.sample(dataset_completo, min(3, len(dataset_completo)))

for i, ej in enumerate(muestra, 1):
    print(f"\n--- Ejemplo {i} ---")
    print(f"System: {ej['messages'][0]['content'][:80]}...")
    print(f"User: {ej['messages'][1]['content']}")
    print(f"Assistant: {ej['messages'][2]['content'][:200]}...")
    print()
```

---

## Paso 11 — Subir al Hugging Face Hub

```python
!pip install datasets huggingface_hub

from huggingface_hub import login
from datasets import load_dataset

# Login con tu token de HF
login(token="tu-token-hf-aqui")

# Cargar el JSONL como dataset
dataset = load_dataset("json", data_files="output/dataset_sxf.jsonl", split="train")

print(f"Dataset cargado: {len(dataset)} ejemplos")
print(f"Columnas: {dataset.column_names}")
print(f"Ejemplo: {dataset[0]}")

# Subir al Hub como privado
dataset.push_to_hub("tu-usuario/waku-waku-code-sxf", private=True)

print("✅ Dataset subido al Hub")
```

---

## Números esperados

```
5 personajes × 10 temas × 4 ejemplos = 200 ejemplos objetivo

Requests a la API:  50 (una por combinación personaje × tema)
Free tier diario:   1,500 requests
Margen:             97% del free tier sin usar

Tiempo de ejecución: ~4 min (50 requests × 4 seg de espera)
Tiempo de revisión:  ~30-60 min
```

---

## Checklist de verificación post-generación

Antes de usar el dataset para fine-tuning, revisa manualmente al menos 20-30 ejemplos:

- [ ] ¿El tono suena auténticamente como el personaje?
- [ ] ¿Las explicaciones de Python son técnicamente correctas?
- [ ] ¿NO se usan analogías del anime para explicar conceptos?
- [ ] ¿Las referencias ambientales son naturales y no confunden?
- [ ] ¿El tutor NO da respuestas directas?
- [ ] ¿Cada ejemplo termina con un ejercicio o pregunta?
- [ ] ¿Los 5 personajes tienen representación similar?
- [ ] ¿Los 10 temas están cubiertos?
- [ ] ¿El formato JSON es consistente?

---

## Troubleshooting

### "Error 429 — Rate limit exceeded"
Incrementa el `time.sleep(4)` a `time.sleep(10)` en el Paso 8.

### "JSON inválido"
Revisa el archivo `output/errores.json`. Puedes re-ejecutar solo las combinaciones que fallaron, o corregir el JSON manualmente.

### "La respuesta no suena al personaje"
Enriquece la guía de tono en el diccionario PERSONAJES con más frases icónicas y ejemplos de cómo habla.

### "Usó analogías del anime para explicar"
Refuerza la regla en el prompt. Añade ejemplos explícitos de lo que NO debe hacer.

### "Las respuestas son muy cortas / muy largas"
Ajusta el rango "80-150 palabras" en el prompt del Paso 6.