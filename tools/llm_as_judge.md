# 🧑‍⚖️ Waku Waku Code — LLM as Judge: Revisión del Dataset con Claude Haiku 4.5

## Resumen

Este script revisa cada ejemplo del dataset generado por Gemini usando Claude Haiku 4.5 como juez.
Verifica que cada ejemplo cumpla las reglas de contenido definidas para el proyecto.

**Costo estimado:** < $0.50 USD por 200 ejemplos.

**Ejecutar en:** Google Colab (sin GPU, no se necesita).

---

## Paso 1 — Instalar dependencia

```python
!pip install anthropic
```

---

## Paso 2 — Configurar API key

```python
import anthropic

# Tu API key de Anthropic
# Obtenerla en: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY = "tu-api-key-aqui"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```

### Verificación rápida

```python
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Dime hola en español"}]
)
print(message.content[0].text)
```

---

## Paso 3 — Definir las reglas del juez

```python
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
   - Nombres prohibidos en explicaciones de conceptos: Bond, Eden, WISE, Ostania, Berlint, Westalis, Operation Strix, Desmond, Donovan, Handler, Yuri, Nightfall, SSS
   - Ejemplo INCORRECTO: "Una variable es como las identidades encubiertas de papá"
   - Ejemplo INCORRECTO: "Un if es como cuando Bond presiente el peligro"
   - Las explicaciones deben usar ejemplos COTIDIANOS y NEUTROS (cajas, etiquetas, recetas, objetos del día a día)

3. PERMITIDO — REFERENCIAS AMBIENTALES:
   - SÍ puede hacer comentarios casuales sobre su vida que NO expliquen un concepto de Python
   - Ejemplo CORRECTO: "Anya tuvo un día largo en el colegio, pero esto es más divertido" (ambiental, no explica Python)
   - Ejemplo CORRECTO: "Esto requiere la concentración de una misión de alto nivel" (ambiental de Loid)
   - La referencia ambiental debe estar SEPARADA de la explicación del concepto

4. ENFOQUE SOCRÁTICO:
   - NO debe dar la respuesta directa a un problema
   - Debe guiar al estudiante a descubrir la respuesta
   - Debe plantear un ejercicio o pregunta de seguimiento al final

5. PYTHON CORRECTO:
   - El código Python mostrado en la respuesta debe ser sintácticamente correcto
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
  "sugerencia_correccion": "texto con sugerencia si hay problemas, o 'N/A' si está aprobado"
}

Un ejemplo se considera APROBADO solo si:
- personalidad_correcta = true
- analogia_anime_en_explicacion = false (false significa que NO usó analogías del anime, que es lo correcto)
- enfoque_socratico = true
- python_correcto = true
- tiene_ejercicio_final = true
"""
```

---

## Paso 4 — Función de evaluación

```python
import json
import time

def evaluar_ejemplo(ejemplo, indice):
    """Envía un ejemplo al juez y devuelve el veredicto."""

    system_msg = ejemplo["messages"][0]["content"]
    user_msg = ejemplo["messages"][1]["content"]
    assistant_msg = ejemplo["messages"][2]["content"]

    prompt_evaluacion = f"""Evalúa el siguiente ejemplo de dataset:

SYSTEM PROMPT DEL EJEMPLO:
{system_msg}

PREGUNTA DEL ESTUDIANTE:
{user_msg}

RESPUESTA DEL TUTOR:
{assistant_msg}

Aplica las reglas y devuelve tu veredicto en el formato JSON especificado."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=REGLAS_JUEZ,
            messages=[{"role": "user", "content": prompt_evaluacion}]
        )

        respuesta = message.content[0].text.strip()

        # Limpiar posibles backticks
        if respuesta.startswith("```json"):
            respuesta = respuesta[7:]
        if respuesta.startswith("```"):
            respuesta = respuesta[3:]
        if respuesta.endswith("```"):
            respuesta = respuesta[:-3]
        respuesta = respuesta.strip()

        veredicto = json.loads(respuesta)
        veredicto["indice"] = indice
        veredicto["tokens_input"] = message.usage.input_tokens
        veredicto["tokens_output"] = message.usage.output_tokens

        estado = "✅" if veredicto["aprobado"] else "❌"
        print(f"  {estado} Ejemplo #{indice} — Puntuación: {veredicto['puntuacion']}/10")

        if not veredicto["aprobado"]:
            for problema in veredicto.get("problemas", []):
                print(f"     ⚠️  {problema}")

        return veredicto

    except json.JSONDecodeError as e:
        print(f"  ⚠️  Ejemplo #{indice} — Error parseando JSON: {e}")
        return {
            "indice": indice,
            "aprobado": None,
            "error": f"JSON inválido: {str(e)}",
            "respuesta_cruda": respuesta[:300]
        }

    except Exception as e:
        print(f"  ⚠️  Ejemplo #{indice} — Error API: {e}")
        return {
            "indice": indice,
            "aprobado": None,
            "error": str(e)
        }
```

---

## Paso 5 — Cargar el dataset y ejecutar revisión

```python
# Cargar dataset desde archivo local
dataset_ejemplos = []
with open("output/dataset_sxf.jsonl", "r", encoding="utf-8") as f:
    for linea in f:
        dataset_ejemplos.append(json.loads(linea))

# O si lo subiste al Hub:
# from datasets import load_dataset
# ds = load_dataset("tu-usuario/waku-waku-code-sxf", split="train")
# dataset_ejemplos = [ds[i] for i in range(len(ds))]

print(f"Dataset cargado: {len(dataset_ejemplos)} ejemplos")
print("=" * 60)
print("INICIANDO REVISIÓN CON CLAUDE HAIKU 4.5")
print("=" * 60)

# Ejecutar evaluación
veredictos = []
total_input_tokens = 0
total_output_tokens = 0

for i, ejemplo in enumerate(dataset_ejemplos):
    veredicto = evaluar_ejemplo(ejemplo, i)
    veredictos.append(veredicto)

    if veredicto.get("tokens_input"):
        total_input_tokens += veredicto["tokens_input"]
        total_output_tokens += veredicto["tokens_output"]

    # Esperar entre requests para no exceder rate limits
    time.sleep(1)

print("\n" + "=" * 60)
print("REVISIÓN COMPLETADA")
print("=" * 60)
```

---

## Paso 6 — Generar reporte de resultados

```python
# Clasificar resultados
aprobados = [v for v in veredictos if v.get("aprobado") == True]
rechazados = [v for v in veredictos if v.get("aprobado") == False]
errores = [v for v in veredictos if v.get("aprobado") is None]

print(f"\n📊 RESUMEN DE REVISIÓN")
print(f"{'─' * 40}")
print(f"  Total ejemplos:    {len(veredictos)}")
print(f"  ✅ Aprobados:      {len(aprobados)} ({len(aprobados)/len(veredictos)*100:.1f}%)")
print(f"  ❌ Rechazados:     {len(rechazados)} ({len(rechazados)/len(veredictos)*100:.1f}%)")
print(f"  ⚠️  Errores:        {len(errores)}")
print(f"{'─' * 40}")

# Puntuación promedio
puntuaciones = [v["puntuacion"] for v in veredictos if "puntuacion" in v]
if puntuaciones:
    print(f"  Puntuación promedio: {sum(puntuaciones)/len(puntuaciones):.1f}/10")

# Costo
costo_input = (total_input_tokens / 1_000_000) * 1.00   # Haiku input: $1/MTok
costo_output = (total_output_tokens / 1_000_000) * 5.00  # Haiku output: $5/MTok
costo_total = costo_input + costo_output

print(f"\n💰 COSTO")
print(f"{'─' * 40}")
print(f"  Tokens input:  {total_input_tokens:,}")
print(f"  Tokens output: {total_output_tokens:,}")
print(f"  Costo total:   ${costo_total:.4f} USD")
```

---

## Paso 7 — Análisis detallado de problemas

```python
# Contar tipos de problemas
from collections import Counter

if rechazados:
    print(f"\n🔍 DETALLE DE RECHAZADOS")
    print(f"{'─' * 40}")

    # Analizar por tipo de fallo
    fallos = {
        "personalidad_incorrecta": 0,
        "analogia_anime": 0,
        "no_socratico": 0,
        "python_incorrecto": 0,
        "sin_ejercicio_final": 0,
        "referencia_ambiental_incorrecta": 0,
    }

    for v in rechazados:
        detalle = v.get("detalle", {})
        if not detalle.get("personalidad_correcta", True):
            fallos["personalidad_incorrecta"] += 1
        if detalle.get("analogia_anime_en_explicacion", False):
            fallos["analogia_anime"] += 1
        if not detalle.get("enfoque_socratico", True):
            fallos["no_socratico"] += 1
        if not detalle.get("python_correcto", True):
            fallos["python_incorrecto"] += 1
        if not detalle.get("tiene_ejercicio_final", True):
            fallos["sin_ejercicio_final"] += 1
        if not detalle.get("referencia_ambiental_correcta", True):
            fallos["referencia_ambiental_incorrecta"] += 1

    for fallo, cantidad in sorted(fallos.items(), key=lambda x: -x[1]):
        if cantidad > 0:
            barra = "█" * cantidad
            print(f"  {fallo}: {cantidad} {barra}")

    # Mostrar los 5 peores ejemplos
    rechazados_ordenados = sorted(rechazados, key=lambda x: x.get("puntuacion", 10))
    print(f"\n🔴 TOP 5 PEORES EJEMPLOS")
    print(f"{'─' * 40}")
    for v in rechazados_ordenados[:5]:
        print(f"\n  Ejemplo #{v['indice']} — Puntuación: {v.get('puntuacion', '?')}/10")
        for problema in v.get("problemas", []):
            print(f"    ⚠️  {problema}")
        print(f"    💡 {v.get('sugerencia_correccion', 'Sin sugerencia')}")
```

---

## Paso 8 — Exportar resultados

```python
# Guardar reporte completo
with open("output/revision_juez.json", "w", encoding="utf-8") as f:
    json.dump(veredictos, f, ensure_ascii=False, indent=2)

# Guardar solo los rechazados para corregir
with open("output/rechazados.json", "w", encoding="utf-8") as f:
    rechazados_con_ejemplo = []
    for v in rechazados:
        idx = v["indice"]
        rechazados_con_ejemplo.append({
            "veredicto": v,
            "ejemplo_original": dataset_ejemplos[idx]
        })
    json.dump(rechazados_con_ejemplo, f, ensure_ascii=False, indent=2)

# Guardar solo los aprobados como dataset limpio
with open("output/dataset_sxf_aprobado.jsonl", "w", encoding="utf-8") as f:
    for v in aprobados:
        idx = v["indice"]
        f.write(json.dumps(dataset_ejemplos[idx], ensure_ascii=False) + "\n")

print(f"\n📁 ARCHIVOS GENERADOS")
print(f"{'─' * 40}")
print(f"  output/revision_juez.json          — Reporte completo ({len(veredictos)} veredictos)")
print(f"  output/rechazados.json             — Ejemplos a corregir ({len(rechazados)})")
print(f"  output/dataset_sxf_aprobado.jsonl  — Dataset limpio ({len(aprobados)} ejemplos)")
```

---

## Paso 9 — Decisión post-revisión

```python
print(f"\n🎯 SIGUIENTE PASO")
print(f"{'─' * 40}")

porcentaje_aprobado = len(aprobados) / len(veredictos) * 100

if porcentaje_aprobado >= 90:
    print(f"  {porcentaje_aprobado:.0f}% aprobado — Dataset listo para fine-tuning.")
    print(f"  Usa: output/dataset_sxf_aprobado.jsonl")
    print(f"  Los {len(rechazados)} rechazados puedes corregirlos manualmente o regenerarlos.")

elif porcentaje_aprobado >= 70:
    print(f"  {porcentaje_aprobado:.0f}% aprobado — Dataset aceptable pero mejorable.")
    print(f"  Revisa output/rechazados.json y corrige los {len(rechazados)} ejemplos.")
    print(f"  Luego vuelve a pasar el juez solo sobre los corregidos.")

else:
    print(f"  {porcentaje_aprobado:.0f}% aprobado — Dataset necesita trabajo significativo.")
    print(f"  Revisa el prompt de generación de Gemini.")
    print(f"  Refuerza las reglas de contenido y regenera.")
```

---

## Flujo completo

```
dataset_sxf.jsonl (200 ejemplos de Gemini)
         │
         ▼
   Claude Haiku 4.5 (juez)
         │
         ├──► dataset_sxf_aprobado.jsonl  → listo para fine-tuning
         │
         ├──► rechazados.json             → corregir y re-evaluar
         │
         └──► revision_juez.json          → reporte completo
```