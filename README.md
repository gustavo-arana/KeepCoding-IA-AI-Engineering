# KeepCoding-IA-AI-Engineering

---

Repositorio de Entrega de la Práctica de AI Engineering

## Contexto

En este repositorio se trabajan los conceptos/herramientas vistas en el módulo Prompt Engineering, AI Code Assistant, Hugging Face, Fine-tuning, generación de datasets y LLM as Judge.

Alineados con el proyecto final "Waku Code", un AI Assistant ambientado con el estilo, tono, palabras y personajes de Spy x Family. Su propósito es enseñar las bases de programación de Python. **Este laboratorio aplica únicamente para el canal de texto; el canal multimodal queda fuera del alcance del lab.**

## Ejecución del Laboratorio

### 1. Generación de datasets sintéticos vía API de Gemini 2.5 Flash

Para resolver nuestro caso de uso y hacer fine-tuning del proyecto, se determinó generar un dataset utilizando Gemini 2.5 Flash, donde se identificaron los personajes principales de la serie y se delimitaron los temas básicos que se quieren abordar en el aprendizaje.

Se creó la función de Python `generate_dataset` que, generando un system prompt por personaje (5) y tema (10), produce 4 resultados consumiendo la API de Gemini, generando un dataset de 200 registros para el entrenamiento.

```python
PERSONAJES = ["Anya Forger", "Loid Forger", "Yor Forger", "Franky Franklin", "Damian Desmond"]

TEMAS = ["que_es_programacion", "que_es_algoritmo", "sintaxis_basica", "print_input",
         "variables_tipos", "conversion_tipos", "operadores_aritmeticos",
         "if_elif_else", "operadores_comparacion", "operadores_logicos"]
```

Los resultados se almacenan en `data/dataset_sxf.jsonl`.

![google-studio-api](images/google-studio-api.jpg)

Dashboard de uso de la API de Gemini 2.5 Flash en Google AI Studio.

### 2. Cargue del dataset a HF

Como plataforma de despliegue se va a utilizar Hugging Face. El dataset se carga con la función [`tools/upload_to_hf.py`](tools/upload_to_hf.py) al repositorio privado `garana-osorio/waku-code-sxf`.

![hf-1](images/hf-1.jpg)

Vista del dataset `garana-osorio/waku-code-sxf` en Hugging Face.

![hf-2](images/hf-2.jpg)

Detalle en JSON de un registro del dataset, con la estructura de tres roles: `system` (personalidad del personaje), `user` (pregunta del estudiante) y `assistant` (respuesta del tutor).

### 3. Entrenamiento del Modelo (Qwen 3 8B) y cargue del adapter en HF

Utilizando como base el Notebook del Entrenamiento de Milei, se realiza la ejecución del laboratorio usando Unsloth para hacer fine-tuning del modelo Qwen 3 8B, para que al hacer una pregunta sobre Python responda con el estilo/tono del rol de uno de los personajes.

El entrenamiento se ejecutó en Kaggle con **GPU T4 x2**. El adapter resultante se publicó en `garana-osorio/waku-code-qlora-adapter` (186 MB).

Notebook: [`Waku-code`](https://www.kaggle.com/)

![kaggle-milei](images/kaggle-milei.jpg)

Notebook base `Practica Milei` en Kaggle, utilizado como referencia para el fine-tuning.

![kaggle-waku-code](images/kaggle-waku-code.jpg)

Notebook `Waku-Code` corriendo en Kaggle con GPU T4 x2 activa (sesión 1h56m).

![hf-3](images/hf-3.jpg)

Repositorio del adapter `garana-osorio/waku-code-qlora-adapter` publicado en Hugging Face.

### 4. Evaluación del dataset con la técnica de LLM as Judge

La idea de la ejecución del LLM as Judge sobre el dataset se incorporó después del entrenamiento del modelo. Por tiempo no se reentrenó, pero se agrega como TODO del proyecto.

Se implementa la función [`tools/judge_dataset.py`](tools/judge_dataset.py) que utiliza la API Key de Anthropic y el modelo Claude Haiku 4.5 para validar que el dataset cumpla las siguientes reglas.

![llm-as-judge-1](images/llm-as-judge-1.jpg)

Ejecución del script `judge_dataset.py` en terminal, mostrando la evaluación de los 200 ejemplos con Claude Haiku 4.5.

#### Resultados

| Métrica | Valor |
| --- | --- |
| Total evaluados | 200 |
| ✅ Aprobados | 126 (63%) |
| ❌ Rechazados | 61 |
| ⚠️ Errores de parseo | 13 |
| Puntuación media | 7.8/10 |
| Costo | $0.47 USD |

**Principal causa de rechazo:** `analogia_anime` (55 casos) — el modelo generador usó referencias del universo Spy x Family para explicar conceptos de Python, violando la regla de ejemplos cotidianos neutros.

![llm-as-judge-2](images/llm-as-judge-2.jpg)

Resumen de la revisión y resultados del LLM as Judge.

![claude-api](images/claude-api.jpg)

Dashboard de uso de la API de Anthropic.

#### Reglas de contenido

1. **PERSONALIDAD DEL PERSONAJE (OBLIGATORIO):**
   - El assistant debe hablar con la personalidad del personaje indicado en el system prompt
   - Debe usar sus muletillas y forma de hablar características
   - Anya: tercera persona, "waku waku", infantil
   - Loid: metódico, preciso, referencias a misiones
   - Yor: amable, cálida, alentadora
   - Franky: informal, quejumbroso, humor
   - Damian: orgulloso, competitivo, retador

2. **PROHIBIDO — ANALOGÍAS DEL ANIME PARA EXPLICAR CONCEPTOS:**
   - NO debe usar elementos del universo de Spy x Family como analogía para explicar conceptos de Python
   - Nombres prohibidos en explicaciones de conceptos: Bond, Eden, WISE, Ostania, Berlint, Westalis, Operation Strix, Desmond, Donovan, Handler, Yuri, Nightfall, SSS
   - Ejemplo INCORRECTO: "Una variable es como las identidades encubiertas de papá"
   - Ejemplo INCORRECTO: "Un if es como cuando Bond presiente el peligro"
   - Las explicaciones deben usar ejemplos COTIDIANOS y NEUTROS (cajas, etiquetas, recetas, objetos del día a día)

3. **PERMITIDO — REFERENCIAS AMBIENTALES:**
   - SÍ puede hacer comentarios casuales sobre su vida que NO expliquen un concepto de Python
   - Ejemplo CORRECTO: "Anya tuvo un día largo en el colegio, pero esto es más divertido"
   - Ejemplo CORRECTO: "Esto requiere la concentración de una misión de alto nivel" (Loid)
   - La referencia ambiental debe estar SEPARADA de la explicación del concepto

4. **ENFOQUE SOCRÁTICO:**
   - NO debe dar la respuesta directa a un problema
   - Debe guiar al estudiante a descubrir la respuesta
   - Debe plantear un ejercicio o pregunta de seguimiento al final

5. **PYTHON CORRECTO:**
   - El código Python mostrado debe ser sintácticamente correcto
   - Las explicaciones técnicas deben ser precisas

### 5. Deploy en HF Spaces con Gradio

Se hace el deploy en un Space de HF usando el archivo [`space/app.py`](space/app.py) con Gradio, el modelo base Qwen3-8B y el adapter `garana-osorio/waku-code-qlora-adapter` generado en el entrenamiento.

![hf-4](images/hf-4.jpg)

Archivos del Space `garana-osorio/waku-code` en Hugging Face: `app.py` con la lógica de Gradio, `requirements.txt` con las dependencias y el `README.md` de configuración del Space.

![hf-6](images/hf-6.jpg)

Interfaz del Space en ejecución con el dropdown para seleccionar entre los 5 personajes-tutor.

![hf-7](images/hf-7.jpg)

Respuesta de Anya Forger a "¿Qué es una variable?".

![hf-8](images/hf-8.jpg)

Respuesta de Loid Forger a "¿Qué es una variable?".

![hf-9](images/hf-9.jpg)

Respuesta de Damian Desmond a "¿Qué es una variable?".

![hf-0](images/hf-0.jpg)

Perfil de Hugging Face con los tres recursos del proyecto: Space, adapter QLoRA y el dataset.

## Requisitos

- **Kaggle** — debe estar autenticada con número de teléfono para acceder a GPU T4. Se decide Kaggle sobre Colab porque permite ejecutar notebooks hasta por 12 horas.
- **Hugging Face** — suscripción Pro requerida para el despliegue en Spaces con ZeroGPU.
- **Google AI Studio** — cuenta y API Key de Gemini Free Tier es suficiente.
- **Anthropic** — cuenta y API Key para el LLM as Judge, se deben cargar créditos.

## Aspectos por mejorar

- [ ] Reentrenar el modelo con el dataset limpio (126 ejemplos aprobados por el juez).
- [ ] Ampliar el dataset de entrenamiento.
- [ ] Implementar un LLM as Judge para evaluar las respuestas del modelo entrenado (Modelo Frontier vs Qwen fine-tuned).
- [ ] Implementar mlflow para evaluar el proceso de entrenamiento del modelo.
- [ ] Implementar Langfuse para evaluar las respuestas del modelo.
- [ ] Hacer un análisis de sentimientos de las respuestas generadas para incluir una métrica que determine las palabras clave de cada rol/ambiente, medir su uso en las respuestas y evaluar el estilo.
