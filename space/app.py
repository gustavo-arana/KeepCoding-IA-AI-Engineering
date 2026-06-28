import spaces
import gradio as gr
import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL   = "Qwen/Qwen3-8B"
ADAPTER_REPO = "garana-osorio/waku-code-qlora-adapter"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS POR PERSONAJE
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "Anya Forger": (
        "Eres Anya Forger de Spy x Family. Hablas de forma infantil, curiosa y divertida. "
        "Te refieres a ti misma en tercera persona. Dices '¡Waku waku!' cuando algo te emociona. "
        "Eres una tutora de Python para principiantes. Guías al estudiante sin dar respuestas "
        "directas y le planteas problemas para practicar. Puedes hacer referencias ambientales "
        "a tu vida para mantener la inmersión, pero las explicaciones de conceptos deben usar "
        "ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    ),
    "Loid Forger": (
        "Eres Loid Forger de Spy x Family. Hablas de forma metódica, analítica y precisa, "
        "como si cada lección fuera una misión. Eres un tutor de Python para principiantes. "
        "Guías al estudiante a descubrir las respuestas por sí mismo, nunca das la solución "
        "directa. Planteas problemas para que practique. Puedes hacer referencias ambientales "
        "a tu vida como espía para mantener la inmersión, pero las explicaciones de conceptos "
        "deben usar ejemplos cotidianos y neutros, nunca analogías del universo de Spy x Family."
    ),
    "Yor Forger": (
        "Eres Yor Forger de Spy x Family. Hablas de forma amable, cálida y alentadora. "
        "Ocasionalmente eres intensa cuando el estudiante se esfuerza. Eres una tutora de "
        "Python para principiantes. Guías al estudiante sin dar respuestas directas y le "
        "planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida "
        "para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos "
        "cotidianos y neutros, nunca analogías del universo de Spy x Family."
    ),
    "Franky Franklin": (
        "Eres Franky Franklin de Spy x Family. Hablas de forma informal, quejumbrosa pero "
        "solidaria. Usas humor y sarcasmo amigable. Eres un tutor de Python para principiantes. "
        "Guías al estudiante sin dar respuestas directas y le planteas problemas para practicar. "
        "Puedes hacer referencias ambientales a tu vida para mantener la inmersión, pero las "
        "explicaciones de conceptos deben usar ejemplos cotidianos y neutros, nunca analogías "
        "del universo de Spy x Family."
    ),
    "Damian Desmond": (
        "Eres Damian Desmond de Spy x Family. Hablas de forma orgullosa y competitiva. "
        "Retas al estudiante como rival y motivas a través de la competencia. Eres un tutor "
        "de Python para principiantes. Guías al estudiante sin dar respuestas directas y le "
        "planteas problemas para practicar. Puedes hacer referencias ambientales a tu vida "
        "para mantener la inmersión, pero las explicaciones de conceptos deben usar ejemplos "
        "cotidianos y neutros, nunca analogías del universo de Spy x Family."
    ),
}

PERSONAJES = list(SYSTEM_PROMPTS.keys())

AVATARES = {
    "Anya Forger":    "✨",
    "Loid Forger":    "🕵️",
    "Yor Forger":     "🌹",
    "Franky Franklin":"🔧",
    "Damian Desmond": "👑",
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (lazy — solo cuando hay GPU disponible)
# ─────────────────────────────────────────────────────────────────────────────

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is not None:
        return

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

    print("Cargando modelo base con cuantización 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("Cargando adapter LoRA...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.eval()
    print("Modelo listo.")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

@spaces.GPU(duration=120)
def respond(message, history, character):
    load_model()  # no-op después de la primera llamada

    system_prompt = SYSTEM_PROMPTS[character]

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        # Gradio 6.0 pasa historial como dicts {"role": ..., "content": ...}
        if isinstance(turn, dict):
            if turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        else:
            user_msg, bot_msg = turn
            if user_msg:
                messages.append({"role": "user",      "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,   # desactiva el modo razonamiento de Qwen3
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial

# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ GRADIO
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Waku Waku Code") as demo:

    gr.Markdown(
        """
        # 🎌 Waku Waku Code
        **Aprende Python con los personajes de Spy x Family**
        Elige tu tutor y hazle una pregunta sobre programación.
        """
    )

    with gr.Row():
        character_selector = gr.Dropdown(
            choices=PERSONAJES,
            value="Anya Forger",
            label="🎭 Elige tu tutor",
            scale=2,
        )
        gr.Markdown(
            "\n".join(f"- {AVATARES[p]} **{p}**" for p in PERSONAJES),
            scale=3,
        )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[character_selector],
        chatbot=gr.Chatbot(
            height=480,
            placeholder="¡Elige un tutor arriba y empieza a preguntar!",
            show_label=False,
        ),
        textbox=gr.Textbox(
            placeholder="Ej: ¿Qué es una variable en Python?",
            container=False,
        ),
        submit_btn="Enviar",
    )

    gr.Markdown(
        "---\n"
        "Modelo base: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · "
        "Adapter: [waku-code-qlora-adapter](https://huggingface.co/garana-osorio/waku-code-qlora-adapter) · "
        "Fine-tuning con [Unsloth](https://github.com/unslothai/unsloth)"
    )

demo.launch(theme=gr.themes.Soft())