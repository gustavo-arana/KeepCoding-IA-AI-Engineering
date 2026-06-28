import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import setup_gemini, generar_ejemplos

client = setup_gemini()
raw = generar_ejemplos(client, "Anya Forger", "variables_tipos", 2)
print(raw)
