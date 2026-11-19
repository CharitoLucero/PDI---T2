from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".model-cache"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["GRADIO_TEMP_DIR"] = str(CACHE_DIR / "gradio")
os.environ["TEMP"] = str(CACHE_DIR)
os.environ["TMP"] = str(CACHE_DIR)

import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CONTROLNET_ID = "neurallove/controlnet-sd21-colorization-diffusers"
CONTROLNET_REVISION = "2083978ca9345c9edac3af2779c17710b7e42258"
BASE_MODEL_ID = "sd2-community/stable-diffusion-2-1-base"
BASE_MODEL_REVISION = "4e63672c03103b6c636b8fb4119ba982469b2955"
CLIP_ID = "openai/clip-vit-base-patch32"
PROMPT = "colorized vintage black and white family photograph, natural realistic colors, preserve the original composition and details"
NEGATIVE_PROMPT = "black and white, grayscale, sepia, changed composition, distorted face, artifacts, text"
ETIQUETAS_CLIP = ["a black and white photograph", "a color photograph"]

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_ID, revision=CONTROLNET_REVISION, use_safetensors=False, torch_dtype=DTYPE
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL_ID, revision=BASE_MODEL_REVISION, controlnet=controlnet, torch_dtype=DTYPE
).to(DEVICE)
pipe.enable_attention_slicing()
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=DTYPE).to(DEVICE).eval()

def preparar_imagen(imagen, lado_maximo=512):
    original = ImageOps.grayscale(imagen).convert("RGB")
    ancho, alto = original.size
    escala = min(1.0, lado_maximo / max(ancho, alto))
    nuevo_ancho = max(64, int(ancho * escala) // 8 * 8)
    nuevo_alto = max(64, int(alto * escala) // 8 * 8)
    reducida = original.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    return original, reducida

def colorizar(imagen, semilla=42, pasos=15):
    original, control = preparar_imagen(imagen)
    generador = torch.Generator(device=DEVICE).manual_seed(int(semilla))
    resultado = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, image=control,
        width=control.width, height=control.height,
        num_inference_steps=int(pasos), guidance_scale=7.5,
        controlnet_conditioning_scale=1.0, generator=generador,
    ).images[0]
    return original, resultado.resize(original.size, Image.Resampling.LANCZOS)

@torch.inference_mode()
def analizar_con_clip(imagenes):
    entradas = clip_processor(
        text=ETIQUETAS_CLIP, images=imagenes, return_tensors="pt", padding=True
    )
    entradas = {nombre: tensor.to(DEVICE) for nombre, tensor in entradas.items()}
    return clip_model(**entradas).logits_per_image.softmax(dim=1).cpu().tolist()

def procesar_foto(imagen, semilla=42, pasos=15):
    if imagen is None:
        raise gr.Error("Cargá una imagen para comenzar.")
    original, colorizada = colorizar(imagen, semilla=semilla, pasos=pasos)
    puntajes = analizar_con_clip([original, colorizada])
    reporte = "| Imagen | Blanco y negro | A color |\n|---|---:|---:|\n"
    for nombre, fila in zip(["Original", "Colorizada"], puntajes):
        reporte += f"| {nombre} | {fila[0]:.1%} | {fila[1]:.1%} |\n"
    reporte += "\n*CLIP compara etiquetas; no comprueba los colores originales.*"
    return original, colorizada, reporte

def crear_interfaz():
    with gr.Blocks(title="Colorizador de fotos familiares") as demo:
        gr.Markdown("# Colorizador de fotos familiares\n**Los colores son una propuesta de IA. Conservá tu original.**")
        with gr.Row():
            entrada = gr.Image(label="Foto en blanco y negro", type="pil")
            original_salida = gr.Image(label="Original en escala de grises", type="pil")
            color_salida = gr.Image(label="Propuesta colorizada", type="pil")
        pasos = gr.Slider(5, 25, value=15, step=1, label="Pasos de difusión")
        semilla = gr.Number(value=42, precision=0, label="Semilla")
        boton = gr.Button("Colorizar y analizar")
        analisis = gr.Markdown()
        boton.click(procesar_foto, [entrada, semilla, pasos], [original_salida, color_salida, analisis])
    return demo

if __name__ == "__main__":
    crear_interfaz().launch(share=False)
