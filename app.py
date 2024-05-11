import streamlit as st
from utils import *
# Limpiar la memoria de la GPU
torch.cuda.empty_cache()

# Función para descargar la imagen utilizando el modelo de difusión
def generate_image(prompt):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_name = "ByteDance/Hyper-SD"
    ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
    
    # Cargar el modelo
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
    pipe.fuse_lora()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    # Generar la imagen
    image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]
    
    return image

# Función para clasificar una imagen
def classify_image(image):
    # Usar un modelo de HuggingFace para clasificar la imagen
    clasificador_imagenes = pipeline("image-classification")
    resultados_clasificacion = clasificador_imagenes(image)

    # Mostrar el resultado con la mayor confianza (top 1)
    top_resultado = resultados_clasificacion[0]  # Tomar el primer resultado
    return top_resultado

# Configurar la página
st.set_page_config(page_title="App de generación y calsificacion de imagenes", page_icon="🤗", layout="wide")

# Título de la aplicación
st.title("App de generacion y clasificacion de iamgenes")

# Sección de Generación de Imágenes y Clasificación de Imágenes
col1, col2 = st.columns(2)

# Sección de Generación de Imágenes (col1)
with col1:
    st.header("Generación de Imágenes")
    texto_generacion = st.text_input("Ingrese el texto para generar una imagen:")

    # Generar imagen cuando se presiona el botón
    if st.button("Generar Imagen"):
        # Generar la imagen utilizando el modelo de difusión
        imagen_generada = generate_image(texto_generacion)

        # Mostrar la imagen generada
        st.image(imagen_generada, caption="Imagen Generada", use_column_width=True)

# Sección de Clasificación de Imágenes (col2)
with col2:
    st.header("Clasificación de Imágenes")
    uploaded_file = st.file_uploader("Subir imagen para clasificación:", type=["jpg", "png"])

    # Clasificar imagen cuando se carga un archivo
    if uploaded_file is not None:
        # Leer el contenido del archivo cargado
        image_bytes = uploaded_file.read()

        # Convertir el contenido en una imagen PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Mostrar la imagen cargada
        st.image(image, caption="Imagen Cargada", use_column_width=True)

        # Clasificar la imagen
        top_resultado = classify_image(image)

        # Mostrar el resultado con la mayor confianza (top 1)
        st.subheader("Resultado de Clasificación:")
        st.write(f"Clase: {top_resultado['label']} - Confianza: {top_resultado['score']}")

# Habilitar el botón para clasificar la imagen generada si existe
if 'imagen_generada' in locals():
    if imagen_generada is not None:
        if st.button("Clasificar Imagen Generada"):
            # Clasificar la imagen generada
            top_resultado = classify_image(imagen_generada)

            # Mostrar el resultado con la mayor confianza (top 1)
            st.subheader("Resultado de Clasificación:")
            st.write(f"Clase: {top_resultado['label']} - Confianza: {top_resultado['score']}")







