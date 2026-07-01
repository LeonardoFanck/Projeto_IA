import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Flowers Recognition",
    page_icon="🌸",
    layout="centered"
)

# Carrega o modelo
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model("melhor_cnn2.keras")

model = carregar_modelo()

# Classes do dataset
classes = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

st.title("🌸 Flowers Recognition")
st.write("Faça upload de uma imagem de uma flor para classificá-la.")

arquivo = st.file_uploader(
    "Selecione uma imagem",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:

    imagem = Image.open(arquivo).convert("RGB")

    st.image(imagem, caption="Imagem enviada", use_container_width=True)

    # Ajuste o tamanho abaixo para o mesmo usado no treinamento
    img = imagem.resize((180, 180))

    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Classificando..."):
        pred = model.predict(img, verbose=0)

    indice = np.argmax(pred)
    confianca = float(pred[0][indice]) * 100

    st.success(f"### 🌼 Resultado: {classes[indice]}")
    st.metric("Confiança", f"{confianca:.2f}%")

    st.subheader("Probabilidade por classe")

    for i, classe in enumerate(classes):
        st.progress(float(pred[0][i]))
        st.write(f"{classe}: {pred[0][i]*100:.2f}%")