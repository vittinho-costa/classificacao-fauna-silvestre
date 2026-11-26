import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gradio as gr

# Carrega o modelo pré-treinado e as classes salvas
print("Carregando o modelo e as classes...")
model = tf.keras.models.load_model('modelo_animais_brasileiros.keras')

with open('classes.json', 'r') as f:
    classes_dict = json.load(f)

# Garante que as classes fiquem ordenadas pelo índice do modelo
classes = [k for k, v in sorted(classes_dict.items(), key=lambda item: item[1])]

# --- FUNÇÃO DE PREDIÇÃO ---
def predizer_interface(imagem_pil):
    if imagem_pil is None:
        return "Nenhuma imagem enviada."

    img = imagem_pil.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicoes = model.predict(img_array, verbose=0)[0]
    return {classes[i]: float(predicoes[i]) for i in range(len(classes))}

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
custom_css = """
/* Fundo da aplicação */
.gradio-container {
    background-color: #f4f7f6;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header estilizado */
.header-container {
    text-align: center;
    background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
    color: white;
    padding: 30px 20px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
}

.header-container h1 {
    font-size: 2.2rem;
    margin-bottom: 8px;
    font-weight: 700;
}

.header-container p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Botão principal com hover animado */
#btn-classificar {
    background: linear-gradient(135deg, #ff8f00 0%, #ff6f00 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0px 4px 10px rgba(255, 111, 0, 0.3) !important;
}

#btn-classificar:hover {
    transform: translateY(-2px);
    box-shadow: 0px 6px 15px rgba(255, 111, 0, 0.4) !important;
}

/* Estilização das colunas */
.custom-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
}

/* Rodapé */
.footer-text {
    text-align: center;
    margin-top: 30px;
    color: #666;
    font-size: 0.9rem;
}
"""

# --- INTERFACE GRADIO ---
with gr.Blocks(css=custom_css, title="Fauna BR - Classifier") as demo:

    # Banner Superior em HTML
    gr.HTML(
        """
        <div class="header-container">
            <h1>🐾 Classificador da Fauna Brasileira</h1>
            <p>Envie uma imagem ou ative sua webcam para identificar espécies nativas em tempo real.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["custom-card"]):
            input_img = gr.Image(
                type="pil",
                label="Envio da Imagem",
                sources=["upload", "webcam"]
            )
            btn_classificar = gr.Button("🔍 Classificar Animal", elem_id="btn-classificar")

        with gr.Column(scale=1, elem_classes=["custom-card"]):
            output_labels = gr.Label(
                num_top_classes=3,
                label="Resultados da Análise (Top 3)"
            )

    # Rodapé em HTML
    gr.HTML(
        """
        <div class="footer-text">
            <p>Desenvolvido com MobileNetV2 e TensorFlow • Projeto de Visão Computacional</p>
        </div>
        """
    )

    btn_classificar.click(
        fn=predizer_interface,
        inputs=input_img,
        outputs=output_labels
    )

print("Iniciando a interface gráfica...")
demo.launch()