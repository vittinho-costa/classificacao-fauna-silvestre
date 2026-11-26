import os
import zipfile
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# --- 1. DESCOMPACTAR IMAGENS ---
zip_path = 'dados_animais.zip'
if os.path.exists(zip_path) and not os.path.exists('dados_animais'):
    print("Descompactando o arquivo de imagens...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Descompactado com sucesso!")

# Configurações de dimensão
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. GERADORES DE DADOS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("Carregando dados de treino...")
train_generator = train_datagen.flow_from_directory(
    'dados_animais/treino',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Carregando dados de validação...")
val_generator = val_datagen.flow_from_directory(
    'dados_animais/validação',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Salva as classes para o arquivo app.py usar depois
with open('classes.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Mapeamento de classes salvo em 'classes.json'.")

# --- 3. CONSTRUÇÃO E TREINAMENTO ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Iniciando o treinamento (pode levar alguns minutos)...")
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Salva o modelo final
model.save('modelo_animais_brasileiros.keras')
print("\n=> Treinamento concluído! Modelo salvo como 'modelo_animais_brasileiros.keras'.")