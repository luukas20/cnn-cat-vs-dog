import glob
import os
import shutil
import random
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras.utils as image
from tensorflow.keras import layers

# Parâmetros
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset'

# Carregar os datasets de treino e validação
train_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    labels='inferred',
    label_mode='binary', # 'binary' para 2 classes. 0 para uma, 1 para outra.
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode='rgb',
    validation_split=0.5, # <-- Diz para reservar 50% dos dados
    subset='training',    # <-- Diz para usar apenas a parte de 'treino' (os primeiros 50%)
    seed=123              # <-- Importante para garantir que a divisão seja sempre a mesma
)

# Carregar os datasets de teste e validação
test_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    labels='inferred',
    label_mode='binary', # 'binary' para 2 classes. 0 para uma, 1 para outra.
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode='rgb',
    validation_split=0.5, # <-- Diz para reservar 50% dos dados
    subset='training',    # <-- Diz para usar apenas a parte de 'treino' (os primeiros 50%)
    seed=123              # <-- Importante para garantir que a divisão seja sempre a mesma
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/validation",
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode='rgb'
)

# Ver as classes que foram encontradas
class_names = train_dataset.class_names
print(f"Classes encontradas: {class_names}")

# Otimizar o pipeline de dados para performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Construindo o modelo...")
model = keras.Sequential([
    # Input Layer e Data Augmentation
    layers.Input(shape=(224, 224, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # Normalização dos pixels para a faixa [0, 1]
    layers.Rescaling(1./255),

    # Bloco Convolucional 1
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Bloco Convolucional 2
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Bloco Convolucional 3
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Achatamento (Flatten) para preparar para o classificador
    layers.Flatten(),

    # Camadas Densas (Classificador)
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.5), # Adicionando Dropout para regularização
    # Camada de Saída: 1 neurônio com ativação sigmoide para classificação binária
    layers.Dense(units=1, activation='sigmoid')
])

# Visualizar a arquitetura do modelo
model.summary()

print("Compilando o modelo...")
model.compile(
    optimizer='adam', # Otimizador Adam é uma excelente escolha padrão
    loss=tf.keras.losses.BinaryCrossentropy(), # Loss para classificação binária com saída sigmoide
    metrics=['accuracy'] # Métrica para acompanhar durante o treino
)

print("Iniciando o treinamento do modelo...")
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=test_dataset
)

print("Salvando o modelo treinado...")
model.save(r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\models\model_cnn.keras')