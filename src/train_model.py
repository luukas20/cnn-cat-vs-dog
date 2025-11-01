import pandas as pd
import kagglehub
import glob
import os
import shutil
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.utils as image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

# Parâmetros
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset'

dataset_path = kagglehub.dataset_download('bhavikjikadara/dog-and-cat-classification-dataset')
print('Data Source import completo.')

directory = os.path.join(dataset_path, 'PetImages')

images = []
labels = []

try:
  for foldr in os.listdir(directory):
    for filee in os.listdir(os.path.join(directory, foldr)):
      images.append(os.path.join(foldr, filee))
      labels.append(foldr)
        
except Exception as e:
  print(f'Error: {e}')

all_df = pd.DataFrame({
    'Images': images,
    'Labels': labels
    })

print(all_df)

train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42, stratify=all_df['Labels'])

print("Preparando o dataset de treino...")
trainimgen = ImageDataGenerator( 
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
    )

train_data = trainimgen.flow_from_dataframe(
    dataframe=train_df,
    directory=directory,
    x_col='Images',
    y_col='Labels',
    target_size=(224,224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
)

print("Preparando o dataset de teste...")
testimgen = ImageDataGenerator()

test_data = testimgen.flow_from_dataframe(
    dataframe=test_df,
    directory=directory,
    x_col='Images',
    y_col='Labels',
    target_size=(224,224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=False
)

print("Construindo o modelo...")
model = tf.keras.Sequential([
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
    train_data,
    epochs=15,
    validation_data=test_data
)

print("Salvando o modelo treinado...")
model.save(r'C:\model_cnn.keras')