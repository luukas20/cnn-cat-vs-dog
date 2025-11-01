import tensorflow as tf
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parâmetros
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset'

model = tf.keras.models.load_model(r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\models\model_cnn.keras')

directory = os.path.join(DATA_DIR, 'validation')

images = []
labels = []

try:
  for foldr in os.listdir(directory):
    for filee in os.listdir(os.path.join(directory, foldr)):
      images.append(os.path.join(foldr, filee))
      labels.append(foldr)
        
except Exception as e:
  print(f'Error: {e}')

val_df = pd.DataFrame({
    'Images': images,
    'Labels': labels
    })

val_df = val_df.sample(frac=1, ignore_index=True)

print(val_df)

print("Preparando o dataset de validacao...")
valimgen = ImageDataGenerator( )

validation_dataset = valimgen.flow_from_dataframe(
    dataframe=val_df,
    directory=directory,
    x_col='Images',
    y_col='Labels',
    target_size=(224,224),
    color_mode='rgb',
    class_mode='binary',
    batch_size=16,
    shuffle=False,
)

# Nomes das classes na ordem correta (geralmente alfabética)
CLASS_NAMES = ['Cat', 'Dog'] # Ou ['caes', 'gatos']

# --- 2. FAZER PREVISÕES E EXTRAIR RÓTULOS VERDADEIROS ---
print("Fazendo previsões em todo o conjunto de validação...")
# model.predict itera por todos os lotes do dataset e retorna um único array
predictions = model.predict(validation_dataset)

# Extrai os rótulos verdadeiros do dataset
# Itera pelo dataset, pega os rótulos (y) de cada lote e os concatena
true_labels = validation_dataset.labels

# Converte as previsões (probabilidades) em rótulos de classe (0 ou 1)
# Se a previsão > 0.5, a classe é 1 (Dog), senão é 0 (Cat)
predicted_labels = (predictions > 0.5).astype(int)

# --- 3. AVALIAÇÃO QUANTITATIVA ---
print("\n--- Relatório de Classificação ---")
# Mostra precisão, recall, f1-score para cada classe
print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES))

print("\n--- Matriz de Confusão ---")
# Mostra quantos foram classificados corretamente e incorretamente
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)
print("\n     Previsto: Cat  | Previsto: Dog")
print(f"Real: Cat |   {cm[0][0]:<5}    |   {cm[0][1]:<5}")
print(f"Real: Dog |   {cm[1][0]:<5}    |   {cm[1][1]:<5}")
print("--------------------------------")


# --- 4. VISUALIZAÇÃO DE EXEMPLOS ---
print("\nExibindo alguns exemplos de previsões...")

# MODIFICAÇÃO 1:
# Em vez de 'for ... in validation_dataset.take(1):'
# Nós usamos 'next()' para pegar o primeiro lote ANTES do loop.
try:
    images, labels = next(validation_dataset)
except StopIteration:
    print("Erro: O gerador de validação está esgotado. Reinicie o script.")
    # Isso pode acontecer se você já iterou por ele (ex: no model.predict)
    # e precisa recriá-lo. Mas vamos tentar primeiro.

plt.figure(figsize=(10, 10))

# O loop 'for i in range(15)' agora está correto
for i in range(15): 
    # Garante que não vamos estourar o índice do lote
    if i >= len(images):
        break 
        
    ax = plt.subplot(5, 4, i + 1)
    
    # MODIFICAÇÃO 2: Removido o .numpy()
    # 'images' já é um array NumPy, então .numpy() não existe e não é necessário.
    # O Keras (antigo) retorna imagens como float, então convertemos para int.
    plt.imshow(images[i].astype("uint8"))
    
    # Pega o rótulo verdadeiro e o previsto para esta imagem
    
    # MODIFICAÇÃO 3: Removido o .numpy()
    # 'labels' também já é um array NumPy
    true_label_index = int(labels[i]) 
    
    # Esta linha está correta, assumingo que 'predicted_labels' foi
    # calculado antes e que shuffle=False foi usado.
    predicted_label_index = predicted_labels[i][0] 
    
    true_class_name = CLASS_NAMES[true_label_index]
    predicted_class_name = CLASS_NAMES[predicted_label_index]
    
    # Define o título e a cor (verde se acertou, vermelho se errou)
    title = f"Real: {true_class_name}\nPrevisto: {predicted_class_name}"
    color = "green" if true_label_index == predicted_label_index else "red"
    
    plt.title(title, color=color)
    plt.axis("off")

plt.tight_layout()
plt.show()