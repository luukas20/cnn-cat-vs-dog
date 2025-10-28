import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Parâmetros
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset'

model = tf.keras.models.load_model(r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\models\model_cnn.keras')

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/validation",
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode='rgb'
)

# Nomes das classes na ordem correta (geralmente alfabética)
CLASS_NAMES = ['Cat', 'Dog'] # Ou ['caes', 'gatos']

# --- 2. FAZER PREVISÕES E EXTRAIR RÓTULOS VERDADEIROS ---
print("Fazendo previsões em todo o conjunto de validação...")
# model.predict itera por todos os lotes do dataset e retorna um único array
predictions = model.predict(validation_dataset)

# Extrai os rótulos verdadeiros do dataset
# Itera pelo dataset, pega os rótulos (y) de cada lote e os concatena
true_labels = np.concatenate([y for x, y in validation_dataset], axis=0)

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

# Pega o primeiro lote de imagens e rótulos para visualização
for images, labels in validation_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(20): # Exibe as 9 primeiras imagens do lote
        ax = plt.subplot(5, 4, i + 1)
        
        # Exibe a imagem
        plt.imshow(images[i].numpy().astype("uint8"))
        
        # Pega o rótulo verdadeiro e o previsto para esta imagem
        true_label_index = int(labels[i].numpy())
        predicted_label_index = predicted_labels[i][0] # O índice corresponde à ordem
        
        true_class_name = CLASS_NAMES[true_label_index]
        predicted_class_name = CLASS_NAMES[predicted_label_index]
        
        # Define o título e a cor (verde se acertou, vermelho se errou)
        title = f"Real: {true_class_name}\nPrevisto: {predicted_class_name}"
        color = "green" if true_label_index == predicted_label_index else "red"
        
        plt.title(title, color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.show()