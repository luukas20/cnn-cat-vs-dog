import os
import random
import shutil

# source_class_path = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset\PetImages\Dog' 
# imagens = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
# print(len(imagens))


def dividir_dataset(source_dir, output_dir, split_counts):
    """
    Divide um diretório de imagens classificado por pastas em conjuntos
    de treino, validação e teste.

    :param source_dir: Caminho para o diretório com as pastas das classes (ex: 'imagens_originais').
    :param output_dir: Caminho para o diretório de saída onde os splits serão criados (ex: 'dataset').
    :param split_ratios: Uma tupla com as proporções para (treino, validação, teste).
    """
    
    # Validação das proporções
    # if sum(split_ratios) != 1.0:
    #     raise ValueError("A soma das proporções de divisão deve ser 1.0")

    # train_ratio, val_ratio, test_ratio = split_ratios
    train_ratio, val_ratio, test_ratio = split_counts

    # Cria o diretório de saída e os subdiretórios se não existirem
    print(f"Criando diretórios de saída em: {output_dir}")
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Pega os nomes das classes (nomes das pastas no diretório fonte)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not classes:
        print("Nenhuma pasta de classe encontrada no diretório de origem.")
        return

    print(f"Classes encontradas: {classes}")

    # Processa cada classe individualmente
    for class_name in classes:
        print(f"\nProcessando a classe: {class_name}")

        # Cria os subdiretórios das classes nos splits de destino
        path_treino = os.path.join(output_dir, 'train', class_name)
        path_validacao = os.path.join(output_dir, 'validation', class_name)
        path_teste = os.path.join(output_dir, 'test', class_name)
        os.makedirs(path_treino, exist_ok=True)
        os.makedirs(path_validacao, exist_ok=True)
        os.makedirs(path_teste, exist_ok=True)

        # Caminho da pasta original da classe
        source_class_path = os.path.join(source_dir, class_name)
        
        # Lista todas as imagens na pasta da classe
        imagens = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        
        # Embaralha a lista de imagens para garantir uma divisão aleatória
        random.shuffle(imagens)
        
        total_imagens = len(imagens)
        print(f"Total de imagens encontradas: {total_imagens}")
        
        # Define os pontos de corte com base nas contagens fixas
        # ponto_corte_treino = train_count
        # ponto_corte_validacao = train_count + val_count
        # ponto_corte_teste = train_count + val_count + test_count

        # Calcula os pontos de corte para a divisão
        ponto_corte_treino = int(train_ratio * total_imagens)
        ponto_corte_validacao = int((train_ratio + val_ratio) * total_imagens)
        
        # Divide a lista de arquivos
        arquivos_treino = imagens[:train_ratio]
        arquivos_validacao = imagens[train_ratio:(train_ratio + val_ratio)]
        arquivos_teste = imagens[(train_ratio + val_ratio):(train_ratio + val_ratio + test_ratio)]
        
        # Função auxiliar para copiar os arquivos
        def copiar_arquivos(lista_arquivos, destino, nome_classe):
            for arquivo in lista_arquivos:
                caminho_origem = os.path.join(source_class_path, arquivo)
                caminho_destino = os.path.join(destino, arquivo)

                # Cria o caminho completo de destino com o novo nome
                # novo_nome_arquivo = f"{nome_classe.lower()}{arquivo}"
                # destino_com_novo_nome = os.path.join(destino, novo_nome_arquivo)
                
                shutil.copy(caminho_origem, caminho_destino)
        
        # Copia os arquivos para seus respectivos diretórios
        print(f"Copiando {len(arquivos_treino)} imagens para o conjunto de treino...")
        copiar_arquivos(arquivos_treino, path_treino, class_name)
        
        print(f"Copiando {len(arquivos_validacao)} imagens para o conjunto de validação...")
        copiar_arquivos(arquivos_validacao, path_validacao, class_name)
        
        print(f"Copiando {len(arquivos_teste)} imagens para o conjunto de teste...")
        copiar_arquivos(arquivos_teste, path_teste, class_name)

    print("\nProcesso de divisão do dataset concluído com sucesso!")

# --- CONFIGURAÇÃO ---
if __name__ == '__main__':
    # Defina o caminho para a pasta que contém suas pastas 'gatos' e 'caes'
    SOURCE_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset\PetImages' 
    
    # Defina o caminho para a nova pasta onde o dataset será criado
    OUTPUT_DIR = r'C:\Users\lucas\OneDrive - Amelyer Company\Documentos\Projetos Python\Dogs vs Cats\dataset'
    
    # Defina as proporções (treino, validação, teste)
    # 70% para treino, 15% para validação, 15% para teste
    # SPLIT_RATIOS = (0.7, 0.15, 0.15)
    SPLIT_COUNTS = (2500, 500, 500)
    
    # Supondo que suas pastas originais sejam 'caes' e 'gatos'
    # Crie a pasta 'imagens_originais' e mova 'caes' e 'gatos' para dentro dela
    # Exemplo:
    # /seu_projeto
    #     /imagens_originais
    #         /caes
    #             - img1.jpg
    #             - img2.jpg
    #         /gatos
    #             - imgA.jpg
    #             - imgB.jpg
    #     - preparar_dataset.py

    dividir_dataset(SOURCE_DIR, OUTPUT_DIR, SPLIT_COUNTS)