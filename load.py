import os

# Caminho para a pasta 'raw_data' (ajuste conforme o local do seu dataset)
pasta_raw_data = 'Dataset'

# Listas para armazenar os títulos e conteúdos
titulos = []
conteudos = []

# Percorrer todas as subpastas e arquivos .txt dentro da pasta 'raw_data'
for raiz, _, arquivos in os.walk(pasta_raw_data):
    for arquivo in arquivos:
        if arquivo.endswith('.txt'):
            caminho_arquivo = os.path.join(raiz, arquivo)
            
            # Abrir o arquivo e ler seu conteúdo
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                # O título está na primeira linha e o conteúdo no restante do arquivo
                linhas = f.readlines()
                if len(linhas) > 1:
                    titulos.append(linhas[0].strip())  # Remover quebras de linha do título
                    conteudos.append(' '.join(linhas[1:]).strip())  # Concatenar o conteúdo

# Exibir os primeiros títulos e conteúdos
print(titulos[:2])
print(conteudos[:2])
