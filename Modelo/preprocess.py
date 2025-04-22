import pandas as pd
import re
import nltk
import os
from nltk.tokenize import sent_tokenize

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Normaliza espaços
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    if not text or not isinstance(text, str):
        return []
    return sent_tokenize(text, language='portuguese')

# Carregue os dados
try:
    caminho_csv = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Historico_de_materias.csv')
    caminho_csv = os.path.normpath(caminho_csv)

    df = pd.read_csv(caminho_csv)
    
    required_columns = ['conteudo_noticia']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")

    # Pré-processamento seguro
    df['conteudo_noticia'] = df['conteudo_noticia'].fillna('').astype(str)
    df['conteudo_limpo'] = df['conteudo_noticia'].apply(clean_text)
    df['frases'] = df['conteudo_limpo'].apply(tokenize_sentences)
    
    # Processar título
    if 'titulo' in df.columns:
        df['titulo_limpo'] = df['titulo'].fillna('').astype(str).apply(clean_text)

    # Salvar    
    caminho2_csv = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Historico_de_materias_processado.csv')
    caminho2_csv = os.path.normpath(caminho2_csv)
    df.to_csv(caminho2_csv, index=False)
    print("Pré-processamento concluído com sucesso!")

except Exception as e:
    print(f"Erro durante o processamento: {str(e)}")