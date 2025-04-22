import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
from nltk.corpus import stopwords
import os
import time
import nltk

# Configuração inicial do NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class LongExtractiveSummarizer:
    def __init__(self, language='portuguese'):
        """Inicializa o sumarizador com configurações para português"""
        print("Inicializando sumarizador para resumos longos...")
        self.language = language
        self.stop_words = stopwords.words(self.language)
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        print("Sumarizador pronto!")
    
    def _calculate_sentence_importance(self, sentences):
        """Calcula a importância de cada frase usando TF-IDF + PageRank"""
        if len(sentences) <= 1:
            return [1.0] * len(sentences)

        # Vetorização TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Matriz de similaridade
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalização
        row_sums = sim_matrix.sum(axis=1) + 1e-6
        norm_sim_matrix = sim_matrix / row_sums[:, np.newaxis]
        
        # PageRank simplificado
        damping = 0.85
        scores = np.ones(len(sentences)) / len(sentences)
        
        for _ in range(10):
            scores = damping * np.dot(norm_sim_matrix.T, scores) + (1 - damping) / len(sentences)
        
        return scores.tolist()
    
    def generate_long_summary(self, clean_text, min_sentences=4, max_sentences=8, max_chars=1000):
        """
        Gera resumos mais longos com controles de qualidade
        Parâmetros:
        - min_sentences: mínimo de frases no resumo
        - max_sentences: máximo de frases
        - max_chars: tamanho máximo em caracteres
        """
        sentences = sent_tokenize(clean_text, language=self.language)
        
        if len(sentences) <= min_sentences:
            return ' '.join(sentences)
        
        # Calcula importância
        scores = self._calculate_sentence_importance(sentences)
        ranked = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Seleciona frases até atingir os limites
        selected_indices = []
        total_chars = 0
        
        for score, idx, sentence in ranked:
            if len(selected_indices) >= max_sentences:
                break
            if total_chars + len(sentence) > max_chars and len(selected_indices) >= min_sentences:
                break
                
            selected_indices.append(idx)
            total_chars += len(sentence)
        
        # Ordena as frases selecionadas pela posição original
        summary = ' '.join([sentences[i] for i in sorted(selected_indices)])
        return self._post_process_summary(summary)
    
    def _post_process_summary(self, summary):
        """Ajustes finais no resumo"""
        sentences = sent_tokenize(summary, language=self.language)
        
        # Remove última frase se for muito curta
        if len(sentences) > 1 and len(sentences[-1].split()) < 5:
            return ' '.join(sentences[:-1])
        
        return summary

def process_dataset(input_path, output_path, sample_size=None):
    """Processa o dataset completo ou uma amostra"""
    try:
        print(f"\nCarregando dados de {input_path}...")
        df = pd.read_csv(input_path)
        
        # Verifica colunas necessárias
        required_cols = ['titulo', 'conteudo_limpo']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória '{col}' não encontrada")
        
        # Aplica filtro de amostra se especificado
        if sample_size:
            df = df.head(sample_size).copy()
            print(f"Processando amostra de {len(df)} registros")
        else:
            print(f"Processando dataset completo ({len(df)} registros)")
        
        # Inicializa sumarizador
        summarizer = LongExtractiveSummarizer()
        
        # Gera resumos longos
        print("\nGerando resumos longos...")
        df['resumo_extrativo'] = df['conteudo_limpo'].apply(
            lambda x: summarizer.generate_long_summary(x) if pd.notna(x) else "")
        
        # Salva resultados
        print(f"\nSalvando em {output_path}...")
        df.to_csv(output_path, index=False)
        print("Processo concluído com sucesso!")
        
    except Exception as e:
        print(f"\nErro no processamento: {str(e)}")

# Configuração de caminhos
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Historico_de_materias_processado.csv'))
output_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Extrativo_Amostra-Gen2.csv'))

# Execução principal
if __name__ == "__main__":
    # Para testar com 10 registros:
    # process_dataset(input_csv, output_csv, sample_size=10)
    
    # Para processar tudo:
    process_dataset(input_csv, output_csv)