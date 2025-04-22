import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
from nltk.corpus import stopwords
import os
import time

class OptimizedNewsSummarizer:
    def __init__(self, language='portuguese'):
        print("Inicializando sumarizador...")
        self.language = language
        self.stop_words = stopwords.words(self.language)
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        print("Sumarizador pronto!")
    
    def _calculate_sentence_importance(self, sentences):
        """Versão otimizada do cálculo de importância"""
        print(f"Processando {len(sentences)} frases...")
        start_time = time.time()
        
        if len(sentences) <= 1:
            return [1.0] * len(sentences)

        # Vetorização TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Matriz de similaridade otimizada
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalização vetorizada
        row_sums = sim_matrix.sum(axis=1) + 1e-6
        norm_sim_matrix = sim_matrix / row_sums[:, np.newaxis]
        
        # PageRank vetorizado
        damping = 0.85
        scores = np.ones(len(sentences)) / len(sentences)
        
        for iteration in range(10):  # Reduzido para 10 iterações
            scores = damping * np.dot(norm_sim_matrix.T, scores) + (1 - damping) / len(sentences)
            print(f"Iteração {iteration + 1}/10 concluída", end='\r')
        
        print(f"\nProcessamento concluído em {time.time() - start_time:.2f} segundos")
        return scores.tolist()
    
    def generate_summary(self, clean_text, num_sentences=3):
        """Gera resumo mantendo ordem original"""
        print("\nIniciando geração de resumo...")
        sentences = sent_tokenize(clean_text, language=self.language)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
            
        scores = self._calculate_sentence_importance(sentences)
        ranked = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        top_indices = sorted([i for (_, i, _) in ranked[:num_sentences]])
        summary = ' '.join([sentences[i] for i in top_indices])
        
        print("Resumo gerado com sucesso!")
        return summary

def process_sample(input_path, output_path, sample_size=10):
    """Processa apenas uma amostra do dataset"""
    try:
        print(f"\nCarregando dataset de {input_path}...")
        df = pd.read_csv(input_path)
        
        # Verifica colunas e pega amostra
        required_cols = ['titulo', 'conteudo_limpo', 'frases']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada")
        
        sample_df = df.head(sample_size).copy()
        print(f"Amostra de {len(sample_df)} notícias selecionada")
        
        summarizer = OptimizedNewsSummarizer()
        
        print("\nIniciando geração de resumos...")
        sample_df['resumo_extrativo'] = sample_df['conteudo_limpo'].apply(
            lambda x: summarizer.generate_summary(x) if pd.notna(x) else "")
        
        print(f"\nSalvando resultados em {output_path}...")
        sample_df.to_csv(output_path, index=False)
        print("Processo concluído com sucesso!")
        
        return sample_df
    
    except Exception as e:
        print(f"\nErro durante o processamento: {str(e)}")
        return None

# Execução principal
if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Historico_de_materias_processado.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Extrativo_Amostra-Gen2.csv')
    
    # Processa apenas 10 notícias
    result = process_sample(
        input_path=os.path.normpath(input_path),
        output_path=os.path.normpath(output_path),
        sample_size=10
    )
    
    if result is not None:
        print("\nPrimeiras linhas do resultado:")
        print(result[['titulo', 'resumo_extrativo']].head())