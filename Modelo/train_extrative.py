#IMPORTS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import os
import time
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class ExtrativeSummarizer:
    #Idioma, stopwords, TF-IDF
    def __init__(self, language='portuguese'):
        print("Inicializando sumarizador extrativo...")
        self.language = language
        self.stop_words = set(stopwords.words(self.language))  
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        print("Inicio Pronto!")
    
    def CalculoImportancia(self, sentences):
        if len(sentences) <= 1:
            return [1.0] * len(sentences)

        # Vetorização TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:
            return [1.0] * len(sentences) 
        
        # Matriz de similaridade
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalização da matriz de similaridade
        row_sums = sim_matrix.sum(axis=1) + 1e-6
        norm_sim_matrix = sim_matrix / row_sums[:, np.newaxis]
        
        # PageRank 
        damping = 0.85
        scores = np.ones(len(sentences)) / len(sentences)
        
        for _ in range(10):
            scores = damping * np.dot(norm_sim_matrix.T, scores) + (1 - damping) / len(sentences)
        
        # Heuristica hibrida (posição + features)
        position_bonus = np.linspace(1.3, 0.9, len(sentences))  # Decaimento gradual
        entity_bonus = np.array([1.2 if self.Nomes(s) else 1.0 for s in sentences])
        
        return (scores * position_bonus * entity_bonus).tolist()
    
    #Substantivos próprios e comuns => bonus
    def Nomes(self, sentence):
        try:
            tagged = pos_tag(word_tokenize(sentence))
            return any(tag in ['NNP', 'NN'] for _, tag in tagged)
        except:
            return False
    
    #Gerar resumo com numero minimo e máximo de frases, bem como os caracteres
    def Sumarizador(self, clean_text, min_sentences=5, max_sentences=7, max_chars=1200):
        #String válida
        if not clean_text or not isinstance(clean_text, str):
            return ""     
        #Quebra em frases       
        try:
            sentences = sent_tokenize(clean_text, language=self.language)
        except:
            sentences = clean_text.split('. ') 
            
        if len(sentences) <= min_sentences:
            return ' '.join(sentences)
        
        # Calcula importância com bônus
        scores = self.CalculoImportancia(sentences)
        
        # Seleção adaptativa
        selected_indices = []
        total_chars = 0
        
        # Garante pelo menos 1 das 2 primeiras frases
        lead_indices = sorted(range(min(2, len(sentences))), key=lambda i: -scores[i])
        #Se couber no limite de frase e caracteres, adiciona a frase
        for idx in lead_indices:
            if (len(selected_indices) < max_sentences and 
                total_chars + len(sentences[idx]) <= max_chars):
                selected_indices.append(idx)
                total_chars += len(sentences[idx])
        
        # Completa com outras frases importantes com base no text rank
        other_indices = [i for i in range(len(sentences)) if i not in lead_indices]
        ranked_others = sorted(other_indices, key=lambda i: -scores[i])
        
        for idx in ranked_others:
            if len(selected_indices) >= max_sentences:
                break
            if (total_chars + len(sentences[idx]) > max_chars and 
                len(selected_indices) >= min_sentences):
                continue
            if idx not in selected_indices:
                selected_indices.append(idx)
                total_chars += len(sentences[idx])
        
        # Ordena e pós-processa
        selected_sentences = [sentences[i] for i in sorted(selected_indices)]
        return self.PosFase(' '.join(selected_sentences))
    
    #Frases curtas e pontuação
    def PosFase(self, summary):
        if not summary:
            return ""
            
        try:
            sentences = sent_tokenize(summary, language=self.language)
        except:
            sentences = summary.split('. ')
            
        if len(sentences) <= 1:
            return summary
            
        # Remove frases muito curtas (exceto se for a única ou última)
        processed = []
        for i, s in enumerate(sentences):
            if i == len(sentences)-1 or len(word_tokenize(s)) >= 5:
                processed.append(s)
        
        # Pontuação
        if processed and processed[-1][-1] not in ['.', '!', '?']:
            processed[-1] = processed[-1].strip() + '.'
            
        return ' '.join(processed).replace(' .', '.').strip()

#Database
def Pro_Database(input_path, output_path, sample_size=None): 
    try:
        #Carrega CSV
        print(f"\nCarregando dados de {input_path}...")
        df = pd.read_csv(input_path, nrows=sample_size if sample_size else None)
        
        #Pŕe requesitos
        req_cols = ['titulo', 'conteudo_limpo']
        miss_cols = [col for col in req_cols if col not in df.columns]
        if miss_cols:
            raise ValueError(f"Colunas obrigatórias não encontradas: {miss_cols}")
        
        print(f"Processando {len(df)} registros...")
        
        #Cria sumarizador
        sumarizador = ExtrativeSummarizer()
        
        #aplicando resumo
        print("\nGerando resumos com abordagem extrativa...")
        df['resumo_extrativo'] = df['conteudo_limpo'].apply(
            lambda x: sumarizador.Sumarizador(str(x)) if pd.notna(x) else "")
        
        #Salvar
        print(f"\nSalvando em {output_path}...")
        df.to_csv(output_path, index=False)
        print("Processo concluído com sucesso!")
        
        # Exemplo de resumo gerado
        if not df.empty:
            print("\nExemplo de resumo:")
            sample_summary = df.iloc[0]['resumo_extrativo']
            print(sample_summary[:500] + ("..." if len(sample_summary) > 500 else ""))
        
    except Exception as e:
        print(f"\nErro no processamento: {str(e)}")
        raise

# Configuração de caminhos
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Historico_de_materias_processado.csv'))
    output_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Extrativo_AmostraGen4.csv'))

    # Teste com 10
    try:
        Pro_Database(input_csv, output_csv, sample_size=10)
    except Exception as e:
        print(f"Erro durante a execução: {e}")