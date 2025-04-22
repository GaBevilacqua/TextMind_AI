import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import os
import time
import nltk

# Configuração inicial do NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class HybridSummarizer:
    def __init__(self, language='portuguese'):
        """Inicializa o sumarizador com configurações para português"""
        print("Inicializando sumarizador híbrido...")
        self.language = language
        self.stop_words = set(stopwords.words(self.language))  # Convertendo para set para performance
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        print("Sumarizador pronto!")
    
    def _calculate_sentence_importance(self, sentences):
        """Calcula importância com bônus posicional e features adicionais"""
        if len(sentences) <= 1:
            return [1.0] * len(sentences)

        # Vetorização TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:
            return [1.0] * len(sentences)  # Retorna importância uniforme se houver erro na vetorização
        
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
        
        # Bônus híbrido (posição + features)
        position_bonus = np.linspace(1.3, 0.9, len(sentences))  # Decaimento gradual
        entity_bonus = np.array([1.2 if self._has_entities(s) else 1.0 for s in sentences])
        
        return (scores * position_bonus * entity_bonus).tolist()
    
    def _has_entities(self, sentence):
        """Verifica se a frase contém entidades (nomes próprios, substantivos)"""
        try:
            tagged = pos_tag(word_tokenize(sentence))
            return any(tag in ['NNP', 'NN'] for _, tag in tagged)
        except:
            return False
    
    def generate_summary(self, clean_text, min_sentences=3, max_sentences=4, max_chars=800):
        """Gera resumo com estratégia híbrida"""
        if not clean_text or not isinstance(clean_text, str):
            return ""
            
        try:
            sentences = sent_tokenize(clean_text, language=self.language)
        except:
            sentences = clean_text.split('. ')  # Fallback básico
            
        if len(sentences) <= min_sentences:
            return ' '.join(sentences)
        
        # Calcula importância com bônus
        scores = self._calculate_sentence_importance(sentences)
        
        # Seleção adaptativa
        selected_indices = []
        total_chars = 0
        
        # Garante pelo menos 1 das 2 primeiras frases
        lead_indices = sorted(range(min(2, len(sentences))), key=lambda i: -scores[i])
        for idx in lead_indices:
            if (len(selected_indices) < max_sentences and 
                total_chars + len(sentences[idx]) <= max_chars):
                selected_indices.append(idx)
                total_chars += len(sentences[idx])
        
        # Completa com outras frases importantes
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
        return self._post_process_summary(' '.join(selected_sentences))
    
    def _post_process_summary(self, summary):
        """Pós-processamento inteligente"""
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
        
        # Garante que termina com pontuação
        if processed and processed[-1][-1] not in ['.', '!', '?']:
            processed[-1] = processed[-1].strip() + '.'
            
        return ' '.join(processed).replace(' .', '.').strip()

def process_dataset(input_path, output_path, sample_size=None):
    """Processa o dataset com a nova abordagem híbrida"""
    try:
        print(f"\nCarregando dados de {input_path}...")
        df = pd.read_csv(input_path, nrows=sample_size if sample_size else None)
        
        required_cols = ['titulo', 'conteudo_limpo']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias não encontradas: {missing_cols}")
        
        print(f"Processando {len(df)} registros...")
        
        summarizer = HybridSummarizer()
        
        print("\nGerando resumos com abordagem híbrida...")
        df['resumo_hibrido'] = df['conteudo_limpo'].apply(
            lambda x: summarizer.generate_summary(str(x)) if pd.notna(x) else "")
        
        print(f"\nSalvando em {output_path}...")
        df.to_csv(output_path, index=False)
        print("Processo concluído com sucesso!")
        
        # Exemplo de resumo gerado
        if not df.empty:
            print("\nExemplo de resumo:")
            sample_summary = df.iloc[0]['resumo_hibrido']
            print(sample_summary[:500] + ("..." if len(sample_summary) > 500 else ""))
        
    except Exception as e:
        print(f"\nErro no processamento: {str(e)}")
        raise

# Configuração de caminhos
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Historico_de_materias_processado.csv'))
    output_csv = os.path.normpath(os.path.join(current_dir, '..', 'Dataset', 'Extrativo_Hibrido.csv'))

    # Teste com 10 registros primeiro
    try:
        process_dataset(input_csv, output_csv, sample_size=10)
        
        # Para processar completo (descomente):
        # process_dataset(input_csv, output_csv)
    except Exception as e:
        print(f"Erro durante a execução: {e}")