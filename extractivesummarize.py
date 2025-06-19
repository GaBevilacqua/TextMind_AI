import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class AdvancedSummarizer:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s.,;!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_importance(self, sentences):
        if len(sentences) <= 1:
            return [1.0] * len(sentences)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
        except ValueError:
            return [1.0] * len(sentences)
        
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalização
        row_sums = sim_matrix.sum(axis=1) + 1e-6
        norm_sim_matrix = sim_matrix / row_sums[:, np.newaxis]
        
        # PageRank 
        damping = 0.85
        scores = np.ones(len(sentences)) / len(sentences)
        
        for _ in range(10):
            scores = damping * np.dot(norm_sim_matrix.T, scores) + (1 - damping) / len(sentences)
        
        # Bônus por posição e entidades nomeadas
        position_bonus = np.linspace(1.3, 0.9, len(sentences))
        entity_bonus = np.array([1.2 if self.has_named_entities(s) else 1.0 for s in sentences])
        
        return (scores * position_bonus * entity_bonus).tolist()
    
    def has_named_entities(self, sentence):
        try:
            tagged = pos_tag(word_tokenize(sentence))
            return any(tag in ['NNP', 'NNPS', 'NN'] for _, tag in tagged)
        except:
            return False
    
    def post_process(self, summary):
        if not summary:
            return ""
            
        try:
            sentences = sent_tokenize(summary, language=self.language)
        except:
            sentences = summary.split('. ')
            sentences = [s + '.' for s in sentences if s and not s.endswith('.')]
            
        if len(sentences) <= 1:
            return summary
            
        processed = []
        for i, s in enumerate(sentences):
            if i == len(sentences)-1 or len(word_tokenize(s)) >= 5:
                processed.append(s)
        
        if processed and not processed[-1].rstrip().endswith(('.', '!', '?')):
            processed[-1] = processed[-1].strip() + '.'
            
        return ' '.join(processed).replace(' .', '.').strip()
    
    def summarize(self, text, min_sentences=3, max_sentences=5, max_chars=800):
        clean_text = self.clean_text(text)
        sentences = sent_tokenize(clean_text, language=self.language)
        
        if len(sentences) <= min_sentences:
            return ' '.join(sentences)
        
        scores = self.calculate_importance(sentences)
        
        selected_indices = []
        total_chars = 0
        
        # Garante pelo menos 1 das 2 primeiras frases
        lead_indices = sorted(range(min(2, len(sentences))), key=lambda i: -scores[i])
        for idx in lead_indices[:1]:
            selected_indices.append(idx)
            total_chars += len(sentences[idx])
        
        remaining_indices = [i for i in range(len(sentences)) if i not in selected_indices]
        ranked_indices = sorted(remaining_indices, key=lambda i: -scores[i])
        
        for idx in ranked_indices:
            if len(selected_indices) >= max_sentences:
                break
                
            if total_chars + len(sentences[idx]) > max_chars:
                if len(selected_indices) >= min_sentences:
                    break
                continue
                
            selected_indices.append(idx)
            total_chars += len(sentences[idx])
            
            if len(selected_indices) >= min_sentences and total_chars >= max_chars * 0.95:
                break
        
        if len(selected_indices) < min_sentences and len(sentences) > min_sentences:
            remaining = [i for i in range(len(sentences)) if i not in selected_indices]
            remaining_by_length = sorted(remaining, key=lambda i: len(sentences[i]))
            
            for idx in remaining_by_length:
                if len(selected_indices) >= min_sentences:
                    break
                if total_chars + len(sentences[idx]) <= max_chars:
                    selected_indices.append(idx)
                    total_chars += len(sentences[idx])
        
        selected_sentences = [sentences[i] for i in sorted(selected_indices)]
        return self.post_process(' '.join(selected_sentences))

def summarize_text(text, min_sentences=2, max_sentences=4, max_chars=512):
    summarizer = AdvancedSummarizer()
    return summarizer.summarize(text, min_sentences, max_sentences, max_chars)

# Exemplo de uso:
user_text = """

Three members of the same family who died in a static caravan from carbon monoxide poisoning would have been unconscious 'within minutes', investigators said today. The bodies of married couple John and Audrey Cook were discovered alongside their daughter, Maureen, at the mobile home they shared on Tremarle Home Park in Camborne, west Cornwall. The inquests have now opened into the deaths last Saturday, with investigators saying the three died along with the family's pet dog, of carbon monoxide poisoning from a cooker. Tragic: The inquests have opened into the deaths of three members of the same family who were found in their static caravan last weekend. John and Audrey Cook are pictured . Awful: The family died following carbon monoxide poisoning at this caravan at the Tremarle Home Park in Camborne, Cornwall . It is also believed there was no working carbon monoxide detector in the static caravan. Cornwall Fire and Rescue Service said this would have resulted in the three being unconscious 'within minutes', . A spokesman for Cornwall coroner Dr Emma Carlyon confirmed the inquests were opened and adjourned yesterday afternoon. They will resume at a later date. Devon and Cornwall Police confirmed on Monday that carbon monoxide poisoning had been established as the cause of death. A police spokesman said the source of the poisoning was 'believed to be from incorrect operation of the gas cooker'. Poisoning: This woman left flowers outside the caravan following the deaths. It has emerged that the trio would have been unconscious 'within minutes' Touching: This tribute was left outside the caravan following news of the deaths . Early readings from experts at the site revealed a potentially lethal level of carbon monoxide present within the caravan at the time it was taken, shortly after the discovery of the bodies. Friends and neighbours have paid tribute to the trio. One . neighbour, Sonya Owen, 53, said: 'It's very distressing. I knew the . daughter, she was living her with her mum and dad. Everybody is really . upset.' Margaret Holmes, 65, who lived near the couple and their . daughter, said: 'They had lived here for around 40 years and they kept . themselves to themselves. 'I just can’t believe this has . happened, it is so sad and I am so shocked, I think we all are, you just . don’t expect this sort of thing to happen on your doorstep. 'Everyone will miss them, we used to chat a lot when we were both in the garden. 'I would just like to send my condolences to their family, I can’t imagine what they’re going through.' Nic Clark, 52, who was good friends with daughter Maureen, added: 'They were a lovely kind family, a great trio. 'Maureen . used to go out and walk her dog, a little Jack Russell, it is so sad . what has happened, I understand the dog went with them. 'They . will be sorely missed and I think everyone is just in shock at the . moment, I would like to send my condolences to the Cook family.'

"""

print("Resumo:")
print(summarize_text(user_text))