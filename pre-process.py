import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixar os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Função de pré-processamento
def preprocess_text(text):
    # Tokenizar o texto
    tokens = word_tokenize(text.lower())
    # Remover stopwords e palavras não alfanuméricas
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Pré-processar os conteúdos
conteudos_processados = [preprocess_text(text) for text in conteudos]

# Exibir os primeiros conteúdos processados
print(conteudos_processados[:2])
