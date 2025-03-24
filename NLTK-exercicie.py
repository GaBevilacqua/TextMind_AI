import nltk


# nltk.download('punkt_tab')
# nltk.download('punkt')  # tokenização
# nltk.download('stopwords')  # stopwords
# nltk.download('wordnet')  # base de dados lexicais


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


#Stopwords em portugues
stopwords_pt = stopwords.words('portuguese')

lematizar = WordNetLemmatizer()

# Texto para tokenização
text = "Olha que coisa mais linda, mais cheia de graça É ela, menina, que vem e que passa Num doce balanço a caminho do mar"
word = "Garota"

# Tokenizar o texto
tokens = word_tokenize(text)

#Remover stopwords de uma lista de tokens
nostop = [palavra for palavra in tokens if palavra.lower() not in stopwords_pt]
lematizado = lematizar.lemmatize(word, pos='s')


# Exibir os tokens
#print(tokens)
#print(nostop)
print(lematizado)
