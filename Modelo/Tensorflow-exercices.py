import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# Baixe os dados necessários do NLTK
nltk.download('punkt')

# Dados de exemplo (textos longos e seus resumos)
textos = [
    "O TensorFlow é uma plataforma de código aberto para machine learning. Ele suporta diversos tipos de modelos e é amplamente utilizado na indústria.",
    "Redes neurais são sistemas de aprendizado inspirados no cérebro humano que podem aprender padrões complexos a partir de dados.",
    "O processamento de linguagem natural é uma área da inteligência artificial que foca na interação entre computadores e linguagem humana."
]

resumos = [
    "TensorFlow: plataforma open-source para ML.",
    "Redes neurais aprendem padrões complexos.",
    "NLP foca na interação computador-linguagem."
]

# Tokenização e preparação dos dados
tokenizer_texto = Tokenizer()
tokenizer_texto.fit_on_texts(textos)
seq_textos = tokenizer_texto.texts_to_sequences(textos)

# Adicionar tokens especiais ao tokenizer do resumo
resumos = ["<inicio> " + r + " <fim>" for r in resumos]
tokenizer_resumo = Tokenizer(filters='')
tokenizer_resumo.fit_on_texts(resumos)
seq_resumos = tokenizer_resumo.texts_to_sequences(resumos)

# Padding para sequências de mesmo tamanho
max_len_texto = max(len(seq) for seq in seq_textos)
max_len_resumo = max(len(seq) for seq in seq_resumos)

X = pad_sequences(seq_textos, maxlen=max_len_texto, padding='post')
Y = pad_sequences(seq_resumos, maxlen=max_len_resumo, padding='post')

# Parâmetros do modelo
vocab_texto = len(tokenizer_texto.word_index) + 1
vocab_resumo = len(tokenizer_resumo.word_index) + 1
embedding_dim = 256
latent_dim = 512

# Modelo encoder
encoder_inputs = Input(shape=(max_len_texto,))
enc_emb = Embedding(vocab_texto, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Modelo decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_resumo, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_resumo, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo completo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Preparar dados para treinamento
encoder_input_data = X
decoder_input_data = Y[:, :-1]
decoder_target_data = Y[:, 1:]

# Treinamento (simplificado)
model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data,
          batch_size=2, 
          epochs=50)

# Modelo de inferência (para geração de resumos)
# Encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    dec_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Função para inferência (gerar resumos)
def resumir(texto):
    # Tokenizar e padronizar o texto de entrada
    seq = tokenizer_texto.texts_to_sequences([texto])
    seq = pad_sequences(seq, maxlen=max_len_texto, padding='post')
    
    # Obter estados do encoder
    states_value = encoder_model.predict(seq)
    
    # Gerar resumo token por token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_resumo.word_index['<inicio>']
    
    stop_condition = False
    resumo = []
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        
        # Pegar o token com maior probabilidade
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_resumo.index_word.get(sampled_token_index, '')
        
        if sampled_word != '<fim>' and len(resumo) < max_len_resumo:
            resumo.append(sampled_word)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        else:
            stop_condition = True
    
    return ' '.join(resumo)

# Teste
texto_teste = "O TensorFlow permite criar modelos de deep learning de forma eficiente."
print(resumir(texto_teste))