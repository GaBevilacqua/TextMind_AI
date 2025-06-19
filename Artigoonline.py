from newspaper import Article

def extrair_texto_noticia(url):
    try:
        artigo = Article(url, language='en')
        artigo.download()
        artigo.parse()
        return artigo.text
    except Exception as e:
        return f"Erro: {e}"

# Exemplo de uso
from newspaper import Article; url=input("Cole o link da notícia: "); a=Article(url, language='en'); a.download(); a.parse(); print("\n=== Texto extraído ===\n", a.text)
texto = extrair_texto_noticia(url)
print("\n=== Texto extraído ===\n")
print(texto)