import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from newspaper import Article
import sys
import os
import threading
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configurações iniciais para o modelo T5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
torch.set_num_threads(4)  

# Baixar recursos do NLTK
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    NLTK_RESOURCES_AVAILABLE = True
except:
    NLTK_RESOURCES_AVAILABLE = False

# Tenta carregar o modelo T5
try:
    model_path = "./abstract_summary_v1"  
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    def summarize(text, max_length=150):
        input_text = "summarize: " + text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    ABSTRACT_SUMMARY_AVAILABLE = True
except Exception as e:
    print(f"Aviso: Não foi possível carregar o modelo T5: {e}")
    ABSTRACT_SUMMARY_AVAILABLE = False

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
    if not NLTK_RESOURCES_AVAILABLE:
        return "Erro: Recursos do NLTK não disponíveis. Instale os pacotes necessários."
    
    summarizer = AdvancedSummarizer(language='portuguese' if any(ord(c) > 127 for c in text) else 'english')
    return summarizer.summarize(text, min_sentences, max_sentences, max_chars)

class RoundedButton(tk.Canvas):
    def __init__(self, master=None, text="", radius=25, btnforeground="#000000", btnbackground="#ffffff", 
                 clicked=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.config(bg='#1e1e1e', highlightthickness=0, bd=0)  # Removendo a borda
        self.btnbackground = btnbackground
        self.btnforeground = btnforeground
        self.clicked = clicked
        self.radius = radius
        
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.text = text
        self.draw_button()
        
    def draw_button(self, color=None):
        self.delete("all")
        color = color if color else self.btnbackground
        
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        
        self.create_round_rect(0, 0, width, height, self.radius, fill=color, outline=color)
        self.create_text(width//2, height//2, text=self.text, fill=self.btnforeground, 
                        font=('Tekton Pro', 10, 'bold'))
    
    def create_round_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def on_press(self, event=None):
        self.draw_button(color='#004d40')
    
    def on_release(self, event=None):
        self.draw_button()
        if self.clicked:
            self.clicked()

class DarkNewsSummarizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Mind AI")
        self.root.geometry("850x700")
        self.root.configure(bg='#121212')
        
        # Variável para controlar se o resumo abstrato está processando
        self.is_processing_abstract = False
        
        self.setup_styles()
        main_frame = ttk.Frame(
            root,
            style='Dark.TFrame',
            borderwidth=8,
            relief='groove',
            padding=(15, 15)
        )
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title = ttk.Label(
            main_frame,
            text="Text Mind AI",
            style='Title.TLabel'
        )
        title.pack(pady=(0, 15))
        
        input_frame = ttk.Frame(main_frame, style='Dark.TFrame', height=30)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Frame para o label e o botão de extrair
        label_frame = ttk.Frame(input_frame, style='Dark.TFrame')
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(
            label_frame,
            text="COLE SUA NOTÍCIA OU LINK:",
            style='Subtitle.TLabel'
        ).pack(side=tk.LEFT, anchor='w')
        
        # Botão para extrair artigo do link
        self.extract_btn = RoundedButton(
            label_frame,
            text="EXTRAIR ARTIGO",
            radius=15,
            btnbackground='#FF6F00',
            btnforeground='white',
            clicked=self.extract_article_from_link,
            width=120,
            height=30
        )
        self.extract_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=85,
            height=13,
            font=('Tekton Pro', 10),
            bg='#252525',
            fg='#e0e0e0',
            insertbackground='white',
            selectbackground='#454545',
            relief='groove',
            borderwidth=2
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        button_frame.pack(fill=tk.X, pady=(5, 15))
        
        btn_container = ttk.Frame(button_frame, style='Dark.TFrame')
        btn_container.pack(expand=True)
        
        # Botão Esquerdo - Resumo Extrativo
        summarize_btn_color = '#00695c' if NLTK_RESOURCES_AVAILABLE else '#757575'
        self.summarize_btn = RoundedButton(
            btn_container,
            text="RESUMO EXTRATIVO",
            radius=20,
            btnbackground=summarize_btn_color,
            btnforeground='white',
            clicked=self.generate_summary,
            width=180,
            height=40
        )
        self.summarize_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)
        
        # Botão Central - Resumo Abstrato (usando o modelo T5)
        abstract_btn_color = '#5D4037' if ABSTRACT_SUMMARY_AVAILABLE else '#757575'
        self.analyze_btn = RoundedButton(
            btn_container,
            text="RESUMO ABSTRATO",
            radius=20,
            btnbackground=abstract_btn_color,
            btnforeground='white',
            clicked=self.generate_abstract_summary,
            width=160,
            height=40
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=10, ipadx=5, ipady=5)
        
        # Botão Direito
        self.knowledge_btn = RoundedButton(
            btn_container,
            text="GERAR ÁRVORE",
            radius=20,
            btnbackground='#00695c',
            btnforeground='white',
            clicked=self.generate_knowledge_tree,
            width=180,
            height=40
        )
        self.knowledge_btn.pack(side=tk.LEFT, ipadx=5, ipady=5)
        
        output_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            output_frame,
            text="RESUMO GERADO:",
            style='Subtitle.TLabel'
        ).pack(anchor='w')
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=85,
            height=10,
            font=('Tekton Pro', 10),
            bg='#252525',
            fg='#e0e0e0',
            state='disabled',
            relief='groove',
            borderwidth=2
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Mostra status dos modelos
        model_status = []
        if ABSTRACT_SUMMARY_AVAILABLE:
            model_status.append("T5 Integrado")
        else:
            model_status.append("T5 Não Encontrado")
            
        if NLTK_RESOURCES_AVAILABLE:
            model_status.append("NLTK OK")
        else:
            model_status.append("NLTK Falta")
            
        footer_text = f"Tela versão 1.7 - {' | '.join(model_status)}"
        
        footer = ttk.Label(
            main_frame,
            text=footer_text,
            style='Footer.TLabel'
        )
        footer.pack(pady=(15, 0))
    
    def setup_styles(self):
        style = ttk.Style()
        
        style.theme_create('dark', settings={
            "TFrame": {
                "configure": {
                    "background": '#1e1e1e',
                    "borderwidth": 0,
                    "relief": 'flat'
                }
            },
            "TLabel": {
                "configure": {
                    "background": '#1e1e1e',
                    "foreground": '#a0a0a0'
                }
            }
        })
        style.theme_use('dark')
        
        style.configure('Dark.TFrame', background='#1e1e1e', borderwidth=0)
        style.configure('Title.TLabel', font=('Tekton Pro', 14, 'bold'), foreground='#4fc3f7')
        style.configure('Subtitle.TLabel', font=('Tekton Pro', 10, 'bold'), foreground='#81c784')
        style.configure('Footer.TLabel', font=('Tekton Pro', 8), foreground='#757575')
    
    def extrair_texto_noticia(self, url):
        """Extrai texto de uma notícia através da URL"""
        try:
            artigo = Article(url, language='pt')
            artigo.download()
            artigo.parse()
            return artigo.text
        except Exception as e:
            return f"Erro ao extrair texto da notícia: {e}"
    
    def extract_article_from_link(self):
        """Nova função para extrair artigo e colar no campo de entrada"""
        current_text = self.input_text.get("1.0", tk.END).strip()
        
        if not current_text:
            messagebox.showwarning("Aviso", "Por favor, cole um link no campo de texto primeiro")
            return
        
        if not (current_text.startswith("http://") or current_text.startswith("https://")):
            messagebox.showwarning("Aviso", "O texto não parece ser um link válido. Certifique-se de incluir http:// ou https://")
            return
        
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, "Extraindo artigo do link... Aguarde...")
        self.root.update()
        
        extracted_text = self.extrair_texto_noticia(current_text)
        
        self.input_text.delete("1.0", tk.END)
        
        if extracted_text.startswith("Erro"):
            self.input_text.insert(tk.END, current_text)
            messagebox.showerror("Erro", extracted_text)
        else:
            self.input_text.insert(tk.END, extracted_text)
            messagebox.showinfo("Sucesso", "Artigo extraído com sucesso! Agora você pode usar os botões de resumo.")
    
    def generate_summary(self):
        """Resumo extrativo usando NLTK"""
        if not NLTK_RESOURCES_AVAILABLE:
            messagebox.showerror("Erro", "Recursos do NLTK não disponíveis. Instale os pacotes:\n\nnltk.download('punkt')\nnltk.download('stopwords')\nnltk.download('averaged_perceptron_tagger')")
            return
        
        news_text = self.input_text.get("1.0", tk.END).strip()
        if not news_text:
            messagebox.showwarning("Aviso", "Por favor, insira um texto ou use o botão 'EXTRAIR ARTIGO' para processar um link")
            return

        if news_text.startswith("http://") or news_text.startswith("https://"):
            messagebox.showinfo("Dica", "Detectamos que você colou um link. Use o botão 'EXTRAIR ARTIGO' primeiro para obter melhores resultados!")
            extracted_text = self.extrair_texto_noticia(news_text)
            if extracted_text.startswith("Erro"):
                messagebox.showerror("Erro", extracted_text)
                return
            news_text = extracted_text

        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Processando resumo extrativo avançado...")
        self.root.update()

        # Executa o resumo em uma thread separada para não travar a interface
        def process_extractive_summary():
            try:
                summary = summarize_text(news_text)
                
                # Atualiza a interface na thread principal
                self.root.after(0, lambda: self.show_extractive_summary(summary))
                
            except Exception as e:
                error_msg = f"Erro ao gerar resumo extrativo: {str(e)}"
                self.root.after(0, lambda: self.show_extractive_error(error_msg))
        
        thread = threading.Thread(target=process_extractive_summary)
        thread.daemon = True
        thread.start()
    
    def show_extractive_summary(self, summary):
        """Mostra o resumo extrativo gerado"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        formatted_summary = f"📝 RESUMO EXTRATIVO (NLTK):\n\n{summary}\n\n🔍 Resumo gerado por algoritmo extrativo baseado em importância de frases."
        self.output_text.insert(tk.END, formatted_summary)
        self.output_text.config(state='disabled')
    
    def show_extractive_error(self, error_msg):
        """Mostra erro do resumo extrativo"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"❌ ERRO NO RESUMO EXTRATIVO:\n\n{error_msg}")
        self.output_text.config(state='disabled')
        messagebox.showerror("Erro", error_msg)
    
    def generate_abstract_summary(self):
        """Resumo abstrato usando o modelo T5"""
        if not ABSTRACT_SUMMARY_AVAILABLE:
            messagebox.showerror("Erro", "Modelo T5 não encontrado! Verifique se:\n\n1. A pasta 'abstract_summary_v1' existe\n2. As dependências estão instaladas (transformers, torch, etc.)")
            return
        
        if self.is_processing_abstract:
            messagebox.showinfo("Aviso", "Já existe um resumo abstrato sendo processado. Aguarde a conclusão.")
            return
        
        news_text = self.input_text.get("1.0", tk.END).strip()
        if not news_text:
            messagebox.showwarning("Aviso", "Por favor, insira um texto ou use o botão 'EXTRAIR ARTIGO' para processar um link")
            return

        if news_text.startswith("http://") or news_text.startswith("https://"):
            messagebox.showinfo("Dica", "Detectamos que você colou um link. Use o botão 'EXTRAIR ARTIGO' primeiro para obter melhores resultados!")
            extracted_text = self.extrair_texto_noticia(news_text)
            if extracted_text.startswith("Erro"):
                messagebox.showerror("Erro", extracted_text)
                return
            news_text = extracted_text

        # Verifica se o texto não é muito longo (limite do modelo)
        if len(news_text.split()) > 400:
            # Pega apenas os primeiras 400 palavras para evitar problemas de memória
            news_text = ' '.join(news_text.split()[:400])
            messagebox.showinfo("Aviso", "Texto muito longo! Usando apenas as primeiras 400 palavras para o resumo abstrato.")

        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Processando resumo abstrato com IA (pode demorar alguns segundos)...")
        self.output_text.config(state='disabled')
        self.root.update()
        
        # Desabilita o botão durante o processamento
        self.is_processing_abstract = True
        self.analyze_btn.btnbackground = '#757575'
        self.analyze_btn.draw_button()
        
        # Executa o resumo em uma thread separada para não travar a interface
        def process_abstract_summary():
            try:
                abstract_summary = summarize(news_text, max_length=200)
                
                # Atualiza a interface na thread principal
                self.root.after(0, lambda: self.show_abstract_summary(abstract_summary))
                
            except Exception as e:
                error_msg = f"Erro ao gerar resumo abstrato: {str(e)}"
                self.root.after(0, lambda: self.show_abstract_error(error_msg))
        
        # Inicia o processamento em thread separada
        thread = threading.Thread(target=process_abstract_summary)
        thread.daemon = True
        thread.start()
    
    def show_abstract_summary(self, summary):
        """Mostra o resumo abstrato gerado"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        formatted_summary = f"🤖 RESUMO ABSTRATO (Modelo T5):\n\n{summary}\n\n✨ Resumo gerado por inteligência artificial usando modelo de linguagem T5."
        self.output_text.insert(tk.END, formatted_summary)
        self.output_text.config(state='disabled')
        
        # Reabilita o botão
        self.is_processing_abstract = False
        self.analyze_btn.btnbackground = '#5D4037'
        self.analyze_btn.draw_button()
    
    def show_abstract_error(self, error_msg):
        """Mostra erro do resumo abstrato"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"❌ ERRO NO RESUMO ABSTRATO:\n\n{error_msg}\n\nTente com um texto menor ou verifique se o modelo está funcionando corretamente.")
        self.output_text.config(state='disabled')
        
        # Reabilita o botão
        self.is_processing_abstract = False
        self.analyze_btn.btnbackground = '#5D4037'
        self.analyze_btn.draw_button()
        
        messagebox.showerror("Erro", error_msg)
    
    def generate_knowledge_tree(self):
        news_text = self.input_text.get("1.0", tk.END).strip()
        if not news_text:
            messagebox.showwarning("Aviso", "Por favor, insira um texto ou use o botão 'EXTRAIR ARTIGO' para processar um link")
            return
        
        if news_text.startswith("http://") or news_text.startswith("https://"):
            messagebox.showinfo("Dica", "Detectamos que você colou um link. Use o botão 'EXTRAIR ARTIGO' primeiro!")
            return
        
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Processando árvore de conhecimento...")
        self.root.update()
        
        self.root.after(1500, lambda: self.show_knowledge_tree(news_text))
    
    def show_knowledge_tree(self, text):
        try:
            tree = self.simulate_knowledge_tree(text)
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, tree)
            self.output_text.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar árvore: {str(e)}")
    
    def simulate_knowledge_tree(self, text):
        keywords = list(set([word for word in text.split() if len(word) > 4 and word.isalpha()][:10]))
        return f"🌳 ÁRVORE DE CONHECIMENTO (SIMULAÇÃO):\n\nConceitos principais:\n{'- ' + '\n- '.join(keywords[:5])}\n\nRelacionamentos:\n- {keywords[0]} → {keywords[3]}\n- {keywords[2]} ↔ {keywords[4]}\n- {keywords[1]} → {keywords[5]}"

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkNewsSummarizer(root)
    root.mainloop()