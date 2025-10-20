import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from newspaper import Article
import sys
import os
import threading
import subprocess
from transformers import T5ForConditionalGeneration, T5TokenizerFast
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
from graphviz import Digraph
from collections import Counter

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
# Tenta carregar o modelo T5
# Tenta carregar o modelo T5
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "Model", "checkpoint-1400"),
        os.path.join(script_dir, "Model"),
        os.path.join(script_dir, "T5_Summarization", "model"),
        os.path.join(script_dir, "checkpoint-1400"),
        "./Model/checkpoint-1400",
        "./Model",
        "./T5_Summarization/model",
        "./checkpoint-1400"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = os.path.join(path, "config.json")
            has_weights = (
                os.path.exists(os.path.join(path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(path, "model.safetensors")) or
                os.path.exists(os.path.join(path, "tf_model.h5"))
            )
            
            if os.path.exists(config_path) and has_weights:
                model_path = path
                print(f"Modelo T5 encontrado em: {model_path}")
                break
    
    if model_path is None:
        print(f"Modelo T5 nao encontrado. Diretorio atual: {os.getcwd()}")
        print(f"Diretorio do script: {script_dir}")
        raise FileNotFoundError("Modelo T5 nao encontrado ou incompleto")
    
    from transformers import T5TokenizerFast
    
    print("Carregando tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    
    print("Carregando modelo T5...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    model = model.to(device)
    
    def summarize(text, max_length=150):
        input_text = "summarize: " + text
        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=30,
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    ABSTRACT_SUMMARY_AVAILABLE = True
    print("Modelo T5 carregado com sucesso!")
    
except Exception as e:
    print(f"Aviso: Nao foi possivel carregar o modelo T5")
    print(f"Erro: {e}")
    ABSTRACT_SUMMARY_AVAILABLE = False
    
except Exception as e:
    print(f"Aviso: Nao foi possivel carregar o modelo T5")
    print(f"Erro: {e}")
    ABSTRACT_SUMMARY_AVAILABLE = False

class MindMapGenerator:
    def __init__(self, keywords_per_branch=8, color="#00C853"):
  
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'and', 'or', 'in', 'on', 'at', 'to', 
                             'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                             'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                             'but', 'if', 'or', 'an', 'as', 'from', 'they', 'their'}
        
        self.keywords_per_branch = keywords_per_branch
        self.color = color
        
    def extract_main_topic(self, text):
   
        words = word_tokenize(text.lower())
        pos_tags = pos_tag(words)
        
        proper_nouns = [word for word, tag in pos_tags if tag in {'NNP', 'NNPS'}]
        
        if proper_nouns:
            most_common = Counter(proper_nouns).most_common(1)[0][0]
            return most_common.title()
        
        nouns = [word for word, tag in pos_tags 
                if tag in {'NN', 'NNS'} and word not in self.stop_words and len(word) > 3]
        
        if nouns:
            most_common = Counter(nouns).most_common(1)[0][0]
            return most_common.title()
        
        return "Main Topic"
    
    def extract_keywords(self, sentence, used_words, max_keywords=8):
      
        words = word_tokenize(sentence.lower())
        pos_tags = pos_tag(words)
        
        candidates = []
        
        for i in range(len(words) - 1):
            word1, tag1 = pos_tags[i]
            word2, tag2 = pos_tags[i + 1]
            
            if (tag1 in {'JJ', 'NN', 'NNP'} and tag2 in {'NN', 'NNS', 'NNP', 'NNPS'}):
                phrase = f"{word1} {word2}"
                if phrase not in used_words and word1 not in self.stop_words:
                    candidates.append((phrase, 8))
                    used_words.add(phrase)
        
        for word, tag in pos_tags:
            if tag == 'CD':
                if word not in used_words:
                    candidates.append((word, 10))
                    used_words.add(word)
                continue
            
            if not word.isalpha() or len(word) < 3:
                continue
            
            if word in self.stop_words or word in used_words:
                continue
            
            priority = 0
            
            if tag in {'NNP', 'NNPS'}:
                priority = 9
            elif tag in {'NN', 'NNS'}:
                priority = 7
            elif tag in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}:
                priority = 6
            elif tag in {'JJ', 'JJR', 'JJS'}:
                priority = 5
            elif tag in {'RB', 'RBR', 'RBS'}:
                priority = 4
            else:
                continue
            
            candidates.append((word, priority))
            used_words.add(word)
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [word.title() for word, _ in candidates[:max_keywords]]
    
    def create_mindmap(self, summary_text, output_file="mindmap", 
                      output_format="png", center_title=None):
      
        sentences = sent_tokenize(summary_text)
        
        if not sentences:
            print("No sentences found.")
            return None
        
        used_words = set()
        
        dot = Digraph(comment='Mind Map', format=output_format, engine='fdp')
        dot.attr(splines='curved', overlap='false', sep='+1.2', K='2.5')
        dot.attr('node', fontname='Arial Bold', fontsize='13', penwidth='2.5')
        dot.attr('edge', penwidth='3')
        
     
        if center_title:
            main_topic = center_title
        else:
            main_topic = self.extract_main_topic(summary_text)
        
        used_words.add(main_topic.lower())
        
        dot.node("CENTER", main_topic,
                shape='ellipse',
                style='filled,bold',
                fillcolor=self.color,
                fontcolor='white',
                fontsize='18',
                width='2.2',
                height='1.3',
                penwidth='4')
        
        total_concepts = 0
        
    
        for idx, sentence in enumerate(sentences):
            branch_label = sentence if len(sentence) <= 65 else sentence[:62] + '...'
            branch_id = f"B{idx}"
            
          
            dot.node(branch_id, branch_label,
                    shape='ellipse',
                    style='filled,bold',
                    fillcolor=self.color,
                    fontcolor='white',
                    fontsize='13',
                    margin='0.4,0.25',
                    penwidth='3')
            
            
            dot.edge("CENTER", branch_id,
                    color=self.color,
                    penwidth='4',
                    weight='3')
            
            
            keywords = self.extract_keywords(
                sentence, used_words, max_keywords=self.keywords_per_branch
            )
            total_concepts += len(keywords)
            
            
            for k_idx, keyword in enumerate(keywords):
                keyword_id = f"B{idx}_K{k_idx}"
                
                dot.node(keyword_id, keyword,
                        shape='ellipse',
                        style='filled',
                        fillcolor=self.color,
                        fontcolor='white',
                        fontsize='18',
                        margin='0.3,0.15',
                        penwidth='2')
                
                dot.edge(branch_id, keyword_id,
                        color=self.color,
                        penwidth='2.5',
                        weight='1')
        
        
        try:
            output_path = dot.render(filename=output_file, cleanup=True)
            print(f"Mind map created: {output_path}")
            print(f"Sentences: {len(sentences)}")
            print(f"Concepts: {total_concepts}")
            print(f"Density: {total_concepts / len(sentences):.1f} concepts/sentence")
            return output_path
        except Exception as e:
            print(f"Error: {e}")
            return None

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

def summarize_text(text, min_sentences=2, max_sentences=3, max_chars=350):
    summarizer = AdvancedSummarizer()
    return summarizer.summarize(text, min_sentences, max_sentences, max_chars)
    


class RoundedButton(tk.Canvas):
    def __init__(self, master=None, text="", radius=25, btnforeground="#000000", btnbackground="#ffffff", 
                 clicked=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.config(bg='#1e1e1e', highlightthickness=0, bd=0) 
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

        self.is_processing_abstract = False
        
        self.mindmap_generator = MindMapGenerator()
        
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
        
     
        label_frame = ttk.Frame(input_frame, style='Dark.TFrame')
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(
            label_frame,
            text="COLE SUA NOTÍCIA OU LINK:",
            style='Subtitle.TLabel'
        ).pack(side=tk.LEFT, anchor='w')
        
      
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
            
        footer_text = f"Tela versão 2.0"
        
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
        current_text = self.input_text.get("1.0", tk.END).strip()
        
        if not current_text:
            messagebox.showwarning("Aviso", "Por favor, cole um link no campo de texto primeiro")
            return
        
        if not (current_text.startswith("http://") or current_text.startswith("https://")):
            messagebox.showwarning("Aviso", "Certifique-se de incluir http:// ou https://")
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
            messagebox.showinfo("Sucesso", "Artigo extraído com sucesso!")
    
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

        def process_extractive_summary():
            try:
                summary = summarize_text(news_text)
                
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
        formatted_summary = f"RESUMO EXTRATIVO (NLTK):\n\n{summary}\n\n Resumo gerado por algoritmo extrativo baseado em importância de frases."
        self.output_text.insert(tk.END, formatted_summary)
        self.output_text.config(state='disabled')
    
    def show_extractive_error(self, error_msg):
        """Mostra erro do resumo extrativo"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f" ERRO NO RESUMO EXTRATIVO:\n\n{error_msg}")
        self.output_text.config(state='disabled')
        messagebox.showerror("Erro", error_msg)
    
    def generate_abstract_summary(self):
        """Resumo abstrato usando o modelo T5"""
        if not ABSTRACT_SUMMARY_AVAILABLE:
            messagebox.showerror("Erro", "Modelo não encontrado! ")
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

        if len(news_text.split()) > 800:
            news_text = ' '.join(news_text.split()[:800])
            messagebox.showinfo("Aviso", "Texto muito longo! Usando apenas as primeiras 800 palavras para o resumo abstrato.")

        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Processando resumo abstrato com IA (pode demorar alguns segundos)...")
        self.output_text.config(state='disabled')
        self.root.update()
        

        self.is_processing_abstract = True
        self.analyze_btn.btnbackground = '#757575'
        self.analyze_btn.draw_button()
        
        def process_abstract_summary():
            try:
                abstract_summary = summarize(news_text, max_length=150)
                
                # Atualiza a interface na thread principal
                self.root.after(0, lambda: self.show_abstract_summary(abstract_summary))
                
            except Exception as e:
                error_msg = f"Erro ao gerar resumo abstrato: {str(e)}"
                self.root.after(0, lambda: self.show_abstract_error(error_msg))
        
        thread = threading.Thread(target=process_abstract_summary)
        thread.daemon = True
        thread.start()
    
    def show_abstract_summary(self, summary):
        """Mostra o resumo abstrato gerado"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        formatted_summary = f" RESUMO ABSTRATO (Modelo T5):\n\n{summary}\n\n"
        self.output_text.insert(tk.END, formatted_summary)
        self.output_text.config(state='disabled')
        

        self.is_processing_abstract = False
        self.analyze_btn.btnbackground = '#5D4037'
        self.analyze_btn.draw_button()
    
    def show_abstract_error(self, error_msg):
        """Mostra erro do resumo abstrato"""
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f" ERRO NO RESUMO ABSTRATO:\n\n{error_msg}\n\nTente com um texto menor ou verifique se o modelo está funcionando corretamente.")
        self.output_text.config(state='disabled')
    
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
            extracted_text = self.extrair_texto_noticia(news_text)
            if extracted_text.startswith("Erro"):
                messagebox.showerror("Erro", extracted_text)
                return
            news_text = extracted_text

        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Gerando resumo extrativo e criando mapa mental...")
        self.root.update()

        try:
            
            if not NLTK_RESOURCES_AVAILABLE:
                messagebox.showerror("Erro", "Recursos do NLTK não disponíveis para gerar o resumo.")
                return
            
            summary = summarize_text(news_text, min_sentences=3, max_sentences=5, max_chars=500)
            
          
            image_path = self.mindmap_generator.create_mindmap(summary)
            
          
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            success_msg = f" Mapa mental gerado com sucesso!\n\nBaseado no resumo extrativo:\n{summary}\n\n---\n\nArquivo salvo em:\n{image_path}"
            self.output_text.insert(tk.END, success_msg)
            self.output_text.config(state='disabled')
            
         
            try:
                if sys.platform == "win32":
                    os.startfile(image_path)
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, image_path])
            except:
                pass
                
        except Exception as e:
            error_msg = f"Erro ao gerar mapa mental:\n{str(e)}"
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, error_msg)
            self.output_text.config(state='disabled')
            messagebox.showerror("Erro", error_msg)
if __name__ == "__main__":
    root = tk.Tk()
    app = DarkNewsSummarizer(root)
    root.mainloop()
