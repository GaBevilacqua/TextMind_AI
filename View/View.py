import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

class DarkNewsSummarizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Mind AI")
        self.root.geometry("800x650")
        self.root.configure(bg='#121212')
        

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
        

        input_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            input_frame,
            text="COLE SUA NOTÍCIA:",
            style='Subtitle.TLabel'
        ).pack(anchor='w')
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=85,
            height=15,
            font=('Tekton Pro', 10),
            bg='#252525',
            fg='#e0e0e0',
            insertbackground='white',
            selectbackground='#454545',
            relief='groove',
            borderwidth=2
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        #Botões
        self.summarize_btn = ttk.Button(
            main_frame,
            text="GERAR RESUMO IA",
            command=self.generate_summary,
            style='Accent.TButton'
        )
        self.summarize_btn.pack(pady=15, ipadx=10, ipady=5)
        self.summarize_btn = ttk.Button(
            main_frame,
            text="GERAR ÁRVORE DE CONHECIMENTO",
            command=self.generate_summary,
            style='Accent.TButton'
        )
        self.summarize_btn.pack(pady=10, ipadx=15, ipady=5)
        

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
        

        footer = ttk.Label(
            main_frame,
            text="Tela versão 1.1",
            style='Footer.TLabel'
        )
        footer.pack(pady=(15, 0))
    
    def setup_styles(self):
        style = ttk.Style()
        

        style.theme_create('dark', settings={
            "TFrame": {
                "configure": {
                    "background": '#1e1e1e',
                    "borderwidth": 2,
                    "relief": 'groove'
                }
            },
            "TLabel": {
                "configure": {
                    "background": '#1e1e1e',
                    "foreground": '#a0a0a0'
                }
            },
            "TButton": {
                "configure": {
                    "anchor": "center",
                    "padding": 5
                }
            }
        })
        style.theme_use('dark')
        

        style.configure('Dark.TFrame', background='#1e1e1e')
        style.configure('Title.TLabel', font=('Tekton Pro', 14, 'bold'), foreground='#4fc3f7')
        style.configure('Subtitle.TLabel', font=('Tekton Pro', 10, 'bold'), foreground='#81c784')
        style.configure('Accent.TButton', 
                      font=('Tekton Pro', 10, 'bold'),
                      foreground='white',
                      background='#00695c',
                      borderwidth=2,
                      relief='raised')
        style.map('Accent.TButton',
                background=[('active', '#00897b'), ('pressed', '#004d40')])
        style.configure('Footer.TLabel', font=('Tekton Pro', 8), foreground='#757575')
    
    def generate_summary(self):
        news_text = self.input_text.get("1.0", tk.END).strip()
        
        if not news_text:
            messagebox.showwarning("Aviso", "Por favor, insira um texto para resumir")
            return
        
        self.summarize_btn.config(state=tk.DISABLED)
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Processando...")
        self.root.update()
        
        self.root.after(1500, lambda: self.show_real_summary(news_text))
    
    def show_real_summary(self, text):
        try:
            summary = self.simulate_ai_summary(text)
            
            self.output_text.config(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, summary)
            self.output_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar resumo: {str(e)}")
        finally:
            self.summarize_btn.config(state=tk.NORMAL)
    
    def simulate_ai_summary(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]
        if len(sentences) > 2:
            return f" RESUMO IA (SIMULAÇÃO):\n\n• {sentences[0]}.\n• {sentences[len(sentences)//2]}.\n• {sentences[-1]}.\n\nPontos-chaves extraídos automaticamente."
        return "RESUMO IA (SIMULAÇÃO):\n\n" + text[:250] + " [...]"

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkNewsSummarizer(root)
    root.mainloop()