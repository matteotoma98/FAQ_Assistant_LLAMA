import streamlit as st
from llama_cpp import Llama
import json
from pathlib import Path
import os
import requests

def download_model(model_url, model_folder):
    """Scarica il modello se non è presente nella cartella"""
    # Estrai il nome del modello dall'URL o dal Content-Disposition
    response = requests.head(model_url, allow_redirects=True)
    if response.status_code == 200:
        filename = model_url.split('/')[-1].split('?')[0]  # Usa il nome del file dall'URL
        model_path = os.path.join(model_folder, filename)
        
        # Se il modello non è già presente, procedi con il download
        if not os.path.exists(model_path):
            st.info(f"🚀 Scaricando il modello da {model_url}...")
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"✅ Modello scaricato con successo: {model_path}")
            else:
                st.error(f"❌ Errore nel download del modello. Codice di stato: {response.status_code}")
                raise Exception("Errore nel download del modello")
        else:
            st.success(f"✅ Modello già presente in: {model_path}")
    else:
        st.error(f"❌ Errore nel verificare il file: Codice di stato {response.status_code}")
        raise Exception("Impossibile verificare il file del modello")
    
    return model_path

# Esegui il controllo e il download all'avvio dell'app
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf?download=true"  # URL del modello
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

model_path = download_model(model_url, model_folder)

# Configurazione ottimizzata per la velocità
@st.cache_resource
def initialize_model(model_path):
    """Inizializza il modello con configurazioni ottimizzate"""
    return Llama(
        model_path=model_path,  # Usa il percorso dinamico del modello scaricato
        n_ctx=512,              # Context window ridotta
        n_threads=6,            # Più threads
        n_batch=512,            # Batch size aumentato
        n_gpu_layers=32         # Usa GPU se disponibile
    )

@st.cache_data
def load_product_faq():
    """Carica e cachea il database FAQ"""
    with open("product_faq.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_compact_prompt(product_faqs, product_name, user_question):
    """Genera un prompt più conciso con istruzioni per rispondere in italiano"""
    relevant_faqs = {
        q: a for q, a in product_faqs[product_name].items() 
        if any(word in q.lower() for word in user_question.lower().split())
    }
    
    if not relevant_faqs:
        relevant_faqs = product_faqs[product_name]
    
    return f"""Product: {product_name}
FAQ: {json.dumps(relevant_faqs, ensure_ascii=False)}
Q: {user_question}
A: Rispondi in italiano:"""  # Aggiunta la specifica per la risposta in italiano

def main():
    st.title("🤖 FAQ Assistant con LLAMA 🦙")
    
    # Inizializza il modello (cache)
    try:
        llm = initialize_model(model_path)
    except Exception as e:
        st.error(f"Errore inizializzazione modello: {str(e)}")
        return
    
    # Carica FAQ (cache)
    product_faqs = load_product_faq()
    
    # Selezione prodotto con layout ottimizzato
    col1, col2 = st.columns([2, 3])
    with col1:
        product_name = st.selectbox(
            "Prodotto:",
            options=list(product_faqs.keys())
        )
    
    with col2:
        user_question = st.text_input("Domanda:")
    
    if user_question:
        with st.spinner("⚡"):
            prompt = generate_compact_prompt(product_faqs, product_name, user_question)
            
            try:
                response = llm(
                    prompt,
                    max_tokens=128,        # Ridotto per risposte più veloci
                    temperature=0.5,       # Ridotto per più determinismo
                    top_p=0.9,
                    stop=["Q:", "\n\n"],
                    stream=True            # Streaming per feedback immediato
                )
                
                # Container per la risposta
                response_container = st.empty()
                full_response = ""
                
                # Streaming della risposta
                for chunk in response:
                    if chunk['choices'][0]['text']:
                        full_response += chunk['choices'][0]['text']
                        response_container.write(full_response)
            
            except Exception as e:
                st.error(f"Errore: {str(e)}")
    
    # FAQ in sidebar per layout più pulito
    with st.sidebar:
        st.subheader(f"📚 FAQ {product_name}")
        for q, a in product_faqs[product_name].items():
            with st.expander(q):
                st.write(a)

if __name__ == "__main__":
    main()
