from transformers import pipeline

try:
    print("Carregando modelo de classificação...")
    classifier = pipeline("zero-shot-classification", model="poltextlab/xlm-roberta-large-portuguese-cap-v3")
    print("Modelo de classificação carregado.")

    print("Carregando modelo de geração (Português)...")
    generator = pipeline("text-generation", model="unicamp-dl/ptt5-base-portuguese-vocab")
    print("Modelo de geração carregado.")

    if generator.tokenizer.pad_token_id is None:
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
        generator.model.config.pad_token_id = generator.model.config.eos_token_id

except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    classifier = None
    generator = None

def classify_email_text(text):
    if not classifier:
        raise RuntimeError("O pipeline de classificação não foi carregado.")
        
    candidate_labels = ["produtivo", "improdutivo"]
    hypothesis_template = "Este email é {}." 
    
    result = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    
    category = result['labels'][0]
    score = result['scores'][0]
    
    return category, score

def generate_suggestion(category):
    if not generator:
        raise RuntimeError("O pipeline de geração não foi carregado.")

    if category == "produtivo":
        prompt = "Escreva uma resposta de email profissional e curta, confirmando o recebimento e dizendo que o assunto será analisado. Comece com 'Prezado(a),'."
        max_tokens = 45
    else:
        prompt = "Escreva uma resposta de email curta e educada, solicitando o cancelamento da inscrição desta lista de emails. Comece com 'Olá,'."
        max_tokens = 35

    result = generator(prompt, max_new_tokens=max_tokens, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    
    generated_text = result[0]['generated_text']
    
    clean_reply = generated_text.replace(prompt, "").strip()
    
    clean_reply = clean_reply.split('\n\n')[0]
    
    if not clean_reply or len(clean_reply) < 15: # Se a resposta for muito curta/vazia
        if category == "produtivo":
            clean_reply = "Prezado(a),\n\nRecebido. Vamos analisar e retornamos em breve.\n\nAtenciosamente."
        else:
            clean_reply = "Olá,\n\nPor favor, remova meu email desta lista de correspondência.\n\nObrigado(a)."

    return clean_reply