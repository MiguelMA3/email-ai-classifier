from transformers import pipeline

try:
    print("Carregando modelo de classificação...")
    CLASSIFIER_MODEL = "FacebookAI/xlm-roberta-base"
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL)
    print(f"Modelo de classificação ({CLASSIFIER_MODEL}) carregado.")

    print("Carregando modelo de geração (Português)...")
    GENERATOR_MODEL = "nicholasKluge/Aira-2-portuguese-124M" 
    generator = pipeline("text-generation", model=GENERATOR_MODEL)
    print(f"Modelo de geração ({GENERATOR_MODEL}) carregado.")

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
        
    LABEL_PRODUTIVO = "tarefa profissional, pedido urgente ou responsabilidade"

    LABEL_IMPRODUTIVO = "mensagem nao solicitada, publicidade ou marketing"
    
    zero_shot_labels = [LABEL_PRODUTIVO, LABEL_IMPRODUTIVO]

    hypothesis_template = "Este email é sobre {}."
    
    result = classifier(text, zero_shot_labels, hypothesis_template=hypothesis_template, multi_label=False)
    
    predicted_label = result['labels'][0]
    score = result['scores'][0]
    
    if predicted_label == LABEL_PRODUTIVO:
        category = "produtivo"
    else:
        category = "improdutivo"
    
    return category, score

def generate_suggestion(category):
    if not generator:
        raise RuntimeError("O pipeline de geração não foi carregado.")

    if category == "produtivo":
        prompt = "Você é um assistente de e-mail profissional. A tarefa é gerar uma **resposta de e-mail** profissional, **muito curta** e direta, confirmando o recebimento de uma mensagem importante e informando que o assunto será revisado. **O texto deve ser em Português do Brasil.** Comece a resposta com 'Prezado(a),' e finalize com uma saudação formal como 'Atenciosamente'. Resposta:"
        max_tokens = 50
    else:
        prompt = "Você é um assistente de e-mail. A tarefa é gerar uma **resposta de e-mail** educada e **muito curta** para **solicitar o cancelamento da inscrição** em uma lista de e-mails. **O texto deve ser em Português do Brasil.** Comece a resposta com 'Olá,' e finalize com 'Obrigado(a)'. Resposta:"
        max_tokens = 40

    result = generator(
        prompt, 
        max_new_tokens=max_tokens, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True,
        temperature=0.7, 
        do_sample=True
    )
    
    generated_text = result[0]['generated_text']

    clean_reply = generated_text.replace(prompt, "").strip()
    if 'Resposta:' in clean_reply and clean_reply.startswith('Resposta:'):
        clean_reply = clean_reply.split('Resposta:')[-1].strip()

    clean_reply = clean_reply.split('\n\n')[0]
    
    if not clean_reply or len(clean_reply) < 15:
        if category == "produtivo":
            clean_reply = "Prezado(a),\n\nRecebido. Vamos analisar e retornamos em breve.\n\nAtenciosamente."
        else:
            clean_reply = "Olá,\n\nPor favor, remova meu email desta lista de correspondência.\n\nObrigado(a)."

    return clean_reply