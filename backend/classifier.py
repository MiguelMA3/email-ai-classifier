from transformers import pipeline

try:
    print("Carregando modelo de classificação...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Modelo de classificação carregado.")

    print("Carregando modelo de geração...")
    generator = pipeline("text-generation", model="distilgpt2")
    print("Modelo de geração carregado.")

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
        prompt = "Escreva uma resposta curta e profissional de e-mail confirmando o recebimento:"
    else:
        prompt = "Escreva uma resposta curta e educada para cancelar a inscrição ou recusar:"

    result = generator(prompt, max_new_tokens=30, num_return_sequences=1)
    
    generated_text = result[0]['generated_text']
    clean_reply = generated_text.replace(prompt, "").strip()
    
    if not clean_reply:
        if category == "produtivo":
            clean_reply = "Obrigado, recebido."
        else:
            clean_reply = "Obrigado, mas não tenho interesse."

    return clean_reply