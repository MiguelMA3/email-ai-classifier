from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import pipeline

CLASSIFICATION_MODEL = "facebook/bart-large-mnli"
GENERATION_MODEL = "t5-small"

CATEGORIES = ["Produtivo", "Improdutivo"]

try:
    classifier = pipeline(
        "zero-shot-classification",
        model=CLASSIFICATION_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )

    text_generator = pipeline(
        "text-generation",
        model=GENERATION_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )

    print("AI Models loaded successfully!")

except Exception as e:
    print(f"Error loading AI models: {e}")
    classifier = None
    text_generator = None

app = FastAPI(title="AI Email Classifier API")

class EmailImput(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    category: str
    suggested_response: str

def generate_response_prompt(category: str, email_text: str) -> str:
    if category == "Produtivo":
        return (
            f"O seguinte email é Produtivo e requer uma ação ou resposta. "
            f"Com base no email abaixo, escreva uma resposta formal e concisa "
            f"solicitando mais detalhes ou informando que o pedido está em análise. "
            f"Email: '{email_text[:150]}...'"
        )
    elif category == "Improdutivo":
        return (
            f"O seguinte email é Improdutivo (por exemplo, agradecimento, saudação). "
            f"Escreva uma resposta curta e cordial de confirmação ou agradecimento. "
            f"Email: '{email_text[:150]}...'"
        )
    return "Resposta padrão. Não foi possível gerar uma resposta específica."

@app.post("/classify-email", response_model=ClassificationResult)
async def classify_and_respond(email_imput: EmailImput):
    if classifier is None or text_generator is None:
        raise HTTPException(
            status_code=503,
            detail="AI unavaiable. Please reload models."
        )

    email_text = email_imput.text.strip()
    if not email_text:
        raise HTTPException(status_code=400, detail="Email can't be NULL")
    
    classification_result = classifier(
        email_text,
        candidate_labels=CATEGORIES
    )

    category = classification_result['labels'][0]
    score = classification_result['scores'][0]

    prompt = generate_response_prompt(category, email_text)

    generated_text = text_generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    suggested_response = generated_text[0]['generated_text'].replace(prompt, '').strip()

    return ClassificationResult(
        category=category,
        suggested_response=suggested_response
    )