from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier import classify_email_text, generate_suggestion

class EmailInput(BaseModel):
    content: str

class AnalysisResponse(BaseModel):
    category: str
    confidence_score: float
    suggested_reply: str

app = FastAPI(
    title="API de Classificação de Email",
    description="Backend para classificar emails e sugerir respostas.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "Hello there! I'm an API."}


@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_email(email: EmailInput):
    if not email.content or email.content.strip() == "":
        raise HTTPException(status_code=400, detail="O conteúdo do email não pode estar vazio.")

    try:
        category, score = classify_email_text(email.content)

        reply_suggestion = generate_suggestion(category)

        return AnalysisResponse(
            category=category,
            confidence_score=score,
            suggested_reply=reply_suggestion
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Erro no servidor de IA: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")