from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import pdfplumber, os, tempfile, requests
from utils.preprocess import preprocess_text
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_MODEL = os.environ.get("GITHUB_MODEL", "gpt-4o")
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"

app = FastAPI(title="Classificador de Emails")
templates = Jinja2Templates(directory="templates")


def consultar_ia(prompt: str, primeira_linha: bool = False, temperatura: float = 0.7, max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    dados = {
        "model": GITHUB_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperatura),
        "max_tokens": int(max_tokens)
    }

    resp = requests.post(GITHUB_MODELS_URL, headers=headers, json=dados, timeout=60)
    resposta = resp.json()["choices"][0]["message"]["content"].strip()

    if primeira_linha:
        resposta = resposta.split('\n')[0]

    return resposta


def classificar_email(texto: str) -> str:
    prompt = f"""Você é um assistente que classifica emails. Responda APENAS com UMA palavra: PRODUTIVO ou IMPRODUTIVO (tudo em maiúsculas), sem explicações adicionais.

Critérios rápidos:
- PRODUTIVO: contém pedido de ação, pergunta que exige resposta, tarefa, prazo, decisão, solicitação de reunião, pedido de confirmação, ou informações que exigem follow-up.
- IMPRODUTIVO: anúncios, newsletters, marketing, confirmação informal sem ação, agradecimento sem pedido de ação.

Se o conteúdo estiver ambíguo mas houver qualquer indicação de ação, marque PRODUTIVO.

Email:
"{texto}"
"""
    return consultar_ia(prompt, primeira_linha=True, temperatura=0.0, max_tokens=60)


def gerar_resposta(categoria: str, texto: str) -> str:
    if "improdutivo" in categoria.lower():
        return "Não é necessária ação imediata."
    
    prompt = f"""Você é um assistente que escreve respostas profissionais e concisas por e-mail. Com base no email abaixo, escreva UMA resposta pronta para envio (máx. 2-3 frases, 1 parágrafo) em português brasileiro, tom profissional, objetiva e direta. Inclua um próximo passo claro quando pertinente. Não invente informações.

Email:
"{texto}"

Responda APENAS com o texto da resposta, sem introduções, sem explicações e sem assinatura."""
    return consultar_ia(prompt, temperatura=0.3, max_tokens=140)


async def extrair_email(texto: str, arquivo: UploadFile) -> str:
    if arquivo and arquivo.filename:
        if arquivo.filename.endswith(".txt"):
            conteudo = await arquivo.read()
            return conteudo.decode("utf-8")
        elif arquivo.filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                conteudo = await arquivo.read()
                tmp.write(conteudo)
                tmp.flush()
                tmp_path = tmp.name
            
            with pdfplumber.open(tmp_path) as pdf:
                texto_pdf = "\n".join(p.extract_text() or "" for p in pdf.pages)
            os.unlink(tmp_path)
            return texto_pdf
    
    return texto or ""


@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def processar_email(
    request: Request,
    email_text: str = Form(""),
    email_file: UploadFile = File(None)
):
    texto = await extrair_email(email_text, email_file)
    
    if not texto:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Por favor, insira um email ou faça upload de um arquivo"
        })
    
    texto = preprocess_text(texto)
    categoria = classificar_email(texto)
    resposta = gerar_resposta(categoria, texto)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "category": categoria,
        "suggestion": resposta
    })
