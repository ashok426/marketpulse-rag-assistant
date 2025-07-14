from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from data_retriever.retriever import Retriever

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

# Initialize your retriever (adjust host/port/collection as needed)
retriever = Retriever("localhost", 6333, "document_chunks_rag")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def get_response(msg: str = Form(...)):
    answer = retriever.rag_pipeline(msg, top_k=15, mmr_k=5)
    return JSONResponse(content=answer)