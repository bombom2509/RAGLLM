from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_utils import process_pdf, get_rag_response
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    process_pdf(file.filename, content)
    return {"message": f"{file.filename} processed and embedded"}

@app.post("/chat")
async def chat_with_pdf(question: str = Form(...)):
    answer = get_rag_response(question)
    return {"answer": answer}
