from fastapi import FastAPI
from .routes.chat_router import router as chat_router

app = FastAPI(title="KrishiMitra Agent", version="0.1")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(chat_router)