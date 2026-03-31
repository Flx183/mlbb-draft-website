from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.draft import router as draft_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://127.0.0.1:5173",
        "https://djsudartha.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(draft_router, prefix= "/draft")

@app.get("/")
def root():
    return {"message": "MLBB Draft Backend Running"}

@app.get("/draft")
def idk():
    pass    