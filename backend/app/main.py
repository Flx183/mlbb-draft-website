from fastapi import FastAPI
from api.draft import router as draft_router

app = FastAPI()
app.include_router(draft_router, prefix="/draft", tags=["draft"])