from fastapi import FastAPI
from api.draft import router as draft_router
from backend.models.draft_model import RecommendationResponse, DraftState


app = FastAPI()
app.include_router(draft_router, prefix="/draft", tags=["draft"])

@app.post("/draft/recommend", response_model=RecommendationResponse)
async def get_recommendation(draft: DraftState):
    pass
