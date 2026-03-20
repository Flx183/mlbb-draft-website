from fastapi import APIRouter
from pydantic import BaseModel
from services.draft_scoring import recommend_heroes

router = APIRouter()

class DraftRequest(BaseModel):
    blue_picks: list[str] = []
    red_picks: list[str] = []
    blue_bans: list[str] = []
    red_bans: list[str] = []
    team: str
    top_k: int = 5

@router.post("/recommend")
def recommend(req: DraftRequest):
    recs = recommend_heroes(
        blue_picks=req.blue_picks,
        red_picks=req.red_picks,
        blue_bans=req.blue_bans,
        red_bans=req.red_bans,
        team=req.team,
        top_k=req.top_k,
    )
    return {"recommendations": recs}