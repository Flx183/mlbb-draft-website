from pydantic import BaseModel
from typing import List, Optional

class DraftState(BaseModel):
    our_team: List[str]        
    enemy_team: List[str]       
    our_bans: List[str]             
    enemy_bans: List[str]
      

class RecommendationResponse(BaseModel):
    recommendations: List[str]  # top picks suggested
    reasoning: str 