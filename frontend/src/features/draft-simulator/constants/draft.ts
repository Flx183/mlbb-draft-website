
export const TIME_PER_ACTION = 30
export const SLOT_COUNT = 5

export interface RecommendationRequest {
  team: "blue" | "red";
  blue_picks: string[];
  red_picks: string[];
  blue_bans: string[];
  red_bans: string[];
  top_k?: number;
  strict_turn?: boolean;
}

export interface Recommendation {
  hero: string;
  rank: number;
  score: number;
  reasons: string[];
}

export interface RecommendationResponse {
  team: "blue" | "red";
  recommendations: Recommendation[];
  reasoning: string,
}