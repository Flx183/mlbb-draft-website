export type Role =
  | "Tank"
  | "Fighter"
  | "Assassin"
  | "Mage"
  | "Marksman"
  | "Support"
  | "Other";

export type DraftPhase = "ban1" | "pick1" | "ban2" | "pick2";

export type DraftAction = "ban" | "pick";

export type Team = "blue" | "red";

export interface Hero {
  id: number;
  name: string;
  role: Role[];     
  image: string;
}

export interface DraftStep {
  phase: DraftPhase;
  team: Team;
  action: DraftAction;
}

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