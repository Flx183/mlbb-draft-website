import type { RecommendationRequest, RecommendationResponse} from "./types/draft"

const API_BASE = "https://ml-2-8lkf.onrender.com";

async function postDraftState(
  endpoint: string,
  payload: RecommendationRequest
): Promise<RecommendationResponse> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }

  return res.json();
}

export function fetchBanRecommendations(payload: RecommendationRequest) {
  return postDraftState("/draft/recommend-bans", payload);
}

export function fetchPickRecommendations(payload: RecommendationRequest) {
  return postDraftState("/draft/recommend-picks", payload);
}