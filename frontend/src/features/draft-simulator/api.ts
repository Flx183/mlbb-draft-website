import type { RecommendationRequest, RecommendationResponse  } from "./types/draft"     

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function fetchRecommendations(
    payload: RecommendationRequest
): Promise<RecommendationResponse> {
    const res = await fetch(`${API_BASE}/draft/recommend`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        throw new Error(`Failed to fetch ban recommendations: ${res.status}`);
    }

    return res.json();
}
