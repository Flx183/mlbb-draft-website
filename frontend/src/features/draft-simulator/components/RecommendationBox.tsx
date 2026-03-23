import React from "react";

export type RecommendationItem = {
  hero: string;
  score?: number;
  reasons?: string[];
};

type RecommendationBoxProps = {
  team: "blue" | "red";
  recommendations: RecommendationItem[];
  visible: boolean;
};

const RecommendationBox: React.FC<RecommendationBoxProps> = ({
  team,
  recommendations,
  visible,
}) => {
  if (!visible) return null;

  return (
    <div className="w-full rounded-xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent backdrop-blur-md p-3 text-white shadow-lg">
      <div className="recommendation-box__header">
        {team === "blue" ? "Blue Suggestions" : "Red Suggestions"}
      </div>

      {recommendations.length === 0 ? (
        <div className="recommendation-box__empty">
          No recommendations yet
        </div>
      ) : (
        <div className="recommendation-box__list">
          {recommendations.map((rec, index) => (
            <div key={rec.hero} className="recommendation-box__item">
              <div className="recommendation-box__rank">#{index + 1}</div>

              <div className="recommendation-box__content">
                <div className="recommendation-box__hero">
                  {rec.hero}
                </div>

                {typeof rec.score === "number" && (
                  <div className="recommendation-box__score">
                    {rec.score.toFixed(3)}
                  </div>
                )}

                {rec.reasons?.[0] && (
                  <div className="recommendation-box__reason">
                    {rec.reasons[0]}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RecommendationBox;