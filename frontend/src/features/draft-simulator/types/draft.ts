// Role type (strongly typed instead of string)
export type Role =
  | "Tank"
  | "Fighter"
  | "Assassin"
  | "Mage"
  | "Marksman"
  | "Support";

// Hero model
export interface Hero {
  id: number;
  name: string;
  role: Role[];     // supports multiple roles
  image: string;
}

// Draft phase name
export type DraftPhase = "ban1" | "pick1";

// Draft action
export type DraftAction = "ban" | "pick";

// Team type
export type Team = "blue" | "red";

// One step in the draft order
export interface DraftStep {
  phase: DraftPhase;
  team: Team;
  action: DraftAction;
}