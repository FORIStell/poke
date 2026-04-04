export type PokemonType =
  | "Normal"
  | "Fire"
  | "Water"
  | "Grass"
  | "Electric"
  | "Ice"
  | "Fighting"
  | "Poison"
  | "Ground"
  | "Flying"
  | "Psychic"
  | "Bug"
  | "Rock"
  | "Ghost"
  | "Dragon"
  | "Dark"
  | "Steel"
  | "Fairy";

export type MoveCategory = "Physical" | "Special" | "Status";

export interface BattlePokemon {
  name: string;
  level: number;
  types: PokemonType[];
  attack: number;
  defense: number;
  spAttack: number;
  spDefense: number;
  status?: "BURN" | "PARALYSIS" | "POISON" | "BADLY_POISONED" | "SLEEP" | "FREEZE";
  ability?: string;
  heldItem?: string;
}

export interface Move {
  name: string;
  type: PokemonType;
  category: MoveCategory;
  basePower: number;
  critRateBonus?: number;
}

export interface BattleWeather {
  type: "NONE" | "RAIN" | "SUN" | "SANDSTORM" | "HAIL";
}
