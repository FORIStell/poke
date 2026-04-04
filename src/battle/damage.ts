import type { BattlePokemon, BattleWeather, Move } from "../data/types.js";
import { getTypeEffectiveness } from "./typeChart.js";

export interface DamageResult {
  damage: number;
  isCritical: boolean;
  effectiveness: number;
}

export function calculateDamage(
  attacker: BattlePokemon,
  defender: BattlePokemon,
  move: Move,
  weather: BattleWeather,
  rng: () => number = Math.random
): DamageResult {
  if (move.category === "Status") return { damage: 0, isCritical: false, effectiveness: 1 };

  const attack = move.category === "Physical" ? attacker.attack : attacker.spAttack;
  const defense = move.category === "Physical" ? defender.defense : defender.spDefense;

  let damage = Math.floor(((2 * attacker.level) / 5 + 2) * move.basePower * (attack / defense) / 50) + 2;

  if (weather.type === "SUN") {
    if (move.type === "Fire") damage = Math.floor(damage * 1.5);
    if (move.type === "Water") damage = Math.floor(damage * 0.5);
  }
  if (weather.type === "RAIN") {
    if (move.type === "Water") damage = Math.floor(damage * 1.5);
    if (move.type === "Fire") damage = Math.floor(damage * 0.5);
  }

  const critStage = Math.min(move.critRateBonus ?? 0, 3);
  const critChance = [1 / 24, 1 / 8, 1 / 2, 1][critStage];
  const isCritical = rng() < critChance;
  if (isCritical) damage = Math.floor(damage * 1.5);

  const randomFactor = Math.floor(rng() * 16 + 85) / 100;
  damage = Math.floor(damage * randomFactor);

  if (attacker.types.includes(move.type)) {
    damage = Math.floor(damage * (attacker.ability === "Adaptability" ? 2 : 1.5));
  }

  const effectiveness = getTypeEffectiveness(move.type, defender.types);
  damage = Math.floor(damage * effectiveness);

  if (attacker.status === "BURN" && move.category === "Physical" && attacker.ability !== "Guts") {
    damage = Math.floor(damage * 0.5);
  }

  if (defender.ability === "Thick Fat" && (move.type === "Fire" || move.type === "Ice")) {
    damage = Math.floor(damage * 0.5);
  }

  return { damage: Math.max(1, damage), isCritical, effectiveness };
}
