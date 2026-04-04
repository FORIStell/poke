import { describe, expect, it } from "vitest";
import { calculateDamage } from "../src/battle/damage.js";
import { getTypeEffectiveness } from "../src/battle/typeChart.js";

describe("type chart", () => {
  it("applies super effective and immunity", () => {
    expect(getTypeEffectiveness("Electric", ["Water", "Flying"])) .toBe(4);
    expect(getTypeEffectiveness("Normal", ["Ghost"])) .toBe(0);
  });
});

describe("damage", () => {
  it("returns positive damage with STAB and effectiveness", () => {
    const result = calculateDamage(
      {
        name: "Charmander",
        level: 10,
        types: ["Fire"],
        attack: 52,
        defense: 43,
        spAttack: 60,
        spDefense: 50
      },
      {
        name: "Bulbasaur",
        level: 10,
        types: ["Grass", "Poison"],
        attack: 49,
        defense: 49,
        spAttack: 65,
        spDefense: 65
      },
      { name: "Ember", type: "Fire", category: "Special", basePower: 40 },
      { type: "NONE" },
      () => 0.99
    );

    expect(result.effectiveness).toBe(2);
    expect(result.damage).toBeGreaterThan(1);
  });
});
