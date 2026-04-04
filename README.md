# poke

This repository now contains a **much larger playable browser Pokémon-like RPG foundation** (single-player, offline, no build step required).

## Quick start
1. Open `index.html` in a browser.
2. Press **New Game**.
3. Pick a starter with **Left/Right**, confirm with **Z/Enter**.

## Implemented gameplay systems
- Title screen with New Game / Continue.
- Starter selection (Bulbasaur / Charmander / Squirtle).
- Multi-map overworld:
  - Pallet Town
  - Route 1
  - Pokémon Center
- Tile collision, map warps, and interactable NPCs.
- NPC roles:
  - Story NPC (starter flow)
  - Healer (Nurse Joy / Mom-style full heal)
  - Trainer battle NPC
  - Shop NPC (Pokémart buy flow)
- Wild encounter grass on Route 1.
- Turn-based battle engine with:
  - Fight (move selection)
  - Ball (wild only)
  - Item (Potion)
  - Run (wild only)
- Type effectiveness + STAB in damage.
- Catching + party/PC overflow behavior.
- EXP gain + leveling progression.
- Overworld Start Menu:
  - Pokémon summary
  - Bag summary
  - Save
  - Load
- Save/load via `localStorage`.
- Black-screen hardening:
  - Canvas initialization checks
  - Runtime error overlay + safe fail behavior

## Controls
- **Move:** WASD / Arrow keys
- **Interact/Confirm:** Z / Enter / Space
- **Cancel/Back:** X / Esc / Backspace
- **Open menu:** C or Esc (overworld)
- **Battle:** Navigate with arrows, confirm with Z/Enter

## Important scope note
A fully complete, content-accurate FireRed/HGSS-scale clone (all regions/maps/systems/151 full data + complete move/AI/event parity) is a large multi-phase project. This code now provides a significantly expanded playable base that includes core loops (exploration, interaction, encounter, battle, capture, progression, shopping, healing, save/load) and is structured to continue expanding.
