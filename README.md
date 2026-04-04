# poke

A **playable browser MVP** for a Pokémon-style RPG now exists in this repository.

## Run
Open `index.html` in a browser.

## What is implemented now
- Title screen and scene transitions.
- Tile-based overworld with collision and grass tiles.
- Player movement with keyboard controls (WASD / arrows).
- Random wild encounters from grass.
- Turn-based battle scene with:
  - Fight action
  - Poké Ball catch attempt
  - Run action
- Basic EXP and level-up loop for the starter.
- HUD and message box flow.

## Files
- `index.html` — bootstraps canvas and controls hint.
- `src/mvpGame.js` — self-contained game loop + overworld + battle systems.

## Scope note
Your mega prompt specifies a full-scale game (maps, gyms, 151 dex entries, full move pool, AI tiers, HM systems, etc.).
That is a multi-phase production effort. This commit upgrades the repository from non-playable scaffolding to a directly playable MVP foundation that can be expanded module-by-module.
