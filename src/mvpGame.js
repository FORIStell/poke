const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const TILE = 16;
const MAP_W = 30;
const MAP_H = 20;
const WIDTH = 480;
const HEIGHT = 320;

const SCENES = {
  TITLE: "TITLE",
  OVERWORLD: "OVERWORLD",
  BATTLE: "BATTLE",
  MESSAGE: "MESSAGE"
};

const map = Array.from({ length: MAP_H }, (_, y) =>
  Array.from({ length: MAP_W }, (_, x) => {
    if (x === 0 || y === 0 || x === MAP_W - 1 || y === MAP_H - 1) return "WALL";
    if ((x > 4 && x < 12 && y > 5 && y < 11) || (x > 16 && x < 27 && y > 2 && y < 8)) return "GRASS";
    return "GROUND";
  })
);

const encounters = [
  { species: "Pidgey", min: 2, max: 4, weight: 55, hp: 18, atk: 10, def: 10, type: "Normal" },
  { species: "Rattata", min: 2, max: 4, weight: 35, hp: 16, atk: 11, def: 8, type: "Normal" },
  { species: "Pikachu", min: 3, max: 5, weight: 10, hp: 17, atk: 12, def: 8, type: "Electric" }
];

const state = {
  scene: SCENES.TITLE,
  message: "",
  messageTimer: 0,
  player: { x: 3, y: 3, facing: "DOWN", steps: 0 },
  party: [
    {
      species: "Charmander",
      level: 5,
      maxHp: 20,
      hp: 20,
      atk: 11,
      def: 10,
      type: "Fire",
      exp: 0,
      toNext: 20,
      balls: 5
    }
  ],
  battle: null
};

const keys = new Set();
window.addEventListener("keydown", (e) => {
  keys.add(e.code);
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) e.preventDefault();
});
window.addEventListener("keyup", (e) => keys.delete(e.code));

function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pickEncounter() {
  const total = encounters.reduce((s, e) => s + e.weight, 0);
  let roll = Math.random() * total;
  for (const e of encounters) {
    roll -= e.weight;
    if (roll <= 0) {
      const level = randInt(e.min, e.max);
      return {
        species: e.species,
        level,
        maxHp: e.hp + level,
        hp: e.hp + level,
        atk: e.atk + Math.floor(level / 2),
        def: e.def + Math.floor(level / 2),
        type: e.type
      };
    }
  }
  return null;
}

function tryMove(dx, dy, facing) {
  const nx = state.player.x + dx;
  const ny = state.player.y + dy;
  state.player.facing = facing;
  if (map[ny]?.[nx] === "WALL") return;
  state.player.x = nx;
  state.player.y = ny;
  state.player.steps += 1;

  if (map[ny][nx] === "GRASS" && Math.random() < 0.12) {
    state.battle = {
      wild: pickEncounter(),
      menu: 0,
      log: "A wild Pokémon appeared!"
    };
    state.scene = SCENES.BATTLE;
  }
}

function showMessage(text, ms = 1300) {
  state.message = text;
  state.messageTimer = ms;
  state.scene = SCENES.MESSAGE;
}

function damage(attacker, defender, power = 40) {
  const base = Math.floor((((2 * 5) / 5 + 2) * power * (attacker.atk / defender.def)) / 50) + 2;
  const rand = randInt(85, 100) / 100;
  return Math.max(1, Math.floor(base * rand));
}

function gainExp(mon, amount) {
  mon.exp += amount;
  while (mon.exp >= mon.toNext) {
    mon.exp -= mon.toNext;
    mon.level += 1;
    mon.maxHp += 3;
    mon.hp = mon.maxHp;
    mon.atk += 1;
    mon.def += 1;
    mon.toNext = Math.floor(mon.toNext * 1.3);
    showMessage(`${mon.species} grew to Lv.${mon.level}!`, 1200);
  }
}

function handleBattleInput() {
  const hero = state.party[0];
  const wild = state.battle.wild;

  if (keys.has("Digit1")) {
    const dealt = damage(hero, wild);
    wild.hp = Math.max(0, wild.hp - dealt);
    state.battle.log = `${hero.species} used Ember! ${dealt} dmg.`;
    if (wild.hp <= 0) {
      const exp = wild.level * 7;
      gainExp(hero, exp);
      showMessage(`You defeated ${wild.species}! +${exp} EXP`);
      state.scene = SCENES.OVERWORLD;
      state.battle = null;
      return;
    }

    const back = damage(wild, hero, 35);
    hero.hp = Math.max(0, hero.hp - back);
    state.battle.log += ` ${wild.species} hit back for ${back}.`;
    if (hero.hp <= 0) {
      hero.hp = 1;
      showMessage("You blacked out! Auto-healed for demo.");
      hero.hp = hero.maxHp;
      state.scene = SCENES.OVERWORLD;
      state.battle = null;
    }
  } else if (keys.has("Digit2")) {
    if (hero.balls <= 0) {
      state.battle.log = "No Poké Balls left!";
      return;
    }
    hero.balls -= 1;
    const hpFactor = (wild.maxHp - wild.hp) / wild.maxHp;
    const chance = 0.25 + hpFactor * 0.55;
    if (Math.random() < chance) {
      showMessage(`Gotcha! ${wild.species} was caught!`);
      state.scene = SCENES.OVERWORLD;
      state.battle = null;
      return;
    }
    state.battle.log = `${wild.species} broke free!`;
  } else if (keys.has("Digit3") || keys.has("Escape") || keys.has("KeyX")) {
    showMessage("Got away safely!");
    state.scene = SCENES.OVERWORLD;
    state.battle = null;
  }
}

function update(dt) {
  if (state.scene === SCENES.TITLE) {
    if (keys.has("Enter") || keys.has("KeyZ")) state.scene = SCENES.OVERWORLD;
    return;
  }

  if (state.scene === SCENES.MESSAGE) {
    state.messageTimer -= dt;
    if (state.messageTimer <= 0) state.scene = SCENES.OVERWORLD;
    return;
  }

  if (state.scene === SCENES.OVERWORLD) {
    if (keys.has("ArrowUp") || keys.has("KeyW")) tryMove(0, -1, "UP");
    else if (keys.has("ArrowDown") || keys.has("KeyS")) tryMove(0, 1, "DOWN");
    else if (keys.has("ArrowLeft") || keys.has("KeyA")) tryMove(-1, 0, "LEFT");
    else if (keys.has("ArrowRight") || keys.has("KeyD")) tryMove(1, 0, "RIGHT");
    return;
  }

  if (state.scene === SCENES.BATTLE) {
    handleBattleInput();
  }
}

function drawOverworld() {
  for (let y = 0; y < MAP_H; y++) {
    for (let x = 0; x < MAP_W; x++) {
      const t = map[y][x];
      if (t === "WALL") ctx.fillStyle = "#2f2f2f";
      else if (t === "GRASS") ctx.fillStyle = "#2f8f2f";
      else ctx.fillStyle = "#7bc67b";
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
    }
  }

  ctx.fillStyle = "#ff4d4d";
  ctx.fillRect(state.player.x * TILE + 3, state.player.y * TILE + 2, 10, 12);

  const hero = state.party[0];
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(0, 0, WIDTH, 20);
  ctx.fillStyle = "#fff";
  ctx.fillText(`Charmander Lv${hero.level} HP ${hero.hp}/${hero.maxHp} Balls:${hero.balls}`, 6, 14);
}

function hpBar(x, y, w, ratio) {
  ctx.fillStyle = "#222";
  ctx.fillRect(x, y, w, 8);
  ctx.fillStyle = ratio > 0.5 ? "#3f3" : ratio > 0.2 ? "#fc3" : "#f44";
  ctx.fillRect(x + 1, y + 1, Math.max(1, (w - 2) * ratio), 6);
}

function drawBattle() {
  const hero = state.party[0];
  const wild = state.battle.wild;

  ctx.fillStyle = "#d6f0ff";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);

  ctx.fillStyle = "#666";
  ctx.beginPath();
  ctx.ellipse(360, 110, 70, 20, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(120, 230, 80, 25, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#f2c94c";
  ctx.fillRect(340, 70, 34, 34);
  ctx.fillStyle = "#f77";
  ctx.fillRect(102, 190, 36, 36);

  ctx.fillStyle = "#111";
  ctx.fillRect(20, 20, 180, 50);
  ctx.fillRect(270, 180, 190, 65);
  ctx.fillStyle = "#fff";
  ctx.fillText(`${wild.species} Lv${wild.level}`, 30, 38);
  hpBar(30, 48, 150, wild.hp / wild.maxHp);

  ctx.fillText(`${hero.species} Lv${hero.level}`, 280, 198);
  hpBar(280, 208, 150, hero.hp / hero.maxHp);
  ctx.fillText(`${hero.hp}/${hero.maxHp}`, 280, 230);

  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillRect(0, 250, WIDTH, 70);
  ctx.fillStyle = "#fff";
  ctx.fillText(state.battle.log, 12, 270);
  ctx.fillText("1:FIGHT  2:BALL  3:RUN", 12, 292);
}

function drawTitle() {
  ctx.fillStyle = "#1f2a44";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "#ffde59";
  ctx.font = "bold 40px monospace";
  ctx.fillText("POKÉ MVP", 112, 140);
  ctx.font = "16px monospace";
  ctx.fillStyle = "#fff";
  ctx.fillText("A tiny playable foundation", 128, 180);
  ctx.fillText("Press Enter / Z", 168, 220);
}

function drawMessage() {
  drawOverworld();
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.fillRect(18, 220, WIDTH - 36, 80);
  ctx.fillStyle = "#fff";
  ctx.fillText(state.message, 30, 262);
}

let last = 0;
function loop(ts) {
  const dt = ts - last;
  last = ts;

  update(dt);

  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.font = "14px monospace";

  if (state.scene === SCENES.TITLE) drawTitle();
  else if (state.scene === SCENES.OVERWORLD) drawOverworld();
  else if (state.scene === SCENES.BATTLE) drawBattle();
  else if (state.scene === SCENES.MESSAGE) drawMessage();

  keys.clear();
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
