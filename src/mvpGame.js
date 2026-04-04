const canvas = document.getElementById("game");
const ctx = canvas?.getContext?.("2d");

if (!canvas || !ctx) {
  const msg = document.createElement("pre");
  msg.textContent = "Failed to initialize canvas. Ensure your browser supports Canvas 2D.";
  msg.style.color = "#fff";
  document.body.appendChild(msg);
  throw new Error("Canvas init failed");
}

const WIDTH = 480;
const HEIGHT = 320;
const TILE = 16;

const Scenes = {
  TITLE: "TITLE",
  INTRO: "INTRO",
  OVERWORLD: "OVERWORLD",
  DIALOG: "DIALOG",
  MENU: "MENU",
  BATTLE: "BATTLE",
  MESSAGE: "MESSAGE",
  SHOP: "SHOP"
};

const TypeChart = {
  Normal: { Rock: 0.5, Ghost: 0 },
  Fire: { Fire: 0.5, Water: 0.5, Grass: 2 },
  Water: { Fire: 2, Water: 0.5, Grass: 0.5 },
  Grass: { Fire: 0.5, Water: 2, Grass: 0.5 },
  Electric: { Water: 2, Grass: 0.5, Electric: 0.5, Ground: 0 },
  Ground: { Fire: 2, Electric: 2, Grass: 0.5 }
};

const starterChoices = [
  { species: "Bulbasaur", type: "Grass", hp: 21, atk: 11, def: 11, moves: [{ name: "Tackle", type: "Normal", power: 40 }, { name: "Vine Whip", type: "Grass", power: 45 }] },
  { species: "Charmander", type: "Fire", hp: 20, atk: 12, def: 10, moves: [{ name: "Scratch", type: "Normal", power: 40 }, { name: "Ember", type: "Fire", power: 40 }] },
  { species: "Squirtle", type: "Water", hp: 22, atk: 10, def: 12, moves: [{ name: "Tackle", type: "Normal", power: 40 }, { name: "Water Gun", type: "Water", power: 40 }] }
];

const speciesData = {
  Pidgey: { type: "Normal", hp: 16, atk: 10, def: 10, catchRate: 0.5, expYield: 45, moves: [{ name: "Tackle", type: "Normal", power: 40 }] },
  Rattata: { type: "Normal", hp: 15, atk: 11, def: 9, catchRate: 0.55, expYield: 51, moves: [{ name: "Tackle", type: "Normal", power: 40 }] },
  Pikachu: { type: "Electric", hp: 17, atk: 12, def: 8, catchRate: 0.35, expYield: 112, moves: [{ name: "Quick Attack", type: "Normal", power: 40 }, { name: "Thunder Shock", type: "Electric", power: 40 }] }
};

const maps = {
  pallet_town: {
    id: "pallet_town",
    name: "Pallet Town",
    width: 30,
    height: 20,
    terrain(x, y) {
      if (x === 0 || y === 0 || x === 29 || y === 19) return "BLOCKED";
      if (x > 9 && x < 20 && y > 7 && y < 11) return "PATH";
      return "WALKABLE";
    },
    npcs: [
      { id: "oak", x: 14, y: 8, kind: "story", name: "Prof. Oak", text: ["Welcome, trainer!", "Pick a starter and begin!"], action: { type: "starter" } },
      { id: "mom", x: 6, y: 15, kind: "healer", name: "Mom", text: ["Take care!", "I healed your Pokémon."] }
    ],
    warps: [{ x: 24, y: 9, to: { map: "route_1", x: 1, y: 9 } }],
    grassRate: 0,
    encounters: []
  },
  route_1: {
    id: "route_1",
    name: "Route 1",
    width: 30,
    height: 20,
    terrain(x, y) {
      if (x === 0 || y === 0 || x === 29 || y === 19) return "BLOCKED";
      if ((x > 4 && x < 12 && y > 3 && y < 15) || (x > 17 && x < 27 && y > 4 && y < 14)) return "GRASS";
      return "WALKABLE";
    },
    npcs: [
      { id: "bugboy", x: 15, y: 9, kind: "trainer", name: "Bug Catcher", text: ["Let's battle!"], defeatedText: ["You got me..."], team: [{ species: "Pidgey", level: 4 }] },
      { id: "martman", x: 25, y: 16, kind: "shop", name: "Merchant", text: ["Need supplies?"], shop: [{ item: "Potion", price: 300 }, { item: "Poké Ball", price: 200 }, { item: "Antidote", price: 100 }] }
    ],
    warps: [
      { x: 1, y: 9, to: { map: "pallet_town", x: 23, y: 9 } },
      { x: 28, y: 9, to: { map: "center", x: 2, y: 5 } }
    ],
    grassRate: 0.14,
    encounters: [
      { species: "Pidgey", min: 2, max: 4, weight: 50 },
      { species: "Rattata", min: 2, max: 4, weight: 40 },
      { species: "Pikachu", min: 3, max: 5, weight: 10 }
    ]
  },
  center: {
    id: "center",
    name: "Pokémon Center",
    width: 15,
    height: 10,
    terrain(x, y) {
      if (x === 0 || y === 0 || x === 14 || y === 9) return "BLOCKED";
      return "WALKABLE";
    },
    npcs: [{ id: "joy", x: 7, y: 2, kind: "healer", name: "Nurse Joy", text: ["Welcome!", "We restored your Pokémon."] }],
    warps: [{ x: 2, y: 9, to: { map: "route_1", x: 27, y: 9 } }],
    grassRate: 0,
    encounters: []
  }
};

const keyMap = {
  up: ["ArrowUp", "KeyW"],
  down: ["ArrowDown", "KeyS"],
  left: ["ArrowLeft", "KeyA"],
  right: ["ArrowRight", "KeyD"],
  a: ["Enter", "KeyZ", "Space"],
  b: ["Escape", "KeyX", "Backspace"],
  menu: ["KeyC", "Escape"],
  one: ["Digit1"],
  two: ["Digit2"],
  three: ["Digit3"],
  four: ["Digit4"]
};

const inputDown = new Set();
const inputPressed = new Set();
window.addEventListener("keydown", (e) => {
  if (!inputDown.has(e.code)) inputPressed.add(e.code);
  inputDown.add(e.code);
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) e.preventDefault();
});
window.addEventListener("keyup", (e) => inputDown.delete(e.code));

function actionDown(action) {
  return keyMap[action].some((k) => inputDown.has(k));
}
function actionPressed(action) {
  return keyMap[action].some((k) => inputPressed.has(k));
}

const game = {
  scene: Scenes.TITLE,
  mapId: "pallet_town",
  player: { x: 14, y: 14, facing: "DOWN" },
  trainerName: "RED",
  rivalName: "BLUE",
  money: 3000,
  badges: Array(8).fill(false),
  storyFlags: { gotStarter: false, gotPokedex: false },
  party: [],
  bag: { "Potion": 3, "Poké Ball": 8, "Antidote": 1 },
  pc: [],
  seenTrainers: {},
  dialog: { pages: [], index: 0, actor: null },
  menuIndex: 0,
  shopIndex: 0,
  battle: null,
  message: "",
  messageTimer: 0,
  titleChoice: 1,
  starterChoice: 1
};

function currentMap() {
  return maps[game.mapId];
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function typeEff(moveType, targetType) {
  return TypeChart[moveType]?.[targetType] ?? 1;
}

function mkPokemon(species, level) {
  const src = speciesData[species] ?? starterChoices.find((s) => s.species === species);
  const maxHp = src.hp + level * 2;
  return {
    species,
    level,
    type: src.type,
    hp: maxHp,
    maxHp,
    atk: src.atk + level,
    def: src.def + level,
    exp: 0,
    toNext: Math.max(20, Math.floor(level * 20)),
    status: null,
    moves: src.moves.slice(0, 4)
  };
}

function drawTextBox(lines) {
  ctx.fillStyle = "rgba(0,0,0,0.76)";
  ctx.fillRect(8, HEIGHT - 84, WIDTH - 16, 76);
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(8, HEIGHT - 84, WIDTH - 16, 76);
  ctx.fillStyle = "#fff";
  lines.forEach((line, i) => ctx.fillText(line, 16, HEIGHT - 60 + i * 18));
}

function setMessage(msg, ms = 1100) {
  game.message = msg;
  game.messageTimer = ms;
  game.scene = Scenes.MESSAGE;
}

function healParty() {
  game.party.forEach((p) => {
    p.hp = p.maxHp;
    p.status = null;
  });
}

function movePlayer(dx, dy, facing) {
  const map = currentMap();
  const nx = game.player.x + dx;
  const ny = game.player.y + dy;
  game.player.facing = facing;

  if (map.terrain(nx, ny) === "BLOCKED") return;

  const npcAt = map.npcs.find((n) => n.x === nx && n.y === ny && !n.hidden);
  if (npcAt) return;

  game.player.x = nx;
  game.player.y = ny;

  const warp = map.warps.find((w) => w.x === nx && w.y === ny);
  if (warp) {
    game.mapId = warp.to.map;
    game.player.x = warp.to.x;
    game.player.y = warp.to.y;
    return;
  }

  if (map.terrain(nx, ny) === "GRASS" && Math.random() < map.grassRate) {
    const wild = rollEncounter(map.encounters);
    if (wild) startBattle({ mode: "wild", enemyName: wild.species, enemyTeam: [mkPokemon(wild.species, wild.level)], reward: 0, trainerId: null });
  }
}

function rollEncounter(table) {
  if (!table.length) return null;
  const total = table.reduce((s, e) => s + e.weight, 0);
  let roll = Math.random() * total;
  for (const e of table) {
    roll -= e.weight;
    if (roll <= 0) return { species: e.species, level: randomInt(e.min, e.max) };
  }
  return null;
}

function tryInteract() {
  const map = currentMap();
  const offs = game.player.facing === "UP" ? [0, -1] : game.player.facing === "DOWN" ? [0, 1] : game.player.facing === "LEFT" ? [-1, 0] : [1, 0];
  const tx = game.player.x + offs[0];
  const ty = game.player.y + offs[1];
  const npc = map.npcs.find((n) => n.x === tx && n.y === ty && !n.hidden);
  if (!npc) return;

  if (npc.kind === "trainer" && !game.seenTrainers[npc.id]) {
    game.seenTrainers[npc.id] = true;
    openDialog(npc, npc.text, () => {
      startBattle({
        mode: "trainer",
        enemyName: npc.name,
        enemyTeam: npc.team.map((m) => mkPokemon(m.species, m.level)),
        reward: 120,
        trainerId: npc.id,
        onWin: () => openDialog(npc, npc.defeatedText || ["..."])
      });
    });
    return;
  }

  if (npc.kind === "healer") {
    openDialog(npc, npc.text, () => {
      healParty();
      setMessage("Your party is fully healed!");
    });
    return;
  }

  if (npc.kind === "shop") {
    openDialog(npc, npc.text, () => {
      game.scene = Scenes.SHOP;
      game.shopIndex = 0;
      game.shopNpc = npc;
    });
    return;
  }

  if (npc.kind === "story" && npc.action?.type === "starter" && !game.storyFlags.gotStarter) {
    game.scene = Scenes.INTRO;
    return;
  }

  openDialog(npc, npc.text);
}

function openDialog(actor, pages, onClose) {
  game.scene = Scenes.DIALOG;
  game.dialog = { actor, pages, index: 0, onClose };
}

function closeDialog() {
  const cb = game.dialog.onClose;
  game.dialog = { pages: [], index: 0, actor: null };
  game.scene = Scenes.OVERWORLD;
  if (cb) cb();
}

function startBattle({ mode, enemyName, enemyTeam, reward, trainerId, onWin }) {
  if (!game.party.length) return;
  game.scene = Scenes.BATTLE;
  game.battle = {
    mode,
    enemyName,
    enemyTeam,
    enemyActive: 0,
    playerActive: 0,
    reward,
    trainerId,
    menu: "MAIN",
    cursor: 0,
    log: `${mode === "wild" ? "A wild" : enemyName} ${mode === "wild" ? "appeared" : "challenges you"}!`,
    onWin
  };
}

function calcDamage(attacker, defender, move) {
  const base = Math.floor((((2 * attacker.level) / 5 + 2) * move.power * (attacker.atk / defender.def)) / 50) + 2;
  const stab = attacker.type === move.type ? 1.5 : 1;
  const eff = typeEff(move.type, defender.type);
  const rand = randomInt(85, 100) / 100;
  return { dmg: Math.max(1, Math.floor(base * stab * eff * rand)), eff };
}

function giveExp(mon, defeated) {
  const amount = Math.floor((defeated.level * (speciesData[defeated.species]?.expYield ?? 60)) / 7);
  mon.exp += amount;
  while (mon.exp >= mon.toNext) {
    mon.exp -= mon.toNext;
    mon.level += 1;
    mon.maxHp += 3;
    mon.hp = mon.maxHp;
    mon.atk += 1;
    mon.def += 1;
    mon.toNext = Math.floor(mon.toNext * 1.2);
    setMessage(`${mon.species} reached Lv.${mon.level}!`);
  }
}

function tryCatch(enemy) {
  if ((game.bag["Poké Ball"] ?? 0) <= 0) {
    game.battle.log = "No Poké Balls left!";
    return false;
  }
  game.bag["Poké Ball"] -= 1;
  const hpFactor = (enemy.maxHp - enemy.hp) / enemy.maxHp;
  const base = (speciesData[enemy.species]?.catchRate ?? 0.35) + hpFactor * 0.45;
  const caught = Math.random() < Math.min(0.95, base);
  if (caught) {
    if (game.party.length < 6) {
      game.party.push(mkPokemon(enemy.species, enemy.level));
      setMessage(`Gotcha! ${enemy.species} joined your party!`);
    } else {
      game.pc.push(mkPokemon(enemy.species, enemy.level));
      setMessage(`Gotcha! ${enemy.species} was sent to PC.`);
    }
    game.scene = Scenes.OVERWORLD;
    game.battle = null;
    return true;
  }
  game.battle.log = `${enemy.species} broke free!`;
  return false;
}

function enemyTurn() {
  const b = game.battle;
  const enemy = b.enemyTeam[b.enemyActive];
  const hero = game.party[b.playerActive];
  const move = enemy.moves[randomInt(0, enemy.moves.length - 1)];
  const { dmg } = calcDamage(enemy, hero, move);
  hero.hp = Math.max(0, hero.hp - dmg);
  b.log += ` ${enemy.species} used ${move.name} (${dmg}).`;
  if (hero.hp <= 0) {
    healParty();
    game.money = Math.max(0, game.money - 100);
    game.mapId = "center";
    game.player.x = 3;
    game.player.y = 8;
    setMessage("You blacked out! Returned to Center.");
    game.battle = null;
    game.scene = Scenes.OVERWORLD;
  }
}

function handleBattleInput() {
  const b = game.battle;
  const enemy = b.enemyTeam[b.enemyActive];
  const hero = game.party[b.playerActive];

  if (b.menu === "MAIN") {
    if (actionPressed("up")) b.cursor = (b.cursor + 3) % 4;
    if (actionPressed("down")) b.cursor = (b.cursor + 1) % 4;
    if (actionPressed("a")) {
      if (b.cursor === 0) b.menu = "FIGHT";
      if (b.cursor === 1 && b.mode === "wild") {
        if (!tryCatch(enemy)) enemyTurn();
      }
      if (b.cursor === 2) {
        const healed = Math.min(hero.maxHp, hero.hp + 20);
        if ((game.bag.Potion ?? 0) > 0) {
          game.bag.Potion -= 1;
          hero.hp = healed;
          b.log = `${hero.species} used a Potion.`;
          enemyTurn();
        } else b.log = "No Potion left.";
      }
      if (b.cursor === 3) {
        if (b.mode === "wild") {
          setMessage("Got away safely!");
          game.scene = Scenes.OVERWORLD;
          game.battle = null;
        } else {
          b.log = "Can't run from trainer battle!";
        }
      }
    }
  } else if (b.menu === "FIGHT") {
    if (actionPressed("up")) b.cursor = (b.cursor + hero.moves.length - 1) % hero.moves.length;
    if (actionPressed("down")) b.cursor = (b.cursor + 1) % hero.moves.length;
    if (actionPressed("b")) {
      b.menu = "MAIN";
      b.cursor = 0;
    }
    if (actionPressed("a")) {
      const move = hero.moves[b.cursor];
      const { dmg, eff } = calcDamage(hero, enemy, move);
      enemy.hp = Math.max(0, enemy.hp - dmg);
      b.log = `${hero.species} used ${move.name}! (${dmg})`;
      if (eff > 1) b.log += " Super effective!";
      if (eff < 1) b.log += " Not very effective.";

      if (enemy.hp <= 0) {
        giveExp(hero, enemy);
        const next = b.enemyActive + 1;
        if (next < b.enemyTeam.length) {
          b.enemyActive = next;
          b.menu = "MAIN";
          b.cursor = 0;
          b.log = `${b.enemyName} sent out ${b.enemyTeam[next].species}!`;
        } else {
          if (b.mode === "trainer") game.money += b.reward;
          setMessage(`Victory! You won ${b.reward}$`);
          const onWin = b.onWin;
          game.scene = Scenes.OVERWORLD;
          game.battle = null;
          if (onWin) onWin();
        }
        return;
      }

      b.menu = "MAIN";
      b.cursor = 0;
      enemyTurn();
    }
  }
}

function saveGame() {
  localStorage.setItem("poke_mvp_save", JSON.stringify(game));
  setMessage("Game saved!");
}

function loadGame() {
  const raw = localStorage.getItem("poke_mvp_save");
  if (!raw) {
    setMessage("No save file found.");
    return;
  }
  const parsed = JSON.parse(raw);
  Object.assign(game, parsed);
  game.scene = Scenes.OVERWORLD;
  setMessage("Game loaded.");
}

function update(dt) {
  if (game.scene === Scenes.TITLE) {
    if (actionPressed("left")) game.titleChoice = 0;
    if (actionPressed("right")) game.titleChoice = 1;
    if (actionPressed("a")) {
      if (game.titleChoice === 0) game.scene = Scenes.INTRO;
      else loadGame();
    }
    return;
  }

  if (game.scene === Scenes.INTRO) {
    if (!game.storyFlags.gotStarter) {
      if (actionPressed("left")) game.starterChoice = (game.starterChoice + 2) % 3;
      if (actionPressed("right")) game.starterChoice = (game.starterChoice + 1) % 3;
      if (actionPressed("a")) {
        const chosen = starterChoices[game.starterChoice];
        game.party = [mkPokemon(chosen.species, 5)];
        game.storyFlags.gotStarter = true;
        game.storyFlags.gotPokedex = true;
        setMessage(`You chose ${chosen.species}!`);
        game.scene = Scenes.OVERWORLD;
      }
    }
    return;
  }

  if (game.scene === Scenes.MESSAGE) {
    game.messageTimer -= dt;
    if (game.messageTimer <= 0 || actionPressed("a")) game.scene = Scenes.OVERWORLD;
    return;
  }

  if (game.scene === Scenes.DIALOG) {
    if (actionPressed("a")) {
      game.dialog.index += 1;
      if (game.dialog.index >= game.dialog.pages.length) closeDialog();
    }
    if (actionPressed("b")) closeDialog();
    return;
  }

  if (game.scene === Scenes.SHOP) {
    const items = game.shopNpc.shop;
    if (actionPressed("up")) game.shopIndex = (game.shopIndex + items.length - 1) % items.length;
    if (actionPressed("down")) game.shopIndex = (game.shopIndex + 1) % items.length;
    if (actionPressed("a")) {
      const s = items[game.shopIndex];
      if (game.money >= s.price) {
        game.money -= s.price;
        game.bag[s.item] = (game.bag[s.item] ?? 0) + 1;
      } else setMessage("Not enough money.");
    }
    if (actionPressed("b")) game.scene = Scenes.OVERWORLD;
    return;
  }

  if (game.scene === Scenes.BATTLE) {
    handleBattleInput();
    return;
  }

  if (game.scene === Scenes.MENU) {
    const opts = ["POKEMON", "BAG", "SAVE", "LOAD", "EXIT"];
    if (actionPressed("up")) game.menuIndex = (game.menuIndex + opts.length - 1) % opts.length;
    if (actionPressed("down")) game.menuIndex = (game.menuIndex + 1) % opts.length;
    if (actionPressed("a")) {
      const selected = opts[game.menuIndex];
      if (selected === "SAVE") saveGame();
      else if (selected === "LOAD") loadGame();
      else if (selected === "EXIT") game.scene = Scenes.OVERWORLD;
      else if (selected === "POKEMON") {
        const p = game.party[0];
        setMessage(p ? `${p.species} Lv${p.level} HP:${p.hp}/${p.maxHp}` : "No Pokémon");
      } else if (selected === "BAG") {
        const balls = game.bag["Poké Ball"] ?? 0;
        const potions = game.bag.Potion ?? 0;
        setMessage(`Bag: Ball x${balls}, Potion x${potions}`);
      }
    }
    if (actionPressed("b")) game.scene = Scenes.OVERWORLD;
    return;
  }

  if (game.scene === Scenes.OVERWORLD) {
    if (!game.storyFlags.gotStarter) {
      game.scene = Scenes.INTRO;
      return;
    }

    if (actionPressed("menu")) {
      game.scene = Scenes.MENU;
      return;
    }

    if (actionPressed("a")) tryInteract();
    else if (actionPressed("up")) movePlayer(0, -1, "UP");
    else if (actionPressed("down")) movePlayer(0, 1, "DOWN");
    else if (actionPressed("left")) movePlayer(-1, 0, "LEFT");
    else if (actionPressed("right")) movePlayer(1, 0, "RIGHT");
  }
}

function drawMap() {
  const map = currentMap();
  for (let y = 0; y < map.height; y++) {
    for (let x = 0; x < map.width; x++) {
      const t = map.terrain(x, y);
      if (t === "BLOCKED") ctx.fillStyle = map.id === "center" ? "#9ba4b0" : "#2f2f2f";
      else if (t === "GRASS") ctx.fillStyle = "#228b22";
      else if (t === "PATH") ctx.fillStyle = "#be9b66";
      else ctx.fillStyle = map.id === "center" ? "#d8dce2" : "#79c879";
      ctx.fillRect(x * TILE, y * TILE, TILE, TILE);
    }
  }

  for (const n of map.npcs) {
    if (n.hidden) continue;
    ctx.fillStyle = n.kind === "trainer" ? "#ffb347" : n.kind === "healer" ? "#f06292" : n.kind === "shop" ? "#42a5f5" : "#ffffff";
    ctx.fillRect(n.x * TILE + 3, n.y * TILE + 2, 10, 12);
  }

  ctx.fillStyle = "#ff4d4d";
  ctx.fillRect(game.player.x * TILE + 3, game.player.y * TILE + 2, 10, 12);
}

function drawHud() {
  const lead = game.party[0];
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(0, 0, WIDTH, 20);
  ctx.fillStyle = "#fff";
  const hp = lead ? `${lead.hp}/${lead.maxHp}` : "--";
  ctx.fillText(`${currentMap().name}  $${game.money}  ${lead ? `${lead.species} Lv${lead.level} HP ${hp}` : "No Pokémon"}`, 6, 14);
}

function drawBattle() {
  const b = game.battle;
  const hero = game.party[b.playerActive];
  const enemy = b.enemyTeam[b.enemyActive];

  ctx.fillStyle = "#dcecff";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);

  ctx.fillStyle = "#6b6b6b";
  ctx.beginPath();
  ctx.ellipse(350, 100, 70, 18, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(110, 230, 85, 24, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#f7d36e";
  ctx.fillRect(332, 66, 34, 34);
  ctx.fillStyle = "#f88379";
  ctx.fillRect(92, 192, 38, 38);

  ctx.fillStyle = "#111";
  ctx.fillRect(20, 20, 185, 48);
  ctx.fillRect(265, 172, 195, 72);
  ctx.fillStyle = "#fff";
  ctx.fillText(`${enemy.species} Lv${enemy.level}`, 30, 36);
  ctx.fillText(`HP ${enemy.hp}/${enemy.maxHp}`, 30, 56);
  ctx.fillText(`${hero.species} Lv${hero.level}`, 275, 190);
  ctx.fillText(`HP ${hero.hp}/${hero.maxHp}`, 275, 210);

  ctx.fillStyle = "rgba(0,0,0,0.78)";
  ctx.fillRect(8, 248, WIDTH - 16, 64);
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(8, 248, WIDTH - 16, 64);
  ctx.fillStyle = "#fff";
  ctx.fillText(b.log, 16, 268);

  if (b.menu === "MAIN") {
    const opts = ["FIGHT", b.mode === "wild" ? "BALL" : "BLOCK", "ITEM", "RUN"];
    opts.forEach((o, i) => {
      ctx.fillText(`${b.cursor === i ? "▶" : " "} ${o}`, 18 + (i % 2) * 120, 288 + Math.floor(i / 2) * 16);
    });
  } else {
    hero.moves.forEach((m, i) => ctx.fillText(`${b.cursor === i ? "▶" : " "} ${m.name}`, 16, 286 + i * 14));
  }
}

function drawTitle() {
  ctx.fillStyle = "#1d2740";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "#ffde59";
  ctx.font = "bold 42px monospace";
  ctx.fillText("POKÉ QUEST", 106, 120);
  ctx.font = "14px monospace";
  ctx.fillStyle = "#fff";
  ctx.fillText("Playable mini RPG foundation", 128, 150);
  ctx.fillText(`${game.titleChoice === 0 ? "▶" : " "} NEW GAME`, 176, 200);
  ctx.fillText(`${game.titleChoice === 1 ? "▶" : " "} CONTINUE`, 176, 220);
}

function drawIntro() {
  ctx.fillStyle = "#09111f";
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "#fff";
  ctx.fillText("Choose your starter Pokémon", 142, 60);

  starterChoices.forEach((s, i) => {
    const x = 60 + i * 140;
    ctx.fillStyle = i === game.starterChoice ? "#ffd54f" : "#263238";
    ctx.fillRect(x, 100, 120, 120);
    ctx.fillStyle = "#fff";
    ctx.fillText(s.species, x + 16, 130);
    ctx.fillText(`Type: ${s.type}`, x + 16, 150);
    ctx.fillText(`HP:${s.hp} ATK:${s.atk}`, x + 16, 170);
    ctx.fillText(`${i === game.starterChoice ? "<" : ""}`, x + 52, 200);
  });
  drawTextBox(["Use ←/→ to choose starter.", "Press Z/Enter to confirm."]);
}

function drawDialog() {
  drawMap();
  drawHud();
  const who = game.dialog.actor?.name ?? "NPC";
  const line = game.dialog.pages[game.dialog.index] ?? "...";
  drawTextBox([`${who}:`, line]);
}

function drawMenu() {
  drawMap();
  drawHud();
  const opts = ["POKEMON", "BAG", "SAVE", "LOAD", "EXIT"];
  ctx.fillStyle = "rgba(0,0,0,0.82)";
  ctx.fillRect(320, 30, 145, 130);
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(320, 30, 145, 130);
  ctx.fillStyle = "#fff";
  opts.forEach((o, i) => ctx.fillText(`${i === game.menuIndex ? "▶" : " "} ${o}`, 332, 52 + i * 22));
}

function drawShop() {
  drawMap();
  drawHud();
  const items = game.shopNpc.shop;
  ctx.fillStyle = "rgba(0,0,0,0.85)";
  ctx.fillRect(40, 40, 400, 220);
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(40, 40, 400, 220);
  ctx.fillStyle = "#fff";
  ctx.fillText("Pokémart", 52, 62);
  items.forEach((it, i) => ctx.fillText(`${i === game.shopIndex ? "▶" : " "} ${it.item} $${it.price}`, 52, 90 + i * 20));
  ctx.fillText(`Money: $${game.money}`, 52, 240);
  ctx.fillText("Z: Buy  X: Exit", 280, 240);
}

function drawMessageScene() {
  drawMap();
  drawHud();
  drawTextBox([game.message]);
}

function render() {
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.font = "13px monospace";
  if (game.scene === Scenes.TITLE) drawTitle();
  else if (game.scene === Scenes.INTRO) drawIntro();
  else if (game.scene === Scenes.OVERWORLD) {
    drawMap();
    drawHud();
  } else if (game.scene === Scenes.DIALOG) drawDialog();
  else if (game.scene === Scenes.MENU) drawMenu();
  else if (game.scene === Scenes.BATTLE) drawBattle();
  else if (game.scene === Scenes.MESSAGE) drawMessageScene();
  else if (game.scene === Scenes.SHOP) drawShop();
}

let last = performance.now();
function loop(now) {
  try {
    const dt = now - last;
    last = now;
    update(dt);
    render();
  } catch (err) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = "#f66";
    ctx.font = "12px monospace";
    ctx.fillText("Runtime error. Open devtools console.", 20, 40);
    console.error(err);
    return;
  } finally {
    inputPressed.clear();
  }
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
