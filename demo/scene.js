/** Static office layout + room rendering for the pixel office sim. */

import {
  drawDesk,
  drawServerRack,
  drawGear,
  drawProtocolBoard,
  drawMainline,
  drawRocket,
  drawPlant,
} from "./sprites.js";

export const W = 1000;
export const H = 500;

export const DESK_SLOTS = [
  { x: 100, y: 178 },
  { x: 215, y: 178 },
  { x: 330, y: 178 },
  { x: 445, y: 178 },
  { x: 100, y: 285 },
  { x: 215, y: 285 },
  { x: 330, y: 285 },
  { x: 445, y: 285 },
];

export const ZONE_SPOTS = {
  ci: { x: 640, y: 262, label: "CI LAB" },
  protocol: { x: 872, y: 262, label: "PROTOCOL ROOM" },
  review: { x: 300, y: 442, label: "REVIEW & MERGE" },
  ship: { x: 838, y: 452, label: "SHIPPING" },
};

const ROOMS = [
  { x: 26, y: 86, w: 514, h: 238, fill: "#39415c", name: "ENGINEERING FLOOR" },
  { x: 556, y: 86, w: 206, h: 238, fill: "#2c4a6e", name: "CI LAB" },
  { x: 778, y: 86, w: 196, h: 238, fill: "#4e3161", name: "PROTOCOL ROOM" },
  { x: 26, y: 340, w: 736, h: 140, fill: "#2b5340", name: "MAINLINE" },
  { x: 778, y: 340, w: 196, h: 140, fill: "#5a3c27", name: "SHIPPING" },
];

function skyColors(hour) {
  if (hour >= 5 && hour < 8) return ["#f9a76c", "#ffd9a0"]; // dawn
  if (hour >= 8 && hour < 17) return ["#4da3e8", "#b7e2ff"]; // day
  if (hour >= 17 && hour < 20) return ["#ef6f5c", "#ffc38b"]; // dusk
  return ["#16204a", "#2e3a73"]; // night
}

function drawClouds(ctx, hour, t) {
  if (hour < 5 || hour >= 20) return;
  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < 4; i++) {
    const x = ((t * (6 + i * 3) + i * 320) % (W + 160)) - 80;
    const y = 18 + i * 13;
    ctx.beginPath();
    ctx.ellipse(x, y, 26, 9, 0, 0, Math.PI * 2);
    ctx.ellipse(x + 20, y + 3, 18, 7, 0, 0, Math.PI * 2);
    ctx.ellipse(x - 18, y + 4, 14, 6, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawBackdrop(ctx, hour, t) {
  ctx.fillStyle = "#171c29";
  ctx.fillRect(0, 0, W, H);

  // ---- window wall ----
  const [c1, c2] = skyColors(hour);
  const sky = ctx.createLinearGradient(0, 0, 0, 76);
  sky.addColorStop(0, c1);
  sky.addColorStop(1, c2);
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, 76);

  const isNight = hour < 5 || hour >= 20;

  // sun / moon travelling across the sky
  const dayT = ((hour + 24 - 5) % 24) / 24;
  const sx = 40 + dayT * (W - 80);
  const sy = 58 - Math.sin(dayT * Math.PI * 2) * 34;
  ctx.save();
  ctx.shadowColor = isNight ? "rgba(232,234,246,0.8)" : "rgba(255,224,102,0.9)";
  ctx.shadowBlur = 22;
  ctx.fillStyle = isNight ? "#eef1fb" : "#ffe066";
  ctx.beginPath();
  ctx.arc(sx, Math.max(12, sy), isNight ? 10 : 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (isNight) {
    for (let i = 0; i < 46; i++) {
      const x = (i * 149) % W;
      const y = (i * 37) % 70;
      ctx.globalAlpha = 0.35 + ((i * 7) % 10) / 22;
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, y, 2, 2);
    }
    ctx.globalAlpha = 1;
  }

  drawClouds(ctx, hour, t);

  // window frames: 5 big panes
  ctx.fillStyle = "#11151f";
  const paneW = W / 5;
  for (let i = 0; i <= 5; i++) ctx.fillRect(i * paneW - 5, 0, 10, 76);
  ctx.fillRect(0, 70, W, 8);
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) ctx.strokeRect(i * paneW + 6, 4, paneW - 12, 64);

  // ---- floor ----
  const floor = ctx.createLinearGradient(0, 78, 0, H);
  floor.addColorStop(0, "#2c3446");
  floor.addColorStop(1, "#212836");
  ctx.fillStyle = floor;
  ctx.fillRect(0, 78, W, H - 78);
  ctx.strokeStyle = "rgba(255,255,255,0.045)";
  for (let x = 0; x < W; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 78);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 78; y < H; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // ---- rooms ----
  for (const r of ROOMS) {
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = r.fill;
    ctx.beginPath();
    ctx.roundRect(r.x, r.y, r.w, r.h, 16);
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(r.x, r.y, r.w, r.h, 16);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "bold 11px 'Courier New', monospace";
    ctx.textAlign = "left";
    ctx.fillText(r.name, r.x + 14, r.y + 20);
  }
}

export function drawProps(ctx, state) {
  const { t, hour, busyDesks, ciHeat, notes, merges, rocketLift } = state;
  const night = hour < 7 || hour >= 20;

  DESK_SLOTS.forEach((d, i) => drawDesk(ctx, d.x, d.y, night, busyDesks.has(i)));

  drawPlant(ctx, 512, 316);
  drawPlant(ctx, 50, 316);

  // CI lab
  drawServerRack(ctx, 596, 300, t, ciHeat);
  drawServerRack(ctx, 644, 300, t, ciHeat);
  drawGear(ctx, 712, 176, 26, t * (0.6 + ciHeat * 6), ciHeat > 0.2 ? "#7dd3fc" : "#47566f");
  drawGear(ctx, 712, 228, 17, -t * (0.9 + ciHeat * 7), ciHeat > 0.2 ? "#58a6ff" : "#3d4a61");

  // protocol room
  drawProtocolBoard(ctx, 876, 190, Math.min(notes, 18), t);

  // mainline
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "bold 11px 'Courier New', monospace";
  ctx.textAlign = "left";
  ctx.fillText("main", 44, 404);
  drawMainline(ctx, 84, 742, 400, merges, t);

  // shipping pad
  ctx.fillStyle = "#4a3722";
  ctx.fillRect(812, 466, 56, 7);
  drawRocket(ctx, 840, 466, rocketLift);
}

/** Night dimming with warm pools of light around active people. */
export function drawLighting(ctx, hour, litPoints) {
  const night = hour < 6 || hour >= 20;
  const dusk = hour === 18 || hour === 19 || hour === 5;
  if (!night && !dusk) return;
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = night ? "rgba(52,60,110,0.42)" : "rgba(92,74,118,0.2)";
  ctx.fillRect(0, 78, W, H - 78);
  ctx.restore();

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const p of litPoints) {
    const g = ctx.createRadialGradient(p.x, p.y, 4, p.x, p.y, p.r || 80);
    g.addColorStop(0, "rgba(255,206,130,0.26)");
    g.addColorStop(1, "transparent");
    ctx.fillStyle = g;
    ctx.fillRect(p.x - 100, p.y - 100, 200, 200);
  }
  ctx.restore();
}

export function drawZoneLabels(ctx) {
  ctx.save();
  ctx.font = "bold 9px 'Courier New', monospace";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(255,255,255,0.3)";
  ctx.fillText("CI LAB", ZONE_SPOTS.ci.x, 316);
  ctx.fillText("PROTOCOL", ZONE_SPOTS.protocol.x, 316);
  ctx.fillText("REVIEW & MERGE", ZONE_SPOTS.review.x, 472);
  ctx.fillText("SHIPPING", ZONE_SPOTS.ship.x, 472);
  ctx.restore();
}
