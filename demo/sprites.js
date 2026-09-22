/** Procedural pixel-art drawing helpers for the office sim. */

export const PALETTE = {
  paul: { hair: "#4a3b2a", shirt: "#e05263", skin: "#f2c9a0" },
  victor: { hair: "#2b2b35", shirt: "#f0a202", skin: "#d9a066" },
  calvin: { hair: "#8a5a2b", shirt: "#3aa6f0", skin: "#f2c9a0" },
  scarlett: { hair: "#b5452f", shirt: "#a56ff0", skin: "#f7d7b8" },
  sean: { hair: "#1f2933", shirt: "#3ec07a", skin: "#c68642" },
  skitty: { hair: "#e8c07d", shirt: "#ff7b72", skin: "#f7d7b8" },
  iris: { hair: "#5b3a8e", shirt: "#7ad0e8", skin: "#e8b88a" },
  will: { hair: "#3d2b1f", shirt: "#ffa657", skin: "#d9a066" },
};

const PX = 3; // pixel scale

function px(ctx, x, y, w, h, color) {
  ctx.fillStyle = color;
  ctx.fillRect(Math.round(x), Math.round(y), w * PX, h * PX);
}

/**
 * Draw a little office worker.
 * `walkPhase` drives the leg cycle, `bob` the idle breathing.
 */
export function drawPerson(ctx, agent, x, y, opts = {}) {
  const { facing = 1, walkPhase = 0, moving = false, glow = null } = opts;
  const pal = PALETTE[agent] || { hair: "#333", shirt: "#888", skin: "#e0b080" };
  const bob = moving ? Math.sin(walkPhase * 2) * 1.2 : Math.sin(walkPhase * 0.6) * 0.8;
  const baseX = x - 5 * PX;
  const baseY = y - 20 * PX + bob;

  // shadow
  ctx.save();
  ctx.globalAlpha = 0.28;
  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.ellipse(x, y + 2, 11, 4, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (glow) {
    ctx.save();
    const g = ctx.createRadialGradient(x, y - 22, 2, x, y - 22, 34);
    g.addColorStop(0, glow);
    g.addColorStop(1, "transparent");
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = g;
    ctx.fillRect(x - 36, y - 58, 72, 72);
    ctx.restore();
  }

  // legs
  const swing = moving ? Math.sin(walkPhase) * PX : 0;
  px(ctx, baseX + 2 * PX, baseY + 15 * PX + swing, 2, 5, "#2f3b52");
  px(ctx, baseX + 6 * PX, baseY + 15 * PX - swing, 2, 5, "#2f3b52");
  // shoes
  px(ctx, baseX + 2 * PX, baseY + 19 * PX + swing, 2, 1, "#161b22");
  px(ctx, baseX + 6 * PX, baseY + 19 * PX - swing, 2, 1, "#161b22");

  // torso
  px(ctx, baseX + 1 * PX, baseY + 9 * PX, 8, 6, pal.shirt);
  // arms
  px(ctx, baseX, baseY + 9 * PX + (moving ? -swing : 0), 1, 5, pal.shirt);
  px(ctx, baseX + 9 * PX, baseY + 9 * PX + (moving ? swing : 0), 1, 5, pal.shirt);
  // hands
  px(ctx, baseX, baseY + 14 * PX + (moving ? -swing : 0), 1, 1, pal.skin);
  px(ctx, baseX + 9 * PX, baseY + 14 * PX + (moving ? swing : 0), 1, 1, pal.skin);

  // head
  px(ctx, baseX + 2 * PX, baseY + 3 * PX, 6, 6, pal.skin);
  // hair
  px(ctx, baseX + 2 * PX, baseY + 2 * PX, 6, 2, pal.hair);
  px(ctx, baseX + 1 * PX, baseY + 3 * PX, 1, 3, pal.hair);
  px(ctx, baseX + 8 * PX, baseY + 3 * PX, 1, 3, pal.hair);
  // eyes
  const eyeY = baseY + 5 * PX;
  px(ctx, baseX + (facing > 0 ? 4 : 3) * PX, eyeY, 1, 1, "#1b1b24");
  px(ctx, baseX + (facing > 0 ? 6 : 5) * PX, eyeY, 1, 1, "#1b1b24");
}

/** Name tag under a character. */
export function drawNameTag(ctx, name, x, y, active) {
  ctx.save();
  ctx.font = "bold 10px 'Courier New', monospace";
  ctx.textAlign = "center";
  const w = ctx.measureText(name).width + 10;
  ctx.fillStyle = active ? "rgba(255,214,102,0.95)" : "rgba(20,22,32,0.7)";
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y + 4, w, 13, 6);
  ctx.fill();
  ctx.fillStyle = active ? "#1b1b24" : "#c9d1d9";
  ctx.fillText(name, x, y + 14);
  ctx.restore();
}

/** Speech bubble with an emoji + short label. */
export function drawBubble(ctx, text, x, y, alpha = 1, accent = "#ffffff") {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = "11px system-ui, sans-serif";
  const w = ctx.measureText(text).width + 16;
  const h = 22;
  const bx = x - w / 2;
  const by = y - h;
  ctx.fillStyle = "rgba(255,255,255,0.96)";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(bx, by, w, h, 8);
  ctx.fill();
  ctx.stroke();
  // tail
  ctx.beginPath();
  ctx.moveTo(x - 5, by + h - 1);
  ctx.lineTo(x, by + h + 6);
  ctx.lineTo(x + 5, by + h - 1);
  ctx.fillStyle = "rgba(255,255,255,0.96)";
  ctx.fill();
  ctx.fillStyle = "#1b1b24";
  ctx.textAlign = "center";
  ctx.fillText(text, x, by + 15);
  ctx.restore();
}

/** A desk with monitor; screen glows at night. */
export function drawDesk(ctx, x, y, night, busy) {
  px(ctx, x - 7 * PX, y - 4 * PX, 14, 2, "#6b4f2a");
  px(ctx, x - 7 * PX, y - 2 * PX, 14, 1, "#4a361d");
  px(ctx, x - 6 * PX, y - 1 * PX, 1, 4, "#4a361d");
  px(ctx, x + 5 * PX, y - 1 * PX, 1, 4, "#4a361d");
  // monitor
  px(ctx, x - 3 * PX, y - 11 * PX, 6, 5, "#2b3245");
  px(ctx, x - 2 * PX, y - 10 * PX, 4, 3, busy ? "#6ee7f0" : night ? "#2a4d6e" : "#3d4a63");
  px(ctx, x - 1 * PX, y - 6 * PX, 2, 2, "#2b3245");
  if (busy || night) {
    ctx.save();
    ctx.globalAlpha = busy ? 0.35 : 0.18;
    const g = ctx.createRadialGradient(x, y - 26, 2, x, y - 26, 40);
    g.addColorStop(0, "#7dd3fc");
    g.addColorStop(1, "transparent");
    ctx.fillStyle = g;
    ctx.fillRect(x - 42, y - 66, 84, 80);
    ctx.restore();
  }
}

/** CI rack: blinking LEDs + spinning gear. */
export function drawServerRack(ctx, x, y, t, intensity) {
  px(ctx, x - 6 * PX, y - 18 * PX, 12, 18, "#252c3a");
  px(ctx, x - 5 * PX, y - 17 * PX, 10, 16, "#171c26");
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 4; col++) {
      const on = Math.sin(t * (2 + row) + col * 1.7 + intensity * 4) > 0.1;
      px(
        ctx,
        x - 4 * PX + col * 2 * PX,
        y - 15 * PX + row * 3 * PX,
        1,
        1,
        on ? (intensity > 0.3 ? "#3ec07a" : "#7dd3fc") : "#31394a"
      );
    }
  }
}

export function drawGear(ctx, x, y, r, angle, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  const teeth = 8;
  for (let i = 0; i < teeth; i++) {
    ctx.rotate((Math.PI * 2) / teeth);
    ctx.fillRect(-r * 0.18, -r - r * 0.28, r * 0.36, r * 0.34);
  }
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#171c26";
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.38, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/** Protocol board; sticky notes pile up as rules are adopted. */
export function drawProtocolBoard(ctx, x, y, notes, t) {
  px(ctx, x - 13 * PX, y - 22 * PX, 26, 20, "#5b4326");
  px(ctx, x - 12 * PX, y - 21 * PX, 24, 18, "#2f3b2a");
  const colors = ["#ffd166", "#f4978e", "#a0e7a0", "#a5d8ff", "#e0aaff"];
  for (let i = 0; i < notes; i++) {
    const col = i % 6;
    const row = Math.floor(i / 6);
    const wob = Math.sin(t * 1.5 + i) * 0.8;
    px(
      ctx,
      x - 11 * PX + col * 4 * PX,
      y - 19 * PX + row * 5 * PX + wob,
      3,
      3,
      colors[i % colors.length]
    );
  }
}

/** Mainline "git" track with merged commits as dots. */
export function drawMainline(ctx, x1, x2, y, merges, t) {
  ctx.save();
  ctx.strokeStyle = "#3ec07a";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.stroke();
  const n = Math.min(merges, 36);
  for (let i = 0; i < n; i++) {
    const cx = x1 + ((x2 - x1) / 36) * i + 6;
    const pulse = i === n - 1 ? 2 + Math.sin(t * 6) * 1.5 : 0;
    ctx.beginPath();
    ctx.fillStyle = i === n - 1 ? "#a7f3d0" : "#3ec07a";
    ctx.arc(cx, y, 3.5 + pulse, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawRocket(ctx, x, y, lift) {
  ctx.save();
  ctx.translate(x, y - lift);
  px(ctx, -2 * PX, -12 * PX, 4, 9, "#e6edf3");
  px(ctx, -2 * PX, -14 * PX, 4, 2, "#f85149");
  px(ctx, -1 * PX, -10 * PX, 2, 2, "#58a6ff");
  px(ctx, -4 * PX, -5 * PX, 2, 3, "#f85149");
  px(ctx, 2 * PX, -5 * PX, 2, 3, "#f85149");
  if (lift > 1) {
    ctx.globalAlpha = 0.85;
    px(ctx, -2 * PX, -3 * PX, 4, 3, "#ffa657");
    px(ctx, -1 * PX, 0, 2, 3, "#ffd166");
  }
  ctx.restore();
}

export function drawPlant(ctx, x, y) {
  px(ctx, x - 2 * PX, y - 4 * PX, 4, 4, "#8a5a2b");
  px(ctx, x - 3 * PX, y - 9 * PX, 6, 5, "#2f8f4e");
  px(ctx, x - 1 * PX, y - 12 * PX, 2, 3, "#3ec07a");
}
