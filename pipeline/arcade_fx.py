"""ARCADE HUD + HIT FEEDBACK for the rendered videos (user's spec, 2026-11-15; branch arcade-hud; cosmetics only).

Classic fighting-game look drawn over real footage: mirrored slanted health bars (ghost damage layer, colour by
health, gloss, shake, flash, shards, low-health pulse, K.O. crack), three tiers of landed-punch feedback (LIGHT jab,
HEAVY cross / hook, CRITICAL uppercut / counter: hit-stop, camera shake, punch-in, flash, chromatic aberration,
starburst, shock ring, speed lines, sweat, glove trail, combo counter, callout), and a clearly different cool-blue
block (shield arc, sideways sparks, contracting ring, 2 px jolt). Every number is in P (seconds / pixels at 1280 wide).

What a video overlay cannot do, and is left out: the round timer (user: ignore it), round-win pips, deforming the
fighters (squash-and-stretch, slide-back, glove bounce, dust), slow motion (it would break sync with the sound),
chip damage (a blocked punch costs nothing here) and the perfect block (there is no block input).
Hit-stop is done as hold-then-catch-up: the picture holds for the hit-stop time and then jumps on, so the video keeps
its length and its sound stays in sync.
"""
import math, os, wave
import cv2, numpy as np

P = dict(
    # ---- health bars
    bar_w=0.38, bar_h=0.04, bar_top=0.03, bar_side=0.02, slant_deg=12.0, bezel=3,
    ghost_hold=0.45, ghost_drain=0.35, bar_shake=(4, 8, 14), bar_shake_t=0.25, bar_flash_t=0.05, colour_blend_t=0.2,
    shine_every=6.0, shine_t=0.8, gloss=0.30, low=25.0, low_hz=2.0, ko_crack_t=0.3, shards_per_dmg=1.5,
    # ---- landed punch, per tier (LIGHT, HEAVY, CRITICAL)
    # ANIME STYLE (user, 2026-11-15: "so exaggerated, like anime"). The spec's values were: hitstop (0.04, 0.09, 0.18), shake (3, 9, 18),
    # shake_t (0.10, 0.20, 0.35), zoom (0, 0.04, 0.10), flash (0, 0.08, 0.20), aberr (0, 2, 5), burst (0.6, 1.0, 1.6), sweat (8, 20, 45).
    hitstop=(0.07, 0.15, 0.28), shake=(8, 20, 38), shake_t=(0.16, 0.30, 0.50), zoom=(0.05, 0.13, 0.26), flash=(0.12, 0.30, 0.55), flash_t=0.06,
    aberr=(2, 6, 12), aberr_t=0.22, burst=(1.2, 2.1, 3.3), burst_px=46, burst_pop=0.03, burst_fade=0.22, ring_t=0.28, lines_t=0.4,
    sweat=(20, 50, 100), victim_white=0.05, victim_red=0.12, trail=3,
    impact_frame=(0.0, 0.07, 0.13),                 # inverted high-contrast "impact frame", seconds per tier
    focus_t=(0.0, 0.28, 0.55), focus_n=(0, 60, 110), focus_hole=(0, 250, 190),     # full-screen concentration lines pointing at the impact
    slash_t=0.16, radial_blur_t=0.2, words=("", ("BAM!", "WHAM!", "POW!", "THUD!"), ""), word_px=(0, 64, 104),
    combo_gap=1.5, combo_pop=0.12, combo_total_t=0.8, callout_in=0.08, callout_hold=0.4, callout_out=0.2, callout_deg=8.0, counter_window=0.35,
    # ---- blocked punch
    block_pop=0.03, block_fade=0.18, block_sparks=(10, 16), block_spark_t=0.2, block_ring_t=0.12, block_hitstop=0.03, block_jolt=2,
    # ---- batch 2 (user, 2026-11-15): combo ranks, damage numbers, GUARD / WHIFF words, glove trails, danger vignette,
    #      K.O. finish, picture-in-picture slow-motion replay of critical hits, intro card, results card
    combo_ranks=((2, "GOOD!", "FFFFFF"), (4, "GREAT!", "FFE033"), (6, "BRUTAL!", "FF8A00"), (8, "UNSTOPPABLE!", "FF3B30")), rank_px=34, rank_pop=0.12,
    dmg_t=0.9, dmg_px=(40, 56, 82), dmg_rise=95,
    word_in=0.06, word_hold=0.35, word_out=0.2, guard_px=54, whiff_px=46,
    trail_n=10, trail_cols=("4FC3FF", "FF8A00", "FF3BD0"),            # straight, hook, uppercut
    vig_hz=2.0, vig_a=(0.18, 0.30),
    ko_t=2.4, ko_in=0.12, ko_grey_in=0.3, ko_out=0.5, ko_letterbox=0.11, ko_cracks=9, ko_px=220,
    pip_w=0.30, pip_pre=0.35, pip_post=0.35, pip_speed=0.25, pip_fade=0.15,
    intro_t=2.6, fight_at=1.7, outro_t=6.0,
    # ---- general
    max_particles=900, reduce=0.25,
)
TIERS = ("LIGHT", "HEAVY", "CRITICAL")
FONT_PATHS = ["/System/Library/Fonts/Supplemental/Impact.ttf", "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", "/Library/Fonts/Impact.ttf"]

def hexc(h):
    h = h.lstrip("#"); return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))          # -> BGR

GRAD = [((hexc("3BE35A"), hexc("1E9E3A"))), ((hexc("FFC530"), hexc("E08A00"))), ((hexc("FF3B30"), hexc("B00010")))]
GHOST, EMPTY, GOLD, METAL, BLUE = hexc("FFE9A0"), hexc("1A0505"), hexc("D4AF37"), (58, 54, 50), hexc("7FD4FF")

class Text:
    """Heavy condensed italic text as BGRA patches (PIL, cached); falls back to OpenCV's font if no system font is found."""
    def __init__(self):
        self.cache = {}; self.font = next((f for f in FONT_PATHS if os.path.exists(f)), None)

    def patch(self, s, px, fill, stroke=(0, 0, 0), sw=3, shadow=True, italic=True):
        key = (s, px, fill, stroke, sw, shadow, italic)
        if key in self.cache:
            return self.cache[key]
        pad = sw + 6
        if self.font:
            from PIL import Image, ImageDraw, ImageFont
            f = ImageFont.truetype(self.font, px); l, t, r, b = f.getbbox(s, stroke_width=sw); w, h = r - l + 2 * pad, b - t + 2 * pad
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
            if shadow:
                d.text((pad - l + 3, pad - t + 4), s, font=f, fill=(0, 0, 0, 200), stroke_width=sw, stroke_fill=(0, 0, 0, 200))
            d.text((pad - l, pad - t), s, font=f, fill=fill[::-1] + (255,), stroke_width=sw, stroke_fill=stroke[::-1] + (255,))
            a = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)
        else:
            sc = px / 30; (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, sc, 2); a = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
            cv2.putText(a, s, (pad, h + pad), cv2.FONT_HERSHEY_DUPLEX, sc, stroke + (255,), 2 + 2 * sw, cv2.LINE_AA); cv2.putText(a, s, (pad, h + pad), cv2.FONT_HERSHEY_DUPLEX, sc, fill + (255,), 2, cv2.LINE_AA)
        if italic:                                                      # shear to the right
            h, w = a.shape[:2]; k = 0.2; a = cv2.warpAffine(a, np.float32([[1, -k, k * h], [0, 1, 0]]), (w + int(k * h), h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        self.cache[key] = a; return a

    def paste(self, im, a, cx, cy, scale=1.0, angle=0.0, alpha=1.0, anchor="c"):
        h, w = a.shape[:2]
        if scale != 1.0 or angle:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale); c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
            M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2; a = cv2.warpAffine(a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)); h, w = nh, nw
        x = int(cx - (w / 2 if anchor == "c" else (0 if anchor == "l" else w))); y = int(cy - h / 2); H, W = im.shape[:2]
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if xb <= xa or yb <= ya:
            return
        src = a[ya - y:yb - y, xa - x:xb - x]; al = src[..., 3:4].astype(np.float32) / 255 * alpha
        im[ya:yb, xa:xb] = (im[ya:yb, xa:xb] * (1 - al) + src[..., :3] * al).astype(np.uint8)

def tier_of(cls):
    return 0 if cls == 1 else (2 if cls >= 5 else 1)                    # L straight = jab; R straight (cross) and hooks; uppercuts

class Arcade:
    def __init__(self, punches, W, H, fps, dmg, names=("BOXER A", "BOXER B"), cols=((0, 140, 255), (255, 220, 0)), reduce=False, target_only=False):
        """punches: dicts with boxer 'A'/'B', cls, verdict ('LANDED' / 'blocked' / 'missed' / None), t_hit, cx, cy (source pixels), type.
        Coordinates given to the drawing calls are in output pixels; k = output / source scale is applied to cx, cy by the caller via `scale`."""
        self.W, self.H, self.fps, self.names, self.cols = W, H, fps, names, cols; self.z = W / 1280; self.red = P["reduce"] if reduce else 1.0
        self.txt = Text(); self.rng = np.random.default_rng(7); self.parts = []; self.held = None; self.spawned = set(); self.col_state = {}; self.ev = []
        last = {0: [], 1: []}
        for n, p in enumerate(sorted(punches, key=lambda p: p["t_hit"])):
            b = "AB".index(p["boxer"]); v = p["verdict"]; t = p["t_hit"]; tr = tier_of(p["cls"]); counter = False
            if v == "LANDED":                                           # counter: the boxer who is hit had just thrown a punch that did not land
                counter = any(0 < t - q["t_hit"] <= P["counter_window"] and q["verdict"] != "LANDED" for q in last[1 - b])
                tr = 2 if counter else tr
            r = np.random.default_rng(1000 + n)
            self.ev.append(dict(n=n, t=t, b=b, v=v, cls=p["cls"], type=p["type"], dmg=dmg[p["cls"]] if v == "LANDED" else 0, tier=tr, c=None if p.get("cx") is None else (p["cx"], p["cy"]),
                                counter=counter, rot=float(r.uniform(0, 6.28)), word="COUNTER!" if counter else "CRUSHING!", hand=(p["cls"] - 1) % 2, dir=None, p=p.get("p"),
                                t0=p.get("t0", t - 0.2), t1=p.get("t1", t), jx=float(r.uniform(-30, 30))))
            last[b].append(p)
        self.hits = {b: [e for e in self.ev if e["v"] == "LANDED" and e["b"] == 1 - b] for b in (0, 1)}             # hits TAKEN by boxer b
        self.combos = {b: [] for b in (0, 1)}                         # per attacker: runs of landed hits not broken by a pause or by the opponent landing
        for b in (0, 1):
            run = []
            for e in [e for e in self.ev if e["v"] == "LANDED"]:
                if e["b"] != b:
                    if run:
                        self.combos[b].append(run); run = []
                    continue
                if run and e["t"] - run[-1]["t"] > P["combo_gap"]:
                    self.combos[b].append(run); run = []
                run.append(e)
            if run:
                self.combos[b].append(run)
        h = max(int(round(P["bar_h"] * H)), 14); w = int(P["bar_w"] * W); y0 = int(P["bar_top"] * H); m = int(P["bar_side"] * W); self.geo = dict(h=h, w=w, y0=y0, x={0: m, 1: W - m}, dx=int(round(h * math.tan(math.radians(P["slant_deg"])))))
        self.kos = []                                                   # (boxer knocked out, time, impact point): the first time each bar reaches zero
        for b in (0, 1):
            e = next((e for e in self.hits[b] if self.hp(b, e["t"]) <= 0), None)
            if e is not None:
                self.kos.append((b, e["t"], e["c"]))
        self.vig = None; self.pip_buf = []; self.pip_q = []; self.pip_seen = set(); self.pip_cur = None

    # ------------------------------------------------------------------ particles (one pool, capped)
    def spawn(self, x, y, vx, vy, life, col, size=2, grav=900.0, kind="dot"):
        if len(self.parts) < P["max_particles"]:
            self.parts.append([x, y, vx, vy, life, life, col, size, grav, kind])

    def draw_parts(self, im, kinds, step=True):
        dt = 1 / self.fps; keep = []
        for q in self.parts:
            if q[9] in kinds:
                if step:
                    q[0] += q[2] * dt; q[1] += q[3] * dt; q[3] += q[8] * dt; q[4] -= dt
                if q[4] <= 0:
                    continue
                a = q[4] / q[5]; x, y = int(q[0]), int(q[1])
                if q[9] == "line":
                    n = math.hypot(q[2], q[3]) + 1e-6; cv2.line(im, (x, y), (int(x - q[2] / n * 10 * self.z), int(y - q[3] / n * 10 * self.z)), q[6], max(int(q[7] * a) + 1, 1), cv2.LINE_AA)
                else:
                    cv2.circle(im, (x, y), max(int(q[7] * self.z), 1) + 1, (255, 255, 255), 1, cv2.LINE_AA); cv2.circle(im, (x, y), max(int(q[7] * self.z * (0.5 + 0.5 * a)), 1), q[6], -1, cv2.LINE_AA)
            keep.append(q)
        self.parts = keep

    # ------------------------------------------------------------------ health
    def hp(self, b, t):
        return max(100.0 - sum(e["dmg"] for e in self.hits[b] if e["t"] <= t), 0.0)

    def ghost(self, b, t):
        """Health snaps; the ghost holds 0.45 s after the LAST hit (so a combo shows as one chunk), then drains in 0.35 s, ease-out."""
        hs = [e for e in self.hits[b] if e["t"] <= t]; hp = self.hp(b, t)
        if not hs:
            return hp
        last = hs[-1]["t"]; u = (t - last - P["ghost_hold"]) / P["ghost_drain"]
        if u >= 1:
            return hp
        k = len(hs) - 1
        while k > 0 and hs[k]["t"] - hs[k - 1]["t"] <= P["ghost_hold"] + P["ghost_drain"]:
            k -= 1
        g0 = self.hp(b, hs[k]["t"] - 1e-6)
        return g0 if u <= 0 else hp + (g0 - hp) * (1 - u) ** 2

    def bar_x(self, b, u, r):
        """x of the point at share u (0 = outer edge) on row r of bar b; the top leans towards the centre."""
        g = self.geo; sh = g["dx"] * (1 - r / max(g["h"] - 1, 1))
        return g["x"][0] + u * g["w"] + sh if b == 0 else g["x"][1] - u * g["w"] - sh

    def hud(self, im, t, stats=None):
        g = self.geo; h, y0 = g["h"], g["y0"]; H, W = im.shape[:2]; z = self.z
        top = im[:int(y0 + h + 78 * z)].astype(np.float32); fade = np.linspace(0.55, 0.0, top.shape[0])[:, None, None]; im[:top.shape[0]] = (top * (1 - fade)).astype(np.uint8)   # soft dark band for legibility
        for b in (0, 1):
            hp, gh = self.hp(b, t), self.ghost(b, t); taken = [e for e in self.hits[b] if e["t"] <= t]; since = t - taken[-1]["t"] if taken else 9.0; tr = taken[-1]["tier"] if taken else 0
            low = 0 < hp < P["low"]; ko = hp <= 0; pulse = 0.5 + 0.5 * math.sin(2 * math.pi * P["low_hz"] * t)
            amp = P["bar_shake"][tr] * z * self.red * max(1 - since / P["bar_shake_t"], 0); ox, oy = (self.rng.uniform(-amp, amp, 2) if amp > 0.3 else (0, 0)); ox, oy = int(ox), int(oy)
            band = 2 if hp < 25 else (1 if hp < 50 else 0); st = self.col_state.setdefault(b, dict(c=np.array(GRAD[band], np.float32)))          # colour blends over 0.2 s at a threshold
            st["c"] += (np.array(GRAD[band], np.float32) - st["c"]) * min(1.0, (1 / self.fps) / P["colour_blend_t"]); ctop, cbot = st["c"]
            shine = (t % P["shine_every"]) / P["shine_t"]
            for r in range(h):
                y = y0 + r + oy
                if not 0 <= y < H:
                    continue
                def seg(ua, ub, col):
                    xa, xb = sorted((self.bar_x(b, ua, r) + ox, self.bar_x(b, ub, r) + ox)); xa, xb = int(np.clip(xa, 0, W)), int(np.clip(xb, 0, W))
                    if xb > xa:
                        im[y, xa:xb] = col
                    return xa, xb
                xa, xb = seg(0, 1, EMPTY if not ko else (22, 22, 22))
                im[y, xa:xb][(np.arange(xa, xb) + y) % 9 == 0] = (38, 16, 16) if not ko else (40, 40, 40)                       # faint diagonal hatch
                if gh > hp:
                    seg(hp / 100, gh / 100, GHOST)
                if hp > 0:
                    c = ctop + (cbot - ctop) * (r / max(h - 1, 1))
                    if r < h / 3:
                        c = c + (255 - c) * P["gloss"]
                    if low:
                        c = c * (0.75 + 0.25 * pulse)
                    if since <= P["bar_flash_t"]:
                        c = np.array((255, 255, 255), np.float32)
                    xa, xb = seg(0, hp / 100, c.astype(np.uint8))
                    if shine < 1 and since > P["bar_flash_t"]:                                                                    # slow shine sweep
                        sx = self.bar_x(b, shine, r) + ox; ia, ib = int(max(min(sx - 14 * z, xb), xa)), int(max(min(sx + 14 * z, xb), xa))
                        if ib > ia:
                            im[y, ia:ib] = (im[y, ia:ib].astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)
            quad = np.array([[self.bar_x(b, 0, 0), y0], [self.bar_x(b, 1, 0), y0], [self.bar_x(b, 1, h - 1), y0 + h], [self.bar_x(b, 0, h - 1), y0 + h]], np.int32) + (ox, oy)
            if low:
                ov = im.copy(); cv2.polylines(ov, [quad], True, (40, 40, 255), int(9 * z), cv2.LINE_AA); im[:] = cv2.addWeighted(ov, 0.25 + 0.3 * pulse, im, 0.75 - 0.3 * pulse, 0)
                if self.rng.random() < 0.06:
                    q = quad[self.rng.integers(0, 4)]; self.spawn(q[0], q[1], *self.rng.uniform(-90, 90, 2), 0.3, (120, 200, 255), 1.5, 500.0, "ui")
            cv2.polylines(im, [quad], True, METAL if not ko else (70, 70, 70), max(int(P["bezel"] * z), 2), cv2.LINE_AA)
            inner = (quad + np.sign(quad.mean(0) - quad) * 2).astype(np.int32); cv2.polylines(im, [inner], True, GOLD if not ko else (120, 120, 120), 1, cv2.LINE_AA)
            cv2.line(im, tuple(int(v) for v in quad[0] + (0, -1)), tuple(int(v) for v in quad[1] + (0, -1)), (150, 150, 150), 1, cv2.LINE_AA)                # bevel highlight on the top edge
            for e in taken:                                             # shards break off the leading edge, once per hit
                if ("shard", e["n"]) not in self.spawned and since < 0.1 and e is taken[-1]:
                    self.spawned.add(("shard", e["n"])); lx = self.bar_x(b, min(gh, 100) / 100, h // 2)
                    for _ in range(int(e["dmg"] * P["shards_per_dmg"])):
                        self.spawn(lx + self.rng.uniform(-6, 6), y0 + self.rng.uniform(0, h), self.rng.uniform(-60, 60) + (60 if b == 0 else -60), self.rng.uniform(-160, 40), self.rng.uniform(0.3, 0.6), GHOST, 2, 700.0, "ui")
            if ko:                                                      # the frame cracks (0.3 s) and the bar stays grey
                tko = next(e["t"] for e in self.hits[b] if self.hp(b, e["t"]) <= 0); u = min((t - tko) / P["ko_crack_t"], 1.0); r = np.random.default_rng(50 + b)
                for _ in range(5):
                    ua = r.uniform(0.05, 0.95); pts = [(self.bar_x(b, ua + r.uniform(-0.03, 0.03) * j, int(j / 4 * (h - 1))), y0 + j / 4 * h) for j in range(5)]; pts = np.array(pts[:max(int(u * 5), 2)], np.int32)
                    cv2.polylines(im, [pts], False, (235, 235, 235), 1, cv2.LINE_AA)
            # name under the outer edge, counts under it
            name = self.txt.patch(self.names[b], int(26 * z), (60, 60, 255) if (low or ko) else (255, 255, 255), (0, 0, 0), 2); ny = y0 + h + int(20 * z); ex = self.bar_x(b, 0, h - 1)
            cv2.rectangle(im, (int(ex) - (0 if b == 0 else int(10 * z)), ny - int(8 * z)), (int(ex) + (int(10 * z) if b == 0 else 0), ny + int(8 * z)), self.cols[b], -1)
            self.txt.paste(im, name, ex + (16 * z if b == 0 else -16 * z), ny, anchor="l" if b == 0 else "r")
            if stats:
                for j, s in enumerate(stats[b]):
                    sc = (0.52 if j == 0 else 0.45) * z; (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, 1); x = int(ex) if b == 0 else int(ex - tw); y = ny + int((26 + 20 * j) * z)
                    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, self.cols[b] if j == 0 else (225, 225, 225), 1, cv2.LINE_AA)
            self.combo(im, b, t, ny + int(84 * z))
        self.draw_parts(im, ("ui",))
        return im

    def combo(self, im, b, t, y):
        z = self.z; x = self.bar_x(b, 0.02, 0)
        for run in self.combos[b]:
            if len(run) < 2 or t < run[1]["t"]:
                continue
            end = run[-1]["t"] + P["combo_gap"]; n = sum(e["t"] <= t for e in run)
            if t <= end:
                since = t - run[n - 1]["t"]; sc = 1.0 + 0.4 * max(1 - since / P["combo_pop"], 0); col = (255, 255, 255) if n < 3 else (hexc("FFE033") if n < 5 else (hexc("FF8A00") if n < 7 else hexc("FF3B30")))
                a = self.txt.patch(f"{n} HITS", int(44 * z), col, (0, 0, 0), 3); self.txt.paste(im, a, x + (a.shape[1] / 2 if b == 0 else -a.shape[1] / 2), y, scale=sc)
                rk = [r for r in P["combo_ranks"] if n >= r[0]]
                if rk:                                                  # combo rank word, slammed on when the combo reaches its threshold
                    th, word, hx = rk[-1]; sr = t - run[th - 1]["t"]; s2 = 1.0 + 1.5 * max(1 - sr / P["rank_pop"], 0)
                    a2 = self.txt.patch(word, int(P["rank_px"] * z), hexc(hx), hexc("D01010") if th >= 6 else (0, 0, 0), 4)
                    self.txt.paste(im, a2, x + (a2.shape[1] / 2 if b == 0 else -a2.shape[1] / 2), y + 46 * z, scale=s2, angle=6 if b == 0 else -6)
            elif t <= end + P["combo_total_t"] + 0.2:
                u = max((t - end - P["combo_total_t"]) / 0.2, 0); a = self.txt.patch(f"{len(run)} HITS  {sum(e['dmg'] for e in run)} DMG", int(34 * z), hexc("FFE033"), (0, 0, 0), 3)
                off = u * 400 * z * (-1 if b == 0 else 1); self.txt.paste(im, a, x + (a.shape[1] / 2 if b == 0 else -a.shape[1] / 2) + off, y, alpha=1 - u)

    # ------------------------------------------------------------------ hit-stop (hold, then catch up)
    def hitstop(self, im, t):
        if self.held is not None and t < self.held[1]:
            return self.held[0].copy()
        self.held = None
        for e in self.ev:
            if e["v"] in ("LANDED", "blocked") and 0 <= t - e["t"] < 1 / self.fps:
                d = P["hitstop"][e["tier"]] if e["v"] == "LANDED" else P["block_hitstop"]
                if d > 1 / self.fps and (self.held is None or e["t"] + d > self.held[1]):
                    self.held = (im.copy(), e["t"] + d)
        return im

    # ------------------------------------------------------------------ impact visuals in the picture
    def effects(self, im, t, k, head, trail):
        """k: output / source scale; head: {boxer: (x, y)} this frame; trail: {(boxer, hand): [glove points, oldest first]}."""
        z = self.z
        for e in self.ev:
            dt = t - e["t"]
            if e["c"] is None or not -0.1 <= dt <= 0.5:
                continue
            c = np.array(e["c"]) * k; ci = (int(c[0]), int(c[1]))
            if e["dir"] is None and dt >= 0:
                o = head.get(e["b"]); d = (c - np.array(o, float)) if o is not None else np.array([1.0 if e["b"] == 0 else -1.0, 0.0]); n = np.linalg.norm(d); e["dir"] = d / n if n > 1 else np.array([1.0 if e["b"] == 0 else -1.0, 0.0])
            if e["v"] == "LANDED":
                tr = e["tier"]; S = P["burst"][tr] * P["burst_px"] * z
                if tr >= 1 and -0.1 <= dt <= 0.1:                        # glove afterimages
                    pts = trail.get((e["b"], e["hand"]), [])[-P["trail"]:]
                    for j, q in enumerate(pts):
                        ov = im.copy(); cv2.circle(ov, (int(q[0]), int(q[1])), int(15 * z), hexc("FF8A00"), -1, cv2.LINE_AA); im[:] = cv2.addWeighted(ov, 0.12 + 0.1 * j, im, 0.88 - 0.1 * j, 0)
                if dt < 0:
                    continue
                d = e["dir"]
                if ("sweat", e["n"]) not in self.spawned:
                    self.spawned.add(("sweat", e["n"]))
                    for _ in range(P["sweat"][tr]):
                        a = math.atan2(d[1], d[0]) + self.rng.uniform(-0.6, 0.6); sp = self.rng.uniform(180, 520) * z
                        self.spawn(c[0], c[1], math.cos(a) * sp, math.sin(a) * sp - 120 * z, self.rng.uniform(0.3, 0.6), (255, 246, 232), self.rng.uniform(1.5, 3), 900.0 * z, "world")
                if dt <= P["victim_white"] + P["victim_red"]:              # the struck area flashes white, then red (no silhouette of the boxer is known)
                    ov = im.copy(); cv2.circle(ov, (int(c[0] + d[0] * 30 * z), int(c[1] + d[1] * 30 * z)), int(1.5 * S), (255, 255, 255) if dt <= P["victim_white"] else (40, 40, 255), -1, cv2.LINE_AA)
                    ov = cv2.GaussianBlur(ov, (0, 0), 9 * z); a = (0.55 if dt <= P["victim_white"] else 0.3) * (self.red if dt <= P["victim_white"] else 1); im[:] = cv2.addWeighted(ov, a, im, 1 - a, 0)
                if tr == 2 and dt <= P["lines_t"]:                        # speed lines
                    u = dt / P["lines_t"]; ov = im.copy(); r = np.random.default_rng(e["n"])
                    for _ in range(22):
                        a = r.uniform(0, 6.28); r0, r1 = S * (1.6 + 2 * u), S * (3.2 + 3 * u + r.uniform(0, 1.5))
                        cv2.line(ov, (int(c[0] + r0 * math.cos(a)), int(c[1] + r0 * math.sin(a))), (int(c[0] + r1 * math.cos(a)), int(c[1] + r1 * math.sin(a))), (255, 255, 255), max(int(2 * z), 1), cv2.LINE_AA)
                    im[:] = cv2.addWeighted(ov, 0.7 * (1 - u), im, 1 - 0.7 * (1 - u), 0)
                if dt <= P["ring_t"]:                                     # shock rings: grow to 2 x (and 3 x) the burst
                    u = dt / P["ring_t"]; ov = im.copy(); cv2.circle(ov, ci, int(S * (1 + u)), (255, 255, 255), max(int(6 * z * (1 - u)) + 1, 1), cv2.LINE_AA)
                    if tr >= 1:
                        cv2.circle(ov, ci, int(S * (1 + 2 * u)), hexc("FFD21E"), max(int(4 * z * (1 - u)) + 1, 1), cv2.LINE_AA)
                    im[:] = cv2.addWeighted(ov, 0.85 * (1 - u), im, 1 - 0.85 * (1 - u), 0)
                if tr >= 1 and dt <= P["slash_t"]:                        # a white slash through the impact, along the punch
                    u = dt / P["slash_t"]; L = S * (3.5 + 3 * u); wd = max(int(S * 0.22 * (1 - u)), 1); ov = im.copy()
                    pa = (int(c[0] - d[0] * L), int(c[1] - d[1] * L)); pb = (int(c[0] + d[0] * L), int(c[1] + d[1] * L)); cv2.line(ov, pa, pb, hexc("FFD21E"), wd + int(8 * z), cv2.LINE_AA); cv2.line(ov, pa, pb, (255, 255, 255), wd, cv2.LINE_AA)
                    im[:] = cv2.addWeighted(ov, 1 - u, im, u, 0)
                if dt <= P["burst_pop"] + P["burst_fade"]:                 # starburst: white core, yellow, orange edge; random rotation per hit
                    sc = 0.7 + 0.5 * min(dt / P["burst_pop"], 1.0); al = 1.0 if dt <= P["burst_pop"] else 1 - (dt - P["burst_pop"]) / P["burst_fade"]; ov = im.copy(); r = np.random.default_rng(e["n"])
                    rad = np.where(np.arange(20) % 2 == 0, 1.0, 0.45) * r.uniform(0.85, 1.15, 20); ang = e["rot"] + np.arange(20) * math.pi / 10
                    if tr >= 1:                                         # long thin spikes behind the star
                        for j in range(14):
                            a = e["rot"] + j * 0.449 + 0.2; L = S * sc * r.uniform(1.5, 2.4); w_ = 0.06
                            cv2.fillPoly(ov, [np.array([[c[0] + math.cos(a) * L, c[1] + math.sin(a) * L], [c[0] + math.cos(a - w_) * S * 0.5, c[1] + math.sin(a - w_) * S * 0.5], [c[0] + math.cos(a + w_) * S * 0.5, c[1] + math.sin(a + w_) * S * 0.5]], np.int32)], (255, 255, 255), cv2.LINE_AA)
                    for f_, col in ((1.0, hexc("FF8A00")), (0.78, hexc("FFD21E")), (0.5, (255, 255, 255))):
                        cv2.fillPoly(ov, [np.stack([c[0] + np.cos(ang) * rad * S * sc * f_, c[1] + np.sin(ang) * rad * S * sc * f_], 1).astype(np.int32)], col, cv2.LINE_AA)
                    im[:] = cv2.addWeighted(ov, al, im, 1 - al, 0)
            elif e["v"] == "blocked" and dt >= 0:                         # cool colours, contained shape
                d = e["dir"]; deg = math.degrees(math.atan2(d[1], d[0]))
                if ("spark", e["n"]) not in self.spawned:
                    self.spawned.add(("spark", e["n"]))
                    for _ in range(int(self.rng.integers(P["block_sparks"][0], P["block_sparks"][1] + 1))):   # ricochet sideways, slightly back towards the attacker
                        s = self.rng.choice((-1, 1)); a = math.atan2(d[1], d[0]) + s * self.rng.uniform(1.3, 2.2); sp = self.rng.uniform(220, 480) * z
                        self.spawn(c[0], c[1], math.cos(a) * sp, math.sin(a) * sp, P["block_spark_t"], (255, 240, 215), 2, 0.0, "line")
                if dt <= P["block_pop"] + P["block_fade"]:
                    al = 1.0 if dt <= P["block_pop"] else 1 - (dt - P["block_pop"]) / P["block_fade"]; ov = im.copy(); cc = (int(c[0] + d[0] * 26 * z), int(c[1] + d[1] * 26 * z))
                    cv2.ellipse(ov, cc, (int(34 * z), int(58 * z)), deg, 110, 250, BLUE, max(int(9 * z), 2), cv2.LINE_AA); cv2.ellipse(ov, cc, (int(34 * z), int(58 * z)), deg, 118, 242, (255, 255, 255), max(int(3 * z), 1), cv2.LINE_AA)
                    im[:] = cv2.addWeighted(ov, al, im, 1 - al, 0)
                if dt <= P["block_ring_t"]:                               # ring contracts inwards
                    u = dt / P["block_ring_t"]; cv2.circle(im, ci, int((44 - 30 * u) * z), BLUE, max(int(2 * z), 1), cv2.LINE_AA)
            elif e["v"] not in ("LANDED", "blocked") and 0 <= dt <= 0.3:  # missed: grey streaks (not in the spec; kept from the old renderer)
                u = dt / 0.3; ov = im.copy()
                for q in (-1, 0, 1):
                    cv2.ellipse(ov, (ci[0], ci[1] + int(q * 14 * z)), (int((30 + u * 70) * z), int(10 * z)), 0, 200, 340, (200, 200, 200), max(int(2 * z), 1), cv2.LINE_AA)
                im[:] = cv2.addWeighted(ov, 1 - u, im, u, 0)
        self.draw_parts(im, ("world", "line"))
        return im

    def camera(self, im, t, k):
        """Shake, punch-in, flash, chromatic aberration for landed hits; a 2 px horizontal jolt for blocks."""
        H, W = im.shape[:2]; dx = dy = 0.0; zoom = 0.0; zc = (W / 2, H / 2); flash = 0.0; ab = 0.0
        for e in self.ev:
            dt = t - e["t"]
            if not 0 <= dt <= 0.4:
                continue
            if e["v"] == "LANDED":
                tr = e["tier"]; u = dt / P["shake_t"][tr]
                if u < 1:
                    a = P["shake"][tr] * self.z * self.red * (1 - u); ddx, ddy = self.rng.uniform(-a, a, 2); dx += ddx; dy += ddy
                    if P["zoom"][tr] * (1 - u) > zoom and e["c"] is not None:
                        zoom = P["zoom"][tr] * (1 - u); zc = (e["c"][0] * k, e["c"][1] * k)
                if dt <= P["flash_t"]:
                    flash = max(flash, P["flash"][tr] * self.red)
                if dt <= P["aberr_t"]:
                    ab = max(ab, P["aberr"][tr] * self.z * self.red * (1 - dt / P["aberr_t"]))
            elif e["v"] == "blocked" and dt <= 2 / self.fps:
                dx += P["block_jolt"] * self.z * (1 if e["b"] == 0 else -1)
        if abs(dx) > 0.5 or abs(dy) > 0.5 or zoom > 0.002:
            s = 1 + zoom; im = cv2.warpAffine(im, np.float32([[s, 0, (1 - s) * zc[0] + dx], [0, s, (1 - s) * zc[1] + dy]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for e in self.ev:                                               # CRITICAL: radial zoom blur around the impact
            dt = t - e["t"]
            if e["v"] == "LANDED" and e["tier"] == 2 and e["c"] is not None and 0 <= dt <= P["radial_blur_t"]:
                u = 1 - dt / P["radial_blur_t"]; cx, cy = e["c"][0] * k, e["c"][1] * k; acc = im.astype(np.float32)
                for s_ in (1.03, 1.06, 1.10):
                    s2 = 1 + (s_ - 1) * u * (0.4 + 0.6 * self.red); acc += cv2.warpAffine(im, np.float32([[s2, 0, (1 - s2) * cx], [0, s2, (1 - s2) * cy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
                im = (acc / 4).astype(np.uint8)
        if ab >= 1:
            a = int(round(ab)); im = im.copy(); im[:, a:, 2] = im[:, :-a, 2]; im[:, :-a, 0] = im[:, a:, 0]
        for e in self.ev:                                               # anime impact frame: inverted, high contrast, tinted
            dt = t - e["t"]
            if e["v"] == "LANDED" and 0 <= dt < P["impact_frame"][e["tier"]] and self.red == 1.0:
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); g = 255 - np.clip((g - 120) * 2.6 + 128, 0, 255); tint = np.array((255, 235, 235) if e["tier"] == 1 else (215, 215, 255), np.float32) / 255
                im = (g[..., None] * tint).astype(np.uint8); break
        if flash > 0.01:
            im = cv2.addWeighted(im, 1 - flash, np.full_like(im, 255), flash, 0)
        return im

    def focus(self, im, t, k):
        """Manga concentration lines: thin wedges from beyond the screen edge pointing at the impact, re-drawn (jittering) every frame."""
        H, W = im.shape[:2]; far = math.hypot(W, H)
        for e in self.ev:
            dt = t - e["t"]
            if e["v"] != "LANDED" or e["c"] is None or not 0 <= dt <= P["focus_t"][e["tier"]]:
                continue
            u = dt / P["focus_t"][e["tier"]]; c = np.array(e["c"]) * k; ov = im.copy(); r = np.random.default_rng(int(t * 1000) + e["n"]); hole = P["focus_hole"][e["tier"]] * self.z * (1 + 0.5 * u)
            for _ in range(P["focus_n"][e["tier"]]):
                a = r.uniform(0, 6.283); w_ = r.uniform(0.006, 0.028); r0 = hole * r.uniform(0.8, 1.7)
                cv2.fillPoly(ov, [np.array([[c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0], [c[0] + math.cos(a - w_) * far, c[1] + math.sin(a - w_) * far], [c[0] + math.cos(a + w_) * far, c[1] + math.sin(a + w_) * far]], np.int32)],
                             (255, 255, 255) if r.random() < 0.8 else (0, 0, 0), cv2.LINE_AA)
            al = 0.85 * (1 - u) ** 0.7 * (0.4 + 0.6 * self.red); im[:] = cv2.addWeighted(ov, al, im, 1 - al, 0)
        return im

    def callouts(self, im, t, k):
        """CRITICAL: a word slammed on near the impact, 250% -> 100% in 0.08 s, held 0.4 s, faded over 0.2 s, rotated 8 degrees.
        Also: GUARD! (blue) on a blocked punch, WHIFF! (grey) on a missed one, and the damage number floating up from a landed hit."""
        H, W = im.shape[:2]; z = self.z
        for e in self.ev:
            dt = t - e["t"]
            if e["c"] is None or dt < 0:
                continue
            cx, cy = e["c"][0] * k, e["c"][1] * k
            dup = any(q["b"] == e["b"] and q["v"] == e["v"] and 0 < e["t"] - q["t"] <= 0.3 for q in self.ev)   # one word per burst of misses / blocks
            if e["v"] in ("blocked", "missed") and not dup and dt <= P["word_in"] + P["word_hold"] + P["word_out"]:
                g = e["v"] == "blocked"; sc = 1.8 - 0.8 * min(dt / P["word_in"], 1.0); al = 1.0 if dt <= P["word_in"] + P["word_hold"] else 1 - (dt - P["word_in"] - P["word_hold"]) / P["word_out"]
                a = self.txt.patch("GUARD!" if g else "WHIFF!", int((P["guard_px"] if g else P["whiff_px"]) * z), BLUE if g else (200, 200, 200), (90, 40, 0) if g else (60, 60, 60), 4)
                x = float(np.clip(cx, a.shape[1] * 0.6, W - a.shape[1] * 0.6)); y = float(np.clip(cy + 50 * z, self.geo["y0"] + 260 * z, H - 60))   # below the impact, clear of the hit words
                self.txt.paste(im, a, x, y, scale=sc, angle=-5 if e["n"] % 2 else 5, alpha=al)
            if e["v"] == "LANDED" and e["dmg"] and dt <= P["dmg_t"]:
                u = dt / P["dmg_t"]; sc = 1.6 - 0.6 * min(dt / 0.08, 1.0); al = 1.0 if u < 0.6 else (1 - u) / 0.4; tr = e["tier"]
                a = self.txt.patch(f"-{e['dmg']}", int(P["dmg_px"][tr] * z), ((255, 255, 255), hexc("FFE033"), hexc("FF3B30"))[tr], (0, 0, 0), 4)
                x = float(np.clip(cx + e["jx"] * z + 60 * z * (1 if e["b"] == 0 else -1), 40, W - 40)); y = float(np.clip(cy - 30 * z - P["dmg_rise"] * z * u, self.geo["y0"] + 150 * z, H - 40))
                self.txt.paste(im, a, x, y, scale=sc, alpha=al)
        for e in self.ev:
            dt = t - e["t"]
            if e["v"] != "LANDED" or e["tier"] == 0 or e["c"] is None or not 0 <= dt <= P["callout_in"] + P["callout_hold"] + P["callout_out"]:
                continue
            word = e["word"] if e["tier"] == 2 else P["words"][1][e["n"] % len(P["words"][1])]
            sc = 2.5 - 1.5 * min(dt / P["callout_in"], 1.0); al = 1.0 if dt <= P["callout_in"] + P["callout_hold"] else 1 - (dt - P["callout_in"] - P["callout_hold"]) / P["callout_out"]
            a = self.txt.patch(word, int(P["word_px"][e["tier"]] * self.z), hexc("FFE033") if e["tier"] == 2 else (255, 255, 255), hexc("D01010"), 6 if e["tier"] == 2 else 5); H, W = im.shape[:2]
            x = float(np.clip(e["c"][0] * k, a.shape[1] * 0.6, W - a.shape[1] * 0.6)); y = float(np.clip(e["c"][1] * k - (110 + 40 * e["tier"]) * self.z, self.geo["y0"] + 260 * self.z, H - 60))     # below the combo counter and rank
            self.txt.paste(im, a, x, y, scale=sc, angle=P["callout_deg"] * (1 if e["n"] % 2 else -1), alpha=al)
        return im

    # ------------------------------------------------------------------ batch 2
    def trails(self, im, t, trail):
        """A ribbon behind the punching glove while a counted punch is in flight, coloured by punch family
        (straight blue, hook orange, uppercut magenta); thicker and brighter towards the newest point."""
        z = self.z; ov = None
        for e in self.ev:
            if not e["t0"] - 0.02 <= t <= max(e["t1"], e["t"]) + 0.1:
                continue
            pts = trail.get((e["b"], e["hand"]), [])[-P["trail_n"]:]
            if len(pts) < 2:
                continue
            ov = im.copy() if ov is None else ov; col = hexc(P["trail_cols"][(e["cls"] - 1) // 2]); n = len(pts)
            for j in range(1, n):
                a, b_ = tuple(int(v) for v in pts[j - 1]), tuple(int(v) for v in pts[j]); w = max(int((3 + 14 * j / n) * z), 1)
                cv2.line(ov, a, b_, col, w, cv2.LINE_AA); cv2.line(ov, a, b_, (255, 255, 255), max(w // 3, 1), cv2.LINE_AA)
        if ov is not None:
            im[:] = cv2.addWeighted(ov, 0.75, im, 0.25, 0)
        return im

    def vignette(self, im, t):
        """Pulsing red vignette on the side of a boxer whose health is below 25% (and not yet K.O.)."""
        H, W = im.shape[:2]
        if self.vig is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); e = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2); base = np.clip((e - 0.55) / 0.6, 0, 1) ** 1.5
            self.vig = {0: base * np.clip(1.15 - xx / W, 0.35, 1), 1: base * np.clip(0.15 + xx / W, 0.35, 1)}
        m = None; pulse = 0.5 + 0.5 * math.sin(2 * math.pi * P["vig_hz"] * t)
        for b in (0, 1):
            hp = self.hp(b, t)
            if 0 < hp < P["low"]:
                a = (P["vig_a"][0] + P["vig_a"][1] * pulse) * (0.6 + 0.4 * self.red); m = self.vig[b] * a if m is None else np.maximum(m, self.vig[b] * a)
        if m is not None:
            m = m[..., None]; im[:] = (im * (1 - m) + np.array((20, 20, 230), np.float32) * m).astype(np.uint8)
        return im

    def _ko(self, t):
        for b, tko, c in self.kos:
            if 0 <= t - tko <= P["ko_t"]:
                return b, t - tko, c
        return None

    def ko_scene(self, im, t, k):
        """K.O. finish, part 1 (under the HUD): picture drains to grey, the screen cracks from the impact, letterbox bars slide in."""
        ko = self._ko(t)
        if ko is None:
            return im
        b, dt, c = ko; H, W = im.shape[:2]; z = self.z; env = min(dt / P["ko_grey_in"], 1.0) * min((P["ko_t"] - dt) / P["ko_out"], 1.0)
        g = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR); g = np.clip((g.astype(np.float32) - 128) * 1.3 + 118, 0, 255).astype(np.uint8)
        im = cv2.addWeighted(g, env, im, 1 - env, 0)
        cx, cy = (c[0] * k, c[1] * k) if c is not None else (W / 2, H / 2); r = np.random.default_rng(77 + b); grow = min(dt / 0.15, 1.0); ov = im.copy()
        for j in range(P["ko_cracks"]):
            a = j * 2 * math.pi / P["ko_cracks"] + r.uniform(-0.25, 0.25); L = W * r.uniform(0.35, 0.7) * grow; pts = [(cx, cy)]
            for s in range(1, 9):
                d = L * s / 8; aa = a + r.uniform(-0.18, 0.18); pts.append((cx + math.cos(aa) * d, cy + math.sin(aa) * d))
            pts = np.array(pts, np.int32); cv2.polylines(ov, [pts], False, (0, 0, 0), max(int(5 * z), 2), cv2.LINE_AA); cv2.polylines(ov, [pts], False, (255, 255, 255), max(int(2 * z), 1), cv2.LINE_AA)
        im = cv2.addWeighted(ov, 0.9 * env, im, 1 - 0.9 * env, 0)
        lb = int(P["ko_letterbox"] * H * min(dt / 0.2, 1.0) * min((P["ko_t"] - dt) / 0.3, 1.0))
        if lb > 0:
            im[:lb] = 0; im[H - lb:] = 0
        return im

    def ko_text(self, im, t):
        """K.O. finish, part 2 (over the HUD): a giant K.O. slammed on, 300% -> 100%."""
        ko = self._ko(t)
        if ko is None:
            return im
        b, dt, _ = ko; H, W = im.shape[:2]; al = min((P["ko_t"] - dt) / P["ko_out"], 1.0); sc = 3.0 - 2.0 * min(dt / P["ko_in"], 1.0)
        a = self.txt.patch("K.O.", int(P["ko_px"] * self.z), hexc("FF3B30"), GOLD, 10); self.txt.paste(im, a, W / 2, H / 2, scale=sc, alpha=al)
        s = self.txt.patch(f"{self.names[b]} IS DOWN", int(40 * self.z), (255, 255, 255), (0, 0, 0), 3); self.txt.paste(im, s, W / 2, H / 2 + 150 * self.z, alpha=al * min(dt / 0.3, 1.0))
        return im

    def pip_capture(self, im, t):
        """Keeps the last ~0.9 s of rendered frames (small) for the slow-motion replay."""
        H, W = im.shape[:2]; pw = int(P["pip_w"] * W); ph = int(pw * H / W)
        self.pip_buf.append((t, cv2.resize(im, (pw, ph), interpolation=cv2.INTER_AREA))); keep = P["pip_pre"] + P["pip_post"] + 0.2
        while self.pip_buf and self.pip_buf[0][0] < t - keep:
            self.pip_buf.pop(0)
        for e in self.ev:                                               # queue each critical hit once its after-part has been rendered
            if e["v"] == "LANDED" and e["tier"] == 2 and e["n"] not in self.pip_seen and t >= e["t"] + P["pip_post"]:
                self.pip_seen.add(e["n"])
                if len(self.pip_q) < 2:                                 # no long backlog: a burst of criticals replays at most two
                    self.pip_q.append((e, [f for tf, f in self.pip_buf if e["t"] - P["pip_pre"] <= tf <= e["t"] + P["pip_post"]]))
        if self.pip_cur is None and self.pip_q:
            e, fr = self.pip_q.pop(0); self.pip_cur = (e, fr, t)

    def pip_draw(self, im, t):
        """Picture-in-picture (bottom right): the latest critical hit again at quarter speed, with its effects."""
        if self.pip_cur is None:
            return im
        e, fr, t0 = self.pip_cur; el = t - t0; n = len(fr); dur = n / (self.fps * P["pip_speed"]); j = int(el * self.fps * P["pip_speed"])
        if not fr or j >= n:
            self.pip_cur = None; return im
        H, W = im.shape[:2]; z = self.z; f = fr[j]; ph, pw = f.shape[:2]; x0, y0 = W - pw - int(18 * z), H - ph - int(18 * z)
        al = min(el / P["pip_fade"], 1.0, (dur - el) / P["pip_fade"]); ov = im.copy()
        cv2.rectangle(ov, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), GOLD, -1); ov[y0:y0 + ph, x0:x0 + pw] = f
        im[:] = cv2.addWeighted(ov, al, im, 1 - al, 0)
        a = self.txt.patch("REPLAY  x1/4", int(24 * z), (255, 255, 255), (0, 0, 0), 2); self.txt.paste(im, a, x0 + 8 * z, y0 + 18 * z, alpha=al, anchor="l")
        w = self.txt.patch(e["word"], int(30 * z), hexc("FFE033"), hexc("D01010"), 3); self.txt.paste(im, w, x0 + pw - 8 * z, y0 + ph - 22 * z, alpha=al, anchor="r")
        return im

    def intro(self, base, s, t0):
        """Intro card, s seconds in: names slide in, VS slams on, then FIGHT! with a flash (bars full)."""
        im = (base.astype(np.float32) * 0.45).astype(np.uint8); H, W = im.shape[:2]; z = self.z; im = self.hud(im, t0 - 1e-3)
        ease = lambda u: 1 - (1 - min(max(u, 0.0), 1.0)) ** 3
        fade = 1.0 if s < P["fight_at"] else max(1 - (s - P["fight_at"]) / 0.15, 0.0)
        if fade > 0:
            for b in (0, 1):
                a = self.txt.patch(self.names[b], int(84 * z), self.cols[b], (0, 0, 0), 5); u = ease((s - 0.15 * b) / 0.45)
                x = -a.shape[1] + (W * 0.27 + a.shape[1]) * u if b == 0 else W + a.shape[1] - (W * 0.27 + a.shape[1]) * u
                self.txt.paste(im, a, x, H * 0.42, alpha=fade)
            if s >= 0.6:
                v = self.txt.patch("VS", int(120 * z), hexc("FFE033"), hexc("D01010"), 7); self.txt.paste(im, v, W / 2, H * 0.42, scale=3 - 2 * min((s - 0.6) / 0.12, 1.0), alpha=fade)
        if s >= P["fight_at"]:
            d = s - P["fight_at"]; al = min((P["intro_t"] - s) / 0.25, 1.0)
            f = self.txt.patch("FIGHT!", int(170 * z), hexc("FFE033"), hexc("D01010"), 9); self.txt.paste(im, f, W / 2, H * 0.5, scale=3 - 2 * min(d / 0.12, 1.0), angle=-4, alpha=al)
            if d < 0.08:
                im = cv2.addWeighted(im, 0.4, np.full_like(im, 255), 0.6, 0)
        return im

    def results(self, base, s, R):
        """Results card, s seconds in, from the real counts R (see count_video.results_stats): winner, then rows one by one."""
        H, W = base.shape[:2]; z = self.z
        if getattr(self, "_res_bg", None) is None or self._res_bg.shape != base.shape:
            self._res_bg = (cv2.GaussianBlur(base, (0, 0), 9).astype(np.float32) * 0.3).astype(np.uint8)
        im = self._res_bg.copy()
        w = R["winner"]; title = "DRAW" if w is None else f"{self.names[w]} WINS"; tc = (255, 255, 255) if w is None else self.cols[w]
        a = self.txt.patch(title, int(96 * z), tc, (0, 0, 0), 6); self.txt.paste(im, a, W / 2, H * 0.13, scale=2.5 - 1.5 * min(s / 0.12, 1.0), alpha=min(s / 0.12, 1.0))
        if s > 0.3:
            r = self.txt.patch(R["how"], int(30 * z), hexc("FFE033"), (0, 0, 0), 3); self.txt.paste(im, r, W / 2, H * 0.13 + 70 * z, alpha=min((s - 0.3) / 0.2, 1.0))
        cx = {0: W * 0.30, 1: W * 0.70}
        for b in (0, 1):
            if s > 0.5:
                n = self.txt.patch(self.names[b], int(40 * z), self.cols[b], (0, 0, 0), 3); self.txt.paste(im, n, cx[b], H * 0.13 + 125 * z, alpha=min((s - 0.5) / 0.2, 1.0))
        y = H * 0.13 + 175 * z
        for j, (label, va, vb, better) in enumerate(R["rows"]):
            t0 = 0.8 + 0.15 * j
            if s < t0:
                break
            al = min((s - t0) / 0.15, 1.0); lab = self.txt.patch(label, int(26 * z), (200, 200, 200), (0, 0, 0), 2, italic=False); self.txt.paste(im, lab, W / 2, y, alpha=al)
            for b, v in ((0, va), (1, vb)):
                hi = better == b; p = self.txt.patch(str(v), int((34 if hi else 30) * z), GOLD if hi else (255, 255, 255), (0, 0, 0), 3, italic=False); self.txt.paste(im, p, cx[b], y, alpha=al)
            y += 38 * z
        return im


def sounds(events, t_start, dur, path, sr=44100, lead=0.0, tail=0.0, cues=()):
    """LIGHT: a tight snap. HEAVY: a meaty thud over a low thump. CRITICAL: a bass drop and a crack, then a (synthesised) crowd swell.
    Blocked: a dull leather thump and muffled slap, no bass. Missed: a quiet whoosh. events: Arcade.ev.
    lead / tail: seconds of intro / results card around the footage (event times are shifted by lead).
    cues: (time in the output, kind) with kind "bell" (one ring), "bell3" (three rings) or "boom" (deep K.O. impact)."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(0); a = np.zeros(int((lead + dur + tail + 2) * sr), np.float32)
    def band(x, lo, hi):
        b, c = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band"); return lfilter(b, c, x)
    def tone(f0, f1, n, decay):
        x = np.arange(n) / sr; f = f1 + (f0 - f1) * np.exp(-x * 14); return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-x * decay)
    def add(s, t):
        i = max(int(t * sr), 0); s = s[:len(a) - i]; a[i:i + len(s)] += s.astype(np.float32)
    def bell(t):
        n = int(1.6 * sr); x = np.arange(n) / sr
        add(sum(g * np.sin(2 * np.pi * f * x) * np.exp(-x * d) for f, g, d in ((820, 0.5, 2.5), (1290, 0.3, 3.5), (2050, 0.2, 5.0), (2690, 0.1, 7.0))) * (1 - np.exp(-x * 400)), t)
    for t, kind in cues:
        if kind == "bell":
            bell(t)
        elif kind == "bell3":
            for j in range(3):
                bell(t + 0.32 * j)
        elif kind == "boom":
            n = int(1.4 * sr); x = np.arange(n) / sr; add(1.3 * tone(90, 28, n, 3.5) + 0.6 * band(rng.standard_normal(n), 60, 400) * np.exp(-x * 6), t)
    for e in events:
        t = e["t"] - t_start
        if not 0 <= t < dur:
            continue
        t += lead
        if e["v"] == "LANDED" and e["tier"] == 0:
            n = int(0.09 * sr); x = np.arange(n) / sr; add(0.55 * band(rng.standard_normal(n), 1200, 5000) * np.exp(-x * 90) + 0.25 * tone(260, 170, n, 60), t)
        elif e["v"] == "LANDED" and e["tier"] == 1:
            n = int(0.30 * sr); x = np.arange(n) / sr; add(0.75 * tone(150, 62, n, 15) + 0.5 * band(rng.standard_normal(n), 250, 1800) * np.exp(-x * 45) + 0.35 * tone(70, 42, n, 9), t)
        elif e["v"] == "LANDED":
            n = int(0.55 * sr); x = np.arange(n) / sr; add(1.0 * tone(120, 34, n, 7) + 0.7 * band(rng.standard_normal(n), 1500, 6000) * np.exp(-x * 70) + 0.4 * band(rng.standard_normal(n), 200, 1200) * np.exp(-x * 30), t)
            n = int(1.1 * sr); x = np.arange(n) / sr; add(0.16 * band(rng.standard_normal(n), 350, 1400) * np.clip(x / 0.25, 0, 1) * np.exp(-np.clip(x - 0.25, 0, None) * 3.2), t + 0.3)
        elif e["v"] == "blocked":
            n = int(0.11 * sr); x = np.arange(n) / sr; add(0.42 * band(rng.standard_normal(n), 400, 1300) * np.exp(-x * 50) + 0.16 * np.sin(2 * np.pi * 230 * x) * np.exp(-x * 45), t)
        else:
            n = int(0.22 * sr); add(0.2 * band(rng.standard_normal(n), 700, 3500) * np.hanning(n), t - 0.10)
    a = np.clip(a / max(np.abs(a).max(), 1e-6) * 0.9, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a * 32767).astype(np.int16).tobytes())
