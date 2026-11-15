# Test footage specification (external videos, TESTING ONLY)

Project rule: external footage is never used for training, only to test the system.

---

## Prompt

I need publicly available boxing videos to **test** (not train) a computer-vision system that
counts and classifies punches from body pose, tracks two fighters separately, and detects whether
a punch made contact. Please find candidate videos in the three categories below, with the
technical requirements, and report each candidate in the required format. Prefer quality over
quantity: 2–4 excellent candidates per category beat 20 mediocre ones.

### Category A: sparring (two fighters)
- Two people boxing each other (sparring or a light bout), **both fully visible head to feet** most of the time.
- **Fixed camera** on a tripod or mount: no handheld shake, no zooming, no panning, no cuts. A slow
  pan that keeps both fighters in frame is acceptable only as a second-choice candidate.
- Camera roughly **side-on** to the fighters (perpendicular to the line between them, ±30°), at
  roughly chest height, 3–8 m away.
- **Clearly different clothing** between the two fighters (e.g. one light shirt, one dark; or
  different trunk colours). Different glove colours are a bonus.
- Contains a natural mix of **landed, blocked and missed punches**, and at least a few
  **clinches or moments where the fighters cross / overlap**.
- At most one other person (e.g. a coach or referee) visible, preferably none.

### Category B: BOB (Body Opponent Bag, human-shaped freestanding bag)
- One person punching a **BOB** (human torso-and-head shaped bag), the whole BOB and the whole
  person visible.
- **Fixed camera**, side-on or 45° to the line between person and BOB.
- A mix of **head and body** punches, ideally with some **misses or punches in the air** between
  combinations (shadow moves, feints).
- Punches varied: jabs, crosses, hooks, uppercuts.

### Category C: punching bags
- C1: a **hanging heavy bag**. C2: a **freestanding (water/sand base) bag**.
- One person, whole body and whole bag visible, **fixed camera**, side-on or 45°.
- The bag **visibly moves** on impact (swing or wobble).
- Ideally includes some punches thrown **without hitting** the bag (feints, shadow work between rounds).

### Technical requirements (all categories)
- Resolution **≥ 720p**, frame rate **≥ 24 fps**, **real-time speed** (no slow motion, no speed ramps,
  no time-lapse), not mirrored/flipped.
- **Continuous segment of ≥ 2 minutes** (ideally 3–5) meeting the requirements, without cuts.
- Bodies not covered by large overlays, banners or picture-in-picture (small corner logos/timers are fine).
- Good lighting; no heavy motion blur in normal movement.
- Raw or lightly edited footage preferred over highlight reels or montages.

### Exclusions
- Professional broadcast fights with constant camera cuts and close-ups.
- Phone footage filmed from inside the ring while moving.
- Footage where the main subjects appear to be minors.
- Videos that are private, unlisted with restricted access, or whose terms forbid downloading for research.
- Do not include the 10 videos already used by this project (YouTube IDs kbgkeTTSau8, JGLLJUgm-0c,
  Cx6JfrRO4eA, IVvOmoB88FM, V59fcX1YC7E, cQW-h8yGbRc, sqU3DmLppcQ, tSNybMUpMKk, WuBPbI0x50o,
  OfZFdwW322E, 3ov6zVAXHJI).

### Required report for each candidate
- **Category** (A / B / C1 / C2) and **URL**
- **Start and end timestamps** of the best qualifying continuous segment (mm:ss–mm:ss)
- **Camera:** fixed / slight pan / other; angle (side-on / 45° / frontal); approximate distance
- **People visible:** number; for A, describe each fighter's clothing and glove colours
- **Content:** approximate number of punches; whether there are clinches/crossings (A), head vs body
  shots (B), misses/feints (B, C)
- **Quality:** resolution, frame rate, any overlays, blur or lighting issues
- **Licence / terms** as stated on the page (e.g. standard YouTube licence, Creative Commons)
- **Any concerns** (e.g. partly out of frame at times, third person walks through)

---

## Why each requirement matters (for the project log)
- Fixed camera: isolates tracker/model errors from camera-motion errors (handheld is a later robustness test).
- Side-on: punches travel across the image, so 2D glove-to-target distance is meaningful for contact detection.
- Different clothing: the identity tracker relies on colour appearance during clinches.
- Clinches/crossings: the real test of identity preservation.
- Landed/blocked/missed, head/body, misses/feints: so contact detection is tested on both outcomes.
- Bag visibly moving: the bag's reaction is one of the contact signals.
- ≥ 720p, ≥ 24 fps, real speed: matches our training data (1280×720, 24–30 fps); slow motion would distort punch speed.
