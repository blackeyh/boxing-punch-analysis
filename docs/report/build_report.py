"""Builds docs/Boxing_Punch_Analysis_Report.pdf (reportlab). Numbers come from the research log (docs/research_log.md).
    python docs/report/charts.py && python docs/report/build_report.py
"""
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle, ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.dirname(HERE); MEDIA = os.path.join(DOCS, "media")
OUT = os.path.join(DOCS, "Boxing_Punch_Analysis_Report.pdf")
INK, MUTED, ACC, LINE, SOFT = colors.HexColor("#1f2430"), colors.HexColor("#6b7280"), colors.HexColor("#d9480f"), colors.HexColor("#e5e7eb"), colors.HexColor("#f6f7f9")
GOOD, BADC = colors.HexColor("#2b8a3e"), colors.HexColor("#c92a2a")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK, spaceBefore=4, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=ACC, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=11, textColor=MUTED)
CAP = ParagraphStyle("CAP", parent=SM, alignment=TA_LEFT, spaceBefore=2, spaceAfter=10)
TC = ParagraphStyle("TC", parent=B, fontSize=8.4, leading=10.6, spaceAfter=0)
TH = ParagraphStyle("TH", parent=TC, fontName="Helvetica-Bold")
KEY = ParagraphStyle("KEY", parent=B, fontSize=10, leading=14, backColor=SOFT, borderPadding=(8, 8, 8, 8), borderColor=LINE, borderWidth=0.5, spaceBefore=6, spaceAfter=12)

W = A4[0] - 4 * cm


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items, s=B):
    return ListFlowable([ListItem(P(i, s), leftIndent=12, value="circle") for i in items], bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8)


def table(rows, widths, head=True, zebra=True):
    data = [[P(c, TH if (head and r == 0) else TC) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        st += [("BACKGROUND", (0, r), (-1, r), SOFT) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t


def img(name, width=W, cap=None):
    p = os.path.join(MEDIA, name)
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(p).getSize(); im = Image(p, width=width, height=width * ih / iw)
    return KeepTogether([im, P(cap, CAP)]) if cap else im


def img_row(names, cap, gap=0.25 * cm):
    from reportlab.lib.utils import ImageReader
    n = len(names); w = (W - gap * (n - 1)) / n; cells = []
    for nm in names:
        p = os.path.join(MEDIA, nm); iw, ih = ImageReader(p).getSize(); cells.append(Image(p, width=w, height=w * ih / iw))
    t = Table([cells], colWidths=[w + (gap if i < n - 1 else 0) for i in range(n)]); t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return KeepTogether([t, P(cap, CAP)])


def pipeline_diagram():
    stages = [("Video", "any camera,\n24-60 fps"), ("RTMPose", "2D skeletons of\nevery person"), ("Boxer selector", "learned, picks\nthe 2 boxers"),
              ("Identity", "OSNet-AIN re-ID:\nA stays A"), ("Punch TCN", "when + which hand\n+ type (skeleton)"), ("Contact scan", "VideoMAE-L:\nmoment + verdict"),
              ("Outputs", "per boxer: thrown,\nlanded / blocked /\nmissed, by type")]
    d = Drawing(W, 3.3 * cm); n = len(stages); gap = 0.32 * cm; bw = (W - gap * (n - 1)) / n; bh = 2.5 * cm; y = 0.4 * cm
    for i, (t, sub) in enumerate(stages):
        x = i * (bw + gap); hot = i in (2, 3, 5)
        d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=colors.HexColor("#fff4ec") if hot else SOFT, strokeColor=ACC if hot else LINE, strokeWidth=0.9))
        d.add(String(x + bw / 2, y + bh - 0.55 * cm, t, fontName="Helvetica-Bold", fontSize=7.6, fillColor=INK, textAnchor="middle"))
        for k, line in enumerate(sub.split("\n")):
            d.add(String(x + bw / 2, y + bh - 1.05 * cm - k * 0.33 * cm, line, fontName="Helvetica", fontSize=6.3, fillColor=MUTED, textAnchor="middle"))
        if i < n - 1:
            ax = x + bw; ay = y + bh / 2; d.add(Line(ax + 1, ay, ax + gap - 2, ay, strokeColor=INK, strokeWidth=0.8))
            d.add(Polygon([ax + gap - 1, ay, ax + gap - 5, ay + 2.2, ax + gap - 5, ay - 2.2], fillColor=INK, strokeColor=INK))
    return d


def on_page(c, doc):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    if doc.page > 1:
        c.drawString(2 * cm, 1.2 * cm, "Boxing punch analysis from video - project report")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    c.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                          title="Boxing Punch Analysis from Video - Project Report", author="blackeyh", subject="Punch detection, type, fighter credit and landed/blocked/missed from monocular video")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=on_page)])
    s = []

    # ------------------------------------------------------------------ title page
    s += [Spacer(1, 2.2 * cm), P("PROJECT REPORT", ParagraphStyle("k", parent=SM, textColor=ACC, fontName="Helvetica-Bold", fontSize=9)),
          P("Boxing punch analysis from ordinary video", ParagraphStyle("T", parent=H1, fontSize=26, leading=31)),
          P("Who threw it, what punch it was, and whether it landed: from a single-boxer classifier to a full two-fighter pipeline "
            "tested on real bouts it never trained on.", ParagraphStyle("st", parent=B, fontSize=12, leading=17, textColor=MUTED)),
          Spacer(1, 0.6 * cm), img("hit.jpg", W), P("Rendered output on a Creative Commons sparring clip (Michal Hradi\u0161, CC BY): tracked skeletons, "
                                                  "a counted punch with its type and verdict, damage number, health bars and per-boxer counts.", CAP), Spacer(1, 0.4 * cm)]
    s.append(P("<b>At a glance</b><br/>"
               "\u2022 Punch type on unseen single-boxer videos: <b>75.2%</b> balanced accuracy over 6 classes (goal was 60%).<br/>"
               "\u2022 Real Olympic bouts never trained on: punches found <b>24.8% -> 79.4%</b>, F1 <b>0.33 -> 0.75</b>.<br/>"
               "\u2022 Telling the two boxers apart for a whole round: <b>99.5-99.8%</b> of frames with person re-identification (motion tracking alone: 84%).<br/>"
               "\u2022 Landed / blocked / missed at the referee's contact moment, unseen gym: <b>74-76%</b> (always guessing 'landed': 62%); landed punches per bout within <b>7.6%</b> (median).<br/>"
               "\u2022 Detection on the BoxingWeb test set on par with the published BoxMind system (F1 0.77-0.79 vs 0.78, loose matching).", KEY))
    s.append(P("Built Nov 12-15, 2026 as a personal research and learning project. Hardware: a MacBook (Apple silicon) and a university HPC "
               "cluster (one RTX 5000 Ada per job). Every experiment, prediction and result was logged as it happened; the full log is "
               "included in the repository as <i>docs/research_log.md</i>.", SM))
    s.append(PageBreak())

    # ------------------------------------------------------------------ contents
    s.append(P("Contents", H1))
    toc = ["1. Goal and definitions", "2. Data", "3. The final system", "4. Phase 1 - punch type from skeletons", "5. Phase 2 - detection: is a punch happening?",
           "6. Phase 3 - two fighters and the sparring wall", "7. Phase 4 - real bouts: pose, boxer selection, identity, detection", "8. Phase 5 - did it land?",
           "9. Phase 6 - type fusion, false counts, bag and BOB route", "10. Phase 7 - the rendered video", "11. Engineering problems along the way",
           "12. Challenges and how they were overcome (summary)", "13. Limitations and next steps", "14. Credits and licences"]
    s += [P(t, ParagraphStyle("toc", parent=B, fontSize=10.5, leading=17)) for t in toc]
    s.append(PageBreak())

    # ------------------------------------------------------------------ 1
    s.append(P("1. Goal and definitions", H1))
    s.append(P("The aim is an app that watches ordinary video of boxing and, for each fighter, counts punches <b>by type</b> and counts "
               "<b>only the ones that landed</b>, on an opponent in sparring or on a heavy bag, standing bag or BOB dummy. It should work with "
               "one camera, YouTube-quality footage, bystanders and a referee in view, and both orthodox and southpaw stances."))
    s.append(table([["Term", "Meaning in this report"],
                    ["Punch type", "6 classes: jab, cross, lead hook, rear hook, lead uppercut, rear uppercut (on real bouts: left/right straight, hook, uppercut)."],
                    ["Landed", "The punch made contact with the head or body, as opposed to <b>blocked</b> (hit the guard) or <b>missed</b>. Agreed definition, matching the Olympic referee labels."],
                    ["LOVO", "Leave-one-video-out: every score is on a video the model never saw, so it reflects a new person, camera and gym."],
                    ["Balanced accuracy", "Mean recall per class, so a rare class (rear hook) counts as much as a common one."],
                    ["Found / real / F1", "Found = share of labelled punches the system counted (recall). Real = share of counts that match a label (precision). F1 combines both."],
                    ["Chance level", "Labels shifted by several seconds and re-scored, to show what a system gets by luck."]], [3.2 * cm, W - 3.2 * cm]))

    # ------------------------------------------------------------------ 2
    s.append(P("2. Data", H1))
    s.append(table([["Source", "What it is", "Labels", "Used for", "Licence"],
                    ["BoxingVI (Kumar et al., arXiv 2511.16524), videos V1-V10", "Single boxer: heavy bag, standing bag, shadowboxing. 5,428 punch clips after cleaning.", "Start/end frame + type", "Phase 1-2 training and LOVO tests", "Research dataset"],
                    ["Olympic Boxing Punch Classification (Stefa\u0144ski, Kozak, Jach, Entropy 2024)", "29 youth bouts, 104 min, 2 gyms, fixed corner camera, 50 fps.", "4,605 punches: head / body / block / miss + hand, box at contact", "Real-bout test; landed model training", "Non-commercial"],
                    ["BoxingWeb (BoxMind, Wang et al., arXiv 2601.11492)", "50 rounds of elite broadcast bouts, 134 min.", "6,872 punches: span, boxer, type, target, 'effective'; per-frame poses of both boxers", "Detector, selector, identity training", "Not stated"],
                    ["Test clips from YouTube", "2 sparring clips (A1, A2), BOB, heavy bag, standing bag.", "None", "Judged by eye only", "A2: CC BY; others standard"]],
                   [3.4 * cm, 4.2 * cm, 3.6 * cm, 2.9 * cm, 2.9 * cm]))
    s.append(P("Data rules changed once, deliberately: at first only BoxingVI was allowed for training (outside footage for testing only). When the "
               "sparring results showed that single-boxer data cannot teach sparring, outside data was allowed for training, while the sparring test "
               "clips and their channels stayed out of training. Rendered videos of third-party footage are never published.", SM))

    # ------------------------------------------------------------------ 3
    s.append(P("3. The final system", H1))
    s.append(pipeline_diagram())
    s.append(P("Figure 1. The chain that runs on a new video. Orange boxes are the parts that had to be invented or replaced after the first design failed on real fights.", CAP))
    s.append(bullets([
        "<b>RTMPose</b> (YOLOX person detector + RTMPose-m) gives 2D skeletons of everyone in the frame, including the referee and the crowd.",
        "<b>Boxer selector</b>: gradient-boosted trees on 65 skeleton-only features over a 2-second window, trained on BoxingWeb's per-frame boxer poses. It decides which two people are fighting.",
        "<b>Identity</b>: the two boxers are cut out when they are apart, embedded with OSNet-AIN (a person re-identification network) and labelled A or B by a small classifier that keeps learning from its own answers; through overlaps the label is carried by motion.",
        "<b>Punch TCN</b>: a dilated temporal convolution network over each boxer's skeleton stream (plus the opponent's) outputs, frame by frame, no punch or one of six punch classes; events are cut from the smoothed probability. Type is the average of the skeleton model and a fine-tuned video model.",
        "<b>Contact scan</b>: for each counted punch, a fine-tuned VideoMAE-Large looks at 16 frames around the glove at 7 time offsets, finds the contact moment and answers landed / blocked / missed.",
        "<b>Bag/BOB route</b> (switch <i>--bob</i>): the target is found with OWLv2, punches with a no-opponent detector, contact by the glove reaching the target outline."]))

    # ------------------------------------------------------------------ 4
    s.append(PageBreak()); s.append(P("4. Phase 1 - punch type from skeletons", H1))
    s.append(P("The project started with the narrowest question: given a clip that is known to contain one punch, which of the six types is it? "
               "The success bar, set before the first experiment, was 60% balanced accuracy on videos the model never saw."))
    s.append(img("charts/phase1_type.png", W, "Figure 2. Each bar adds one change. The two orange bars use 3D: MediaPipe 3D joints with low-confidence joints filled, and an ensemble of 2D, lifted 3D and MediaPipe 3D models."))
    s.append(P("Challenges and how they were overcome", H2))
    s.append(table([["Challenge", "What we found", "What we did", "Result"],
                    ["Random 80/20 split looked great (91%)", "It measured 'which video is this', not punches: LOVO was 46.5%.", "Reported LOVO only, per video.", "Honest numbers from day one."],
                    ["Video V4 scored below chance (2-10%)", "Confusion matrix: jab<->cross, lead<->rear swapped. A wrist test showed the right hand leads: V4 is a southpaw (later confirmed from the source video).", "Mirror southpaws to orthodox before the model ('stance canonicalisation'). Mirroring everything as augmentation was tested and hurt (50%).", "57.1% -> 63.1%, goal met; 5 seeds confirmed it (63.2 \u00b1 0.3)."],
                    ["Random forest ignores time", "Punches of different length put the same phase in different columns.", "Resample to 10 frames + velocities, then 1D-CNN / LSTM / ST-GCN.", "66.8% (RF) -> 72.7% (CNN)."],
                    ["Plateau at ~73% across 4 architectures and 9 variants", "Errors cluster by camera angle: side-on jabs are read as lead hooks 47% of the time vs 9% front-on. The punch moves in depth, which 2D cannot see.", "Tried view augmentation, frontalness input and foreshortening (no gain), then real 3D: VideoPose3D lifting and MediaPipe 3D.", "Side-on jab->hook error 47% -> 22%."],
                    ["MediaPipe 3D first made things worse (57.9%)", "Rare low-confidence joints were extreme outliers that derailed training.", "Replace joints below a confidence threshold (5% of joint-frames).", "57.9% -> 72.5%; best single model 73.7%; ensemble 75.2%."],
                    ["Mac kernel panic during training", "PyTorch's built-in LSTM on Apple's GPU leaked driver memory (46.7 GB on a 36 GB machine).", "Memory guard imported before torch in every GPU script; a hand-written LSTM with no leak.", "No further crashes."]],
                   [3.2 * cm, 5.0 * cm, 4.6 * cm, 4.2 * cm]))
    s.append(P("Lesson: most of the gain came from understanding the data (stance, viewpoint, outliers), not from bigger models. Every model since the first CNN landed within 71-73% until real depth information was added.", KEY))

    # ------------------------------------------------------------------ 5
    s.append(P("5. Phase 2 - detection: is a punch happening?", H1))
    s.append(P("A real app sees continuous video, so it must also say <i>no punch</i>. The obvious negatives, the gaps between labelled punches, "
               "turned out to be contaminated: about 15% of random unlabelled moments contained a real punch (whole stretches of bag work were never labelled)."))
    s.append(bullets([
        "<b>Two independent detectors</b> were built so that their agreement could define clean data: a skeleton detector trained with positive-unlabelled learning, and a pixel detector (a linear probe on frozen VideoMAE-L features, mean per-video AUC 0.93).",
        "<b>A blind review page</b> showed the user 280 moments without scores, including hidden control punches (7 of 7 controls recognised). Against those verdicts the pixel detector reached AUC 0.96; 'pixel score < 0.02' contained 0 punches out of 24.",
        "<b>The no-punch class</b> was then cut only where both detectors were confident (2,981 clips). The 7-class model kept its type accuracy (73.2%) while missing 10.0% of punches with 2.8% false alarms.",
        "<b>A real-time counter</b> streams 8/12/16-frame windows every 2 frames with hysteresis and splits combinations where the punching hand switches. On a held-out video: 50 of 51 punches found, type right 78%, 0.4 ms per decision."]))

    # ------------------------------------------------------------------ 6
    s.append(P("6. Phase 3 - two fighters and the sparring wall", H1))
    s.append(P("To count per fighter, a training-free identity tracker (position, motion, clothing colour, limb proportions, clinch mode) and a "
               "stance-by-model rule (run the counter mirrored and not; more punches = the real stance; 10/10 videos right) were added. "
               "On real sparring the counts were clearly wrong, and since nobody wanted to hand-label sparring, an open vision-language model "
               "(Qwen3-VL-8B) was validated as an automatic judge first: 91% balanced agreement on our own labelled windows, with the pass bar fixed in advance."))
    s.append(img("charts/sparring.png", 0.62 * W, "Figure 3. About half of the old model's sparring counts were not punches: mostly a high guard read as 'uppercuts', or the opponent's arm in view."))
    s.append(table([["Attempt to fix sparring", "Result"],
                    ["Speed augmentation (fast no-punch clips)", "The model was fooled by fast movement only 4% of the time anyway: speed was not the cause."],
                    ["Fresh pose detection on every frame", "Found both fighters LESS often (54% -> 18.5% of frames)."],
                    ["'Hand must go towards the opponent' gate", "Removed 30% of counts but precision moved only for one fighter."],
                    ["Bigger judge (Qwen3-VL-32B)", "Same agreement (91.5%), 4x slower; kept the 8B."],
                    ["Speeding up the judge", "The vision tower's 3D convolution took 54 s per call; replaced by the mathematically identical matrix multiply: 65x faster (rank correlation of outputs 0.999)."]],
                   [6 * cm, W - 6 * cm]))
    s.append(P("Diagnosis: a model that has only ever seen one boxer alone has never seen guard movement, parries, feints or clinches, and "
               "MediaPipe saw both fighters in only 30-59% of sparring frames. The project had to move to real two-person data and a different pose model.", KEY))

    # ------------------------------------------------------------------ 7
    s.append(PageBreak()); s.append(P("7. Phase 4 - real bouts", H1))
    s.append(P("Two labelled real-bout datasets made it possible to measure instead of guess: the Olympic set (fixed gym camera, referee labels at the "
               "contact moment) and BoxingWeb (broadcast bouts with punch spans, types and per-frame poses of both boxers)."))
    s.append(P("7.1 The pose step was the bottleneck", H2))
    s.append(img("charts/pose_ceiling.png", W, "Figure 4. At the moment of a labelled punch, MediaPipe often returned only the referee or half a boxer. RTMPose saw both boxers almost always."))
    s.append(P("The old pipeline, unchanged, found only 24.8% of the 4,605 referee-labelled Olympic punches (chance 15.8%). This became the baseline every fix was judged against."))
    s.append(P("7.2 Picking the two boxers among referee and crowd", H2))
    s.append(table([["Version", "Idea", "Outcome"],
                    ["v1-v2", "9 hand-made features + logistic regression, free labels from the referee's boxes", "Puncher kept 95-98% but spectators picked; kit colour failed (navy reads as 'other', both in red)"],
                    ["Depth cues", "Feet on the ring floor; same apparent size for the pair", "Feet: no gain. Same size: wrong picks halved (21.3% -> 11.8%)"],
                    ["Learned (final)", "65 skeleton-only features, boosted trees, trained on BoxingWeb's per-frame truth", "Exactly the 2 boxers 88.5% of frames (was 61.4%); unseen Olympic gyms: picks away from the punch 0.7-1.3%"]],
                   [2.6 * cm, 7.0 * cm, W - 9.6 * cm]))
    s.append(P("7.3 Who is who", H2))
    s.append(img("charts/identity.png", W, "Figure 5. *Motion continuity even with the identity reset by hand at every camera cut. Person re-identification networks keep the two boxers apart for a whole round without any correction."))
    s.append(bullets(["Measured first: following skeletons frame to frame was a <b>coin flip</b> (51.8%), and 'keep the left/right order through an overlap' failed in 28% of overlaps. Broadcast camera cuts (4.9 per minute) were found with a pixel cut detector.",
                      "Recognition instead of tracking (the user's idea): re-identification is <b>self-correcting</b>, so a mistake does not carry over. In the full chain 94.4% of punches are credited to the right boxer.",
                      "On sparring clip A1, where both fighters wear black, 24 of 24 sampled frames were consistent: the model separates headgear, gloves and shoes."]))
    s.append(P("7.4 Punch detection rebuilt on real bouts", H2))
    s.append(table([["System (BoxingWeb test, 1,178 punches)", "Precision", "Recall", "F1", "Notes"],
                    ["Ours, 3-seed ensemble, loose matching", "-", "-", "0.786", "type (3 families) macro-F1 0.670"],
                    ["Ours, single model, loose", "0.769", "0.772", "0.771 \u00b1 0.006", "hand right 85%"],
                    ["Ours, strict (temporal IoU >= 0.5)", "0.667", "0.604", "0.634", "boundaries are the weak part"],
                    ["BoxMind (published)", "0.806", "0.763", "0.783", "matching rule not stated"],
                    ["Ours, full chain (no truth given)", "0.767", "0.692", "0.727", "selector + identity + detector"]],
                   [5.6 * cm, 1.7 * cm, 1.5 * cm, 2.4 * cm, W - 11.2 * cm]))
    s.append(img("charts/olympic_chain.png", W, "Figure 6. The same 29 Olympic bouts, the same scoring, before and after. Nothing in the new detector was trained on the Olympic set."))

    # ------------------------------------------------------------------ 8
    s.append(PageBreak()); s.append(P("8. Phase 5 - did it land?", H1))
    s.append(P("Landed was the hardest part. Four ideas failed before one worked, and the remaining gap was traced to its real cause."))
    s.append(table([["Attempt", "Why it should work", "What happened"],
                    ["Physics rule on the arm (sudden stop before full extension)", "A fist that hits something decelerates early", "<font color='#c92a2a'>Failed</font>: bag punches and air punches gave identical results; at 25-30 fps the fist stops within 1-3 frames"],
                    ["19 skeleton cues (reach to head, guard in the way, head snap...)", "Contact moves the target", "<font color='#c92a2a'>Failed</font>: 57% (always 'landed' 62%); boxers are ~250 px tall, so 10 cm is pose jitter"],
                    ["Frozen video features + linear probe", "Pixels show the glove on the head", "65.7%: better, not usable"],
                    ["Fine-tuned VideoMAE-B / -L on 4,105 contact clips", "Teach the network to look at contact", "<font color='#2b8a3e'>71.3% / 74.2%</font>, AUC 0.83; +BoxingWeb second head and averaging: 76.1%"]],
                   [4.8 * cm, 4.0 * cm, W - 8.8 * cm]))
    s.append(img("charts/landed.png", W, "Figure 7. 3-way accuracy on a gym the model never saw, given the referee's contact moment. The learning curve was still rising (62.3% / 70.8% / 74.1% at 25 / 50 / 100% of the data)."))
    s.append(P("8.1 Behind our own detector the gain disappeared, and why", H2))
    s.append(P("Chained after our detector the verdict first dropped to 61%, below always-'landed'. A breakdown showed the landed model was fine "
               "when shown the right moment and the right glove (72%); the chain's contact time was within 2 frames only 27% of the time. "
               "Training on mistimed clips did not help ('training on mistimed clips does not fix mistimed clips'). What helped was fixing the timing upstream:"))
    s.append(bullets(["a skeleton contact-moment model (timing within 2 frames: 27% -> 48%);",
                      "letting the video model itself scan 7 time offsets and pick the contact (84% within 2 frames on referee clips);",
                      "<b>finding a labelling misunderstanding</b>: the referee's box sits on the <i>target</i>, not the puncher. The 'nearest glove' rule credited the defender in about a third of the labels. With the corrected rule and fair one-to-one matching, the chained verdict reached <b>69.1%</b>."]))
    s.append(img("charts/landed_chain.png", W, "Figure 8. Steps that recovered the landed verdict inside the full chain."))
    s.append(P("Per bout, the totals are more useful than single verdicts: landed punches per bout within 7.6% of the referee (median, correlation 0.96), thrown per fighter within 10.9%, landed per fighter within 17.5%.", KEY))

    # ------------------------------------------------------------------ 9
    s.append(P("9. Phase 6 - type fusion, false counts, bag and BOB", H1))
    s.append(bullets(["<b>Punch type from pixels</b>: a fine-tuned video model alone was no better than the skeleton (68.4% vs 72.3% family accuracy), but they fail on different punches; averaging gives <b>77.0%</b> on true spans and 78.1% behind our own detector. Adopted.",
                      "<b>Rejecting false counts</b>: a cheap rejector (AUC 0.78) and a pixel 'real punch' head (AUC 0.74-0.78) traded found punches for precision with no net F1 gain; not adopted. False counts were mixed: clinches, the other boxer's movement, referee breaks, and a few real punches the referee did not label.",
                      "<b>Things that looked promising and were dropped after measuring</b>: a marker drawn on the puncher's arm, a wider time window (71.2% vs 74.1%), self-training on BoxingWeb (the model's own confident answers were 92% 'landed'), a smaller landed model (faster, but less tolerant of rough timing).",
                      "<b>Bag/BOB route</b>: OWLv2 finds the dummy or bag in 99-100% of sampled frames. The bout-trained detector missed many bag jabs, so a rescue rule counts a glove entering the target fast; the box follows a swinging bag. The route is chosen by a command-line switch, never by 'only one boxer found', because a boxer can be lost for seconds and a BOB is itself drawn as a person by the pose model."]))

    # ------------------------------------------------------------------ 10
    s.append(P("10. Phase 7 - the rendered video", H1))
    s.append(P("The output video is styled like a fighting game, with one rule from the user: every effect must be driven by something the system "
               "really measured. Health drops only on LANDED punches, by type; GUARD! and WHIFF! show blocked and missed verdicts; trails are coloured by punch family; "
               "critical hits are uppercuts and counters; the results card is built from the real counts."))
    s.append(img_row(["intro.jpg", "guard.jpg"], "Figure 9. Intro card, and a blocked punch (GUARD!) with damage from an earlier hit."))
    s.append(img_row(["ko.jpg", "results.jpg"], "Figure 10. K.O. finish with the slow-motion replay window, and the results card (winner by first K.O., then health, then landed punches)."))
    s.append(P("Dropped on purpose: a super meter and other game mechanics with no measured signal behind them; full-body silhouettes (a person-mask model on every frame would cost about 3 hours per video on the laptop).", SM))

    # ------------------------------------------------------------------ 11
    s.append(P("11. Engineering problems along the way", H1))
    s.append(table([["Problem", "Fix"],
                    ["Mac kernel panic: LSTM on Apple's GPU leaked memory", "Memory cap + watchdog before torch; leak-free LSTM; test memory on a short run before a long one"],
                    ["HPC sync froze after the laptop slept (dead shared ssh connection)", "Time limits and keep-alives on every ssh/rsync call; stall detection in the progress bar"],
                    ["HPC jobs frozen on Lustre file locks; one unkillable process", "Jobs write to node-local disk and publish snapshots by copy + rename; separate HPC folders per workstream"],
                    ["onnxruntime silently ran on CPU on the cluster", "Removed the CPU build that hid the GPU one; pinned a version that matches the node driver (CUDA 12)"],
                    ["Vision-language model took 52 s per clip", "Profiled: the patch-embedding 3D convolution; replaced by an identical linear map: 0.8 s"],
                    ["NumPy archive re-decompressed on every access (tracker took 18 min)", "Load arrays once: 9 s"],
                    ["Pickle files from a dataset could run code when loaded", "Disassembled first, then loaded with a restricted unpickler that allows only NumPy arrays"],
                    ["A 1.9 GB log blocked every push to GitHub for a day", "Removed from the unpushed history; logs and rendered videos are now ignored"],
                    ["Third-party footage must not be republished", "Rendered videos kept local; only the Creative Commons clip is shown, with credit"]],
                   [7.2 * cm, W - 7.2 * cm]))

    # ------------------------------------------------------------------ 12
    s.append(P("12. Challenges and how they were overcome (summary)", H1))
    s.append(table([["#", "Challenge", "How it was overcome", "Evidence"],
                    ["1", "Optimistic random splits", "Leave-one-video-out / held-out gym everywhere, chance levels by shifted labels", "46.5% LOVO vs 91% random"],
                    ["2", "Southpaw video wrecked accuracy", "Detected from wrist motion, mirrored to orthodox; later stance chosen by the model itself", "V4 2% -> 38%; stance 10/10 videos"],
                    ["3", "Camera angle (side-on jabs look like hooks)", "Real 3D from MediaPipe with confidence filling", "Side-on error 47% -> 22%"],
                    ["4", "Unlabelled punches hiding in 'no punch' data", "Two independent detectors + blind human review with control punches", "0/24 punches below the strict pixel threshold"],
                    ["5", "Sparring counts half fake", "Validated an AI judge, ruled out speed and gating, moved to real two-person data", "Diagnosis confirmed on 3 datasets"],
                    ["6", "Pose model missed boxers in real bouts", "Switched from MediaPipe to RTMPose", "Both boxers seen 15% -> 96%"],
                    ["7", "Referee and crowd picked as boxers", "Learned skeleton-only selector trained on per-frame truth", "Wrong picks near punches 0.7-1.3%"],
                    ["8", "Identity swaps (coin flip by motion)", "Person re-identification, self-updating classifier", "99.5-99.8% of frames"],
                    ["9", "Contact invisible to skeletons", "Fine-tuned video model at the glove", "57% -> 74-76%"],
                    ["10", "Landed model failed inside the chain", "Traced to timing and hand; contact scan and corrected puncher rule", "61% -> 69.1%"],
                    ["11", "Dataset label misread (box on the target)", "Checked contact sheets, rewrote the puncher rule", "Rules agreed on only 65% of labels"],
                    ["12", "Compute: slow models, flaky cluster", "Profiling, identical faster operators, robust job and sync design", "65x speed-up; no lost runs after fixes"]],
                   [0.7 * cm, 4.4 * cm, 6.8 * cm, W - 11.9 * cm]))

    # ------------------------------------------------------------------ 13
    s.append(P("13. Limitations and next steps", H1))
    s.append(bullets(["<b>Sparring accuracy is not measured.</b> No labels exist for the sparring clips and the user chose not to label them; accuracy is quoted from the Olympic and BoxingWeb bouts only.",
                      "<b>Landed inside the chain</b> is 69% against 63% for always-'landed'; per-bout totals are the reliable output today. Blocked punches remain the weakest class.",
                      "<b>Punch boundaries</b>: with strict temporal matching detection F1 falls to 0.63, below the published BoxMind number if that is their rule.",
                      "<b>Not real time yet</b>: the video models run per punch (about 0.25-0.3 s each on a laptop GPU); a phone would need smaller models trained the same way.",
                      "<b>Human ceiling</b>: a judging page for 90 referee-labelled punches was built but not completed; it would show how far 74-76% is from a person.",
                      "<b>Next</b>: more labelled contact clips (the learning curve is still rising), better punch boundaries, a lighter landed model, and finishing the human ceiling test."]))

    # ------------------------------------------------------------------ 14
    s.append(P("14. Credits and licences", H1))
    s.append(bullets(["BoxingVI: Kumar et al., arXiv 2511.16524. Olympic Boxing Punch Classification Video Dataset: Stefa\u0144ski, Kozak, Jach, <i>Entropy</i> 26(8):617, 2024 (non-commercial). BoxingWeb / BoxMind: Wang et al., arXiv 2601.11492.",
                      "Models: RTMPose and YOLOX via rtmlib; MediaPipe Pose Landmarker; VideoPose3D (Meta); VideoMAE (MCG-NJU, CC-BY-NC-4.0); OSNet / OSNet-AIN (K. Zhou, MIT); Qwen3-VL (Apache-2.0); OWLv2 (Google).",
                      "Demo footage: 'My first boxing sparring' by Michal Hradi\u0161, YouTube, Creative Commons Attribution. No dataset footage or other third-party video is redistributed. VideoMAE and the Olympic dataset are non-commercial, so the trained landed model is for research and personal use only."], SM))
    doc.build(s); print("report ->", OUT)


if __name__ == "__main__":
    build()
