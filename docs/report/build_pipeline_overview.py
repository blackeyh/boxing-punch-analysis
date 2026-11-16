"""Builds docs/Pipeline_Overview.pdf: a short description of the FINAL pipeline (how it works, not how we got there).
    python docs/report/build_pipeline_overview.py
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, PageBreak, Spacer

from build_report import (ACC, B, CAP, DOCS, H1, H2, H3, KEY, MUTED, SM, W, P, bullets, img, img_row, on_page, pipeline_diagram, table)

OUT = os.path.join(DOCS, "Pipeline_Overview.pdf")


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                          title="Boxing Punch Analysis - Pipeline Overview", author="blackeyh")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=on_page)])
    s = [P("PIPELINE OVERVIEW", ParagraphStyle("k", parent=SM, textColor=ACC, fontName="Helvetica-Bold", fontSize=9)),
         P("How the boxing punch analyser works", ParagraphStyle("T", parent=H1, fontSize=22, leading=27)),
         P("From an ordinary video of a fight to, for each boxer: every punch thrown, its type, and whether it landed, was blocked or missed.",
           ParagraphStyle("st", parent=B, fontSize=11.5, leading=16, textColor=MUTED)), Spacer(1, 0.3 * cm)]

    s.append(P("In one picture", H2))
    s.append(pipeline_diagram())
    s.append(P("Figure 1. Six stages. The first four work on skeletons (cheap, every frame); the last one looks at pixels, only around each counted punch.", CAP))
    s.append(P("<b>Input:</b> one video, one camera, 24-60 fps, any resolution from 720p. People besides the boxers (referee, crowd, coaches) are fine. "
               "<b>Output:</b> a list of every punch (which boxer, when, type, verdict and confidences), per-boxer totals, and an annotated video.", KEY))

    s.append(P("Stage by stage", H2))
    s.append(table([["Stage", "What it does", "Model", "Trained on"],
                    ["1. Pose", "Finds every person in every frame and their 17 body joints (2D).", "YOLOX-m person detector + RTMPose-m (rtmlib, ONNX), up to 20 people per frame", "Off the shelf (COCO)"],
                    ["2. Boxer selector", "Decides which two people are the fighters.", "Gradient-boosted trees on 65 skeleton-only features over a 2 s window (guard height, arm and hip motion, size relative to others, nearest same-size person...)", "BoxingWeb per-frame boxer poses (425k person-frames)"],
                    ["3. Identity", "Keeps boxer A as A and B as B for the whole video, through overlaps and camera cuts.", "OSNet-AIN re-identification embeddings + a per-video A-vs-B classifier that keeps learning from its own answers; motion carries the label through overlaps", "Off the shelf (MSMT17) + learns each video on the fly"],
                    ["4. Punch detector", "For each boxer, frame by frame: no punch, or which punch.", "Dilated temporal CNN over 157 skeleton channels (own pose, motion, opponent's pose), +-1 s context, 3 models averaged", "BoxingWeb (40 rounds, 5,537 punches)"],
                    ["5. Contact + verdict", "Finds the contact moment and judges landed / blocked / missed.", "Skeleton contact model for a rough moment, then a fine-tuned VideoMAE-Large scans 7 time offsets of a glove-centred clip", "Olympic referee labels (3,810 punches) "],
                    ["6. Type fusion", "Final punch type.", "Average of the detector's type and a fine-tuned VideoMAE type model; the hand comes from the detector", "BoxingWeb"]],
                   [2.4 * cm, 4.2 * cm, 6.6 * cm, W - 13.2 * cm]))

    s.append(PageBreak()); s.append(P("How each stage works", H1))
    s.append(P("1. Pose", H3))
    s.append(P("A person detector draws a box around everyone; the pose model places 17 joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) "
               "with a confidence each. Nothing is boxing-specific yet: the crowd gets skeletons too."))
    s.append(P("2. Boxer selector", H3))
    s.append(P("Skeletons are linked over time into short tracks. For every person, features over a 2-second window describe how a boxer moves and stands "
               "(hands held high in guard, quick arm movement, mobile hips, legs apart) and how they relate to others (similar size and distance to another person). "
               "A boosted-tree classifier gives each person a probability of being a boxer; the two most likely are kept. It uses skeletons only, "
               "so kit colour or camera position do not matter."))
    s.append(P("3. Identity", H3))
    s.append(P("The video is split into stretches where the two boxers stand apart, and at camera cuts. From each stretch, crops of each boxer go through "
               "OSNet-AIN, a network trained to recognise people across cameras. The first stretch defines A and B; every later stretch is labelled by a "
               "small classifier on these embeddings, which adds its confident answers to its training set as the video goes on. While the boxers overlap, "
               "each keeps its label by following motion. Because every stretch is recognised afresh, a mistake does not carry over."))
    s.append(P("4. Punch detector", H3))
    s.append(P("Each boxer becomes a stream of 157 numbers per frame: their joints relative to their hips and scaled by body size, how fast each joint moves, "
               "the opponent's joints in the same frame of reference, and joint confidences. A temporal convolution network reads about one second either "
               "side and outputs, for every frame, the probability of no punch and of each of six punches (left/right straight, hook, uppercut). "
               "A punch is counted where the punch probability rises above 0.40; a run is split where the punching hand changes, so a jab-cross counts as two."))
    s.append(P("5. Contact moment and verdict", H3))
    s.append(P("A second skeleton network trained on referee contact moments gives a rough time of contact for each counted punch. Around it, the video "
               "model looks at a square crop centred on the punching glove: 16 frames (0.32 s) at 50 fps, at 7 positions spaced 4 frames apart. For each "
               "position it answers landed, blocked, missed, or 'contact not centred here'. The most centred position wins and gives the verdict and its confidence."))
    s.append(P("6. Type fusion and outputs", H3))
    s.append(P("A second video clip around the middle of the punch goes through a fine-tuned type model; its straight / hook / uppercut probabilities are averaged "
               "with the detector's (the two make different mistakes). The hand stays the detector's. Every punch is written to a JSON list; per-boxer totals "
               "(thrown, landed, blocked, missed, by type) go to a summary."))
    s.append(P("Rendered video", H3))
    s.append(P("Skeletons in each boxer's colour, a label per punch with its verdict, and a fighting-game HUD in which every effect follows a real output: health "
               "drops only on landed punches (more for heavier types), GUARD! and WHIFF! mark blocked and missed punches, critical hits are uppercuts and counters, "
               "and the closing card is built from the counts."))
    s.append(img_row(["hit.jpg", "results.jpg"], "Figure 2. A landed punch with its type, verdict and damage; the results card at the end (demo footage: Michal Hradiš, CC BY)."))
    s.append(P("Bag and BOB mode", H3))
    s.append(P("With <i>--bob</i> or <i>--bag</i>, one boxer and a target: OWLv2 (a detector prompted with text such as 'a punching bag') finds the target and "
               "follows it as it swings; a detector trained without an opponent counts punches; a punch is 'touched' when the glove reaches the target's outline."))

    s.append(P("How well it works", H1))
    s.append(table([["Measure", "Test data (never trained on)", "Result"],
                    ["Punches found / counts that are real", "29 Olympic bouts, 4,605 referee-labelled punches", "79% / 70%, F1 0.75"],
                    ["Punch detection F1", "BoxingWeb test rounds", "0.77-0.79 (BoxMind, published: 0.78)"],
                    ["Punch credited to the right boxer", "BoxingWeb test rounds", "94%"],
                    ["Right identity, whole round", "BoxingWeb test rounds", "99.5% of frames"],
                    ["Punch family (straight / hook / uppercut)", "BoxingWeb test, behind our detector", "78% when the hand is right"],
                    ["Landed / blocked / missed", "Olympic, unseen gym, inside the full chain", "69% (always 'landed': 63%)"],
                    ["Landed punches per bout", "Olympic, full chain", "within 7.6% of the referee (median)"]],
                   [5.4 * cm, 6.2 * cm, W - 11.6 * cm]))
    s.append(P("Per-bout totals are the most reliable output; single verdicts are less certain, and blocked punches are the hardest class.", SM))

    s.append(P("Speed and hardware", H2))
    s.append(table([["Stage", "Where the time goes"],
                    ["Pose", "Every frame. About 40 frames/s on a workstation GPU; 1-4 frames/s on a laptop CPU. The main cost on a laptop."],
                    ["Selector, identity, detector", "Light; the detector looks about 1 s ahead, so results trail the video by about a second."],
                    ["Contact scan + type", "Only per counted punch: about 2 s of laptop-GPU work per punch (7 clips of ~0.25 s, plus the type clip)."]],
                   [5.0 * cm, W - 5.0 * cm]))
    s.append(P("Today it runs as an after-the-fight analyser: stages are cached and the video is rendered at the end. On a desktop NVIDIA GPU it could run "
               "nearly live, with verdicts appearing a few seconds after each punch."))

    s.append(P("Running it", H2))
    s.append(P("<font face='Courier'>python pipeline/count_video.py CLIP</font> - sparring or bout: counts, verdicts and the rendered video<br/>"
               "<font face='Courier'>python pipeline/count_video.py CLIP --bob</font> - one boxer on a BOB dummy or a bag<br/>"
               "<font face='Courier'>python pipeline/count_video.py CLIP --no-render</font> - counts only; <font face='Courier'>--no-cards</font> - no intro / results card", B))
    s.append(P("Outputs: <i>CLIP_punches.json</i> (every punch), <i>CLIP_summary.txt</i> (per-boxer totals by type), <i>CLIP.mp4</i> (annotated video).", B))
    s.append(Spacer(1, 0.3 * cm))
    s.append(P("Models: RTMPose / YOLOX (rtmlib), OSNet-AIN (MIT), VideoMAE (CC-BY-NC-4.0), OWLv2. Training data: BoxingWeb (Wang et al., arXiv 2601.11492) and the "
               "Olympic Boxing Punch Classification dataset (Stefański, Kozak, Jach, Entropy 2024, non-commercial). Research and personal use only. "
               "The full story of how this pipeline was reached is in the project report.", SM))
    doc.build(s); print("overview ->", OUT)


if __name__ == "__main__":
    build()
