"""Human ceiling for landed / blocked / missed from this one camera: cut 90 referee punches (30 per class, both gyms,
fixed seed) as short clips and write a local page where a person judges each one; the page scores against the referee.
Each clip: 1.2 s around the contact (0.7 s before, 0.5 s after), crop 2x the model's glove crop so both boxers show,
played at normal speed then repeatable at half speed in the page. A yellow square (the referee's box) is shown until
0.24 s before contact, so the viewer knows WHICH punch to judge when both boxers punch. The model's own answers for the same punches
(held-out gym, referee's moment) are stored in the page for a side-by-side score.
Output (not committed, third-party footage): external/human_test/index.html + clips/*.mp4
    python olympic/human_test.py
"""
import glob, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
N_PER, PRE, POST = 30, 35, 25

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"))); y = np.array([c["y"] for c in clips])
    P = np.full((len(y), 3), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", "cl_large_kam*.npz"))):
        z = np.load(f); P[z["idx"]] = z["P3"]
    rng = np.random.default_rng(7); pick = np.concatenate([rng.choice(np.flatnonzero(y == k), N_PER, replace=False) for k in range(3)]); rng.shuffle(pick)
    out = os.path.join(ROOT, "external", "human_test"); os.makedirs(os.path.join(out, "clips"), exist_ok=True); items = []
    for n, i in enumerate(pick):
        c = clips[i]; src = os.path.join(ROOT, "external", "raw", c["id"] + ".mp4"); side = int(min(max(2 * c["side"], 360), 1080)) // 2 * 2
        x = int(np.clip(c["cx"] - side / 2, 0, 1920 - side)); yy = int(np.clip(c["cy"] - side / 2, 0, 1080 - side)); t0 = max(c["frame"] - PRE, 0) / 50
        dst = os.path.join(out, "clips", f"{n:03d}.mp4")
        k = 540 / side; mx, my = int(np.clip((c["bx"] - x) * k - 45, 0, 450)), int(np.clip((c["by"] - yy) * k - 45, 0, 450))      # yellow square = where the referee marked THIS punch
        if not os.path.exists(dst):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0:.3f}", "-i", src, "-frames:v", str(PRE + POST), "-vf", f"crop={side}:{side}:{x}:{yy},scale=540:540,drawbox=x={mx}:y={my}:w=90:h=90:color=yellow:t=3:enable='lt(n,{PRE - 12})'",
                            "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", dst], check=True)
        items.append(dict(f=f"clips/{n:03d}.mp4", y=int(y[i]), m=int(P[i].argmax()), hand=c["hand"]))
        print(n, c["id"], c["outcome"], flush=True)
    page = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Landed judging test</title>
<style>body{font:16px system-ui;margin:0 auto;max-width:600px;padding:16px;background:#111;color:#eee}video{width:100%;background:#000}button{font:inherit;padding:12px 16px;margin:6px 4px 0 0;border:0;border-radius:6px;background:#2d6cdf;color:#fff;cursor:pointer}button.alt{background:#444}#res{white-space:pre-wrap}</style>
<h2>Landed, blocked or missed?</h2><p id=info></p><video id=v muted playsinline loop autoplay></video>
<div id=btn><button onclick="ans(0)">Landed (1)</button><button onclick="ans(1)">Blocked (2)</button><button onclick="ans(2)">Missed (3)</button><button class=alt onclick="slow()">Half speed (S)</button><button class=alt onclick="score()">Score so far</button></div><div id=res></div>
<script>const D=__DATA__,N=['landed','blocked','missed'];let i=0,A=[];try{A=JSON.parse(localStorage.getItem('ht2')||'[]')}catch(e){}i=A.length;const v=document.getElementById('v');
function show(){if(i>=D.length)return done();v.playbackRate=1;v.src=D[i].f;v.play();document.getElementById('info').textContent='Clip '+(i+1)+' of '+D.length+': the punch is thrown with the '+D[i].hand+' hand, contact about 0.7 s in. Judge only the punch that arrives where the yellow square was.'}
function slow(){v.playbackRate=v.playbackRate==1?0.5:1}
function ans(k){if(i>=D.length)return;A.push(k);try{localStorage.setItem('ht2',JSON.stringify(A))}catch(e){}i++;show()}
function score(){let n=A.length,h=0,m=0,ag=0,c=[[0,0,0],[0,0,0],[0,0,0]];if(!n)return;D.slice(0,n).forEach((d,j)=>{h+=A[j]==d.y;m+=d.m==d.y;ag+=A[j]==d.m;c[d.y][A[j]]++});
let t='After '+n+' clips:\\nYou agree with the referee on '+h+' of '+n+' ('+Math.round(100*h/n)+'%).\\nThe model agrees with the referee on '+m+' ('+Math.round(100*m/n)+'%).\\nYou and the model agree on '+ag+'.\\n\\nReferee said -> you said (landed, blocked, missed):\\n';
N.forEach((n2,k)=>t+=n2+': '+c[k].join(', ')+'\\n');t+='\\nAnswers: '+JSON.stringify(A);document.getElementById('res').textContent=t}
function done(){v.pause();document.getElementById('btn').style.display='none';score();document.getElementById('info').textContent='Finished.'}
document.addEventListener('keydown',e=>{if('123'.includes(e.key))ans(+e.key-1);if(e.key=='s'||e.key=='S')slow()});show()</script>"""
    open(os.path.join(out, "index.html"), "w").write(page.replace("__DATA__", json.dumps(items)))
    m = np.mean([it["m"] == it["y"] for it in items]); print(f"{len(items)} clips written to {out}; the model is right on {m:.1%} of these (balanced classes)")

if __name__ == "__main__":
    main()
