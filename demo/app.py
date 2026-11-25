"""Interactive SkillShift demo (Hugging Face Spaces / Gradio).

Paste a SKILL.md and see two detectors score it: one trained on the whole merged benchmark (which keys
on source style and over-blocks) and one trained only on both-label sources (the fix). The gap between
them is the shortcut this project is about.

The two word-TF-IDF + logistic-regression detectors are precomputed and shipped in models.joblib, so the
app loads them instantly and the Space starts without a dataset download or any training. Regenerate the
file with scripts/build_demo_models.py.
"""

import os
os.environ.setdefault("GRADIO_SSR_MODE", "false")  # plain server, no SSR Node proxy (health-check safe)

import time

import gradio as gr
import joblib

_M = joblib.load(os.path.join(os.path.dirname(__file__), "models.joblib"))
VEC_ALL, M_ALL = _M["all"]
VEC_MIX, M_MIX = _M["mix"]


def score(text):
    if not text or len(text.strip()) < 20:
        return "Paste a `SKILL.md` (at least a few lines) to compare the two detectors."
    pa = float(M_ALL.predict_proba(VEC_ALL.transform([text]))[:, 1][0])
    pm = float(M_MIX.predict_proba(VEC_MIX.transform([text]))[:, 1][0])
    va = "MALICIOUS" if pa >= 0.5 else "benign"
    vm = "MALICIOUS" if pm >= 0.5 else "benign"
    note = ("\n\n_The two detectors disagree. The all-data detector is reacting to source **style**, not "
            "content: that is the shortcut this project diagnoses, and the fix trades recall to avoid it._"
            if va != vm else "")
    return (f"**All-data detector** (trained on the whole merged benchmark): "
            f"**{va}** ({pa * 100:.0f}% malicious)\n\n"
            f"**Fixed detector** (trained only on both-label sources): "
            f"**{vm}** ({pm * 100:.0f}% malicious){note}")


demo = gr.Interface(
    fn=score,
    inputs=gr.Textbox(lines=16, label="SKILL.md content", placeholder="Paste a SKILL.md file here..."),
    outputs=gr.Markdown(label="Detector verdicts"),
    title="SkillShift — do malicious-skill detectors learn the source or the threat?",
    description=("Two text detectors for malicious agent skills. On 400 unlabeled public GitHub skills the "
                 "all-data detector flags ~30% as malicious and the fixed one ~5%. Paste a SKILL.md to see "
                 "them differ. github.com/Aryanbhargava18/SkillShift"),
)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    # Keep the process (and the web server thread) alive if launch() returns instead of blocking.
    while True:
        time.sleep(3600)
