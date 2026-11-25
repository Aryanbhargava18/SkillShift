"""Interactive SkillShift demo (Hugging Face Spaces / Gradio).

Paste a SKILL.md and see two detectors score it: one trained on the whole merged benchmark (which keys
on source style and over-blocks) and one trained only on both-label sources (the fix). The gap between
them is the shortcut this project is about. The two word-TF-IDF + logistic-regression detectors are fit
from MaliciousSkillBench on startup (a few seconds).
"""

import os
os.environ.setdefault("GRADIO_SSR_MODE", "false")  # plain server, no SSR Node proxy (health-check safe)

import time

import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}


def _fit(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40_000, sublinear_tf=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)
    model.fit(vec.fit_transform(train.text), train.label)
    return vec, model


print("fitting detectors...", flush=True)
_d = snapshot_download("ProtectSkills/MaliciousSkillBench", repo_type="dataset", local_dir="msb")
_p = pd.read_parquet(f"{_d}/primary.parquet")
_p["text"] = _p.skill_text.fillna(_p.public_skill_text).fillna("")
_sp = pd.read_parquet(f"{_d}/splits/source_disjoint.parquet")
_df = _sp[["benchmark_id", "split"]].merge(
    _p[["benchmark_id", "text", "label", "source_name"]], on="benchmark_id")
_df["label"] = _df.label.astype(int)
_train = _df[_df.split == "train"]
VEC_ALL, M_ALL = _fit(_train)
VEC_MIX, M_MIX = _fit(_train[_train.source_name.isin(MIXED)])
print("ready.", flush=True)


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
    allow_flagging="never",
)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    # Some gradio/Spaces combinations return from launch() instead of blocking; keep the process
    # (and therefore the web server thread) alive so the Space stays up.
    while True:
        time.sleep(3600)
