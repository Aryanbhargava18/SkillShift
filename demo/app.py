"""Interactive SkillShift demo (Hugging Face Spaces / Gradio).

Paste a SKILL.md and see two detectors score it: one trained on the whole merged benchmark (which keys
on source style and over-blocks) and one trained only on both-label sources (the fix). The gap between
them is the shortcut this project is about.

The two word-TF-IDF + logistic-regression detectors are fit on startup from MaliciousSkillBench. Fitting
runs in a background thread so the web server binds immediately and the Space passes its health check;
the first scored request waits for fitting to finish (~15 s), later ones are instant.
"""

import os
os.environ.setdefault("GRADIO_SSR_MODE", "false")  # plain server, no SSR Node proxy (health-check safe)

import threading

import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}
_STATE = {}
_LOCK = threading.Lock()


def _fit(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40_000, sublinear_tf=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)
    model.fit(vec.fit_transform(train.text), train.label)
    return vec, model


def _load():
    d = snapshot_download("ProtectSkills/MaliciousSkillBench", repo_type="dataset", local_dir="msb")
    p = pd.read_parquet(f"{d}/primary.parquet")
    p["text"] = p.skill_text.fillna(p.public_skill_text).fillna("")
    sp = pd.read_parquet(f"{d}/splits/source_disjoint.parquet")
    df = sp[["benchmark_id", "split"]].merge(
        p[["benchmark_id", "text", "label", "source_name"]], on="benchmark_id")
    df["label"] = df.label.astype(int)
    train = df[(df.split == "train")]
    return _fit(train), _fit(train[train.source_name.isin(MIXED)])


def _ensure():
    with _LOCK:
        if not _STATE:
            print("fitting detectors...", flush=True)
            _STATE["all"], _STATE["mix"] = _load()
            print("ready.", flush=True)
    return _STATE


# Warm up in the background so launch() can bind the server right away.
threading.Thread(target=_ensure, daemon=True).start()


def score(text):
    if not text or len(text.strip()) < 20:
        return "Paste a `SKILL.md` (at least a few lines) to compare the two detectors."
    s = _ensure()
    vec_all, m_all = s["all"]
    vec_mix, m_mix = s["mix"]
    pa = float(m_all.predict_proba(vec_all.transform([text]))[:, 1][0])
    pm = float(m_mix.predict_proba(vec_mix.transform([text]))[:, 1][0])
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
        demo.launch(ssr_mode=False)
    except TypeError:
        demo.launch()
