"""Interactive SkillShift demo (Hugging Face Spaces / Gradio).

Paste a SKILL.md and see two detectors score it: one trained on the whole merged benchmark (which
keys on source style and over-blocks) and one trained only on both-label sources (the fix). The gap
between them is the shortcut this project is about. Self-contained: downloads MaliciousSkillBench and
fits the two word-TF-IDF + logistic-regression detectors on startup (~15 s cold start).
"""

import gradio as gr
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}


def load_split():
    d = snapshot_download("ProtectSkills/MaliciousSkillBench", repo_type="dataset", local_dir="msb")
    p = pd.read_parquet(f"{d}/primary.parquet")
    p["text"] = p.skill_text.fillna(p.public_skill_text).fillna("")
    sp = pd.read_parquet(f"{d}/splits/source_disjoint.parquet")
    df = sp[["benchmark_id", "split"]].merge(p[["benchmark_id", "text", "label", "source_name"]],
                                             on="benchmark_id")
    df["label"] = df.label.astype(int)
    return df[df.split != "excluded"]


def fit(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
    m.fit(vec.fit_transform(train.text), train.label)
    return vec, m


print("loading data and fitting detectors...")
_df = load_split()
_train = _df[_df.split == "train"]
VEC_ALL, M_ALL = fit(_train)
VEC_MIX, M_MIX = fit(_train[_train.source_name.isin(MIXED)])
print("ready.")


def score(text):
    if not text or len(text.strip()) < 20:
        return "Paste a SKILL.md (at least a few lines)."
    pa = float(M_ALL.predict_proba(VEC_ALL.transform([text]))[:, 1][0])
    pm = float(M_MIX.predict_proba(VEC_MIX.transform([text]))[:, 1][0])
    va = "MALICIOUS" if pa >= 0.5 else "benign"
    vm = "MALICIOUS" if pm >= 0.5 else "benign"
    note = ("\n\n_The two detectors disagree — the all-data detector is reacting to source **style**, "
            "not content. That is the shortcut this project diagnoses and fixes._"
            if va != vm else "")
    return (f"**All-data detector** (trained on the whole merged benchmark): "
            f"**{va}** ({pa*100:.0f}% malicious)\n\n"
            f"**Fixed detector** (trained only on both-label sources): "
            f"**{vm}** ({pm*100:.0f}% malicious){note}")


demo = gr.Interface(
    fn=score,
    inputs=gr.Textbox(lines=16, label="SKILL.md content", placeholder="Paste a SKILL.md file here..."),
    outputs=gr.Markdown(label="Detector verdicts"),
    title="SkillShift — do malicious-skill detectors learn the source or the threat?",
    description=("Two text detectors for malicious agent skills. The all-data one wrongly flags ~30% "
                 "of real GitHub skills; the fixed one flags ~5%. Paste a SKILL.md to see them differ. "
                 "github.com/Aryanbhargava18/SkillShift"),
)

if __name__ == "__main__":
    demo.launch()
