import json, joblib, numpy as np, math

M = joblib.load("demo/models.joblib")

def export(vec, model):
    vocab = vec.vocabulary_                      # term -> col
    idf = vec.idf_
    coef = model.coef_[0]
    inter = float(model.intercept_[0])
    # term -> [idf, coef] (only terms in vocab matter; L2 norm is over present terms)
    terms = {t: [round(float(idf[c]), 5), round(float(coef[c]), 6)] for t, c in vocab.items()}
    params = dict(
        lowercase=vec.lowercase, sublinear_tf=vec.sublinear_tf,
        ngram_range=list(vec.ngram_range), norm=vec.norm,
        token_pattern=vec.token_pattern,
    )
    return {"terms": terms, "intercept": inter, "params": params}

out = {"all": export(*M["all"]), "mix": export(*M["mix"])}
# sanity: vocab sizes + a reference score we can check JS against
for k in ("all", "mix"):
    print(k, "vocab:", len(out[k]["terms"]), "intercept:", out[k]["intercept"])
print("params:", out["all"]["params"])

json.dump(out, open("demo/web/models.json", "w"))
import os
print("json bytes:", os.path.getsize("demo/web/models.json"))
