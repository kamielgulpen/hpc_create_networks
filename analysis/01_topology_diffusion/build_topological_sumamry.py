import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os
os.environ["PIPELINE_SCALE"] = "0.10"  # Limit the number of threads used by OpenMP to 1
import data_lake as dl

LAYER = "werkschool"

N_ITER         = 25
CORR_THRESHOLD = 0.75          # must match the threshold the consensus matrix was built at
SENTINEL       = "random"
XFLOOR         = 200.0         # importance must exceed noise floor by this factor
FREQ_BAR       = 0.9          # group must be present (any member selected) in >= this fraction
DROP           = ["optimize", "max_coreness"]   # config col + coreness duplicate

print(dl.ROOT)
frames = []
for i in range(N_ITER):
    p = (dl.ROOT / "analysis" / "topology_diffusion" / "rf_shap_analysis"
         / LAYER / "topology_to_diffusion" / "threshold_0.2" / f"seed_{i}" / "shap_importance.csv")
    if p.exists():
        frames.append(pd.read_csv(p, index_col=0).squeeze("columns").rename(i))
print(frames)
allimp  = pd.concat(frames, axis=1)
summary = pd.DataFrame({"mean": allimp.mean(axis=1), "n_seeds": allimp.notna().sum(axis=1)})
summary = summary.drop(index=[d for d in DROP if d in summary.index])

consensus = pd.read_csv(
    dl.ROOT / "analysis" / "topology_diffusion" / "rf_shap_analysis"
    / LAYER / "topology_to_diffusion" / "threshold_0.3" / "consensus_matrix.csv", index_col=0)
consensus = consensus.drop(index=DROP, columns=DROP, errors="ignore")

d = np.clip(1 - consensus.values, 0, None); np.fill_diagonal(d, 0)
Z = linkage(squareform(d, checks=False), "average")
group = pd.Series(fcluster(Z, 1 - CORR_THRESHOLD, "distance"), index=consensus.index)

floor = summary.loc[SENTINEL, "mean"]
f = summary.copy()
f["group"]  = group.reindex(f.index)
f["freq"]   = f["n_seeds"] / N_ITER
f["xfloor"] = f["mean"] / floor
# label each group by its most FREQUENTLY selected member (the stable representative)
f["label"]  = f["group"].map(
    f.dropna(subset=["group"]).sort_values("freq", ascending=False)
     .groupby("group").apply(lambda g: g.index[0]))

# influence = best member's SHAP (never sum). union_freq = "any member selected" ≈ min(1, Σ freq)
G = f.dropna(subset=["group"]).groupby("label").agg(
        influence  = ("mean", "max"),
        union_freq = ("freq", lambda s: min(1.0, s.sum())),
        rep_freq   = ("freq", "max"),
        members    = ("mean", "size"))
G["xfloor"] = G["influence"] / floor
G["driver"] = (G["xfloor"] >= XFLOOR) & (G["union_freq"] >= FREQ_BAR) & (G.index != SENTINEL)

G["kind"]   = np.where(G["rep_freq"] >= FREQ_BAR, "stable_rep", "split")
G = G.sort_values("influence", ascending=False)

print(f"noise floor = {floor:.5f}\n")
print(G.round(4).to_string())

print("\n=== DRIVERS ===")
for lbl, r in G[G.driver].iterrows():
    if r["kind"] == "stable_rep":
        print(f"{lbl:<50} {r['xfloor']:>6.0f}× floor  (named feature)")
    else:
        reps = f[(f["label"] == lbl) & (f["freq"] > 0.1)].index.tolist()
        print(f"{lbl:<50} {r['xfloor']:>6.0f}× floor  (theme, rotates: {reps})")

f["gfreq"] = f["label"].map(G["union_freq"])
f["gdrv"]  = f["label"].map(G["driver"])
plt.figure(figsize=(9, 6))
for drv, g in f.dropna(subset=["group"]).groupby("gdrv"):
    plt.scatter(g["mean"], g["gfreq"], s=45, c="#16a34a" if drv else "#9ca3af",
                label="driver" if drv else "non-driver", edgecolors="#333", lw=0.3)
plt.scatter(floor, 1.0, s=90, c="#dc2626", marker="X", label="random")
plt.axvline(floor * XFLOOR, ls=":", c="#f59e0b", label=f"{XFLOOR}× floor")
plt.axhline(FREQ_BAR, ls=":", c="#2563eb", label="presence bar")
plt.xscale("log"); plt.xlabel("influence (max member mean|SHAP|)")
plt.ylabel("group presence (union freq)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()

out = G.copy()
out.index.name = "topology_feature"          

out["rotating_members"] = ""
for lbl in out.index:
    if out.loc[lbl, "kind"] == "split":
        reps = f[(f["label"] == lbl) & (f["freq"] > 0.1)].index.tolist()
        out.loc[lbl, "rotating_members"] = ";".join(reps)

out = out[["influence", "xfloor", "union_freq", "rep_freq",
           "members", "driver", "kind", "rotating_members"]]
out = out.round(6).sort_values("xfloor", ascending=False)

out_path = (dl.ROOT / "analysis" / "topology_diffusion" / "rf_shap_analysis"
            / LAYER / "topology_to_diffusion" / "threshold_0.3" / "topology_drivers.csv")
out.to_csv(out_path)
print(f"\nwrote {out_path}")
print(out.to_string())