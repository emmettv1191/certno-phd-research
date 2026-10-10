#!/usr/bin/env python
"""Figures generated from REAL experiment JSON (results/data/*.json)."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
D = json.load(open("results/data/paper_tables.json"))
SEL = json.load(open("results/data/selective.json"))
FNO = json.load(open("results/data/fno_certified.json"))

c = np.array([r["contrast"] for r in D["contrast"]])
eg = np.array([r["eff_global"]["median"] for r in D["contrast"]])
ec = np.array([r["eff_cg"]["median"] for r in D["contrast"]])

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.3))
ax[0].semilogy(c, eg, "o-", color="#c0392b", lw=1.4, ms=3.5,
               label=r"global: $\|r\|_2/\alpha$")
ax[0].set_xlabel(r"contrast $a_{\max}/a_{\min}$")
ax[0].set_ylabel(r"median effectivity $B/\|e\|_2$")
ax[0].set_title("(a) global bound", fontsize=8)
ax[0].grid(alpha=.3, ls="--"); ax[0].legend(frameon=False)

ax[1].semilogy(c, np.maximum(ec, 1e-12), "s-", color="#1e8449", lw=1.4, ms=3.5,
               label=r"CG-corrected $B_{\rm CG}$")
ax[1].axhline(1.0, color="k", ls=":", lw=.9)
ax[1].set_ylim(0.5, 3)
ax[1].set_xlabel(r"contrast $a_{\max}/a_{\min}$")
ax[1].set_ylabel(r"median effectivity $B/\|e\|_2$")
ax[1].set_title("(b) CG-corrected", fontsize=8)
ax[1].grid(alpha=.3, ls="--"); ax[1].legend(frameon=False)
plt.tight_layout()
for e in ("pdf", "png"):
    plt.savefig(f"figures/fig1_effectivity.{e}", bbox_inches="tight")
plt.close()

# Fig 2: effectivity distribution (global vs CG) pooled, log scale
fig, ax = plt.subplots(figsize=(3.3, 2.3))
allg = np.concatenate([[r["eff_global"]["q25"], r["eff_global"]["median"],
                        r["eff_global"]["q75"]] for r in D["contrast"]])
allc = np.concatenate([[r["eff_cg"]["q25"], r["eff_cg"]["median"],
                        r["eff_cg"]["q75"]] for r in D["contrast"]])
bp = ax.boxplot([allg, allc], tick_labels=["global", "CG-corr."], patch_artist=True,
                widths=.55, showfliers=False)
for patch, col in zip(bp["boxes"], ["#c0392b", "#1e8449"]):
    patch.set_facecolor(col); patch.set_alpha(.55)
ax.set_yscale("log"); ax.set_ylabel(r"effectivity $B/\|e\|_2$")
ax.grid(alpha=.3, ls="--", axis="y")
plt.tight_layout()
for e in ("pdf", "png"):
    plt.savefig(f"figures/fig2_effectivity_box.{e}", bbox_inches="tight")
plt.close()

# Fig 3: selective verification acceptance
fig, ax = plt.subplots(figsize=(3.3, 2.3))
tau = np.array([r["tau"] for r in SEL["rows"]])
ag = np.array([r["accept_rate_global"] for r in SEL["rows"]])
ac = np.array([r["accept_rate_cg"] for r in SEL["rows"]])
ax.semilogx(tau, ag, "o-", color="#c0392b", lw=1.4, ms=4, label="global")
ax.semilogx(tau, ac, "s-", color="#1e8449", lw=1.4, ms=4, label="CG-corrected")
ax.set_xlabel(r"tolerance $\tau$"); ax.set_ylabel("acceptance rate")
ax.set_ylim(-.05, 1.08); ax.grid(alpha=.3, ls="--"); ax.legend(frameon=False)
plt.tight_layout()
for e in ("pdf", "png"):
    plt.savefig(f"figures/fig3_selective.{e}", bbox_inches="tight")
plt.close()

# Fig 4: FNO ID/OOD real predictions
fig, ax = plt.subplots(figsize=(3.3, 2.3))
names = list(FNO.keys())
xg = [FNO[k]["analytic"]["median_effectivity"] for k in names]
xc = [FNO[k]["cg_corrected"]["median_effectivity"] for k in names]
ix = np.arange(len(names))
ax.bar(ix - .19, xg, .36, color="#c0392b", alpha=.85, label="global")
ax.bar(ix + .19, xc, .36, color="#1e8449", alpha=.85, label="CG-corrected")
ax.set_yscale("log"); ax.set_xticks(ix); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel(r"median effectivity"); ax.grid(alpha=.3, ls="--", axis="y")
ax.legend(frameon=False, fontsize=6.5)
plt.tight_layout()
for e in ("pdf", "png"):
    plt.savefig(f"figures/fig4_fno.{e}", bbox_inches="tight")
plt.close()

# Fig 5: cost scaling
fig, ax = plt.subplots(figsize=(3.3, 2.2))
nn = np.array([r["n"] for r in D["size"]])
tt = np.array([r["median_time_ms"] for r in D["size"]])
ax.plot(nn, tt, "o-", color="#2471a3", lw=1.4, ms=4)
ax.set_xlabel(r"interior nodes $n$"); ax.set_ylabel("median cert. time [ms]")
ax.grid(alpha=.3, ls="--")
plt.tight_layout()
for e in ("pdf", "png"):
    plt.savefig(f"figures/fig5_cost.{e}", bbox_inches="tight")
plt.close()
print("figures ok")
