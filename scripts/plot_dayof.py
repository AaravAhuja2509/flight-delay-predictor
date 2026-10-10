"""Chart for the day-of experiment: reports/dayof.png (reads reports/dayof_experiment*.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "reports"
main = json.loads((R / "dayof_experiment.json").read_text())["results"]
lead3 = json.loads((R / "dayof_experiment_lead180.json").read_text())["results"][0]
names = ["Current app\n(schedule+weather)", "+ aircraft\nrotation", "+ inbound plane\nstatus", "+ airport\nstatus", "All day-of\nfeatures"]
auc = [r["roc_auc"] for r in main]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
cols = ["#868e96", "#748ffc", "#2f9e44", "#f08c00", "#1971c2"]
b = ax[0].bar(names, auc, color=cols)
for bar, v in zip(b, auc):
    ax[0].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set(ylim=(0.6, 0.86), ylabel="ROC-AUC, all Oct-Dec 2017 flights", title="Predicting 1 hour before departure")
ax[0].tick_params(axis="x", labelsize=8)
x = ["Days ahead\n(schedule only)", "3 hours\nbefore", "1 hour\nbefore"]
y = [main[1]["roc_auc"], lead3["roc_auc"], main[4]["roc_auc"]]
ax[1].plot(x, y, "o-", color="#1971c2", lw=2)
for xi, yi in zip(x, y):
    ax[1].annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax[1].set(ylim=(0.6, 0.86), title="Accuracy rises as departure gets closer")
ax[1].tick_params(axis="x", labelsize=8)
for a in ax:
    a.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(R / "dayof.png", dpi=130)
print("saved", R / "dayof.png")
