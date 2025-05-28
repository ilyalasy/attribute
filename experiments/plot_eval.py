#%%
import json
from pathlib import Path
from matplotlib import pyplot as plt
from transformers import AutoConfig
import seaborn as sns

acc_path = Path("results/accuracy")
expansion_factors = []
l0s = []
accuracies = []
kl_divs = []
names = []
for acc_file in acc_path.glob("*.json"):
    with open(acc_file, "r") as f:
        acc_data = json.load(f)
    model_name = acc_data["model"]
    model_config = AutoConfig.from_pretrained(model_name)
    if "mlp_trim" in acc_data:
        l0 = acc_data["mlp_trim"]
        model_cfg = vars(model_config)
        expansion_factor = model_cfg.get("intermediate_size", model_cfg.get("n_inner")) // model_config.hidden_size
        names.append(f"MLP (top {l0} weights)")
    else:
        transcoder_path = Path(acc_data["transcoder_path"])
        print(acc_file, transcoder_path)
        with open(transcoder_path / "config.json", "r") as f:
            transcoder_config = json.load(f)
        expansion_factor = transcoder_config["sae"]["expansion_factor"]
        l0 = transcoder_config["sae"]["k"]
        is_clt = transcoder_config["cross_layer"] > 0
        is_skip = transcoder_config["sae"]["skip_connection"]
        short_name = f"{'ST SST CLT CLST'.split()[is_clt * 2 + is_skip]} (x{expansion_factor}, k={l0})"
        names.append(f"{short_name} (x{expansion_factor}, k={l0})")
    expansion_factors.append(expansion_factor)
    l0s.append(l0)
    accuracies.append(acc_data["accuracy"]["mean"])
    kl_divs.append(acc_data["kl_divergence"]["mean"])
sns.set_theme()
# x_axis = "l0"
x_axis = "expansion_factor"
y_axis = "accuracy"
# y_axis = "kl_div"
argsort_by_name = sorted(range(len(names)), key=lambda i: names[i])
for i in argsort_by_name:
    l0 = l0s[i]
    expansion_factor = expansion_factors[i]
    accuracy = accuracies[i]
    kl_div = kl_divs[i]
    plt.scatter(
        dict(l0=l0, expansion_factor=expansion_factor)[x_axis],
        dict(kl_div=kl_div, accuracy=accuracy)[y_axis],
        label=names[i]
    )
plt.xscale("log")
if y_axis == "kl_div":
    plt.yscale("log")
plt.xlabel(x_axis.capitalize())
plt.ylabel(y_axis.capitalize())
plt.legend()
plt.savefig(f"results/accuracy_{x_axis}_{y_axis}.png", dpi=300)
plt.show()
# %%
