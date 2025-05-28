#%%
%load_ext autoreload
%autoreload 2
from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="INFO")
#%%
from attribute.caching import TranscodedModel
from matplotlib import pyplot as plt
from attribute.mlp_attribution import AttributionGraph, AttributionConfig
import random
import torch
from collections import defaultdict
import numpy as np

# model_name = "roneneldan/TinyStories-33M"
# transcoder_path = "../e2e/checkpoints/clt-ts/test"
model_name = "nev/GELU_4L512W_C4_Code"
transcoder_path = "../e2e/checkpoints/gelu-4l-clt/k64"

model = TranscodedModel(
    model_name=model_name,
    transcoder_path=transcoder_path,
    device="cuda",
)
#%%
# prompt = "When John and Mary went to the store, John gave a bag to"
prompt = "<|BOS|>National Data Attribution Graph ("
og_activations = model(prompt)
#%%
config = AttributionConfig(
    name="test-1-ts",
    scan="default",
)
graph = AttributionGraph(model, og_activations, config)
graph.flow()
#%%
adj_matrix, dedup_node_names = graph.adjacency_matrix(normalize=False, absolute=False)
# adj_matrix, dedup_node_names = graph.adjacency_matrix(normalize=False, absolute=True)
dedup_node_indices = {name: i for i, name in enumerate(dedup_node_names)}
#%%
identity = np.eye(len(dedup_node_names))
inf_matrix = np.linalg.inv(identity - adj_matrix) - identity
#%%
output_mask = np.zeros((len(dedup_node_names),))
for i, name in enumerate(dedup_node_names):
    if not name.startswith("output_"):
        continue
    output_mask[i] = 1
output_mask /= output_mask.sum()
#%%
plt.xscale("log")
plt.yscale("log")
plt.hist(inf_matrix.flatten())
# %%

def collect_values(activations):
    values = {}
    for l, o in activations.mlp_outputs.items():
        act, loc = o.activation, o.location
        for seq in range(loc.shape[1]):
            for f, a in zip(loc[0, seq].tolist(), act[0, seq].tolist()):
                values[(seq, l, f)] = a
    return values

per_layer = defaultdict(set)
for l, o in og_activations.mlp_outputs.items():
    act, loc = o.activation, o.location
    per_layer[l].update(loc.flatten().unique().tolist())
per_layer = {k: sorted(v) for k, v in per_layer.items()}
og_values = collect_values(og_activations)
#%%
dist_random = []
est_random = []
dist_kl_small = []
dist_kl_large = []
dist_act_large = []
dist_act_small = []
N_EXPERIMENTS = 4
N_SAMPLES = 100
seed = 5
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
for sample_idx in range(N_SAMPLES + N_EXPERIMENTS):
    remove_n = 4
    if sample_idx >= N_EXPERIMENTS:
        sample_removed = {k: random.sample(v, remove_n) for k, v in per_layer.items()}
        if sample_idx == N_SAMPLES + N_EXPERIMENTS - 1 and False:
            to_leave = random.choice(list(sample_removed.keys()))
            sample_removed = {k: (v if k == to_leave else []) for k, v in sample_removed.items()}
    elif sample_idx in (0, 1):
        sample_removed = {k: [] for k in per_layer}
        for k in per_layer:
            nodes_and_influences = []
            for f in per_layer[k]:
                removed_mask = np.zeros((len(dedup_node_names),))
                for i, name in enumerate(dedup_node_names):
                    if not name.startswith("intermediate_"):
                        continue
                    seq_idx, layer_idx, feature_idx = map(int, name.split("_")[1:])
                    if f != feature_idx:
                        continue
                    k_idx = og_activations.mlp_outputs[layer_idx].location[0, seq_idx].tolist().index(feature_idx)
                    removed_mask[i] = og_activations.mlp_outputs[layer_idx].activation[0, seq_idx, k_idx]
                influence = (inf_matrix @ removed_mask) @ output_mask
                nodes_and_influences.append((influence, f))
            nodes_and_influences.sort(key=lambda x: abs(x[0]), reverse=sample_idx == N_SAMPLES + 1)
            nodes_and_influences = nodes_and_influences[:remove_n]
            for influence, f in nodes_and_influences:
                sample_removed[k].append(f)
    elif sample_idx in (2, 3):
        for k in per_layer:
            nodes_and_activations = {f: 0 for f in per_layer[k]}
            for f in per_layer[k]:
                for key, activation in og_values.items():
                    if key[2] == f and key[1] == k:
                        nodes_and_activations[f] += abs(activation)
            nodes_and_activations = sorted(nodes_and_activations.items(), key=lambda x: x[1], reverse=sample_idx == N_SAMPLES + 3)
            nodes_and_activations = nodes_and_activations[:remove_n]
            for f, _ in nodes_and_activations:
                sample_removed[k].append(f)

    removed_mask = np.zeros((len(dedup_node_names),))
    for i, name in enumerate(dedup_node_names):
        if not name.startswith("intermediate_"):
            continue
        seq_idx, layer_idx, feature_idx = map(int, name.split("_")[1:])
        if feature_idx in sample_removed[layer_idx]:
            k_idx = og_activations.mlp_outputs[layer_idx].location[0, seq_idx].tolist().index(feature_idx)
            removed_mask[i] = og_activations.mlp_outputs[layer_idx].activation[0, seq_idx, k_idx]

    masked_activations = model(prompt, mask_features=sample_removed, errors_from=og_activations)
    masked_logprobs = masked_activations.logits.log_softmax(-1)
    og_logprobs = og_activations.logits.log_softmax(-1)
    est_logprobs = og_logprobs.clone()
    estimated_influence = (inf_matrix @ (-removed_mask))
    # estimated_influence = (adj_matrix @ (-removed_mask))
    for i, name in enumerate(dedup_node_names):
        if not name.startswith("output_"):
            continue
        node = graph.nodes[name]
        est_logprobs[..., node.token_position, node.token_idx] += estimated_influence[i]
    kl_div = torch.nn.functional.kl_div(masked_logprobs, og_logprobs, reduction="batchmean", log_target=True).item()
    # kl_div_est = torch.nn.functional.kl_div(est_logprobs, og_logprobs, reduction="batchmean", log_target=True).item()
    kl_div_est = (est_logprobs - og_logprobs).norm().item()
    if sample_idx >= N_EXPERIMENTS:
        dist_random.append(kl_div)
        est_random.append(kl_div_est)
    elif sample_idx == 0:
        dist_kl_small.append(kl_div)
    elif sample_idx == 1:
        dist_kl_large.append(kl_div)
    elif sample_idx == 2:
        dist_act_small.append(kl_div)
    elif sample_idx == 3:
        dist_act_large.append(kl_div)
#%%
from scipy.stats import spearmanr
spearman_corr = spearmanr(est_random, dist_random).correlation
plt.scatter(est_random, dist_random)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Estimated KL divergence")
plt.ylabel("Actual KL divergence")
plt.title(f"Spearman correlation: {spearman_corr}")
plt.show()

#%%
plt.xscale("log")
plt.yscale("log")
bins = np.logspace(-1, 2)
plt.hist(dist_random, bins=bins, alpha=0.5, label="random")
plt.hist(dist_kl_small, bins=bins, alpha=0.5, label="kl small")
plt.hist(dist_kl_large, bins=bins, alpha=0.5, label="kl large")
plt.hist(dist_act_small, bins=bins, alpha=0.5, label="act small")
plt.hist(dist_act_large, bins=bins, alpha=0.5, label="act large")
plt.legend()
# %%
masked_values = collect_values(masked_activations)
xs = []
ys = []
for k, v in og_values.items():
    (seq_idx, layer_idx, feature_idx) = k
    node_index = f"intermediate_{seq_idx}_{layer_idx}_{feature_idx}"
    if node_index not in dedup_node_names:
        continue
    inf_estimate = inf_matrix[dedup_node_indices[node_index]] @ (-removed_mask)
    # inf_estimate = np.abs(inf_matrix[dedup_node_indices[node_index]]) @ removed_mask
    # inf_estimate = adj_matrix[dedup_node_indices[node_index]] @ (-removed_mask)
    rel_estimate = abs(inf_estimate) / v

    if k in masked_values:
        difference = masked_values[k] - v
        rel = abs(difference) / v
        xs.append(rel)
        ys.append(rel_estimate)
        if inf_estimate == 0:
            print(k)

xs = np.array(xs)
ys = np.array(ys)
plt.scatter(xs, ys, s=1)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.xlabel("Relative difference")
plt.ylabel("Relative difference estimate")
mask = ys != 0
xs = xs[mask]
ys = ys[mask]
spearman_corr = spearmanr(xs, ys, nan_policy="omit").correlation
plt.title(f"Spearman correlation (nonzero): {spearman_corr}")
plt.show()

# %%
