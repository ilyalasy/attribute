#%%
from IPython import get_ipython
import json
import re
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from attribute.caching import TranscodedModel


if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

model = TranscodedModel(
    "meta-llama/Llama-3.2-1B-Instruct",
    transcoder_path="EleutherAI/skip-transcoder-Llama-3.2-1B-131k",
)
# %%

prompt = "Michael Jordan plays the sport of"
og_activations = model(prompt)

supernodes_str = """
supernodes [["okpkopkop","intermediate_5_11_55357-0","intermediate_5_11_19006-0"],["dewfewokpkopk","intermediate_5_12_57243-0","intermediate_5_12_86448-0"]]
"""
superedges_str = """
superedges [{"source":"okpkopkop","target":"dewfewokpkopk","weight":-0.010766710163786134}]
"""

supernodes = json.loads(supernodes_str.strip().partition("supernodes ")[2])
superedges = json.loads(superedges_str.strip().partition("superedges ")[2])

supernodes = {
    a: b for a, *b in supernodes
}
# %%

def get_supernode_activations(activations):
    super_activations = {}
    for supernode, node_list in supernodes.items():
        total_activation = 0
        for node in node_list:
            seq_idx, layer_idx, node_idx = re.match(r"intermediate_(\d+)_(\d+)_(\d+)-0", node).groups()
            seq_idx = int(seq_idx)
            layer_idx = int(layer_idx)
            node_idx = int(node_idx)
            mlp_outputs = activations.mlp_outputs[layer_idx]
            locations, acts = (
                mlp_outputs.location[0, seq_idx].tolist(),
                mlp_outputs.activation[0, seq_idx].tolist()
            )
            total_activation += acts[locations.index(node_idx)]
        super_activations[supernode] = total_activation
    return super_activations
supernode_activations = get_supernode_activations(og_activations)
supernode_activations
# %%

steered_matrix = np.zeros((len(supernodes), len(supernodes)))
for supernode, node_list in supernodes.items():
    steered = defaultdict(list)
    for node in node_list:
        seq_idx, layer_idx, node_idx = re.match(r"intermediate_(\d+)_(\d+)_(\d+)-0", node).groups()
        seq_idx = int(seq_idx)
        layer_idx = int(layer_idx)
        node_idx = int(node_idx)
        mlp_outputs = og_activations.mlp_outputs[layer_idx]
        locations, acts = (
            mlp_outputs.location[0, seq_idx].tolist(),
            mlp_outputs.activation[0, seq_idx].tolist()
        )
        act_strength = acts[locations.index(node_idx)]
        steered[layer_idx].append((seq_idx, node_idx, -2 * act_strength))
    steered_activations = model(
        prompt,
        steer_features=steered,
        errors_from=og_activations,
        latents_from_errors=True
    )
    steered_supernode_activations = get_supernode_activations(steered_activations)
    i = list(supernodes.keys()).index(supernode)
    for supernode_target in steered_supernode_activations:
        j = list(supernodes.keys()).index(supernode_target)
        steered_matrix[i, j] = steered_supernode_activations[supernode_target]
steered_matrix /= (steered_matrix * np.eye(len(supernodes)) + 1e-6).sum(axis=1, keepdims=True)

x_ticks = list(supernodes.keys())
plt.imshow(steered_matrix)
plt.colorbar()

# Add orange outlines for edges
for edge in superedges:
    source_idx = list(supernodes.keys()).index(edge['source'])
    target_idx = list(supernodes.keys()).index(edge['target'])
    # Draw rectangle around the cell
    base_color = "FFA500" if edge['weight'] > 0 else "0000FF"
    opacity = abs(edge['weight']) ** 0.5
    op_hex = int(opacity * 255)
    op_hex = str(hex(op_hex))[2:].zfill(2)
    rect = plt.Rectangle((target_idx-0.5+0.04, source_idx-0.5+0.04),
                         0.93, 0.93,
                        fill=False,
                        edgecolor=f"#{base_color}{op_hex}",
                        linewidth=10)
    plt.gca().add_patch(rect)

plt.xticks(range(len(x_ticks)), x_ticks, rotation=90)
plt.yticks(range(len(x_ticks)), x_ticks)
plt.show()
# %%
