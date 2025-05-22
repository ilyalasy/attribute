import sys
from pathlib import Path

import fire
from loguru import logger

from .caching import TranscodedModel
from .mlp_attribution import AttributionConfig, AttributionGraph


async def main(
    prompt="When John and Mary went to the store, John gave a bag to",
    model_name="HuggingFaceTB/SmolLM2-135M",
    save_dir = Path("attribution-graphs-frontend"),
    transcoder_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/sparsify/checkpoints/single_128x",
    cache_path = "/mnt/ssd-1/gpaulo/smollm-decomposition/attribution_graph/results/transcoder_128x/latents",
    name = "test-1-ts",
    scan = "default",
    remove_prefix = 0,
):
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = AttributionConfig(
        name=name,
        scan=scan,
    )
    model = TranscodedModel(
        model_name=model_name,
        transcoder_path=transcoder_path,
        device="cuda",
    )
    transcoded_outputs = model(prompt)

    if remove_prefix > 0:
        transcoded_outputs.input_ids = transcoded_outputs.input_ids[:, remove_prefix:]
        # only ever accessed w/ [-1], removing BOS doesn't matter
        # transcoded_outputs.last_layer_activations = transcoded_outputs.last_layer_activations[:, 1:]
        transcoded_outputs.logits = transcoded_outputs.logits[:, remove_prefix:]
        for k, mlp_output in transcoded_outputs.mlp_outputs.items():
            mlp_output.ln_factor = mlp_output.ln_factor[:, remove_prefix:]
            mlp_output.activation = mlp_output.activation[:, remove_prefix:]
            mlp_output.location = mlp_output.location[:, remove_prefix:]
            mlp_output.error = mlp_output.error[:, remove_prefix:]
            # we don't remove BOS from source nodes because we take gradients to them

    attribution_graph = AttributionGraph(model, transcoded_outputs, config)
    attribution_graph.get_dense_features(cache_path)
    attribution_graph.flow()
    attribution_graph.save_graph(save_dir)
    await attribution_graph.cache_features(cache_path, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
