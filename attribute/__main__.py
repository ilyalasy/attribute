import sys
from pathlib import Path

import fire
from loguru import logger

from attribute.caching import TranscodedModel
from attribute.mlp_attribution import AttributionConfig, AttributionGraph


async def main(
    prompt="Once in a land of ice and snow, there lived a boy named",
    model_name="SimpleStories/SimpleStories-35M",
    save_dir=Path("attribution-graphs-frontend"),
    transcoder_path="/share/ilya.lasy/transcoders/clt-16k-jumprelu",
    cache_path="/share/ilya.lasy/attribution_graph/",
    name="test-1-ts",
    scan="default",
    remove_prefix=0,
    pre_ln_hook=False,
    **kwargs,
):
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = AttributionConfig(
        name=name,
        scan=scan,
        **kwargs
    )
    model = TranscodedModel(
        model_name=model_name,
        transcoder_path=transcoder_path,
        device="cuda",
        pre_ln_hook=pre_ln_hook,
    )
    transcoded_outputs = model([prompt] * config.batch_size)
    transcoded_outputs.remove_prefix(remove_prefix)

    attribution_graph = AttributionGraph(model, transcoded_outputs, config)
    attribution_graph.get_dense_features(cache_path)
    attribution_graph.flow()
    attribution_graph.save_graph(save_dir)
    attribution_graph.cache_features(cache_path, save_dir)
    attribution_graph.cache_self_explanations(cache_path, save_dir)
    await attribution_graph.cache_contexts(cache_path, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
