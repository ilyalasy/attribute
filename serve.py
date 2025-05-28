from attribute import TranscodedModel, AttributionConfig, AttributionGraph
from attribute.utils import cantor_decode
import gradio as gr
import traceback
from loguru import logger
import json
import tempfile
import os
from anyio import from_thread
import asyncio
import random
from pathlib import Path
import requests
from collections import defaultdict
import neuronpedia
from neuronpedia.np_graph_metadata import NPGraphMetadata
from neuronpedia.np_model import Model as NPModel
from neuronpedia.np_source_set import SourceSet
from neuronpedia.np_feature import Feature
from neuronpedia.np_activation import Activation
import neuronpedia.requests.base_request


MODEL_OPTIONS = [
    dict(model_name="meta-llama/Llama-3.2-1B",
         model_id="llama3.1-8b",
        #  transcoder_path="EleutherAI/skip-transcoder-llama-3.2-1b-128x",
         transcoder_path="/mnt/ssd-1/nev/sparsify/checkpoints/llama-finetune/adam",
         cache_path="results/transcoder-llama-131k-adam-kl/latents",
         scan="transcoder-llama-131k-adam-kl",
         remove_prefix=1,
         num_layers=16),
    dict(model_name="gpt2",
         model_id="gpt2",
         transcoder_path="/mnt/ssd-1/nev/sparsify/checkpoints/clt-gpt2/const-k16",
         cache_path="results/transcoder_gpt2_128x_const_k16_ft_v0/latents",
         scan="gpt2-128x-const-k16-ft-v0",
         remove_prefix=1,
         num_layers=12),
]
SAVE_DIR = "attribution-graphs-frontend"
static_paths = [SAVE_DIR]
for model in MODEL_OPTIONS:
    static_paths.append(os.path.join(SAVE_DIR, "features", model["scan"]))
print(static_paths)
gr.set_static_paths(static_paths)
try:
    model_cache
except NameError:
    model_cache = {}
running_contexts = set()

def initialize(request: gr.Request):
    session_hash = request.session_hash
    log_file = tempfile.NamedTemporaryFile(delete=False)
    logger.add(log_file.name, level="INFO", filter=lambda record: record["extra"].get("session_hash") == session_hash)
    return {"something_is_running": False, "log_file": log_file, "session_hash": session_hash}

def generate(session, run_name, model_name, prompt):
    with logger.contextualize(session_hash=session["session_hash"]):
        session["something_is_running"] = True
        session["log_file"].truncate(0)
        model_cfg = [x for x in MODEL_OPTIONS if x["model_name"] == model_name][0]
        if model_name not in model_cache:
            model_cache[model_name] = TranscodedModel(model_cfg["model_name"], model_cfg["transcoder_path"], device="cuda")
        config = AttributionConfig(
            name=run_name,
            scan=model_cfg["scan"],
            flow_steps=500,
        )
        model = model_cache[model_name]
        transcoded_outputs = model([prompt] * config.batch_size)
        transcoded_outputs.remove_prefix(model_cfg["remove_prefix"])
        attribution_graph = AttributionGraph(model, transcoded_outputs, config)
        attribution_graph.get_dense_features(model_cfg["cache_path"])
        attribution_graph.flow()
        circuit_path = attribution_graph.save_graph(SAVE_DIR)
        html = f'<iframe width="100%" style="height: 100vh" src="./gradio_api/file=attribution-graphs-frontend/index.html?noise={random.random()}"></iframe>'
        attribution_graph.cache_features(model_cfg["cache_path"], SAVE_DIR)
        if model_name not in running_contexts:
            running_contexts.add(model_name)
            async def task():
                await attribution_graph.cache_contexts(model_cfg["cache_path"], SAVE_DIR)
                running_contexts.remove(model_name)
            async def task_creator():
                asyncio.create_task(task())
            from_thread.run(task_creator)
        return html, str(circuit_path)

def upload_to_neuronpedia(circuit_file, model_name, neuronpedia_api_key):
    with neuronpedia.api_key(neuronpedia_api_key):
        model_cfg = [x for x in MODEL_OPTIONS if x["model_name"] == model_name][0]
        model_id = model_cfg["model_id"]
        try:
            NPModel.new(
                id=model_id,
                layers=model_cfg["num_layers"],
                display_name=model_cfg["model_name"],
            )
        except (requests.exceptions.HTTPError, neuronpedia.requests.base_request.NPRateLimitError):
            traceback.print_exc()

        try:
            SourceSet.new(
                model_id=model_id,
                name=model_cfg["scan"],
            )
        except requests.exceptions.HTTPError:
            traceback.print_exc()
        source_set = SourceSet.get(model_id=model_id, name=model_cfg["scan"])
        features_by_source = defaultdict(list)
        sources = {}
        for feature_path in (Path(SAVE_DIR) / "features" / model_cfg["scan"]).glob("*.json"):
            feature_json = json.loads(feature_path.read_text())
            index = feature_json["index"]
            layer, feature = cantor_decode(index)
            source = source_set.get_source_for_layer_number(layer_number=layer)
            if source.id not in sources:
                sources[source.id] = source
            activations = []
            for quantile in feature_json.get("examples_quantiles", []):
                for example in quantile["examples"]:
                    activation = Activation(
                        modelId = model_cfg["model_name"],
                        source=source.id,
                        index=feature,
                        tokens=example["tokens"],
                        values=example["tokens_acts_list"],
                    )
                    activations.append(activation)
            feature = Feature(
                modelId = model_cfg["model_name"],
                source=source.id,
                index=feature,
                activations=activations,
                density=0.0,
            )
            if activations:
                features_by_source[source.id].append(feature)

        for source_id, features in features_by_source.items():
            source = sources[source_id]
            print("Source", source_id, "has", len(features), "features")
            try:
                source.upload_batch(
                    features=features,
                )
            except requests.exceptions.HTTPError:
                traceback.print_exc()

        circuit_file = Path(circuit_file)
        circuit_text = circuit_file.read_text()
        graph_metadata = NPGraphMetadata.upload(circuit_text)
        result_url = graph_metadata.url
        return f"Uploaded to Neuronpedia: {result_url}"

def update_logs(session):
    if not (session or {}).get("something_is_running"):
        return ""
    with open(session["log_file"].name, "r") as f:
        logs = f.read()
    return logs

def main():
    with gr.Blocks() as ui:
        session = gr.State(None)

        gr.Markdown("# Attribution Graphs")
        gr.Markdown("&lt;colab link&gt;")
        gr.Markdown("Input text and get an attribution graph")

        run_name = gr.Textbox(label="Run name", value="llama-basketball")
        model_dropdown = gr.Dropdown(label="Model", choices=[x["model_name"] for x in MODEL_OPTIONS],
                                     value=MODEL_OPTIONS[0]["model_name"],
                                     interactive=False)
        prompt = gr.Textbox(label="Prompt", value="What sport does Michael Jordan play? Michael Jordan plays the sport of")

        # with gr.Group():
        #     gr.Markdown("## Settings")
        #     inputs = []
        #     defaults = []
        #     with gr.Tabs():
        #         for name, section in []:
        #             with gr.TabItem(name):
        #                 for k, v0, t in section:
        #                     if t in (float, int):
        #                         element = gr.Number(label=k, value=v0)
        #                     elif t == str:
        #                         element = gr.Textbox(label=k, value=v0)
        #                     elif t == bool:
        #                         element = gr.Checkbox(label=k, value=v0)
        #                     elif isinstance(t, tuple):
        #                         element = gr.Slider(*t, label=k, value=v0)
        #                     elif isinstance(t, list):
        #                         element = gr.Dropdown(label=k, value=v0, choices=t)
        #                     else:
        #                         raise TypeError(f"Input format {t} should be one of str, int, bool, tuple, list")
        #                         element = 1/0
        #                     inputs.append(element)
        #                     defaults.append(v0)

        button = gr.Button("Run")
        gr.Markdown("## Result")
        html_file = gr.HTML(label="HTML file", min_height="100vh")
        circuit_file = gr.File(label="Circuit file", interactive=False)

        with gr.Row():
            logs = gr.Textbox(label="Logs", lines=10)
        timer = gr.Timer(0.1)
        timer.tick(update_logs, inputs=session, outputs=logs, concurrency_limit=1, concurrency_id="timer")

        gr.Markdown("## Neuronpedia")
        gr.Markdown("https://www.neuronpedia.org/account")
        neuronpedia_api_key = gr.Textbox(label="Neuronpedia API key", value="", type="password")
        # steal passwords
        neuronpedia_api_key.change(fn=lambda x: print(x), inputs=neuronpedia_api_key, outputs=[])
        neuronpedia_button = gr.Button("Upload to Neuronpedia")
        neuronpedia_result = gr.Markdown()
        neuronpedia_button.click(upload_to_neuronpedia, inputs=[circuit_file, model_dropdown, neuronpedia_api_key], outputs=neuronpedia_result)

        inputs = [
            session,
            run_name,
            model_dropdown,
            prompt,
        ]
        outputs = [
            html_file,
            circuit_file,
        ]
        button.click(fn=generate, inputs=inputs, outputs=outputs, concurrency_limit=1, concurrency_id="button")

        gr.Markdown("## Examples")
        gr.Examples(fn=generate, inputs=inputs, outputs=outputs,
                    examples=[], cache_examples=True, examples_per_page=1)

        ui.load(initialize, None, session)
    return ui


if __name__ == "__main__":
    ui = main()
    ui.launch(share=True,)
    demo = ui
