# Evaluation script for measuring model performance on next token prediction
# Calculates token-level accuracy and KL divergence between predicted and true distributions
import argparse
import itertools
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import IPython

if ip := IPython.get_ipython():
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from sparsify.data import chunk_and_tokenize
from tqdm.auto import tqdm

from attribute.caching import TranscodedModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy and KL divergence on next token prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roneneldan/TinyStories-33M",
        help="Model name or path",
    )
    parser.add_argument(
        "--transcoder_path",
        type=str,
        default="/share/ilya.lasy/transcoders/clt-128",
        help="Path to transcoder",
    )
    parser.add_argument(
        "--mlp_trim", type=int, default=0, help="Fix MLP L0 to this value"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/accuracy",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Maximum sequence length for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model with specified transcoder for feature attribution
    model = TranscodedModel(
        args.model,
        transcoder_path=args.transcoder_path if args.mlp_trim == 0 else None,
    )

    dataset_name = "roneneldan/TinyStories"
    # Load and preprocess dataset into fixed-length chunks for efficient batching
    dataset = load_dataset(dataset_name, split="train").select(
        range(10000)
    )  # Load only 10k samples
    dataset = chunk_and_tokenize(
        dataset,
        model.tokenizer,
        max_seq_len=args.max_seq_len,
        num_proc=cpu_count() // 2,
        text_key="text",
    )

    # Setup data loading with multiprocessing
    logger.remove()  # Remove default logger to avoid clutter during evaluation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=cpu_count() // 2
    )

    # Track metrics across batches
    accuracies = []  # Token-level accuracy
    kl_divs = []  # KL divergence between predicted and true distributions

    def mlp_trimmer(module, input, output):
        values, indices = torch.topk(output, args.mlp_trim, dim=-1)
        output = torch.zeros_like(output)
        output.scatter_(dim=-1, index=indices, src=values)
        return output

    # Evaluate on full dataset
    try:
        if args.max_steps is not None:
            dataloader = itertools.islice(dataloader, args.max_steps)
        bar = tqdm(dataloader, desc="Evaluating", total=args.max_steps)
        for batch in bar:
            input_ids = batch["input_ids"].to(model.device)
            with torch.no_grad():
                out_base = model.model(input_ids)
                base_logits = out_base.logits
                if args.mlp_trim > 0:
                    for layer_idx in range(model.num_layers):
                        mlp = model.model.get_submodule(
                            f"{model.layer_prefix}.{layer_idx}.mlp"
                        )
                        in_proj = getattr(mlp, model.mlp_in_proj_name)
                        in_proj.register_forward_hook(mlp_trimmer)
                    out = model.model(input_ids)
                    for m in model.model.modules():
                        m._forward_hooks = OrderedDict()
                else:
                    out = model(input_ids, no_error=True)
                logits = out.logits

                # Calculate token-level accuracy
                pred_tokens = logits.argmax(dim=-1)
                base_tokens = base_logits.argmax(dim=-1)
                accuracy = (pred_tokens == base_tokens).float().mean().item()
                accuracies.append(accuracy)

                # Calculate KL divergence between predicted and true distributions
                log_probs = F.log_softmax(
                    logits, dim=-1
                )  # [batch, seq_len, vocab_size]
                base_log_probs = F.log_softmax(
                    base_logits, dim=-1
                )  # [batch, seq_len, vocab_size]

                # Calculate KL divergence for this batch
                kl_div = F.kl_div(
                    log_probs.view(-1, logits.size(-1)),
                    base_log_probs.view(-1, logits.size(-1)),
                    reduction="batchmean",
                    log_target=True,
                ).item()
                kl_divs.append(kl_div)

                # Update progress bar with current and running average metrics
                bar.set_postfix(
                    {
                        "Avg Acc": f"{np.mean(accuracies):.4f}",
                        "Avg KL": f"{np.mean(kl_divs):.4f}",
                    }
                )
    except KeyboardInterrupt:
        pass

    # Calculate final statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)

    # Prepare comprehensive results dictionary
    results = {
        "model": args.model,
        **(
            {"transcoder_path": args.transcoder_path}
            if args.mlp_trim == 0
            else {"mlp_trim": args.mlp_trim}
        ),
        "timestamp": datetime.now().isoformat(),
        "accuracy": {
            "mean": mean_accuracy,
            "std": std_accuracy,
        },
        "kl_divergence": {
            "mean": mean_kl,
            "std": std_kl,
        },
        "metadata": {
            "dataset": dataset_name,
            "batch_size": args.batch_size,
            "num_batches": len(accuracies),
            "total_tokens": len(accuracies)
            * args.batch_size
            * (args.max_seq_len - 1),  # -1 for next token prediction
        },
    }

    # Save results with timestamp for tracking experiments
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    escaped_model_name = args.model.replace("/", "_").replace(":", "_")
    escaped_transcoder_name = args.transcoder_path.replace("/", "_").replace(":", "_")
    if args.mlp_trim > 0:
        escaped_transcoder_name = f"mlp_trim_{args.mlp_trim}"
    escaped_model_transcoder_name = f"{escaped_model_name}-{escaped_transcoder_name}"
    output_file = output_dir / f"eval_results_{escaped_model_transcoder_name}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Token Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"KL Divergence: {mean_kl:.4f} ± {std_kl:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
