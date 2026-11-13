"""Store a single-model checkpoint with compressed weights (no retraining).

fp16 keeps every tensor as float16; int8_rowwise keeps the embedding table as
symmetric per-row int8 with a float32 scale per row and the small layers as
float32. language_detector.model_from_checkpoint restores float32 weights at
load, so inference code is unchanged. The storage is recorded in
config["weight_storage"].

Ensemble checkpoints (such as the bundled model.pt) and checkpoints that are
already compressed are refused: this tool only handles one member.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STORAGES = ("fp16", "int8_rowwise")


def compress_checkpoint(checkpoint: dict, storage: str) -> dict:
    from language_detector import compress_state_dict, model_from_checkpoint

    if storage not in STORAGES:
        raise ValueError(f"Unknown storage: {storage!r}")
    if checkpoint.get("ensemble"):
        raise ValueError("Ensemble checkpoints are not supported; compress a single-model checkpoint")
    config = dict(checkpoint.get("config", {}))
    if config.get("weight_storage", "fp32") != "fp32":
        raise ValueError(f"Checkpoint is already stored as {config['weight_storage']}")
    model_from_checkpoint(checkpoint)  # refuse checkpoints that do not load
    config["weight_storage"] = storage
    compressed = dict(checkpoint)
    compressed["config"] = config
    compressed["state_dict"] = compress_state_dict(checkpoint["state_dict"], storage)
    compressed["compression"] = {"storage": storage, "source_config_storage": "fp32"}
    return compressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage", choices=STORAGES, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    if args.output.exists():
        parser.error(f"{args.output} exists; refusing to overwrite")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    from language_detector import model_from_checkpoint

    try:
        compressed = compress_checkpoint(checkpoint, args.storage)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compressed, args.output)
    try:
        model_from_checkpoint(torch.load(args.output, map_location="cpu", weights_only=True))
    except (ValueError, RuntimeError) as error:
        args.output.unlink()
        print(f"Error: written checkpoint does not load ({error}); removed {args.output}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}: {args.input.stat().st_size} -> {args.output.stat().st_size} bytes ({args.storage})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
