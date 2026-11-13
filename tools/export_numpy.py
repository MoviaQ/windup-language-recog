"""Export a PyTorch checkpoint to a torch-free ``.npz`` release file.

The release file holds the weights of every network (an ensemble keeps all
members), the language list, feature configuration, ensemble weights and the
acceptance policy (``calibration``). language_detector.LanguageDetector loads
it with numpy only.

``--storage int8_rowwise`` stores each embedding table as symmetric per-row
int8 with one float32 scale per row (same arithmetic as
tools/compress_checkpoint.py); ``fp16`` stores every array as float16. Members
that are already compressed keep their storage. The other top-level checkpoint
entries (training history, adaptation notes) are copied to
``source_metadata`` when they are plain JSON, and the export is refused
otherwise.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import numpy_runtime  # noqa: E402

STORAGES = ("keep", "fp16", "int8_rowwise")
_MODEL_KEYS = {"state_dict", "languages", "config", "calibration", "ensemble"}


def convert_member(config: dict, arrays: dict[str, np.ndarray], storage: str) -> tuple[dict, dict]:
    current = config.get("weight_storage", "fp32")
    if storage == "keep" or current == storage:
        return config, arrays
    if current != "fp32":
        raise ValueError(f"Member is stored as {current}; cannot convert it to {storage}")
    config = dict(config, weight_storage=storage)
    if storage == "fp16":
        return config, {name: value.astype(np.float16) for name, value in arrays.items()}
    converted = dict(arrays)
    weight = converted.pop("embedding.weight")
    converted["embedding.weight"], converted["embedding.weight" + numpy_runtime.ROW_SCALE_SUFFIX] = (
        numpy_runtime.quantize_rowwise_int8(weight))
    return config, converted


def export(checkpoint: dict, storage: str, source: dict) -> tuple[dict, list[dict]]:
    if storage not in STORAGES:
        raise ValueError(f"Unknown storage: {storage!r}")
    meta, member_arrays = numpy_runtime.split_checkpoint(checkpoint)
    for index, member in enumerate(meta["members"]):
        member["config"], member_arrays[index] = convert_member(member["config"], member_arrays[index], storage)
    extras = {key: value for key, value in checkpoint.items() if key not in _MODEL_KEYS}
    try:
        if json.loads(json.dumps(extras, allow_nan=False)) != extras:
            raise ValueError("changed by a JSON round trip")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint metadata is not plain JSON ({error}); refusing to drop it silently") from error
    meta["source_metadata"] = extras
    meta["export"] = dict(source, storage=storage)
    return meta, member_arrays


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Release file (.npz)")
    parser.add_argument("--storage", choices=STORAGES, default="keep")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error(f"{args.output} exists; refusing to overwrite")
    import torch

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    try:
        meta, member_arrays = export(checkpoint, args.storage,
                                     {"source_file": args.input.name, "source_sha256": file_sha256(args.input)})
        numpy_runtime.save_npz(args.output, meta, member_arrays)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}: {args.input.stat().st_size} -> {args.output.stat().st_size} bytes ({args.storage})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
