import os
import shutil
from pathlib import Path

from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(("/", ".", "~")) or "\\" in value or ":" in value


def _validate_paths(model_id: str, save_dir: str, local_files_only: bool) -> None:
    source_path = Path(model_id).expanduser()
    destination_path = Path(save_dir).expanduser()

    if local_files_only or _looks_like_local_path(model_id):
        if not source_path.exists():
            raise FileNotFoundError(
                "MODEL_ID must point to an existing path inside the container. "
                f"Got {model_id!r}. Mount the host model directory under /models "
                "and use a container path such as /models/Qwen3.5-2B."
            )

        if source_path.resolve() == destination_path.resolve():
            raise ValueError(
                "QUANTIZED_MODEL_DIR must be different from MODEL_ID so the "
                "full-precision source checkpoint is not overwritten."
            )

    destination_path.mkdir(parents=True, exist_ok=True)


def _model_class_for_config(config):
    if getattr(config, "model_type", None) == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration

    return AutoModelForCausalLM


def _copy_vl_assets(model_id: str, save_dir: str) -> None:
    source_path = Path(model_id).expanduser()
    destination_path = Path(save_dir).expanduser()
    if not source_path.exists() or not source_path.is_dir():
        return

    asset_names = [
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ]
    for asset_name in asset_names:
        source = source_path / asset_name
        destination = destination_path / asset_name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
            print(f"[quantize] Copied VL asset: {asset_name}")


def _save_mtp_tensors_if_available(model_id: str, save_dir: str) -> None:
    try:
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
    except Exception as exc:
        print(f"[quantize] MTP tensor copy skipped: {exc}")
        return

    try:
        save_mtp_tensors_to_checkpoint(source_model=model_id, dest_dir=save_dir)
        print("[quantize] Copied MTP tensors from source checkpoint")
    except Exception as exc:
        print(f"[quantize] MTP tensor copy skipped: {exc}")


def quantize_qwen3_5() -> None:
    """Perform GPTQ W4A16 quantization on a local Qwen 3.5 checkpoint."""
    model_id = os.environ.get("MODEL_ID", "/models/Qwen3.5-2B")
    save_dir = os.environ.get(
        "QUANTIZED_MODEL_DIR",
        "/models/Qwen3.5-2B-GPTQ-W4A16-G128",
    )
    num_cal_samples = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "512"))
    max_seq_len = int(os.environ.get("CALIBRATION_MAX_SEQUENCE_LENGTH", "2048"))
    local_files_only = _env_bool("LOCAL_FILES_ONLY", True)
    trust_remote_code = _env_bool("TRUST_REMOTE_CODE", True)
    calibration_dataset = os.environ.get(
        "CALIBRATION_DATASET",
        "HuggingFaceH4/ultrachat_200k",
    )
    calibration_split = os.environ.get(
        "CALIBRATION_DATASET_SPLIT",
    ) or f"train_sft[:{num_cal_samples}]"
    seed = int(os.environ.get("CALIBRATION_SEED", "42"))
    ignore = [
        target.strip()
        for target in os.environ.get(
            "GPTQ_IGNORE",
            "lm_head,re:.*visual.*,re:.*linear_attn.*",
        ).split(",")
        if target.strip()
    ]

    _validate_paths(model_id, save_dir, local_files_only)

    print(f"[quantize] Transformers version: {transformers.__version__}")
    print(f"[quantize] Loading base model: {model_id}")
    print(f"[quantize] Saving output to: {save_dir}")
    print(f"[quantize] LOCAL_FILES_ONLY={local_files_only}")
    config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = _model_class_for_config(config)
    print(f"[quantize] Model class: {model_cls.__name__}")
    model = model_cls.from_pretrained(
        model_id,
        config=config,
        dtype="auto",
        device_map="auto",
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    print(
        "[quantize] Preparing calibration data: "
        f"{calibration_dataset}:{calibration_split}, "
        f"{num_cal_samples} samples, max_seq_len={max_seq_len}"
    )
    ds = load_dataset(calibration_dataset, split=calibration_split)
    ds = ds.shuffle(seed=seed)

    def preprocess(example):
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        elif "text" in example:
            text = example["text"]
        else:
            raise KeyError(
                "Calibration samples must contain either a 'messages' field "
                "or a 'text' field."
            )
        return {"text": text}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore,
    )
    print(f"[quantize] GPTQ ignore targets: {ignore}")

    print("[quantize] Starting quantization...")
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_cal_samples,
    )

    print(f"[quantize] Saving quantized checkpoint to {save_dir}")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    _copy_vl_assets(model_id, save_dir)
    _save_mtp_tensors_if_available(model_id, save_dir)
    print("[quantize] Finished quantization")


if __name__ == "__main__":
    quantize_qwen3_5()
