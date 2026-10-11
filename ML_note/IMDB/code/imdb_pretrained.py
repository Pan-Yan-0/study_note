"""One-model-at-a-time IMDB fine-tuning, separate from the historical GloVe notebook.

Required packages: torch, transformers, scikit-learn; peft for LoRA;
sentencepiece for DeBERTa-v2.
Version pins belong in the tested environment file, not in this reusable module.

Example::
    cfg = TrainingConfig(output_dir="result/bert_lora", num_epochs=3)
    run = train_model("bert_lora", train_frame, validation_frame, config=cfg)
    predict_submit(run, test_frame, "result/bert_lora/submission.csv")

The registry uses the actual 1.5B DeBERTa-v2-xxlarge, not a smaller replacement.
Its default is LoRA plus pooler/classifier training. Each T4 is an independent
16 GiB device: cuda:0 and cuda:1 are selectable, never implicitly combined.
Full XXL tuning requires explicit opt-in and a conservative free-memory check.
Registry/source overrides and injected tiny models support offline smoke tests.
Injected adapter/head-only models need a matching local model_name_or_path for
automatic reload, or an explicitly supplied base_model in load_run().
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelSpec:
    model_name_or_path: str
    tokenizer_name_or_path: str
    default_batch_size: int
    default_eval_batch_size: int
    default_max_length: int
    default_accumulation_steps: int
    lora_targets: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ("classifier",)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "bert_full": ModelSpec("bert-base-uncased", "bert-base-uncased", 8, 16, 256, 2),
    "bert_lora": ModelSpec(
        "bert-base-uncased", "bert-base-uncased", 8, 16, 256, 2, ("query", "value")
    ),
    "deberta_v2_xxlarge": ModelSpec(
        "microsoft/deberta-v2-xxlarge", "microsoft/deberta-v2-xxlarge",
        1, 1, 128, 16, ("query_proj", "value_proj"), ("classifier", "pooler"),
    ),
}


@dataclass
class TrainingConfig:
    model_key: Optional[str] = None
    output_dir: str = "result/pretrained_imdb"
    num_epochs: int = 3
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    max_length: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = 1.0
    amp: bool = True
    seed: int = 42
    deterministic: bool = True
    device: Optional[str] = None
    num_workers: int = 0
    review_column: str = "review"
    label_column: str = "sentiment"
    xxl_mode: str = "lora"  # lora, head_only, or full
    allow_xxl_full: bool = False
    xxl_min_free_gib: float = 40.0
    gradient_checkpointing: Optional[bool] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    local_files_only: bool = False
    monitor: str = "accuracy"  # accuracy, auc, or loss; AUC matches the Kaggle metric


@dataclass
class TrainingRun:
    model: nn.Module
    tokenizer: Any
    config: TrainingConfig
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation: dict[str, Any]
    output_dir: Path
    metadata: dict[str, Any]


@dataclass
class PredictionResult:
    submission_path: Path
    probability_path: Path
    n_examples: int
    ids: list[str]
    labels: list[int]
    positive_probabilities: list[float]


def _resolve_config(model_key: str, config: Optional[TrainingConfig]) -> TrainingConfig:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key {model_key!r}; choose {tuple(MODEL_REGISTRY)}")
    cfg = replace(config or TrainingConfig())
    if cfg.model_key is not None and cfg.model_key != model_key:
        raise ValueError("config.model_key and train_model model_key disagree")
    spec = MODEL_REGISTRY[model_key]
    cfg.model_key = model_key
    cfg.batch_size = cfg.batch_size if cfg.batch_size is not None else spec.default_batch_size
    cfg.eval_batch_size = (
        cfg.eval_batch_size if cfg.eval_batch_size is not None else spec.default_eval_batch_size
    )
    cfg.max_length = cfg.max_length if cfg.max_length is not None else spec.default_max_length
    cfg.gradient_accumulation_steps = (
        cfg.gradient_accumulation_steps
        if cfg.gradient_accumulation_steps is not None else spec.default_accumulation_steps
    )
    if cfg.gradient_checkpointing is None:
        cfg.gradient_checkpointing = model_key == "deberta_v2_xxlarge" and cfg.xxl_mode != "head_only"
    for field in ("num_epochs", "batch_size", "eval_batch_size", "max_length", "gradient_accumulation_steps", "lora_r"):
        value = getattr(cfg, field)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if cfg.num_workers < 0 or cfg.learning_rate <= 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid workers, learning rate, or weight decay")
    if cfg.xxl_mode not in {"lora", "head_only", "full"}:
        raise ValueError("xxl_mode must be lora, head_only, or full")
    if not 0 <= cfg.lora_dropout < 1 or cfg.lora_alpha <= 0:
        raise ValueError("Invalid LoRA alpha/dropout")
    if cfg.monitor not in {"accuracy", "auc", "loss"}:
        raise ValueError("monitor must be accuracy, auc, or loss")
    if cfg.max_grad_norm is not None and cfg.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    if cfg.xxl_min_free_gib < 30:
        raise ValueError("XXL full memory guard must allow at least 30 GiB before activations")
    cfg.output_dir = str(Path(cfg.output_dir).resolve())
    return cfg


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Seed training and loader shuffle; deterministic CUDA behavior is best-effort."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _device(device: Optional[str]) -> torch.device:
    result = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("This training interface supports cpu or a single cuda device")
    return result


def _guard_xxl_full(cfg: TrainingConfig, device: torch.device, injected: bool,
                    parameter_count: Optional[int] = None) -> None:
    if cfg.model_key != "deberta_v2_xxlarge" or cfg.xxl_mode != "full":
        return
    if not cfg.allow_xxl_full:
        raise ValueError("XXL full tuning needs allow_xxl_full=True; default XXL mode is lora")
    source = cfg.model_name_or_path or MODEL_REGISTRY[cfg.model_key].model_name_or_path
    # Explicit tiny local overrides/injections are for tests, not an alternate XXL model.
    if parameter_count is not None and parameter_count < 1_000_000_000:
        return
    if parameter_count is None and (injected or source != MODEL_REGISTRY[cfg.model_key].model_name_or_path):
        return
    if device.type != "cuda":
        raise RuntimeError("Real 1.5B XXL full tuning requires a suitable CUDA device")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 2**30
    if free_gib < cfg.xxl_min_free_gib:
        raise RuntimeError(
            f"XXL full Adam needs about 27 GB before activations. {device} has "
            f"{free_gib:.1f} GiB free; this run requires {cfg.xxl_min_free_gib:.1f}. "
            "Two 16 GiB T4s do not form one 32 GiB device. Choose lora/head_only."
        )


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(" ")


def strip_html(review: str) -> str:
    """Preserve punctuation/case for the selected model's own tokenizer."""
    if not isinstance(review, str):
        raise ValueError("Each review must be a non-missing string")
    parser = _HTMLText()
    parser.feed(review)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def _records(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, (str, Path)):
        with Path(frame).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if hasattr(frame, "to_dict"):
        return frame.to_dict(orient="records")
    return [dict(row) for row in frame]


class ReviewDataset(Dataset):
    def __init__(self, frame: Any, review_column: str = "review", label_column: Optional[str] = "sentiment"):
        rows = _records(frame)
        if not rows:
            raise ValueError("Dataset is empty")
        self.texts: list[str] = []
        self.labels: Optional[list[int]] = [] if label_column is not None else None
        for row in rows:
            if review_column not in row:
                raise ValueError(f"Missing review column {review_column!r}")
            self.texts.append(strip_html(row[review_column]))
            if label_column is not None:
                value = row.get(label_column)
                try:
                    label = int(value)
                    valid = label in (0, 1) and float(value) == label
                except (ValueError, TypeError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError("Training labels must be 0 or 1")
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {"text": self.texts[index]}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item

    def fingerprint(self) -> str:
        payload = json.dumps([self.texts, self.labels], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ReviewCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in examples], truncation=True,
            padding=True, max_length=self.max_length, return_tensors="pt",
        )
        batch = {key: value for key, value in encoded.items()
                 if key in {"input_ids", "attention_mask", "token_type_ids"}}
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("Tokenizer must return input_ids and attention_mask")
        if "labels" in examples[0]:
            batch["labels"] = torch.tensor([item["labels"] for item in examples], dtype=torch.long)
        return batch


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def make_data_loader(dataset: ReviewDataset, tokenizer: Any, cfg: TrainingConfig, *, training: bool) -> DataLoader:
    cfg = _resolve_config(cfg.model_key or "bert_full", cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        dataset, batch_size=cfg.batch_size if training else cfg.eval_batch_size,
        shuffle=training, drop_last=False, num_workers=cfg.num_workers,
        collate_fn=ReviewCollator(tokenizer, cfg.max_length),
        generator=generator, worker_init_fn=_seed_worker,
    )


def _guard_bin_checkpoint_loader(model_key: str, source: str) -> None:
    # The official XXL repository supplies a PyTorch .bin checkpoint. Current
    # Transformers requires Torch >=2.6 for its safe weights-only loading path.
    if model_key != "deberta_v2_xxlarge" or source != MODEL_REGISTRY[model_key].model_name_or_path:
        return
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError("Loading the official DeBERTa-v2 XXL .bin checkpoint requires torch>=2.6")


def build_model_and_tokenizer(
    model_key: str, config: Optional[TrainingConfig] = None, *,
    model: Optional[nn.Module] = None, tokenizer: Any = None,
) -> tuple[nn.Module, Any]:
    """Build one registry choice. Source overrides must match the injected model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    cfg = _resolve_config(model_key, config)
    set_reproducibility(cfg.seed, cfg.deterministic)
    spec = MODEL_REGISTRY[model_key]
    device = _device(cfg.device)
    injected = model is not None
    _guard_xxl_full(cfg, device, injected)
    source = cfg.model_name_or_path or spec.model_name_or_path
    token_source = cfg.tokenizer_name_or_path or cfg.model_name_or_path or spec.tokenizer_name_or_path
    if tokenizer is None:
        token_revision = cfg.tokenizer_revision or (cfg.revision if token_source == source else None)
        tokenizer = AutoTokenizer.from_pretrained(
            token_source, revision=token_revision, local_files_only=cfg.local_files_only,
        )
    if model is None:
        _guard_bin_checkpoint_loader(model_key, source)
        dtype = torch.float16 if (
            model_key == "deberta_v2_xxlarge" and cfg.xxl_mode != "full"
            and device.type == "cuda" and cfg.amp
        ) else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=2, torch_dtype=dtype, revision=cfg.revision,
            local_files_only=cfg.local_files_only,
        )
    if getattr(model.config, "num_labels", 2) != 2:
        raise ValueError("The model must have exactly two classification outputs")
    _guard_xxl_full(cfg, device, injected,
                    parameter_count=sum(parameter.numel() for parameter in model.parameters()))
    use_lora = model_key == "bert_lora" or (model_key == "deberta_v2_xxlarge" and cfg.xxl_mode == "lora")
    if use_lora:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        if not isinstance(model, PeftModel):
            module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
            missing = set(spec.lora_targets) - module_names
            if missing:
                raise ValueError(f"Model does not have requested LoRA targets: {sorted(missing)}")
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, target_modules=list(spec.lora_targets),
                modules_to_save=list(spec.modules_to_save), bias="none",
            ))
        scope = "lora_adapters_classifier_pooler" if model_key == "deberta_v2_xxlarge" else "lora_adapters_classifier"
    elif model_key == "deberta_v2_xxlarge" and cfg.xxl_mode == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.split(".", 1)[0] in {"classifier", "pooler", "score", "pre_classifier"}
        scope = "head_only_classifier_pooler"
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        scope = "full_fine_tuning"
    # GradScaler cannot unscale FP16 parameter gradients. Keep the small trainable
    # adapters/task heads in FP32 while a frozen XXL CUDA backbone can stay FP16.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype == torch.float16:
            parameter.data = parameter.data.float()
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("No trainable parameters; check task-head or adapter configuration")
    if cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if use_lora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    actual_source = cfg.model_name_or_path or (
        getattr(model.config, "_name_or_path", None) if injected else source
    )
    model._imdb_runtime = {
        "training_scope": scope, "checkpoint_kind": "adapter" if use_lora else ("head_only" if scope.startswith("head_only") else "full"),
        "base_model_source": actual_source or None, "injected_model": injected,
        "source_override": source != spec.model_name_or_path,
        "tokenizer_source": token_source, "lora_targets": list(spec.lora_targets) if use_lora else [],
        "modules_to_save": list(spec.modules_to_save) if use_lora else [],
    }
    return model, tokenizer


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                          enabled=bool(amp and device.type == "cuda"))


def _logits(model: nn.Module, batch: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    output = model(**inputs)
    logits = output.logits
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [batch, 2] logits")
    return logits


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: Optional[str] = None, *, amp: bool = False) -> dict[str, Any]:
    target = _device(device or str(next(model.parameters()).device))
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    all_labels, positive = [], []
    for batch in data_loader:
        labels = batch["labels"].to(target)
        with _autocast(target, amp):
            logits = _logits(model, batch, target)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        loss_sum += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        count += labels.numel()
        all_labels.extend(labels.cpu().tolist())
        positive.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    if count == 0:
        raise ValueError("Validation loader is empty")
    if len(set(all_labels)) == 2:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(all_labels, positive))
    else:
        auc = None
    return {"loss": loss_sum / count, "accuracy": correct / count, "auc": auc, "n_examples": count}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _versions() -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {}
    for package in ("torch", "transformers", "peft", "accelerate", "sentencepiece", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _save_checkpoint(model: nn.Module, tokenizer: Any, path: Path, kind: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if kind == "head_only":
        model.config.save_pretrained(path)
        state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}
        torch.save(state, path / "task_head.pt")
    else:
        # PEFT saves adapters AND modules_to_save (including the random task head).
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_model(
    model_key: str, train_frame: Any, validation_frame: Any, *,
    config: Optional[TrainingConfig] = None, model: Optional[nn.Module] = None,
    tokenizer: Any = None,
) -> TrainingRun:
    """Train ONE model, export history/config, and return the restored best model.

    Accumulation groups divide SUM losses by the group's actual example count,
    including short final batches and the incomplete final accumulation group.
    """
    cfg = _resolve_config(model_key, config)
    train_data = ReviewDataset(train_frame, cfg.review_column, cfg.label_column)
    validation_data = ReviewDataset(validation_frame, cfg.review_column, cfg.label_column)
    output_dir = Path(cfg.output_dir)
    if (output_dir / "training_metadata.json").exists():
        raise FileExistsError("This output_dir already has a run; choose a new directory")
    set_reproducibility(cfg.seed, cfg.deterministic)
    target = _device(cfg.device)
    cfg.device = str(target)
    model, tokenizer = build_model_and_tokenizer(model_key, cfg, model=model, tokenizer=tokenizer)
    model.to(target)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    amp_enabled = bool(cfg.amp and target.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = make_data_loader(train_data, tokenizer, cfg, training=True)
    validation_loader = make_data_loader(validation_data, tokenizer, cfg, training=False)
    runtime = dict(model._imdb_runtime)
    metadata = {
        "schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg), "registry_model_source": MODEL_REGISTRY[model_key].model_name_or_path,
        **runtime, "base_revision": cfg.revision,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "train_examples": len(train_data), "validation_examples": len(validation_data),
        "train_fingerprint": train_data.fingerprint(), "validation_fingerprint": validation_data.fingerprint(),
        "metric_aggregation": "sample_weighted", "amp_enabled": amp_enabled,
        "gpu_memory_policy": "one selected CUDA device, never pooled across GPUs",
        "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
        "versions": _versions(), "checkpoint_directory": "best_checkpoint",
    }
    _write_json(output_dir / "training_metadata.json", metadata)
    history: list[dict[str, Any]] = []
    best_epoch, best_validation, best_state = 0, None, None
    total_steps, skipped_steps = 0, 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        loss_sum, correct, count, epoch_steps = 0.0, 0, 0, 0
        batch_iterator = iter(train_loader)
        while True:
            group = list(islice(batch_iterator, cfg.gradient_accumulation_steps))
            if not group:
                break
            group_count = sum(batch["labels"].numel() for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                labels = batch["labels"].to(target)
                with _autocast(target, amp_enabled):
                    logits = _logits(model, batch, target)
                    summed_loss = F.cross_entropy(logits.float(), labels, reduction="sum")
                    normalized_loss = summed_loss / group_count
                if not torch.isfinite(summed_loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(normalized_loss).backward()
                loss_sum += summed_loss.detach().item()
                correct += (logits.detach().argmax(dim=-1) == labels).sum().item()
                count += labels.numel()
            scaler.unscale_(optimizer)
            if cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                epoch_steps += 1
                total_steps += 1
            else:
                skipped_steps += 1
        validation = evaluate(model, validation_loader, str(target), amp=amp_enabled)
        if validation.get(cfg.monitor) is None:
            raise ValueError("AUC selection requires both classes in validation; use monitor='accuracy' or 'loss'")
        row = {
            "epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "validation_auc": validation.get("auc"),
            "train_examples": count, "validation_examples": validation["n_examples"],
            "optimizer_steps": epoch_steps,
        }
        history.append(row)
        better = best_validation is None or (
            validation[cfg.monitor] < best_validation[cfg.monitor] if cfg.monitor == "loss"
            else validation[cfg.monitor] > best_validation[cfg.monitor]
        )
        tied_lower_loss = best_validation is not None and (
            validation[cfg.monitor] == best_validation[cfg.monitor]
            and validation["loss"] < best_validation["loss"]
        )
        if better or tied_lower_loss:
            best_epoch, best_validation = epoch, dict(validation)
            best_state = {name: parameter.detach().cpu().clone()
                          for name, parameter in model.named_parameters() if parameter.requires_grad}
            _save_checkpoint(model, tokenizer, output_dir / "best_checkpoint", runtime["checkpoint_kind"])
        _write_csv(output_dir / "training_history.csv", history, list(row))
        print(f"Epoch {epoch}/{cfg.num_epochs}: Train Loss {row['train_loss']:.4f} | "
              f"Train Acc {row['train_accuracy']:.4f} | Val Acc {row['validation_accuracy']:.4f}")
    # Only trainable parameters changed; restore them in-place, without loading a
    # second 1.5B backbone into GPU memory. No running-stat buffers in these models.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name].to(parameter.device))
    model.eval()
    metadata.update({
        "status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch, "best_validation": best_validation,
        "optimizer_steps": total_steps, "skipped_amp_steps": skipped_steps,
        "selection_rule": ("minimum" if cfg.monitor == "loss" else "maximum") +
                          f" validation {cfg.monitor}; ties use lower validation loss",
        "returned_model": "restored best validation checkpoint",
    })
    _write_json(output_dir / "training_metadata.json", metadata)
    _write_json(output_dir / "results.json", {
        "model_key": model_key, "training_scope": runtime["training_scope"],
        "best_epoch": best_epoch, "best_validation": best_validation,
        "final_epoch_validation": {"loss": history[-1]["validation_loss"], "accuracy": history[-1]["validation_accuracy"],
                                   "auc": history[-1]["validation_auc"]},
        "metric_aggregation": "sample_weighted", "kaggle_leaderboard_score": None,
    })
    return TrainingRun(model, tokenizer, cfg, history, best_epoch, best_validation, output_dir, metadata)


def load_run(output_dir: str | Path, *, device: Optional[str] = None,
             base_model: Optional[nn.Module] = None, tokenizer: Any = None) -> TrainingRun:
    """Reload the best checkpoint. Adapter/head-only runs share their original base."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root = Path(output_dir).resolve()
    metadata = json.loads((root / "training_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("Run did not complete; its checkpoint has no completed-run contract")
    cfg = TrainingConfig(**metadata["config"])
    target = _device(device or cfg.device)
    cfg.device = str(target)
    checkpoint = root / metadata["checkpoint_directory"]
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    kind = metadata["checkpoint_kind"]
    dtype = torch.float16 if (cfg.model_key == "deberta_v2_xxlarge" and kind != "full" and target.type == "cuda" and cfg.amp) else torch.float32
    if kind == "full":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=dtype, local_files_only=True)
    else:
        source = metadata.get("base_model_source")
        if base_model is None:
            if not source:
                raise ValueError("Injected run has no base source; supply its matching base_model")
            _guard_bin_checkpoint_loader(cfg.model_key, source)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                source, num_labels=2, torch_dtype=dtype,
                revision=metadata.get("base_revision"), local_files_only=cfg.local_files_only,
            )
        if kind == "adapter":
            from peft import PeftModel
            for name, parameter in base_model.named_parameters():
                if name.split(".", 1)[0] in {"classifier", "pooler", "score", "pre_classifier"}:
                    parameter.data = parameter.data.float()
            model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=False)
        else:
            model = base_model
            for name, parameter in model.named_parameters():
                is_head = name.split(".", 1)[0] in {"classifier", "pooler", "score", "pre_classifier"}
                parameter.requires_grad = is_head
                if is_head:
                    parameter.data = parameter.data.float()
            state = torch.load(checkpoint / "task_head.pt", map_location="cpu", weights_only=True)
            unexpected = model.load_state_dict(state, strict=False).unexpected_keys
            if unexpected:
                raise ValueError(f"Checkpoint task-head keys do not match: {unexpected}")
    model.to(target).eval()
    rows = _records(root / "training_history.csv")
    history = [{key: (None if value == "" else (int(value) if key in {"epoch", "train_examples", "validation_examples", "optimizer_steps"} else float(value)))
                for key, value in row.items()} for row in rows]
    return TrainingRun(model, tokenizer, cfg, history, metadata["best_epoch"], metadata["best_validation"], root, metadata)


def _normalize_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Prediction IDs must not be missing")
    result = str(value).strip().strip('"')
    if not result:
        raise ValueError("Prediction IDs must not be empty")
    return result


@torch.no_grad()
def predict_submit(
    run: TrainingRun, test_frame: Any, filename: str | Path, *,
    probability_filename: Optional[str | Path] = None, id_column: str = "id",
    review_column: Optional[str] = None, sample_submission: Any = None,
) -> PredictionResult:
    """Export hard labels by default, and positive-class probabilities separately.

    Without a template, preserve test-frame order. With a template, validate a
    one-to-one ID match and reorder both exports to the template's ID order.
    """
    rows = _records(test_frame)
    if any(id_column not in row for row in rows):
        raise ValueError(f"Missing prediction ID column {id_column!r}")
    ids = [_normalize_id(row[id_column]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction IDs must be unique")
    dataset = ReviewDataset(rows, review_column or run.config.review_column, label_column=None)
    loader = make_data_loader(dataset, run.tokenizer, run.config, training=False)
    target = next(run.model.parameters()).device
    run.model.eval()
    hard_labels, positive = [], []
    for batch in loader:
        with _autocast(target, run.config.amp):
            logits = _logits(run.model, batch, target)
        hard_labels.extend(logits.argmax(dim=-1).cpu().tolist())
        positive.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    if sample_submission is not None:
        template_rows = _records(sample_submission)
        if any(id_column not in row for row in template_rows):
            raise ValueError("Submission template has no ID column")
        template_ids = [_normalize_id(row[id_column]) for row in template_rows]
        if len(template_ids) != len(set(template_ids)) or set(template_ids) != set(ids):
            raise ValueError("Submission template must match prediction IDs one-to-one")
        predictions = dict(zip(ids, zip(hard_labels, positive)))
        hard_labels = [predictions[item][0] for item in template_ids]
        positive = [predictions[item][1] for item in template_ids]
        ids = template_ids
    if not (len(ids) == len(hard_labels) == len(positive)) or not all(math.isfinite(p) and 0 <= p <= 1 for p in positive):
        raise ValueError("Prediction length or probabilities are invalid")
    submission_path = Path(filename)
    probability_path = Path(probability_filename) if probability_filename else submission_path.with_name(submission_path.stem + "_probabilities.csv")
    if submission_path.resolve() == probability_path.resolve():
        raise ValueError("Hard labels and probabilities require separate output paths")
    _write_csv(submission_path, [{id_column: item, "sentiment": label} for item, label in zip(ids, hard_labels)], [id_column, "sentiment"])
    _write_csv(probability_path, [{id_column: item, "positive_probability": p} for item, p in zip(ids, positive)], [id_column, "positive_probability"])
    _write_json(submission_path.with_suffix(".metadata.json"), {
        "model_key": run.config.model_key, "training_scope": run.metadata["training_scope"],
        "prediction_checkpoint": "best_validation", "best_epoch": run.best_epoch,
        "n_examples": len(ids), "submission_values": "hard_labels_0_or_1",
        "probability_values": "softmax_probability_of_label_1", "template_order_used": sample_submission is not None,
        "submission_path": str(submission_path), "probability_path": str(probability_path),
        "kaggle_submitted": False, "kaggle_leaderboard_score": None,
    })
    return PredictionResult(submission_path, probability_path, len(ids), ids, hard_labels, positive)


__all__ = [
    "MODEL_REGISTRY", "ModelSpec", "TrainingConfig", "TrainingRun", "PredictionResult",
    "ReviewDataset", "ReviewCollator", "strip_html", "set_reproducibility",
    "make_data_loader", "build_model_and_tokenizer", "train_model", "evaluate", "load_run", "predict_submit",
]
