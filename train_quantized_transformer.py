#!/usr/bin/env python3
"""Train and export a quantized tiny Transformer language model."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("quantized_transformer")
DEFAULT_BOOK_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"


@dataclass
class Config:
    # Data
    book_path: str = "book.txt"
    book_url: str = DEFAULT_BOOK_URL
    train_split: float = 0.9
    max_chars: int = 0

    # Model
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    use_fake_quant: bool = True
    fake_quant_bits: int = 8

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 5
    grad_clip: float = 1.0
    eval_max_batches: int = 50
    seed: int = 42

    # Generation
    prompt: str = "The "
    max_new_tokens: int = 180
    temperature: float = 0.9
    top_k: int = 40

    # Runtime
    output_dir: str = "artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate tiny transformer and export dynamic INT8 model.")
    parser.add_argument("--book-path", default="book.txt", help="Path to corpus text file.")
    parser.add_argument("--book-url", default=DEFAULT_BOOK_URL, help="Corpus download URL if file is missing.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--block-size", type=int, default=128, help="Context window size.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer embedding width.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--layers", type=int, default=4, help="Transformer blocks.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--max-chars", type=int, default=0, help="Use only first N characters (0 = full corpus).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Override device (cpu or cuda).")
    parser.add_argument("--disable-fake-quant", action="store_true", help="Disable fake quant during training.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        book_path=args.book_path,
        book_url=args.book_url,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        lr=args.lr,
        max_chars=args.max_chars,
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        use_fake_quant=not args.disable_fake_quant,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_book_file(path: Path, url: str) -> None:
    if path.exists():
        return
    LOGGER.info("book file not found at %s; downloading from %s", path, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return total_bytes / (1024**2)


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class BookDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def fake_quantize_tensor(x: torch.Tensor, num_bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    if not x.is_floating_point():
        return x
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -2 ** (num_bits - 1)
    max_abs = x.detach().abs().max()
    scale = torch.clamp(max_abs / max(qmax, 1), min=eps)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


class FakeQuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_bits: int = 8, enable_fake_quant: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.num_bits = num_bits
        self.enable_fake_quant = enable_fake_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_fake_quant:
            return self.linear(x)
        x_q = fake_quantize_tensor(x, self.num_bits)
        w_q = fake_quantize_tensor(self.linear.weight, self.num_bits)
        return F.linear(x_q, w_q, self.linear.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim**-0.5

        linear = self._linear_factory(cfg)
        self.q_proj = linear(cfg.d_model, cfg.d_model)
        self.k_proj = linear(cfg.d_model, cfg.d_model)
        self.v_proj = linear(cfg.d_model, cfg.d_model)
        self.out_proj = linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _linear_factory(cfg: Config):
        if cfg.use_fake_quant:
            return lambda in_f, out_f, bias=True: FakeQuantLinear(
                in_f,
                out_f,
                bias=bias,
                num_bits=cfg.fake_quant_bits,
                enable_fake_quant=True,
            )
        return lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tsz, channels = x.shape
        q = self.q_proj(x).view(bsz, tsz, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tsz, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tsz, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(tsz, tsz, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.attn_dropout(F.softmax(attn, dim=-1))

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bsz, tsz, channels)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.use_fake_quant:
            linear = lambda in_f, out_f, bias=True: FakeQuantLinear(
                in_f, out_f, bias=bias, num_bits=cfg.fake_quant_bits, enable_fake_quant=True
            )
        else:
            linear = lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias)

        self.net = nn.Sequential(
            linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, FakeQuantLinear):
            nn.init.normal_(module.linear.weight, mean=0.0, std=0.02)
            if module.linear.bias is not None:
                nn.init.zeros_(module.linear.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, tsz = idx.shape
        if tsz > self.cfg.block_size:
            raise ValueError(f"Sequence length {tsz} exceeds block size {self.cfg.block_size}")

        pos = torch.arange(0, tsz, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


@torch.no_grad()
def evaluate(model: TinyTransformerLM, loader: DataLoader, device: str, max_batches: int) -> float:
    model.eval()
    losses: List[float] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def train_one_epoch(model: TinyTransformerLM, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, grad_clip: float, log_every: int = 100) -> None:
    model.train()
    running = 0.0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item()
        if step % log_every == 0:
            LOGGER.info("step=%d train_loss=%.4f", step, running / log_every)
            running = 0.0


def export_dynamic_quantized_model(fp32_model: TinyTransformerLM) -> nn.Module:
    fp32_model = fp32_model.cpu().eval()
    return torch.quantization.quantize_dynamic(fp32_model, {nn.Linear}, dtype=torch.qint8)


def save_checkpoint(path: Path, model: nn.Module, cfg: Config, tokenizer: CharTokenizer, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "metrics": metrics,
        },
        path,
    )


def write_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cfg: Config) -> Dict[str, object]:
    set_seed(cfg.seed)

    book_path = Path(cfg.book_path)
    ensure_book_file(book_path, cfg.book_url)
    text = book_path.read_text(encoding="utf-8")
    if cfg.max_chars > 0:
        text = text[: cfg.max_chars]

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    split_idx = int(len(token_ids) * cfg.train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = BookDataset(train_ids, cfg.block_size)
    val_ds = BookDataset(val_ids, cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    model = TinyTransformerLM(cfg, vocab_size=tokenizer.vocab_size).to(cfg.device)

    LOGGER.info("device=%s chars=%d vocab=%d", cfg.device, len(text), tokenizer.vocab_size)
    LOGGER.info("trainable_params=%d fp32_size_mb=%.2f", trainable_params(model), estimate_model_size_mb(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_train = float("inf")
    t0 = time.time()

    out_dir = Path(cfg.output_dir)
    fp32_ckpt = out_dir / "best_fp32_model.pt"
    int8_ckpt = out_dir / "best_int8_dynamic_model.pt"
    metrics_json = out_dir / "metrics.json"

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, cfg.device, cfg.grad_clip)

        train_loss = evaluate(model, train_loader, cfg.device, cfg.eval_max_batches)
        val_loss = evaluate(model, val_loader, cfg.device, cfg.eval_max_batches)

        best_train = min(best_train, train_loss)
        elapsed = time.time() - epoch_t0
        LOGGER.info("epoch=%d/%d train_loss=%.4f val_loss=%.4f epoch_time_s=%.1f", epoch, cfg.max_epochs, train_loss, val_loss, elapsed)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                fp32_ckpt,
                model,
                cfg,
                tokenizer,
                {"train_loss": train_loss, "val_loss": val_loss, "best_val": best_val},
            )
            LOGGER.info("saved fp32 checkpoint: %s", fp32_ckpt)

    prompt_ids = tokenizer.encode(cfg.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    generated_fp32 = tokenizer.decode(
        model.generate(x, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)[0].tolist()
    )

    quantized_model = export_dynamic_quantized_model(model)
    save_checkpoint(
        int8_ckpt,
        quantized_model,
        cfg,
        tokenizer,
        {"best_val": best_val},
    )

    x_cpu = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    generated_int8 = tokenizer.decode(
        quantized_model.generate(x_cpu, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)[0].tolist()
    )

    result = {
        "device": cfg.device,
        "chars": len(text),
        "vocab_size": tokenizer.vocab_size,
        "trainable_params": trainable_params(model),
        "fp32_size_mb": estimate_model_size_mb(model),
        "int8_size_mb": estimate_model_size_mb(quantized_model),
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "runtime_sec": round(time.time() - t0, 2),
        "fp32_checkpoint": str(fp32_ckpt),
        "int8_checkpoint": str(int8_ckpt),
        "generated_fp32": generated_fp32,
        "generated_int8": generated_int8,
    }
    write_metrics(metrics_json, result)
    LOGGER.info("saved metrics: %s", metrics_json)
    return result


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = build_config(args)
    result = run(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
