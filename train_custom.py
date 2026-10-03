import torch
import time
import math
import os
import functools
import warnings
from architecture import MyCustomLLM, CustomBlock
from data_loader import CustomDataLoader

# Harmless deprecation noise from the still-working FSDP1 state_dict API
warnings.filterwarnings('ignore', message='.*FSDP.state_dict_type.*')

# --- Configuration for Kaggle T4 x2 (16GB VRAM each) ---
batch_size = 16
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size
block_size = 1024
bpe_vocab_size = 32000

# --- Distributed Setup (auto-detects single vs multi-GPU) ---
# Launched with: torchrun --nproc_per_node=2 train_custom.py
# Falls back to single GPU if launched with: python train_custom.py
distributed = int(os.environ.get('RANK', -1)) != -1
if distributed:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    is_main = (local_rank == 0)
    if is_main:
        print(f"FSDP initialized: {dist.get_world_size()} GPUs")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_rank = 0
    is_main = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# --- Data ---
train_dataset = CustomDataLoader(block_size, split='train', vocab_size=bpe_vocab_size)
val_dataset = CustomDataLoader(block_size, split='val', vocab_size=bpe_vocab_size)
vocab_size = train_dataset.tokenizer.vocab_size

# --- Model (1B Parameters) ---
model_config = dict(
    vocab_size=vocab_size,
    dim=1536,
    n_layers=12,
    n_heads=24,
    n_kv_heads=8,
    max_seq_len=block_size,
    num_experts=4,
    top_k=2,
    dropout=0.05,
    window_size=512,
    n_predict=2,
)
model = MyCustomLLM(**model_config)
model.to(device)
# Full param count BEFORE FSDP shards (after wrapping, .parameters() only sees the local shard)
full_param_count = sum(p.numel() for p in model.parameters())

# --- Wrap with FSDP (shards optimizer states across GPUs) ---
if distributed:
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CustomBlock},
    )
    mp_policy = MixedPrecision(
        param_dtype=ptdtype,
        reduce_dtype=ptdtype,
        buffer_dtype=ptdtype,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(device),
        limit_all_gathers=True,
    )
    if is_main:
        print("Model wrapped with FSDP (FULL_SHARD)")
elif torch.cuda.device_count() > 1:
    if is_main:
        print(f"Detected {torch.cuda.device_count()} GPUs but not using FSDP.")
        print("For multi-GPU, launch with: torchrun --nproc_per_node=2 train_custom.py")

# Note: torch.compile can conflict with FSDP. Skipping for stability.
# To try: model = torch.compile(model)

if is_main:
    print(f"MyCustomLLM: {full_param_count/1e6:.2f}M params (total)")
    print(f"Architecture: {model_config['n_layers']}L / {model_config['n_heads']}H / "
          f"{model_config['num_experts']}E+shared-top{model_config['top_k']} / "
          f"predict-{model_config['n_predict']} / window-{model_config['window_size']}")

# --- Training Config ---
max_iters = 3000
eval_interval = 200
eval_iters = 10
grad_clip = 1.0
time_limit_hours = 10

max_lr = 1e-3
min_lr = 1e-4
warmup_iters = 200

def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split_name, ds in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(micro_batch_size, device)
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                _, loss = model(X, Y)
                if loss.dim() > 0:
                    loss = loss.mean()
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out

# --- Checkpoint helpers (FSDP-aware) ---
def _strip_checkpoint_wrappers(state_dict):
    """Remove checkpoint_wrapper .module. prefixes from state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('.module.', '.')
        new_state_dict[new_key] = v
    return new_state_dict

def save_checkpoint(model, path, config, iter_num, val_loss):
    if distributed:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        state_dict = _strip_checkpoint_wrappers(state_dict)
    else:
        state_dict = model.state_dict()
    if is_main:
        torch.save({
            'model_state_dict': state_dict,
            'config': config,
            'iter': iter_num,
            'val_loss': val_loss,
        }, path)

def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if distributed:
        # Full state dict must be loaded inside the same FSDP context used for saving
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('iter', -1) + 1, checkpoint.get('val_loss', float('inf'))

# --- Auto-Resume Logic ---
start_iter = 0
best_val_loss = float('inf')

if os.path.exists('best_model.pth'):
    if is_main: print("Found best_model.pth! Resuming training...")
    start_iter, best_val_loss = load_checkpoint(model, 'best_model.pth', device)
    if is_main: print(f"Resumed from Step {start_iter-1} with Val Loss: {best_val_loss:.4f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# --- Training Loop ---
t0 = time.time()

for iter in range(start_iter, max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        X, Y = train_dataset.get_batch(micro_batch_size, device)
        with torch.autocast(device_type=device_type, dtype=ptdtype):
            logits, loss = model(X, Y)
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / gradient_accumulation_steps

        # FSDP: skip gradient sync on non-final micro-steps
        if distributed and micro_step < gradient_accumulation_steps - 1:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    if distributed:
        model.clip_grad_norm_(grad_clip)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - t0
        train_loss = float(losses['train'])
        val_loss = float(losses['val'])
        # Rank 0 decides; the decision is broadcast so ALL ranks enter
        # save_checkpoint together. FSDP full-state-dict gather is a collective:
        # skipping it on one rank deadlocks NCCL (10-min timeout).
        should_save = 0
        if is_main:
            print(f"Step {iter:4d} | {elapsed:6.0f}s | LR {lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                should_save = 1
        if distributed:
            should_save_t = torch.tensor([should_save], device=device)
            dist.broadcast(should_save_t, src=0)
            should_save = int(should_save_t.item())
            best_t = torch.tensor([best_val_loss if is_main else 0.0], device=device)
            dist.broadcast(best_t, src=0)
            best_val_loss = float(best_t.item())
        if should_save:
            save_checkpoint(model, 'best_model.pth', model_config, iter, best_val_loss)
            if distributed:
                dist.barrier()

    # --- Time Cap Check (rank 0 decides, all ranks save together, then all exit) ---
    elapsed_hours = (time.time() - t0) / 3600.0
    time_up = 1 if elapsed_hours >= time_limit_hours else 0
    if distributed:
        time_up_t = torch.tensor([time_up if is_main else 0], device=device)
        dist.broadcast(time_up_t, src=0)
        time_up = int(time_up_t.item())
    if time_up:
        if is_main:
            print(f"\n[TIME CAP REACHED] {elapsed_hours:.2f} hours elapsed. Saving and exiting safely.")
        save_checkpoint(model, 'latest_backup_model.pth', model_config, iter, best_val_loss)
        if distributed:
            dist.barrier()
        break

t1 = time.time()
if is_main:
    print(f"\nTraining completed in {t1 - t0:.2f}s. Best val loss: {best_val_loss:.4f}")
    print("Best model saved to best_model.pth")

if distributed:
    dist.destroy_process_group()
