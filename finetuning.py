"""
Fine-tuning GPT-2 with TRUE Tensor Parallelism + Flash Communication
Dataset: wikitext-103-raw-v1
Fixes applied:
  1. optimizer.zero_grad() moved before forward pass
  2. Gradient clipping added
  3. Learning rate lowered to 3e-5
  4. Padding tokens masked in labels
  5. Dataset switched to wikitext-103-raw-v1
  6. Compression ratio fix (no longer subscripting float)
  7. Performance metrics added (no training logic changed)
"""
import os
import torch
import torch.distributed as dist
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import time

from models.quantization import MixedPrecisionQuantizer
from models.flash_allreduce import FlashAllReduce
from models.tensor_parallel_gpt2 import TensorParallelGPT2


def setup_distributed():
    rank        = int(os.environ['RANK'])
    world_size  = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    print(f"[Rank {rank}] Initializing Tensor Parallelism + Flash Communication...")

    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

    return rank, world_size


def load_data(tokenizer, rank):
    if rank == 0:
        print("Loading wikitext-103-raw-v1 dataset...")

    dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)

    if rank == 0:
        print(f"Filtered dataset size: {len(dataset)} samples")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    if rank == 0:
        print(f"✓ Dataset loaded: {len(tokenized_dataset)} samples")
        print(f"  Batches per epoch: {len(dataloader)}")
        print("  NOTE: All nodes process SAME batches (Tensor Parallelism)")

    return dataloader


# ─────────────────────────────────────────────
# Performance tracker — no effect on training
# ─────────────────────────────────────────────
class PerfTracker:
    """
    Tracks performance metrics without touching training logic.
    All timing is additive — just wraps what already exists.
    """
    BATCH_SIZE = 4
    SEQ_LEN    = 128

    def __init__(self, world_size):
        self.world_size       = world_size

        # Step-level
        self.step_times       = []          # wall time per step (s)
        self.comm_times       = []          # Flash AllReduce time per step (s)
        self.losses           = []          # loss per logged step

        # Epoch-level
        self.epoch_start      = None

        # Run-level
        self.training_start   = None

    def start_training(self):
        self.training_start = time.time()

    def start_epoch(self):
        self.epoch_start = time.time()

    def record_step(self, step_time, comm_time, loss):
        self.step_times.append(step_time)
        self.comm_times.append(comm_time)
        self.losses.append(loss)

    # ── derived metrics ──────────────────────

    def samples_per_sec(self, step_time):
        """Sequences processed per second across all nodes."""
        return self.BATCH_SIZE / step_time

    def tokens_per_sec(self, step_time):
        """Tokens processed per second (batch × seq_len)."""
        return (self.BATCH_SIZE * self.SEQ_LEN) / step_time

    def comm_overhead_pct(self, step_time, comm_time):
        """What fraction of step time was communication."""
        return (comm_time / step_time) * 100 if step_time > 0 else 0

    def compute_time(self, step_time, comm_time):
        """Pure compute time = step time minus comm time."""
        return step_time - comm_time

    def avg_step_time(self):
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0

    def total_training_time(self):
        return time.time() - self.training_start if self.training_start else 0

    def bandwidth_saved_gb(self, num_calls, tensor_size_bytes):
        """
        Estimates how many GB of network traffic Flash AllReduce saved
        vs standard FP32 all-reduce.

        Flash uses INT4 for reduce-scatter (8x smaller) and INT8 for
        all-gather (4x smaller). Standard would send full FP32 both ways.
        """
        # Standard FP32 all-reduce would transfer 2 × tensor_size per call
        standard_bytes  = num_calls * 2 * tensor_size_bytes
        # Flash: reduce-scatter at 1/8, all-gather at 1/4
        flash_bytes     = num_calls * (tensor_size_bytes / 8 + tensor_size_bytes / 4)
        saved           = standard_bytes - flash_bytes
        return saved / (1024 ** 3)   # convert to GB

    def print_step(self, step, loss, step_time, comm_time, current_lr, comm_count):
        """Prints the per-step metrics row."""
        sps  = self.samples_per_sec(step_time)
        tps  = self.tokens_per_sec(step_time)
        ovhd = self.comm_overhead_pct(step_time, comm_time)
        comp = self.compute_time(step_time, comm_time)

        print(
            f"{step:<6} "
            f"{loss:<9.4f} "
            f"{current_lr:<10.2e} "
            f"{step_time:<10.3f}s "
            f"{comp:<10.3f}s "
            f"{comm_time*1000:<10.1f}ms "
            f"{ovhd:<8.1f}% "
            f"{sps:<10.1f} "
            f"{tps:<12.0f} "
            f"{comm_count:<6}"
        )

    def print_summary(self, step, flash_stats, reduce_ratio, gather_ratio):
        """Prints the full performance summary at end of training."""

        total_time   = self.total_training_time()
        avg_step     = self.avg_step_time()
        avg_sps      = self.BATCH_SIZE / avg_step if avg_step > 0 else 0
        avg_tps      = (self.BATCH_SIZE * self.SEQ_LEN) / avg_step if avg_step > 0 else 0
        total_samples= step * self.BATCH_SIZE
        total_tokens = total_samples * self.SEQ_LEN

        avg_comm     = sum(self.comm_times) / len(self.comm_times) if self.comm_times else 0
        avg_comp     = avg_step - avg_comm
        avg_ovhd     = (avg_comm / avg_step * 100) if avg_step > 0 else 0

        # Approximate gradient tensor size: GPT-2 hidden=768, seq=128, batch=4
        # RowParallel output shape = [4, 128, 768] = 393216 floats = 1.57 MB per call
        approx_tensor_bytes = self.BATCH_SIZE * self.SEQ_LEN * 768 * 4
        saved_gb = self.bandwidth_saved_gb(flash_stats['num_calls'], approx_tensor_bytes)

        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)

        print("\n── Training Throughput ─────────────────────────────")
        print(f"  Total training time:        {total_time:.2f}s  ({total_time/60:.1f} min)")
        print(f"  Total steps:                {step}")
        print(f"  Total samples processed:    {total_samples:,}")
        print(f"  Total tokens processed:     {total_tokens:,}")
        print(f"  Avg step time:              {avg_step:.3f}s")
        print(f"  Avg samples/sec:            {avg_sps:.1f}")
        print(f"  Avg tokens/sec:             {avg_tps:.0f}")

        print("\n── Compute vs Communication Breakdown ──────────────")
        print(f"  Avg compute time/step:      {avg_comp:.3f}s  ({100-avg_ovhd:.1f}% of step)")
        print(f"  Avg comm time/step:         {avg_comm*1000:.1f}ms  ({avg_ovhd:.1f}% of step)")
        print(f"  Flash AllReduce calls:      {flash_stats['num_calls']}")
        print(f"  Avg time per AllReduce:     {flash_stats['avg_time']*1000:.2f}ms")
        print(f"  Comms per step:             {flash_stats['num_calls']/step:.0f}  (expected 48)")

        print("\n── Flash Communication Savings ─────────────────────")
        print(f"  Reduce-Scatter ratio:       {reduce_ratio:.1f}x  (INT4 vs FP32)")
        print(f"  All-Gather ratio:           {gather_ratio:.1f}x  (INT8 vs FP32)")
        print(f"  Est. bandwidth saved:       {saved_gb:.2f} GB vs standard FP32 all-reduce")
        print(f"  Total comm time:            {flash_stats['total_time']:.2f}s")

        print("\n── Tensor Parallelism Efficiency ───────────────────")
        print(f"  Nodes:                      {self.world_size}")
        print(f"  Params per node:            ~{124.4/self.world_size:.1f}M  (of 124.4M total)")
        print(f"  Comm overhead:              {avg_ovhd:.1f}%  (lower is better)")
        print(f"  Parallel efficiency:        {100 - avg_ovhd:.1f}%")

        print("=" * 70)


def train():
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 70)
        print("GPT-2 FINE-TUNING: TENSOR PARALLELISM + FLASH COMMUNICATION")
        print(f"Dataset: wikitext-103-raw-v1")
        print(f"Nodes: {world_size}")
        print("=" * 70)

    # Flash Communication
    quantizer       = MixedPrecisionQuantizer(group_size=128)
    flash_allreduce = FlashAllReduce(quantizer, world_size, rank)

    if rank == 0:
        reduce_ratio = quantizer.quant_4bit.get_compression_ratio()
        gather_ratio = quantizer.quant_8bit.get_compression_ratio()
        print(f"\n✓ Flash Communication initialized")
        print(f"  Reduce-Scatter compression: {reduce_ratio:.1f}x (INT4)")
        print(f"  All-Gather compression:     {gather_ratio:.1f}x (INT8)")

    # Model
    if rank == 0:
        print(f"\nLoading GPT-2 with Tensor Parallelism ({world_size} nodes)...")

    model     = TensorParallelGPT2.from_pretrained('gpt2', flash_allreduce)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print(f"✓ Model split across {world_size} nodes")
        print(f"  Each node: ~{124.4 / world_size:.1f}M parameters")

    dataloader = load_data(tokenizer, rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    warmup_steps = 100
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )

    model.train()

    # ── Performance tracker (no effect on training) ──
    perf = PerfTracker(world_size) if rank == 0 else None
    if rank == 0:
        perf.start_training()

    if rank == 0:
        print("\nStarting training...")
        print("=" * 100)
        print(
            f"{'Step':<6} {'Loss':<9} {'LR':<10} "
            f"{'StepTime':<10} {'Compute':<10} {'CommTime':<10} "
            f"{'CommOvhd':<8} {'Samples/s':<10} {'Tokens/s':<12} {'Comm#':<6}"
        )
        print("=" * 100)

    step      = 0
    max_steps = 1000
    log_every = 10

    for epoch in range(10):
        if rank == 0:
            perf.start_epoch()

        for batch in dataloader:
            if step >= max_steps:
                break

            step_start  = time.time()
            comm_before = flash_allreduce.num_calls

            input_ids      = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels         = input_ids.clone()

            labels[attention_mask == 0] = -100

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            comm_count = flash_allreduce.num_calls - comm_before
            step_time  = time.time() - step_start

            # ── record + print metrics (rank 0 only, no effect on training) ──
            if rank == 0:
                comm_time  = flash_allreduce.total_time - sum(perf.comm_times)
                current_lr = scheduler.get_last_lr()[0]
                perf.record_step(step_time, comm_time, loss.item())

                if step % log_every == 0:
                    perf.print_step(
                        step, loss.item(), step_time,
                        comm_time, current_lr, comm_count
                    )

            step += 1

        if step >= max_steps:
            break

    if rank == 0:
        print("=" * 100)
        print("TRAINING COMPLETED!")

        stats        = flash_allreduce.get_stats()
        reduce_ratio = quantizer.quant_4bit.get_compression_ratio()
        gather_ratio = quantizer.quant_8bit.get_compression_ratio()

        # Flash Communication Statistics (unchanged)
        print(f"\nFlash Communication Statistics:")
        print(f"  Total All-Reduce calls:   {stats['num_calls']}")
        print(f"  Total communication time: {stats['total_time']:.2f}s")
        print(f"  Average per call:         {stats['avg_time'] * 1000:.2f}ms")
        print(f"  Reduce-Scatter ratio:     {reduce_ratio:.1f}x (INT4)")
        print(f"  All-Gather ratio:         {gather_ratio:.1f}x (INT8)")
        comms_per_batch = stats['num_calls'] / step if step > 0 else 0
        print(f"  Communications per batch: {comms_per_batch:.0f}")
        print(f"    Expected: ~48 (24 forward + 24 backward for 12 layers)")

        # Full performance summary
        perf.print_summary(step, stats, reduce_ratio, gather_ratio)

        print("\nSaving model...")
        torch.save(model.state_dict(), 'gpt2_finetuned_wikitext103.pt')
        print("✓ Model saved to gpt2_finetuned_wikitext103.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()