# Future Ideas

> Ideas that don't yet come from experiment results.
> These are NOT in ARCHITECTURE_v1. They go here until validated by data.
>
> Rule: An idea moves from this file into a new architecture version
> only after an experiment demonstrates it's needed or beneficial.

---

## Energy Efficiency Metric (Post-V3)

After V3 is working, measure **Joules per token** as an additional evaluation axis.

```
Energy per token = Power (Watts) / Throughput (tokens/sec)

Example:
  AirLLM:  100W / 10 tok/s  =  10.0 J/token
  HAMR V3: 120W / 18 tok/s  =   6.7 J/token
```

Why it matters: If HAMR uses slightly more power but runs models 3× larger
at better efficiency, that's a publishable advantage beyond throughput alone.

**Do not measure this until V3 is complete and externally benchmarked.**



---

## Multi-GPU Streaming

Stream chunks across multiple GPUs using NVLink or PCIe.
Relevant when: a single GPU's VRAM is the bottleneck.
Requires: Multi-device ResourceManager, chunk routing.

---

## Training Support

Stream gradients, optimizer states (Adam momentum/variance) from SSD.
Relevant when: we want to fine-tune models larger than VRAM.
Challenge: Optimizer state = 2–4× model size. Massive I/O amplification.
Requires: Separate GradientChunk / OptimizerChunk pipeline.

---

## Mixture-of-Experts (MoE) Expert Selection

For MoE models (Mixtral, DeepSeek), only 2–8 of N experts activate per token.
Could enable streaming only the active experts instead of all N.
Requires: Expert activation prediction (soft prediction, not certainty).

---

## Neural Network Scheduler

Replace the rule-based Adaptive Cost Model with a tiny neural network
trained on execution traces. Could generalize across hardware.
Requires: Sufficient execution trace data (Phase 4+).

---

## GGUF Format Direct Streaming

Read weight chunks directly from GGUF files (Ollama models on disk).
Avoids format conversion step.
Deferred reason: Adds debugging complexity in early phases.
When to revisit: After V3 (Adaptive Runtime) is fully validated.

---

## Cross-Machine Distributed Streaming

Stream chunks from a remote NAS or distributed storage.
Relevant when: model exceeds local SSD capacity (5T+ parameter models).
Requires: Network bandwidth model, latency tolerance.

---

## Hardware Decompression

Use GPU hardware decompression (NVIDIA Ampere+ has built-in decompression)
to decompress chunks as they land in VRAM — reducing PCIe transfer size.
Requires: Specific GPU feature. Not available on RTX 3050 (our test hardware).

---

## Speculative Prefetching for MoE

For models where expert routing is partially predictable,
prefetch likely experts before the routing decision is made.
This reintroduces the "prediction" problem we deliberately removed.
Only worth it if MoE models prove dominant.

---

## SSD RAID Striping

Use both NVMe SSDs in parallel (RAID-0 style) to double read bandwidth.
On our hardware: Micron 2550 + Sandisk SN740 = potentially 8–12 GB/s combined.
Requires: OS-level setup or custom parallel reader.

---

## Chunk Deduplication

If two operations use identical weight chunks (shared parameters),
load once and reference from multiple ComputeOperations.
Relevant for: models with weight tying (e.g., embedding ↔ output head).
