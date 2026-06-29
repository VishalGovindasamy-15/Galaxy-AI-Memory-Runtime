# Experiment 01 — SSD Bandwidth

**Date:** 20260628_134323
**Objective:** Measure SSD read throughput and latency for block sizes 1-128 MB
**Hypothesis:** H1
**Git Commit:** 154826c8dfe9649a4f4384650d5c62bc72343509

## Observations
**Observation 1: Sequential and random throughput are similar.**
Sequential (32MB): 1.55 GB/s. Random (32MB): 1.78 GB/s. 
Normally sequential > random. The DRAM-less architecture and block alignment may cause caching/readahead behavior inside the NVMe controller that equalizes these.

**Observation 2: High variance at larger block sizes.**
At 16MB and 32MB, the standard deviation is extremely high (e.g. 11ms on a 21ms mean). The SSD is not deterministic. The runtime scheduler must plan for *worst-case stalls*, not just average throughput.

**Observation 3: Hard limit around ~1.5 - 2.2 GB/s.**
Bandwidth plateaus quickly. A secondary verification with `dd iflag=direct` measured 2.2 GB/s, confirming that this is the real hardware limit of the SSD, not a Python overhead artifact. The cost model must use ~2 GB/s, not theoretical PCIe maximums.

**Observation 4: Long Tail Latencies (from exp01b)**
A 100-iteration repeatability test at 32MB block size revealed:
- Mean latency: 25.30 ms
- P95 latency:  51.15 ms
- Max latency:  57.10 ms
The 95th percentile latency is double the mean. Transfer jitter is huge. The H1 safety margin (`M > σ_transfer`) is absolutely necessary.

## Conclusions
The SSD transfer time is highly stochastic. The GAMR scheduler cannot assume a fixed latency; it must use the latency distribution to calculate risk.

## Research Direction Shift (Emerged from Exp01)

HAMR's central problem is **not** "hiding bandwidth."
HAMR's central problem is **managing uncertainty in hierarchical memory systems.**

This changes the core objective from:

```
Minimize average transfer time
```
to:
```
Minimize P(stall)  ←  probability that transfer time exceeds compute time
```

**Observation 5: Possible bimodal distribution.**
Some histograms show two clusters of latency values rather than a single Gaussian. This could indicate the DRAM-less NVMe controller switching between internal states (SLC cache available / exhausted, or different NAND dies being accessed, or FTL garbage collection events). Not yet confirmed — keep in notebook, do not add to architecture.

**New Hypothesis H4 (created by Exp01 data):**
> Does a distribution-aware (risk-based) scheduler outperform a mean-based scheduler under stochastic storage latency?

This hypothesis did not exist when the project started. The experiment created it.

## Next Action
Proceed to Exp02 (PCIe bandwidth). Measure PCIe as a stochastic system, not just a deterministic channel. The key question is: Is PCIe more predictable than the SSD? If yes, that contrast directly influences scheduler design.

