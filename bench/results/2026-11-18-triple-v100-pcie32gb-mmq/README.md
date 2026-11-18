# Triple NVIDIA Tesla V100-PCIE-32GB (`sm_70`, fully resident) — `STRATA_MMQ` A/B and the v0.2.15 tree delta vs the 0.1.58 production build (2026-11-18)

Follow-up measurement to PR #6023 and `bench/results/2026-11-18-dual-v100-sxm2-256k-mmq` (dual SXM2-16GB, 54% Tier-0 residency + pinned-RAM tier): the same `STRATA_MMQ` question asked on a **fully-resident** 3x32GB PCIe rig, where the answer inverts. Includes a same-day paired deep-band run that turns the tree delta itself into a band trade-off.

## Rig

- **GPUs:** 3 x Tesla V100-PCIE-32GB (`sm_70`, TCC, 250 W) + idle GTX 1060 (excluded from compute); PCIe Gen3 x16
- **Host:** 2 x Xeon E5-2696 v3 (36C/72T), 128 GiB DDR3L-1600 ECC, Windows 10 Enterprise LTSC 2021, driver 581.80
- **Shape:** `--layer-split 16,32` (16/16/16 over 48 layers), `--max-context 262144`, `--kv int8`, `--kv-resident 32768`, `--pipeline-windows 2`, `--spec 4 --spec-min-p 0.70` (MTP drafter + suffix n-gram), `--prefill-pipe 1536`, Q8_0 engram via `--ple-gguf` (standalone PLE, direct-read probe 31,588 rows/s), 24576/24576 expert slots (100% residency) on **every** engine below

## Model

Qwen3.8-Flash-Next 125B (Huihui abliterated v2), Unsloth UD-IQ4_XS **v31 requant** (96/96 patch coverage; i-quant experts: IQ4_XS + Strata Q2_0 + Q8_0 layers), 4 shards on NVMe.

## Engines under test

| arm | engine | build |
|---|---|---|
| prod baseline | 0.1.58 **prebuilt** (upstream release; its on-disk BUILD.json is stale 0.1.55 metadata — binary identity verified via `--version` + md5 against the kept backup) | upstream CUDA 12.9-era recipe |
| MMQ=ON | v0.2.15 tag + local cherry 9393777 (PLE standalone fix, cf. PR #5942) | self-build: MSVC + CUDA 12.4, `-DCMAKE_CUDA_ARCHITECTURES=70`, `-DSTRATA_MMQ=ON` (`build-0215.bat`); binary self-reports 0.1.61 |
| MMQ=OFF (control) | identical sources, identical recipe, flag omitted (`build-0215-nommq.bat`) | self-build, same toolchain |
| historical | 0.1.59.1 self-build of v0.2.13.1 (measured 2026-11-14) | same recipe family |

Both v0.2.15 arms ran with `STRATA_PREFILL_LEND_PCT=0` (issue #5943 guard) and passed the deployment verifier (single serve instance, layer split, conversation cache, watchdog state) plus `/metrics expert_slots = 24576`. i-quant MMQ engages automatically once built (`moe_mmq.cu supported()`: IQ4_XS / Q2_0 / Q8_0 ungated; only K-quant experts need `STRATA_MMQ_KQ=1`, absent here).

## Method

`flash-ab/bench_fnfl_ab.py`: engine-side per-request timing lines, deterministic fresh pseudo-word prompts (`reused = 0` enforced per round — nothing served from cache), n=3 rounds + warm-up, medians; `temperature 0`. Deep band uses the same seeds for both engines (96,784 / 96,841 / 96,732 prompt tokens, identical across arms). Quality: 10 mechanical graders (`bench_quality_probes.py`).

## Results

**Mid band — 11,794-token prompt, 200 generated (n=3 medians):**

| engine | prefill tok/s | decode tok/s |
|---|---:|---:|
| 0.1.58 prebuilt (2026-11-14 retest, same protocol) | 2099.8 | 103.1 |
| 0.1.59.1 self-build of v0.2.13.1 (2026-11-14) | 1857.2 | 104.6 |
| **v0.2.15 self-build, `STRATA_MMQ=OFF`** | **2471.9** (2418/2472/2475) | 105.9 |
| v0.2.15 self-build, `STRATA_MMQ=ON` | 1832.2 (1826/1832/1860) | 101.7 |

**Deep band — 96.8k-token prompt, 200 generated (same seeds both engines, n=3 medians, measured 2026-11-18):**

| engine | prefill tok/s | decode tok/s |
|---|---:|---:|
| **0.1.58 prebuilt** | **3499.2** (3478/3499/3501) | 99.4 |
| v0.2.15 self-build, `STRATA_MMQ=OFF` | 2796.3 (2759/2810/2796) | 98.0 |

**Short band decode (n=3):** 113.5 / 117.4 (2026-11-14 arms) vs 109.5 / 118.2 (v0.2.15 arms) — inside the speculative-sampling jitter band (single rounds scatter 108–126); decode is parity across all four engines.

## Findings

1. **`STRATA_MMQ=ON` is a large prefill regression on a fully-resident i-quant rig: -25.9% at 11.8k vs the same tree with the flag off** (single-flag A/B, same toolchain). The +45% measured on the tiered dual-SXM2-16GB rig does not transfer: MMQ's win there comes from skipping dequant staging under VRAM starvation; with 100% HBM residency Strata's own sm_70 kernels win for IQ4_XS by a wide margin. Practical rule: **fully-resident Volta rigs on i-quant models should build with `STRATA_MMQ=OFF`** (it is build-time only — no runtime footgun; the kernels simply don't ship).
2. **The v0.2.15 tree vs the 0.1.58 prebuilt is a band trade-off, not a uniform win**: +17.7% prefill at 11.8k, **-20.1% at 96.8k**, decode parity. The two engines scale differently with depth (0.1.58: 2099.8 → 3499.2 from 11.8k to 96.8k; v0.2.15-OFF: 2471.9 → 2796.3), crossing somewhere in between. The production rig therefore stays on 0.1.58. Unexplored lever: the v0.2.15 engine logs `--prefill auto may take chunks up to 32768 (STRATA_NV_CHUNK=0 keeps 8192, =16 keeps 16384)` — deep-band chunk tuning was not attempted in this pass.
3. **Quality probes are equal across both v0.2.15 arms (8/10, identical pattern)**: `recur`'s stored expectation (-49198) is itself arithmetically wrong — a(n+1)=3a(n)-7 with a1=1 gives a10 = (7 - 5*3^9)/2 = **-49204**, which both arms answered (the older-engine reference answered -16384, wrong in a different way); and `bigmul` now exhausts `max_tokens=400` with empty output on **both** v0.2.15 arms (it passed on the older engine) — a tree-level behavior change worth a look, not an MMQ effect.

## Notes

- The runtime dirs of the experimental arms ship their own matching `strata_kq.dll` (SHARED, built only with `STRATA_MMQ=ON`) — no cross-version DLL mixing.
- The v0.2.15 builds compiled against CUDA 12.4 headers and ran against the production dir's 12.9-era `cublas64_12.dll` / `cublasLt64_12.dll` / `cudart64_12.dll` (minor-version compatible; both arms clean).
- PR #5942 (the cherry-picked PLE standalone fix, required by this rig's `--ple-gguf` shape) and issue #5943 (prefill slot lending) are still open upstream; `STRATA_PREFILL_LEND_PCT=0` is the local guard on any 0.1.59+ build.

## Files

- `results-fnfl-mmq0215.json` / `results-fnfl-off0215.json` — mid-band, per-round rows (prefill/decode tok/s, accept rates, engine + PLE lines, per-GPU memory)
- `results-fnfl-off0215-deep.json` / `results-fnfl-v0158-deep-1118.json` — deep band, both engines
- `quality-off0215.json` / `quality-mmq0215.json` / `quality-mmq0215b.json` / `quality-base10.json` — probe transcripts (2026-11-18 arms + 2026-11-01 older-engine reference)
