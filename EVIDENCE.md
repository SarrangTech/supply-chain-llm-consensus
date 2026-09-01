# Evidence: these runs really executed on Northeastern's Discovery cluster

This file exists because a summary/narrative (NOTES_AND_ASSUMPTIONS.md
section (f)) is not independently verifiable on its own -- it's just
prose. This file captures the raw, cluster-generated artifacts that back
every claim there: SLURM's own accounting records, physical GPU hardware
identities, and the model loader's own confirmation of which weights were
actually loaded. None of this is something a script running locally
(without cluster access) could produce.

**To re-verify any of this yourself**, independent of anything written
here: `ssh explorer-login` (or your own account on the same cluster) and
run the commands shown before each block below directly.

## 1. SLURM accounting database (the cluster's own system of record)

```
sacct -u bhawesh --starttime=2026-08-29 -o JobID,JobName%25,Partition,NodeList,State,Elapsed,Start,End -X
```

```
JobID                          JobName  Partition        NodeList      State    Elapsed               Start                 End
9808333                    ollama-test    sharing           d1027  COMPLETED   00:00:06 2026-08-29T15:44:39 2026-08-29T15:44:45
9808352                   ollama-test2    sharing           d1027  COMPLETED   00:01:47 2026-08-29T15:46:40 2026-08-29T15:48:27
9808377                ollama-70b-test    sharing           d4077  COMPLETED   00:19:54 2026-08-29T16:05:46 2026-08-29T16:25:40
9808451                ollama-cpu-test      short           d0139    TIMEOUT   00:20:05 2026-08-29T16:14:46 2026-08-29T16:34:51
9808599               ollama-cpu-test2      short           d0139  COMPLETED   00:05:16 2026-08-29T16:34:59 2026-08-29T16:40:15
9808671              ollama-cpu-avx512      short           d0032  COMPLETED   00:00:27 2026-08-29T22:14:58 2026-08-29T22:15:25
9818544                 ollama-70b-cpu      short           d0002    TIMEOUT   00:30:28 2026-08-30T23:44:54 2026-08-31T00:15:22
9835369                pilot-neg-large        gpu           d4054  COMPLETED   00:11:37 2026-08-31T12:48:50 2026-08-31T13:00:27
9836195             p-standalo-sm-cost      short           d0020  COMPLETED   00:02:16 2026-08-31T13:50:55 2026-08-31T13:53:11
9836196             p-standalo-lg-cost        gpu           d4055  COMPLETED   00:03:00 2026-08-31T13:43:40 2026-08-31T13:46:40
9836197             p-standalo-sm-bull      short           d0020  COMPLETED   00:02:08 2026-08-31T13:53:58 2026-08-31T13:56:06
9836198             p-standalo-lg-bull        gpu           d4053  COMPLETED   00:03:27 2026-08-31T13:44:42 2026-08-31T13:48:09
9836199             p-info_sha-sm-cost      short           d0020  COMPLETED   00:02:36 2026-08-31T13:57:01 2026-08-31T13:59:37
9836200             p-info_sha-lg-cost        gpu           d4052     FAILED   00:03:19 2026-08-31T13:46:02 2026-08-31T13:49:21
9836201             p-info_sha-sm-bull      short           d0020  COMPLETED   00:02:35 2026-08-31T14:00:03 2026-08-31T14:02:38
9836202             p-info_sha-lg-bull        gpu           d4052  COMPLETED   00:03:20 2026-08-31T13:46:02 2026-08-31T13:49:22
9836203             p-standalo-sm-cost      short           d0020  COMPLETED   00:02:16 2026-08-31T14:03:04 2026-08-31T14:05:20
9836204             p-standalo-lg-cost        gpu           d4055  COMPLETED   00:00:52 2026-08-31T13:46:48 2026-08-31T13:47:40
9836205             p-standalo-sm-bull      short           d0029    TIMEOUT   00:45:11 2026-08-31T17:00:12 2026-08-31T17:45:23
9836206             p-standalo-lg-bull        gpu           d4055  COMPLETED   00:00:51 2026-08-31T13:47:49 2026-08-31T13:48:40
9836207             p-info_sha-sm-cost      short           d0029    TIMEOUT   00:45:02 2026-08-31T17:45:27 2026-08-31T18:30:29
9836208             p-info_sha-lg-cost        gpu           d4055  COMPLETED   00:01:03 2026-08-31T13:48:51 2026-08-31T13:49:54
9836209             p-info_sha-sm-bull      short           d0005  COMPLETED   00:03:44 2026-08-31T17:46:53 2026-08-31T17:50:37
9836210             p-info_sha-lg-bull        gpu           d4053  COMPLETED   00:01:07 2026-08-31T13:48:51 2026-08-31T13:49:58
9836211             p-negotiat-sm-cost      short           d0005  COMPLETED   00:29:26 2026-08-31T17:50:55 2026-08-31T18:20:21
9836215             p-negotiat-sm-bull      short           d0005  COMPLETED   00:29:14 2026-08-31T18:20:31 2026-08-31T18:49:45
9836370             p-negotiat-lg-cost        gpu           d4052  COMPLETED   00:19:08 2026-08-31T13:49:52 2026-08-31T14:09:00
9836378             p-negotiat-lg-bull        gpu           d4052     FAILED   00:19:07 2026-08-31T13:49:52 2026-08-31T14:08:59
9851695               info_sha-la-cost        gpu           d4054  COMPLETED   00:01:37 2026-08-31T20:23:37 2026-08-31T20:25:14
9852815               standalo-sm-bull      short           d0005  COMPLETED   00:02:09 2026-08-31T21:10:11 2026-08-31T21:12:20
9852816               info_sha-sm-cost      short           d0005  COMPLETED   00:02:44 2026-08-31T21:12:43 2026-08-31T21:15:27
```

Two jobs show `FAILED`: 9836200 (real data loss, rerun as 9851695) and
9836378 (cosmetic -- rerun wasn't needed, results were already valid; see
NOTES_AND_ASSUMPTIONS.md section (f), "port-collision bug"). Two show
`TIMEOUT`: 9836205 and 9836207 (noisy-neighbor CPU contention, rerun as
9852815/9852816 with `--exclusive`). Cross-reference `pilot_job_ids.txt`
for which shard each job ID corresponds to.

## 2. Physical GPU hardware used (UUIDs are burned into each card)

```
grep -h "UUID: GPU" logs/*.log | sort -u
```

```
GPU 0: NVIDIA H200 NVL (UUID: GPU-c38b1db7-0298-dede-10e9-be23e466e60d)
GPU 0: NVIDIA H200 (UUID: GPU-98443ff6-a2ed-3db4-aabb-18eea5cc23e5)
GPU 0: Tesla V100-SXM2-32GB (UUID: GPU-052915b0-e58c-186b-39b1-fd0e7278bc46)
```

Three distinct physical GPUs across three different jobs -- consistent
with the cluster's scheduler placing jobs on whichever node/GPU was free,
not a single reused environment.

## 3. Model loader's own confirmation of which weights loaded

70B (`llama3.1:70b`, "large" tier):
```
llama_model_loader: - kv   2:  general.name str = Meta Llama 3.1 70B Instruct
print_info: general.name          = Meta Llama 3.1 70B Instruct
```

8B (`llama3.1:8b`, "small" tier):
```
llama_model_loader: - kv   2:  general.name str = Meta Llama 3.1 8B Instruct
print_info: general.name          = Meta Llama 3.1 8B Instruct
```

These come directly from llama.cpp's own GGUF metadata parser reading the
downloaded model file's embedded name field -- not a config value we set,
a fact read back out of the weights file itself.

## 4. CUDA vs. Vulkan backend confirmation (explains the throughput table in section (f))

```
grep -h "library=CUDA\|library=Vulkan" logs/*.log | sort -u
```

H200 nodes (driver >= 550, real CUDA path):
```
library=CUDA compute=9.0 name=CUDA0 description="NVIDIA H200" driver=12.8 total="139.8 GiB"
```

V100 node (driver 545, below Ollama's CUDA floor, falls back to Vulkan --
still real GPU offload, confirmed separately via `ollama ps` showing
`100% GPU` in that job's log):
```
library=Vulkan compute=0.0 name=Vulkan0 description="Tesla V100-SXM2-32GB" total="32.0 GiB"
```

## 5. AVX512 CPU confirmation (small-tier Cascade Lake nodes)

```
grep -h "AVX512" logs/pilot_standalone_small_cost_9836195.log | head -1
```

```
system_info: n_threads = 56 (n_threads_batch = 56) / 112 | CPU : ... AVX512 = 1 | AVX512_VNNI = 1 | ...
```

## How this maps to the repo's other evidence files

- `pilot_job_ids.txt` -- which shard (framework/tier/metric) each job ID
  above corresponds to.
- `results/pilot_*.json` -- the actual cost/bullwhip/elapsed_sec output
  each job produced.
- `NOTES_AND_ASSUMPTIONS.md` section (f) -- the narrative explaining what
  each of these numbers means and the two infra bugs found along the way.
