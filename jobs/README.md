# Job-ID provenance files

Each file is a flat text log, one SLURM job per line, format:

```
<tier> <framework> <metric> <job_id>
```

e.g. `small negotiation_tool cost 9836211` — lets you independently verify
via `sacct -j <job_id>` (on the Explorer cluster) that a given result file
was actually produced by a real cluster job, not fabricated.

| File | Campaign | Date |
|---|---|---|
| `pilot.txt` | 20-shard pilot grid (`--steps 20` smoke test) | 2026-08-31 |
| `full.txt` | First full 200-step x 25-config grid | 2026-09-02 |
| `full2.txt` | Full grid rerun after the 3 negotiation bug fixes | 2026-09-13 |

## Known gap: no file for several later campaigns

NOTES_AND_ASSUMPTIONS.md section (l.2) references a `gemma_full_job_ids.txt`
that **was never actually created** — the Gemma small-tier grid's job IDs
were tracked only in conversation/commit history, not saved to a file. This
is a real, pre-existing gap, documented here rather than retroactively
patched with a fabricated file.

The same is true, by omission, for every campaign after that one: the
48-hour push, the RQ2 paired-ablation reruns, the RQ3 transcript-capture
reruns, and the RQ2 mechanism-investigation reruns all have their job IDs
recorded only in NOTES_AND_ASSUMPTIONS.md prose and git commit messages,
not in a `jobs/*.txt` file. If a dedicated file is wanted for any of these
later campaigns, it would need to be reconstructed from `sacct` history or
the relevant NOTES sections — not assumed to already exist.
