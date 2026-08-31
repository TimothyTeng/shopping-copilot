# TechJam Challenge 4 — Shopping Copilot

Conversational product search over a frozen 50,000-product Amazon catalog. An
agent asks a simulated shopper questions and returns 10 recommendations per
turn; the session ends when the hidden target appears in the list.

- `copilot/` — all of our work (the agent + tooling). Pure standard library on
  the scored path: no network, no LLM, no third-party runtime deps.
- `techjam-conversational-search/` — the organizers' participant kit (evaluator,
  starter, docs). Vendored **without** its git history and **without** the
  catalog data (see below). Treated as read-only; we import its evaluator.
- `CLAUDE.md` — project notes: scoring rules, measured results, working rules.

## 1. Prerequisites

- **Python 3.12** (developed on 3.12.10). The agent itself needs only the
  standard library.
- Optional, for retraining the category classifier only (never on the scored
  path): `numpy`, `scipy`, `scikit-learn`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
```

## 2. Download the catalog data (required, not in this repo)

The 50,000-row catalog is **not** committed — it is derived from
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab,
UCSD, category `Clothing_Shoes_and_Jewelry`) and must be fetched from the
organizers' GitHub Release. See `techjam-conversational-search/DATA_ATTRIBUTION.md`.

1. Download `catalog.jsonl.gz` from the releases page of
   <https://github.com/TechJam2026/techjam-conversational-search/releases>
2. Place it in `techjam-conversational-search/data/` and decompress it to
   `catalog.jsonl` (expected: 50,000 rows):

   ```bash
   cd techjam-conversational-search/data
   gzip -dk catalog.jsonl.gz          # -k keeps the .gz
   ```

3. Verify integrity against the shipped checksum:

   ```bash
   sha256sum -c SHA256SUMS            # checks catalog.jsonl.gz
   ```

`public_set.jsonl` (200 labeled dev sessions) is already included in the kit.

## 3. Run

All commands run from `copilot/`. `PYTHONIOENCODING=utf-8` is required on
Windows (the cp1252 console default crashes on product titles with emoji or
typographic dashes).

Score against the official evaluator (200 sessions, ~11s):

```bash
cd copilot
PYTHONIOENCODING=utf-8 python -m tools.harness run
```

Independent natural-language stress test (built to find failures, not confirm
success):

```bash
PYTHONIOENCODING=utf-8 python -m tools.stress --show
```

Talk to it yourself:

```bash
PYTHONIOENCODING=utf-8 python -m tools.demo chat
```

See `CLAUDE.md` and `copilot/README.md` for the full command list, scoring
mechanics, and the measured-and-rejected log.

## Current state

Benchmark `score 0.9456  hit@10 1.000  MRR 0.911  MTTC 2.38` (baseline 0.1067).

The LLM augmentation is **built and measured**: `src/backends/` holds the seam
(protocol + null impl) and a HyDE rewriter that generates the product listing a
shopper is describing and retrieves with that. It is **off by default**
(`backend="null"`), so the scored path stays stdlib-only and runs
network-disabled. Measured: it lifts the natural-language prose path
`0.6339 → 0.6599` with recall held exactly, and *costs* the graded path
(`0.9390`), so it is a free-text augmentation only. See `copilot/README.md`,
"The optional model tier".

To try it, with a local OpenAI-compatible server running:

```bash
COPILOT_LLM_BASE=http://localhost:30800/v1 \
COPILOT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
PYTHONIOENCODING=utf-8 python -m tools.llm_check      # smoke test + cost
PYTHONIOENCODING=utf-8 python -m tools.llm_check --offline   # fallback proof
PYTHONIOENCODING=utf-8 python -m tools.demo chat --backend hyde
PYTHONIOENCODING=utf-8 python -m tools.stress --retrieval bm25 --set backend=hyde
```

## Data use

The catalog derives from Amazon Reviews 2023. Follow the source dataset's terms;
use for the competition, research, and other permitted purposes only. The
organizer does not claim ownership of the underlying Amazon product content.
