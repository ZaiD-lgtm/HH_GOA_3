# hhg3 — face scan → social post → blockchain anchor

HH Goa 2026 shortlisting, **Task 3**. A command-line pipeline that takes a face
scan, finds a real matching social-media post on the open web, and writes a
tamper-evident record of that finding to a blockchain — then proves the record
still matches by re-verifying it against the chain.

> **Status: prototype skeleton.** Every stage boundary, data shape and CLI
> command is in place and the local-chain path runs end to end. The pieces that
> still need to be turned on for the submission are listed under
> [What is not done yet](#what-is-not-done-yet).

```
samples/probe.jpg
      │
      ▼  1. detect + embed (YuNet → SFace 128-d, or InsightFace ArcFace 512-d)
   probe crop ──────────────────────────────────────────────┐
      │                                                     │
      ▼  2. reverse image search of the SOURCE image
         (SerpAPI Google Lens / Cloud Vision / Yandex)
   N candidate pages ─── keep social platforms only         │
      │                                                     │
      ▼  3. confirm: download each candidate image,         │
         re-detect, re-embed, cosine vs. probe ◄────────────┘
   best match above threshold
      │
      ▼  4. canonical record → sha256 → chain
   runs/<id>/bundle.json + tx receipt
      │
      ▼  hhg3 verify --run runs/<id>
   recompute hash locally == hash read back from chain
```

## Why it is built this way

The search provider is deliberately not trusted to decide identity. It only
proposes URLs; stage 3 downloads whatever image the candidate page actually
serves, runs detection and embedding on it again, and scores it against the
probe. So the "match" is a face-recognition decision made locally with a
recorded similarity score and threshold — not a search-engine ranking, and not
a hardcoded result.

That split also fixes a trap worth knowing about: **reverse image search matches
photographs, not faces.** Measured on this pipeline — same photo, same provider —
searching the cropped face returned **0** results from Google Lens, while
searching the full source image returned **59 visual matches, 13 of them on
social platforms** across Instagram, X, LinkedIn, Facebook, Reddit, YouTube and
TikTok. So stage 2 sends the source image and stage 3 uses the face crop.
`--search-crop` restores the old behaviour if you want to see this for yourself.

Only a 32-byte hash goes on-chain. The face embedding never leaves the machine
and is excluded from the record ([`evidence.py`](src/hhg3/evidence.py)), which
matters because a public chain is permanent and biometric data is not something
to publish.

## Install

```bash
git clone https://github.com/Pragyan330/HH_goa.git
cd HH_goa
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e .
```

That is enough for a real run: the default face backends are **YuNet**
(detection) and **SFace** (128-d recognition), both bundled with OpenCV. Their
ONNX weights (~340 KB and ~37 MB) download themselves into `models/` on first
use.

Optional extras:

```bash
pip install -r requirements-face.txt   # insightface + onnxruntime (ArcFace 512-d)
pip install -r requirements-evm.txt    # web3 + py-solc-x
```

> `onnxruntime` has no wheels for Python 3.14 yet, so the InsightFace backend
> needs **3.11 or 3.12**. Everything else, YuNet and SFace included, runs on
> 3.14 — `hhg3 doctor` shows which backends actually resolved.

Copy `.env.example` to `.env` and fill in whichever keys you have:

```bash
cp .env.example .env
```

Check what is actually wired up:

```bash
hhg3 doctor
```

## Run

```bash
# full pipeline, local chain (no keys, no gas)
hhg3 run --image samples/probe.jpg

# real search + public testnet
hhg3 run --image samples/probe.jpg --provider serpapi --chain evm

# re-verify a finished run against the chain
hhg3 verify --run runs/20260903T014500Z-a1b2c3
```

Useful flags: `--threshold 0.30` (looser match), `--any-domain` (do not restrict
to social platforms), `--search-crop` (search the face crop instead of the source
image), `--provider mock --allow-mock` (offline plumbing test),
`--embedder sface|insightface|fallback`, `--json`.

The match threshold defaults to whatever the active embedder declares, because a
cosine score is only meaningful relative to the model that produced it: **0.363**
for SFace (OpenCV's documented same-identity threshold) and **0.45** for ArcFace.
`--threshold` overrides it.

Each run writes `runs/<id>/`:

| file | contents |
| --- | --- |
| `probe_crop.jpg` | the detected face, padded |
| `candidates/cand_NN.jpg` | every candidate image that was downloaded and scored |
| `record.json` | the canonical record — exactly what gets hashed |
| `bundle.json` | record + record hash + chain receipt + full search trace |

## Which blockchain

Two backends, selected with `--chain`:

- **`local`** (default) — a hash-linked proof-of-work chain in
  `runs/_localchain.json`. Fully offline, used by the tests, and `validate()`
  detects any edit to an earlier block.
- **`evm`** — any JSON-RPC chain; defaults target **Ethereum Sepolia**. Two modes:
  - `EVM_MODE=calldata` (default) — a 0-value self-transaction carrying
    `"HHG3" || record_hash || metadata` as input data. No contract, no deploy.
  - `EVM_MODE=contract` — [`AnchorRegistry.sol`](contracts/AnchorRegistry.sol),
    a `bytes32 → {submitter, timestamp, uri}` mapping with an `Anchored` event
    and first-write-wins semantics. Deploy with
    `python scripts/deploy_contract.py`, then set `CONTRACT_ADDRESS`.

Re-verification is the same in both cases: recompute `sha256(canonical_json(record))`
from the local bundle, read the record back off the chain, compare.

## Tests

```bash
pytest
```

Covers canonical hashing, local-chain integrity, and — the ones that matter —
that editing `record.json` after anchoring makes `hhg3 verify` fail.

## What is not done yet

- [x] Confirmed end-to-end run against a live social post — SerpAPI Google Lens
      returned 25 candidates, 9 on social platforms, 6 scored by face similarity
      (0.83–0.95), matched an X post, anchored and re-verified.
- [ ] Testnet run on Sepolia with a funded key, and the tx link recorded here.
- [ ] Thresholds sanity-checked on real pairs. The defaults are the model
      authors' published numbers, not values measured on this pipeline's crops.
- [ ] Screen recording of the end-to-end run.

## Known limitations

- **This finds the photo, not the person.** Stage 2 can only surface pages
  hosting that same photograph. A different photo of the same face will not be
  found — that would need a face-search index (PimEyes and similar), which this
  project deliberately does not use.
- **Social platforms serve crawler gateways, not image files.** Both search
  providers hand back `lookaside.fbsbx.com` / `lookaside.instagram.com` URLs.
  Facebook's returns HTML to a browser user-agent and the real JPEG to a crawler
  one; Instagram's returns HTML to everything but carries an `og:image` pointing
  at the actual CDN file. [verify/match.py](src/hhg3/verify/match.py) falls back
  through both, which took the social scoring rate from **2/9 to 9/9** on a test
  probe. Sites that do neither are skipped and logged rather than scored.
- **The Yandex provider scrapes HTML** and will break when the markup changes or
  when it is served a captcha. It is a fallback, not the primary path.
- **The source image is uploaded to a public host** (catbox.moe) because Lens
  needs a URL it can fetch, and catbox is intermittently flaky — uploads retry
  three times. tmpfiles.org was tried and rejected: Lens returns nothing for its
  links. The Cloud Vision provider POSTs the image directly and avoids the
  public host entirely; prefer it when the probe is someone else's photo.
- **Bing Visual Search is not available.** Microsoft retired the entire Bing
  Search API family on 2025-08-11 — no new signups, 410 on existing keys — so
  that provider was removed rather than left as a trap.
- **No liveness or spoof detection.** A printed photo or a screen would pass.
- **The chain proves *when*, not *what*.** An anchor shows this exact record
  existed at that block — it does not prove the match was correct.

## Scope and use

Built for a shortlisting task, on images the operator is entitled to search.
Reverse face search on people who have not consented is a privacy harm and is
restricted or unlawful in several jurisdictions; nothing here is intended for
surveillance or for identifying strangers.

## Layout

```
src/hhg3/
  cli.py            argparse entry point (run / verify / doctor / chain)
  pipeline.py       stage orchestration
  evidence.py       canonical record + bundle read/write
  hashing.py        canonical JSON + sha256
  face/             detect.py, embed.py, compare.py  (pluggable backends)
  models.py         lazy download/cache for the YuNet + SFace ONNX weights
  search/           serpapi_lens.py, gcv_web.py, yandex.py, mock.py
  verify/match.py   re-detect + re-embed candidates, decide the match
  chain/            local.py (PoW chain), evm.py (calldata | contract)
contracts/          AnchorRegistry.sol
scripts/            deploy_contract.py
```

Each of `face`, `search` and `chain` resolves its backend through a small
registry and degrades gracefully: `auto` walks the preference order and takes the
first backend that actually imports, logging what it settled on. Swapping SFace
for ArcFace, or Sepolia for Polygon Amoy, is a flag or an env var — no other file
changes.
