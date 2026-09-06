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
      ▼  1. detect + embed (YuNet → ArcFace R50, 512-d)
   face box + 60% padding ──────────────────────────────────┐
      │         (no face? fall back to the whole image)     │
      ▼  2. reverse image search                            │
         (SerpAPI Google Lens / Cloud Vision / Yandex)      │
   N candidate pages ─── keep social platforms only         │
      │                                                     │
      ▼  3. confirm: download each candidate image,         │
         re-detect, re-embed, cosine vs. probe ◄────────────┘
         (no face? perceptual-hash the whole image instead)
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

### What gets sent to the search provider

Stage 2 sends the **face box grown by 60%**, clamped to the image so the padding
is always more of the actual photograph and never black bars. Measured on one
probe, counting pages that land on a social platform:

| search input | Cloud Vision | Google Lens |
| --- | --- | --- |
| whole image | 12 social | 10 social |
| face box + 25% | 8 social | — |
| **face box + 60%** | **15 social** | **13 social** |
| face box + 100% | 14 social | — |

The padded crop beats the full frame on both providers. Enough context survives
for the engines to match, while the face dominates the frame rather than
competing with background, other people and captions.

> An earlier version of this README claimed a cropped search returned **0**
> results from Lens. That measurement was wrong — it did not reproduce. Repeated
> across two providers and several padding levels, crops match at least as well
> as the full frame. `--search-image source` still sends the whole image if you
> want to compare.

When **no face is detected**, stage 2 sends the whole image and stage 3 switches
to perceptual-hash agreement (`method: "image-phash"` in the record), so object
photos work too. `--require-face` turns that fallback off.

Only a 32-byte hash goes on-chain. The face embedding never leaves the machine
and is excluded from the record ([`evidence.py`](src/hhg3/evidence.py)), which
matters because a public chain is permanent and biometric data is not something
to publish.

## Install

```bash
git clone https://github.com/ZaiD-lgtm/HH_GOA_3.git
cd HH_GOA_3
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e .
```

That is enough for a real run: detection is **YuNet** and recognition falls back
to **SFace** (128-d), both bundled with OpenCV, weights auto-cached in `models/`.

For the better embedder — recommended:

```bash
pip install -r requirements-face.txt   # onnxruntime -> ArcFace R50, 512-d
pip install -r requirements-evm.txt    # web3 + py-solc-x
```

With `onnxruntime` present the pipeline picks **ArcFace R50 (512-d)**
automatically and downloads its weights once (275 MB, from the official
insightface release). Without it, SFace is used and everything still works.
`hhg3 doctor` shows which backends actually resolved.

Why it is worth the download — worst impostor score across three different
people, against a 0.30 threshold:

| embedder | worst impostor | headroom |
| --- | --- | --- |
| SFace 128-d | 0.2123 | 0.09 |
| **ArcFace 512-d** | **0.0928** | **0.21** |

Genuine matches sit at 0.85–0.91 either way, so the gain is entirely in not
mistaking a stranger for your probe.

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

Useful flags: `--threshold 0.45` (stricter match), `--any-domain` (do not restrict
to social platforms), `--search-crop` (search the face crop instead of the source
image), `--provider mock --allow-mock` (offline plumbing test),
`--embedder sface|insightface|fallback`, `--no-image`, `--json`.

A finished run prints the probe and the matched post side by side, drawn in the
terminal with half-block characters, above a panel carrying the post URL, the
similarity, the record hash and the chain receipt — so a screen recording ends on
a frame that shows both the face that was found and the anchor that proves it.

Each preview is cropped to the face box the matcher actually scored. A candidate
is usually a whole post — a video thumbnail, a group shot — in which the face is
a small corner of the frame, so the crop is what makes it recognisable on camera;
`--preview-full` shows the uncropped post instead. Previews size themselves to
the terminal (bounded by its height, since a square crop that is wide is also
tall), and `--preview-width N` overrides that. The pictures need a colour-capable
TTY; piped or redirected output, `NO_COLOR`, and `--no-image` all fall back to
the panel alone, which carries the same facts.

The match threshold defaults to **0.30**. Each embedder also declares its own
same-identity threshold — 0.363 for SFace (OpenCV's published figure), 0.45 for
ArcFace — because a cosine score is only meaningful relative to the model that
produced it; those apply when `match_threshold` is set to `None` in
[config.py](src/hhg3/config.py). `--threshold` overrides either.

0.30 sits deliberately below SFace's published figure: it admits more candidates
so a demo is less likely to end with no match at all. Every accepted match records
the score and the threshold it cleared, so a marginal one is visible rather than
hidden. Raise it to 0.363+ if you care more about precision than recall.

Each run writes `runs/<id>/`:

| file | contents |
| --- | --- |
| `probe_crop.jpg` | the detected face, padded 25% — kept as evidence |
| `search_crop.jpg` | what was actually sent to the search provider |
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
  models.py         lazy download/cache for the ONNX weights
  imagehash.py      DCT perceptual hash, used when there is no face
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
