# hhg3: face scan → social post → blockchain anchor

HH Goa 2026 shortlisting, Task 3. A command-line pipeline: it takes a face scan,
finds a real matching social-media post on the open web, writes a tamper-evident
record of that finding to a blockchain, then re-verifies the record against the
chain.
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

The search provider never gets to decide identity. It only proposes URLs. Stage
3 downloads whatever image the candidate page actually serves, runs detection
and embedding on it again, and scores it against the probe. The "match" is
therefore a face-recognition decision made on this machine, carrying the
similarity score and the threshold it cleared; the search engine's own ranking
never settles it, and no result is hardcoded.

### What gets sent to the search provider

Stage 2 sends the face box grown by 60%, clamped to the image so the padding is
always more of the actual photograph and never black bars. Measured on one
probe, counting pages that land on a social platform:

| search input | Cloud Vision | Google Lens |
| --- | --- | --- |
| whole image | 12 social | 10 social |
| face box + 25% | 8 social | not measured |
| face box + 60% | 15 social | 13 social |
| face box + 100% | 14 social | not measured |

The padded crop beats the full frame on both providers. Enough context survives
for the engines to match, and the face still dominates the frame instead of
competing with background, other people and captions.

When no face is detected, stage 2 sends the whole image and stage 3 switches to
perceptual-hash agreement (`method: "image-phash"` in the record), so object
photos work too. `--require-face` turns that fallback off.

Only a 32-byte hash goes on-chain. The face embedding never leaves the machine
and is excluded from the record ([`evidence.py`](src/hhg3/evidence.py)). A
public chain is permanent, and biometric data does not belong on one.

## Install

```bash
git clone https://github.com/ZaiD-lgtm/HH_GOA_3.git
cd HH_GOA_3
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e .
```

That is enough for a real run: detection is YuNet and recognition falls back to
SFace (128-d), both bundled with OpenCV, weights auto-cached in `models/`.

For the better embedder (recommended):

```bash
pip install -r requirements-face.txt   # onnxruntime -> ArcFace R50, 512-d
pip install -r requirements-evm.txt    # web3 + py-solc-x
```

With `onnxruntime` present the pipeline picks ArcFace R50 (512-d) automatically
and downloads its weights once (275 MB, from the official insightface release).
Without it, SFace is used and everything still works. `hhg3 doctor` shows which
backends actually resolved.

Why it is worth the download. Worst impostor score across three different
people, against a 0.30 threshold:

| embedder | worst impostor | headroom |
| --- | --- | --- |
| SFace 128-d | 0.2123 | 0.09 |
| ArcFace 512-d | 0.0928 | 0.21 |

Genuine matches sit at 0.85 to 0.91 either way, so the whole gain is in not
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
similarity, the record hash and the chain receipt. A screen recording therefore
ends on one frame holding both the face that was found and the anchor that
proves it.

Each preview is cropped to the face box the matcher actually scored. A candidate
is usually a whole post (a video thumbnail, a group shot) in which the face is a
small corner of the frame, so the crop is what makes it recognisable on camera.
`--preview-full` shows the uncropped post instead. Previews size themselves to
the terminal (bounded by its height, since a square crop that is wide is also
tall), and `--preview-width N` overrides that. The pictures need a colour-capable
TTY; piped or redirected output, `NO_COLOR`, and `--no-image` all fall back to
the panel alone, which carries the same facts.

The match threshold defaults to 0.30. Each embedder also declares its own
same-identity threshold: 0.363 for SFace (OpenCV's published figure), 0.45 for
ArcFace. A cosine score is only meaningful relative to the model that produced
it. Those per-model values apply when `match_threshold` is set to `None` in
[config.py](src/hhg3/config.py), and `--threshold` overrides either.

0.30 sits below SFace's published figure on purpose: it admits more candidates,
so a demo is less likely to end with no match at all. Every accepted match
records the score and the threshold it cleared, so a marginal one stays visible
in the bundle. Raise it to 0.363+ if you care more about precision than recall.

Each run writes `runs/<id>/`:

| file | contents |
| --- | --- |
| `probe_crop.jpg` | the detected face, padded 25%, kept as evidence |
| `search_crop.jpg` | what was actually sent to the search provider |
| `candidates/cand_NN.jpg` | every candidate image that was downloaded and scored |
| `record.json` | the canonical record: exactly what gets hashed |
| `bundle.json` | record + record hash + chain receipt + full search trace |

## Which blockchain

Two backends, selected with `--chain`:

- `local` (default): a hash-linked proof-of-work chain in
  `runs/_localchain.json`. Fully offline, used by the tests, and `validate()`
  detects any edit to an earlier block.
- `evm`: any JSON-RPC chain; the defaults target Ethereum Sepolia. Two modes:
  - `EVM_MODE=calldata` (default): a 0-value self-transaction carrying
    `"HHG3" || record_hash || metadata` as input data. No contract, no deploy.
  - `EVM_MODE=contract`: [`AnchorRegistry.sol`](contracts/AnchorRegistry.sol),
    a `bytes32 → {submitter, timestamp, uri}` mapping with an `Anchored` event
    and first-write-wins semantics. Deploy with
    `python scripts/deploy_contract.py`, then set `CONTRACT_ADDRESS`.

Re-verification is the same in both cases: recompute `sha256(canonical_json(record))`
from the local bundle, read the record back off the chain, compare.

## Tests

```bash
pytest
```
Covers canonical hashing, local-chain integrity, and the case the whole claim
rests on: editing `record.json` after anchoring makes `hhg3 verify` fail.

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
registry and degrades quietly: `auto` walks the preference order and takes the
first backend that actually imports, then logs what it settled on. Swapping
SFace for ArcFace, or Sepolia for Polygon Amoy, is one flag or one env var, with
no other file changes.
