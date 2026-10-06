# Environment notes

Facts about this machine and dataset that cost real time to discover.

## Hard constraints

Violating any of these breaks the build in ways that are slow to diagnose.

- **`transformers>=4.55,<5`.** jina-clip-v2 ships remote code that breaks on 5.x;
  SigLIP2 and Qwen2.5-VL both need ≥4.49. Use the **`jina_env`** conda env
  (transformers 4.57.6). `base` and `aio2026_env2` have transformers 5.x and
  will fail.
- **Run everything from the repo root** with `python -m`. Running a module by
  path breaks `core.*` imports.
- **Load `.env` first**: `set -a; . ./.env; set +a`.
- **Qdrant payloads stay nested** — `video.name`, `frame.timestamp_seconds`,
  `shot_id`, `file_path`. `fusion.py` keys on `file_path`, the shot rollup keys
  on `shot_id`, and `parsing.py` reads `video.name`. Flattening breaks all three.
- **Elasticsearch documents put their text in `content`**, not `text`.
  `ElasticsearchService.search` queries `content`, `content.ngram` and
  `content.exact`; a `text` field indexes fine and matches nothing.
- **Tests are stdlib `unittest`**, never pytest, no linter:
  `python3 -m unittest discover -s tests -v`
- **12GB VRAM.** Online holds SigLIP2 (~1.6GB) + jina-clip-v2 (~1.8GB) +
  Qwen2.5-VL-3B (~7GB) ≈ 10.4GB. Never load Whisper in the same process as
  Qwen. On OOM set `QWEN_VL_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
- **SigLIP2 text needs lowercase + padding to exactly 64 tokens.**
  `core/models/SigLIP_embedding.py` already does this; do not "simplify" it.

## Filesystem layout, and why it is split

`/media/nekoartisan/Lexar` is **exFAT**: no POSIX ownership, no symlinks, no
Lucene-compatible file locking.

| What | Where | Why |
|---|---|---|
| Videos, keyframes, artifacts | `/media/nekoartisan/Lexar/vqa-data` (exFAT) | Plain file reads; exFAT is fine |
| Elasticsearch + Qdrant data | `/mnt/data/vqa-stores` (ext4) | ES writes `node.lock`; exFAT cannot |
| `HF_HOME` | `/mnt/data/vqa-hf` (ext4) | The HF blob cache uses symlinks |
| Repo | `/` (ext4, ~17GB free) | Keep large files off it |

## Environment facts that cost time to rediscover

- **pypi.org and the npm registry are unreachable** from this network. Four
  packages were replaced rather than installed, and `requirements.txt` records
  each substitution:
  `opencv-python` → ffmpeg CLI + PyAV (the videos are HEVC, which cv2 often
  cannot decode anyway); `librosa` → faster-whisper's own PyAV decoding;
  `imagehash` → `offline/phash.py`; `qwen-vl-utils` → `AutoProcessor` takes PIL
  images directly. `express` is reused from AIC's `node_modules`.
- **The Docker registry CDN is unreachable.** `docker-compose.yml` pins
  Elasticsearch `8.17.0` because that tag is already in the local image cache.
- **`hf download` is unusable for bulk fetches.** After one timeout its client
  raises `Cannot send a request, as the client has been closed` for every
  subsequent file — it failed 9 of 10 dataset zips without transferring a byte.
  `scripts/fetch_videomme2.sh` uses `curl --retry 999 -C -` instead. For single
  model repos `hf download` works, but **`HF_HUB_DISABLE_XET=1` is required**:
  hf_xet hangs indefinitely here.
- Link speed is ~1.75 MB/s and drops intermittently. Parallel connections do
  not help; that is the whole pipe.

## Failure modes already diagnosed

Each of these cost real time to find. Don't re-derive them.

**Elasticsearch wedges on the disk watermark.** Symptom: documents never arrive,
`Lỗi upsert: Connection timed out`, and `number_of_pending_tasks` stuck above zero
with `cluster_reroute(disk threshold monitor)` and
`update_tsdb_data_stream_end_times` aged in hours. `/mnt/data` sits at 91% from
unrelated projects, above ES's default 90% high watermark, so the reroute task
wedges the cluster-state queue and every `put-mapping` behind it times out — the
call to raise the watermark included, since that is itself a cluster-state update.
Writes to an *existing* mapping keep working, which masks it: `asr_data` looked
healthy while 30,251 OCR documents landed as zero.
Fix: `docker restart vqa-elasticsearch`, then confirm `pending` is 0. Watermarks
are set to 95/97/98 persistently; the indices total a few megabytes.

**Never delete anything under `/mnt/data` to free space.** `aic25-selected-cache`
(36G), `birdnet_forest_multilabel` (16G), `hf_cache` (9.8G) and `conda_pkgs` (7.8G)
belong to the user's other projects.

**`pgrep -cf '<pattern>'` overcounts**: the pattern matches the inspecting shell's
own command line. This produced two wrong diagnoses — a watcher concluded a dead
fetcher was alive and stopped respawning it, and a measurement reported four
concurrent curls where there was one. Use `ps | grep` and read the list. Scripts
coordinate with `flock`, never with pgrep.

**Never edit a shell script while it runs.** Bash reads by byte offset, so inserted
lines make it resume mid-token. Let the current step finish, kill the wrapper, then
relaunch; the scripts are resumable.

**Two writers on one download corrupt it silently.** `from_pretrained` does not
verify safetensors checksums. `scripts/fetch_model_curl.sh` takes an flock, and
drives retries from its own loop rather than curl's: curl's internal `--retry`
truncates and rewrites from byte 0 when the CDN answers a resumed request with 200
instead of 206, which collapsed one transfer from 261MB to 0MB twice.

**Byte count is not "loadable".** A weights gate that only summed `.safetensors`
passed while `model.safetensors.index.json` was still downloading, and transformers
reported `no file named model.safetensors`. Check the shard index and that every
shard it names exists.

**Models that cannot be downloaded here, and where they already are:**

| Model | Use this |
|---|---|
| SigLIP2 so400m | `SIGLIP_MODEL=/mnt/data/aic25-clean-models/siglip` (complete, 4,544,143,072 bytes) |
| faster-whisper large-v3 | `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3` (2.9GB) |
| Qwen2.5-VL-3B | `/mnt/data/vqa-models/qwen2.5-vl-3b` (fetched with curl) |
| TransNetV2 | `/mnt/data/AIC_2025-main/model_weights/transnetv2-pytorch-weights.pth` |

`SigLIP_embedding.py` hardcoded the hub id, so every run silently dropped the
siglip vector and reported jina-only numbers. It now reads `SIGLIP_MODEL`.

**Gemini OCR:** the keys are Vertex AI Express keys bound to project 26275598820
in asia-southeast1. `gemini-2.0-flash` 404s there; `gemini-2.5-flash` answers.
`MultiLLMOCR`'s defaults are also wrong for this job — 128 images against 500
output tokens truncates the JSON mid-string and `_parse_array_of_pairs` then
returns all-None, so OCR "succeeds" with no text. Use 24 images, 16384 tokens, 30
clients. A video with genuinely no on-screen text (062, 063, 183) is not a failure.

**`--stages index_text`, not `index --force`, to push ASR/OCR text.** The latter
also re-upserts all 54,802 Qdrant vectors at roughly a video a minute — 3.3 hours
to deliver documents that take minutes. `index --force` is right only when the
vectors themselves changed.
