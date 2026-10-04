# VQA trên Video-MME-v2 — Thiết kế

Ngày: 2026-10-04

## 1. Mục tiêu

Trả lời câu hỏi trắc nghiệm về video bằng cách truy hồi bằng chứng (keyframe + lời nói +
chữ trên hình) rồi đưa cho VLM local chọn đáp án.

Câu hỏi nghiên cứu, không phải "đạt accuracy cao nhất":

> Retrieval-augmented VQA thắng hay thua uniform frame sampling, và thắng/thua ở loại
> câu hỏi nào?

Đây là lý do baseline uniform-sampling là thành phần bắt buộc, không phải tuỳ chọn.

## 2. Dữ liệu

[Video-MME-v2](https://hf.co/datasets/MME-Benchmarks/Video-MME-v2), MIT license, tiếng Anh.

| | |
|---|---|
| Video | 800 × 1080p, nguồn YouTube, >80% đăng 2025+ |
| Câu hỏi | 3.200 MCQ (4 câu/video), 3.138 câu có đúng 8 đáp án A–H |
| Baseline đoán bừa | 12,8% |
| Subtitle | 800 file JSONL, **timestamp cấp từ** |
| Dung lượng | ~105GB (40 zip) |

Phạm vi lần này: **10 zip = 200 video = 800 câu hỏi** (khoảng tin cậy 95% ±3,5%).
Lý do giới hạn: băng thông đo được chỉ 1,75 MB/s → full 105GB mất ~20 giờ tải cộng
~31 giờ index. Mở rộng lên 40 zip bằng `./fetch_videomme2.sh 40`.

### 2.0 Chia dev / test

Tách **trước khi** tinh chỉnh bất cứ thứ gì, theo `video_id` (không theo câu hỏi — 4 câu
cùng video phải nằm cùng phía, nếu không là leak):

| Tập | Video | Câu hỏi | Dùng để |
|---|---|---|---|
| dev | 50 đầu (`001`–`050`) | ~200 | Tinh chỉnh trọng số RRF, `top_k`, ngưỡng, đo giá trị OCR |
| test | 150 còn lại (`051`–`200`) | ~600 | Chỉ chạy **một lần** cho mỗi cấu hình ở §8 |

Mọi con số trong báo cáo lấy từ test. Chạm vào test nhiều lần để chọn tham số là tự
vô hiệu hoá kết quả.

Vị trí: `DATA_ROOT=/media/nekoartisan/Lexar/vqa-data` (ổ `/` chỉ còn 18GB, không chứa được).

### 2.1 Phân bố câu hỏi theo Level

| Level | Nhóm `second_head` | Số câu | Kỳ vọng với pipeline retrieval |
|---|---|---|---|
| 1 | `Frame-Only` 575, `Frames & Audio` 478 | 1.053 | Sở trường — top-k shot đủ trả lời |
| 2 | Action/Change/Order/Temporal | 1.049 | Yếu — top-k độc lập mất thứ tự thời gian |
| 3 | Plot/Physical/Social/Knowledge | 1.098 | Bất lợi — cần hiểu toàn video |

Phân bố Level 1 thành `Frame-Only` vs `Frames & Audio` là **thiết kế ablation có sẵn**:
bật/tắt nhánh ASR thì `Frames & Audio` phải tăng còn `Frame-Only` phải không đổi. Nếu
`Frame-Only` cũng tăng thì có leak hoặc nhánh ASR đang vá lỗi nhánh visual.

### 2.2 Chấm điểm

`group_type` chia 2 loại: `relevance` (2.076 câu, chấm thường) và `logic` (1.124 câu,
**first error truncation** — sai câu đầu thì các câu sau trong nhóm tính sai hết).

Báo cáo cả hai: accuracy per-question (để so sánh giữa các cấu hình của ta) và điểm nhóm
chính thức (để so leaderboard).

## 3. Nền tảng tái sử dụng

Nguồn: `/mnt/data/AIC_2025-main` — hệ thống AIC 2025, ~28.600 dòng Python đang chạy.
Dùng lại thay vì viết lại, vì nó đã chứa các fix đã trả giá để học.

### 3.1 Copy nguyên — 17 file

```
core/config.py
core/models/SigLIP_embedding.py          # fix: lowercase + pad đúng 64 token
core/models/JinaCLIP_embedding.py
core/models/Whisper_VAD.py
core/models/MultiLLM_OCR.py              # luân phiên 30 GOOGLE_API_KEY_n
core/models/transnetv2_pytorch.py
core/services/vector_store_optimized.py
core/services/elasticsearch_service.py
core/utils/parsing.py
core/utils/video_identity.py
online/backend/fusion.py                 # SRRF
online/backend/utils.py
online/backend/schemas.py
online/backend/search_kis.py             # sạch: chỉ phụ thuộc Qdrant/ES/parsing
offline/video_trans_detection.py
offline/save_images_and_gifs.py
offline/database_processing.py
```

### 3.2 Copy rồi cắt — 3 chỗ

| File | Cắt gì |
|---|---|
| `offline/ingest_data.py` | 2 import BEiT3 + YOLOE |
| `online/backend/resources.py` | init BEiT3 / BLIP2 / GPT4o / events |
| `online/frontend/` | DRES, TRAKE, bookmark, submission UI |

### 3.3 Viết mới — 5 file + 1 routes

| File | Trách nhiệm |
|---|---|
| `data/videomme.py` | Đọc `test.parquet`; gộp subtitle cấp từ → `Segment` |
| `core/models/qwen_vl.py` | Wrapper Qwen2.5-VL-3B |
| `online/backend/vqa.py` | Dựng prompt MCQ, sắp theo thời gian, parse đáp án |
| `eval/run_eval.py` | Accuracy + group score + breakdown theo Level |
| `eval/baseline.py` | Uniform sampling 10 frame |
| `online/backend/routes.py` | **Viết mới, nhỏ** — không port được |

`routes.py` gốc dính 39 ref tới `events`, 36 tới `guided`, 18 tới BEiT3, 14 tới agent.
Port nó là kéo cả hệ thống AIC sang. Thay bằng file mới chỉ có `/search/kis`,
`/search/audio`, `/search/ocr`, `/vqa/answer`, `/health`, `/files`.

### 3.4 Bỏ hẳn

BEiT3, BLIP2, OpenCLIP, YOLOE, GPT4o, `core/events/*`, `guided*`, `agent*`,
`search_{temporal,composed,events,intelligent}`, `submission_checker`, DRES,
caption/BGE-M3.

Kéo theo requirements rụng: `timm==0.6.13`, `torchscale==0.3.0`, `FlagEmbedding`,
`ultralytics`, `streamlit`.

## 4. Mô hình

| Vai trò | Model | Chạy ở | VRAM |
|---|---|---|---|
| Tách shot | TransNetV2 | local GPU | ~0,5GB |
| Visual embed 1 | `google/siglip2-so400m-patch14-384` | local GPU | ~1,6GB |
| Visual embed 2 | `jinaai/jina-clip-v2` | local GPU | ~1,8GB |
| ASR | faster-whisper large-v3 + Silero VAD | local GPU | ~3GB |
| OCR | Gemini Flash (30 key luân phiên) | API | 0 |
| Mở rộng query | DeepSeek | API | 0 |
| Trả lời | `Qwen/Qwen2.5-VL-3B-Instruct` | local GPU | ~7GB |

Offline và online không chạy cùng lúc, nên VRAM không xung đột:

- Offline: Whisper (~3GB) → rồi SigLIP2+Jina (~3,4GB)
- Online: SigLIP2+Jina+Qwen = **~10,4GB / 12GB**

Dự phòng OOM: đổi sang `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (~4GB), một dòng config.

### 4.1 Ràng buộc version

`transformers>=4.55,<5`. Giới hạn trên vì remote code của jina-clip-v2 vỡ trên 5.x;
giới hạn dưới vì SigLIP2. Qwen2.5-VL cần ≥4.49 nên **nằm trong khoảng này** — cả ba
model dùng chung một version, không xung đột.

## 5. Pipeline offline

Nguyên tắc: mỗi stage chạy qua *toàn bộ* video (không phải mỗi video chạy hết stage), để
model load **một lần mỗi stage**. Trạng thái trong SQLite `manifest.db` khoá
`(video_id, stage)` nên crash ở video thứ 180 không mất 179 cái trước.

| Stage | Vào | Ra | Chạy trên |
|---|---|---|---|
| `shots` | mp4 | `shots/{vid}.json` | GPU TransNetV2 |
| `keyframes` | mp4 + shots | `keyframes/{vid}/{shot:04d}_{k}.jpg` | CPU ffmpeg |
| `embed` | keyframes | `embeds/{vid}.npz` | GPU SigLIP2+Jina |
| `asr` | mp4 *hoặc* subtitle GT | `asr/{vid}.json` | GPU Whisper / CPU parse |
| `ocr` | keyframes | `ocr/{vid}.json` | Gemini API |
| `index` | tất cả trên | Qdrant + ES | — |

**Mọi artifact ghi ra file trước, `index` mới đẩy vào store.** Đổi trọng số RRF hay mapping
ES thì chạy lại `index` trong vài phút, không chạy lại model nào. Với một dự án sẽ tinh
chỉnh fusion hàng chục lần, đây là khác biệt giữa vài phút và vài giờ mỗi vòng lặp.

GIF **không** sinh trước — `online/gif.py` sinh bằng ffmpeg khi frontend xin rồi cache.
Tiết kiệm ~96GB ở quy mô full và cắt hẳn một stage.

### 5.1 Chọn keyframe

Mỗi shot lấy 3 frame ở 10%/50%/90% thời lượng. Shot <1s → 1 frame. Shot >10s → cứ 3s một
frame, tối đa 5. Dedup trong shot bằng perceptual hash.

### 5.2 Subtitle cấp từ → segment

Subtitle gốc mỗi dòng một từ:

```json
{"text": "Hi,", "start_time": 115.6, "end_time": 116.0}
```

BM25 trên từ đơn là vô nghĩa. Gộp thành segment: ngắt khi lặng >0,6s **hoặc** đủ 15s.
Kết quả có **đúng schema mà Whisper sinh ra**:

```python
Segment(text: str, start: float, end: float)
```

Vì cùng schema, `asr_source` trong config đổi qua lại được:

| `asr_source` | Ý nghĩa |
|---|---|
| `subtitle` | Giới hạn trên — transcript hoàn hảo |
| `whisper` | Thực tế |
| Khoảng cách | Chính xác phần accuracy mất do lỗi ASR |

Phụ phẩm: đo WER của nhánh Whisper miễn phí bằng cách so với GT.

## 6. Schema store

### 6.1 Qdrant — point là **keyframe**

Collection `videomme_keyframes`, named vectors khớp code AIC có sẵn:

| Vector | Chiều | Metric |
|---|---|---|
| `siglip` | 1152 | cosine |
| `jina` | 1024 | cosine |

Point id: UUID5 tất định từ `f"{video_name}:{shot_number}:{frame_index}"` — chạy lại
`index` không tạo bản trùng.

Payload **giữ đúng schema lồng nhau của repo AIC**, không phẳng hoá. Đây là ràng buộc
cứng: `fusion.py`, `parsing.py` và frontend đều đọc theo các khoá này.

```python
{
  "file_path": str,                  # khoá fuse của rrf_weighted_fuse
  "filename": str,
  "source_type": "video_keyframe",
  "shot_id": f"{video_name}_shot_{shot_number:03d}",   # khoá group theo shot
  "video": {"name": "001", "filename": "001.mp4"},
  "shot":  {"number": int, "position": int, "start": float, "end": float},
  "frame": {"index": int, "timestamp_seconds": float, "timestamp_formatted": str},
}
```

`video.name` là `"001"`.. `"800"` (tên file Video-MME-v2), không phải định dạng AIC.

`setup_collection()` đã tự tạo payload index cho `frame.timestamp_seconds` và `video.name`
— không cần thêm gì cho chế độ in-video.

### 6.2a Ba chỗ phải sửa khi port, nếu không sẽ vỡ ngầm

Phát hiện khi đọc code nguồn, không phải suy đoán:

1. **`parsing.py` có `_VIDEO_NAME_RE = r"^[A-Z]\d+_V\d+$"`** — chỉ nhận tên kiểu
   `L01_V001`. Với `"001"` thì `metadata_needs_repair()` **luôn trả True**. Phải mở rộng
   regex để nhận thêm `^\d{3}$`.
2. **Không port `enrich_results_metadata`** — nó chỉ tồn tại để hydrate collection
   caption, mà ta đã bỏ. Giữ lại là tự gọi `repair_result_metadata` lên mọi kết quả.
3. **`rrf_weighted_fuse` nhận object có `.score` và `.payload`** (ScoredPoint của Qdrant),
   nhưng `ElasticsearchService.search` trả **dict**. Cần adapter bọc kết quả ES trước khi
   fuse — xem §6.3.

### 6.2 Elasticsearch — doc là **segment** / **keyframe**

`asr_data`: `video_id` keyword, `text` text(english), `start` float, `end` float,
`source` keyword (`subtitle`|`whisper`).

`ocr_data`: `video_id` keyword, `shot_idx` integer, `timestamp` float,
`text` text(english), `file_path` keyword.

Dùng `english` analyzer (có stemming + stopword) vì dataset là tiếng Anh — khác AIC gốc
vốn xử lý tiếng Việt.

### 6.3 Fusion: keyframe trước, rồi gộp lên shot

`rrf_weighted_fuse` có sẵn khoá theo `payload["file_path"]`, tức fuse ở cấp **keyframe**.
Giữ nguyên hàm đó (đã được dùng thật trong AIC) rồi **gộp lên shot ở bước sau** bằng
`payload["shot_id"]`, lấy điểm cao nhất mỗi shot. Hai bước, không viết lại fusion.

ES trả dict nên cần adapter cho đồng dạng với ScoredPoint:

```python
@dataclass
class _FusableHit:
    score: float
    payload: dict
    id: str | None = None
```

Mỗi hit ASR được map sang các keyframe có `frame.timestamp_seconds` nằm trong
`[start, end]` của segment; mỗi hit OCR map trực tiếp sang keyframe của nó. Nhờ vậy cả 4
nguồn vào `rrf_weighted_fuse` cùng một dạng.

## 7. Luồng online

```
câu hỏi + 8 đáp án
   │
   ├─ DeepSeek mở rộng query (tuỳ chọn, bật/tắt bằng config)
   │
   ├─ 4 nguồn song song, mỗi nguồn trả ranked list shot
   │    siglip : Qdrant vector=siglip  → keyframe → gộp theo shot (max)
   │    jina   : Qdrant vector=jina    → keyframe → gộp theo shot (max)
   │    asr    : ES BM25 asr_data      → segment  → shot chồng thời gian
   │    ocr    : ES BM25 ocr_data      → keyframe → shot
   │
   ├─ RRF: score(shot) = Σ_s w_s / (rrf_k + rank_s),  rrf_k=60
   │    trọng số mặc định: siglip 1.0, jina 1.0, asr 0.8, ocr 0.5
   │    mỗi nguồn lấy candidate_depth=50 shot trước khi fuse
   │
   ├─ top_k=10 shot → **sắp theo timestamp tăng dần, KHÔNG theo điểm**
   │
   └─ Qwen2.5-VL-3B → chữ cái A–H
```

Ba tham số cần phân biệt, dễ lẫn nhau:

| Tham số | Mặc định | Nghĩa |
|---|---|---|
| `candidate_depth` | 50 | Mỗi nguồn trả bao nhiêu shot vào RRF |
| `rrf_k` | 60 | Hằng số làm mượt trong công thức RRF |
| `top_k` | 10 | Bao nhiêu shot thực sự vào prompt của VLM |

Cả ba tinh chỉnh trên dev set (§2.0).

Không có reranker riêng. Quyết định có chủ đích: đo RRF thuần trước đã, rồi mới biết
rerank có đáng không.

Trọng số OCR để thấp 0,5 vì 33 `third_head` của Video-MME-v2 **không có task nào về đọc
chữ trên hình**. Xem §9.

### 7.1 Sắp theo thời gian, không theo điểm

Sau RRF, top-k được sắp **tăng dần theo timestamp**. Chi phí gần bằng 0 nhưng vá được
phần lớn điểm yếu ở Level 2 (Order, Temporal Reasoning): VLM nhìn thấy các bằng chứng
đúng trình tự chúng xảy ra. Mốc thời gian ghi thẳng vào prompt.

### 7.2 Lớp ngữ cảnh toàn video

Ngoài top-k shot, prompt luôn kèm 8 frame cách đều toàn video + transcript rút gọn. Mục
đích: cứu các câu Level 3 cần hiểu mạch truyện, thứ mà top-k rời rạc không bao giờ đủ.

### 7.3 Prompt MCQ và parse đáp án

Thứ tự prompt: lớp ngữ cảnh toàn video → các shot top-k theo thời gian (ảnh + `[mm:ss]`
ASR + OCR) → câu hỏi → 8 đáp án → chỉ thị *"Answer with the single letter only."*

Parse 3 tầng:

1. Regex chữ cái A–H đứng độc lập
2. Nếu thất bại: so log-prob của các token chữ cái
3. Nếu vẫn thất bại: trả `A` và **đánh dấu `unparsed`**

Tỉ lệ `unparsed` phải được báo cáo. Nếu nó cao thì accuracy không có nghĩa.

Ngân sách ảnh: tối đa 12 ảnh retrieved + 8 global, cạnh dài ≤448px, JPEG q70. Lấy từ bài
học đã đo trong repo AIC (`2b45157 perf(vision): downscale keyframes to half + JPEG q70`).

## 8. Đánh giá

`eval/run_eval.py --config <name> --limit N`

Chỉ tính các câu thuộc video đã tải. **Luôn báo N** — 800/3.200 câu thì phải nói rõ.

Chỉ số: accuracy tổng; theo Level 1/2/3; theo `second_head`; điểm nhóm có first-error
truncation; tỉ lệ `unparsed`.

Các cấu hình so sánh:

| Tên | Nội dung |
|---|---|
| `baseline-uniform` | 10 frame cách đều, không Qdrant/ES |
| `visual-only` | siglip + jina |
| `+asr-gt` | thêm ASR từ subtitle GT |
| `+asr-whisper` | thêm ASR từ Whisper |
| `+ocr` | thêm OCR |
| `full` | tất cả + mở rộng query |

Ra `results/eval/<name>.json` kèm bảng markdown.

## 9. Rủi ro đã biết

**OCR có thể vô dụng trên dataset này.** 33 `third_head` không có task đọc chữ. Video-MME-v2
là YouTube đa dạng, không phải news/TV. Nên: chạy OCR trên dev set ~50 video trước, đo
contribution. Nếu +<1% thì giữ code nhưng tắt mặc định và viết vào báo cáo như *negative
result có số liệu*. Không chạy OCR toàn corpus trước khi đo.

**Level 2/3 có thể thua baseline.** Đây là kết quả hợp lệ, không phải lỗi — miễn là
breakdown theo Level được báo cáo (§8). Nó trả lời đúng câu hỏi nghiên cứu ở §1.

**Băng thông 1,75 MB/s và không ổn định.** Downloader dùng curl `--retry 999 -C -` nên tự
nối lại sau khi mạng sập. `hf download` **không** dùng được: sau timeout đầu tiên nó raise
`Cannot send a request, as the client has been closed` cho mọi file sau đó — 9/10 zip fail
mà không truyền một byte.

**VRAM 10,4/12GB khi online.** Prefill nhiều ảnh có thể OOM. Dự phòng: Qwen AWQ 4-bit.

## 10. Kiểm thử

Theo quy ước repo AIC: `unittest` thuần, không pytest, không linter.

```bash
python3 -m unittest discover -s tests -v
```

| Test | Nội dung |
|---|---|
| `test_videomme_loader.py` | Gộp từ → segment: 1 từ, lặng dài, chạm trần 15s, file rỗng |
| `test_rrf.py` | RRF với rank biết trước; trọng số 0 phải loại hẳn một nguồn |
| `test_mcq_parse.py` | Parse chữ cái từ output bẩn: `"B."`, `"The answer is C"`, rỗng |
| `test_integration.py` | Tự skip khi Qdrant/ES/backend chưa chạy |

## 11. Vận hành

```bash
docker compose up -d qdrant elasticsearch     # chỉ 2 service
export HF_HOME=/media/nekoartisan/Lexar/vqa-data/hf_home
python -m offline.run --stages all            # từ repo root
python -m online.backend.api_server           # :8000
cd online/frontend && npm start               # :3000
```

Phải chạy từ repo root — chạy theo path làm vỡ import `core.*`.

API key copy từ `/mnt/data/AIC_2025-main/.env`: `GEMINI_API_KEY`,
`GOOGLE_API_KEY_1..30`, `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`.
`.env` nằm trong `.gitignore`.

## 12. Thứ tự triển khai

Phạm vi này quá lớn cho một lần làm liền mạch, nên chia 5 giai đoạn. Mỗi giai đoạn kết
thúc bằng một thứ **chạy được và kiểm chứng được**, không phải một nửa tính năng.

| GĐ | Nội dung | Kiểm chứng xong khi |
|---|---|---|
| 1 | Scaffold: copy 17 file, cắt 3 chỗ, `requirements.txt`, `docker-compose.yml`, `.env` | `docker compose up -d` xanh; `import core.*` không lỗi; unittest chạy |
| 2 | `data/videomme.py` + offline stages + `manifest.db` | 20 video có sẵn index xong vào Qdrant + ES; `/health` đếm đúng số point |
| 3 | `routes.py` mới + RRF retrieval, **chưa có VLM** | `/search/kis` trả shot đúng cho query thủ công |
| 4 | `core/models/qwen_vl.py` + `online/backend/vqa.py` | `/vqa/answer` trả chữ cái A–H cho 1 câu hỏi thật |
| 5 | `eval/baseline.py` + `eval/run_eval.py` | Bảng accuracy theo Level cho `baseline-uniform` và `visual-only` |

Giai đoạn 1–2 chạy được ngay với 20 video đã tải (zip 001). Không cần chờ download xong.

Frontend (`online/frontend/`) làm sau giai đoạn 5 — nó không cần thiết để có số liệu, và
mục tiêu ở §1 là số liệu.
