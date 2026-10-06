# Retrieval-Augmented Video Question Answering on Video-MME-v2

## Abstract

Long-form video question answering requires a system to identify sparse evidence in a long temporal sequence, combine visual and textual modalities, and reason over events that may be separated by substantial time. Processing an entire video directly with a vision-language model is often impractical under limited GPU memory, while uniform frame sampling can omit short but decisive events. This report presents a retrieval-augmented video question answering pipeline evaluated on a 200-video subset of Video-MME-v2. The system decomposes videos into shots, extracts and deduplicates keyframes, indexes visual embeddings from SigLIP2 and jina-clip-v2 in Qdrant, and indexes speech and on-screen text in Elasticsearch. At query time, each retrieval channel is deduplicated to one representative hit per shot before weighted Reciprocal Rank Fusion; fused entries are then aggregated by shot and expanded into temporally ordered windows containing frames from before, during, and after each retrieved moment. These local windows are combined with uniformly sampled global frames and a condensed transcript before a local Qwen2.5-VL-3B model answers the multiple-choice question. The evaluation is designed as an ablation study against a uniform-sampling baseline and reports conventional accuracy, the official Video-MME-v2 Non-Linear group score, category-level breakdowns, output parsing reliability, and paired statistical tests. On the 600-question test split, the strongest configuration, `+ocr`, obtains 22.67% accuracy and a 10.08% Non-Linear score, compared with 20.83% and 9.42% for uniform sampling. The 1.83-point accuracy gain is directionally positive but is not statistically significant under an exact McNemar test (p = 0.193). These results indicate that retrieval provides a modest benefit under the current local-model and evidence-budget constraints, while also exposing a substantial gap between development and test performance.

**Keywords:** video question answering, long-video understanding, multimodal retrieval, retrieval-augmented generation, temporal reasoning, Video-MME-v2

## 1. Introduction

Video Question Answering (VideoQA) extends visual question answering from static images to temporally ordered, multimodal observations. A successful system must not only recognize objects and scenes but also identify actions, track changes, interpret speech and on-screen text, and reason about relationships between events. These requirements become more difficult as video duration grows: only a small fraction of the frames may be relevant to a question, yet aggressively reducing the video to a fixed set of uniformly sampled frames can discard the critical evidence.

Large multimodal models can answer questions from multiple images or video inputs, but their cost grows with the number and resolution of visual tokens. This project targets a constrained local environment with approximately 12 GB of GPU memory. Instead of asking the answer model to process the full video, it performs computationally expensive feature extraction once, stores the resulting representations, and retrieves a compact evidence set for every question.

The central research question is:

> Does retrieval-augmented video question answering outperform uniform frame sampling, and for which categories of questions is retrieval most effective?

The project makes the following engineering contributions:

1. It implements a resumable offline pipeline for shot detection, keyframe extraction, visual embedding, speech processing, OCR, and multimodal indexing.
2. It combines two visual and two textual retrieval channels after per-source shot deduplication.
3. It converts isolated retrieval hits into bounded temporal windows so that the answer model receives evidence of change and ordering rather than a single representative image.
4. It provides a controlled ablation harness aligned with the hierarchical and group-based evaluation protocol of Video-MME-v2.

## 2. Problem Formulation and Research Objectives

Let a video be denoted by \(V\), a question by \(q\), and its candidate answers by

\[
\mathcal{A} = \{a_1, a_2, \ldots, a_n\}, \qquad 1 \leq n \leq 8.
\]

The task is to predict the label of the correct answer using evidence drawn from the video, its speech transcript, and visible text. Rather than passing all of \(V\) to the answer model, the system first retrieves a compact evidence set

\[
E = R(V, q),
\]

where \(R\) is a multimodal retrieval pipeline. A vision-language model then predicts

\[
\hat{a} = G(q, \mathcal{A}, E, C),
\]

where \(C\) is a global context layer sampled across the full video.

The evaluation addresses four subordinate questions:

1. Does visual retrieval improve over uniform frame sampling?
2. How much do ground-truth subtitles, automatic speech recognition, and OCR contribute?
3. Does query expansion improve retrieval sufficiently to justify an external API call?
4. How does each configuration behave across the three Video-MME-v2 reasoning levels and finer question categories?

The design treats the uniform-frame configuration as the mandatory reference point. Absolute accuracy alone cannot establish whether retrieval is useful because every configuration answers the same questions and shares the same answer model.

## 3. Related Work

Video-MME introduced a broad benchmark spanning short, medium, and long videos with visual, subtitle, and audio inputs [1]. Video-MME-v2 makes the evaluation more demanding through a progressive three-level hierarchy and a Non-Linear group score that rewards consistency and coherent multi-step reasoning rather than isolated correct guesses [2]. LongVideoBench similarly frames long-video understanding as retrieving and reasoning over details in interleaved video-language context [3].

Long-video systems generally reduce the visual sequence before generation. MovieChat compresses dense video tokens into sparse short- and long-term memory [4], whereas retrieval-based systems explicitly select relevant content. VRAG retrieves and refines relevant video segments before answering [5], and VideoRAG combines multimodal retrieval with graph-based textual grounding for extremely long videos [6]. The present project follows the retrieval-based direction but uses independently inspectable visual, speech, and OCR indexes, followed by rank fusion and bounded temporal-window construction.

The visual retrieval layer uses SigLIP2 [7] and jina-clip-v2 [8], which align images and text in shared embedding spaces. The answer stage uses Qwen2.5-VL [9], selected in its 3B form to fit the available hardware. Weighted Reciprocal Rank Fusion follows the rank-based aggregation principle of Cormack et al. [10], avoiding direct comparison between cosine similarity and Elasticsearch relevance scores.

## 4. Methodology

### 4.1 Dataset

Video-MME-v2 is an English multiple-choice video understanding benchmark designed around a progressive hierarchy of reasoning difficulty [2]. The full release contains 800 videos and 3,200 questions. The current project indexes 200 videos, corresponding to 10 of the 40 distributed video archives and approximately 20 GB of 1080p HEVC video. Each indexed video has four questions, producing 800 questions in total. Approximately 97% of the questions have eight answer choices, giving an empirical random-choice baseline of approximately 12.8%.

The benchmark hierarchy contains three levels:

- **Level 1 — Retrieval and aggregation:** questions require locating and combining directly observable evidence.
- **Level 2 — Temporal and dynamic understanding:** questions emphasize order, motion, change, and relations between events.
- **Level 3 — Complex multimodal reasoning:** questions require broader plot, intent, causal, or cross-modal reasoning.

The local evaluation split is defined by video identifier rather than by question:

| Split | Video IDs | Videos | Questions | Purpose |
|---|---:|---:|---:|---|
| Development | `001`–`050` | 50 | 200 | Debugging, ablation design, and hyperparameter selection |
| Test | `051`–`200` | 150 | 600 | Final evaluation after the pipeline is frozen |

Splitting by video is necessary because the four questions associated with one video share visual and textual evidence. A question-level random split would place evidence from the same video in both development and test sets and would therefore overstate generalization.

The annotation package contains word-level subtitles. The loader groups adjacent words into sentence-like segments, starting a new segment after a silence longer than 0.6 seconds or when a segment exceeds 15 seconds. The resulting structure is intentionally compatible with Whisper output so that ground-truth and automatically transcribed speech can be compared through the same retrieval interface.

### 4.2 System Architecture

The implementation is divided into an offline indexing pipeline and an online retrieval-and-answering pipeline.

```text
Offline
video
  ├── TransNetV2 ───────────────► shots
  ├── PyAV + perceptual hash ───► keyframes
  ├── SigLIP2 + jina-clip-v2 ──► Qdrant
  ├── subtitles / Whisper ──────► Elasticsearch (ASR)
  └── Gemini OCR ───────────────► Elasticsearch (OCR)

Online
question
  ├── visual retrieval ─────────► candidate shots
  ├── ASR retrieval ────────────► candidate shots
  ├── OCR retrieval ────────────► candidate shots
  └── weighted RRF ─────────────► temporal windows
                                         │
uniform global frames + transcript ──────┤
                                         ▼
                                  Qwen2.5-VL-3B
                                         │
                                         ▼
                                     label A–H
```

All offline stages write intermediate artifacts before indexing them. A SQLite manifest records the state of each `(video, stage)` pair, allowing long GPU jobs to resume without reprocessing completed videos. Large media artifacts are stored separately from the repository, while Qdrant and Elasticsearch use an ext4 volume to satisfy database locking and ownership requirements.

### 4.3 Shot Detection and Keyframe Extraction

TransNetV2 [11] detects shot boundaries from reduced-resolution video frames. Inference uses overlapping 100-frame windows with 25 frames of context on each side; only the central 50 predictions are retained. Consecutive frames above the transition threshold are treated as one transition, preventing a dissolve from being interpreted as multiple cuts.

Keyframe sampling depends on shot duration:

- shots shorter than one second contribute one middle frame;
- ordinary shots contribute frames near 10%, 50%, and 90% of their duration;
- shots longer than ten seconds are sampled approximately every three seconds, subject to a maximum of 40 frames per shot.

Frames are resized to a maximum side length of 448 pixels. A 64-bit perceptual hash removes near-duplicate frames within each shot using a Hamming-distance threshold of four. The process preserves broader coverage for unusually long static shots while limiting redundant embedding and visual-token costs.

### 4.4 Visual Representation and Vector Indexing

Each retained keyframe is encoded by two complementary vision-language encoders:

- SigLIP2 produces a 1,152-dimensional vector;
- jina-clip-v2 produces a 1,024-dimensional vector.

Embedding is performed in chunks of 64 frames so that peak GPU memory does not scale with video length. Vectors are written to compressed NumPy artifacts before being upserted into a Qdrant collection with named vector fields. Each point receives a deterministic UUID and a nested payload containing the video ID, shot number and bounds, frame index, timestamp, and relative file path.

The nested schema is operationally important. Retrieval filters on `video.name`, temporal processing reads `frame.timestamp_seconds`, and shot-level aggregation groups points using `shot_id`.

### 4.5 Speech and OCR Processing

The speech branch supports two sources:

- **Ground-truth subtitles**, used to estimate an upper bound for speech-assisted retrieval;
- **faster-whisper transcripts**, based on Whisper [12] and used to estimate performance in a deployable setting.

The two sources are stored in separate artifact directories and Elasticsearch indexes, preventing one source from overwriting the other. Speech documents contain their video ID, text content, source, and temporal interval.

The OCR branch sends keyframes to Gemini 2.5 Flash and stores non-empty on-screen text with its video, shot, keyframe path, and timestamp. During answer construction, OCR text is grouped and deduplicated at shot level rather than being read only from the single representative keyframe. This retains text that appears briefly elsewhere within a retrieved shot.

Elasticsearch indexes textual evidence in the `content` field. Search combines phrase matching, conjunctive and disjunctive term matching, fuzzy matching, and an n-gram or wildcard fallback. Video IDs are stored as keyword fields so that evaluation queries can be restricted to the question's source video.

### 4.6 Multimodal Retrieval and Rank Fusion

For a question \(q\), the system queries each enabled source independently. SigLIP2 and jina-clip-v2 encode the question and search their corresponding named vectors in Qdrant. Elasticsearch retrieves ASR and OCR documents. An OCR hit maps directly to its keyframe; an ASR hit maps to the keyframes whose timestamps fall inside the segment interval.

Before fusion, each source is collapsed to one entry per shot by retaining that source's highest-scoring keyframe. This step prevents long shots from receiving more opportunities to rank merely because they contain more indexed frames. In the earlier keyframe-level formulation, retrieved shots averaged more keyframes than the corpus mean, indicating a systematic length bias.

The remaining source-specific rankings are combined using weighted Reciprocal Rank Fusion:

\[
\operatorname{RRF}(d)
= \sum_{m \in \mathcal{M}}
\frac{w_m}{k + \operatorname{rank}_m(d)},
\]

where \(d\) is the representative indexed entry, \(\mathcal{M}\) is the set of enabled retrieval sources, \(w_m\) is a source weight, and \(k\) is the RRF smoothing constant. The fused entries are subsequently aggregated by `shot_id`. The default source weights are 1.0 for each visual encoder, 0.8 for ASR, and 0.5 for OCR. A source with zero weight is not queried.

### 4.7 Temporal Evidence Windows

A single representative frame is insufficient for questions involving change, order, or reaction. The current implementation therefore converts high-ranked shots into temporal windows.

For each selected center shot, the system includes the neighboring shot on either side and samples the first, middle, and last available keyframes across the combined span. Hits separated by fewer than eight seconds are merged so that adjacent detections do not consume multiple evidence slots. A merged window is bounded to at most five shots. The default answer path retains three temporal windows, giving up to nine retrieved frames ordered by timestamp.

Speech is attached by interval overlap rather than by testing whether one representative timestamp falls inside a transcript segment. The current answer path uses the retrieved center shot's start and end bounds with a 1.5-second margin. OCR is attached from all indexed keyframes belonging to that center shot and is deduplicated case-insensitively.

### 4.8 Global Context and Answer Generation

Retrieval may find a decisive moment but omit the broader narrative required by Level 3 questions. The answer prompt therefore combines local and global evidence:

1. eight frames sampled uniformly at the midpoint of equal temporal intervals;
2. a transcript digest capped at 1,800 characters and sampled across the full timeline;
3. up to three retrieved temporal windows, each represented by several chronological frames;
4. speech and OCR annotations associated with those windows;
5. the question and its labeled answer options.

Qwen2.5-VL-3B receives at most 20 images, each resized to a maximum side length of 448 pixels. Generation is deterministic and limited to eight new tokens. The prompt requests a single answer letter, and a bounded regular expression accepts only labels valid for the number of options. Any output without a valid label is marked as unparsed rather than silently converted into an answer.

The backend exposes health, search, answer, file-serving, and on-demand GIF-preview endpoints through FastAPI. A small Express application serves the browser interface and proxies requests to the backend.

## 5. Experimental Setup and Results

> **Result status:** The tables below were generated from the current JSON files in `results/eval/`. All five configurations in the automated sweep were rerun after the temporal-window and Non-Linear-score corrections. The optional `+asr-whisper` configuration was not rerun and is therefore reported as not available rather than mixed with results from an earlier code revision.

### 5.1 Ablation Configurations

Six configurations are registered in the evaluation harness. The automated sequential sweep currently runs five of them and leaves `+asr-whisper` available as an optional diagnostic run.

| Configuration | Uniform global context | Visual retrieval | Speech retrieval | OCR retrieval | Query expansion |
|---|---|---|---|---|---|
| `baseline-uniform` | Yes | No | No | No | No |
| `visual-only` | Yes | SigLIP2 + jina-clip-v2 | No | No | No |
| `+asr-gt` | Yes | SigLIP2 + jina-clip-v2 | Ground-truth subtitles | No | No |
| `+asr-whisper` | Yes | SigLIP2 + jina-clip-v2 | Whisper transcript | No | No |
| `+ocr` | Yes | SigLIP2 + jina-clip-v2 | Ground-truth subtitles | Yes | No |
| `full` | Yes | SigLIP2 + jina-clip-v2 | Ground-truth subtitles | Yes | DeepSeek |

The intended comparisons are:

- `visual-only − baseline-uniform`: contribution of visual retrieval;
- `+asr-gt − visual-only`: upper-bound contribution of speech retrieval;
- `+asr-whisper − +asr-gt`: degradation caused by ASR error;
- `+ocr − +asr-gt`: contribution of on-screen text;
- `full − +ocr`: contribution of query expansion.

### 5.2 Evaluation Metrics

**Question accuracy** is the fraction of questions answered correctly:

\[
\operatorname{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\hat{a}_i = a_i].
\]

Accuracy is also reported by Video-MME-v2 Level and by `second_head` question category. Every category result must be accompanied by its sample count because several development-set categories contain too few examples for reliable interpretation.

**Non-Linear group score** evaluates groups of four related questions. For relevance/consistency groups with \(c\) correct answers, the score is

\[
\left(\frac{c}{4}\right)^2.
\]

For logic/coherence groups, only the leading run of correct answers before the first error receives credit:

\[
\frac{\text{number of consecutive correct answers from the start}}{4}.
\]

This metric penalizes isolated correct guesses and broken reasoning chains more heavily than ordinary accuracy.

**Unparsed rate** measures the fraction of model outputs from which no valid answer label can be extracted. A rate above 5% is treated as a prompt or runtime failure that must be resolved before interpreting accuracy.

Because configurations answer the same questions, pairwise differences should be tested with the exact McNemar test. The report should include the number of baseline-correct/configuration-wrong and baseline-wrong/configuration-correct pairs, not only a p-value.

### 5.3 Runtime Environment

The answer model and offline neural components run in the `jina_env` Conda environment with `transformers>=4.55,<5`. The upper bound is required by jina-clip-v2 remote code, while SigLIP2 and Qwen2.5-VL require a recent 4.x release. Qdrant and Elasticsearch 8.17 run as local Docker services; the GPU-dependent backend runs directly on the host.

The available GPU memory is approximately 12 GB. During evaluation, the two text encoders are placed on CPU while Qwen remains on GPU. This avoids loading SigLIP2, jina-clip-v2, and Qwen concurrently on the GPU, which previously caused out-of-memory failures. Evaluation is sequential across configurations, and an output file with an excessive unparsed rate is rejected rather than accepted as a zero-accuracy result.

The current unit test suite contains 158 tests. In the latest local verification, the suite completed successfully with 10 integration tests skipped because the backend and VLM were not running.

### 5.4 Main Results

| Configuration | Split | Questions | Accuracy | Non-Linear score | Unparsed rate |
|---|---|---:|---:|---:|---:|
| `baseline-uniform` | Dev | 200 | 25.50% | 14.63% | 0.00% |
| `visual-only` | Dev | 200 | 31.00% | 16.50% | 0.00% |
| `+asr-gt` | Dev | 200 | 31.50% | 16.63% | 0.00% |
| `+asr-whisper` | Dev | Not run | N/A | N/A | N/A |
| **`+ocr`** | **Dev** | **200** | **32.50%** | **17.88%** | **0.00%** |
| `full` | Dev | 200 | 30.00% | 16.25% | 0.00% |
| `baseline-uniform` | Test | 600 | 20.83% | 9.42% | 0.00% |
| `visual-only` | Test | 600 | 21.50% | 9.88% | 0.00% |
| `+asr-gt` | Test | 600 | 21.83% | 10.00% | 0.00% |
| `+asr-whisper` | Test | Not run | N/A | N/A | N/A |
| **`+ocr`** | **Test** | **600** | **22.67%** | **10.08%** | **0.00%** |
| `full` | Test | 600 | 22.33% | 9.88% | 0.00% |

`+ocr` answers 136 of 600 test questions correctly, versus 125 for `baseline-uniform`. Its absolute gain is therefore 11 questions, or 1.83 percentage points. All configurations remain above the empirical 12.8% random-choice baseline, and the zero unparsed rate indicates that the differences are not caused by answer-format failures.

### 5.5 Results by Reasoning Level

| Configuration | Split | Level 1 accuracy | Level 2 accuracy | Level 3 accuracy |
|---|---|---:|---:|---:|
| `baseline-uniform` | Dev | 48.28% | 19.61% | 22.50% |
| `visual-only` | Dev | 48.28% | 29.41% | 27.50% |
| `+asr-gt` | Dev | 51.72% | 27.45% | 28.33% |
| `+ocr` | Dev | 55.17% | 29.41% | 28.33% |
| `full` | Dev | 55.17% | 29.41% | 24.17% |
| `baseline-uniform` | Test | 31.45% | 12.42% | 19.64% |
| `visual-only` | Test | 31.45% | 12.42% | 21.07% |
| `+asr-gt` | Test | 31.45% | 14.29% | 20.71% |
| **`+ocr`** | **Test** | **31.45%** | **14.29%** | **22.50%** |
| `full` | Test | 29.56% | 17.39% | 21.07% |

The development split contains 29 Level 1, 51 Level 2, and 120 Level 3 questions; the test split contains 159, 161, and 280, respectively. On test, `+ocr` leaves Level 1 unchanged, improves Level 2 by 1.86 points, and improves Level 3 by 2.86 points over the baseline. The `full` configuration obtains the best Level 2 result, but its lower Level 1 and Level 3 scores prevent it from achieving the best overall accuracy.

### 5.6 Results by Question Category

| `second_head` category | Sample count | Baseline accuracy | Selected configuration accuracy | Difference |
|---|---:|---:|---:|---:|
| Action & Motion | 45 | 11.11% | 11.11% | +0.00 pp |
| Change | 33 | 18.18% | 18.18% | +0.00 pp |
| Complex Plot Comprehension | 55 | 25.45% | 30.91% | +5.45 pp |
| Frame-Only | 89 | 32.58% | 31.46% | -1.12 pp |
| Frames & Audio | 70 | 30.00% | 31.43% | +1.43 pp |
| Order | 52 | 11.54% | 15.38% | +3.85 pp |
| Physical World Reasoning | 68 | 19.12% | 16.18% | -2.94 pp |
| Social Behavior Analysis | 74 | 18.92% | 25.68% | +6.76 pp |
| Temporal Reasoning | 31 | 9.68% | 12.90% | +3.23 pp |
| Video-Based Knowledge Acquisition | 83 | 16.87% | 19.28% | +2.41 pp |

This table uses the 600-question test split and selects `+ocr`, the best overall configuration. The largest observed gains occur in Social Behavior Analysis and Complex Plot Comprehension, while Physical World Reasoning decreases. These are descriptive subgroup results; no correction for multiple comparisons was applied.

### 5.7 Paired Significance Tests

| Test-set comparison | Baseline wrong / config correct | Baseline correct / config wrong | Net gain | Exact McNemar p-value |
|---|---:|---:|---:|---:|
| `visual-only` vs. `baseline-uniform` | 30 | 26 | +4 | 0.689 |
| `+asr-gt` vs. `baseline-uniform` | 31 | 25 | +6 | 0.504 |
| `+ocr` vs. `baseline-uniform` | 35 | 24 | +11 | 0.193 |
| `full` vs. `baseline-uniform` | 32 | 23 | +9 | 0.281 |

None of the test-set improvements reaches the conventional 0.05 significance threshold. On development, the corresponding p-values are 0.052, 0.036, 0.013, and 0.108 for `visual-only`, `+asr-gt`, `+ocr`, and `full`, respectively. The contrast between the apparently strong development result and the non-significant test result reinforces the need to avoid selecting conclusions from the smaller tuning split.

### 5.8 Efficiency

| Configuration | Questions | End-to-end time | Questions/second | Peak GPU memory | External API calls |
|---|---:|---:|---:|---:|---:|
| `baseline-uniform` | 600 | 10 min 13 s | 0.98 | Not logged | 0 |
| `visual-only` | 600 | 32 min 21 s | 0.31 | Not logged | 0 |
| `+asr-gt` | 600 | 34 min 54 s | 0.29 | Not logged | 0 |
| `+ocr` | 600 | 36 min 05 s | 0.28 | Not logged | 0 |
| `full` | 600 | 42 min 25 s | 0.24 | Not logged | 600 DeepSeek calls |

The times are wall-clock measurements from the sequential test log. Gemini OCR is an offline indexing cost and is therefore not counted as an evaluation-time API call. Relative to `+ocr`, query expansion adds 6 minutes 20 seconds and 600 external calls while reducing accuracy by 0.33 points; it is not justified by the present result.

### 5.9 Comparison with Published Video-MME-v2 Results

The following models were selected because their published average accuracy is numerically close to the local system's 22.67%. The official results evaluate all 3,200 Video-MME-v2 questions, whereas this project evaluates a custom 600-question test subset drawn from 150 of the 200 locally indexed videos. Input budgets, answer models, and subtitle/audio settings also differ. The table is therefore contextual rather than a head-to-head leaderboard, and the numerical proximity must not be interpreted as model equivalence.

| Model or system | Evaluation set | Frames | Side information | Average accuracy | Non-Linear score |
|---|---:|---:|---|---:|---:|
| **This work: Qwen2.5-VL-3B + retrieval (`+ocr`)** | **Custom test, 600 questions** | **Up to 20** | **GT subtitles + OCR** | **22.67%** | **10.08%** |
| Qwen3-VL-30B-A3B-Instruct [2] | Full, 3,200 questions | 64 | None | 21.3% | 7.8% |
| GLM4.1V-Think [2] | Full, 3,200 questions | 64 | None | 23.7% | 9.4% |
| Keye-VL-1.5-8B [2] | Full, 3,200 questions | 64 | None | 23.8% | 8.9% |
| LLaVA-Video-7B-Qwen2 [2] | Full, 3,200 questions | 64 | Subtitles/audio | 24.0% | 9.7% |
| VITA-1.5-7B [2] | Full, 3,200 questions | 16 | Subtitles/audio | 25.8% | 11.3% |

Two comparisons are particularly informative. First, the local system uses a much smaller 3B answer model and a smaller visual budget than most listed systems, but supplements it with retrieval, OCR, and ground-truth subtitles. Second, official results change materially with the input setting: a model's visual-only result and subtitle-assisted result should not be treated as the same operating point. A valid leaderboard claim would require evaluating the frozen local pipeline on the complete 3,200-question benchmark under the official protocol.

### 5.10 Experimental Discussion

The ablation trend is monotonic from the baseline through visual retrieval, ground-truth speech retrieval, and OCR: test accuracy increases from 20.83% to 21.50%, 21.83%, and 22.67%. The effect sizes are nevertheless small. Visual retrieval contributes only four net additional correct answers, speech adds two beyond visual retrieval, and OCR adds five beyond `+asr-gt`. The Non-Linear score rises by only 0.67 points from baseline to `+ocr`, implying that retrieval has not yet produced a comparably strong improvement in group-level consistency.

Performance is substantially lower on test than on development for every configuration. For `+ocr`, the decrease is 9.83 points, from 32.50% to 22.67%. The two splits have different level distributions, but distribution alone does not explain the gap: Level 1, Level 2, and Level 3 accuracy all decline. This pattern is consistent with development-set selection effects, variation in video content, and limited statistical power rather than a stable seven-point retrieval gain.

The category breakdown suggests that retrieved text and temporally localized context may help narrative and social questions, but the system remains weak on motion, ordering, and temporal reasoning. Sparse still frames cannot reliably represent fine-grained movement, and shot-level retrieval does not guarantee that the sampled first, middle, or last keyframe captures the decisive transition. The negative Physical World Reasoning result further shows that additional context can distract the answer model when retrieved evidence is only weakly relevant.

The `full` configuration does not improve on `+ocr`. Query expansion increases the test runtime by approximately 18% relative to `+ocr` and requires one DeepSeek call per question, yet loses two correct answers. The current external expansion stage should therefore be disabled in the selected configuration unless a future retrieval-level evaluation demonstrates higher evidence recall.

Finally, these results should not yet be interpreted as clean causal modality ablations. As detailed in Section 6.2, the answer-construction path can expose ground-truth transcript and OCR content even when the corresponding retrieval channel is disabled. The tables accurately describe the behavior of the named runtime configurations, but only a rerun after prompt-side modality gating can establish the isolated contribution of visual retrieval, speech, and OCR.

## 6. Limitations

### 6.1 Partial Dataset Coverage

The project evaluates 200 of the 800 Video-MME-v2 videos. Although this produces 800 questions, conclusions may not generalize to the full benchmark. The custom development split is also skewed toward Level 3 questions, so overall development accuracy is sensitive to the hardest category.

### 6.2 Experimental Isolation Requires Further Validation

The retrieval weights correctly disable Qdrant or Elasticsearch queries for excluded sources. However, the answer-construction path currently loads the ground-truth subtitle artifact and shot-level OCR independently of those retrieval weights. Consequently, a configuration intended to isolate visual retrieval can still receive transcript context, and retrieved visual windows can receive OCR annotations even when OCR retrieval is disabled. In addition, the prompt-side ASR loader currently reads the ground-truth `asr/` directory even when the selected retrieval index is Whisper. These paths must be explicitly gated by configuration before the ablation results can be interpreted as clean modality comparisons.

### 6.3 Incomplete Temporal Representation

Temporal windows improve on single-frame evidence but remain sparse summaries of continuous video. Three frames cannot fully represent fine-grained motion, camera transitions, gestures, or actions occurring between samples. The current neighborhood rule is also fixed rather than conditioned on the question type or shot duration.

### 6.4 Question-Only Retrieval

Retrieval uses the question text but not the candidate answers. In multiple-choice tasks, discriminative visual entities or actions may appear only in the options. Option-aware candidate generation could improve evidence recall but would also require careful controls to avoid retrieving misleading support for every distractor.

### 6.5 Fixed Fusion and Evidence Budgets

RRF weights, candidate depth, temporal-gap threshold, number of windows, and global-frame budget are fixed for all questions. Speech-heavy, OCR-heavy, global-narrative, and motion-centric questions likely require different retrieval and context allocation strategies. The current system also lacks a learned or cross-modal reranker after fusion. Moreover, RRF identifies entries by `file_path`: if two sources select different representative keyframes from the same shot, their agreement is not combined until the later `shot_id` aggregation step. A canonical shot-level fusion key would make the intended source agreement explicit.

### 6.6 No Direct Retrieval Ground Truth

The evaluation measures end-to-end answer quality but does not contain temporal evidence labels for each question. A wrong answer cannot therefore be attributed unambiguously to failed retrieval, insufficient visual representation, or failed VLM reasoning. An oracle-evidence subset or manually annotated temporal spans would enable Recall@K, MRR, and conditional answer-accuracy measurements.

### 6.7 Hardware-Constrained Answer Model

Qwen2.5-VL-3B and the 20-image cap are practical choices for a 12 GB GPU rather than an upper bound on VideoQA quality. Larger answer models, higher-resolution images, denser temporal samples, or native video input may change both absolute performance and the relative value of retrieval.

### 6.8 External-Service Reproducibility

OCR and query expansion depend on Gemini and DeepSeek APIs. Service availability, model version changes, quotas, latency, and nondeterministic backend updates may affect reproducibility. Ground-truth subtitles are useful for estimating an ASR ceiling but are unavailable in many real deployments; the optional Whisper configuration is therefore necessary for a realistic assessment.

### 6.9 Statistical Power and Tuning Risk

The 200-question development set is small for fine-grained subgroup analysis. Repeatedly choosing parameters based on this split can overfit the development set even without model training. The test split must remain untouched until the implementation, prompt, fusion weights, and evidence budget are frozen.

## 7. Future Work

The highest-priority next step is to make modality exposure explicit in every evaluation configuration and rerun the full ablation suite. After that correction, the following improvements are recommended:

1. retrieve with the question and each answer option separately, then fuse and diversify the candidate pool;
2. introduce question-dependent routing and dynamic allocation between global frames and local windows;
3. add evidence-level annotations for a representative subset and report temporal Recall@K and oracle-answer accuracy;
4. compare fixed RRF with calibrated shot-level fusion and a lightweight reranking stage;
5. adapt the number and duration of temporal windows to motion, ordering, and global-reasoning questions;
6. benchmark an API-based answer model as a separate experimental setting rather than conflating it with pipeline improvements;
7. evaluate the frozen system on the remaining Video-MME-v2 videos.

## References

[1] C. Fu et al., [“Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis,”](https://arxiv.org/abs/2405.21075) CVPR, 2025.

[2] C. Fu et al., [“Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding,”](https://arxiv.org/abs/2604.05015) arXiv:2604.05015, 2026.

[3] H. Wu, D. Li, B. Chen, and J. Li, [“LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding,”](https://arxiv.org/abs/2407.15754) NeurIPS Datasets and Benchmarks Track, 2024.

[4] E. Song et al., [“MovieChat: From Dense Token to Sparse Memory for Long Video Understanding,”](https://openaccess.thecvf.com/content/CVPR2024/html/Song_MovieChat_From_Dense_Token_to_Sparse_Memory_for_Long_Video_Understanding_CVPR_2024_paper.html) CVPR, 2024.

[5] B. T. Gia et al., [“VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos,”](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/html/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.html) CVPR Workshops, 2025.

[6] X. Ren et al., [“VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos,”](https://arxiv.org/abs/2502.01549) arXiv:2502.01549, 2025.

[7] M. Tschannen et al., [“SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features,”](https://arxiv.org/abs/2502.14786) arXiv:2502.14786, 2025.

[8] A. Koukounas et al., [“jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images,”](https://arxiv.org/abs/2412.08802) arXiv:2412.08802, 2024.

[9] S. Bai et al., [“Qwen2.5-VL Technical Report,”](https://arxiv.org/abs/2502.13923) arXiv:2502.13923, 2025.

[10] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, [“Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods,”](https://doi.org/10.1145/1571941.1572114) SIGIR, 2009.

[11] T. Souček and J. Lokoč, [“TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection,”](https://arxiv.org/abs/2008.04838) arXiv:2008.04838, 2020.

[12] A. Radford et al., [“Robust Speech Recognition via Large-Scale Weak Supervision,”](https://arxiv.org/abs/2212.04356) ICML, 2023.
