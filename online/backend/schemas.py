from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, model_validator


class AgentAutoRequest(BaseModel):
    """Agent tự query một lượt (không hỏi lại người dùng)."""
    query: str = Field(min_length=1)
    task_type: str = Field(default="auto")  # auto|kis|qa|trake
    num_results: int = Field(default=100, ge=1, le=100)
    max_rounds: int = Field(default=1, ge=1, le=5)
    enable_vision: bool = True
    video_filter: Optional[str] = None  # giới hạn tìm trong 1 video (vd "L28_V012")


class AgentAutoResponse(BaseModel):
    task_type: str
    modality: str
    results: List[Dict[str, Any]]
    answer: Optional[str] = None
    rounds: int
    trace: List[str] = Field(default_factory=list)


class AgentChatRequest(BaseModel):
    """Lịch sử hội thoại do frontend giữ (stateless ở backend)."""
    messages: List[Dict[str, Any]]


class KISSearchRequest(BaseModel):
    text_query: Optional[str] = None
    image_base64: Optional[str] = None
    num_results: int = Field(default=20)
    score_threshold: float = Field(default=0.2)
    query_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    auto_expand: bool = True
    num_expansions: int = Field(default=4, ge=2, le=10)

    # Rerank
    use_rerank: bool = True
    rerank_top_k: int = Field(default=100, ge=5, le=500)
    rerank_score_type: str = "itm"   # "itc" nhanh hơn; "itm" chính xác hơn

    # RRF
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rrf_weights: Optional[Dict[str, float]] = None  # {"siglip":1.0, "beit3":1.0}

    # Chọn từng kênh embedding; nếu tắt hết thì backend tự bật lại SigLIP.
    use_siglip: bool = True
    use_jina: bool = True
    use_beit3: bool = True
    use_bge: bool = True

    # Giới hạn tìm trong 1 video (video.name đã có payload index)
    video_filter: Optional[str] = None

    # Lọc theo object detection (payload object_detection.* do offline/ingest_object_detection.py nạp)
    # - object_class rỗng -> lọc theo TỔNG số object; có class -> lọc số object của class đó
    # - khớp trong khoảng [object_count - tolerance, object_count + tolerance]
    # Lọc nhiều class cùng lúc: [{"cls": "person", "count": 2}, {"cls": "car"}]
    # (cls rỗng = tổng số object; count rỗng = chỉ cần có mặt ít nhất 1)
    object_filters: Optional[List[Dict[str, Any]]] = None
    # Dạng 1 class (giữ cho tương thích, gộp chung vào object_filters khi xử lý)
    object_class: Optional[str] = None
    object_count: Optional[int] = Field(default=None, ge=0)
    object_count_tolerance: int = Field(default=3, ge=0, le=50)
    # "hard": cắt hẳn ngoài khoảng ±tolerance (Qdrant filter)
    # "soft": không cắt — đúng số lượng thì x1.0, lệch càng xa nhân điểm giảm dần
    #         về object_score_floor (tolerance làm thang decay)
    object_filter_mode: str = Field(default="hard", pattern="^(hard|soft)$")
    object_score_floor: float = Field(default=0.25, ge=0.0, le=1.0)

class TextSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=20)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    use_fuzzy: bool = False
    filters: Optional[Dict[str, Any]] = None

class ShotDetailRequest(BaseModel):
    shot_id: str
    limit: int = Field(default=10)

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    message: Optional[str] = None


class CompositionWeights(BaseModel):
    text: float = Field(default=0.20, ge=0, le=2)
    reference: float = Field(default=0.45, ge=0, le=2)
    feedback: float = Field(default=0.35, ge=0, le=2)
    positive: float = Field(default=0.25, ge=0, le=2)
    negative: float = Field(default=0.25, ge=0, le=2)


class ComposedSearchRequest(BaseModel):
    original_text: str = Field(min_length=1)
    reference_point_id: Union[str, int]
    feedback: str = Field(min_length=1)
    positive_point_ids: List[Union[str, int]] = Field(default_factory=list)
    negative_point_ids: List[Union[str, int]] = Field(default_factory=list)
    num_results: int = Field(default=100, ge=1, le=500)
    score_threshold: float = Field(default=0.0, ge=0, le=1)
    weights: CompositionWeights = Field(default_factory=CompositionWeights)
    use_rerank: bool = False

    @model_validator(mode="after")
    def no_feedback_overlap(self):
        if len(self.positive_point_ids) != len(set(self.positive_point_ids)):
            raise ValueError("duplicate positive_point_ids")
        if len(self.negative_point_ids) != len(set(self.negative_point_ids)):
            raise ValueError("duplicate negative_point_ids")
        if set(self.positive_point_ids) & set(self.negative_point_ids):
            raise ValueError("a point cannot be both positive and negative")
        return self


class EventSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    num_results: int = Field(default=10, ge=1, le=100)
    per_event_k: int = Field(default=100, ge=10, le=1000)
    max_events: int = Field(default=8, ge=2, le=8)
    max_gap_s: Optional[float] = Field(default=None, gt=0)
    temporal_scale_s: float = Field(default=60.0, gt=0)
    soft_gap_s: float = Field(default=30.0, gt=0)


class ComposedSearchResponse(SearchResponse):
    query_state: Dict[str, Any]


class EventSearchResponse(SearchResponse):
    plan: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

# ---- Enhanced Temporal
class EnhancedTemporalSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=10)
    per_event_k: int = Field(default=30)
    max_events: int = Field(default=5)
    score_threshold: float = Field(default=0.1)
    same_video_only: bool = True
    max_gap_seconds: Optional[float] = None
    decay_rate: float = Field(default=0.01, ge=0.001, le=0.1)  # α parameter
    rrf_k: int = Field(default=60)
    rrf_weights: Optional[Dict[str, float]] = None  # {"siglip":1.0,"beit3":0.3}

class EventSpec(BaseModel):
    kis: Optional[str] = None   # mô tả hình ảnh/scene (KIS)
    ocr: Optional[str] = None   # text trong ảnh/biển số (OCR)
    asr: Optional[str] = None   # từ khoá lời nói (ASR)

class TemporalCombinateSearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=5, ge=1, le=100)
    per_event_k: int = Field(default=50, ge=10, le=1000)
    max_events: int = Field(default=5, ge=2, le=8)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    same_video_only: bool = True
    max_gap_seconds: Optional[float] = None
    decay_rate: float = Field(default=0.01, ge=0.001, le=0.1)
    events_explicit: Optional[List[EventSpec]] = None

    # Trọng số multimodal cho MỖI EVENT
    weights: Optional[Dict[str, float]] = None  # {"kis":0.5,"ocr":0.3,"asr":0.2}

    # KIS (SRRF + BLIP2) cho MỖI EVENT
    rrf_k: int = Field(default=60, ge=10, le=200)
    rrf_weights: Optional[Dict[str, float]] = None   # {"siglip":1.0, "beit3":0.3}
    use_rerank_per_event: bool = True
    rerank_score_type: str = "itm"                  # "itc" nhanh hơn
    rerank_top_k: int = Field(default=100, ge=10, le=500)


class IntelligentSearchRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None
    num_results: int = Field(default=20, ge=5, le=100)
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    same_video_only: bool = True
    enable_kis: bool = True
    enable_ocr: bool = True
    enable_asr: bool = True

    # Advanced options
    show_analysis: bool = True
    custom_weights: Optional[Dict[str, float]] = None
    rrf_k: int = Field(default=60, ge=1, le=1000)
    rrf_weights: Optional[Dict[str, float]] = None


class GuidedCandidateSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    candidate_video_count: int = Field(default=20, ge=1, le=50)
    frames_per_video: int = Field(default=3, ge=1, le=5)
    anchor_global_k: int = Field(default=500, ge=50, le=3000)
    max_events: int = Field(default=8, ge=2, le=8)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class GuidedCandidateSearchResponse(BaseModel):
    success: bool
    plan: Dict[str, Any]
    plan_digest: str
    anchor_event: Dict[str, Any]
    videos: List[Dict[str, Any]]
    total_count: int
    search_time: float
    diagnostics: Dict[str, Any]


class GuidedRefineSearchRequest(BaseModel):
    plan: Dict[str, Any]
    plan_digest: str = Field(min_length=1)
    selected_video_ids: List[str] = Field(min_length=1, max_length=20)
    num_results: int = Field(default=10, ge=1, le=100)
    per_event_k: int = Field(default=100, ge=10, le=1000)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_gap_s: Optional[float] = Field(default=None, gt=0)
    temporal_scale_s: float = Field(default=60.0, gt=0)
    soft_gap_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_selected_videos(self):
        cleaned = [str(v).strip() for v in self.selected_video_ids if v and str(v).strip()]
        if len(cleaned) != len(self.selected_video_ids):
            raise ValueError("selected_video_ids must contain non-empty string IDs")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("selected_video_ids must not contain duplicates")
        return self


class GuidedRefineSearchResponse(BaseModel):
    success: bool
    plan: Dict[str, Any]
    selected_video_ids: List[str]
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    diagnostics: Dict[str, Any]
