# model/MultiLLM_OCR.py
# -*- coding: utf-8 -*-
"""
Multi-LLM OCR + Caption (song song, nhiều API key)
- JSON-only: {"text": string|null, "caption": string|null}
- Retry chỉ khi thiếu caption (text có/không không quan trọng)
- Prompt tối ưu cho video keyframe retrieval (caption 20–40 từ, EN)
"""

import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image, ImageOps
from dotenv import load_dotenv
from google import genai
from google.genai import types


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MultiLLMOCR:
    # ----- Prompt chi tiết cho retrieval (caption 20–40 từ) -----
    PROMPT_SINGLE = (
        """Return ONLY JSON: {"text": string|null, "caption": string}.

        OCR "text": extract all readable text exactly as seen (numbers, dates, app/site names, signs),
        left→right then top→bottom; keep original language/casing; join lines with a single space;
        if no text -> text=null.

        Caption (semantic, retrieval-oriented):
        - ONE English sentence (14–28 words) describing the visible scene: main subjects and action, setting type
        (e.g., street, classroom, office, kitchen, news studio, slide, map, chart, mobile/desktop UI),
        and 1–2 distinctive visual cues.
        - Prefer generic phrases over exact strings: write “TV channel logo”, “warning sign”, “news ticker”,
        “timestamp overlay” instead of quoting the tokens.
        - DO NOT include exact on-screen words, brand names, channel codes, timestamps, or numbers in the caption.
        - Keep camera/view hints minimal and only if obvious (e.g., “wide street view”).
        - Include coarse time/lighting only if clear (day/night, indoor/outdoor).

        Style: neutral, factual, present tense, ASCII, no lists, no hedging, no file names, no comments.
        Always provide a non-empty caption. No extra keys
        """
    )

    # Khi gửi nhiều ảnh cùng lúc, yêu cầu trả về JSON array cùng thứ tự
    PROMPT_MULTI = (
        """Return ONLY JSON array of length N, where N = number of images sent, in the SAME ORDER: [{"text": string|null, "caption": string}, ...]. For EACH image do: • OCR: extract all readable text exactly as seen; left→right, top→bottom; keep original language/casing; join lines with single space; if none -> text=null. • Caption: ONE English sentence (20–40 words) optimized for keyframe retrieval; include subject/action, scene type, framing/viewpoint, lighting, and 3–6 salient on-screen text tokens (summarized, not quoted fully); add notable attributes and people count/age if clear; neutral tone. Always provide a best-effort, non-null caption. Return ONLY valid JSON, no extra keys, no markdown.
        """
    )

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        num_llms: int = 6,
        batch_size: int = 128,
        max_workers: Optional[int] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        timeout: int = 120,
        retries: int = 1,
        backoff: float = 1.5,
        caption_min_words: int = 20,
        caption_max_words: int = 40,
        max_output_tokens: int = 500,
    ):
        """
        - num_llms: số key (hoặc số model instance) dùng song song
        - batch_size: số ảnh / request / model
        - max_workers: số thread song song (mặc định = num_llms)
        """
        load_dotenv()

        # Gom POOL key phân biệt (Vertex Express: token dạng AQ.*):
        # GOOGLE_API_KEY_1.., GEMINI_API_KEY_1.., rồi GEMINI_API_KEY / GOOGLE_API_KEY.
        pool: List[str] = []
        i = 1
        while True:
            k = os.getenv(f"GOOGLE_API_KEY_{i}") or os.getenv(f"GEMINI_API_KEY_{i}")
            if not k:
                break
            if k not in pool:
                pool.append(k)
            i += 1
        for k in (os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")):
            if k and k not in pool:
                pool.append(k)
        if not pool:
            raise ValueError("Không tìm thấy key Gemini/Google nào trong .env")
        # Cycle pool cho đủ num_llms slot -> cho phép NHIỀU client/key (tăng concurrency).
        # vd num_llms=40 với 8 key -> 5 client/key.
        self.api_keys: List[str] = [pool[j % len(pool)] for j in range(num_llms)]
        self.key_pool_size = len(pool)

        # Mỗi key -> 1 Client Vertex AI (Express mode). Token AQ.* bắt buộc vertexai=True.
        self.clients = []
        self.model_locks = []
        for key in self.api_keys:
            self.clients.append(
                genai.Client(
                    vertexai=True,
                    api_key=key,
                    http_options=types.HttpOptions(timeout=int(timeout) * 1000),
                )
            )
            self.model_locks.append(Lock())

        self.num_llms = len(self.clients)
        self.batch_size = int(max(1, batch_size))
        self.max_workers = max_workers or self.num_llms

        self.model_name = model_name
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.caption_min_words = int(caption_min_words)
        self.caption_max_words = int(caption_max_words)
        self.max_output_tokens = int(max_output_tokens)

    # ------------- ảnh: mở an toàn -------------
    def _open_image_rgb(self, path: str) -> Image.Image:
        with Image.open(path) as im0:
            im = ImageOps.exif_transpose(im0)
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                im = im.convert("RGBA")
                bg.paste(im, mask=im.split()[3])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < 512:
                scale = 512.0 / float(min(w, h))
                im = im.resize((int(w * scale), int(h * scale)))
            return im

    # ------------- lấy danh sách ảnh -------------
    def get_image_files(self, directory: str) -> List[str]:
        # Đệ quy: keyframe nằm trong subdir theo video (vd keyframes/K20_V004/*.jpg)
        files: List[str] = []
        for root, _dirs, names in os.walk(directory):
            for name in names:
                _, ext = os.path.splitext(name.lower())
                if ext in self.SUPPORTED_FORMATS:
                    files.append(os.path.abspath(os.path.join(root, name)))
        files.sort()
        return files

    # ------------- helpers -------------
    @staticmethod
    def _wc(s: Optional[str]) -> int:
        return len((s or "").strip().split())

    @staticmethod
    def _clean_json_text(s: Optional[str]) -> str:
        if not s:
            return ""
        t = s.strip()
        if t.startswith("```"):
            t = t.replace("```json", "").replace("```", "").strip()
        return t

    def _parse_array_of_pairs(self, response_text: Optional[str], expected: int) -> List[Dict[str, Optional[str]]]:
        """
        Kỳ vọng JSON array: [{"text":..., "caption":...}, ...] có length==expected.
        Nếu sai format → trả danh sách length=expected với None.
        """
        clean = self._clean_json_text(response_text)
        out = [{"text": None, "caption": None} for _ in range(expected)]
        if not clean:
            return out
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                for i in range(min(expected, len(data))):
                    item = data[i]
                    if isinstance(item, dict):
                        out[i]["text"] = (item.get("text") or None)
                        out[i]["caption"] = (item.get("caption") or None)
                return out
            if isinstance(data, dict):  # đôi khi model trả 1 object
                # gán hết object cho phần tử đầu
                out[0]["text"] = data.get("text") or None
                out[0]["caption"] = data.get("caption") or None
                return out
        except Exception:
            return out
        return out

    def _parse_single_pair(self, response_text: Optional[str]) -> Dict[str, Optional[str]]:
        """
        Kỳ vọng 1 object: {"text":..., "caption":...}
        """
        clean = self._clean_json_text(response_text)
        if not clean:
            return {"text": None, "caption": None}
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return {
                    "text": (data.get("text") or None),
                    "caption": (data.get("caption") or None),
                }
            if isinstance(data, list) and data and isinstance(data[0], dict):
                f = data[0]
                return {"text": (f.get("text") or None), "caption": (f.get("caption") or None)}
        except Exception:
            pass
        return {"text": None, "caption": None}

    # ------------- call model -------------
    _TRANSIENT = ("NOT_FOUND", "404", "503", "UNAVAILABLE", "500", "INTERNAL",
                  "timed out", "SSL", "RESOURCE_EXHAUSTED", "429", "Bad Gateway", "502")

    def _gen(self, model_index: int, contents) -> Optional[str]:
        """generate_content với retry cho lỗi transient (404 flakiness region asia-southeast1,
        503/500/timeout/SSL/429). Mỗi lần retry ĐỔI key (xoay model_index) + backoff để né burst."""
        n = len(self.clients)
        last = None
        for att in range(4):  # 1 lần + 3 retry, mỗi lần 1 key khác
            mi = (model_index + att) % n
            try:
                with self.model_locks[mi]:
                    resp = self.clients[mi].models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        ),
                    )
                return getattr(resp, "text", None)
            except Exception as e:
                if not any(t in str(e) for t in self._TRANSIENT):
                    raise
                last = e
                time.sleep(0.5 * (2 ** att))  # 0.5, 1, 2s
        raise last

    def _call_multi(self, model_index: int, images: List[Image.Image]) -> Optional[str]:
        """Gọi 1 lần cho N ảnh → JSON array N phần tử (có retry transient)."""
        return self._gen(model_index, [self.PROMPT_MULTI] + images)

    def _call_single(self, model_index: int, image: Image.Image) -> Optional[str]:
        """
        Gọi 1 ảnh đơn lẻ → object {"text","caption"}.
        """
        return self._gen(model_index, [self.PROMPT_SINGLE, image])

    # ------------- xử lý 1 batch -------------
    def _process_batch_with_model(
        self, batch_id: int, image_paths: List[str], model_index: int
    ) -> List[Dict[str, Any]]:
        """
        - Gọi 1 lần multi-ảnh.
        - Ảnh nào thiếu caption → retry theo ẢNH-ĐƠN (tối đa self.retries).
        """
        results: List[Dict[str, Any]] = []
        imgs: List[Image.Image] = []
        val_paths: List[str] = []

        # Load ảnh (RGB + xử lý an toàn)
        for p in image_paths:
            try:
                imgs.append(self._open_image_rgb(p))
                val_paths.append(p)
            except Exception as e:
                results.append({
                    "file_path": os.path.abspath(p),
                    "status": "error",
                    "error": f"load_image: {e}",
                    "batch_id": batch_id,
                    "model_used": model_index + 1,
                })

        if not imgs:
            return results

        try:
            # Gọi 1 lần cho cả batch
            raw = self._call_multi(model_index, imgs)
            pairs = self._parse_array_of_pairs(raw, expected=len(imgs))

            # Với từng ảnh, nếu thiếu caption → retry đơn lẻ
            for i, p in enumerate(val_paths):
                pair = pairs[i] if i < len(pairs) else {"text": None, "caption": None}
                text = (pair.get("text") or None)
                caption = (pair.get("caption") or None)

                # Kiểm tra caption hợp lệ (không rỗng & đủ số từ)
                ok = bool(caption and self._wc(caption) >= self.caption_min_words)

                attempts = 0
                last_err = None
                # Retry theo ảnh đơn lẻ khi caption thiếu/yếu
                while (not ok) and (attempts < self.retries):
                    attempts += 1
                    try:
                        single = self._call_single(model_index, imgs[i])
                        pair2 = self._parse_single_pair(single)
                        text2 = (pair2.get("text") or None)
                        caption2 = (pair2.get("caption") or None)
                        if caption2 and self._wc(caption2) >= self.caption_min_words:
                            text, caption = (text2 or text), caption2
                            ok = True
                            break
                        time.sleep((self.backoff ** (attempts - 1)) * 0.5)
                    except Exception as e:
                        last_err = str(e)
                        time.sleep((self.backoff ** (attempts - 1)) * 0.5)

                results.append({
                    "file_path": os.path.abspath(p),
                    "ocr_result": {
                        "text": (text if (text and str(text).strip()) else None),
                        "caption": (caption if (caption and str(caption).strip()) else None),
                    },
                    "status": "success",
                    "batch_id": batch_id,
                    "model_used": model_index + 1,
                    "error": None if ok else (last_err or None),
                })

        except Exception as e:
            # lỗi cấp-batch (hiếm) → đánh dấu tất cả p trong batch
            for p in val_paths:
                results.append({
                    "file_path": os.path.abspath(p),
                    "status": "error",
                    "error": f"batch_call: {e}",
                    "batch_id": batch_id,
                    "model_used": model_index + 1,
                })

        # Close ảnh
        for im in imgs:
            try:
                im.close()
            except Exception:
                pass

        print(f"✓ Batch {batch_id} xong (Model {model_index + 1}): {len(val_paths)} ảnh")
        return results

    # ================== PUBLIC API: stream ==================
    def process_directory_iter(self, input_directory: str):
        """
        Generator: yield từng result dict ngay khi một batch hoàn tất.
        """
        t0 = time.time()
        files = self.get_image_files(input_directory)
        if not files:
            print("Không tìm thấy ảnh nào")
            return
            yield  # giữ generator

        print(f"Tìm thấy {len(files)} ảnh")
        print(f"Sử dụng {self.num_llms} LLMs, batch_size={self.batch_size}, max_workers={self.max_workers}")

        # Tạo batches round-robin qua các model
        batches: List[Tuple[int, List[str], int]] = []
        for i in range(0, len(files), self.batch_size):
            batch_paths = files[i : i + self.batch_size]
            batch_id = len(batches) + 1
            model_index = (batch_id - 1) % self.num_llms
            batches.append((batch_id, batch_paths, model_index))

        ok = err = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {
                ex.submit(self._process_batch_with_model, bid, paths, mid): (bid, paths, mid)
                for (bid, paths, mid) in batches
            }
            for fut in as_completed(fut_map):
                try:
                    batch_results = fut.result() or []
                except Exception as e:
                    bid, paths, mid = fut_map[fut]
                    print(f"✗ Batch {bid} exception: {e}")
                    batch_results = [
                        {"file_path": p, "status": "error", "error": f"executor: {e}", "batch_id": bid, "model_used": mid + 1}
                        for p in paths
                    ]

                for item in batch_results:
                    if item.get("status") == "success":
                        ok += 1
                    else:
                        err += 1
                    yield item

        dt = time.time() - t0
        print("\n=== STREAM DONE ===")
        print(f"Success: {ok} | Error: {err} | Elapsed: {dt:.1f}s | Rate: {(ok+err)/max(dt,1e-6):.1f} img/s")

    # ================== PUBLIC API: list ==================
    def process_directory(self, input_directory: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chạy xong trả list đầy đủ (gom từ stream).
        Nếu output_file được truyền, sẽ lưu JSON (không bắt buộc).
        """
        all_results: List[Dict[str, Any]] = []
        for res in self.process_directory_iter(input_directory):
            all_results.append(res)

        if output_file:
            try:
                payload = {
                    "metadata": {
                        "created_at": _now_iso(),
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "timeout": self.timeout,
                        "retries": self.retries,
                        "batch_size": self.batch_size,
                        "num_llms": self.num_llms,
                        "total_images": len(all_results),
                    },
                    "results": all_results,
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️  Không lưu được output_file: {e}")

        return all_results
