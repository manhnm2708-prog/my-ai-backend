import asyncio
import ipaddress
import json
import mimetypes
import os
import socket
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile


GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
DEFAULT_TEXT_MODEL = "openai/gpt-oss-20b"
MAX_SOURCE_BYTES = int(os.getenv("MAX_SOURCE_MB", "25")) * 1024 * 1024
REQUEST_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120"))

LANGUAGE_CODES = {
    "Tiếng Anh": "en",
    "Tiếng Việt": "vi",
    "Tiếng Nhật": "ja",
    "Tiếng Hàn": "ko",
    "Tiếng Pháp": "fr",
}

MODEL_ALIASES = {
    "": DEFAULT_TRANSCRIPTION_MODEL,
    "gpt-transcribe": DEFAULT_TRANSCRIPTION_MODEL,
    "gpt-4o-transcribe": DEFAULT_TRANSCRIPTION_MODEL,
    "gpt-4o-mini-transcribe": DEFAULT_TRANSCRIPTION_MODEL,
}


def _allowed_origins() -> list[str]:
    value = os.getenv(
        "ALLOWED_ORIGINS",
        "https://nghe-noi-video.manhnm2708.chatgpt.site",
    )
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


app = FastAPI(title="Nghe Noi AI Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Backend chưa có biến môi trường GROQ_API_KEY trên Render.",
        )
    return key


def _groq_error(response: httpx.Response) -> HTTPException:
    message = "Groq từ chối yêu cầu. Hãy kiểm tra API key, model và giới hạn tài khoản."
    try:
        body = response.json()
        upstream = body.get("error", {}).get("message")
        if upstream:
            message = f"Groq: {upstream}"
    except (ValueError, TypeError, AttributeError):
        pass
    return HTTPException(status_code=502, detail=message)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "Nghe Noi AI Backend",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    _api_key()
    return {
        "ok": True,
        "provider": "groq",
        "capabilities": ["speech-to-text", "translation-evaluation"],
        "transcription_model": DEFAULT_TRANSCRIPTION_MODEL,
        "text_model": DEFAULT_TEXT_MODEL,
    }


def _is_public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL nguồn phải dùng HTTP hoặc HTTPS.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="URL nguồn không được chứa thông tin đăng nhập.")
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="Không phân giải được tên miền của URL nguồn.") from exc
    addresses = {item[4][0] for item in infos}
    if not addresses or any(not _is_public_ip(address) for address in addresses):
        raise HTTPException(status_code=400, detail="URL nguồn trỏ tới mạng nội bộ hoặc địa chỉ bị chặn.")


def _filename_from_url(url: str, content_type: str) -> str:
    name = Path(urlparse(url).path).name or "media"
    if "." not in name:
        extensions = {
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
        }
        name += extensions.get(content_type.split(";")[0].lower(), ".mp4")
    return name[:180]


async def _download_direct_media(source_url: str) -> tuple[str, bytes, str]:
    current_url = source_url
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=False) as client:
        for _ in range(6):
            await _validate_public_url(current_url)
            async with client.stream(
                "GET",
                current_url,
                headers={"User-Agent": "NgheNoiBackend/1.0"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise HTTPException(status_code=400, detail="URL chuyển hướng nhưng không có đích.")
                    current_url = urljoin(current_url, location)
                    continue
                if response.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"Không tải được URL nguồn (HTTP {response.status_code}).")
                content_type = response.headers.get("content-type", "application/octet-stream")
                if content_type.startswith("text/html"):
                    raise HTTPException(
                        status_code=422,
                        detail="URL này là trang web, không phải liên kết trực tiếp tới tệp video/âm thanh. Với YouTube hoặc Vimeo, hãy tải tệp hợp pháp rồi dùng chế độ Tải tệp.",
                    )
                declared_size = int(response.headers.get("content-length", "0") or 0)
                if declared_size > MAX_SOURCE_BYTES:
                    raise HTTPException(status_code=413, detail="Tệp nguồn vượt giới hạn của backend.")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_SOURCE_BYTES:
                        raise HTTPException(status_code=413, detail="Tệp nguồn vượt giới hạn của backend.")
                    chunks.append(chunk)
                return _filename_from_url(current_url, content_type), b"".join(chunks), content_type
        raise HTTPException(status_code=400, detail="URL nguồn chuyển hướng quá nhiều lần.")


def _download_platform_media_sync(source_url: str) -> tuple[str, bytes, str]:
    """Download one public YouTube/Vimeo audio stream without cookies or DRM bypass."""
    with tempfile.TemporaryDirectory(prefix="nghenoi-") as temp_dir:
        output_template = str(Path(temp_dir) / "source.%(ext)s")
        options = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "restrictfilenames": True,
            "socket_timeout": 20,
            "retries": 2,
            "fragment_retries": 2,
            "max_filesize": MAX_SOURCE_BYTES,
            "overwrites": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=True)
                requested = info.get("requested_downloads") or []
                candidates = [Path(item.get("filepath", "")) for item in requested if item.get("filepath")]
                prepared = Path(downloader.prepare_filename(info))
                candidates.extend([prepared, *Path(temp_dir).glob("source.*")])
        except yt_dlp.utils.DownloadError as exc:
            raise HTTPException(
                status_code=422,
                detail="Không thể lấy âm thanh công khai từ nguồn này. Video có thể riêng tư, giới hạn khu vực, cần đăng nhập hoặc được bảo vệ.",
            ) from exc

        media_path = next((path for path in candidates if path.is_file()), None)
        if media_path is None:
            raise HTTPException(status_code=422, detail="Nguồn không cung cấp luồng âm thanh có thể xử lý.")
        size = media_path.stat().st_size
        if size > MAX_SOURCE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Âm thanh vượt giới hạn {MAX_SOURCE_BYTES // (1024 * 1024)} MB. Hãy dùng video ngắn hơn hoặc tải tệp đã nén.",
            )
        media_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        return media_path.name, media_path.read_bytes(), media_type


async def _download_platform_media(source_url: str) -> tuple[str, bytes, str]:
    await _validate_public_url(source_url)
    return await asyncio.to_thread(_download_platform_media_sync, source_url)


async def _translate_segments(
    segments: list[dict[str, Any]], target_language: str, model: str
) -> list[dict[str, Any]]:
    if not target_language or not segments:
        return segments
    translated = [dict(segment) for segment in segments]
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for offset in range(0, len(translated), 40):
            batch = translated[offset : offset + 40]
            source_items = [{"id": index, "text": item["text"]} for index, item in enumerate(batch)]
            payload = {
                "model": model or DEFAULT_TEXT_MODEL,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Bạn là dịch giả phụ đề. Dịch tự nhiên, giữ đúng nghĩa và không thêm giải thích. "
                            "Chỉ trả JSON hợp lệ dạng {\"translations\":[{\"id\":0,\"text\":\"...\"}]}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Ngôn ngữ đích: {target_language}\nCác câu: {json.dumps(source_items, ensure_ascii=False)}",
                    },
                ],
            }
            response = await client.post(f"{GROQ_API_BASE}/chat/completions", headers=headers, json=payload)
            if response.status_code >= 400:
                raise _groq_error(response)
            try:
                content = response.json()["choices"][0]["message"]["content"]
                items = json.loads(content).get("translations", [])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=502, detail="Groq trả về bản dịch phụ đề không hợp lệ.") from exc
            by_id = {int(item["id"]): str(item["text"]).strip() for item in items if "id" in item and "text" in item}
            for index, item in enumerate(batch):
                item["translation"] = by_id.get(index, "")
    return translated


async def _transcribe_media(
    filename: str,
    content: bytes,
    content_type: str,
    model: str,
    language: str,
    target_language: str = "",
    text_model: str = "",
) -> dict[str, Any]:
    if not content:
        raise HTTPException(status_code=400, detail="Tệp video/âm thanh rỗng.")
    if len(content) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Tệp vượt giới hạn của backend.")
    resolved_model = MODEL_ALIASES.get(model.strip(), model.strip()) or DEFAULT_TRANSCRIPTION_MODEL
    data = {
        "model": resolved_model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }
    language_code = LANGUAGE_CODES.get(language)
    if language_code:
        data["language"] = language_code
    headers = {"Authorization": f"Bearer {_api_key()}"}
    files = {"file": (filename, content, content_type or "application/octet-stream")}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{GROQ_API_BASE}/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
        )
    if response.status_code >= 400:
        raise _groq_error(response)
    result = response.json()
    segments = [
        {
            "start": float(segment.get("start", 0)),
            "end": float(segment.get("end", 0)),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in result.get("segments", [])
        if str(segment.get("text", "")).strip()
    ]
    segments = await _translate_segments(segments, target_language, text_model or DEFAULT_TEXT_MODEL)
    return {
        "provider": "groq",
        "model": resolved_model,
        "text": result.get("text", ""),
        "segments": segments,
    }


@app.post("/transcribe")
async def transcribe(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploaded = form.get("file")
        if not isinstance(uploaded, UploadFile):
            raise HTTPException(status_code=400, detail="Thiếu tệp video/âm thanh trong trường file.")
        content = await uploaded.read(MAX_SOURCE_BYTES + 1)
        return await _transcribe_media(
            uploaded.filename or "media",
            content,
            uploaded.content_type or "application/octet-stream",
            str(form.get("model", "")),
            str(form.get("language", "")),
            str(form.get("target_language", "")),
            str(form.get("text_model", "")),
        )

    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="JSON không hợp lệ.") from exc
        if body.get("data_policy") != "backend":
            raise HTTPException(status_code=400, detail="Hãy chọn 'Cho phép backend tải URL nguồn' trong Cấu hình AI.")
        source_url = str(body.get("source_url", "")).strip()
        source_type = str(body.get("source_type", "")).lower()
        if source_type in {"youtube", "vimeo"}:
            filename, content, media_type = await _download_platform_media(source_url)
        else:
            filename, content, media_type = await _download_direct_media(source_url)
        return await _transcribe_media(
            filename,
            content,
            media_type,
            str(body.get("model", "")),
            str(body.get("language", "")),
            str(body.get("target_language", "")),
            str(body.get("text_model", "")),
        )

    raise HTTPException(status_code=415, detail="Chỉ hỗ trợ multipart/form-data hoặc application/json.")


@app.post("/evaluate-translation")
async def evaluate_translation(request: Request) -> dict[str, Any]:
    body = await request.json()
    source_text = str(body.get("source_text", "")).strip()
    answer = str(body.get("answer", "")).strip()
    target_language = str(body.get("target_language", "")).strip()
    if not source_text or not answer or not target_language:
        raise HTTPException(status_code=400, detail="Thiếu câu gốc, bản dịch hoặc ngôn ngữ đích.")
    model = str(body.get("model", "")).strip() or DEFAULT_TEXT_MODEL
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là giáo viên ngoại ngữ. Đánh giá bản dịch theo nghĩa và ngữ cảnh, "
                    "không yêu cầu giống từng chữ. Chỉ trả JSON hợp lệ với ba trường: "
                    "accepted (boolean), feedback (chuỗi tiếng Việt ngắn), reference (bản dịch gợi ý)."
                ),
            },
            {
                "role": "user",
                "content": f"Câu gốc: {source_text}\nNgôn ngữ đích: {target_language}\nBản dịch của học viên: {answer}",
            },
        ],
    }
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(f"{GROQ_API_BASE}/chat/completions", headers=headers, json=payload)
    if response.status_code >= 400:
        raise _groq_error(response)
    try:
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Groq trả về kết quả đánh giá không hợp lệ.") from exc
    return {
        "accepted": bool(result.get("accepted", False)),
        "feedback": str(result.get("feedback", "Đã đánh giá xong.")),
        "reference": str(result.get("reference", "")),
        "provider": "groq",
        "model": model,
    }
