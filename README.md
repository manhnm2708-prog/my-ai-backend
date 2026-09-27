# Nghe Noi AI Backend

Backend FastAPI kết nối ứng dụng Nghe Nói Video với Groq mà không đưa API key vào trình duyệt.

## API

- `GET /health`
- `POST /transcribe` (tệp multipart hoặc URL trực tiếp trong JSON)
- `POST /evaluate-translation`

URL YouTube/Vimeo không phải liên kết media trực tiếp nên backend chủ động từ chối. Hãy dùng phụ đề hoặc tải lên tệp mà bạn có quyền sử dụng.

## Chạy cục bộ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="gsk_..."
uvicorn main:app --reload
```

Không commit khóa thật vào GitHub. Tệp `.env` đã được loại khỏi Git bằng `.gitignore`.

## Render

Repository root phải chứa `main.py` và `requirements.txt` này. Render có thể đọc `render.yaml`; hoặc cấu hình thủ công:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Secret environment variable: `GROQ_API_KEY`
- Environment variable: `ALLOWED_ORIGINS=https://nghe-noi-video.manhnm2708.chatgpt.site`

Sau khi deploy, mở `https://TEN-DICH-VU.onrender.com/health`. Khi khóa hợp lệ, API trả `ok: true`.

Trong web Nghe Nói Video, điền:

- URL backend AI: URL Render, không thêm `/health`
- Model nhận diện: `whisper-large-v3-turbo`
- Model đánh giá bản dịch: `openai/gpt-oss-20b`
- Cách gửi dữ liệu: `Cho phép backend tải URL nguồn` nếu dùng URL trực tiếp đến MP4/MP3
