# JellyDub Engine

**Đưa video tiếng Anh vào, nhận video có giọng đọc và phụ đề tiếng Việt.**

JellyDub tự chép lời từ video, dịch sang tiếng Việt, tạo giọng rồi xuất video. Không cần SRT, file chỉ dẫn, giọng tham chiếu hoặc bước nghe thử. Video gốc được giữ nguyên.

Hiện dùng bằng lệnh trong terminal. Mặc định dùng **VieNeu**, chạy cục bộ và không cần API key.

## 1. Cài đặt một lần

Cần **Python 3.11–3.13**, **Git**, **FFmpeg** và **uv**. Nếu chưa có uv, cài theo [hướng dẫn này](https://docs.astral.sh/uv/getting-started/installation/).

Ví dụ trên Ubuntu hoặc WSL:

```bash
sudo apt install git ffmpeg

git clone --branch enhance/pipeline-performance https://github.com/QuyCHunnn/AphroDiag-Engine.git
cd AphroDiag-Engine
uv run python scripts/setup_engine.py
```

Lệnh cuối cài các thành phần cần thiết và kiểm tra môi trường. Lần đầu cần mạng để tải thư viện/model. Từ đây, chạy các lệnh trong thư mục `AphroDiag-Engine`.

## 2. Chạy video

Đặt video tiếng Anh vào thư mục repo rồi chạy:

```bash
uv run --no-sync jellydub auto "video.mp4" --out-dir output
```

Thay `video.mp4` bằng đường dẫn video của bạn. Engine nhận dạng ngôn ngữ và dừng nếu kết quả không phải tiếng Anh. Nhận dạng có thể sai với video ít lời hoặc nhiều ngôn ngữ.

Quy trình chạy lần lượt: chép lời → dịch → tạo giọng → xuất video → kiểm tra file. Không tìm hoặc tải phụ đề có sẵn, không dừng để nghe thử. Tạo giọng từng câu một theo mặc định để giảm tải máy; thời gian chạy tùy độ dài video và phần cứng.

## 3. Nhận kết quả

Mở thư mục `output` sau khi lệnh hoàn tất:

- `video.AI-Dub.vi.mkv`: video có giọng đọc và phụ đề tiếng Việt, mở bằng VLC.
- `video.vi.srt`: phụ đề tiếng Việt rời.

Track **Tiếng Việt (AI Dub)** chỉ có giọng mới, không giữ nhạc nền/hiệu ứng hoặc trộn tiếng Anh. MKV vẫn giữ các track gốc riêng; chọn track Việt trong VLC nếu trình phát chưa chọn sẵn.

Engine tự kiểm tra stream âm thanh, phụ đề và thời lượng đầu ra. Kiểm tra này không đánh giá được ý nghĩa bản dịch hoặc độ tự nhiên của giọng.

## Nếu bị gián đoạn

Chạy lại đúng lệnh trên. Engine tái sử dụng các bước và câu đã có cache hợp lệ; câu TTS bị lỗi được thử lại một lần trong mỗi lượt chạy. Không cần bắt đầu lại toàn bộ. Giữ thư mục cache mặc định `~/.cache/jellydub` để dùng cơ chế này.

Nếu báo thiếu thành phần, chạy:

```bash
uv run --no-sync jellydub doctor
```

Lần đầu cần mạng để tải thư viện/model. Các lần sau có thể dùng lại model đã tải.

## Khi cần thêm

- [Hướng dẫn CLI](docs/cli-guide.md): thiết bị, cache, xuất thêm MP4/MP3 và chọn bộ tạo giọng.
- Các lệnh kỹ thuật `translate`, `dub`, `preview` vẫn dành cho xử lý riêng; không thuộc quy trình `auto`. `auto` không nhận `--srt`, `--source-lang`, `--mix` hoặc `--pipeline-mode`.
- [Gemini TTS](docs/gemini-tts.md), [VoxCPM2](docs/voxcpm2.md), [chỉ dẫn cách đọc](docs/expressive-speech.md): cấu hình nâng cao.
- [Live API](docs/live-dubbing-vi.md): tích hợp ứng dụng/Jellyfin. HLS còn thử nghiệm; Live/HLS chưa có xác thực.
- Phát triển: `uv sync --group dev`, sau đó `uv run pytest -q`. Cấu hình đầy đủ ở [jellydub.example.toml](jellydub.example.toml).

**Giấy phép:** mã nguồn MIT; thư viện, model và dịch vụ có điều khoản riêng. Xem [ghi chú bên thứ ba](LICENSES-THIRD-PARTY.md).
