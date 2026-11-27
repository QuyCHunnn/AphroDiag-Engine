# Dùng JellyDub bằng CLI

Phạm vi hiện tại là repo/CLI, chưa có giao diện hoặc gói ứng dụng. Cần Python 3.11–3.13, uv và FFmpeg/ffprobe. Trên Ubuntu/WSL cài FFmpeg bằng `sudo apt install ffmpeg`. Chạy các lệnh dưới đây từ thư mục repo.

## Setup một lần

```bash
uv run python scripts/setup_engine.py
```

Script tự cài profile ASR/dịch/VieNeu; trên Windows/Linux x86-64 thêm runtime Intel. Không yêu cầu tự chọn CUDA/OpenVINO để xử lý video. GPU phù hợp được ưu tiên, không có backend tương thích thì CPU. Trên máy NVIDIA, CTranslate2 vẫn cần CUDA/cuDNN tương thích để thực thi CUDA; thiếu runtime sẽ báo fallback CPU. Intel đã kiểm chứng trên Iris Xe trong WSL; NVIDIA mới kiểm thử logic chọn thiết bị/fallback, chưa thử phần cứng NVIDIA. Windows native/macOS chưa có kiểm thử độc lập trong lượt này.

Script không tự cài driver, restart WSL hoặc gỡ package ngoài profile. Lockfile dùng PyTorch/torchvision CPU khớp nhau cho runtime CPU; Intel ASR sử dụng OpenVINO, không dựa vào torch.cuda.

## Một lệnh cho một video

```bash
uv run --no-sync jellydub auto "video.mp4" --out-dir output
```

Chỉ đưa video tiếng Anh vào. Engine tự nhận dạng ngôn ngữ, chép lời trực tiếp từ video, dịch rồi tạo giọng; dừng nếu ASR trả về ngôn ngữ khác tiếng Anh. Không cần SRT hoặc nghe thử. Không tìm/tải phụ đề có sẵn. Nhận dạng ngôn ngữ có thể sai với video ít lời hoặc nhiều ngôn ngữ.

`auto` chạy tuần tự, mặc định một câu TTS mỗi lượt. Track Việt chỉ có giọng mới (`dub-only`), không tách nhạc nền/hiệu ứng. Nhận hai file:

- `video.AI-Dub.vi.mkv`: giữ video gốc, thêm audio Việt và SRT Việt bật/tắt được.
- `video.vi.srt`: phụ đề Việt rời.

`auto` không còn nhận `--srt`, `--source-lang`, `--mix`, `--pipeline-mode`. Các giá trị source/mix/pipeline trong TOML được thay bằng chế độ tự nhận dạng ngôn ngữ, chỉ giọng mới và xử lý tuần tự. Muốn dùng SRT riêng hoặc chế độ trộn khác, dùng lệnh kỹ thuật `dub`; các lựa chọn đó không thuộc luồng chính.

MKV giữ các track nguồn và đánh dấu audio/phụ đề Việt mặc định. Nếu thiết lập riêng của player chọn track khác, trong VLC chọn track **Tiếng Việt (AI Dub)** và phụ đề **Phụ đề tiếng Việt**. File xuất xem được độc lập, không cần Jellyfin/TinyTV/server hoặc giữ engine chạy.

## Tự động chọn thiết bị và cache

- ASR: ưu tiên NVIDIA/CTranslate2 khi CUDA khả dụng; tiếp theo Intel/OpenVINO nếu đã có runtime phù hợp; còn lại CPU.
- Intel ASR dùng Optimum/OpenVINO, giữ beam mặc định 5. Đường GenAI từng lỗi remote tensor không được dùng trong engine. Model và VAD tải lần đầu theo backend; lần sau dùng cache.
- `auto` thử CPU nếu suy luận GPU lỗi, giữ model/beam. Không âm thầm giảm beam. `--asr-device` dùng để ép thiết bị khi chẩn đoán; ép GPU/CUDA sẽ báo lỗi nếu thiết bị thất bại thay vì đổi thiết bị ngầm.
- Dịch OPUS dùng PyTorch CPU/CUDA theo runtime cài đặt; chưa có backend dịch Intel trong lần triển khai này. VieNeu tạo giọng cục bộ (Iris Xe hiện dùng CPU); Gemini/Edge-TTS được chọn riêng và tạo giọng trực tuyến. Video MKV là stream-copy nên không cần GPU encode.
- Cache mặc định `~/.cache/jellydub`. Giọng và clip đã căn thời gian dùng chung theo nội dung/tham số; đổi thư mục xuất không tạo lại giọng. Cache/QC kiểm tra hash trước khi tái sử dụng.
- Lỗi TTS: lệnh auto thử lại một lần; có thể chạy lại cùng lệnh để tiếp tục các câu chưa xong.

Log/báo cáo JSON nằm trong cache, đường dẫn in ở cuối. Báo cáo ASR ghi backend/device thực thi và fallback. `completed` chỉ xác nhận xử lý; `needs_review` chỉ ra nội dung/timing cần xem lại. Không có bước duyệt chặn quy trình; kiểm tra tự động không xác nhận được ý nghĩa bản dịch hoặc chất lượng giọng cảm nhận.

## Tuỳ chọn nâng cao

```bash
# Chẩn đoán và đường CPU
uv run --no-sync jellydub doctor
uv run --no-sync jellydub auto video.mp4 --asr-device cpu --out-dir output

# Opt-in thêm MP4 hoặc MP3; mặc định không tạo hai định dạng này
uv run --no-sync jellydub auto video.mp4 --mp4 --audio --out-dir output
```

VieNeu là TTS mặc định, chạy cục bộ sau lần tải model đầu tiên. Gemini chỉ dùng khi chọn `--tts gemini` hoặc profile Gemini; xem [hướng dẫn Gemini](gemini-tts.md). Edge-TTS vẫn có thể chọn riêng và cần mạng cho câu chưa cache. Lần đầu cần mạng để tải model; không đưa API key vào repo nếu chọn translator cần key. Nguồn/đầu ra cũ không bị tự xóa. Dùng thư mục output mới nếu muốn nhìn rõ chỉ các file của lần chạy mới; engine không xóa MP4/MP3 cũ có sẵn trong thư mục.

## Bằng chứng triển khai

Kế hoạch/trạng thái: [optimization-plan.md](optimization-plan.md). Kết quả từng checkpoint: `artifacts/optimization-checkpoints/`. Audit trước sửa: [performance-audit-2026-11-16.md](performance-audit-2026-11-16.md).

## Nhiều giọng theo người nói

SRT hỗ trợ tên người nói như `LUNA: ...` hoặc `[[LUNA]] ...`. Thêm mapping vào TOML:

```toml
[jellydub]
tts = "edge"
voice = "en-US-AndrewMultilingualNeural"

[jellydub.speaker_voices]
LUNA = "en-US-EmmaMultilingualNeural"
RAPTOR = "en-US-AndrewMultilingualNeural"
```

Engine chọn giọng theo nhãn, dùng `voice` nếu không có mapping; chưa tự nhận diện người nói/giới tính từ video. Hai giọng trên đã tạo tiếng Việt trong bài Delta Force, nhưng vẫn là dịch vụ mạng có thể lỗi. Xem [bản bàn giao Delta Force](delta-force-run.md) để biết các bước tự động và phần đã biên tập.

VieNeu cũng hỗ trợ cùng mapping, với tên preset VieNeu thay cho tên giọng Edge. Ví dụ `tts="vieneu"`, `voice="Hải Đăng"`, mapping `LUNA="Trúc Ly"`, `RAPTOR="Hải Đăng"`. Không cần audio reference khi dùng preset; nếu cấu hình reference cho người nói thì reference được ưu tiên. VieNeu v3 Turbo hiện chạy ONNX/CPU trên môi trường Intel này.
