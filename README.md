# JellyDub Engine

Hướng dẫn luồng mới **một lệnh → MKV + SRT**, setup và GPU/fallback: [docs/cli-guide.md](docs/cli-guide.md).

**Tự động tạo thuyết minh tiếng Việt và phụ đề đồng bộ từ video.**

Engine Python độc lập dành cho xử lý file, phát trực tiếp và tích hợp Jellyfin/TinyTV. Có thể tự tìm phụ đề, nhận dạng lời nói khi không có phụ đề, dịch sang tiếng Việt, tạo giọng đọc, căn thời gian rồi xuất video. File phim gốc được giữ nguyên.

`Video → Tìm SRT / ASR → Dịch tiếng Việt → Tổng hợp giọng nói → Căn thời gian → Trộn / Xuất`

## Điểm nổi bật

- **Ba cách dùng:** CLI xử lý file, HTTP API tạo giọng theo vị trí phát, HLS thử nghiệm cho Jellyfin.
- **Linh hoạt backend:** OPUS-MT hoặc Gemini để dịch; Edge-TTS, VieNeu hoặc Gemini TTS để đọc.
- **Tái sử dụng cache:** lưu bản dịch và từng đoạn giọng nói, tiếp tục công việc khi chạy lại.
- **Tìm phụ đề:** hỗ trợ file cạnh phim, phụ đề nhúng và nguồn trực tuyến tùy chọn.
- **Tự động hóa CLI:** một lệnh tìm phụ đề hoặc chép lời, lồng tiếng, mặc định xuất MKV + SRT và lưu QC trong cache.

## Bắt đầu nhanh

Yêu cầu **Python 3.11–3.13**, **FFmpeg** và [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

git clone --branch enhance/pipeline-performance https://github.com/QuyCHunnn/AphroDiag-Engine.git
cd AphroDiag-Engine
uv sync --extra vieneu
uv run --no-sync jellydub doctor

# Tạo thuyết minh từ phụ đề tiếng Việt
uv run --no-sync jellydub dub movie.mkv --srt movie.vi.srt \
  --source-lang vi --out-dir ./dubbed
```

Mặc định dùng **VieNeu cục bộ**, giọng `Hải Đăng`; cài extra `vieneu`. Model cần tải ở lần đầu, sau đó có thể tạo giọng offline. **Gemini chỉ chạy khi chọn `--tts gemini` hoặc profile Gemini**; engine không tự chuyển backend khi hết quota.

### Dịch phụ đề tiếng Anh

```bash
uv sync --extra translate --extra vieneu
uv run --no-sync jellydub translate movie.mkv --srt movie.en.srt --out-dir ./dubbed

# Kiểm tra/chỉnh bản dịch trước khi tạo giọng
uv run --no-sync jellydub dub movie.mkv --srt dubbed/movie.vi.srt --out-dir ./dubbed
```

OPUS-MT chạy cục bộ sau lần tải model đầu tiên. Để dùng Gemini, đặt biến môi trường `GEMINI_API_KEY` rồi thêm `--translator gemini` vào lệnh dịch. Phụ đề tiếng Việt bỏ qua bước dịch; dùng `--source-lang vi` nếu nhận diện tự động chưa đúng.

## Chọn cách sử dụng

### 1. Tạo file thuyết minh

```bash
# Hạ tiếng gốc khi có lời thuyết minh
uv run --no-sync jellydub dub movie.mkv --srt movie.vi.srt --mix voiceover --out-dir ./dubbed

# Xuất MKV có track phụ đề bật/tắt và MP4 có phụ đề gắn cố định
uv run --no-sync jellydub dub movie.mkv --srt movie.vi.srt --mix voiceover \
  --mkv --mp4 --out-dir ./dubbed

# Tự tìm phụ đề trong máy, không tìm trên mạng
uv run --no-sync jellydub dub movie.mkv --no-download --out-dir ./dubbed
```

Mặc định xuất MKV và SRT tiếng Việt. Dùng `--audio` nếu cần thêm MP3 thuyết minh. `--mkv` thêm audio
và SRT tiếng Việt thành các track riêng, giữ nguyên các stream nguồn.
`--mp4` xuất video H.264/AAC với SRT được render vào hình; font mặc định là
`DejaVu Sans` và có thể đổi bằng `--subtitle-font`.

| Chế độ trộn | Kết quả |
| --- | --- |
| `dub-only` — mặc định khi xử lý file | Chỉ giọng thuyết minh |
| `voiceover` | Giọng thuyết minh + tiếng gốc được hạ âm lượng |
| `replace` | Tách lời gốc, giữ nhạc/hiệu ứng; cần extra `separate` |

Có thể tải phụ đề trực tuyến bằng extra `opensubtitles`. Xem cấu hình mẫu tại [jellydub.example.toml](jellydub.example.toml) và hướng dẫn [tách nền âm thanh](docs/separation.md).

### 2. Tự động hóa từ video

```bash
# Cài ASR; thêm translate nếu cần dịch SRT tiếng Anh
uv sync --extra asr --extra translate --extra vieneu

# Không cần chỉ định SRT: tìm phụ đề phù hợp, hoặc tự chép lời nếu không tìm thấy
uv run --no-sync jellydub auto movie.mp4 --out-dir ./dubbed

# Có thể cung cấp SRT tường minh
uv run --no-sync jellydub auto movie.mp4 --srt movie.en.srt --out-dir ./dubbed
```

`auto` tự nhận diện tiếng Anh/Việt, dịch tiếng Anh, tạo thuyết minh và mặc định xuất
MKV (audio Việt và phụ đề bật/tắt được) cùng SRT rời. MP4 và MP3 chỉ xuất khi thêm `--mp4` / `--audio`. Nếu không
tìm thấy SRT, lệnh dùng `faster-whisper` (mặc định model `small`; model được tải
ở lần đầu). Kết quả mặc định chỉ có MKV và SRT; báo cáo JSON nằm trong thư mục cache `reports/`, đường dẫn được in sau khi chạy. Báo cáo kiểm
tra cấu trúc phụ đề, lỗi đồng bộ và thông tin stream; nó không thể xác nhận
ngữ nghĩa bản dịch hoặc chất lượng giọng nghe bằng tai. Các cảnh báo trong báo
cáo cần được xem lại.

### 3. Live API cho ứng dụng phát video

```bash
uv run --no-sync jellydub serve --port 8765
# Dịch bằng Gemini:
uv run --no-sync jellydub serve --port 8765 --translator gemini
```

Client gửi SRT, cập nhật vị trí phát và lấy các đoạn WAV đã tạo. Engine ưu tiên đoạn quanh vị trí đang xem và điều chỉnh khi tua.

| Endpoint | Chức năng |
| --- | --- |
| `GET /v1/health` | Kiểm tra server |
| `POST /v1/sessions` | Tạo phiên với JSON gồm `srt`, `source_lang`, `title` |
| `GET /v1/sessions/{id}/cues?from=0&to=90` | Lấy trạng thái các câu trong khoảng thời gian (giây) |
| `GET /v1/sessions/{id}/audio/{n}.wav` | Lấy đoạn giọng đọc |
| `DELETE /v1/sessions/{id}` | Đóng phiên |

Xem thêm [hướng dẫn live bằng tiếng Việt](docs/live-dubbing-vi.md). Cài extra `live` nếu cần quảng bá server qua mDNS cho TinyTV.

### 4. HLS cho Jellyfin · Thử nghiệm

```bash
uv run --no-sync jellydub stream movie.mkv --srt movie.vi.srt \
  --public-url http://192.168.1.10:8766 \
  --strm './library/Movie Vietnamese.strm'
```

Thay IP bằng địa chỉ LAN của máy chạy engine. Đặt `.strm` trong thư viện Jellyfin, quét lại thư viện và giữ lệnh chạy khi xem. Jellyfin và thiết bị phát phải truy cập được địa chỉ này.

HLS tạo segment theo yêu cầu, hỗ trợ tua và tái sử dụng cache. Hiện xuất **H.264 SDR tối đa 1080p/30fps + AAC stereo**; nguồn là file video cục bộ. Độ trễ phụ thuộc CPU, dịch và TTS; khả năng phát/tua cần kiểm tra trên thiết bị thực tế.

### Chọn VieNeu hoặc Gemini TTS

Sau khi cài các extra, dùng `--no-sync` để giữ nguyên môi trường đã cài.

```bash
uv sync --extra vieneu --extra translate --extra asr

# Mặc định: VieNeu, không gọi API Gemini để tạo giọng
uv run --no-sync jellydub auto movie.mp4 --mix voiceover --out-dir dubbed/vieneu

# Chọn VieNeu tường minh; có thể dùng lại cache VieNeu đã có
uv run --no-sync jellydub auto movie.mp4 --tts vieneu --voice "Hải Đăng" \
  --out-dir dubbed/vieneu

# Chỉ dùng Gemini khi có quota: nhập key kín, không lưu vào mã/config
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
printf '\n'
uv run --no-sync jellydub auto movie.mp4 --config jellydub.gemini.toml
unset GEMINI_API_KEY

# Quay về VieNeu: chạy lại không kèm profile Gemini
uv run --no-sync jellydub auto movie.mp4 --tts vieneu --out-dir dubbed/vieneu
```

Profile Gemini dùng `Charon`; VieNeu dùng tên preset riêng. Khi đổi backend, bỏ `--voice` cũ hoặc chọn giọng đúng engine. Cache audio được phân biệt theo engine và giọng. Không đổi tự động giữa chừng, nên một video không bị trộn giọng Gemini/VieNeu vì lỗi quota. Chọn `--translator gemini` là đổi **bộ dịch**, độc lập với `--tts gemini` là đổi **bộ tạo giọng**; mặc định dịch vẫn là OPUS-MT.

Lệnh `serve`, `stream`, `dub` và `preview` cũng dùng VieNeu mặc định; truyền `--tts gemini` để chọn Gemini. Ví dụ nghe thử SRT tiếng Việt đã duyệt:

```bash
uv run --no-sync jellydub preview --srt movie.vi.srt --tts vieneu --out-dir preview/vieneu
uv run --no-sync jellydub preview --srt movie.vi.srt --config jellydub.gemini.toml --out-dir preview/gemini
```

Cấu hình Gemini TTS: [hướng dẫn](docs/gemini-tts.md) và [profile mẫu](jellydub.gemini.toml). API key lấy từ biến môi trường, không lưu trong cấu hình.

Kết quả tối ưu và giới hạn: [checkpoint](docs/checkpoints-2026-11-17.md), [kế hoạch và thực thi](docs/next-optimization-execution-2026-11-17.md), [review lỗi](docs/bug-review-2026-11-17.md). GenAI ASR và các thử nghiệm TTS GPU vẫn cần chọn tường minh; chưa đạt mục tiêu video 3 phút trong 40–60 giây.

## Backend và cấu hình

| Thành phần | Lựa chọn | Ghi chú |
| --- | --- | --- |
| Dịch | `opus-mt` — mặc định | Cục bộ; cài extra `translate` |
| Dịch | `gemini` | API trực tuyến; cần API key và quota |
| Dịch | `agy`, `agy-stream` | Thử nghiệm; cần Antigravity CLI đã đăng nhập |
| Giọng đọc | `vieneu` — mặc định | Cục bộ; cài extra `vieneu`, hỗ trợ giọng tham chiếu |
| Giọng đọc | `gemini` — chọn riêng | API; cần key/quota, giọng mặc định Charon |
| Giọng đọc | `edge` — chọn riêng | Trực tuyến; client không chính thức, không clone giọng |

```bash
# Cài các thành phần tùy chọn cần dùng
uv sync --extra translate --extra vieneu --extra separate

# Xem toàn bộ tùy chọn
uv run --no-sync jellydub --help
uv run --no-sync jellydub dub --help
uv run --no-sync jellydub stream --help
```

Các lệnh hỗ trợ `--config jellydub.example.toml`; tham số CLI được ưu tiên. Giữ API key trong biến môi trường. Cache mặc định nằm tại `~/.cache/jellydub`.

> **Lưu ý:** Edge-TTS gửi văn bản tới Microsoft; Gemini và các backend Antigravity gửi văn bản tới Google. Live/HLS chưa có xác thực, chỉ nên dùng trong LAN hoặc tailnet tin cậy. Câu quá dài có thể cần rút gọn để khớp thời gian; engine chưa tự nhận diện người nói.

## Roadmap

Các phase tiếp theo là định hướng, chưa cam kết ngày phát hành.

| Phase | Trọng tâm | Mốc hoàn thành |
| --- | --- | --- |
| **0 · Độc lập** — hiện tại | Package, CLI, live API, HLS thử nghiệm, tài liệu | Cài đặt độc lập; bộ test offline chạy qua |
| **1 · Phát hành** | CI, phiên bản phát hành, chuẩn hóa API/cấu hình | Build wheel tái lập; chính sách tương thích rõ ràng |
| **2 · Streaming ổn định** | Hủy tác vụ khi tua, giới hạn hàng đợi/cache, retry | Đo và kiểm chứng thời gian mở phim/tua trên Jellyfin và Android TV |
| **3 · Chất lượng** | Bộ đánh giá bản dịch, phát âm, gán giọng nhân vật | So sánh chất lượng và độ trễ có thể tái lập |
| **4 · Triển khai** | Container, API có xác thực, ví dụ tích hợp | Triển khai và kết nối client theo tài liệu |
| **5 · Mở rộng** | Hàng đợi bền vững, metrics, giới hạn tài nguyên, GPU worker | Kiểm thử tải và phục hồi; công bố năng lực xử lý |

## Phát triển

```bash
uv sync --group dev
uv run pytest -q
uv build
```

Mã nguồn nằm trong `src/jellydub`; `pipeline.py` điều phối xử lý file, `live/` phục vụ API và HLS. Tên package/lệnh vẫn là `jellydub`. API Python chưa ổn định; ưu tiên CLI hoặc HTTP API khi tích hợp.

Test mặc định bỏ qua nhóm `e2e` và `slow` cần mạng/model. Không commit thông tin đăng nhập, phim, model hay cache.

**Giấy phép:** mã nguồn công bố theo MIT; thư viện, model và dịch vụ có điều khoản riêng. Xem [ghi chú giấy phép bên thứ ba](LICENSES-THIRD-PARTY.md).
