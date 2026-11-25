# JellyDub Engine

**Tạo giọng đọc và phụ đề tiếng Việt cho video tiếng Anh hoặc tiếng Việt.**

Bạn đưa vào một video. JellyDub tìm phụ đề hoặc chép lại lời nói, dịch nếu cần, tạo giọng tiếng Việt rồi xuất video mới. Video gốc được giữ nguyên.

Hiện dùng bằng lệnh trong terminal. Cách bắt đầu dưới đây dùng **VieNeu**, chạy cục bộ và không cần API key.

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

## 2. Chạy video đầu tiên

Đặt video vào thư mục vừa tải. Ví dụ file tên `video.mp4`:

```bash
uv run --no-sync jellydub auto "video.mp4" --mix dub-only --out-dir output
```

Nếu video có tên khác, thay `video.mp4` bằng tên hoặc đường dẫn của bạn. JellyDub tự tìm phụ đề; nếu không có, nó nhận dạng lời nói. Thời gian chạy tùy độ dài video và phần cứng.

Khi xong, mở thư mục `output`:

| File | Dùng để làm gì? |
| --- | --- |
| `video.AI-Dub.vi.mkv` | Video có giọng đọc và phụ đề tiếng Việt; có thể mở bằng VLC |
| `video.vi.srt` | Phụ đề tiếng Việt, có thể mở bằng trình soạn thảo để sửa |

Trong VLC, chọn track **Tiếng Việt (AI Dub)** nếu chưa được chọn sẵn. MKV vẫn giữ các track gốc.

**Đã có phụ đề tiếng Việt?** Đưa thẳng vào lệnh để dùng bản đã duyệt:

```bash
uv run --no-sync jellydub dub "video.mp4" --srt "video.vi.srt" \
  --source-lang vi --mix dub-only --out-dir output
```

## 3. Chọn cách xử lý tiếng gốc

Ví dụ trên dùng `--mix dub-only`: track Việt chỉ có giọng đọc mới.

| Giá trị `--mix` | Bạn sẽ nghe gì trong track Việt? |
| --- | --- |
| `dub-only` | Chỉ giọng đọc mới |
| `voiceover` | Giọng đọc mới cùng tiếng gốc đã hạ âm lượng; vẫn có thể nghe tiếng Anh |
| `replace` | Giọng đọc mới cùng nền nhạc/hiệu ứng được tách từ video gốc |

`replace` cần cài thêm thành phần tách âm thanh và có thể còn sót giọng gốc. Xem [cách giữ nhạc nền](docs/separation.md) khi cần dùng chế độ này.

## 4. Chọn bộ tạo giọng

Bộ tạo giọng, còn gọi là TTS, quyết định giọng nói và khả năng diễn đạt.

| Lựa chọn | Khi nào dùng? | Thiết lập |
| --- | --- | --- |
| **VieNeu** — mặc định | Muốn bắt đầu ngay, chạy cục bộ, không cần key; nhấn nhá còn hạn chế | Đã có sau bước cài đặt |
| **Gemini** | Muốn chỉ dẫn cảm xúc/cách đọc qua dịch vụ Google | Cần API key và quota; [hướng dẫn](docs/gemini-tts.md) |
| **VoxCPM2** | Muốn thử cách đọc có chỉ dẫn và giọng tham chiếu trên máy | Cần tải model và build runtime GPU; [hướng dẫn](docs/voxcpm2.md) |

VieNeu vẫn chạy nếu bạn không chọn bộ khác. Engine không tự đổi bộ tạo giọng khi hết quota hoặc gặp lỗi GPU.

## 5. Muốn điều chỉnh cách đọc?

Dùng phụ đề tiếng Việt đã duyệt. Bạn tạo thêm hai file bên cạnh video:

**`speech.json`** — ghi cách đọc cho từng câu. Ví dụ dưới đây dành cho câu số `1`, có đúng lời “Thôi được rồi, anh nói đúng.” trong SRT:

```json
[
  {
    "index": 1,
    "text_sub": "Thôi được rồi, anh nói đúng.",
    "direction": {
      "pace": "chậm",
      "nonverbal": "thở dài trước câu"
    }
  }
]
```

**`cau-hinh.toml`** — cho engine biết sẽ dùng file chỉ dẫn nào:

```toml
[jellydub]
tts = "vieneu"
speech_plan = "speech.json"
mix = "dub-only"
```

Sửa `index` và `text_sub` theo câu trong phụ đề của bạn. Phụ đề hiển thị vẫn giữ nguyên; chỉ cách đọc thay đổi.

Nghe thử 10 giây đầu trước:

```bash
uv run --no-sync jellydub preview --srt "video.vi.srt" \
  --config cau-hinh.toml --seconds 10 --out-dir nghe-thu
```

Mở `nghe-thu/preview.wav`. Nếu chưa vừa ý, sửa `speech.json` rồi chạy lại. Khi đã nghe ổn, dùng cùng cấu hình để tạo video:

```bash
uv run --no-sync jellydub dub "video.mp4" --srt "video.vi.srt" \
  --source-lang vi --config cau-hinh.toml --out-dir output
```

Cùng một speech plan có thể dùng cho Gemini và VoxCPM2 sau khi thiết lập bộ tạo giọng tương ứng. Bạn còn có thể chỉ định cảm xúc, từ cần nhấn, giọng cuối câu và mẫu giọng. **Mức hỗ trợ khác nhau giữa các bộ tạo giọng**; VieNeu sẽ cảnh báo phần không thực hiện được. Xem [hướng dẫn cách đọc và JSON mẫu](docs/expressive-speech.md).

## Khi cần thêm

- **Lệnh báo thiếu thành phần:** chạy `uv run --no-sync jellydub doctor`; xem [hướng dẫn CLI](docs/cli-guide.md).
- **Bản dịch hoặc phát âm chưa đúng:** sửa SRT tiếng Việt hoặc lời đọc trong speech plan rồi chạy lại. Các câu đã tạo được lưu để tái sử dụng khi phù hợp.
- **Nhiều nhân vật, giọng tham chiếu, chỉ dẫn chi tiết:** xem [speech plan](docs/expressive-speech.md).
- **Tích hợp ứng dụng/Jellyfin:** xem [Live API](docs/live-dubbing-vi.md) và lệnh `uv run --no-sync jellydub stream --help`. HLS còn thử nghiệm; Live/HLS chưa có xác thực.
- **Phát triển:** chạy `uv sync --group dev`, sau đó `uv run pytest -q`. Cấu hình đầy đủ ở [jellydub.example.toml](jellydub.example.toml).

Hãy nghe duyệt bản dịch, phát âm và cảm xúc trước khi dùng kết quả.

**Giấy phép:** mã nguồn MIT; thư viện, model và dịch vụ có điều khoản riêng. Xem [ghi chú bên thứ ba](LICENSES-THIRD-PARTY.md).
