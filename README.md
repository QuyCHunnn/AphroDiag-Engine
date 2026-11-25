# JellyDub Engine

Hướng dẫn luồng mới **một lệnh → MKV + SRT**, setup và GPU/fallback: [docs/cli-guide.md](docs/cli-guide.md).

**Tự động tạo thuyết minh tiếng Việt và phụ đề đồng bộ từ video.**

Engine Python độc lập dành cho xử lý file, phát trực tiếp và tích hợp Jellyfin/TinyTV. Có thể tự tìm phụ đề, nhận dạng lời nói khi không có phụ đề, dịch sang tiếng Việt, tạo giọng đọc, căn thời gian rồi xuất video. File phim gốc được giữ nguyên.

`Video → Tìm SRT / ASR → Dịch tiếng Việt → Tổng hợp giọng nói → Căn thời gian → Trộn / Xuất`

## Điểm nổi bật

- **Ba cách dùng:** CLI xử lý file, HTTP API tạo giọng theo vị trí phát, HLS thử nghiệm cho Jellyfin.
- **Linh hoạt backend:** OPUS-MT hoặc Gemini để dịch; Edge-TTS, VieNeu, Gemini TTS hoặc VoxCPM2 GGUF để đọc.
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

## Chọn backend và chuẩn bị TTS

VieNeu vẫn là mặc định. Gemini, VoxCPM2 và Edge chỉ chạy khi chọn tường minh; engine không tự chuyển backend khi hết quota hoặc lỗi GPU. `--translator gemini` chọn **bộ dịch**, còn `--tts gemini` chọn **bộ tạo giọng**.

| Backend | Cài đặt / tài nguyên | Chỉ dẫn cách diễn | Giọng tham chiếu |
| --- | --- | --- | --- |
| **VieNeu** | `uv sync --extra vieneu`; tải model lần đầu, sau đó chạy cục bộ | Tag phi ngôn ngữ và dấu câu; nhiều yêu cầu chỉ được báo chưa hỗ trợ | WAV 3–8 giây; hoặc preset như `Hải Đăng` |
| **Gemini** | Môi trường JellyDub + `GEMINI_API_KEY`; cần mạng và quota | Style chung và `direction` riêng từng câu | Adapter hiện dùng named voice như `Charon`, không clone từ WAV |
| **VoxCPM2** | Runtime C++ `voxcpm2-cli` + hai file GGUF; không cần Python extra TTS | Prefix chỉ dẫn chung và riêng từng câu | WAV 3–8 giây; không nhận preset VieNeu/Gemini |
| **Edge** | Có trong dependency cơ bản; cần mạng | Không hỗ trợ `direction` diễn xuất, engine sẽ cảnh báo | Không clone giọng |

Dùng `uv run --no-sync` sau khi đã cài môi trường. Nếu cần dịch/ASR/tách nền, cài thêm các extra tương ứng trong cùng lệnh `uv sync`; không cần chúng cho preview SRT tiếng Việt.

### VieNeu: profile cục bộ

Tạo `tts.vieneu.toml`:

```toml
[jellydub]
tts = "vieneu"
voice = "Hải Đăng"
tts_device = "cpu"
speech_plan = "speech.json"
mix = "dub-only"
```

Cấu hình này chọn CPU tường minh. VieNeu không dùng backend Intel OpenCL của VoxCPM2. Có thể thêm `reference = ["refs/peter.wav"]` để dùng audio tham chiếu; bỏ trường này nếu chỉ dùng preset.

### Gemini: profile và API key

Tạo `tts.gemini.toml`:

```toml
[jellydub]
tts = "gemini"
tts_model = "gemini-3.8-flash-lite-tts"
voice = "Charon"
tts_style = "Speak Vietnamese naturally. Follow each line's acting direction."
tts_workers = 2
speech_plan = "speech.json"
mix = "dub-only"
```

Nhập key trong terminal trước khi chạy preview/dub:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
printf '\n'
# Chạy các lệnh --config tts.gemini.toml bên dưới trong terminal này.
# Sau khi dùng xong:
# unset GEMINI_API_KEY
```

Adapter hiện đọc `GEMINI_API_KEY` hoặc `GOOGLE_API_KEY`; chưa tích hợp OAuth/ADC hay token đăng nhập CLI. Không có cơ chế lấy quota Code Assist/CLI để tạo giọng. Key không được ghi vào TOML hoặc cache. Profile có sẵn [jellydub.gemini.toml](jellydub.gemini.toml) dùng `voiceover`; nếu muốn không trộn tiếng Anh gốc, dùng `dub-only` như ví dụ trên. Xem [chi tiết Gemini](docs/gemini-tts.md).

### VoxCPM2: build Intel GPU trên WSL

Máy phải có Intel OpenCL runtime/driver hoạt động. Cài công cụ build và headers không tự cài hay sửa driver GPU. Lệnh `clinfo -l` cần liệt kê Intel Graphics; WSL host cần truy cập được `/dev/dxg`.

Chạy từ thư mục gốc repo:

```bash
sudo apt install build-essential cmake ninja-build git ocl-icd-opencl-dev opencl-headers clinfo
clinfo -l

# Runtime đã dùng để kiểm tra; áp patch OpenCL trong repo này.
git clone https://github.com/tc-mb/llama.cpp-omni.git runtime/llama.cpp-omni
git -C runtime/llama.cpp-omni checkout eefebf4e399a3cc99119cb097d6f4c9e8f329365
git -C runtime/llama.cpp-omni apply ../../patches/voxcpm2-opencl-audio.patch
cmake -S runtime/llama.cpp-omni -B runtime/voxcpm2-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_CUDA=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF
cmake --build runtime/voxcpm2-build --target voxcpm2-cli -j 2

# Hai file khoảng 3,55 GB tổng cộng; tải đúng revision đã thử.
uvx --from huggingface-hub hf download DennisHuang648/VoxCPM2-GGUF \
  VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf \
  --revision 169f64d8b98bbaab1761e4ca3a83e6af653456cc \
  --local-dir models/voxcpm2
```

Chỉ áp patch một lần vào runtime đúng commit. Tạo `tts.voxcpm2.toml`:

```toml
[jellydub]
tts = "voxcpm2"
tts_model = "models/voxcpm2"
voxcpm2_binary = "runtime/voxcpm2-build/bin/voxcpm2-cli"
voxcpm2_timeout = 600
tts_device = "gpu"
tts_style = "Speak Vietnamese naturally. Follow each line's acting direction."
reference = ["refs/peter.wav"]
tts_workers = 1
speech_plan = "speech.json"
mix = "dub-only"
```

Chuẩn bị clip `refs/peter.wav` dài 3–8 giây của giọng muốn tham chiếu. Nếu bỏ `reference`, model tạo giọng theo lời/chỉ dẫn, không bảo đảm giữ cùng danh tính qua nhiều câu. Thư mục `tts_model` phải chứa đúng một `VoxCPM2-BaseLM-*.gguf` và một `VoxCPM2-Acoustic-*.gguf`. Có thể bỏ `voxcpm2_binary` nếu `voxcpm2-cli` đã nằm trên PATH. Bỏ `voice`/`speaker_voices` của backend trước khi chuyển sang VoxCPM2.

`gpu` và `auto` yêu cầu runtime xác nhận toàn bộ lớp BaseLM cùng các thành phần âm thanh dùng GPU; không chấp nhận fallback CPU. `cuda` còn yêu cầu backend CUDA. Muốn chạy CPU phải chọn `tts_device = "cpu"`. Tokenization, điều khiển, I/O và một số nút phụ trợ vẫn xử lý trên host; GPU không có nghĩa mọi thao tác đều chạy GPU.

[jellydub.voxcpm2.toml](jellydub.voxcpm2.toml) là profile của máy thử, trỏ vào `artifacts`; clone repo mới không có model/binary/audio ở đó. Trên máy khác dùng profile ở trên với đường dẫn đã chuẩn bị. Xem [kiểm tra kernel và chi tiết VoxCPM2](docs/voxcpm2.md).

## Chỉ dẫn cách đọc bằng speech plan chung

Một JSON dùng cho **Gemini, VieNeu và VoxCPM2**. Chỉ thay profile backend; phụ đề và chỉ dẫn được quản lý riêng. Cần SRT tiếng Việt cuối cùng đã duyệt trước khi viết plan. Chưa có bước tự xem video rồi suy ra cảm xúc.

Ví dụ câu số 1 trong SRT:

```srt
1
00:00:00,000 --> 00:00:04,000
Thôi được rồi, anh nói đúng.
```

Tạo `speech.json` tương ứng, hoặc sửa từ [JSON mẫu](examples/speech-plan.json):

```json
[
  {
    "index": 1,
    "text_sub": "Thôi được rồi, anh nói đúng.",
    "text_tts": "Thôi… được rồi. Anh nói đúng.",
    "direction": {
      "emotion": "buồn, miễn cưỡng chấp nhận",
      "pace": "chậm",
      "emphasis": ["được rồi", "anh nói đúng"],
      "ending": "hạ giọng, nhỏ dần",
      "nonverbal": "thở dài trước câu"
    }
  }
]
```

| Trường | Ý nghĩa |
| --- | --- |
| `index` | Số câu trong SRT tiếng Việt cuối cùng, không phải vị trí trong mảng JSON |
| `text_sub` | Nội dung phụ đề phải khớp SRT để phát hiện plan bị lệch; không dùng để thay SRT |
| `text_tts` | Lời thực sự đọc; bỏ trường này để dùng lời SRT, hoặc đặt `""` để bỏ đọc câu |
| `direction.emotion` | Cảm xúc và ý định diễn, ví dụ miễn cưỡng chấp nhận |
| `direction.pace` | Nhịp đọc, ví dụ chậm |
| `direction.emphasis` | Mảng các từ/cụm từ muốn nhấn |
| `direction.ending` | Cách kết thúc câu, ví dụ hạ giọng |
| `direction.nonverbal` | Hành động phi ngôn ngữ, ví dụ thở dài trước câu |
| `ref_audio` | Tùy chọn ngoài `direction`: đường dẫn mẫu giọng riêng cho câu |

Mỗi trường trong `direction` đều tùy chọn; trừ `emphasis` là mảng string, các giá trị còn lại là string không rỗng. Chỉ dẫn nên cụ thể về từ cần nhấn, nhịp và cuối câu; tránh ghi chung chung “đọc truyền cảm”. JSON sai kiểu/trường lạ hoặc `text_sub` không khớp sẽ báo lỗi trước khi tạo giọng.

Cách chuyển chỉ dẫn phụ thuộc backend:

| Backend | Cách thực hiện và giới hạn |
| --- | --- |
| **Gemini** | Gửi chỉ dẫn riêng từng câu trong speech metadata/prompt, tách khỏi transcript. Kết hợp với `tts_style` chung |
| **VoxCPM2** | Chuyển thành một prefix `(Emotion: ...; Pace: ...)`, kết hợp với `tts_style`. Prefix không được ghi vào SRT |
| **VieNeu** | `thở dài`, `cười`, `hắng giọng` trước câu được chuyển thành tag; nhịp `chậm` được gợi ý bằng dấu ba chấm ở chỗ ngắt có sẵn. `emotion`, `emphasis`, `ending` chưa có điều khiển trực tiếp và được báo không áp dụng |

VieNeu không tự biến cảm xúc “buồn” thành thở dài. Tag viết tay như `[thở dài]` trong `text_tts` chỉ dùng cho VieNeu; muốn dùng cùng plan cho ba backend, đặt hành động trong `direction.nonverbal`. Cùng một chỉ dẫn không bảo đảm các model diễn giống nhau hoặc làm đúng tuyệt đối.

`speech_plan` và reference trong TOML tính từ thư mục chạy lệnh. `ref_audio` trong JSON tính từ thư mục JSON; với VieNeu/VoxCPM2, mẫu phải dài 3–8 giây. Gemini hiện bỏ qua `ref_audio` trong plan có `direction` và báo cảnh báo; không đặt `reference` trong profile Gemini. Nếu dùng trường `emotion` cũ cùng `direction.emotion`, hai giá trị phải giống nhau. [Schema và tương thích plan cũ](docs/expressive-speech.md).

### Nhiều nhân vật

Gán speaker bằng prefix trong SRT, ví dụ `PETER: Không! Anh không ổn!`. Parser tách label khỏi lời đọc. Trong profile VieNeu/VoxCPM2 có thể dùng:

```toml
[jellydub]
# Thêm vào profile hiện có, không tạo hai bảng [jellydub] trong cùng file.
reference = ["PETER=refs/peter.wav", "LOIS=refs/lois.wav"]
```

Với Gemini, dùng named voice thay cho WAV:

```toml
[jellydub.speaker_voices]
PETER = "Charon"
LOIS = "Kore"
```

Engine chưa tự nhận diện người nói. VoxCPM2 có reference theo label nhưng thiếu label của câu sẽ báo lỗi nếu không có reference mặc định; có thể thêm một đường dẫn không label vào `reference` làm mặc định.

## Preview, duyệt rồi chạy toàn video

Chọn **một** profile đã chuẩn bị. Ví dụ với VoxCPM2:

```bash
# Nghe các câu bắt đầu trong 10 giây đầu; có thể đổi --start để chọn đoạn khác.
uv run --no-sync jellydub preview --srt movie.vi.srt \
  --config tts.voxcpm2.toml --start 0 --seconds 10 --out-dir preview/voxcpm2

# Sau khi nghe và chỉnh speech.json, dùng cùng profile để dub.
uv run --no-sync jellydub dub movie.mp4 --srt movie.vi.srt --source-lang vi \
  --config tts.voxcpm2.toml --out-dir dubbed/voxcpm2
```

Đổi thành `tts.vieneu.toml` hoặc `tts.gemini.toml` để dùng plan này với backend khác. Preview nhận SRT đã duyệt; không dịch hay ASR. Mỗi lượt preview chọn tối đa 30 giây theo thời điểm bắt đầu câu và không cắt ngang lời đọc.

Preview xuất `preview.wav`, `preview.vi.srt` và `review.json`. Kiểm tra `speech_directions` để xem chỉ dẫn đã nhận, `speech_warnings` để biết phần chưa hỗ trợ/bị bỏ qua, `execution` để xem backend thực thi và `timing_warnings` để xem cảnh báo căn thời gian. Trạng thái vẫn là `pending_listening_review`; cần người nghe duyệt phát âm, cảm xúc, nhịp, đúng nhân vật và độ vừa câu. Lệnh `dub` không ép người dùng phải duyệt preview trước.

Muốn loại tiếng gốc khỏi **track Việt**, chọn `dub-only`. Muốn giữ nhạc/hiệu ứng, cài extra `separate` rồi chọn `replace`; chất lượng tách lời gốc cần nghe kiểm tra, chưa bảo đảm sạch hoàn toàn. `voiceover` vẫn trộn tiếng gốc ở mức thấp nên có thể nghe tiếng Anh. MKV giữ cả các track nguồn; trong trình phát cần chọn đúng track tiếng Việt.

### Cache, lỗi thường gặp và tốc độ

- **Chạy lại sau lỗi:** các câu thành công được cache. Chỉ dẫn khác nhau của Gemini/VoxCPM2, giọng, reference hoặc model khác nhau tạo cache riêng. VieNeu có thể tái sử dụng audio khi chỉ thay trường chưa hỗ trợ; cảnh báo mới vẫn được ghi.
- **Gemini báo key/quota:** kiểm tra biến môi trường, quyền dùng model và quota của project; engine không tự đổi sang VieNeu/VoxCPM2.
- **GPU bị từ chối hoặc lỗi `SIN`/`SUM`:** kiểm tra `clinfo -l`, đúng binary có OpenCL và patch đúng revision. Sandbox có thể không thấy GPU dù WSL host nhận được; engine không tự vượt sandbox.
- **Sai `text_sub` hoặc index:** sửa plan theo bản SRT tiếng Việt cuối cùng; chỉ những index có trong cửa sổ preview mới được xử lý.
- **Câu quá dài:** chia hoặc rút gọn câu rồi xem lại phụ đề/thời gian. VoxCPM2 giới hạn 200 patch, tương ứng 32 giây ở model đã thử; audio chạm giới hạn bị từ chối. Timeout mỗi câu mặc định 600 giây, điều chỉnh bằng `voxcpm2_timeout`.
- **Tốc độ VoxCPM2:** hiện nạp model lại mỗi câu, chưa dùng worker thường trú; nạp model/cấp phát bộ nhớ có thể tốn thời gian hơn suy luận. Không suy từ tốc độ một sample sang thời gian xử lý toàn video.

Kiểm tra ngày 25/11/2026: **575 Python tests đạt**, 5 bài mạng/model bị loại theo marker; **26 ca kiểm tra số OpenCL đạt**. Đã kiểm tra preview VoxCPM2 trên Iris Xe/OpenCL, dùng lại cache và xuất clip dub ngắn; chưa chạy lại toàn video bằng VoxCPM2. Gemini `direction` đã kiểm tra bằng request mô phỏng; không phải xác nhận diễn xuất bằng lượt API thật. Các số đo và giới hạn GPU: [VoxCPM2](docs/voxcpm2.md). Các checkpoint tối ưu trước đó: [checkpoint](docs/checkpoints-2026-11-17.md), [kế hoạch](docs/next-optimization-execution-2026-11-17.md), [review lỗi](docs/bug-review-2026-11-17.md). Chưa đạt mục tiêu video 3 phút trong 40–60 giây.

## Backend và cấu hình

| Thành phần | Lựa chọn | Ghi chú |
| --- | --- | --- |
| Dịch | `opus-mt` — mặc định | Cục bộ; cài extra `translate` |
| Dịch | `gemini` | API trực tuyến; cần API key và quota |
| Dịch | `agy`, `agy-stream` | Thử nghiệm; cần Antigravity CLI đã đăng nhập |
| Giọng đọc | `vieneu` — mặc định | Cục bộ; cài extra `vieneu`, hỗ trợ giọng tham chiếu |
| Giọng đọc | `gemini` — chọn riêng | API; cần key/quota, giọng mặc định Charon |
| Giọng đọc | `voxcpm2` — chọn riêng | Cục bộ; runtime C++ + GGUF, GPU được kiểm tra, hỗ trợ reference/direction |
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
