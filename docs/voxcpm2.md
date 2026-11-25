# VoxCPM2 trong JellyDub

Chọn `tts = "voxcpm2"` hoặc `--tts voxcpm2`. VieNeu vẫn là mặc định; không tự đổi backend. VoxCPM2 dùng runtime C++ GGUF cục bộ, không cần API key hay Python extra mới.

## Profile đã chạy trên máy này

Chạy từ thư mục gốc repo:

```sh
.venv/bin/jellydub preview --config jellydub.voxcpm2.toml \
  --srt artifacts/family-guy-vieneu/reviewed.vi.srt \
  --start 90 --seconds 4 --out-dir delivery/voxcpm2-preview

.venv/bin/jellydub dub "videos/Family Guy - Peter falls down the stairs.mp4" \
  --srt artifacts/family-guy-vieneu/reviewed.vi.srt \
  --config jellydub.voxcpm2.toml
```

Profile [jellydub.voxcpm2.toml](../jellydub.voxcpm2.toml) trỏ tới runtime và model trong `artifacts` của checkout này. Các file model/binary không được đóng gói vào Git/wheel. Trên máy khác cần build/tải rồi đổi đường dẫn. Đường dẫn config tính từ thư mục chạy lệnh.

`tts_model` là thư mục chứa đúng một `VoxCPM2-BaseLM-*.gguf` và một `VoxCPM2-Acoustic-*.gguf`. Mẫu dùng BaseLM Q8_0 + Acoustic F16. `voxcpm2_binary` là executable; bỏ cấu hình này nếu `voxcpm2-cli` đã có trên PATH. `voxcpm2_timeout` giới hạn mỗi câu, mặc định 600 giây.

`tts_device = "gpu"` yêu cầu toàn bộ lớp BaseLM và backend các thành phần âm thanh dùng GPU. `auto` cũng yêu cầu GPU. Engine kiểm tra log runtime; thiếu bằng chứng hoặc fallback CPU sẽ báo lỗi. `cuda` còn yêu cầu backend CUDA. Chỉ `cpu` mới cho phép toàn CPU; truyền cả `--cpu` và `--n-gpu-layers 0` vì `--cpu` của runtime chỉ điều khiển các thành phần âm thanh.

"Toàn GPU" ở đây là BaseLM và các graph thành phần âm thanh. Tokenization, lấy mẫu, điều khiển, I/O và một số nút phụ trợ của llama.cpp vẫn có xử lý host. Không phải mọi thao tác trên máy đều chạy GPU.

Profile dùng `dub-only`, không trộn giọng Anh gốc. Muốn giữ nền và hiệu ứng, chọn `mix = "replace"` và chuẩn bị bộ tách âm thanh riêng; chất lượng loại giọng gốc cần nghe kiểm tra.

## Giọng và nhấn nhá

`reference = ["peter.wav"]` đặt giọng mặc định. Nhiều nhân vật dùng `reference = ["PETER=peter.wav", "LOIS=lois.wav"]`; speaker trong SRT phải khớp label. Có thể thêm một reference không label làm mặc định. Clip tham chiếu phải đọc được, dài 3–8 giây. Backend không nhận `voice` hoặc `speaker_voices` kiểu preset của VieNeu/Gemini; bỏ chúng khi đổi backend. Không có reference thì model tạo giọng theo text/style, không đảm bảo giữ một danh tính giọng qua nhiều câu.

`tts_style` đặt hướng dẫn chung. Speech plan JSON chung cho Gemini/VieNeu/VoxCPM2 cho phép chỉ đạo từng câu qua [`direction`](expressive-speech.md). Trường `emotion` cũ và prefix `(direction)` trong `text_tts` vẫn tương thích:

```json
[
  {
    "index": 25,
    "text_sub": "Không! Anh không ổn!",
    "text_tts": "Không! Anh không ổn!",
    "emotion": "angry, forceful denial; strongly stress the two negatives",
    "ref_audio": "peter.wav"
  }
]
```

Đặt `speech_plan = "path/to/speech.json"` trong TOML. `ref_audio` tính từ thư mục JSON. Với VoxCPM2, `emotion` được đưa vào prefix điều khiển model, kết hợp với `tts_style`; không ghi vào SRT. Tag VieNeu như `[thở dài]` vẫn chỉ dành cho VieNeu; dùng chỉ dẫn tự nhiên cho VoxCPM2.

Cache phân biệt nội dung đọc, chỉ dẫn, reference, device, hash model, executable và thư viện runtime. Synthesis tuần tự, một process/câu; model hiện được nạp lại mỗi câu. Không phải runtime thường trú. Tham số hiện cố định seed 42, tối đa 200 patch, 10 timesteps, CFG 2. Audio chạm giới hạn 32 giây sẽ bị từ chối; chia câu ngắn hơn. Model/reference/runtime cần giữ nguyên trong một lượt chạy.

Preview, dub, auto và worker overlap cùng dùng factory/options này. Kiểm tra decode/cache không thay thế nghe duyệt phát âm, cảm xúc và độ giống giọng.

## Build Intel OpenCL cho WSL

Runtime đã kiểm tra: [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni), commit `eefebf4e399a3cc99119cb097d6f4c9e8f329365`. Model: [DennisHuang648/VoxCPM2-GGUF](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF); revision/checksum của lượt chạy trong `artifacts/voxcpm2-trial-2026-11-25/model-source.json`.

Bản upstream ở commit này thiếu các phép toán OpenCL mà VoxCPM2 cần. Bản vá cục bộ [voxcpm2-opencl-audio.patch](../patches/voxcpm2-opencl-audio.patch) bổ sung đường F32 contiguous cho SIN, COS, ARANGE, SUM, ABS, STEP, ROUND, transposed convolution 1D và chuyển F32 sang I32. Không gửi contribution lên upstream. Giữ patch ngoài thư mục artifact để có thể tái tạo build.

Trên Ubuntu/WSL, chuẩn bị công cụ build và kiểm tra driver trước. Các gói dưới đây không tự cài/sửa driver Intel; `clinfo -l` phải thấy Intel Graphics:

```sh
sudo apt install build-essential cmake ninja-build git ocl-icd-opencl-dev opencl-headers clinfo
clinfo -l
```

Trên host có OpenCL headers/loader và CMake, Ninja, C++ compiler, chạy từ thư mục gốc repo:

```sh
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
```

Tải hai file model đã kiểm tra, tổng khoảng 3,55 GB. Model không có sẵn trong Git:

```sh
uvx --from huggingface-hub hf download DennisHuang648/VoxCPM2-GGUF \
  VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf \
  --revision 169f64d8b98bbaab1761e4ca3a83e6af653456cc \
  --local-dir models/voxcpm2
```

Tạo `tts.voxcpm2.toml` cho máy vừa build:

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
mix = "dub-only"
```

Chuẩn bị `refs/peter.wav` dài 3–8 giây, hoặc bỏ `reference` nếu chưa có mẫu. Thêm `speech_plan = "speech.json"` khi đã chuẩn bị chỉ dẫn. Preview/dub bằng `--config tts.voxcpm2.toml`; không cần profile `artifacts` của máy thử.

Kiểm tra số các kernel GPU sau build (26 ca, đối chiếu phép tính tham chiếu trên CPU):

```sh
g++ -O2 -std=c++17 tests/native/voxcpm2_opencl_ops.cpp \
  -I runtime/llama.cpp-omni/ggml/include -L runtime/voxcpm2-build/bin \
  -Wl,-rpath,"$PWD/runtime/voxcpm2-build/bin" -lggml -lggml-base \
  -o /tmp/voxcpm2-opencl-ops
/tmp/voxcpm2-opencl-ops
```

Đổi `voxcpm2_binary` sang `runtime/voxcpm2-build/bin/voxcpm2-cli`. Không áp lại patch vào bản runtime đã vá. WSL cần `/dev/dxg` và Intel OpenCL host; sandbox không thấy thiết bị này dù host dùng được. Profile không tự chạy ngoài sandbox.

Lượt mẫu trước tích hợp: GPU tạo WAV 1,44 giây trong 26,133 giây suy luận (gồm xử lý reference), tổng process 130,074 giây. Lượt CPU trước là 57,113 giây suy luận. Đây là các lượt khác thời điểm, chưa phải benchmark kiểm soát; nạp model và cấp phát bộ nhớ vẫn tốn thời gian. Kernel đã kiểm tra độc lập trên GPU; log và kết quả tại `artifacts/voxcpm2-gpu-2026-11-25/`.
