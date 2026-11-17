# Intel Iris Xe trong WSL — thiết lập và kiểm chứng

Đã thiết lập ngày 2026-11-16 trên máy hiện tại. Không thay dependency hoặc backend trong `.venv` của JellyDub.

## Trạng thái xác minh

- Windows: Intel Iris Xe Graphics, PCI `8086:9A49`, driver `32.0.101.7077`, trạng thái OK.
- WSL 2.7.14.0, Ubuntu 24.04; `/dev/dxg` và thư viện Direct3D đã tồn tại trước khi cài runtime.
- Cài mới từ Ubuntu: `intel-opencl-icd`, `clinfo`, `libze1` và dependency; không nâng cấp/gỡ gói sẵn có.
- `clinfo -l` nhận Intel OpenCL Graphics / Intel Graphics `[0x9a49]`.
- Môi trường riêng `.venv-igpu`: OpenVINO 2025.4.1. Phiên bản dependency tại `artifacts/igpu-setup/requirements.lock.txt`.
- Ba phép nhân ma trận float32 kích thước 256, 512, 1024 đều chạy trên `GPU.0`; kết quả so với NumPy đạt `rtol=1e-3, atol=1e-4`. Chỉ định thiết bị `GPU`, không dùng `AUTO` hoặc fallback CPU.
- Số đo cụ thể tại `artifacts/igpu-setup/verification.json`. Đây là smoke test phần cứng/runtime, không phải benchmark Whisper/dịch/TTS hay bằng chứng toàn pipeline nhanh hơn.

## Chạy lại

Trong terminal Ubuntu WSL thông thường, từ thư mục repo:

```bash
clinfo -l
.venv-igpu/bin/python scripts/check_igpu.py
```

Sandbox giới hạn của agent có thể che `/dev/dxg`; khi dùng agent, phép kiểm tra GPU cần chạy ngoài sandbox với quyền được cho phép. Không kết luận WSL thiếu GPU chỉ từ việc không thấy thiết bị trong sandbox.

`torch.cuda.is_available()` là kiểm tra CUDA, không phải kiểm tra tổng quát Intel GPU. `False` không chứng minh Iris Xe không hoạt động.

## Tái lập môi trường thử nghiệm

Trên cùng loại hệ thống WSL và driver Windows đã hỗ trợ GPU passthrough:

```bash
sudo apt-get install --no-install-recommends intel-opencl-icd clinfo libze1
python3 -m venv .venv-igpu
.venv-igpu/bin/python -m pip install -r artifacts/igpu-setup/requirements.lock.txt
.venv-igpu/bin/python scripts/check_igpu.py
```

Không cài Linux kernel/display driver i915/DKMS cho bước này. Thư viện compute phía Ubuntu sử dụng GPU được Windows cung cấp qua WSL. Không cần restart trên máy này; chưa thay cấu hình khởi động hoặc driver Windows.

## Ranh giới với JellyDub

JellyDub hiện chưa tích hợp OpenVINO. Backend faster-whisper/CTranslate2 của repo chưa có lựa chọn Intel OpenVINO; model dịch PyTorch chọn CPU/CUDA; Edge-TTS tạo giọng trên dịch vụ trực tuyến. Chưa chỉnh các backend này hoặc bật GPU toàn pipeline.

Các bước tiếp theo nếu triển khai tăng tốc: chọn backend Intel tương thích cho ASR/dịch, đo cùng model/cùng dữ liệu trên CPU và GPU, so nội dung/thời gian/tải tài nguyên. Với mục tiêu chỉ MKV + SRT, video nên stream-copy; không cần GPU encode video. Bản profiling cũ đang dừng theo yêu cầu, chưa có kết luận benchmark hoàn chỉnh.

## Tài liệu nguồn

- Intel OpenVINO GPU/WSL setup: https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html
- Intel Compute Runtime, nền tảng được hỗ trợ: https://github.com/intel/compute-runtime

## Kết quả thử ASR thực tế ngày 2026-11-16

Đã cài thêm OpenVINO GenAI 2025.4.1.0 trong `.venv-igpu` và tải model Whisper Tiny/Small INT8 công khai. Thử ba đoạn âm thanh 20/40/60 giây trên CPU và GPU, chạy tuần tự với cùng cấu hình mỗi phép so sánh.

- Tiny beam=1 chạy được cả hai device; GPU không nhanh hơn nhất quán ở các mẫu này.
- Small beam=1 chạy được; GPU có lợi ở mẫu 40/60 giây, chưa tính load/compile, nhưng không có lợi ở mẫu 20 giây.
- Small beam=5 chạy được CPU nhưng GPU lỗi `IRemoteTensor: Not Implemented`, tái hiện hai lần. Không hạ beam âm thầm hoặc bật backend này mặc định cho engine.
- Chưa có backend OpenVINO production trong engine; đây là cấu hình môi trường và thử nghiệm riêng. Báo cáo số liệu/giới hạn: [performance-audit-2026-11-16.md](performance-audit-2026-11-16.md).
