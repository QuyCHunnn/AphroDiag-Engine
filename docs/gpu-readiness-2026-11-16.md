# Kiểm tra trước khi chạy lại pipeline — 2026-11-16

## Kết quả thực thi

- WSL có `/dev/dxg`, `/dev/dri/card0`, `/dev/dri/renderD128`.
- OpenVINO: Intel Graphics PCI 0x9a49 (Iris Xe iGPU); CPU i7-11390H. Không thấy CUDA: CTranslate2 CUDA count = 0, không có lệnh nvidia-smi. Đây là thiết bị hệ điều hành/runtime đang nhìn thấy, chưa chứng minh nhận thêm GPU rời.
- PyTorch 2.14.1+cpu: CUDA unavailable. OPUS auto chọn CPU; chưa có adapter dịch Intel GPU.
- Ma trận tính thật GPU.0, đối chiếu NumPy: pass.
- ASR mẫu 20 giây, Whisper Small, beam=5, ép OpenVINO GPU: pass, 52.98 giây gồm load/compile. Encoder và decoder GPU.0, không fallback. Số đo một lần này không thể ngoại suy tốc độ video 3 phút.
- Edge-TTS NamMinh: câu 1 pass 1.55 giây; câu 2 lỗi NoAudioReceived sau 3.61 giây. Các câu chạy tuần tự; lỗi vẫn tái hiện ngay cả không chạy nhiều request đồng thời. Đây là kiểm tra trực tiếp provider, không gồm retry cấp pipeline.

## Trạng thái pipeline

| Bước | Thiết bị / trạng thái |
|---|---|
| Chép lời | Intel GPU hoạt động trên mẫu ngắn; lỗi timestamp bài 3 phút chưa sửa |
| Dịch OPUS | CPU với môi trường hiện tại |
| Tạo giọng Edge | Dịch vụ mạng; lỗi NoAudioReceived còn tái hiện |
| Căn thời gian / trộn tiếng | FFmpeg/CPU |
| Xuất MKV + SRT | Copy luồng hình, không cần GPU encode |

Kiểm tra VAD trên video 180 giây thấy các vùng 0–0.59, 3.79–88.62, 90.45–92.59, 94.32–133.68, 137.20–140.50, 141.94–180 giây. Vùng dài ~85 giây là đầu mối kiểm tra đường timestamp dài, chưa phải nguyên nhân đã được xác nhận.

## Điều kiện để chạy lại đầy đủ

1. Chẩn đoán và sửa timestamp ASR, kiểm chứng trên đúng đầu vào 180 giây; giữ beam=5, không bỏ câu lỗi.
2. Xử lý lỗi Edge-TTS hoặc chọn backend thay thế đã đo tốc độ/chất lượng. Giảm concurrency đơn thuần chưa đủ bằng chứng khắc phục vì request tuần tự cũng lỗi.
3. Giữ đầu ra MKV + SRT, nguồn nguyên vẹn; tách lần không cache xử lý với lần dùng lại cache. Model đã tải sẵn cần ghi rõ.

Chưa chạy lại toàn bộ pipeline 3 phút trong lượt kiểm tra này. Chưa thay backend, driver hay cấu hình GPU của engine.

Bằng chứng: `artifacts/gpu-readiness-2026-11-16/readiness.json`; script `check.py` cùng thư mục. Bài lỗi trước: `artifacts/optimization-checkpoints/benchmark-180s/` và `benchmark-180s-en/`.
