# Plan tối ưu JellyDub theo checkpoint

Phạm vi: engine và CLI; chưa làm giao diện hoặc đóng gói ứng dụng. Người dùng hướng tới một lệnh `jellydub auto VIDEO`, mặc định nhận MKV + SRT. GPU ưu tiên cho model có backend tương thích; CPU fallback giữ khả năng sử dụng. Video được stream-copy, không encode lại để ép dùng GPU.

| Checkpoint | Nội dung | Điều kiện qua | Trạng thái |
|---|---|---|---|
| CP0 | Audit baseline | Số đo stage/CPU/cache và kiểm chứng GPU | Đã xong; xem performance-audit-2026-11-16.md |
| CP1 | Đầu ra MKV + SRT | Mặc định không MP4/MP3, WAV→AAC trực tiếp, QC lưu nội bộ; test và media thật đạt | Đã đạt: 160 test; media thật/hash/track/decode đạt (artifacts/optimization-checkpoints/cp1) |
| CP2 | Cache và QC | Đổi output/font không tạo lại giọng; sửa một câu chỉ TTS câu đó; WAV metadata nhanh; file hỏng bị phát hiện; benchmark trước/sau | Đã đạt: hồi quy 432 test, test cache/corruption; warm 1.34s, 4 probe, đổi output 1.52s; media/hash/decode đạt |
| CP3 | GPU ưu tiên/fallback | Chọn thiết bị đúng backend; Intel chạy cấu hình đã kiểm chứng; lỗi GPU fallback CPU, không âm thầm đổi beam/giọng/chất lượng | Đã đạt: Small beam=5, encoder/decoder GPU.0; lỗi GPU tiêm có kiểm soát → suy luận CPU thật; 174 test phạm vi |
| CP4 | CLI/hướng dẫn và hồi quy | Một lệnh thông thường; hướng dẫn setup ngắn; test phạm vi và các luồng EN/VI/ASR, nguồn giữ nguyên | Đang triển khai |

Thực hiện tuần tự, báo kết quả từng checkpoint và tiếp tục khi kiểm tra đạt, không cần hỏi xác nhận lại. Giữ bằng chứng benchmark cũ và dữ liệu bàn giao; không xóa file nguồn hoặc tự commit/deploy.

CP1 giữ MP3/MP4 dưới dạng opt-in cho các luồng kỹ thuật cũ. CP2 phân biệt cache giọng, timing và export; cache QC phải gắn với nội dung/version, không chỉ sự tồn tại file. CP3 không coi CUDA=False là không có Intel GPU; giới hạn OpenVINO beam=5 phải được phản ánh rõ và không che bằng giảm beam. ASR/dịch/TTS online có benchmark riêng; không lấy mức giảm thời gian media để cam kết cho mọi video.

Các giới hạn cần ghi nhận khi bàn giao: Edge-TTS cần mạng cho câu chưa cache; thời gian phụ thuộc máy; test/FFmpeg không thay thế nghe nghiệm thu. Trạng thái hỗ trợ hệ điều hành/GPU chỉ dựa trên cấu hình đã kiểm chứng thực tế.
