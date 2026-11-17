# Delta Force — video tiếng Việt hoàn chỉnh

Nguồn: `videos/YTSave.to_YouTube_Media_W39avTeRKjs_Delta-Force-ALL-Operators-Menu-Animations-Voicelines_005_360p.mp4`, 189.800 giây, 640×360. Đầu ra nằm ở `delivery/delta-force/`, chỉ MKV và SRT.

## Quy trình đã thực hiện

1. Whisper Small, OpenVINO Intel GPU, beam=5: 16 câu, 65.13 giây gồm load/compile; encoder và decoder thực thi trên GPU.0, không fallback CPU.
2. OPUS tự dịch Anh → Việt. Giữ bản dịch tự động ở `artifacts/delta-force/translated/`. Bản tự động có lỗi nghĩa game/thành ngữ; đã biên tập 16 câu ở `delta-force.reviewed.vi.srt`, lưu đối chiếu trong `translation-review.json`. Hai câu Uluru/Gizmo được rút gọn để giọng không vượt thời gian phụ đề.
3. Gán người nói theo tên nhân vật hiện trên video. Đây là mapping cảnh đã kiểm tra cho đúng video này, không phải chức năng tự nhận diện giới tính/diarization cho mọi video. Cấu hình `speaker-map.json` và `settings.toml`.
4. Edge-TTS Andrew Multilingual (nam) cho D-Wolf, Vyron, Nox, Stinger, Shepherd, Uluru, Sineva, Gizmo, Raptor; Emma Multilingual (nữ) cho Tempest, Toxik, Vlinder, Luna, Hackclaw. Tổng 10 câu nam, 6 câu nữ. Không clone giọng nhân vật gốc. Hai giọng mặc định Việt Nam lỗi trong phép thử; hai giọng thay thế tạo được đầy đủ 16 câu.
5. Căn thời gian, trộn voiceover giữ âm thanh/hiệu ứng nguồn và xuất MKV. Giữ track gốc, thêm audio Việt và subtitle Việt mặc định. Xuất thêm SRT rời.

## Bản sửa engine

- VAD ngắn từng cắt một đoạn còn 0.592 giây nhưng Whisper trả timestamp bắt đầu 1 giây. Chỉ với đoạn dưới một giây: bổ sung ngữ cảnh tới ba giây nếu khoảng cách với đoạn bên cạnh cho phép; các cửa sổ không chồng nhau, không bỏ vùng speech. Vẫn từ chối interval sai nếu model tiếp tục lỗi, không âm thầm bỏ câu. Cache ASR tăng phiên bản để không dùng bản chép lời cũ.
- Kiểm chứng lại đúng video từng lỗi: pass trên GPU, beam=5, 30 cue, 131.58 giây. Đây là kiểm tra kỹ thuật, chưa có nghiệm thu nội dung cho video cũ.
- Thêm `[jellydub.speaker_voices]` cho Edge: chọn giọng theo nhãn người nói trong SRT; cache phân biệt cấu hình giọng. Không có nhãn thì dùng `voice` mặc định. Ví dụ:

```toml
[jellydub]
tts = "edge"
voice = "en-US-AndrewMultilingualNeural"

[jellydub.speaker_voices]
LUNA = "en-US-EmmaMultilingualNeural"
RAPTOR = "en-US-AndrewMultilingualNeural"
```

Lỗi dịch vụ Edge-TTS chưa thể sửa triệt để từ engine. Không tự chuyển giọng ngầm: giọng dùng cho lần bàn giao này được cấu hình rõ. [Upstream issue về NoAudioReceived](https://github.com/rany2/edge-tts/issues/512).

## Tái lập bản đã duyệt

Từ thư mục repo:

```bash
.venv/bin/python scripts/rebuild_delta_force.py
```

Script kiểm tra SHA-256 nguồn trước khi áp dụng bản dịch/mapping của video này. Sử dụng lại cache, tạo mới phần còn thiếu. Xóa cache sẽ cần mạng để tạo lại giọng. Script dựng lại bản đã duyệt; không tuyên bố chạy lại bước dịch hoặc nhận diện nhân vật.

Bằng chứng các bước từ nguồn: `artifacts/delta-force/transcribe_probe.py`, `prepare.py`, `translate.log`, `review_translation.py`, `dub-final.log`. Đây là workflow có bước biên tập; muốn xử lý video khác tự động bằng CLI dùng [cli-guide.md](cli-guide.md), rồi duyệt transcript/bản dịch/mapping mới.

## Kiểm tra bàn giao

- 16/16 câu có clip giọng, không cảnh báo lệch/chồng âm thanh; các clip kết thúc trong thời gian phụ đề.
- Hash video stream nguồn và MKV giống nhau. SHA-256 nguồn lưu trong `verification.json`.
- Audio Việt và SRT Việt có language `vie`, default=1; audio gốc default=0.
- Toàn bộ luồng hình/âm thanh giải mã bằng FFmpeg không lỗi.
- Render phụ đề kiểm tra tại 00:04, 01:40.3, 02:56.7: tiếng Việt đúng dấu và không bị cắt chữ. Ảnh `subtitle-start.jpg`, `subtitle-middle.jpg`, `subtitle-end.jpg` chỉ là kiểm tra render, không phải quay lại từ VLC.
- Bộ kiểm thử: 445 passed, 5 deselected, 3 warnings, 64.98 giây; sau chỉnh sửa cuối, 110 test liên quan tiếp tục pass. Các test model/network e2e tự động bị bỏ qua theo mặc định repo; file đầu ra đã kiểm tra thực tế riêng.
- Chưa nghe nghiệm thu thủ công bằng VLC. ASR có những từ không chắc dưới hiệu ứng game; cần nghe lại nếu dùng làm bản dịch chính thức (đặc biệt câu Stinger, Uluru, Sineva). Không có số đo end-to-end tự động sạch vì lượt này có bước review và tạo lại hai câu.

Mở MKV bằng VLC, chọn audio **Tiếng Việt (AI Dub)** nếu player không tự chọn, bật/tắt **Phụ đề tiếng Việt**. File mở độc lập, không cần engine/server chạy kèm.
