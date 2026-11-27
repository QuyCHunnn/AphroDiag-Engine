# Handover: tự động thuyết minh và phụ đề tiếng Việt

> Tài liệu lịch sử trước khi tinh gọn quy trình. `auto` hiện chỉ nhận video tiếng Anh, tự chép lời, dùng `dub-only` và chạy tuần tự; không nhận `--srt`, `--source-lang`, `--mix`, `--pipeline-mode`. Dùng [hướng dẫn CLI hiện tại](cli-guide.md) để vận hành.

## Mục tiêu

Từ một video, tự tìm phụ đề phù hợp hoặc nhận dạng lời nói nếu không có phụ đề; dịch tiếng Anh sang tiếng Việt khi cần; tạo thuyết minh; xuất video có phụ đề và báo cáo QC. File nguồn được giữ nguyên. Kết quả video có thể mở bằng trình phát thông thường, không cần Jellyfin hay server JellyDub.

## Đã triển khai

- Thêm lệnh `jellydub auto VIDEO [--srt FILE]`. Không chỉ định SRT thì engine tìm phụ đề cục bộ/nhúng/nguồn trực tuyến theo cấu hình; nếu không tìm được phụ đề dùng ASR `faster-whisper`.
- Thêm adapter ASR tùy chọn, nhận diện tiếng Anh/Việt, tạo SRT và cache bản chép lời cùng model để dùng lại. Cấu hình có thể chọn model và thiết bị (`auto`, `cpu`, `cuda`).
- Luồng tự động nhận diện ngôn ngữ, dịch SRT tiếng Anh, bỏ qua dịch với SRT tiếng Việt, tạo giọng đọc và căn thời gian.
- Tự xuất SRT tiếng Việt, MP3 thuyết minh, MKV và MP4:
  - MKV có track audio thuyết minh tiếng Việt và track SRT tiếng Việt bật/tắt được; giữ các stream nguồn.
  - MP4 H.264/AAC có phụ đề tiếng Việt render cố định lên hình và audio tiếng Việt.
- Sinh báo cáo JSON gồm trạng thái, nguồn phụ đề, cảnh báo, các vấn đề QC, codec/track, thời lượng và dung lượng file. Báo cáo không giả định đã xác minh ngữ nghĩa bản dịch hoặc chất lượng nghe.
- Tự thử lại một lần khi TTS báo lỗi các đoạn; cache giúp tái sử dụng những đoạn đã tạo thành công.
- Khi có cặp SRT `.en.srt` / `.vi.srt` cùng tên trong cùng thư mục, có thể dùng bản tiếng Anh làm tham chiếu kiểm tra thời gian cho bản tiếng Việt.
- Bổ sung kiểm thử cho CLI, pipeline, xuất MKV/MP4 và adapter ASR; cập nhật README, cấu hình mẫu và dependency extra.

## Cách chạy

Yêu cầu Python 3.11–3.13, `uv`, FFmpeg và kết nối mạng cho tải model/Edge-TTS ở lần chạy đầu. Cài các thành phần cần cho tự động hóa:

```bash
uv sync --extra asr --extra translate
uv run jellydub doctor
```

Từ video không có SRT:

```bash
uv run jellydub auto video.mp4 --out-dir ./output
```

Với SRT đầu vào:

```bash
uv run jellydub auto video.mp4 --srt video.en.srt --out-dir ./output
```

Model ASR mặc định trong cấu hình mẫu là `small`; lần chạy đầu có thể tải model. Có thể chỉnh `asr_model` và `asr_device` trong cấu hình. Với SRT tiếng Anh, cần cài extra `translate`. Không đặt API key trực tiếp vào repo; nếu dùng backend cần key, truyền qua biến môi trường.

## Chạy thử và kết quả

### Chạy tự động không có SRT

Video mẫu “Compare vendor quotations using alternatives in RFQs in Odoo [Part 1]” được nhận dạng thành 35 cue tiếng Anh, dịch, tạo thuyết minh và xuất đầy đủ bộ file trong:

`validation/compare-vendor/auto-asr/`

Các file đầu ra có hậu tố `.vi.srt`, `.AI-Dub.vi.mp3`, `.AI-Dub.vi.mkv`, `.AI-Dub.vi.mp4` và `.AI-Dub.vi.report.json`. Báo cáo ghi trạng thái `completed`, chất lượng `needs_review`; thời lượng MKV khoảng 211.369 giây và MP4 khoảng 211.348 giây. MKV có audio và phụ đề tiếng Việt; MP4 có video H.264 và audio AAC tiếng Việt. Cả hai file đã được giải mã bằng FFmpeg thành công.

Lần chạy phát sinh lỗi nhận audio Edge-TTS ở các đoạn 1, 4, 29 và 30; engine đã thử lại toàn bộ tác vụ một lần. Chưa nghe/kiểm tra thủ công để xác nhận các đoạn này đã được phục hồi hoàn toàn.

QC đánh dấu 22 cue phụ đề dài hơn ngưỡng heuristic; đồng bộ ASR chưa được đối chiếu với SRT tham chiếu. Cần duyệt các cue này và kiểm tra đầu/giữa/cuối video trước khi bàn giao như bản hoàn chỉnh.

### Chạy với SRT tiếng Việt có sẵn

Đã chạy `auto` với phụ đề tiếng Việt trong `validation/compare-vendor/vi/`. Engine tìm thấy SRT tiếng Anh ghép cặp, khớp 38 cue với offset `+0.000s` và xuất MKV/MP4 cùng các sản phẩm khác. QC vẫn yêu cầu xem lại vì có cue dài.

## Kiểm tra đã thực hiện

- `uv run jellydub doctor`: core, translate, ASR, VieNeu và separation đều được nhận diện là đã cài.
- `uv run pytest -q`: **423 passed, 5 deselected**.
- MKV và MP4 của lần chạy ASR giải mã thành công với FFmpeg.
- `git diff --check` đã qua trước khi lập handover.

## Giới hạn còn lại

- ASR tạo transcript nhưng không đảm bảo hoàn toàn đúng; khi không có phụ đề tham chiếu, thời gian cue và độ khớp lời thoại với hình chưa được kiểm chứng độc lập.
- QC hiện là heuristic: có thể phát hiện một số cue dài và cảnh báo pipeline, nhưng không đánh giá được bản dịch đúng nghĩa hay giọng đọc tự nhiên.
- Lần chạy mẫu có cảnh báo TTS và nhiều cue dài; cần duyệt SRT và nghe các vị trí liên quan trước khi phát hành.
- Các tiêu chí kiểm tra đầu, giữa, cuối video bằng người nghe/người xem chưa được ghi nhận timestamp, vì vậy chưa thể tuyên bố nghiệm thu đầy đủ.
- Cần thử chạy lại để đo thời gian/dung lượng/cache hit nếu số liệu này là yêu cầu bàn giao; hiện mới ghi dung lượng và thời lượng trong báo cáo JSON.

## Việc tiếp theo đề xuất

1. Mở SRT đầu ra, sửa các cue dài và các câu bị cảnh báo dịch; nếu cần, so với audio gốc.
2. Nghe/đối chiếu đầu, giữa và cuối video, ghi timestamp đã kiểm tra và xác nhận không mất câu/lệch sync rõ rệt.
3. Phát MKV bằng VLC để xác nhận chọn được track tiếng Việt và bật/tắt phụ đề; phát MP4 để kiểm tra audio, font, dấu tiếng Việt, xuống dòng và vùng an toàn.
4. Chạy lại cùng lệnh để ghi thời gian xử lý lần đầu/lần sau và xác nhận cache hoạt động.
5. Sau khi duyệt, lưu bộ sản phẩm và báo cáo QC vào vị trí bàn giao; không coi trạng thái tự động `completed` là xác nhận chất lượng nội dung.

## Thay đổi trong worktree

Các thay đổi hiện gồm mã pipeline/CLI/cấu hình/xuất file, adapter ASR, test, tài liệu và lockfile. Ngoài ra còn có thư mục dữ liệu chạy thử (`validation/`, `videos/`, `dubbed/`, `demo-reel/`) và script. Worktree có thay đổi sẵn; cần xem `git status --short` và chọn lọc nội dung cần commit, không xóa hoặc hoàn tác các file ngoài phạm vi.

## Bổ sung bàn giao ngày 2026-11-13

Bộ mẫu được tập trung tại `delivery/vendor-quotations/`, dùng SRT Việt 38 cue đã đối chiếu với SRT Anh ở mức văn bản. Không dùng nhánh ASR để thay thế bản mẫu này. Chi tiết thuật ngữ, số liệu và giới hạn duyệt tại `evidence/subtitle-review.md`.

- Có script `scripts/complete_handover.py VIDEO VI_SRT OUTPUT`: chạy CLI hai lần, lưu lệnh đầy đủ, log, thời gian, thay đổi cache, hash nguồn/đầu ra, ffprobe và full decode mọi audio/video track.
- Lần đo thứ nhất với giọng có sẵn trong cache: **178.455 giây**; lần chạy lại: **8.184 giây**, không tạo hoặc sửa file cache. Đây không phải phép đo cold-start từ dịch/TTS hoàn toàn mới. Lần đo trước bị gián đoạn không được dùng làm số liệu hoàn thành.
- Hash video và SRT nguồn trước/sau giống nhau. MKV, MP4 và MP3 trước bước định dạng phụ đề đã full decode thành công.
- Tái sử dụng đủ 38 đoạn TTS theo khóa khớp provider/voice/tham số/nội dung/người nói; bằng chứng `evidence/cache-reuse.json`. Kế hoạch timing từ thời lượng giọng: đủ 38/38 cue, không shift, không warning, tempo tối đa khoảng 1.175. Đây là kiểm tra tự động, không thay thế nghe.
- `scripts/package_reviewed_sample.py` đóng gói lại mẫu với SRT xuống dòng tối đa 42 ký tự, tối đa ba dòng/cue; giữ nguyên lời và timing. Kết quả cuối, full decode, track metadata và ảnh trích xuất được ghi riêng trong `evidence/packaging.json`; chỉ coi bước này xong khi script exit 0 và báo cáo tồn tại.
- `jellydub doctor` nhận diện FFmpeg/ffprobe, core, translate, ASR, VieNeu, separation. Đã chạy lại nhóm test export/pipeline/CLI/transcribe: **159 passed in 10.95s**. Đây là kiểm tra mới theo phạm vi, không phải chạy lại toàn bộ suite 423 test.
- Hướng dẫn chạy video mới và hai đề xuất tiếp theo kèm công sức/cách kiểm chứng nằm tại `delivery/vendor-quotations/README.md`.
- `delivery/engine-changes.patch` và `delivery/SOURCE-SETUP.md` giúp tái lập phần source chưa commit trên commit gốc `465f566657ee0fbc0619dbd3f960a283aa9a0571`; bản vá đã được kiểm tra áp dụng trên source sạch.

**Trạng thái nghiệm thu:** còn cần nghe/xem thực tế các khoảng 00:04–00:18, 01:47–02:04, 02:55–03:23 và câu cuối đến 03:26.920; xác nhận audio Việt, bật/tắt SRT trong VLC và phát trên máy nhận không kết nối engine. Chưa có người vận hành xác nhận các bước này. Chưa đo cold-start đầy đủ hoặc có người khác chạy độc lập trên video mới. Không tự chuyển các mục này sang đạt từ kết quả test/FFmpeg.

Bước đóng gói mẫu đã hoàn tất và xác minh exit 0: mất **194.151 giây**; MKV **19,193,373 byte**, MP4 **16,402,022 byte**. Hai file cuối full decode thành công; metadata/track đúng, SRT nhúng đủ 38 cue với offset đồng nhất **+21 ms** so với file rời (ngưỡng kiểm tra 25 ms, không có drift giữa các cue). Đã xem ảnh MP4 cuối tại 00:08, 01:50, 03:20 và 03:21: dấu tiếng Việt và khung chữ hiển thị đủ ở các ảnh này. Chưa xem phát liên tục hoặc nghe. Xem báo cáo cuối `delivery/vendor-quotations/RESULTS.md`.
