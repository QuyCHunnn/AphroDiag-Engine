# Kế hoạch tối ưu tiếp: video 3 phút trong 40–60 giây trên Iris Xe

Ngày: 2026-11-17. Trạng thái: kế hoạch, chưa triển khai các checkpoint N0–N6 bên dưới.

Cập nhật thực thi: đã bắt đầu N0 và spike N1; xem `next-optimization-execution-2026-11-17.md` cùng các report trong `artifacts/next-optimization/`. N0/N1 chưa được coi hoàn thành toàn bộ chỉ vì đã có một số lượt benchmark. Giữ nguyên các gate của kế hoạch dưới đây.

## 1. Mục tiêu và phạm vi

Mục tiêu do người dùng đặt: video 180 giây xử lý trong 40–60 giây khi dùng GPU; Iris Xe là máy đối chứng chính, sau đó kiểm tra khả năng dùng GPU tầm thấp/trung. Được phép thay kiến trúc engine/pipeline. Giữ giọng VieNeu hiện tại làm đối chứng; đổi model/backend là nhánh thử nghiệm và cần đánh giá chất lượng.

Đầu vào mục tiêu: video EN chưa có phụ đề; tự ASR → dịch VI → TTS → căn thời gian → voiceover → MKV + SRT. Model đã tải sẵn. Không dùng ledger hoặc bản dịch soạn trước trong benchmark tự động. VI SRT có sẵn là workload riêng, không dùng nó để xác nhận mục tiêu EN end-to-end.

Đề xuất tiêu chí tốc độ: median ≤60 giây, ít nhất 4/5 lượt ≤60 giây trên mỗi clip 180 giây của bộ nghiệm thu, processing cache rỗng mỗi lượt, worker/model đã sẵn. 40 giây là mục tiêu mở rộng. Đo thêm process lạnh và lần đầu compile; chưa cam kết cold process cũng ≤60 giây. Đây là định nghĩa đề xuất cần chốt, không phải yêu cầu đã được người dùng xác nhận.

Thời điểm bắt đầu: nhận yêu cầu xử lý video. Kết thúc: MKV/SRT được ghi atomic, probe/kiểm tra bắt buộc hoàn tất, job được báo completed. Không báo completed trước QC. Full decode bổ sung trong nghiệm thu đo riêng, không âm thầm bỏ các QC đang có để làm đẹp số.

Không có cam kết đạt 40–60 giây trước khi đo. Nếu backend/model giữ nguyên không đáp ứng, báo số đo tốt nhất và lựa chọn đánh đổi cụ thể.

## 2. Baseline đã có và khoảng cách

| Workload | Kết quả | Giới hạn |
|---|---:|---|
| Family Guy 156,665s, automatic EN→VI, native ASR GPU, TTS INT8 CPU, tuần tự | 312,09s | 1 lượt; một preset |
| Cùng workload, overlap đã sửa | 209,60s | 1 lượt; chưa fixed-seed; chưa nghe duyệt INT8 |
| SRT/decode/video stream | 41 cue giống hash; decode đạt; hình giữ nguyên | Không xác nhận nghĩa/giọng |
| ASR native Small/beam=5 riêng, toàn Family Guy | 82,23s | Nhanh hơn Optimum nhưng riêng ASR đã vượt 60s |
| ASR native chạy lại từ cache | 3,17s | Không đại diện video mới |

Nguồn: `artifacts/checkpoints-2026-11-17/overlap-comparison.json`, `native-cli/report.json`, `docs/checkpoints-2026-11-17.md`.

**Hệ quả:** tối ưu dịch/TTS/overlap mà giữ thời gian ASR hiện tại sẽ không đạt ≤60s. ASR phải có checkpoint riêng. Từ 209,60s tới 60s cần khoảng 3,49× trên chính clip hiện có; không ngoại suy tuyến tính sang clip 180s có mật độ thoại khác.

Chưa có profile critical path đủ chi tiết của overlap: các stage chồng nhau nên không cộng thời gian stage để suy wall time. N0 phải xác định thời gian chờ queue, load, compute và contention.

## 3. Thứ tự và điều kiện ra quyết định

N0 → N1 → N2 → N3 → N4 → N5 → N6. N1 là spike dịch OpenVINO theo đề xuất đã trao đổi; N2 xử lý giới hạn ASR bắt buộc. Không cần chờ người dùng duyệt những tối ưu giữ chất lượng để tiếp tục; chỉ chờ khi nhánh phụ thuộc lựa chọn đánh đổi chất lượng chưa được chốt.

Không bật thêm tính song song mù quáng trên Iris. Thử nghiệm các cấu hình tuần tự trước, rồi chạy cùng scheduler để đo tranh CPU/GPU/RAM. Thay đổi backend chỉ thắng microbenchmark chưa đủ để đổi mặc định.

### N0 — Chuẩn hóa baseline và bộ đo

**Việc làm**

- Chọn 3 clip 180s: hướng dẫn một người nói; hội thoại nhiều vai; nói nhanh kèm tiếng hét/nhạc. Giữ Family Guy 156,665s làm regression, không gọi nó là clip 3 phút.
- Ghi hash video, SRT/reference, model revision, runtime lock, cấu hình beam/precision/voice, nguồn dữ liệu và mật độ lời nói.
- Thêm trace theo job/region/batch: import/load/compile, inference, normalize, queue wait, thời điểm có câu đầu, tail sau ASR, export/QC. Ghi process RSS, peak RAM của cây process; GPU measurement nếu khả dụng. Nếu chỉ có device identity thì không gọi đó là GPU utilization.
- Chia 3 mode: cold process + cache xử lý rỗng; worker/model sẵn + cache xử lý rỗng; processing cache hit. Tách model download/compile đầu tiên.
- Mỗi A/B đủ clip chạy ≥3 cặp theo thứ tự cân bằng; nghiệm thu cuối 5 lượt mỗi clip. Process riêng cho cold, cache riêng mỗi variant, không chạy test/benchmark khác đồng thời. Ghi tải nền không kiểm soát được.

**Đầu ra:** benchmark runner có manifest/trace, baseline median/min/max và tập audio/transcript đối chứng. Script từ chối nhầm thư mục cache cũ.

**Qua checkpoint:** mọi cue có thể truy vết; không missing/duplicate; decode/hình/source đạt; xác định được critical path. Chưa có bộ reference thì vẫn đo hiệu năng nhưng đánh dấu quality chưa đủ bằng chứng.

### N1 — Prototype dịch OPUS bằng OpenVINO

**Việc làm**

- Giữ đúng model/revision OPUS hiện tại; thử export tokenizer + encoder/decoder, KV-cache và generation tương thích. Trước tiên xác nhận API/export hỗ trợ, không coi đổi tên device là đã có backend.
- So PyTorch CPU với OpenVINO CPU và Iris GPU. Giữ num_beams=4, repetition guard, giới hạn token, context pairing hiện tại. Đo cả load và warm inference.
- Thử batch 1/4/8/16 theo độ dài token, giữ mapping về cue/speaker; bounded wait để không đợi đủ batch quá lâu. Không ghép ngữ cảnh xuyên người nói chỉ để tăng batch.
- Cache phân biệt backend/model revision/precision/generation policy khi output có thể khác. Kiểm tra sửa một câu chỉ invalidation dependency đúng.
- Đo standalone rồi thử cùng ASR trên Iris. Nếu tranh GPU làm critical path xấu đi, giữ dịch CPU hoặc chạy theo lịch phân đoạn.

**Qua checkpoint:** không mất/đảo câu; đủ EN→VI; thuật ngữ, tên, phủ định, số không sai mới so reference; backend đạt chất lượng. Mức tăng ≥15% warm inference là gate chọn ứng viên; chỉ tích hợp khi end-to-end median cải thiện >5% và biến thiên không phủ nhận lợi ích. Các mức này là ngưỡng kỹ thuật đề xuất.

**Dừng nhánh:** export/generation không tương thích, sai nghĩa, hoặc chỉ nhanh khi đứng riêng nhưng toàn pipeline chậm. Ghi số đo và giữ backend cũ.

### N2 — Giảm critical path ASR trên Iris

**Việc làm giữ model/beam trước**

- Pin native GenAI nightly đã chạy được; kiểm tra khả năng chuyển sang stable có cùng hành vi trước khi coi production-ready. Giữ worker native sẵn qua nhiều job, tái sử dụng compile cache đúng model/device/runtime.
- Profile VAD/context regions, token generation, phần load và đồng bộ. Đo gộp các vùng ngắn có giới hạn so với giải mã hiện tại; giữ offset/timestamp và guard tiếng hét, tránh thêm quá nhiều audio im lặng.
- Thử precision/runtime properties theo backend hỗ trợ và có A/B chất lượng; không tự giảm beam, model hoặc guard.
- So transcript/reference, missing words, mốc đổi người nói, timestamp drift, lỗi lặp và token-cap truncation. Dùng cùng audio đã giải mã để tránh đo lẫn khác preprocessing.

**Điều kiện cần:** warm ASR toàn clip 180s phải thấp hơn 60s; ngân sách sơ bộ mong muốn ≤25–30s để còn tail dịch/TTS/export. Ngân sách này là mục tiêu thiết kế, chưa có bằng chứng khả thi.

**Nếu không đạt:** mở nhánh model ASR nhỏ/distilled hoặc beam thấp hơn thành profile riêng; trình mẫu sai khác và quality score. Không nhập nhánh giảm chất lượng vào baseline Small/beam=5. Việc người dùng cho đổi kiến trúc không đồng nghĩa đã chấp nhận giảm chất lượng transcript.

### N3 — TTS batch và lựa chọn engine phù hợp Iris

**Nhánh A: cùng VieNeu/giọng**

- Xác nhận SDK có batch thực trên Intel/OpenVINO hay chỉ tuần tự. SDK batch CUDA không được coi là Intel đã hỗ trợ.
- Prototype batch 1/2/4/8 câu cùng voice/reference và độ dài gần nhau; đo frames/s, tổng audio/s, RAM, thời gian câu đầu/tail. Batch gồm các câu đã sẵn, không chờ video kết thúc.
- Bảo toàn owner mapping qua split/chunk/join; guard hủy, silent audio, timeout và retry chỉ batch/câu lỗi; cache theo từng câu, không mất reuse khi một speaker đổi.
- Hoàn tất A/B đủ 44 câu FP32/INT8, sample nghe Peter/Lois/Stewie và lời thoại dài. Precision cache tách biệt. Khác hash cần nghe và so nội dung, không dùng hash làm tiêu chí duy nhất cho backend số học khác.

**Nhánh B: model/backend khác**

- Chỉ mở nếu nhánh A không đạt throughput cần thiết. Chọn tối đa 2 ứng viên có export/runtime thực trên Iris và hỗ trợ VI; xác minh license, weights, voice/reference và yêu cầu RAM trước khi tải/port lớn.
- Dùng nguyên bộ text/reference, không chọn riêng câu dễ. So phát âm, rơi từ/thêm từ, chất giọng, tuổi/cảm xúc, timing và tốc độ. Không dùng dịch vụ cloud để tuyên bố đã tăng tốc GPU local.

**Qua checkpoint:** nghe đạt, không rơi/thêm câu; end-to-end tốt hơn. Nếu đổi chất giọng/model thì người dùng duyệt sample trước bulk. Bản CPU tối ưu vẫn là fallback; không bật đường GPU chậm hơn chỉ vì có GPU. Prototype heads/frame GPU trước đây đã chậm hơn và không được lặp lại nguyên trạng nếu chưa có giả thuyết mới.

### N4 — Worker/session bền và scheduler theo tài nguyên

**Việc làm**

- Một manager + worker ASR/dịch/TTS có lifecycle rõ: starting/ready/busy/failed/stopping. Model load một lần, kiểm tra health, restart có giới hạn, release khi idle hoặc thiếu RAM. Giữ CLI chạy một job không phụ thuộc daemon nếu không cần.
- Queue giới hạn theo số cue và tổng độ dài/token/audio dự kiến, không chỉ số message. Ghi backpressure và queue age; bảo toàn index/speaker/context/order.
- So các lịch: ASR GPU + dịch/TTS CPU; ASR/dịch chia GPU theo lượt; batch TTS khi có đủ câu. Chỉ dùng lịch được đo trên Iris, tránh oversubscription BLAS/ORT/Torch.
- Mỗi job/config có generation ID; cancel job chỉ dừng đúng job, không trả kết quả cũ sang job mới. Khóa/reset state riêng của backend; test hai video liên tiếp khác giọng/ngôn ngữ, một video lỗi hoặc hủy giữa chừng.
- Cache ghi atomic/hash, retry typed, resume chỉ phần chưa đạt. Phân biệt fallback dự kiến với lỗi scheduler bị che.

**Qua checkpoint:** 2–3 job liên tiếp giữ model nhưng không tăng RAM vô hạn, không lẫn giọng/kết quả; cancel và crash recovery đúng; throughput end-to-end có lợi. RAM budget phải chốt ở N0, không mặc định model nào cũng có thể giữ cùng lúc trên máy RAM 8GB.

### N5 — Giảm tail normalize/mix/export sau khi compute đã giảm

**Việc làm**

- Đo process spawn từng cue, decode/trim/resample; chỉ thay bằng in-process hoặc batch nếu giữ sample rate/trim/loudness/timing và xử lý file lỗi đúng.
- Chuẩn bị original PCM song song khi tài nguyên cho phép; pipeline buffer/streaming mix chỉ sau khi đảm bảo không cắt speech vượt slot.
- Giữ video stream-copy, AAC track và SRT bật/tắt. Không encode lại hình để ép dùng GPU. Đo mux/audio encode và QC riêng.

**Qua checkpoint:** hình/source hash không đổi, audio không cắt/mất, duration/track/loudness/timing đạt; wall time cải thiện. Không bỏ full-decode nghiệm thu hoặc QC lỗi để giảm số đo.

### N6 — Nghiệm thu 180s và chốt profile

- Chạy 3 clip 180s + Family Guy regression; 5 lượt/profile/clip với cache xử lý rỗng. Công bố cold và warm worker riêng, cache-hit riêng.
- Báo median/min/max, tỷ lệ lượt ≤60s, peak RAM, device/provider thực, load/compile, cue count, fallback và critical path. Không lấy trung bình toàn bộ clip để che clip fail.
- So chất lượng reference đã nghe duyệt; nghiệm thu translation/voice/timing. Nhiều vai tự động chưa có diarization thì ghi rõ một preset; benchmark nhiều vai thủ công là profile khác.
- Nếu đạt: cập nhật CLI/config/doctor/hướng dẫn setup, runtime locks, fallback và regression checks. Giữ upstream đối chứng độc lập, không tự commit/deploy.
- Nếu chưa đạt: báo thời gian tốt nhất, điểm nghẽn còn lại, chất lượng từng profile và bằng chứng giới hạn phần cứng/backend. Không công bố đạt dựa trên thời gian dự phóng hoặc cache hit.

## 4. Những điểm cần làm rõ

| ID | Cần chốt | Đề xuất tạm dùng | Ảnh hưởng và thời điểm cần câu trả lời |
|---|---|---|---|
| Q01 | 40–60s gồm nạp/compile model hay worker sẵn? | Model tải sẵn; mục tiêu chính là worker sẵn, cache xử lý rỗng; cold báo riêng | Quyết định giữ worker bền và tiêu chí N6. Chốt trước nghiệm thu tốc độ |
| Q02 | Mốc 60s là median hay mọi lượt? | Median ≤60s, ≥4/5 lượt ≤60s cho từng clip | Quyết định cách công bố năng lực; chốt N0/N6 |
| Q03 | Có được đổi ASR model/beam để lấy tốc độ? | Giữ Small/beam=5 trong nhánh không đổi chất lượng; nhánh khác opt-in riêng | Nếu N2 không đạt thì cần quyết định này trước tích hợp nhánh giảm compute |
| Q04 | INT8 có giữ chất lượng đủ tốt? | Chưa coi đạt; so/nghe sample cùng text, giọng, context | Chặn đổi INT8 thành mặc định; không chặn đo benchmark |
| Q05 | Có chấp nhận đổi chất giọng khi model TTS khác nhanh hơn? | Giữ VieNeu làm baseline; model khác cần duyệt sample | Chặn N3 nhánh B bulk/mặc định, không chặn khảo sát |
| Q06 | Đầu vào nghiệm thu chủ yếu hội thoại dày hay tutorial? | Bộ 3 clip 180s có cả hai, ghi tỷ lệ speech/audio | Không suy tốc độ từ độ dài video đơn thuần; chốt bộ clip N0 |
| Q07 | Một giọng hay nhiều vai tự nhận diện? | Mục tiêu tự động ban đầu một preset; multi-voice có nhãn benchmark riêng | Nếu bắt buộc diarization tự động phải thêm scope và đo chi phí; chưa coi việc gán ledger là tự động |
| Q08 | Ngân sách RAM và worker idle? | Dùng máy Iris hiện tại khoảng 8GB; đo tree RSS; chưa đặt auto giữ mọi model | Quyết định batch/model residency N3/N4; cần số RAM tối thiểu dự kiến triển khai |
| Q09 | Dữ liệu ground truth cho nghĩa/ASR/giọng do ai duyệt? | Chuẩn bị sample/SRT đối chiếu để người dùng nghe/duyệt; test kỹ thuật không thay nghiệm thu | Chặn tuyên bố quality đạt ở N2/N3/N6 |
| Q10 | Nightly runtime có được dùng khi bàn giao? | Cách ly môi trường, pin version/hash; chưa thay runtime mặc định | Quyết định packaging và vận hành; nếu chỉ stable phải xác minh bản stable có bản sửa |
| Q11 | Chỉ local hay có cloud fallback? | Local-only cho benchmark mục tiêu GPU Iris | Cloud là phương án riêng; không thay số đo local bằng latency dịch vụ |
| Q12 | Tài nguyên máy có được dành riêng khi đo? | Dừng benchmark/test do mình chạy; ghi tải nền không kiểm soát được | Nếu user có workload khác, baseline chỉ là đo quan sát, chưa phải A/B sạch |

Q01/Q03/Q05/Q07 là các lựa chọn ảnh hưởng phạm vi nhiều nhất. Có thể triển khai N0 và spike N1 với giả định tạm ở bảng mà không chờ toàn bộ câu trả lời. Ghi giả định vào manifest từng benchmark; không biến giả định thành quyết định đã chốt.

## 5. Ngân sách hiệu năng để kiểm tra tính khả thi

Đây là phân bổ mục tiêu thử nghiệm cho worker warm, không phải dự báo:

| Phần critical path | Ngân sách mong muốn |
|---|---:|
| ASR GPU cho 180s | 25–30s |
| Dịch/TTS chồng ASR, phần còn lại sau ASR | 10–15s |
| Căn thời gian/mix/mux/QC bắt buộc sau compute | 10–15s |
| Tổng wall mục tiêu | 45–60s |

Không cộng thời gian standalone dịch/TTS vào ASR khi chúng đã overlap. Nếu đo N2/N3 cho thấy ngân sách này không khả thi, cập nhật kế hoạch và báo mức tốt nhất; không tiếp tục tuning nhỏ vô hạn để hứa mốc chưa có căn cứ.

## 6. Quy tắc triển khai và bàn giao

- Bắt đầu N0 rồi N1; ghi checkpoint đã đạt/không đạt cùng artifact thực. N2 là điều kiện bắt buộc của mục tiêu ≤60s, không trì hoãn vì spike dịch có kết quả đẹp.
- Mỗi thay đổi có baseline, giả thuyết, A/B và quality gate. Test cần bám rủi ro: mapping batch, context, invalidation cache, cancel/resume, worker crash, audio/track/timestamp.
- Không đo benchmark nặng cùng pytest hoặc model job khác do mình quản lý. Sau thay đổi chạy phạm vi cần thiết; hồi quy toàn bộ khi chốt tích hợp.
- Mặc định chỉ đổi khi có evidence và quality gate đạt; experimental flags có mô tả giới hạn. Model tải mới và nâng runtime thực hiện riêng để dễ rollback.
- Mỗi checkpoint bàn giao: code/config, script tái lập, report machine-readable, sample nghe/SRT, trạng thái chất lượng, điểm chưa làm và lệnh fallback.
- Workspace tối ưu: `/home/quytrung22521557/jellydub-engine`. Bản upstream `/home/quytrung22521557/jellydub-engine-default` dùng đối chiếu, không sửa từ kế hoạch này.
