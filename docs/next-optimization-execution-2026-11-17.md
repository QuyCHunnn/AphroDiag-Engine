# Thực thi N0–N1 — 2026-11-17

Kế hoạch: `next-optimization-plan-2026-11-17.md`. Trạng thái hiện tại: triển khai bộ đo N0 và sàng lọc N1; chưa đạt 40–60s, chưa đổi backend dịch mặc định.

## N0

- `scripts/prepare_optimization_corpus.py` tạo clip stream-copy, giữ nguồn, manifest SHA256/duration. Tutorial 180,017s, game/effects 180,033s, Family Guy regression 156,665s. Không padding Family Guy để gọi là 180s.
- Bộ nghiệm thu còn thiếu hội thoại nhiều vai đủ 180s và reference được nghe duyệt. Hai clip 180s có mật độ lời khác nhau; không dùng một clip để xác nhận tất cả workload.
- `scripts/trace_pipeline_job.py` chạy CLI với processing cache mới, chặn download subtitle/model, ghi message JSONL, config/command, sample RSS cây process mỗi 0,5s. RSS cộng có thể đếm shared pages nhiều lần; không phải RAM vật lý duy nhất hoặc GPU occupancy.
- `performance_trace.py` bật bằng `JELLYDUB_TRACE_FILE`; span theo process/monotonic cho ASR load/generate, dịch load/generate, TTS load/infer/synthesize/normalize, FFmpeg, queue submit/drain. Không ghi text thoại trong trace. Span lỗi giữ nguyên exception; trace I/O lỗi không làm pipeline fail.
- `scripts/summarize_pipeline_trace.py` phân biệt inclusive span time, union wall và overlap ASR/TTS; không cộng span lồng nhau thành exclusive compute. Import/VAD và chi tiết export vẫn cần phân bổ thêm; critical-path trace chưa được coi hoàn chỉnh.
- Chế độ worker-ready với processing cache rỗng chưa được runner hỗ trợ; không dùng cache-hit thay thế.

## N1 — Export và sàng lọc OPUS

Giữ `Helsinki-NLP/opus-mt-en-vi`, revision `989c9fb9ec63987901022baf0182dcec3e149be6`. Export OpenVINO không quantization, các Const weights float32; manifest hash model/tokenizer ở `artifacts/next-optimization/opus-openvino-fp32/export-manifest.json`.

Export bằng môi trường CLI bị chặn: Optimum Intel 2.1.0 chỉ cho export với Transformers tối đa 4.57.6, CLI hiện là 5.5.4. Tạo `.venv-opus-export` với Transformers 4.57.6/tokenizers 0.22.2/HF Hub 0.36.2, tham chiếu thư viện có sẵn của môi trường chính. Đây là môi trường export tạm phụ thuộc base, không phải môi trường bàn giao độc lập. Không hạ Transformers trong CLI. Cần lock độc lập nếu dùng lâu dài.

`translate_openvino.py` là adapter prototype, chưa đăng ký trong factory/config mặc định. Tái sử dụng chính hàm generation của `OpusMTTranslator`: beam 4, repetition retry/guard, token budget, context pairing. Cache identity tách backend/device/export manifest. Device thực được xác nhận từ compiled encoder/decoder, không suy từ tên process.

### Phép đo 41 cue

`benchmark_opus_openvino.py`: cùng 41 cue EN Family Guy, batch=16, hai lượt inference với cache dịch rỗng riêng mỗi lượt; model loaded một lần/process/backend. Chạy Torch→OV CPU→OV GPU tuần tự. Chưa phải 3 cặp A/B cân bằng theo N0, chưa có khoảng tin cậy.

| Backend | Lượt 1 | Lượt 2 | Median | Load/compile riêng |
|---|---:|---:|---:|---:|
| PyTorch CPU | 65,05s | 43,00s | 54,03s | 22,11s |
| OpenVINO CPU | 75,75s | 74,80s | 75,27s | 21,96s |
| OpenVINO Iris GPU | 55,17s | 42,78s | 48,97s | 18,75s |

- 41/41 cue text giống Torch reference ở cả hai lượt của cả ba backend; cảnh báo collapse lặp vẫn ở cue 2/3 và 32/33. Đây là agreement, không phải chứng minh nghĩa đúng.
- OV CPU chậm hơn khoảng 39,3%; không chọn làm thay thế.
- OV GPU median giảm khoảng 9,4%; thấp hơn gate spike ≥15%, biến thiên lớn. Chưa đo contention với ASR GPU hoặc end-to-end; không tích hợp mặc định.
- Mẫu 6 cue batch4 từng cho GPU chậm hơn CPU. Kết quả phụ thuộc batch/độ dài và không được cherry-pick sample ngắn.
- Chưa sweep batch1/4/8/16 đủ corpus hoặc length bucketing; N1 chưa đóng toàn bộ. Nếu tiếp tục nhánh GPU phải đo full pipeline với ASR trước khi chấp nhận.

Bằng chứng: `artifacts/next-optimization/opus-{probe,full}/comparison.json`, report từng backend, script export/benchmark/compare. API tham chiếu chính thức: [supported models](https://huggingface.co/docs/optimum-intel/en/openvino/models), [inference](https://huggingface.co/docs/optimum-intel/en/openvino/inference).

## Hai baseline 180s với trace (cold process)

Cấu hình chung: native ASR Small/beam5/f32 Iris GPU, OPUS CPU, VieNeu INT8 CPU, overlap, processing cache rỗng, models đã tải, không ledger. Chỉ một lượt mỗi clip, chưa đủ số lần lặp theo gate N0. Model INT8/nghĩa dịch chưa được nghe duyệt.

| Chỉ tiêu | Game/effects 180,033s | Tutorial 180,017s |
|---|---:|---:|
| Wall CLI đến completed | 129,59s | 217,22s |
| Cue count | 16 | 30 |
| ASR generation union wall | 36,96s | 62,80s |
| Translation load spans | 37,15s | 21,90s |
| Translation generation union | 11,03s | 35,06s |
| TTS load spans | 8,14s | 16,52s |
| TTS inference union | 24,12s | 92,27s |
| ASR generation chồng TTS inference | 0,00s | 3,20s |
| Queue drain sau ASR | 51,12s | 111,49s |
| Peak sampled process-tree RSS sum | 1.632,9 MiB | 2.107,3 MiB |

Đây là span wall time có lồng nhau; không cộng cột để suy tổng. `translation.load` gồm import/load ở lần gọi đầu và no-op các lần sau, không phải model bị load 12 lần. Queue drain gồm công việc dịch/TTS chưa xong, không phải chỉ overhead queue. RSS sum có double-count shared pages. Chưa có phép đo GPU occupancy.

Full decode bổ sung: hai file đều returncode 0, không lỗi; hash luồng hình hai output khớp clip đầu vào. Wall CLI không gồm full decode bổ sung này. Evidence mỗi clip: `artifacts/next-optimization/baselines/{effects180-cold-01,tutorial180-cold-01}/{report.json,spans.jsonl,trace-summary.json,media-verification.json}`.

Hệ quả đo được: worker lạnh trì hoãn TTS khiến overlap ít hơn kỳ vọng. N4 worker-ready/residency cần benchmark riêng; chưa được trừ số load khỏi tổng rồi gọi đó là wall warm. Tutorial riêng ASR compute đã >60s, nên N2 vẫn bắt buộc. N0 chưa đóng gate: thiếu corpus/reference nói trên, ≥3 cặp cân bằng, worker-ready empty-cache và đầy đủ attribution import/VAD/export.

## Spike N2 — ASR precision, giữ model/beam

`benchmark_native_precision.py` chạy hai lượt riêng cho Family Guy regression, processing cache rỗng, native Small/beam5, cùng VAD/region/token-budget/no-repeat và GPU_ENABLE_SDPA_OPTIMIZATION=False. Worker nhận precision experimental qua request; CLI vẫn mặc định f32 và chưa có option đổi precision. Region cache key gồm precision, không reuse nhầm f32/f16.

- f32: **57,35s**, 41 cue.
- f16: **61,18s**, 40 cue, chậm hơn khoảng 6,7%.
- Word sequence agreement khoảng **99,09%**, không giống hoàn toàn: có thay `at a` thành `to`, thêm `ugh`, thay tiếng hét; segmentation cũng khác. So positional cue list bị lệch sau merge nên report ghi thêm normalized toàn transcript và chú thích không coi index zip là alignment.
- Không chọn f16; giữ f32. Hai số này là A/B ở thời điểm khác với baseline ASR 82s cũ; không quy mức thay đổi so lịch sử cho một tối ưu không tồn tại.
- Chưa nghe duyệt hoặc đo nhiều lượt/clip; không tuyên bố f32/f16 tương đương chất lượng, không giảm beam/model để ép đạt tốc độ.

Evidence: `artifacts/next-optimization/native-precision/family-regression/report.json`. N2 còn các nhánh worker residency, cách phân vùng, runtime properties và benchmark đủ corpus; spike precision này chưa hoàn thành N2.

## Trạng thái gate

| Checkpoint | Đã thực hiện | Chưa đủ để đóng |
|---|---|---|
| N0 | Corpus local/hash, trace đa process, hai clip 180s cold chạy thật, decode/hình đạt | 180s multi-role reference, số lần lặp, warm worker empty-cache, attribution đầy đủ |
| N1 | Export cùng revision FP32, adapter riêng, 6/41 cue CPU/GPU, output agreement | ≥15% spike gain chưa đạt; batch sweep/length bucketing và end-to-end contention chưa đo |
| N2 | Precision A/B Small/beam5 GPU; loại f16 | Các nhánh khác và chất lượng/reference nhiều clip |
| N3 | Đã thử tiếp batch acoustic thực CPU/Iris GPU, hai nhóm câu | GPU chưa thắng CPU đồng thời; waveform graph batch chưa nghiệm thu; xem báo cáo riêng |
| N4–N6 | Chưa triển khai đợt này | Theo kế hoạch |

Không đăng ký adapter dịch OpenVINO vào factory hoặc đổi config mặc định vì chưa qua gate. Giữ prototype và artifact để tái lập. Bản upstream đối chứng không bị sửa trong đợt này.

Kiểm chứng cuối: **492 passed, 5 deselected**, exit **0**, `artifacts/next-optimization/tests-final.log`; `git diff --check` đạt. Test mới kiểm tra trace success/failure không đổi lỗi gốc, cache export/device tách Torch và f32/f16 cache riêng. Suite đầu có một lỗi stale class do đang sửa signature worker/cache trong lúc suite chạy; đã chạy lại process mới sau khi chốt mã, phạm vi 17 test và toàn bộ 492 test đều đạt. Các test không thay benchmark GPU và nghiệm thu nghe.

Cập nhật sau yêu cầu thử TTS GPU: xem `tts-iris-batch-experiment-2026-11-17.md`. Prototype đã có batch row thật và execution GPU.0, nhưng chưa đạt gate tốc độ/chất lượng để tích hợp. CPU đồng thời là ứng viên tốt hơn trong hai nhóm câu ngắn, chưa phải kết quả toàn pipeline. Hồi quy mới nhất của đợt đó: **500 passed, 5 deselected**, exit **0**, `tests-gpu-batch.log`.
