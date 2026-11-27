# Tiếp tục tối ưu pipeline — 2026-11-17

> Tài liệu lịch sử trước khi tinh gọn quy trình. `auto` hiện chỉ nhận video tiếng Anh, tự chép lời, dùng `dub-only` và chạy tuần tự; không nhận `--srt`, `--source-lang`, `--mix`, `--pipeline-mode`. Dùng [hướng dẫn CLI hiện tại](cli-guide.md) để vận hành.

Theo CP1–CP6 trong `pipeline-bottlenecks-2026-11-16.md` (khác bộ checkpoint đầu ra MKV cũ trong `optimization-plan.md`). Người dùng chọn tiếp tục tối ưu giọng VieNeu hiện tại; lựa chọn này không được hiểu là đã nghiệm thu diễn xuất/phát âm.

Mục tiêu bổ sung: **video 180 giây trong 40–60 giây trên GPU**, ưu tiên GPU tầm thấp/trung và dùng **Iris Xe hiện tại làm đối chứng chính**. Người dùng cho phép đổi kiến trúc engine/pipeline. Mục tiêu này **chưa đạt**. Không tính lượt reuse cache thành benchmark video mới; thời gian tải model, nạp/compile và warm worker cần báo riêng.

## Thay đổi

- **CP1:** chạy lại A/B toàn bộ 44 câu nói với seed theo câu, BLAS 8 so với 2; so hash từng WAV normalized. Giữ nguyên kết quả đang dở trong `artifacts/cp1-performance/`. Phép đo chính mới ở `artifacts/cp1-performance-clean/`; lượt thăm dò `cp1-performance-resumed` có chạy test đồng thời nên không dùng làm đối chứng cuối.
- **CP2:** thêm `jellydub preview` tạo mẫu nghe có nhiều vai, SRT và `review.json`. Cùng voice/reference/thread settings và cùng cache TTS với pipeline đầy đủ. Giữ trọn câu bắt đầu trong cửa sổ; file có thể dài hơn số giây yêu cầu để tránh cắt tiếng. Review luôn ghi `pending_listening_review`, không suy nghiệm thu giọng từ QC file. Chưa áp cổng phê duyệt bắt buộc lên mọi lệnh cũ.
- **CP3:** thêm `tts_device` và `translate_device` trong TOML, thêm option tương ứng ở `auto`. Nếu không đặt, kế thừa `device`. Auto dùng Torch CUDA nếu thực sự khả dụng, nếu không dùng CPU; yêu cầu CUDA khi không có CUDA hoặc Intel GPU cho backend chưa hỗ trợ sẽ báo lỗi. ASR vẫn giữ lựa chọn OpenVINO/Faster Whisper riêng. `doctor` phân biệt GPU ASR và local Torch models; VieNeu báo backend/provider sau khi load. Intel/CPU có thể kiểm chứng trên máy này; NVIDIA chỉ được kiểm tra logic chọn thiết bị bằng mock.
- **CP4, phần resume/retry:** checkpoint từng vùng ASR đã phân tích timestamp thành công. Key gồm hash audio, toàn bộ nội dung model/config/tokenizer, runtime versions, device, ngôn ngữ, beam, precision và phiên bản guard. JSON có checksum, ghi atomic dưới khóa liên tiến trình. Khi ngắt, vùng đã hoàn thành không cần giải mã lại; vẫn phải load model/VAD/ngôn ngữ khi resume. Lỗi cấu trúc/timestamp transcript không kích hoạt chạy lại toàn clip trên CPU. Chưa phân loại hết mọi lỗi backend khác.

Chưa fuse graph TTS/ASR, đổi precision, thêm backend Intel cho TTS/dịch, batch model xuyên job hay diarization tự động (CP5). Chưa tuyên bố CP4/CP5/CP6 đã hoàn thành.

## Lệnh sử dụng

```bash
.venv/bin/jellydub preview \
  --srt artifacts/family-guy-vieneu/reviewed.vi.srt \
  --config artifacts/checkpoints-2026-11-17/preview.toml \
  --start 0 --seconds 20 \
  --out-dir artifacts/checkpoints-2026-11-17/listening-sample

.venv/bin/jellydub auto VIDEO.mp4 --tts vieneu \
  --asr-backend openvino --asr-device gpu \
  --tts-device cpu --translate-device cpu

.venv/bin/python scripts/benchmark_cp1_tts.py baseline8 --out-dir artifacts/NEW-BENCHMARK
.venv/bin/python scripts/benchmark_cp1_tts.py optimized2 --out-dir artifacts/NEW-BENCHMARK
.venv/bin/python scripts/compare_cp1_tts.py artifacts/NEW-BENCHMARK
```

Benchmark từ chối thư mục variant đã tồn tại để không đo nhầm cache hit. Chạy hai variant tuần tự, tránh test/model inference khác cùng lúc. Đây là TTS toàn bộ corpus, không phải thời gian ASR/dịch/export toàn video. Model đã tải sẵn, cache xử lý rỗng, không có thời gian người biên tập/duyệt.

## Kiểm chứng

- Lượt hồi quy ban đầu: 464 passed, 2 failed do kỳ vọng test cũ vẫn đòi truyền `auto` vào translator. Đã cập nhật kỳ vọng chọn thiết bị thực tế và chạy lại phạm vi thay đổi: 193 passed.
- Test mới gồm: device CPU/CUDA/Intel không hỗ trợ, cache đổi audio/model/beam hoặc hỏng checksum, ngắt sau vùng ASR đầu rồi resume, lỗi transcript không rerun CPU, preview giữ câu trọn vẹn và reuse cache, từ chối ghi đè SRT đầu vào.
- Mẫu nghe thật: `artifacts/checkpoints-2026-11-17/listening-sample/preview.wav`, hai vai Peter/Lois, dùng cache cũ mà không gọi TTS mới. Có SRT và phiếu review cạnh file WAV.

## Kết quả A/B CP1 toàn bộ corpus

| Phép đo | BLAS 8 | BLAS 2 |
|---|---:|---:|
| 44 câu, load + TTS + normalize | 244,96s | 120,55s |
| Load model (nằm trong tổng) | 7,59s | 6,64s |
| WAV có hash giống đối chứng | — | 44/44 |

Nhanh **2,032 lần**, giảm **50,8%** thời gian khối này. Cùng thứ tự câu, seed theo speaker/text, giọng, model, CPUExecutionProvider và ORT 4 luồng. Cả hai lượt bắt đầu với cache xử lý rỗng; model đã tải sẵn, process riêng, chạy tuần tự và không chạy test/model khác đồng thời. Chỉ một cặp A/B trên một máy, chưa có khoảng tin cậy hoặc kết luận cho phần cứng khác.

Bằng chứng: `artifacts/cp1-performance-clean/{baseline8,optimized2}/report.json` và `ab-result.json`.

## Toàn pipeline và cache vùng ASR

Lượt mới `artifacts/family-guy-vieneu/optimized-2026-11-17-host/report.json`: **325,00s** gồm full decode QA; TTS **132,22s**, ASR **135,42s**, dịch **24,65s**, export **13,38s**. Cache xử lý ban đầu rỗng, model đã tải sẵn. Encoder/decoder xác nhận GPU.0, CPUExecutionProvider cho TTS; Small/beam=5 và các giọng được giữ. Transcript khớp ledger cũ trước khi gán vai. Chỉ MKV + SRT trong thư mục output; decode, hash luồng hình và nguồn không đổi đều đạt.

So lượt lịch sử 615,50s: giảm khoảng **47,2%** (1,89 lần). Đây là đối chiếu hai thời điểm, không phải cặp full-pipeline fixed-seed được kiểm soát; dùng A/B 44 câu phía trên để kết luận riêng hiệu quả thread tuning. Lượt này vẫn có ledger và bản dịch biên tập chuẩn bị sẵn; không phải workflow tự động hoàn toàn hoặc giọng đã được duyệt.

Sandbox không truy cập được Intel GPU; lượt `optimized-2026-11-17` thất bại trước inference và được giữ lại. Lượt `-host` được cấp phép chạy ngoài sandbox, không chuyển sang CPU để lấy kết quả.

ASR thực trên mẫu 20 giây: cache vùng rỗng **20,85s / 61 decoder calls**, chạy lại **3,29s / 0 decoder calls**, transcript giống nhau, cùng Small/beam=5/GPU.0. SDK import **không** nằm trong hai thời gian này, vẫn đo riêng trong report. Warm vẫn load/compile model và chạy VAD; không gọi lại decoder. Bằng chứng: `artifacts/checkpoints-2026-11-17/asr-region-cache/report.json`; tái lập bằng `scripts/verify_asr_region_cache.py` ở thư mục output mới.

## Prototype thay đổi engine

Theo ủy quyền mở rộng của người dùng: thử đưa acoustic output heads từ NumPy vào ONNX graph (`src/jellydub/tts/fused.py`), giữ model, sampling/history, guard và giọng. Không sửa SDK cài đặt. Prototype không nằm trên đường chạy mặc định. ABBA ba vai, seed cố định, warm cả hai đường; setup graph và load được đo riêng. Chỉ xem xét tích hợp nếu waveform bằng nhau và tốc độ tăng có ý nghĩa; vẫn cần A/B đủ corpus trước khi đổi mặc định. Chưa fuse sampling hoặc vòng 16 codebook/frame.

Đã thử tiếp graph gộp toàn bộ **16 codebook/frame + heads + sampling** (`tts/frame_graph.py`): mỗi frame gọi runtime một lần, history và uniform random truyền tường minh. Sampling ở graph có thể khác số học NumPy; chưa được phép coi là audio tương đương khi hash khác. Chỉ hỗ trợ các tham số sampling mặc định; tham số khác quay về SDK. Vòng generation toàn câu vẫn ở SDK.

| Thử nghiệm | Tốc độ tương đối SDK CPU | Hash mẫu | Quyết định |
|---|---:|---|---|
| Fuse heads, ORT CPU | 0,929× | Giống | Không tích hợp mặc định |
| ORT 1/2/4 luồng, BLAS 2 | 4 luồng tốt nhất; 2 gần bằng; 1 chậm hơn | Giống | Giữ mặc định SDK 4 |
| Gộp frame, ORT CPU | 1,073× | Có khác | Chưa đạt gate audio |
| Gộp frame, Iris GPU FP32 | 0,755× | Giống 3 mẫu | Không tích hợp mặc định |
| Gộp frame, Iris GPU FP16 | 0,129× | Khác | Loại khỏi cấu hình mặc định |

Đây là ABBA 3 vai/12 lượt, không phải full-corpus hoặc xác nhận chất lượng nghe. FP16 có thể kích hoạt đường generation/babble guard khác, vì thế số đo chậm không được diễn giải thành chỉ throughput kernel thấp.

Prototype đưa riêng prefill/decode/acoustic transformer lên GPU (`tts/openvino_session.py`) cũng chạy thật GPU.0 nhưng chậm hơn: Peter CPU 3,49s, acoustic GPU 5,70s, tất cả transformer GPU 7,23s. Codec, heads và sampler vẫn CPU. Dữ liệu ở `openvino-tts-staged/`; giữ hai lỗi nạp symlink trước đó ở thư mục khác. Bản sao regular weights khoảng 1,1GB, không sửa cache HF gốc.

Native ASR: tạo `.venv-native-asr` riêng với GenAI/OpenVINO 2026.4.1; Small/beam=5/GPU/f32, tắt SDPA optimization vẫn lỗi `IRemoteTensor: Not Implemented`. Không đổi CLI sang backend này. Report ở `artifacts/performance-audit/asr/small/native-2026-11-17/gpu.json`. [PR upstream #4498](https://github.com/openvinotoolkit/openvino.genai/pull/4498) đã merge sửa đường remote encoder, nhưng không suy rằng wheel stable đang cài đã chứa hoặc chạy được bản sửa trên máy này. Không tự hạ beam để gọi là cùng chất lượng.

## Hướng tiếp theo theo mục tiêu 40–60 giây

Tuning luồng đã có hiệu quả, nhưng chưa đủ. Các prototype cho thấy chuyển từng graph sang GPU hoặc gộp riêng acoustic frame chưa giải quyết toàn bộ chi phí. Cần khảo sát backend giữ generation/KV state trong runtime và batch được nhiều câu trên Iris; nếu backend hiện tại không có đường phù hợp, cần đánh giá model/backend khác bằng cùng bộ câu và gate nghe. Việc chọn backend/model mới chưa có bằng chứng để chốt. Pipeline có thể dùng bounded queues giữa ASR/dịch/TTS sau khi backend ổn; phải đo tranh CPU/RAM và giữ context dịch, không chỉ cộng thêm worker.

Các thay đổi được áp dụng vào đường mặc định trong lượt này chỉ gồm preview, device selection, cấu hình chung cho CLI/live, resume ASR và phân loại lỗi transcript. Các graph/backend thử nghiệm chưa được bật mặc định.

## Backend ASR native đã nối vào CLI (opt-in)

Wheel nightly chính thức `2026.5.1.dev20261116` đã chạy được Small/beam=5/GPU/f32 trên máy này. Phép đo GenAI trực tiếp mẫu 20 giây: 13,72s / 11,21s, chưa gồm nạp/compile. Chưa lấy số này làm thời gian toàn pipeline.

CLI hỗ trợ `--asr-backend genai --asr-python INTERPRETER`. Native generation chạy trong process riêng, dùng một WhisperPipeline cho mọi vùng trong một job. Parent xử lý VAD/context; worker giữ language/beam/token budget/no-repeat, cache vùng theo backend/runtime/model/audio. Lượt reuse hoàn toàn cache không cần load model. JSON IPC không dùng pickle; worker được terminate khi parent bị ngắt. Điều này tách phiên bản runtime nightly khỏi môi trường Torch/Optimum hiện tại. Mặc định auto vẫn giữ backend cũ; GenAI hiện cần chỉ định EN/VI, chưa hỗ trợ auto language ở adapter này.

Môi trường kiểm chứng:

```bash
uv venv .venv-native-asr-next --python .venv/bin/python
uv pip install --python .venv-native-asr-next/bin/python \
  --prerelease allow --index-strategy unsafe-best-match \
  --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
  openvino-genai numpy

.venv/bin/jellydub auto VIDEO.mp4 --source-lang en \
  --asr-backend genai --asr-device gpu --asr-beam-size 5 \
  --asr-python .venv-native-asr-next/bin/python \
  --tts vieneu --tts-device cpu --translate-device cpu
```

Lệnh install ở trên lấy nightly mới nhất tại lúc chạy, không phải lock. Phiên bản đã đo: OpenVINO/GenAI/Tokenizers `2026.5.1.dev20261116` (GenAI/Tokenizers có đoạn `.0` trong version). Khi tái lập nên pin chính xác các version trong môi trường đã lưu. Runtime nightly và transcript toàn clip vẫn cần kiểm chứng riêng trước khi đổi mặc định.

Kiểm chứng adapter thật (`native-cli/report.json`): mẫu 20s cache lạnh **21,94s**, trong đó load/compile **7,96s**; warm cache **1,34s**, không load model. Family Guy ASR cache lạnh **82,23s**, load/compile **6,25s**, warm cache **3,17s** / 9 vùng cache hit, không load model. Hai lượt cold/warm trả cùng cue list. Đối chiếu backend cũ: **41 cue có text khớp hoàn toàn**, 328 từ khớp, timestamp lệch tối đa khoảng **0,00000076s**. Đây là độ nhất quán với ASR cũ, không phải WER với ground truth đã nghe duyệt. So 135,42s của ASR Optimum trong lượt full-video trước: nhanh khoảng **1,65×**; cold compile/cache ở hai lượt không hoàn toàn giống nhau.

Backend native chưa bật mặc định. Không cộng/trừ thời gian stage để gọi là benchmark full-pipeline mới; lượt MKV/SRT native phải đo riêng.

Lượt full pipeline native đã hoàn thành: `artifacts/family-guy-vieneu/native-2026-11-17/report.json`, **303,64s** gồm full decode QA. ASR **86,20s**, dịch **42,07s**, TTS FP32 **145,17s**. Ledger được kiểm tra text như cũ, model/beam/voices giữ nguyên; MKV stream-copy, decode và nguồn không đổi đều đạt. Tổng chỉ nhanh hơn lượt Optimum mới 325,00s khoảng **6,6%**, vì chi phí import Torch chuyển sang stage dịch và TTS không seed cố định có biến thiên generation. Không lấy mức tăng tốc ASR riêng để tuyên bố toàn pipeline tăng tương ứng.

## Đường overlap và INT8 opt-in

- `--pipeline-mode overlap`: ASR phát các cue sau mỗi vùng/segment đã hoàn thành; process riêng dịch và TTS trong lúc ASR tiếp tục. Pipe giới hạn dung lượng cung cấp backpressure. Mỗi worker giữ translator/provider cho nhiều vùng. Khi hủy, process group của worker được dừng; lỗi prefetch được báo và pipeline chuẩn dùng cache để tiếp tục.
- Worker giữ lại singleton chưa kết thúc câu để nhận ngữ cảnh từ vùng sau. Logic nhóm câu dùng chung với `translate_cues`. Chỉ hỗ trợ source EN/VI được chỉ định; overlap EN hiện chỉ dùng OPUS, chưa áp sang backend LLM contextual. Backend native và các ASR adapter có callback chung; final pipeline vẫn căn thời gian/QC/export như cũ.
- `--tts-precision int8 --tts-device cpu`: chọn export INT8 upstream cùng revision `61b85e3d937fbbacb387714180e8182823512523`; cache tách FP32/INT8, không nhập nhầm cache cũ. Default vẫn FP32. ABBA ba vai nhanh **1,426×** trong mẫu ngắn, **waveform khác**; file nghe đối chiếu trong `tts-precision/`, chưa có nghiệm thu nghe hay benchmark full-corpus INT8.
- Tránh import Torch chỉ để kiểm tra CUDA khi metadata wheel đã xác nhận `+cpu`; cache hit không phải trả chi phí import model framework vô ích.

Overlap và INT8 đã có code/test chức năng, **chưa benchmark end-to-end**, không ghi nhận là đã đạt mục tiêu 40–60s. Hồi quy hiện tại: **485 passed, 5 deselected**, tiến trình thoát **0**, `artifacts/checkpoints-2026-11-17/tests-current.log`; `git diff --check` đạt. Bao gồm cache/context qua ranh giới vùng, worker IPC/cancel, native worker/model reuse và lỗi transcript. Không thay nghiệm thu nghe hoặc benchmark real GPU bằng pytest.

Người dùng yêu cầu thêm bản upstream độc lập để test phương án khác: clone `https://github.com/tecix/jellydub-engine.git`, main `465f566657ee0fbc0619dbd3f960a283aa9a0571`, tại `/home/quytrung22521557/jellydub-engine-default`. Không chép các tối ưu trong workspace này sang bản đối chứng.

## Đo overlap thực tế và sửa IPC

Tiếp tục cùng ngày, chạy `scripts/benchmark_overlap.py` với source Family Guy **156,665s**, native Whisper Small/beam=5/f32 trên Iris, OPUS CPU, VieNeu INT8 CPU, một preset mặc định. Không có ledger/bản dịch chuẩn bị thủ công. Mỗi lượt dùng cache xử lý riêng ban đầu rỗng; model tải sẵn. Đây là workflow tự động khác benchmark nhiều vai có ledger ở trên, không so hai bộ như cùng workload.

Lượt overlap đầu phát hiện lỗi thật: FFmpeg normalization kế thừa stdin của worker và ăn mất dữ liệu JSON. Worker báo JSONDecodeError rồi pipeline fallback tuần tự. Đã dừng lượt này, giữ log `overlap-ab/attempt.log`, đánh dấu report không hợp lệ. Sửa `run_ffmpeg` bằng `-nostdin` và `stdin=DEVNULL`; thêm test process thật để xác nhận JSON stdin không bị mất khi FFmpeg chạy.

| Chế độ | Thời gian đến MKV/SRT + QC CLI | Full decode bổ sung | SRT |
|---|---:|---|---|
| Tuần tự, INT8 | **312,09s** | Pass | 41 cue |
| Overlap đã sửa, INT8 | **209,60s** | Pass | Cùng hash, 41 cue |

Nhanh **1,489×**, giảm **32,8%**. Overlap chuẩn bị đủ 41 speech cues trong **185,36s**; final translate/TTS reuse cache, TTS final không load model. Không có prefetch fallback. Báo cáo `artifacts/checkpoints-2026-11-17/overlap-comparison.json`; chi tiết từng lượt ở `overlap-ab/sequential/report.json` và `overlap-fixed/overlap/report.json`.

Giới hạn: một lượt mỗi mode, khác thời điểm, load máy có thể biến động, không fixed seed TTS. Sequential đầu thuộc process benchmark ban đầu; overlap đo lại bằng process mới sau sửa IPC. Model download bị tắt nhưng OS file/driver compile cache không được xóa. Thời gian full decode bổ sung không nằm trong hai số wall CLI này. SRT giống nhau xác nhận nội dung/timing đầu vào TTS, chưa xác nhận waveform/giọng tương đương. OPUS vẫn có cảnh báo collapse lặp ở cue 2/3 và 32/33; cần review nghĩa. INT8 chưa nghe nghiệm thu. Không công bố đạt 40–60s; **209,60s cho clip 156,665s còn cách xa mục tiêu**.

Tái lập với thư mục mới:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_overlap.py --run NEW-RUN
```

Script hiện chạy hai mode bằng process riêng, từ chối thư mục đã có dữ liệu. Lệnh CLI tương ứng thêm `--pipeline-mode overlap --tts-precision int8 --tts-device cpu --source-lang en --translator opus-mt` vào cấu hình native ASR ở trên. Giữ mặc định sequential/FP32 cho đến khi có nghiệm thu chất lượng và đo nhiều lượt.

Kiểm chứng cuối đợt này: **487 passed, 5 deselected**, exit **0** (`tests-overlap-fix.log`), `git diff --check` đạt; cả hai MKV có hash luồng hình giống nguồn. Không thay đổi bản đối chứng `jellydub-engine-default` trong đợt tối ưu này.
