# Thử TTS batch thực trên Iris Xe — 2026-11-17

Theo yêu cầu thử chuyển TTS sang GPU. Kết luận: đã chạy batch acoustic thực trên GPU.0, nhưng **chưa có cấu hình GPU thắng các phương án CPU một cách đủ rõ và giữ audio tương đương**. Không đổi provider/config mặc định.

## Đã làm gì khác lần trước

- Graph acoustic hiện có khai báo batch=1, nhưng các reshape bên trong đọc batch dimension từ input. Tạo graph mới vector hóa theo từng row; không sửa SDK hoặc trọng số upstream.
- Mỗi row có hidden state, repetition history, uniform random và EOS riêng. Gom 16 codebook của mỗi frame trong graph, K/V local nằm trong graph. Không có attention hoặc history xuyên các câu.
- Graph capacity 1/2/4 được compile và chạy thật bằng OpenVINO GPU f32. Execution devices xác nhận GPU.0; không gọi CPU fallback để ghi nhãn GPU.
- Coordinator gom request từ các cue đang xử lý, có queue giới hạn, max wait, Future mapping và lỗi/shutdown rõ. SDK clone nhẹ chia sẻ trọng số/ORT sessions nhưng giữ lock, state và RNG riêng mỗi cue. Giữ normalization, global decoding, codec và babble guard của SDK.
- **Chỉ acoustic graph lên GPU. Global prefill/decode và codec vẫn CPU.** Đây chưa phải engine TTS hoàn toàn GPU hoặc generation toàn câu nằm trong native graph.

Mã prototype: `tts/batched_frame_graph.py`, `tts/gpu_batch_runtime.py`; graph gốc hỗ trợ build-only để tái sử dụng. Không đăng ký prototype vào factory CLI.

## Probe acoustic frame

8 input frame lấy từ inference SDK thật; dùng cùng token/history/uniform. Bỏ lượt chạy graph đầu, báo median hai lượt sau:

| Backend/graph capacity | Thời gian cho 8 row-frame |
|---|---:|
| ORT CPU batch 1 | 0,130s |
| ORT CPU batch 2 | 0,097s |
| Iris GPU batch 1 | 0,818s |
| Iris GPU batch 2 | 0,527s |
| Iris GPU batch 4 | 0,263s |

Batch GPU cải thiện throughput so GPU batch1, nhưng vẫn thua CPU graph ở workload này. Tăng batch có khác sampled codes; không coi cùng seed là bit-exact khi math/kernel thay đổi. SDK CPU frame samples lấy khi capture chưa warm cân bằng nên không dùng nó làm đối chứng timing chính.

Evidence: `artifacts/next-optimization/tts-gpu-batch-probe/report.json`; `scripts/probe_tts_gpu_batch.py`. Đây là acoustic-only, không ngoại suy thành tốc độ TTS toàn câu.

## TTS đầy đủ trên hai nhóm câu

Mỗi mode hai lượt, cùng seed theo speaker/text, không speech cache, model warm. Setup/compile đo riêng. Watermark tắt ở mọi mode, không bỏ frame cap hoặc babble guard. Batchers dùng chung một InferRequest qua một service thread; không gọi đồng thời cùng request.

| Chế độ | 4 câu/4 vai, median | 4 câu Peter độ dài gần nhau, median |
|---|---:|---:|
| SDK CPU tuần tự | **8,46s** | **12,30s** |
| SDK CPU đồng thời, RNG riêng | **5,79s** | **8,22s** |
| CPU acoustic graph capacity 4 | **4,82s** | **8,15s** |
| GPU acoustic graph capacity 4 | **8,96s** | **11,63s** |

- GPU chậm hơn CPU tuần tự khoảng 6% ở nhóm nhiều vai; nhanh hơn khoảng 5,4% ở nhóm cùng vai. Không nhất quán đủ để chọn.
- GPU chậm hơn CPU đồng thời khoảng **55% / 41%** ở hai nhóm.
- CPU đồng thời nhanh hơn CPU tuần tự khoảng **1,46× / 1,50×**, hash waveform giữ nguyên ở cả hai lượt mỗi nhóm. Đây là kết quả prototype với RNG riêng và shared immutable sessions, chưa tích hợp vào provider đang có scoped BLAS lock.
- Các mode graph batch CPU/GPU đều có waveform khác baseline. Thời lượng vẫn cùng cấp độ nhưng có thay đổi; chưa nghe/reference ASR kiểm chứng. Không có quality acceptance.
- GPU mixed: active rows mỗi lần gọi là 1/2/3, mean khoảng 1,63 trên capacity4. Nhóm Peter với wait40ms đã có active batch4 nhưng chỉ 2 lần trong hai lượt; nhiều bước tail vẫn chỉ 1/3 row. Report ghi histogram thực, không gọi capacity4 là luôn chạy batch4.
- Setup GPU capacity4: khoảng 16,52s ở nhóm mixed, đo riêng khỏi wall inference; cold job còn phải trả chi phí này nếu không giữ session/compile cache.

Nguyên nhân có bằng chứng: GPU batch giúp so với GPU single-row nhưng không đủ thắng CPU graph; scheduler không duy trì đủ active rows; EOS/độ dài từng câu khác nhau tạo tail. Global prefill/decode và codec chưa được chuyển sang backend batch GPU. Không có kernel/occupancy trace nên không gán toàn bộ chậm cho copy, PCIe hoặc một kernel cụ thể. Iris chia sẻ RAM, không mặc định coi đây là copy qua PCIe.

## Artifact và tái lập

- `artifacts/next-optimization/tts-gpu-batch-summary.json`: tổng hợp, giới hạn, quyết định không áp GPU mặc định.
- `tts-gpu-batch-full/`: report, runtime versions, audio CPU/GPU của Peter/Lois/Stewie/Brian.
- `tts-gpu-batch-matched/`: 4 câu Peter, cue 2/10/16/4, max wait40ms, report/audio.
- `scripts/benchmark_tts_gpu_batch.py --run NEW-RUN --cohort mixed`.
- `scripts/benchmark_tts_gpu_batch.py --run NEW-RUN --cohort matched --batch-wait .04`.
- `scripts/summarize_tts_gpu_batch.py`: tổng hợp các thư mục đã đo.

Runtime graph revision: `61b85e3d937fbbacb387714180e8182823512523`; package versions lưu cạnh report. Chưa phải A/B cân bằng ≥3 cặp, chưa benchmark 44 cue/toàn video. Model warm, fixed mode order, waveform seeds cố định nhưng các mode graph có math khác. Không lấy số này để công bố 180s video xử lý trong 40–60s.

## Quyết định

1. Giữ TTS CPU mặc định; không gọi prototype này là production-ready GPU backend.
2. Hướng có lợi hơn trong mẫu là CPU đồng thời, nhưng cần đo full corpus, model/thread contention và cache/live-worker lifecycle trước khi tích hợp. Prototype CPU graph cũng cần nghe duyệt do waveform thay đổi.
3. Nếu tiếp tục GPU: cần giả thuyết mới cho batching global decoder/prefill, attention masks cho prompt/KV khác độ dài, GPU KV residency và scheduler tail. Không chỉ tăng Python worker hoặc đổi chuỗi device. Chưa triển khai các phần này trong thử nghiệm này.

Kiểm chứng: **500 passed, 5 deselected**, exit **0**, `artifacts/next-optimization/tests-gpu-batch.log`; `git diff --check` đạt. Test mới kiểm tra mapping/độc lập row và partial batch, RNG riêng không đụng global NumPy, propagation khi runtime lỗi, đóng worker và tham số wait không hợp lệ. Không dùng pytest thay nghiệm thu nghe.
