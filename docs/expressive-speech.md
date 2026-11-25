# Speech plan chung: Gemini, VieNeu và VoxCPM2

SRT giữ nguyên lời hiển thị và thời gian. JSON speech plan đặt lời đọc và chỉ dẫn diễn theo `index` của SRT tiếng Việt cuối cùng. Không tự phân tích cảm xúc từ video hay tự viết speech plan.

Mẫu có thể chỉnh trực tiếp: [examples/speech-plan.json](../examples/speech-plan.json).

```json
[
  {
    "index": 1,
    "text_sub": "Thôi được rồi, anh nói đúng.",
    "text_tts": "Thôi… được rồi. Anh nói đúng.",
    "direction": {
      "emotion": "buồn, miễn cưỡng chấp nhận",
      "pace": "chậm",
      "emphasis": ["được rồi", "anh nói đúng"],
      "ending": "hạ giọng, nhỏ dần",
      "nonverbal": "thở dài trước câu"
    },
    "ref_audio": "refs/peter_weary.wav"
  }
]
```

`ref_audio` tùy chọn; bỏ trường này nếu không có mẫu. Với VieNeu/VoxCPM2, đường dẫn tính từ thư mục JSON và clip cần đọc được, dài 3–8 giây. Gemini hiện dùng named voice, bỏ qua reference trong plan có `direction` và báo cảnh báo; reference không cần tồn tại trên máy khi chạy Gemini. Cấu hình giọng/runtime vẫn riêng cho mỗi backend. Chung speech plan không có nghĩa ba model tạo cùng một giọng hoặc diễn giống nhau.

## Cấu hình và chạy

Thêm vào profile backend đang dùng:

```toml
[jellydub]
tts = "vieneu" # hoặc gemini / voxcpm2
speech_plan = "speech.json"
```

```sh
.venv/bin/jellydub preview --srt movie.vi.srt --config expressive.toml --out-dir preview/expressive
.venv/bin/jellydub dub movie.mp4 --srt movie.vi.srt --source-lang vi --config expressive.toml
```

Giữ model, giọng, API key hoặc đường dẫn binary/model theo [Gemini](gemini-tts.md) / [VoxCPM2](voxcpm2.md). Đường dẫn `speech_plan` tính từ thư mục chạy lệnh.

## Cách chuyển chỉ dẫn

| Trường | Gemini | VoxCPM2 | VieNeu v3 Turbo |
|---|---|---|---|
| `emotion` | Metadata style / prompt riêng từng câu | Prefix điều khiển | Báo không có điều khiển trực tiếp; reference có thể ảnh hưởng |
| `pace` | Hướng dẫn nhịp đọc | Hướng dẫn nhịp đọc | `chậm`, `slow`, `slowly`: thay dấu phẩy/chấm phẩy và chấm giữa câu bằng dấu ba chấm; báo đây chỉ là xấp xỉ. Giá trị khác báo không hỗ trợ |
| `emphasis` | Hướng dẫn nhấn các từ cụ thể | Hướng dẫn nhấn các từ cụ thể | Báo không hỗ trợ trực tiếp; không sửa từ bằng viết HOA |
| `ending` | Hướng dẫn giọng cuối câu | Hướng dẫn giọng cuối câu | Báo không hỗ trợ trực tiếp |
| `nonverbal` | Hướng dẫn hành động trong style/prompt | Hướng dẫn hành động trong prefix | Chỉ hỗ trợ thở dài, cười nhẹ, hắng giọng trước câu qua tag; hành động khác báo không hỗ trợ |

Chỉ dẫn của Gemini được tách khỏi transcript: model 3.8 dùng speech metadata; model cũ dùng đoạn hướng dẫn riêng trước transcript. VoxCPM2 dùng một prefix `(Emotion: ...; Pace: ...)`, kết hợp với `tts_style`. Dấu ngoặc trong giá trị direction được đổi sang ngoặc toàn chiều rộng để không làm vỡ prefix. `tts_style` chung và chỉ dẫn từng câu được kết hợp, nên tránh viết mâu thuẫn.

VieNeu nhận tag `[thở dài]`, `[cười]`, `[hắng giọng]`; nhận tên tiếng Anh `sigh`, `chuckle`, `clear throat` trong `direction.nonverbal`, có thể thêm hậu tố `before speaking` / `before the sentence`. Tên tiếng Việt có thể thêm `trước câu` / `trước khi nói`. Không tự suy ra "buồn" phải thở dài hay "vui" phải cười. Punctuation và tag chỉ gợi ý model, không bảo đảm hành động hay tốc độ đúng như mong muốn.

## Kiểm tra và giới hạn

`direction` là object, chỉ nhận `emotion`, `pace`, `emphasis`, `ending`, `nonverbal`. Mọi trường đều tùy chọn. Giá trị phải là string không rỗng; riêng `emphasis` là array các string không rỗng. Sai kiểu/trường lạ báo lỗi trước khi sinh audio. `text_sub` nên có để phát hiện plan lệch bản SRT. `text_tts` có thể rỗng để bỏ đọc câu; thêm chỉ dẫn không tự bật đọc lại câu này.

Yêu cầu không hỗ trợ được báo trên stderr; pipeline dub đưa cảnh báo vào reporter. Preview ghi `speech_directions` và `speech_warnings` trong `review.json`, kể cả khi dùng lại cache. Chỉ dẫn khác nhau của Gemini/VoxCPM2 dùng cache riêng. Với VieNeu, thay đổi trường chưa hỗ trợ không bắt sinh lại audio nếu lời đọc/reference thực tế vẫn giống nhau; cảnh báo vẫn phản ánh yêu cầu mới.

Plan cũ với `text_tts`, `emotion`, `ref_audio` vẫn dùng được. `emotion` cũ chỉ là metadata với VieNeu, là prefix với VoxCPM2. Muốn dùng Gemini theo câu, chuyển sang `direction`. Nếu đồng thời ghi `emotion` cũ và `direction.emotion`, hai giá trị phải giống nhau. Tag VieNeu viết trực tiếp trong `text_tts` vẫn chỉ dành cho VieNeu; để dùng cùng một plan cho cả ba backend, đặt hành động vào `direction.nonverbal`.

Nghe preview để duyệt phát âm, nhấn nhá, giọng nhân vật và độ vừa khung thời gian trước khi chạy toàn video. Kiểm tra request/đầu ra/cache không xác nhận model diễn đúng ý người nghe.
