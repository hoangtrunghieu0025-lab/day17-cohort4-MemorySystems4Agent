# Phân tích kết quả – Day 17: Memory Systems for AI Agent

Số liệu lấy từ `python src/benchmark.py` (chế độ offline, deterministic; ngưỡng compact = 900 token, giữ 4 message gần nhất).
Token được ước lượng bằng `len(text)/4`, nên con số tuyệt đối chỉ mang tính tương đối – điều quan trọng là *tỉ lệ* giữa hai agent.

**Standard Benchmark** (10 hội thoại × ~10 lượt, user `dungct`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2444 | 12862 | 0.00 | 0.15 | 0    | 0 |
| Advanced | 2634 | 27343 | 1.00 | 1.00 | 1109 | 0 |

**Long-Context Stress Benchmark** (1 hội thoại 14 lượt rất dài, user `dungct_stress`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline | 2516 | 21607 | 0.00 | 0.15 | 0   | 0 |
| Advanced | 2580 | 11299 | 1.00 | 1.00 | 692 | 3 |

## 1. Vì sao Advanced recall tốt hơn Baseline

Recall được đo ở **thread mới**. Baseline chỉ giữ message theo `thread_id`, nên sang thread mới nó không còn gì → recall 0 (nó trả lời "chưa có thông tin", không đoán). Advanced trích fact ổn định (tên, nơi ở, nghề, đồ uống, style…) từ mỗi lượt và ghi vào `User.md` theo `user_id`; thread mới vẫn đọc lại được profile → recall 1.0, kể cả với correction (Đà Nẵng→Huế, backend→MLOps) và nhiễu (Hà Nội chỉ là nơi đi họp, "product manager" chỉ là câu đùa).

Ba lớp memory được tách bạch:

| Lớp | Nằm ở đâu | Sống bao lâu | Chứa gì |
|---|---|---|---|
| short-term | `CompactMemoryManager.messages` | trong 1 thread | vài message gần nhất, nguyên văn |
| compact | `CompactMemoryManager.summary` | trong 1 thread | mỗi lượt cũ một câu chính, tối đa 6 dòng (lossy) |
| persistent | `state/profiles/<user>/User.md` | qua mọi thread | fact ổn định, có confidence/mentions/recency |

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

Ở Standard, `Prompt tokens processed` của Advanced **cao hơn 112%**. Hội thoại chỉ ~10 lượt nên chưa vượt ngưỡng compact (0 compaction), tức là Advanced vẫn phải mang toàn bộ lịch sử *và* trả thêm chi phí cố định mỗi lượt: system prompt dài hơn + `User.md` được inject vào prompt (`User.md` lớn dần lên tới vài trăm token). `Agent tokens only` cũng nhỉnh hơn (+7.8%) vì mỗi lần ghi memory agent in thêm dấu vết `[User.md đã cập nhật: …]`. Memory không miễn phí: nó chỉ có lãi khi recall xuyên phiên đáng giá hơn khoản overhead này.

## 3. Vì sao compact giúp Advanced ở hội thoại dài

Ở Baseline, mỗi lượt gửi lại toàn bộ lịch sử nên chi phí prompt tăng **bậc hai** theo độ dài thread. Trong stress test, mỗi lượt dài ~200 token nên Baseline xử lý 21.607 token prompt. Advanced nén lịch sử cũ thành summary khi vượt ngưỡng (3 lần compact), nên context mỗi lượt bị chặn trên (≈ system + `User.md` + summary + 4 message gần nhất) → còn 11.299 token (**-47,7%**), trong khi recall vẫn 1.0 vì thông tin quan trọng nằm ở `User.md`, không phụ thuộc phần lịch sử đã bị nén.

Compact **chỉ tối ưu `Prompt tokens processed`**. `Agent tokens only` gần như không đổi (2516 vs 2580, Advanced vẫn nhỉnh hơn một chút) vì chỉ số này đếm lượng chữ thực sự trao đổi với người dùng; compact không làm user nói ít đi hay agent trả lời ngắn lại, nó chỉ làm agent *đọc lại* ít hơn.

## 4. File memory tăng trưởng ra sao và rủi ro

- `User.md` của `dungct` lớn thêm 1109 byte sau 10 hội thoại; của `dungct_stress` 692 byte sau 14 lượt rất dài. Tăng trưởng theo **số fact**, không theo số chữ nói ra, và bị chặn trên: tối đa 6 interest, 6 style tag, 5 giá trị bị thay thế. Nhờ đó file đi ngang thay vì phình vô hạn.
- Nhưng mỗi byte trong file là token phải trả **ở mọi lượt** (xem mục 2), nên "file nhỏ" cũng là yêu cầu về chi phí, không chỉ về dung lượng đĩa.
- **Lưu sai fact**: extractor là regex/heuristic. Nó có thể nhận nhầm câu đùa, câu giả định hoặc câu hỏi thành fact (đã chặn các ca trong dataset, nhưng không thể bao phủ mọi cách nói). Fact sai bị ghi vào `User.md` sẽ *lặp lại ở mọi phiên sau* nên lỗi bền hơn lỗi trong một thread.
- **Summary là lossy**: mỗi lượt cũ chỉ còn một câu đầu (≤110 ký tự). Chi tiết nằm ngoài `User.md` và ngoài 4 message gần nhất (ví dụ con số cụ thể trong tin tức) sẽ mất sau compact. Đây là chủ ý, nhưng là rủi ro thật nếu use case cần chi tiết cũ.
- **Bảo mật/quyền riêng tư**: `User.md` lưu văn bản do người dùng nói và được đưa vào prompt ở các phiên sau, nên cần quyền xoá/sửa cho user, và cần coi nội dung đó là dữ liệu không đáng tin (nguy cơ prompt injection bền vững).

## 5. Bonus đã làm

| Bonus | Giải quyết vấn đề gì | Hiệu quả | Rủi ro mới |
|---|---|---|---|
| **Confidence threshold** (`CONFIDENCE_THRESHOLD = 0.6`) | Không ghi fact mơ hồ/giả định vào `User.md` | Câu hỏi không bao giờ ghi file (test `test_questions_never_write_to_user_md`); câu đùa → conf 0; câu có "có thể/nếu/thử…" đứng *trước* fact bị trừ điểm → không ghi. Giữ `User.md` sạch hơn, tiết kiệm token | Ngưỡng cứng có thể bỏ sót fact đúng nhưng nói dè dặt (false negative → recall giảm) |
| **Conflict handling** | Correction mới phải thay fact cũ, không giữ song song | Fact đơn trị (nơi ở, nghề…) được thay thế; giá trị cũ chỉ còn trong mục `Superseded` (audit) và **không** được inject vào prompt. Câu phủ định "không còn ở/làm X" bị bỏ qua. Câu mới yếu hơn nhiều không đè được fact chắc chắn | Luật "mới thắng" có thể bị lợi dụng bằng một câu nói sai có vẻ chắc chắn; mục `Superseded` tốn thêm vài chục byte |
| **Memory decay** (half-life 40 lần cập nhật) | File không phình vô hạn với sở thích cũ | Fact đa trị (interest/style) có điểm = confidence × (1+log2 mentions) × 0.5^(tuổi/40); vượt giới hạn thì loại điểm thấp nhất. Fact lặp lại nhiều (Python) được giữ, sở thích nhất thời bị rụng (test `test_memory_decay_prunes_stale_low_value_interests`) | Có thể rụng mất một sở thích hiếm nhưng quan trọng; fact đơn trị (tên, nơi ở) cố ý **không** decay |
| **Entity extraction có cấu trúc** | Tách fact thành field thay vì ghi cả câu | Mỗi fact có `key`, `value`, `conf`, `n`, `seq`; style được chuẩn hoá thành tag (`ngắn gọn`, `3 bullet`, `ví dụ thực tế`) nên correction/tổng hợp style không làm mất tag cũ; `User.md` đọc/ghi hai chiều và vẫn sửa tay được qua `edit_text` | Schema cố định (8 loại fact); thông tin ngoài schema không được nhớ |

## 6. Giới hạn cần nói thẳng

- Extractor và các regex được tinh chỉnh khi nhìn dataset này, nên recall 1.0 nói về *dataset này*, chưa phải khả năng khái quát cho tiếng Việt tự do. Muốn đo khái quát cần dataset held-out hoặc extractor dùng LLM.
- `Response quality` ở offline là heuristic (70% độ phủ fact, 15% ngắn gọn, 15% không từ chối), không phải đánh giá bằng judge. `judge_model` đã có trong config để mở rộng sang LLM-as-judge.
- Đường live (`--live`) được kiểm tra bằng model giả (`test_live_agents_wire_up_with_a_fake_model`), xác nhận tool, dynamic prompt và summarization middleware dựng được trên LangChain 1.2. Nó **chưa được chạy với API thật** của bất kỳ provider nào (không có key trong môi trường làm bài).
