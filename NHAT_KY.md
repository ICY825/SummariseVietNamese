# Nhật ký tiến trình — để làm tiếp mà không phải bắt đầu lại

Cập nhật: 17/09/2026. Số liệu chi tiết và lập luận nằm trong `README.md`; file này chỉ
ghi **đang ở đâu, việc gì còn dở, chạy lệnh gì tiếp**, cùng những quyết định và cái bẫy
đã gặp để phiên sau không phải hỏi lại.

## Đang ở đâu

- **Tuần 1–7: xong, đã commit và push.**
- **Tuần 8: phần đo đạc xong.** Mọi tầng đã chấm trên `test` (ROUGE và BERTScore), đã
  commit và push ở `1cea014`.
- **Hướng mới (một hệ thống, đủ ý và không sai sự thật):** giai đoạn 0–3 xong, hệ thống cuối là giai đoạn 1; không dùng
  BARTpho trong hệ thống cuối; tiếp theo giai đoạn 4 (xem mục "HƯỚNG MỚI" bên dưới).
- **Còn lại:** báo cáo và slide (bạn đã chọn để sau), kiểm tra tái lập trên máy sạch.
- Demo Gradio chạy đầu-cuối được trên máy này (kiểm lại lần cuối 16/09 sau mọi thay đổi
  của tuần 8).

Các commit mốc của phiên 16/09/2026, theo thứ tự:

| Commit | Nội dung |
|---|---|
| `4942e35` | tuần 7: kiểm chứng máy chấm bằng 2 người chấm mẫu 12 bài |
| `27d25b7` | tuần 7: phân tích lỗi định tính — bốn họ lỗi trên 190 ghi chú |
| `8580709` | tuần 7 xong: demo Gradio bốn tầng, kernel lấy lại checkpoint từ Kaggle |
| `ae23bfc` | tuần 8: mở khoá `test` bằng cờ tường minh, kernel chấm test |
| `1cea014` | tuần 8: kết quả cuối trên `test` — tầng 2, 3, 4 và BERTScore |

## Kết quả chính đã có (chi tiết: README)

**Kết quả cuối — trên `test`, 2.000 bài, chấm một lần:**

| Hệ thống | ROUGE-1 | BERTScore |
|---|---|---|
| Lead-3 | 27,22 | 85,51 |
| Tầng 2 — PhoBERT `k2` | 31,43 | 86,40 |
| **Tầng 3 — BARTpho `train_20k`** | **34,55** | **87,21** |
| Tầng 4 — lọc `lexrank` rồi viết lại | 34,61 | 87,22 |
| Oracle-3 (trần extractive) | 48,14 | 89,05 |

Lead-3 < tầng 2 < tầng 3 ở cả hai thước đo, mọi cặp p < 0,0001. Tầng 4 không hơn tầng 3
(ROUGE +0,06 p = 0,50; BERTScore +0,01 p = 0,58). Không có ViT5 trên `test` (xem mục 5).

**Kết quả trên `val` và các tuần trước:**

| Hạng mục | Kết quả |
|---|---|
| Tầng 3 | BARTpho `train_20k` 35,23 ROUGE-1 trên `val` — tốt nhất; ViT5 33,40 |
| Tầng 2 | PhoBERT chọn câu, `k2` (2 câu, chọn trên `tune`) 31,00; hơn mốc Lead-2 +1,91 |
| Tầng 4 vòng 1 | lọc câu lúc suy luận không lấy lại được thiệt hại do cắt bài |
| Tầng 4 vòng 2 | huấn luyện lại trên đầu vào đã lọc cũng không (hiệu hai hiệu −0,17 [−2,35, +2,00]) |
| Đối chứng `--no-train` | trùng khít hai bản lọc 898/898 ở nhóm không lọc |
| Tuần 7, máy chấm | 2 lượt LLM chấm 50 bài × 4 hệ thống, alpha 0,76–0,92; BARTpho trôi chảy nhất nhưng thiếu ý và kém trung thực hơn extractive; tương quan với ROUGE-1 chỉ +0,12 |
| Tuần 7, người chấm mẫu | 2 người × 12 bài (48 bản); `day_du`/`trung_thuc` kiểm chứng đạt, `troi_chay` không đạt (máy phóng đại); 9/9 kết luận cùng chiều; người vs ROUGE-1 −0,02 |
| Tuần 7, phân tích lỗi | bốn họ lỗi; hồ sơ lỗi extractive và abstractive bù trừ nhau; ba ví dụ "bịa" rõ nhất đều thuộc sapo |

## HƯỚNG MỚI (16/09/2026) — một hệ thống: đủ ý và không sai sự thật

Bạn quyết định đổi cả đề tài: không chia tầng nữa mà làm **một** hệ thống tóm tắt, với tiêu
chí **đủ ý và không sai sự thật, không cần càng ngắn càng tốt**. Các tầng cũ trở thành phần
phân tích dẫn tới hệ thống đó. Kế hoạch 5 giai đoạn và quy tắc quyết định nằm ở README, mục
"Hướng mới".

**Giai đoạn 0 — XONG, đã commit và push:**
- `src/eval/chinh_xac.py`: ROUGE-1/2 recall, độ phủ chi tiết (tên riêng + con số của sapo),
  tỷ lệ có chi tiết lạ. `src/eval/cham_chinh_xac.py`: script chấm dùng lại cho mọi giai đoạn.
  Selftest mục 8 có 21 phép kiểm tính tay, đều đạt.
- Bộ đo đã kiểm: đối chứng âm 0/1000 cho extractive (sau hai lần sửa do chính phép kiểm này
  phát hiện); đối chứng dương bắt 3/3 lỗi bịa/đổi chi tiết, 0/5 lỗi gán nhầm đối tượng (giới
  hạn đã biết); soi tay 15 cờ BARTpho: 8 lỗi thật, 4 thêm ngoài bài/chưa rõ, 3 bắt nhầm.
- Mốc trên `val`: bảng xếp hạng đảo ngược so với F1 — BARTpho F1 cao nhất nhưng recall 35,6,
  phủ chi tiết 50,9, **11,8% có chi tiết lạ**; Lead-3 phủ chi tiết 67,6 và PhoBERT 3 câu recall
  55,6, cả hai 0% chi tiết lạ.
- Quy tắc đã chốt: chi tiết lạ ≤ 1,0%; ≤ 110 âm tiết trung bình, ≤ 4 câu; phải hơn cả Lead-3
  lẫn PhoBERT 3 câu về độ phủ chi tiết và recall, có ý nghĩa. Dò trên `tune`, xác nhận `val`,
  `test` một lần.

**Giai đoạn 1 — XONG (17/09/2026):** `src/models/chon_cau.py`, chọn câu tham lam theo ngân
sách âm tiết, điểm = PhoBERT + ưu tiên vị trí + LexRank + thưởng phủ ý mới.
- Dò 216 cấu hình trên `tune`; thắng: ngân sách 110, `w_vt` 1, `w_lex` 0,5, `lam` 1, không
  nối tiền đề, có lọc rác.
- `val`: recall 57,2, phủ chi tiết 69,1, **0,0% chi tiết lạ**, 100 âm tiết, F1 28,6. Hơn
  Lead-3 (+2,79 recall, +1,47 phủ) và PhoBERT 3 câu (+1,61, +2,75), mọi khoảng tin cậy không
  chứa 0 → **đạt quy tắc**. Hơn Lead-3 về phủ chi tiết mỏng (cận dưới +0,19, p = 0,021 —
  không qua nếu hiệu chỉnh Bonferroni cho 4 phép so; phải nói rõ khi báo cáo).
- Review 17/09: kiểm lại gán điểm PhoBERT đúng câu, chấm nhanh khớp bộ chấm chính thức,
  22 lần ghép học hàm đều đúng, không rò rỉ nội dung giữa các split — đều đạt.
- `val` đã chấm hai lần; từ giai đoạn 2 chỉ soi lỗi và sửa trên `tune`.
- Lỗi phát hiện trên `val` (guid 14423): cắt câu sau học hàm "TS ." làm mất tên chuyên gia.
  Đã sửa ghép mảnh (TS, GS, PGS, ThS, TSKH, St; cố ý không ghép NSND/NSƯT/CN), selftest
  `models` mục 12, rồi **chạy lại từ `tune`**: cấu hình thắng không đổi, 5/1.000 bản `val` đổi.
- Giới hạn mang sang: 152/1.000 bản có câu treo (câu nối thiếu tiền đề); bộ chấm đếm > 4 câu
  ở 23 bản do tách câu mịn hơn (Lead-3 cũng bị, 8 bản) — không phải vi phạm.
- Bản tóm tắt `val` không commit: sinh lại tất định trong 5 giây (đã kiểm trùng khít).

**Bạn đã quyết (17/09): bỏ BARTpho khỏi hệ thống cuối**, giữ nó làm phần phân tích. Lý do trình
bày với hội đồng: mô hình sinh thắng ROUGE nhưng hỏng tiêu chí không sai sự thật, và hai lần dùng
có kiểm soát đều thất bại. Câu hỏi hội đồng dễ hỏi nhất — "deep learning nằm ở đâu" — đã có số:

- **PhoBERT góp +2,37 recall** (tune, tắt PhoBERT rồi dò lại cả lưới, p < 0,0001); phủ chi
  tiết +1,27 chưa có ý nghĩa. `chon_cau.py do --khong-phobert` rồi `so-phobert`.

**Giai đoạn 2 — XONG (17/09/2026), kết quả: giữ hai phiên bản, giai đoạn 3 chọn.**
- Thử nghiệm âm 1 (`viet_lai.py thu-ghep`): ghép câu BARTpho vào đầu bản chọn câu → giảm cả
  recall lẫn phủ chi tiết; không lọc thì 12,2% chi tiết lạ.
- Thử nghiệm âm 2 (`viet_lai.py ung-vien`/`phan-tich`): BARTpho viết lại 67 câu treo trên tune →
  22 có chi tiết lạ, chỉ 15 giữ ý; lỗi đảo chủ thể/bịa chức vụ lọt bộ đo (#17, #27, #63).
  BARTpho trên CPU máy này tái lập đúng từng ký tự bản Kaggle (6/6).
- Thay bằng luật (`chon_cau.py do-treo`, `sinh --gd 2`): bỏ từ nối ("Tuy nhiên, "...; CỐ Ý giữ
  "Trước đó/Sau đó" vì là dấu thời gian) + `w_treo` 1 trừ điểm câu treo. Chốt trên tune theo
  quy tắc dung sai 0,5 điểm.
- `val`: câu treo 152 → 21; recall 56,9, phủ 68,8, 0% chi tiết lạ. **Trượt một điều**: phủ chi
  tiết hơn Lead-3 +1,18 [−0,14, +2,52], p = 0,08. So giai đoạn 1: −0,21 recall, −0,29 phủ, đều
  không có ý nghĩa.
- `val` giờ đã chấm ba lần (gđ1 hai lần, gđ2 một lần) — không chỉnh gì thêm theo `val`.
- Quy tắc chọn gđ1/gđ2 ở giai đoạn 3 đã chốt trong README (troi_chay hơn có ý nghĩa và
  day_du/trung_thuc không kém quá 0,25). Tuần 7: LLM chấm troi_chay không khớp người.

**Khung trình bày đã chốt (17/09): "phát hiện → giải quyết".** Hướng cũ (tuần 1–8) là phần phát
hiện, hướng mới là phần giải quyết. Kế hoạch giai đoạn 3–4 được mở rộng để lấp bốn chỗ hở (README,
mục "Giai đoạn 3–4 — kế hoạch"):
1. Gđ 3: phiếu chấm gồm gđ1, gđ2, **BARTpho, Lead-3** — người đọc xác nhận cú đảo chiều.
2. Gđ 4: chấm thước đo mới trên `test` cho **mọi hệ thống** (cần mở khoá `test` trong
   `cham_chinh_xac.py`).
3. Gđ 4: demo hệ thống cuối cạnh BARTpho, tô màu chi tiết lạ.
4. Gđ 4: viết lại mở đầu README/báo cáo, thêm câu hỏi nghiên cứu 4, đặt tầng 4 vào mạch.

**Giai đoạn 3 — XONG (17/09).** Bạn quyết: **dùng lại 50 bài tuần 7** (không rút phiếu mới),
máy chấm hai lượt bằng tác tử con, hai người chấm mẫu cũ chấm tiếp; mẫu nhỏ thì giữ quy tắc,
mặc định gđ1.
- `src/eval/cham_gd3.py prepare` đã chạy, phiếu đã commit (`754cd9c`): máy 46 bài / 69 bản
  (59 mới + 10 bản mốc đo độ trôi hai đợt), người 11 bài / 19 bản. `gd3/khoa.json` không commit.
- Quy tắc phân tích chốt trước khi có điểm: README, mục "Giai đoạn 3 — chấm blind".
- **Cả hai lượt máy** do tác tử con chấm, mỗi lượt một thư mục riêng ngoài repo, chỉ đọc
  `phieu_phan1..5.md`. Lượt 1 không do phiên chính chấm vì phiên này đã đọc bản gđ1/gđ2.
- **Phần máy chấm XONG** (`cham_gd3.py analyze`, README "Kết quả phần máy chấm"): hai lượt
  đồng thuận alpha 0,85–0,92. Độ trôi hai đợt: `day_du` −0,55 và `troi_chay` +0,35 **vượt ngưỡng**
  → so chéo đợt hai tiêu chí này chỉ tham khảo; `trung_thuc` −0,25 đạt.
  - Kết luận được: gđ1 **trung thực hơn BARTpho** +0,55, p < 0,0001.
  - Chọn phiên bản theo quy tắc: **gđ1** (16 bài, gđ2 − gđ1 gần 0 ở cả ba tiêu chí).
  - Phát hiện: 7 bản gđ1 bị trừ trung thực vì **ghép câu mất ngữ cảnh** (B43 "9h cùng ngày" đọc
    thành 13/3) → không được nói "không thể sai sự thật", chỉ "không bịa".
  - Chưa giải: máy chấm cho gđ1 kém Lead-3 về đầy đủ (−0,25, nhưng trôi −0,55) trái với chỉ số
    tự động.
- **Bạn quyết: không chấm người mới.** Dùng lại người chấm tuần 7 chỉ cho phần phát hiện và
  làm chỗ dựa cho kết luận trung thực (chuỗi người → máy cũ → máy mới). Phiếu 19 bản vẫn để trống
  trong repo, có người chấm thì bổ sung, `analyze` tự thêm phần 4. Cách nói khi trình bày (được /
  không được) ghi ở README, mục "Quyết định: không chấm người mới".
- **Giai đoạn 3 — XONG. Hệ thống cuối: giai đoạn 1** (`chon_cau.py sinh --gd 1`).

**Giai đoạn 4a — `test` ĐÃ CHẤM, MỘT LẦN (17/09). KHÔNG chấm lại, không sửa hệ thống theo `test`.**
- 9 hệ thống, `results/tables/chinh_xac_test.json` + `_so_cap.json`. Hệ thống cuối: recall 57,4,
  phủ chi tiết 71,1 (cao nhất, kể cả Oracle-3), 0,15% chi tiết lạ, 100 âm tiết, F1 28,7.
- **Đạt cả bốn điều của quy tắc**: hơn Lead-3 +3,39 recall / +2,77 phủ, hơn PhoBERT 3 câu +1,76 /
  +3,93, mọi p < 0,0001. Chỗ mỏng trên `val` (phủ hơn Lead-3) giờ chắc (cận dưới +1,82).
- Đảo chiều lặp lại: BARTpho F1 34,6 cao nhất (sau Oracle) nhưng recall 35,1, phủ 48,4, **10,0%
  chi tiết lạ**.
- 3 cờ chi tiết lạ của hệ thống cuối: 2 bộ đo bắt nhầm ở chỗ nối câu, 1 lỗi thật (cắt câu sau
  "TX." mất tên Cai Lậy) — giới hạn đã ghi, không sửa.

**Việc tiếp theo: giai đoạn 4b–c** — (b) demo hệ thống cuối cạnh BARTpho, tô chi tiết lạ;
(c) viết lại mở đầu README theo khung "phát hiện → giải quyết", thêm câu hỏi nghiên cứu 4.

**Bẫy riêng của hướng mới:** recall tự tăng theo độ dài — mọi so sánh đủ ý phải ở cùng ngân
sách độ dài, không thì "dài hơn" sẽ luôn thắng.

## Việc còn dở — theo thứ tự nên làm

### 1. Báo cáo và slide (tuần 8) — **chưa làm, bạn đã chọn để sau**

Chưa chốt định dạng (đã hỏi: Markdown trong `report/`, Word, hay LaTeX). Nguyên liệu đã
đủ và đã kiểm chứng trong README: ba câu hỏi nghiên cứu đều có câu trả lời trên `test`.

### 2. Kiểm tra tái lập trên máy sạch (tuần 8)

Không làm được trên máy hiện tại, vì máy này đã cài sẵn mọi môi trường và có sẵn
checkpoint. Cần một máy/một thư mục clone mới, làm theo đúng README từ đầu.

### 3. Các quyết định còn treo về demo — **chưa làm, chỉ mới bàn**

- **Chia sẻ demo.** Hiện `app/app.py` gọi `demo.launch()` mặc định nên chỉ chạy ở
  `127.0.0.1:7860` — chỉ máy này xem được. Ba cách đã bàn: (1) mạng nội bộ
  `server_name="0.0.0.0"`; (2) link công khai tạm `share=True`, hết hạn khoảng 72 giờ, máy
  phải bật suốt; (3) Hugging Face Spaces, lâu dài nhưng phải tải lên 2,46 GB checkpoint.
- **Tải file lên.** Demo hiện **chỉ nhận văn bản dán vào** (`gr.Textbox`). Nếu thêm, khuyến
  nghị chỉ `.txt` và `.docx`; bỏ PDF vì tiếng Việt dễ vỡ dấu khi đọc. Nhớ ghi chú trên giao
  diện: mô hình huấn luyện cho tin tức và tầng 3 chỉ đọc 1.024 token đầu.
- **Bài mẫu trong demo chưa làm nổi bật khác biệt.** Với bài mẫu hiện tại, tầng 3 và tầng 4
  ra **giống hệt nhau** (bài ngắn hơn 1.024 token nên tầng 4 không lọc gì — đúng thiết kế),
  và Lead-3 trùng LexRank (trùng hợp vì bài ít câu). Đề xuất: chọn từ `test` một bài dài mà
  bốn tầng khác nhau rõ, thay làm bài mẫu.
- **Khi trình bày:** bấm thử một lần trước — lần đầu 22,6 s (nạp mô hình), các lần sau 4,2 s.

### 4. Nợ cũ (không bắt buộc)

- ViT5 `train_2k` (~22 phút GPU) — **đẩy lên kernel riêng**, tuyệt đối không đẩy vào
  `dl-summarisevn-vit5`: làm thế là ghi đè output và mất checkpoint BARTpho trên Kaggle
  (bản sao cục bộ vẫn còn, xem mục "File quan trọng không nằm trong git").
- Đường cong học cho BARTpho (~2 giờ GPU).
- Sinh lại vòng 2 tầng 4 từ `checkpoint-2500` (~15 phút GPU) để loại trừ ảnh hưởng chọn
  epoch 3 so với epoch 2. Checkpoint vẫn còn trong output kernel `dl-summarisevn-tang4-train`.

## Đã xong trong phiên 16/09/2026 — ghi lại cách làm

### Người chấm mẫu (tuần 7)

Hai người đã chấm đủ 48/48 bản, `so-sanh` đã chạy, kết luận đã viết vào README (mục
"Kiểm chứng máy chấm bằng một mẫu người chấm"). Kết quả ở
`results/tables/nguoi_vs_may_val.json`. `day_du` và `trung_thuc` đạt chuẩn kiểm chứng đã
chốt trước (alpha người–máy 0,789 và 0,764, **cao hơn** alpha người–người 0,747 và 0,568);
`troi_chay` **không** đạt (0,504 so với 0,585) — máy phóng đại khác biệt trôi chảy, nên phát
biểu về trôi chảy chỉ được nêu theo máy chấm. Cả 9 kết luận chính cùng chiều. Người chấm
tương quan với ROUGE-1 −0,02 và BERTScore −0,12 (n = 36).

**Ba phiếu `cham_nguoi1..3.csv` đã bỏ có chủ ý.** Đó là phiếu trống của phương án "3 người
chấm đủ 50 bài" — không thực hiện. Hệ quả: `human_eval.py analyze` mặc định tìm
`cham_nguoi*.csv` nên **không còn đầu vào**. Nếu cần, lấy lại từ commit `2638a78`:

```bash
git show 2638a78:results/human_eval/cham_nguoi1.csv > results/human_eval/cham_nguoi1.csv
```

### Phân tích lỗi định tính (tuần 7)

Mục "Phân loại lỗi — bốn họ, phân bố rất khác nhau theo tầng" trong README, dựng trên 190
ghi chú (115 của hai lượt máy, 75 của hai người chấm) ghép hệ thống qua `khoa.json`. Bốn
họ: thiếu ý chính, sai sự thật, mạch văn đứt, rác kế thừa từ dữ liệu. Không được gộp ba
tiêu chí thành một điểm vì hai hồ sơ lỗi bù trừ nhau.

### Demo Gradio (tuần 7)

`app/pipeline.py` (đường suy luận) và `app/app.py` (giao diện). Chạy:

```bash
~/.venvs/demo/Scripts/python.exe app/app.py     # http://127.0.0.1:7860
```

Cả bốn tầng chạy với trọng số thật; `tom_tat_tat_ca` trả 5/5 tầng; máy chủ trả HTTP 200. Ba
chốt chặn đã kiểm cả hai chiều: trọng số 0 byte, tách từ hỏng, và chọn nhầm bản checkpoint
hỏng khi trên máy có nhiều bản.

### Chấm `test` (tuần 8)

- `test` mở bằng cờ `--cho-phep-test` ở `vit5.py` và `phobert_select.py`; chốt từ chối vẫn
  còn cho mọi lần chạy không có cờ.
- Tầng 2 chạy tại máy: `score --split test --cho-phep-test` (401 s CPU) rồi lệnh mới
  `cham-test --cho-phep-test`, lệnh này **đọc** quy tắc thắng từ bảng dò trên `tune` (không
  dò lại trên `test`) và từ chối ghi đè.
- Tầng 3–4 chạy bằng kernel `dl-summarisevn-test-final` (khoảng 60 phút từ lúc đẩy tới lúc
  xong, gồm cả thời gian xếp hàng; hai cấu hình).
- BERTScore: 9 hệ thống × 2.000 bài trong `~/.venvs/torch`, khoảng 45 phút CPU. Script chỉ
  tự so với Lead-3; các cặp khác tính thêm bằng `paired_bootstrap` trên `per_article`.
- **Không có ViT5 trên `test`.** Checkpoint ViT5 `train_20k` bị ghi đè khi chính kernel
  `dl-summarisevn-vit5` huấn luyện BARTpho (12/09). Đã kiểm mọi đường: output mới nhất chỉ
  có BARTpho, CLI không lấy được version cũ, trên đĩa không có checkpoint ViT5 nào. Huấn
  luyện lại tốn ~232 phút GPU mà không câu hỏi nghiên cứu nào cần; kết luận BARTpho > ViT5
  dẫn từ bảng `val`. Nếu muốn làm lại: phải đẩy vào **kernel riêng**.

## Tuyệt đối không làm

- **Không** chạy lại `human_eval.py prepare` hay `prepare-mau` — cả hai từ chối khi
  `khoa.json` / `mau.json` đã có; xoá chúng để chạy lại là làm mất khớp với phiếu đã phát.
- **Không** sửa `phieu_doc.html` hay `phieu_doc_mau.html` sau khi đã phát —
  `analyze`/`so-sanh` so mã băm và sẽ từ chối chạy.
- **Không** sửa điểm trong `llm_judge.csv` hay `llm_judge_2.csv` sau khi đã thấy lượt kia
  hoặc điểm người — sẽ mất tính độc lập.
- **Không** điền điểm máy vào `cham_nguoi*.csv` hay `cham_mau_nguoi*.csv` — đó là phiếu người.
- **Không chấm lại `test`.** `test` đã dùng đúng một lần; mọi kết quả đã commit. Các lệnh
  chấm từ chối ghi đè bảng cũ — đừng xoá bảng để chạy lại.
- **Không đẩy `notebooks/kernel-metadata.json`**, và **không chạy `kaggle_push.py` mà thiếu
  `--dir`**: `--dir` mặc định là `notebooks/`, tức kernel `dl-summarisevn-vit5` chứa
  checkpoint BARTpho. Đẩy nhầm là huấn luyện lại và ghi đè mất checkpoint trên Kaggle.

## File quan trọng không nằm trong git

- **Checkpoint (2,46 GB)** — BARTpho `train_20k` 1.911.406.472 byte, tầng 2 540.015.440 byte
  kèm `head.pt`. Có ba bản:
  - `~/.cache/dl-summarisevn/xuat/ckpt/{bartpho,tang2}` — bản demo đang dùng
  - `~/Documents/dl-summarisevn-checkpoints/{bartpho,tang2}` — **bản sao lưu**, đối chiếu từng byte
  - trên Kaggle, output kernel `dl-summarisevn-vit5` và `dl-summarisevn-tang2`
- `~/.cache/dl-summarisevn/{bartpho,tang2}/` là **bản tải hỏng** (trọng số 0 byte) từ lần
  dùng `kernels output`. Vô hại vì `tim_checkpoint()` ưu tiên bản có trọng số thật; xoá được.
- `results/human_eval/khoa.json` — khoá chấm blind, cố ý bỏ khỏi git (`.gitignore`) để
  người chấm đọc repo không biết nhãn nào là hệ thống nào. Nó **vẫn nằm trên máy**. Nếu mất:
  `prepare` tất định — chạy lại **vào một thư mục khác** (đặt `human_eval.OUT`) ra đúng khoá
  (đã kiểm từng byte), rồi chép `khoa.json` về.
- File dự đoán của baseline (`results/predictions/baselines_*`) — bỏ khỏi git có chủ ý,
  sinh lại bằng `run_baselines.py` trên CPU.

## Môi trường

| Môi trường | Dùng cho |
|---|---|
| `.venv` (dự án) | mọi lệnh thường: `run_baselines.py`, `phobert_select.py select`/`cham-test`, `human_eval.py`, selftest. Cố ý **không** có `torch` |
| `~/.venvs/torch` | cần `torch`/`transformers`: `run_bertscore.py`, `tang4_sosanh.py`, `phobert_select.py score`. Đã cài thêm `sentencepiece` |
| `~/.venvs/demo` | demo Gradio: torch 2.14.0+cpu, transformers 5.0.0, gradio 6.27.0, underthesea 9.5.0, sentencepiece |
| `~/.venvs/kaggle` | Kaggle CLI 2.2.4: `notebooks/kaggle_push.py`, `kaggle kernels status/output`. Token ở `~/.kaggle/access_token` |

Tự kiểm tra sau mỗi lần sửa code (vài giây, không cần mạng):

```bash
.venv/Scripts/python.exe src/eval/selftest.py
.venv/Scripts/python.exe src/models/selftest.py
```

## Kernel Kaggle (tài khoản `minh12605`)

| Kernel | Nội dung | Trạng thái |
|---|---|---|
| `dl-summarisevn-vit5` | checkpoint BARTpho `train_20k` gốc (version mới nhất) — **đừng đẩy đè** | xong |
| `dl-summarisevn-tang2` | checkpoint tầng 2 PhoBERT | xong |
| `dl-summarisevn-tang2-score` | điểm câu tầng 2 trên `tune`/`val` | xong, đã tải về |
| `dl-summarisevn-tang-4` | đối chứng `--no-train` không lọc | xong, đã tải về |
| `dl-summarisevn-tang4-train` | tầng 4 vòng 2, có `checkpoint-2500` và `3750` | xong, đã tải về |
| `dl-summarisevn-xuat-ckpt` | CPU; chép checkpoint BARTpho và tầng 2 ra `/kaggle/working` để tải về | xong, đã tải về |
| `dl-summarisevn-test-final` | GPU; tầng 3 và tầng 4 trên `test` | xong, đã tải về |

## Bẫy đã gặp (đỡ mất thời gian lần sau)

- **`guid` đánh số riêng theo split.** Trùng `guid` giữa `val` và `train` không có nghĩa là
  rò rỉ (đã kiểm: 181/181 là bài khác nhau).
- **File dự đoán của `vit5.py` lưu `guid` dạng số**, bảng và split dùng chuỗi — so thẳng hai
  danh sách sẽ báo lệch oan; đổi sang chuỗi thì khớp tuyệt đối.
- **Windows ghi `\r\n`.** So mã băm file với chuỗi Python sẽ lệch; so văn bản đã đọc, hoặc
  so file với file. Git tự chuyển về LF (`.gitattributes`).
- **PowerShell + `python -c "..."`** vỡ khi code có ngoặc nhọn/nháy → dùng
  `@'...'@ | python -`, hoặc ghi script ra file.
- **Commit message pipe từ PowerShell** bị dính BOM ở đầu → viết message ra file rồi
  `git commit -F <file>`.
- **Console Windows cp1252** chết khi in ký tự Việt hoặc dấu trừ U+2212 → đặt
  `PYTHONIOENCODING=utf-8`.
- **Mất mạng khi chạy BERTScore** (`getaddrinfo failed`) → đặt `HF_HUB_OFFLINE=1` và
  `TRANSFORMERS_OFFLINE=1`, mô hình đọc từ cache.
- **Tên file của `run_baselines.py` có `k`**: `--k 2 --systems leadk` ghi `baselines_val_k2_leadk.json`.
- **Kaggle `kernels logs` rỗng** với phiên đã xong; kết quả nằm trong `_run.json` của output.
- **`kaggle kernels output` trả file trọng số 0 BYTE.** Mọi file nhỏ tải về đầy đủ và
  đúng, riêng `model.safetensors` rỗng — không báo lỗi gì. Với BARTpho nó còn bỏ sót hẳn
  thư mục `final/`. Trọng số vẫn còn bên Kaggle (các kernel đọc qua `kernel_sources` chạy
  bình thường), nên cách lấy về là kernel CPU `dl-summarisevn-xuat-ckpt` chép sang
  `/kaggle/working` rồi tải — **không phải huấn luyện lại**. Đừng bao giờ kết luận "mất
  checkpoint" từ việc `from_pretrained` báo lỗi; kiểm **kích thước** file trước.
- **`kaggle kernels output <id>/<version>` bỏ qua số version**, luôn trả bản mới nhất (đã
  thử xin version 999 vẫn ra file). CLI không lấy được output của version cũ.
- **`kernel_sources` chỉ gắn version mới nhất**, và ở chế độ chỉ-đọc — đọc qua nó không làm
  hỏng kernel nguồn.
- **Tải file lớn từ Kaggle hay đứt** (`IncompleteRead`), và pip cũng đứt (`getaddrinfo
  failed`) khi mạng chập chờn. Cứ chạy lại; pip nối tiếp được phần đã tải. Chỉ tải file cần
  bằng `--file-pattern` (ví dụ chỉ zip kết quả) cho nhẹ.
- **`du -sh` báo GiB chứ không phải GB** — 2,3G của `du` là 2,46 GB.
- **gradio 6 đã bỏ `show_copy_button`** của `gr.Textbox`. `gr.Examples` là **hàm** chứ không
  phải class, nên `inspect.signature(gr.Examples.__init__)` trả `(*args, **kwargs)` — đọc
  chữ ký của chính `gr.Examples`.
- **Schema file kết quả:** `length` có khoá `mean_syllables` (không phải `mean`); `novel` có
  `1gram`/`2gram`/`4gram`; chỉ số ROUGE/BERTScore trong `corpus` là dict `{mean, lo, hi}`.
  In cấu trúc ra trước khi viết code đọc số.

## Mở phiên làm việc mới

Nhắn cho Claude, ví dụ:

> Đọc `NHAT_KY.md` và phần "Tiến độ" trong `README.md`, kiểm tra lại trạng thái hiện tại
> (git log, selftest), rồi làm tiếp việc số … trong "Việc còn dở".

Lịch sử hội thoại với Claude Code cũng được lưu cục bộ trên máy; mở lại phiên gần nhất bằng
`claude --continue`, hoặc chọn phiên cũ bằng `claude --resume`. Bản lưu đó chỉ nằm trên máy
này — nhật ký này (đã push lên GitHub) mới là bản không mất.
