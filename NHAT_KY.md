# Nhật ký tiến trình — để làm tiếp mà không phải bắt đầu lại

Cập nhật: 16/09/2026. Số liệu chi tiết và lập luận nằm trong `README.md`; file này chỉ
ghi **đang ở đâu, việc gì còn dở, chạy lệnh gì tiếp**, cùng những quyết định và cái bẫy
đã gặp để phiên sau không phải hỏi lại.

## Đang ở đâu

- **Tuần 1–7: xong, đã commit và push.**
- **Tuần 8: phần đo đạc xong.** Mọi tầng đã chấm trên `test` (ROUGE và BERTScore), đã
  commit và push ở `1cea014`.
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

**Việc tiếp theo: giai đoạn 1** — extractive có chủ đích trên `tune`, dùng điểm PhoBERT đã có
sẵn (`results/predictions/phobert-sent-train_20k_{tune,val}_len256_scores.json`), không cần GPU.

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
