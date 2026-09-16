# Nhật ký tiến trình — để làm tiếp mà không phải bắt đầu lại

Cập nhật: 16/09/2026. Số liệu chi tiết và lập luận nằm trong `README.md`; file này chỉ
ghi **đang ở đâu, việc gì còn dở, chạy lệnh gì tiếp**.

## Đang ở đâu

- **Tuần 1–6: xong và đã commit.** Commit tuần 6: `cfcc199`.
- **Tuần 7: đang làm dở.** Commit ngay sau file này chứa toàn bộ phần tuần 7 đến thời
  điểm dừng (chưa push lên GitHub).
- Tuần 8 (báo cáo, chấm `test`, kiểm tái lập) chưa bắt đầu.

## Kết quả chính đã có (chi tiết: README)

| Hạng mục | Kết quả |
|---|---|
| Tầng 3 | BARTpho `train_20k` 35,23 ROUGE-1 trên `val` — tốt nhất; ViT5 33,40 |
| Tầng 2 | PhoBERT chọn câu, `k2` (2 câu, chọn trên `tune`) 31,00; hơn mốc Lead-2 +1,91 |
| Tầng 4 vòng 1 | lọc câu lúc suy luận không lấy lại được thiệt hại do cắt bài |
| Tầng 4 vòng 2 | huấn luyện lại trên đầu vào đã lọc cũng không (hiệu hai hiệu −0,17 [−2,35, +2,00]) |
| Đối chứng `--no-train` | trùng khít hai bản lọc 898/898 ở nhóm không lọc |
| Tuần 7, máy chấm | 2 lượt LLM chấm 50 bài × 4 hệ thống, alpha 0,76–0,92; BARTpho trôi chảy nhất nhưng thiếu ý và kém trung thực hơn extractive; tương quan với ROUGE-1 chỉ +0,12 |
| Tuần 7, người chấm mẫu | 2 người × 12 bài (48 bản); `day_du`/`trung_thuc` kiểm chứng đạt, `troi_chay` không đạt (máy phóng đại); 9/9 kết luận cùng chiều; người vs ROUGE-1 −0,02 |

## Việc còn dở — theo thứ tự nên làm

### 1. Người chấm mẫu (tuần 7) — **XONG** 16/09/2026

Hai người đã chấm đủ 48/48 bản, `so-sanh` đã chạy, kết luận đã viết vào README (mục
"Kiểm chứng máy chấm bằng một mẫu người chấm" và mục "Tiến độ"). Kết quả ở
`results/tables/nguoi_vs_may_val.json`. Tóm tắt: `day_du` và `trung_thuc` đạt chuẩn kiểm
chứng đã chốt trước (alpha người–máy 0,789 và 0,764, **cao hơn** alpha người–người 0,747
và 0,568); `troi_chay` **không** đạt (0,504 so với 0,585) — máy phóng đại khác biệt trôi
chảy, ba so cặp máy thấy có ý nghĩa mà người thì không, nên phát biểu về trôi chảy chỉ
được nêu theo máy chấm. Cả 9 kết luận chính cùng chiều. Người chấm tương quan với ROUGE-1
−0,02 và BERTScore −0,12 (n = 36), củng cố câu trả lời cho câu hỏi 3.

**Ba phiếu `cham_nguoi1..3.csv` đã bỏ có chủ ý** (16/09/2026). Đó là phiếu trống của
phương án "3 người chấm đủ 50 bài" — phương án không thực hiện, vì 50 bài được chấm bằng
hai lượt mô hình ngôn ngữ rồi kiểm chứng bằng phiếu mẫu 12 bài nói trên. Hệ quả cần nhớ:
`human_eval.py analyze` mặc định tìm `cham_nguoi*.csv` nên **không còn đầu vào**. Nếu sau
này cần, lấy phiếu trống ra từ commit `2638a78`:

```bash
git show 2638a78:results/human_eval/cham_nguoi1.csv > results/human_eval/cham_nguoi1.csv
```

### 2. Phân tích lỗi định tính (tuần 7) — **XONG** 16/09/2026

Đã viết mục "Phân loại lỗi — bốn họ, phân bố rất khác nhau theo tầng" trong README, dựng
trên 190 ghi chú (115 của hai lượt máy, 75 của hai người chấm) ghép hệ thống qua
`khoa.json`. Bốn họ: thiếu ý chính, sai sự thật, mạch văn đứt, rác kế thừa từ dữ liệu.
Kết quả chính: hồ sơ lỗi của extractive và abstractive **bù trừ nhau** (extractive 0–4%
sai sự thật nhưng hỏng mạch văn; BARTpho trôi chảy nhưng thiếu ý ở 45–54% số bản và là
tầng duy nhất hoán đổi chi tiết), nên không được gộp ba tiêu chí thành một điểm. Ba ví dụ
"bịa" rõ nhất (B31, B11, B37) đều thuộc **sapo**, tức thói thêm thông tin ngoài bài là học
từ dữ liệu. Mọi số trong mục đều tính lại được từ bốn file phiếu và `khoa.json`.

### 3. Demo Gradio (tuần 7) — **XONG** 16/09/2026

`app/pipeline.py` (đường suy luận) và `app/app.py` (giao diện). Chạy:

```bash
~/.venvs/demo/Scripts/python.exe app/app.py     # http://127.0.0.1:7860
```

Đã kiểm bằng chạy thật, không suy diễn: cả bốn tầng chạy với **trọng số thật** (tầng 2
16,9 s, tầng 3 18,6 s, tầng 4 6,1 s kể cả nạp mô hình), `tom_tat_tat_ca` trả 5/5 tầng,
máy chủ Gradio lên trong 1,2 s và trả HTTP 200 với đúng 6 textbox trong `/config`.

Checkpoint nằm **ngoài repo** ở `~/.cache/dl-summarisevn/` (2,46 GB) — xem README mục demo
để biết cách lấy lại. Môi trường `~/.venvs/demo` (gradio 6.27.0, underthesea 9.5.0).

Ba chốt chặn đã kiểm cả chiều thuận lẫn chiều nghịch: trọng số 0 byte, tách từ hỏng, và
chọn nhầm bản checkpoint hỏng khi trên máy có nhiều bản.

### 4. Nợ cũ (không bắt buộc)

- ViT5 `train_2k` (~22 phút GPU) — **phải chấm `test` trước**, hoặc đẩy lên kernel riêng,
  vì đẩy lên `dl-summarisevn-vit5` làm checkpoint BARTpho không còn lấy được qua
  `kernel_sources`.
- Đường cong học cho BARTpho (~2 giờ GPU).
- Sinh lại vòng 2 tầng 4 từ `checkpoint-2500` (~15 phút GPU) để loại trừ ảnh hưởng chọn
  epoch 3 so với epoch 2. Checkpoint vẫn còn trong output kernel `dl-summarisevn-tang4-train`.

### 5. Tuần 8

Chấm mọi tầng trên `test`, viết báo cáo, kiểm tra tái lập.

## Tuyệt đối không làm

- **Không** chạy lại `human_eval.py prepare` hay `prepare-mau` — cả hai từ chối khi
  `khoa.json` / `mau.json` đã có; xoá chúng để chạy lại là làm mất khớp với phiếu đã phát.
- **Không** sửa `phieu_doc.html` hay `phieu_doc_mau.html` sau khi đã phát —
  `analyze`/`so-sanh` so mã băm và sẽ từ chối chạy.
- **Không** sửa điểm trong `llm_judge.csv` hay `llm_judge_2.csv` sau khi đã thấy lượt kia
  hoặc điểm người — sẽ mất tính độc lập.
- **Không** điền điểm máy vào `cham_nguoi*.csv` hay `cham_mau_nguoi*.csv` — đó là phiếu người.

## File quan trọng không nằm trong git

- `results/human_eval/khoa.json` — khoá chấm blind, cố ý bỏ khỏi git (`.gitignore`) để
  người chấm đọc repo không biết nhãn nào là hệ thống nào. Nó **vẫn nằm trên máy**. Nếu mất:
  `prepare` tất định — chạy lại **vào một thư mục khác** (đặt `human_eval.OUT`) ra đúng khoá
  (đã kiểm từng byte), rồi chép `khoa.json` về.
- File dự đoán của baseline (`results/predictions/baselines_*`) — bỏ khỏi git có chủ ý,
  sinh lại bằng `run_baselines.py` trên CPU.

## Môi trường

| Môi trường | Dùng cho |
|---|---|
| `.venv` (dự án) | mọi lệnh thường: `run_baselines.py`, `phobert_select.py select`, `human_eval.py`, selftest |
| `~/.venvs/torch` | cần `torch`/`transformers`: `run_bertscore.py`, `tang4_sosanh.py`. Đã cài thêm `sentencepiece` (tokenizer BARTpho) |
| `~/.venvs/kaggle` | Kaggle CLI: `notebooks/kaggle_push.py`, `kaggle kernels status/output` |

Tự kiểm tra sau mỗi lần sửa code (vài giây, không cần mạng):

```bash
.venv/Scripts/python.exe src/eval/selftest.py
.venv/Scripts/python.exe src/models/selftest.py
```

## Kernel Kaggle (tài khoản `minh12605`)

| Kernel | Nội dung | Trạng thái |
|---|---|---|
| `dl-summarisevn-vit5` | checkpoint BARTpho `train_20k` gốc (version mới nhất) | xong |
| `dl-summarisevn-tang2` | checkpoint tầng 2 PhoBERT | xong |
| `dl-summarisevn-tang2-score` | điểm câu tầng 2 trên `tune`/`val` | xong, đã tải về |
| `dl-summarisevn-tang-4` | đối chứng `--no-train` không lọc | xong, đã tải về |
| `dl-summarisevn-tang4-train` | tầng 4 vòng 2, có `checkpoint-2500` và `3750` | xong, đã tải về |

## Bẫy đã gặp (đỡ mất thời gian lần sau)

- **`guid` đánh số riêng theo split.** Trùng `guid` giữa `val` và `train` không có nghĩa là
  rò rỉ (đã kiểm: 181/181 là bài khác nhau).
- **Windows ghi `\r\n`.** So mã băm file với chuỗi Python sẽ lệch; so văn bản đã đọc, hoặc
  so file với file. Git tự chuyển về LF (`.gitattributes`).
- **PowerShell + `python -c "..."`** vỡ khi code có ngoặc nhọn/nháy → dùng
  `@'...'@ | python -`, hoặc ghi script ra file.
- **Commit message pipe từ PowerShell** bị dính BOM ở đầu → viết message ra file rồi
  `git commit -F <file>`.
- **Mất mạng khi chạy BERTScore** (`getaddrinfo failed`) → đặt `HF_HUB_OFFLINE=1` và
  `TRANSFORMERS_OFFLINE=1`, mô hình đọc từ cache.
- **Tên file của `run_baselines.py` có `k`**: `--k 2 --systems leadk` ghi `baselines_val_k2_leadk.json`.
- **Kaggle `kernels logs` rỗng** với phiên đã xong; kết quả nằm trong `_run.json` của output.
- **`kaggle kernels output` trả file trọng số 0 BYTE.** Mọi file nhỏ tải về đầy đủ và
  đúng, riêng `model.safetensors` rỗng — không báo lỗi gì. Với BARTpho nó còn bỏ sót hẳn
  thư mục `final/`. Trọng số vẫn còn bên Kaggle (các kernel đọc qua `kernel_sources` chạy
  bình thường), nên cách lấy về là kernel CPU `dl-summarisevn-xuat-ckpt` chép sang
  `/kaggle/working` rồi tải — **không phải huấn luyện lại**. Hệ quả: đừng bao giờ kết luận
  "mất checkpoint" từ việc `from_pretrained` báo lỗi; kiểm **kích thước** file trước.
- **Tải file lớn từ Kaggle hay đứt** (`IncompleteRead`), và pip cũng đứt (`getaddrinfo
  failed`) khi mạng chập chờn. Cứ chạy lại; pip nối tiếp được phần đã tải.

## Mở phiên làm việc mới

Nhắn cho Claude, ví dụ:

> Đọc `NHAT_KY.md` và phần "Tiến độ" trong `README.md`, kiểm tra lại trạng thái hiện tại
> (git log, selftest), rồi làm tiếp việc số … trong "Việc còn dở".
