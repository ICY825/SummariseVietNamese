# DL-SummariseVN

Tóm tắt tin tức tiếng Việt — bài tập lớn môn Deep Learning.

So sánh có kiểm soát giữa hướng **extractive** (chọn câu có sẵn) và **abstractive**
(sinh câu mới) trên bộ tin tức VietNews/VNDS, cùng một pipeline lai kết hợp hai hướng.
Toàn bộ thiết kế chạy được trong giới hạn của Google Colab bản miễn phí.

## Câu hỏi nghiên cứu

1. Mô hình abstractive được fine-tune có thực sự vượt baseline extractive không, và vượt ở khía cạnh nào?
2. Việc cắt bài theo giới hạn token gây mất mát bao nhiêu, và pipeline lai có bù lại được không?
3. ROUGE có phản ánh đúng cảm nhận của người đọc không?

## Dữ liệu

Nguồn chính: [`nam194/vietnews`](https://huggingface.co/datasets/nam194/vietnews) —
143.816 bài, các cột `guid`, `title`, `abstract`, `article`.

Kết quả kiểm tra dữ liệu (`src/data/inspect_vietnews.py`, chạy lại ngày 09/09/2026).
Các kiểm tra toàn bộ split chạy trên cả 143.816 bài; các kiểm tra lấy mẫu dùng 4.000
bài **ngẫu nhiên với `seed=13`** — bộ này xếp theo nguồn/chuyên mục nên lấy 4.000 bài
đầu sẽ cho số liệu lệch.

| Kiểm tra | Kết quả | Hệ quả |
|---|---|---|
| Split | 99.134 / 22.184 / 22.498 | Dùng được split có sẵn |
| Trùng lặp nội bộ | 0,00% ở cả ba split | Không cần khử trùng lặp |
| Rò rỉ giữa các split | `train ∩ test` 1 bài; hai cặp còn lại 0 | Không cần chia lại split |
| Chuẩn hoá Unicode | 4,5% chưa ở dạng NFC, 0 ký tự U+FFFD | **Bắt buộc** chuẩn hoá NFC |
| Sapo chép câu đầu | Lead-1 0,7% (bao phủ tb 0,251); Lead-3 3,3% (0,491) | Bộ dữ liệu abstractive thật |
| Số câu mỗi bài | tb 16,6, p50 14, p90 29; không bài nào chỉ có 1 câu | Tầng extractive có đủ câu để chọn |
| Độ dài bài — token đã tách từ | tb 366, p90 641, p95 744 | Kích thước đầu vào PhoBERT |
| Độ dài bài — âm tiết thật | tb 495, p90 871, **p95 1.001** | Kích thước đầu vào ViT5/BARTpho |
| Tỷ lệ nén (sapo / bài) | 0,085 | Sapo dài tb 25 token / 35 âm tiết |

Con số quyết định `max_input_length` là **âm tiết**, không phải token: ViT5 và
BARTpho-syllable đọc văn bản thô đã khử gạch dưới. Tỷ lệ bài bị cắt theo từng ngưỡng
(cùng mẫu 4.000 bài):

| Ngưỡng | 512 | 768 | 1.024 | 1.536 |
|---|---|---|---|---|
| Bài bị cắt | 37,5% | 15,3% | 4,5% | 0,2% |

Đây là **cận dưới**: một âm tiết tiếng Việt thường tách thành nhiều subword của
SentencePiece, nên số token thật của ViT5 còn cao hơn và tỷ lệ cắt thực tế còn lớn
hơn các con số trên. Phải đo lại bằng chính tokenizer của mô hình ở tuần 2. Chênh
lệch giữa ngưỡng 512 và 1.024 chính là mức mất mát mà câu hỏi nghiên cứu số 2 và
pipeline lai ở tầng 4 phải xử lý.

**Lưu ý quan trọng:** văn bản trong bộ này **đã được tách từ sẵn** bằng VnCoreNLP
(`Khởi_tố`, `ma_tuý`) và dấu câu cũng đã tách rời. PhoBERT và phép tính ROUGE dùng
được trực tiếp; ViT5 và BARTpho-syllable thì **không** — phải khử dấu gạch dưới để
lấy lại văn bản thô trước. Bản thô được lưu làm bản chuẩn, bản tách từ sinh lại bằng
`underthesea` khi cần.

Hai hệ quả kỹ thuật phải nhớ khi viết bất cứ đoạn xử lý văn bản nào cho bộ này:

- **Đếm độ dài:** regex `\w+` coi `_` là ký tự chữ nên gộp `Khởi_tố` thành một đơn
  vị và đếm hụt khoảng 35% so với số âm tiết thật. Muốn đếm âm tiết phải dùng
  `[^\W_]+`. `src/data/inspect_vietnews.py` tách sẵn hai hàm `tokens()` và
  `syllables()` cho hai mục đích này.
- **Cắt câu:** dấu câu là token đứng riêng (` . `), nên phải cắt bằng
  `(?<=\s[.!?])\s+`. Cắt bằng `(?<=[.!?])\s+` sẽ đứt ngay ở viết tắt như `TP.` và
  cho ra câu cụt — mọi baseline extractive ở tầng 0–2 đều xây trên ranh giới câu này,
  nên dùng chung hàm `sentences()` thay vì viết lại.

### Đây có phải bài toán abstractive thật không?

Tỷ lệ n-gram xuất hiện trong sapo nhưng **không** có trong bài gốc (mẫu 4.000 bài):

| | 1-gram | 2-gram | 3-gram | 4-gram |
|---|---|---|---|---|
| Mới | 19,8% | **59,5%** | 77,0% | 85,4% |

Gần 60% bigram của sapo là mới, nên mô hình buộc phải viết lại chứ không thể chép câu
mà đạt điểm cao. Đây là điều kiện cần để câu hỏi nghiên cứu số 1 có nghĩa. Kèm theo,
tỷ lệ token OOV của test so với từ vựng train chỉ 1,11% — không có vấn đề thưa dữ liệu.

Kiểm tra mẫu bất thường trên 4.000 bài: **không có** bài nào dưới 50 âm tiết, không có
sapo nào dài hơn bài gốc, không có sapo rỗng. Không cần viết bộ lọc chất lượng.

## Tập con cố định

Dữ liệu đầy đủ vượt xa ngân sách của Colab bản miễn phí, nên `src/data/make_splits.py`
đóng băng sẵn các tập con vào `data/splits/` (`seed=13`). **Mọi tầng đều nạp qua
`data.splits.load_split()`, không tầng nào được tự lấy mẫu lại** — có vậy tầng 0 và
tầng 4 mới được chấm trên đúng cùng một tập bài.

| Tập | Cỡ | Vai trò |
|---|---|---|
| `train_2k` ⊂ `train_5k` ⊂ `train_10k` ⊂ `train_20k` | 2k–20k | Lồng nhau, cho đường cong học |
| `val` | 1.000 | Theo dõi qua từng epoch |
| `tune` | 500 | Dò tham số sinh văn bản (tuần 5), rời hẳn `val` và `test` |
| `test` | 2.000 | Chấm điểm cuối cùng, dùng một lần |

**Vì sao test chỉ 2.000 bài.** Đo trên tập test thật: độ lệch chuẩn ROUGE-1 giữa các
bài là 9,8, độ lệch chuẩn của *hiệu* khi so cặp đôi là 12,5. Nửa khoảng tin cậy 95%
khi so hai hệ thống:

| n | 500 | 1.000 | **2.000** | 5.000 | 22.498 |
|---|---|---|---|---|---|
| ± điểm ROUGE | 1,10 | 0,78 | **0,55** | 0,35 | 0,16 |

Khoảng cách giữa các tầng thường là 2–5 điểm nên ±0,55 dư dùng; chạy toàn bộ test chỉ
siết xuống ±0,16 mà tốn gấp 11 lần, chưa kể mỗi hệ thống neural phải sinh lại từng ấy
bản tóm tắt. Bài duy nhất rò rỉ `train ∩ test` (guid 16992) đã bị loại khỏi `test`.

**Vì sao train tối đa 20.000 bài.** Một epoch trên 99.134 bài với ViT5-base ở đầu vào
1.024 token vượt giới hạn một phiên Colab free. ViT5 đã pretrain nên fine-tune tóm tắt
bão hoà sớm; phần GPU tiết kiệm được đổ vào đường cong học và khảo sát tham số sinh sẽ
cho kết quả có nội dung hơn là thêm 1–2 điểm ROUGE. Đường cong học chính là bằng chứng
để bảo vệ lựa chọn này trong báo cáo.

## Các tầng mô hình

| Tầng | Phương pháp | Tính chất |
|---|---|---|
| 0 | Lead-1, Lead-3, Random-3, Oracle extractive | Không học |
| 1 | TextRank, LexRank (TF-IDF và embedding PhoBERT) | Không giám sát |
| 2 | PhoBERT phân loại câu, kiểu BERTSum | Có giám sát |
| 3 | Fine-tune ViT5-base; đối chứng BARTpho-syllable | Sinh văn bản |
| 4 | Extractive lọc trước, abstractive viết lại | Lai |

## Đánh giá

ROUGE-1/2/L, BERTScore, thống kê độ dài và tỷ lệ n-gram mới, đánh giá của người chấm
theo quy trình blind trên 50 bài, phân tích lỗi định tính, và bootstrap để ước lượng
khoảng tin cậy.

**Dạng văn bản để chấm điểm — quyết định đã chốt.** Đầu ra tầng 0–2 là văn bản tách
từ, đầu ra tầng 3 là văn bản thô. Chấm trực tiếp hai thứ đó với nhau là so sánh vô
nghĩa: `học_sinh` đếm một đơn vị còn `học sinh` đếm hai. Riêng việc đổi dạng đã dịch
ROUGE-2 của Lead-3 từ 10,19 lên 14,21 — lớn hơn khoảng cách kỳ vọng giữa các hệ thống.

Chuẩn chính là **dạng thô**, và mọi bản tóm tắt của mọi tầng — kể cả tham chiếu — phải
đi qua `data.text.for_scoring()` trước khi chấm. Lý do chọn dạng thô chứ không phải
dạng tách từ: khử tách từ là chiều **tất định** mà hệ thống nào cũng làm được, còn
chiều ngược lại phải nhờ `underthesea`, khác công cụ với VnCoreNLP đã tách tham chiếu,
và sai khác công cụ đó chỉ giáng lên phía abstractive. Có thể báo cáo thêm ROUGE trên
dạng tách từ để đối chiếu với các bài báo trước, kèm ghi chú về bất lợi này.

Quy trình blind cũng dùng đúng dạng đó: nếu bản tóm tắt extractive còn nguyên gạch
dưới còn bản của ViT5 thì không, người chấm nhận ra ngay đâu là hệ thống nào.

## Cấu trúc

```
data/raw/          dữ liệu tải về, không chỉnh sửa
data/processed/    đã chuẩn hoá và khử tách từ
data/splits/       file ID cố định của train/val/test
notebooks/         notebook trình bày
src/data/          text.py (3 phép biến đổi dùng chung), splits.py (nạp),
                   make_splits.py (đóng băng), inspect_vietnews.py (kiểm tra)
src/models/        các tầng mô hình
src/eval/          ROUGE tiếng Việt, BERTScore, bootstrap
app/               demo Gradio
results/           đầu ra thô, bảng chỉ số, phiếu chấm
report/            báo cáo và slide
```

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

## Chạy

```bash
.venv/Scripts/python.exe src/data/inspect_vietnews.py   # kiểm tra dữ liệu
.venv/Scripts/python.exe src/data/make_splits.py        # đóng băng tập con (chạy MỘT lần)
```

Từ tuần 3 trở đi, mọi tầng nạp dữ liệu như sau:

```python
import sys; sys.path.insert(0, "src")
from data.splits import load_split
from data.text import sentences, for_scoring

test = load_split("test")        # 2.000 bài, có sẵn article_raw / abstract_raw
cau = sentences(test[0]["article"])   # cắt câu dùng chung cho mọi tầng extractive
```

## Tiến độ

- [x] Tuần 1 — Dựng khung dự án, xác minh dữ liệu
- [x] Tuần 2 — Cố định split, ba phép biến đổi văn bản dùng chung, chốt dạng chấm điểm
- [ ] Tuần 3 — Baseline tầng 0–1 và khung đánh giá
- [ ] Tuần 4 — Fine-tune ViT5 lần đầu
- [ ] Tuần 5 — Huấn luyện đầy đủ, khảo sát tham số sinh văn bản
- [ ] Tuần 6 — Tầng 2 và tầng 4
- [ ] Tuần 7 — Người chấm, phân tích lỗi, demo Gradio
- [ ] Tuần 8 — Báo cáo, kiểm tra tái lập
