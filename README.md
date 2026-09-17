# DL-SummariseVN

Tóm tắt tin tức tiếng Việt — bài tập lớn môn Deep Learning.

So sánh có kiểm soát giữa hướng **extractive** (chọn câu có sẵn) và **abstractive**
(sinh câu mới) trên bộ tin tức VietNews/VNDS, cùng một pipeline lai kết hợp hai hướng.
Toàn bộ thiết kế chạy được trong giới hạn của Google Colab bản miễn phí.

> **Câu chuyện của dự án (từ 17/09/2026): phát hiện → giải quyết.** Tuần 1–8 là phần
> **phát hiện**: mô hình sinh (BARTpho) đứng đầu ROUGE, nhưng người chấm thấy nó thiếu ý và
> kém trung thực, và ROUGE gần như không tương quan với người đọc. Mục "Hướng mới" là phần
> **giải quyết**: đổi tiêu chí thành *đủ ý và không sai sự thật*, bảng xếp hạng đảo ngược, hai
> lần dùng BARTpho có kiểm soát đều thất bại, và hệ thống cuối là PhoBERT chọn câu có chủ đích.
> Phần mở đầu này sẽ được viết lại theo khung đó ở giai đoạn 4; kế hoạch ở mục
> "Giai đoạn 3–4 — kế hoạch".

## Câu hỏi nghiên cứu

1. Mô hình abstractive được fine-tune có thực sự vượt baseline extractive không, và vượt ở khía cạnh nào?
2. Việc cắt bài theo giới hạn token gây mất mát bao nhiêu, và pipeline lai có bù lại được không?
3. ROUGE có phản ánh đúng cảm nhận của người đọc không?

## Dữ liệu

Nguồn chính: [`nam194/vietnews`](https://huggingface.co/datasets/nam194/vietnews) —
143.816 bài, các cột `guid`, `title`, `abstract`, `article`.

### Bộ dữ liệu này từ đâu ra

Đây là bộ **VNDS**, công bố kèm bài báo *VNDS: A Vietnamese Dataset for Summarization*
(Nguyen và cộng sự, NAFOSTED NICS 2019, IEEE 9023886), phát hành tại
[`ThanhChinhBK/vietnews`](https://github.com/ThanhChinhBK/vietnews). Bài viết gồm tin
2016–2019 thuộc bốn mục (thế giới, tin tức, pháp luật, kinh doanh) của `tuoitre.vn`,
`vnexpress.net` và `nguoiduatin.vn`; sapo (`abstract`) là tóm tắt do người viết bài
đặt, và chính nó là tham chiếu vàng.

Bản `nam194/vietnews` là bản chuyển đổi của repo GitHub đó: mỗi `guid N` ứng với đúng
file `N.txt.seg` (tiêu đề / sapo / thân bài ngăn bằng dòng trống). Đã đối chiếu 5 bài
bất kỳ — trùng khớp từng ký tự, khác biệt duy nhất là dấu ngoặc kép cong `“ ”` bị đổi
thành thẳng `" "`.

**Nhưng bản HuggingFace thiếu 4,6% so với bộ gốc.** `guid` lớn nhất mỗi split trùng
khít con số trong Bảng II của bài báo, nhưng số dòng thì ít hơn — tức đánh số vẫn là
1..N của bài báo, chỉ là có lỗ hổng rải rác:

| Split | Bài báo | Bản đang dùng | `guid` lớn nhất | Thiếu |
|---|---|---|---|---|
| train | 105.418 | 99.134 | 105.418 | 6.284 |
| val | 22.642 | 22.184 | 22.642 | 458 |
| test | 22.644 | 22.498 | 22.644 | 146 |
| **Tổng** | **150.704** | **143.816** | | **6.888** |

Lấy 5 `guid` bị thiếu đầu tiên (460, 516, 730, 790, 1688) kiểm tra trên GitHub thì
**cả 5 đều có thật** — nên chúng mất ở khâu chuyển đổi, không phải bị tác giả loại.
Việc này **không** ảnh hưởng kết quả của đề tài: mọi so sánh ở đây là so sánh nội bộ
trên cùng một tập bài đã đóng băng, và đề tài chỉ dùng tối đa 20.000 bài train cùng
2.000 bài test. Điều chưa kiểm được là 6.888 bài mất đi có ngẫu nhiên hay không; nếu
không thì bản này sạch hơn bộ gốc một chút, và đó là giới hạn của mọi phát biểu dạng
"kết quả này đúng cho VNDS".

**Không so bảng ROUGE của đề tài với bảng của bài báo.** Bài báo chọn **2 câu** (đề
tài này chọn 3) và chấm bằng `ROUGE 1.5.5` qua `pyrouge` với tham số `-a -c 95 -m -n 2
-w 1.2` — trong đó `-m` bật Porter stemmer, một bộ rút gọn từ **tiếng Anh**. Phần chữ
nói dùng F-score nhưng công thức (2) in trong bài lại là công thức recall. Kết quả là
bảng của họ không tự nhất quán với thang đo ở đây: Sumbasic của họ đạt 52,65 ROUGE-1
trong khi **Oracle-3 đo được ở đây chỉ 48,14** — mà Oracle-3 là trần tuyệt đối của mọi
hệ thống chọn 3 câu, còn họ chỉ chọn 2.

Ngược lại, đề tài này **mâu thuẫn có bằng chứng** với một kết luận của bài báo. Họ viết
*"the Lead-m method does not obtain high ROUGE-scores... the content first sentences are
quite different from the abstract"* (Lead-2 = 5,86). Đo lại trên chính bộ dữ liệu ấy thì
ngược hẳn: Lead-1 = 27,03 và Lead-3 = 27,22, ngang nhau (p = 0,50), và cả ba phương pháp
đồ thị đều **thua** lead — đúng như cấu trúc tháp ngược của tin tức dự đoán.

Kết quả kiểm tra dữ liệu (`src/data/inspect_vietnews.py`, chạy lại ngày 09/09/2026).
Các kiểm tra toàn bộ split chạy trên cả 143.816 bài của bản HuggingFace; các kiểm tra
lấy mẫu dùng 4.000 bài **ngẫu nhiên với `seed=13`** — bộ này xếp theo nguồn/chuyên mục
nên lấy 4.000 bài đầu sẽ cho số liệu lệch.

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

Con số quyết định `max_input_length` phải đo bằng **chính tokenizer của mô hình**, và
trên văn bản thô đã khử gạch dưới — đó là thứ ViT5 và BARTpho-syllable thực sự đọc.
Đo bằng `src/models/measure_tokens.py` trên 2.000 bài của `train_20k` (`seed=13`); số
liệu đầy đủ nằm ở `results/tables/token_lengths_<mô hình>_raw.json`:

| Ngưỡng | 512 | 768 | 1.024 | 1.536 |
|---|---|---|---|---|
| Bài bị cắt — **token ViT5** | **49,3%** | **24,6%** | **10,2%** | 0,4% |
| Bài bị cắt — **token BARTpho** | 50,6% | 24,9% | 11,2% | 0,5% |
| Bài bị cắt — âm tiết (cận dưới) | 37,5% | 16,5% | 4,2% | 0,1% |

Ước lượng theo âm tiết đúng là **cận dưới** như đã dự đoán: hệ số nở token/âm tiết đo
được là **1,18** với ViT5 và **1,20** với BARTpho, nên ở ngưỡng 1.024 tỷ lệ cắt thật
gấp gần 2,4 lần con số theo âm tiết. Hai mô hình gần như trùng nhau, nên dùng chung
một ngưỡng — nhờ vậy so ViT5 với BARTpho là so mô hình chứ không phải so ngân sách
đầu vào.

**Quyết định: `max_input_length = 1024`, `max_target_length = 80`.**

Không chọn 1.536 dù nó hạ tỷ lệ cắt xuống 0,4%. Dài gấp 1,5 lần thì chi phí attention
gấp khoảng 2,25 lần, đủ để `train_20k` không chạy xong trong một phiên Colab free. Và
quan trọng hơn: **việc cắt bài chính là đối tượng nghiên cứu** của câu hỏi số 2, đồng
thời là lý do tồn tại của tầng 4 — nới cửa sổ cho nó biến mất là xoá luôn câu hỏi.
Mức 10,2% đủ lớn để đo được tác động, đủ nhỏ để không phá kết quả chung.

`max_target_length = 80` vì sapo có p99 là 74 token (ViT5) và 78 (BARTpho); 80 che
được 99% và là bội của 8, thuận cho fp16. Bài dài nhất 144 token là ngoại lệ.

**Lưu ý quan trọng:** văn bản trong bộ này **đã được tách từ sẵn** bằng `vitk`
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

Có **hai tầng chia** chồng lên nhau, đừng lẫn: bộ dữ liệu đã chia sẵn train/val/test từ
phía tác giả, còn đề tài này chỉ đóng băng các tập con *bên trong* ba split đó.

### Tầng 1 — bộ gốc chia sẵn 70/15/15, và chia ngẫu nhiên

Đề tài **không tự chia** train/val/test; `load_split()` nạp thẳng ba split có sẵn. Nhờ
vậy kết quả còn đặt cạnh các công trình khác dùng cùng bộ này được.

| Split | Bài báo | Tỷ lệ | Bản đang dùng | Tỷ lệ |
|---|---|---|---|---|
| `train` | 105.418 | 70,0% | 99.134 | 68,9% |
| `validation` | 22.642 | 15,0% | 22.184 | 15,4% |
| `test` | 22.644 | 15,0% | 22.498 | 15,6% |

Tỷ lệ của bản đang dùng lệch nhẹ khỏi 70/15/15 vì phần thiếu 4,6% rơi chủ yếu vào
`train` (6.284 trong tổng số 6.888 bài).

**Bài báo chỉ nói tỷ lệ, không nói chia bằng cách nào**, nên phải đo. Lấy mẫu ngẫu nhiên
3.000 bài mỗi split (`seed=13`; đếm bằng `\w+` nên từ ghép tính một đơn vị và dấu câu
không tính):

| Chỉ số | train | val | test |
|---|---|---|---|
| số từ trong bài | 367,59 | 370,68 | 373,82 |
| số từ trong sapo | 25,61 | 25,76 | 25,63 |
| số câu trong bài | 16,74 | 16,89 | 16,95 |

Ba split gần như trùng nhau trên cả ba chỉ số, và Bảng II của bài báo cũng vậy. Kiểm
thêm năm xuất hiện trong bài ở bốn đoạn khác nhau của `train`: cả bốn đều trộn lẫn
2014–2019. Kết luận: **chia ngẫu nhiên trên toàn corpus**, không cắt theo mốc thời gian
cũng không chia theo nguồn báo.

Hệ quả tốt: train và test cùng phân phối, không có dịch chuyển phân phối cần xử lý.
**Hệ quả phải nêu trong báo cáo:** vì chia ngẫu nhiên chứ không cắt theo thời gian, kết
quả ở đây *không* đo được khả năng khái quát sang tin tức của giai đoạn sau — đó là giới
hạn của bộ dữ liệu, không phải của đề tài.

*(Đối chiếu cách đếm: Bảng II của bài báo ghi 418,37 từ mỗi bài và 28,48 từ mỗi sapo.
Đếm token tách theo khoảng trắng — dấu câu tính là một token, đúng định dạng của bộ này
— cho 412,81 và 28,21, tức khớp. Khoảng lệch so với bảng trên chỉ là khác quy ước đếm,
không phải khác dữ liệu.)*

### Hai cạm bẫy của cách chia này

**`guid` đánh số riêng cho từng split, đều bắt đầu từ 1.** `guid=501` của `train` và
`guid=501` của `test` là hai bài khác hẳn nhau. Đã xác nhận: `guid` lớn nhất mỗi split
đúng bằng cỡ split đó theo bài báo (105.418 / 22.642 / 22.644), nhỏ nhất đều là 1. Vì
thế `make_splits.py` lưu kèm trường `split` trong mỗi file, và **không bao giờ được gộp
`guid` giữa hai split rồi so**.

**Trong mỗi split, bài xếp theo khối** — lấy N bài đầu của bất kỳ split nào cũng cho số
liệu lệch. Đo độ dài bài trung bình trên từng khối 1.500 bài liên tiếp của `train`:

| Vị trí | 0 | 22.000 | 44.000 | 77.000 |
|---|---|---|---|---|
| số từ trung bình | 354,8 | 408,4 | 302,3 | 470,1 |

Độ lệch chuẩn giữa các khối liên tiếp là **52,71**, trong khi giữa các mẫu ngẫu nhiên
cùng cỡ chỉ là **4,80** — gấp 11 lần. Đó là lý do `make_splits.py` **xáo một lần rồi
mới cắt tiền tố**, và mọi khâu kiểm tra dữ liệu đều lấy mẫu ngẫu nhiên với `seed=13`.

### Tầng 2 — tập con do đề tài đóng băng

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

**Vì sao các tập train lồng nhau.** Xáo pool `train` đúng một lần rồi cắt tiền tố, nên
`train_2k ⊂ train_5k ⊂ train_10k ⊂ train_20k`. Lấy bốn mẫu ngẫu nhiên độc lập thì chênh
lệch trên đường cong học sẽ lẫn cả dao động do lấy mẫu, và không đọc được gì.

**`val` và `tune` cắt từ hai đoạn khác nhau của cùng một pool đã xáo**, nên rời hẳn
nhau — đã kiểm lại: `val ∩ tune = 0`. Đây là phép kiểm giao nhau *duy nhất* có nghĩa
giữa các tập của đề tài, vì chỉ hai tập này mới cùng đến từ split `validation`; mọi phép
so `guid` giữa `train`, `val` và `test` đều vô nghĩa do ba split đánh số độc lập.

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
cho kết quả có nội dung hơn là thêm 1–2 điểm ROUGE.

**Đường cong học đo xong đã bác vế "bão hoà" của lập luận này** (xem mục Đường cong
học): từ 5k lên 20k, ROUGE-1 vẫn tăng +1,30 [+0,40, +2,19]. Giới hạn 20.000 bài vì vậy
phải được trình bày là ràng buộc ngân sách GPU, không phải là điểm mô hình hết học.

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
chiều ngược lại phải nhờ `underthesea`, khác công cụ với `vitk` đã tách tham chiếu,
và sai khác công cụ đó chỉ giáng lên phía abstractive. Có thể báo cáo thêm ROUGE trên
dạng tách từ để đối chiếu với các bài báo trước, kèm ghi chú về bất lợi này.

Quy trình blind cũng dùng đúng dạng đó: nếu bản tóm tắt extractive còn nguyên gạch
dưới còn bản của ViT5 thì không, người chấm nhận ra ngay đâu là hệ thống nào.

**Việc phải làm ở tuần 7, đừng quên.** File trong `results/predictions/` **chưa** được
chuẩn hoá: bản của tầng 0-2 còn nguyên gạch dưới (`Huỳnh_Ngọc_Bích`), bản của ViT5 và
BARTpho thì không. Điểm số không hề bị ảnh hưởng — mọi hàm chấm điểm đều tự gọi
`for_scoring()` trước — nhưng **phiếu chấm blind thì phải tự chạy `for_scoring()` lên
từng bản tóm tắt trước khi in ra**, nếu không người chấm nhận diện được hệ thống chỉ qua
dấu gạch dưới và toàn bộ khâu blind mất giá trị. Để file dự đoán ở dạng gốc là có chủ ý:
chúng là đầu ra thô của từng lần chạy, việc đổi dạng thuộc về khâu dùng chúng.

**Cài đặt.** `src/eval/` tự cài đặt ROUGE thay vì gọi thư viện ngoài, vì bộ tách token
mặc định của `rouge_score` (Google) chỉ giữ `[a-z0-9]` nên **xoá sạch dấu tiếng Việt**.
Bản cài đặt ở đây đã được đối chiếu với `rouge_score` (truyền tokenizer tiếng Việt)
trên **1.200 cặp thật** của cả sáu hệ thống baseline và **khớp tuyệt đối cả bốn chỉ
số** (lệch lớn nhất 0,0). Tái lập: cài `rouge-score` vào một thư mục riêng
(`pip install --target=... rouge_score`, đừng cài vào `.venv` của dự án), truyền
tokenizer `syllables()`, và với `rougeLsum` phải tự cắt câu bằng `sentences_raw()` rồi
nối lại bằng ký tự xuống dòng, vì `rouge_score` cắt câu theo xuống dòng. Mọi hàm
công khai chỉ nhận chuỗi thô rồi tự gọi `for_scoring()`, không nhận danh sách token
tách sẵn — làm đúng là việc duy nhất làm được.

Biến thể L chính là `rougeL` (LCS toàn chuỗi), **không phải** `rougeLsum`: `rougeLsum`
buộc phải cắt câu trên văn bản thô, mà ở đó "TP." trông y hệt dấu chấm kết câu, và sai
số ấy lệch một chiều vì chỉ đầu ra abstractive mới cần cắt câu ở dạng thô. `rougeLsum`
vẫn được tính kèm (có bộ chặn viết tắt) để đối chiếu với các bài báo khác.

**Mọi kết quả phải kèm khoảng tin cậy.** Độ lệch chuẩn ROUGE-1 giữa các bài là 9,8 —
lớn hơn khoảng cách giữa các hệ thống. `eval.report.compare()` dùng bootstrap cặp đôi:
lấy lại mẫu chỉ số bài rồi áp **cùng bộ chỉ số** cho cả hai hệ thống, vì hai hệ thống
chấm trên cùng một bài có điểm tương quan mạnh và giữ ghép cặp sẽ khử phần lớn phương
sai do độ khó của bài.

**Ghép cặp chỉ hợp lệ khi hai hệ thống chấm trên đúng cùng những bài đó**, mà cỡ mẫu
bằng nhau thì không hề bảo đảm điều ấy. Nên `evaluate()` lưu luôn danh sách `guid` vào
bảng kết quả và `compare()` đối chiếu hai danh sách trước khi bootstrap, sai thì báo
lỗi chứ không chạy tiếp. Chốt chặn này không thừa: `vit5.py` nạp mốc Lead-3 từ
`results/tables/baselines_<split>.json` do **một lần chạy khác, ở thời điểm khác** ghi
ra; nếu `data/splits/` bị sinh lại giữa hai lần chạy thì hai bên vẫn cùng 1.000 bài
nhưng khác bài, và khoảng tin cậy sinh ra sẽ sai mà không để lại dấu vết nào trong bảng.
Bảng của tuần 3b đã được bổ sung `guid` lấy từ file dự đoán của chính lần chạy đó, và
đối chiếu lại bằng cách chấm lại vài bài theo `guid` để chắc chắn chúng trỏ đúng dòng.

Chạy tự kiểm tra sau mỗi lần sửa module đánh giá:

```bash
.venv/Scripts/python.exe src/eval/selftest.py
```

## Kết quả baseline (tầng 0–1)

Chấm trên `test` (2.000 bài), 3 câu mỗi bản tóm tắt, bootstrap 10.000 lần, `seed=13`.
Số trong ngoặc là **nửa khoảng tin cậy 95%**. Tái lập bằng
`src/models/run_baselines.py`.

| Hệ thống | rouge1 | rouge2 | rougeL | rougeLsum | Độ dài | 2-gram mới |
|---|---|---|---|---|---|---|
| Random-3 | 23,44 ±0,42 | 10,65 ±0,38 | 15,76 ±0,35 | 18,95 ±0,39 | 91 | 2,0% |
| Lead-1 | 27,03 ±0,65 | 14,77 ±0,58 | 20,73 ±0,58 | 21,14 ±0,58 | 36 | 0,0% |
| Lead-3 | 27,22 ±0,44 | 14,73 ±0,43 | 19,17 ±0,41 | 22,32 ±0,43 | 102 | 0,0% |
| TextRank | 23,14 ±0,39 | 11,83 ±0,37 | 16,37 ±0,34 | 18,77 ±0,36 | 132 | 1,4% |
| LexRank | 24,81 ±0,40 | 12,52 ±0,38 | 17,26 ±0,35 | 20,08 ±0,38 | 112 | 1,6% |
| Oracle-3 | 48,14 ±0,63 | 31,89 ±0,76 | 36,21 ±0,76 | 40,19 ±0,71 | 50 | 1,1% |

**Lead-1 ngang Lead-3.** Chênh lệch ROUGE-1 chỉ −0,19 [−0,76, +0,37], p = 0,50 — không
đủ bằng chứng để nói cái nào hơn. Nguyên nhân là độ dài: sapo dài trung bình 35 âm tiết
còn Lead-1 dài 36, khớp gần như hoàn hảo; Lead-3 dài 102 âm tiết nên được recall cao
hơn nhưng mất đúng chừng ấy precision, và F1 triệt tiêu hai chiều. Hệ quả cho tuần 4:
mốc phải vượt là 27,2 chứ không phải một con số dễ hơn, và độ dài sinh ra của ViT5 phải
được kiểm soát chứ không thả nổi.

**Hai phương pháp đồ thị đều THUA lead.** TextRank −4,08 [−4,49, −3,67] và LexRank
−2,42 [−2,84, −2,00] so với Lead-3, cả hai p < 0,0001. Đây không phải lỗi cài đặt mà là
đặc trưng của thể loại: tin tức viết theo tháp ngược nên thông tin quan trọng nhất nằm
ngay câu đầu, trong khi xếp hạng theo độ trung tâm lại chuộng câu dài nhiều từ chung —
TextRank ra 132 âm tiết, gần gấp bốn lần sapo. Trung tâm của đồ thị tương đồng không
phải là "đáng tóm tắt".

**Trần của extractive là 48,1.** Oracle-3 hơn Lead-3 +20,91 [+20,29, +21,53]. Con số
này chia đôi câu chuyện của cả đề tài: khoảng 21 điểm còn nằm trong tầm với chỉ nhờ
**chọn câu khéo hơn** — đó chính là việc của tầng 2; còn phần từ 48 lên 100 thì
extractive không bao giờ với tới, vì 59,5% bigram của sapo vốn không có trong bài. Chỉ
tầng 3 mới lấy được phần đó, và đó là lý do tồn tại của nó.

**Cỡ mẫu đúng như thiết kế.** Nửa khoảng tin cậy quan sát được nằm trong khoảng ±0,39
đến ±0,65, khớp với ước lượng ±0,55 lúc chốt n = 2.000 ở phần trên.

**Kiểm tra tính nhất quán.** Tỷ lệ 1-gram mới bằng 0,0% ở cả sáu hệ thống, đúng như
định nghĩa extractive. Tỷ lệ 2-gram mới khác 0 chỉ ở những hệ thống ghép các câu **không
liền nhau** (Random-3 2,0%, LexRank 1,6%, TextRank 1,4%): bigram "mới" đó sinh ra ngay
tại chỗ nối hai câu rời. Lead-1 và Lead-3 ghép câu liền nhau nên đúng 0,0%. Không có
bản tóm tắt rỗng nào.

**Tầng 1 khử câu trùng nội dung, tầng 0 thì không.** 179 trong 2.000 bài test chứa sẵn
câu lặp lại (thường là chú thích ảnh xuất hiện hai lần). Hai bản sao của cùng một câu
có điểm trung tâm bằng hệt nhau nên bộ xếp hạng vơ cả hai, khiến 2,2% bản tóm tắt của
TextRank và 1,5% của LexRank thực chất chỉ còn hai câu nội dung. Đo tác động trước khi
sửa: bỏ lặp và chọn bù câu kế tiếp chỉ đổi ROUGE-1 thêm **+0,018 (p = 0,31)** cho
TextRank và **+0,003 (p = 0,76)** cho LexRank — tức đây **không** phải nguyên nhân
khiến hai phương pháp đồ thị thua Lead-3. Vẫn sửa, vì lý do khác: tuần 7 có khâu người
chấm blind và một bản tóm tắt lặp nguyên một câu thì người chấm nhận ra ngay. Lead-k và
Random-k cố ý giữ nguyên câu lặp — định nghĩa của chúng là "k câu đầu" và "k câu rút
ngẫu nhiên", sửa đi thì không còn là mốc ngây thơ nữa.

**LexRank bản nhúng PhoBERT: thua cả bản TF-IDF.** Thay ma trận tương đồng TF-IDF bằng
cosin giữa các vector câu PhoBERT (mean-pooling, chuẩn hoá L2), giữ nguyên `pagerank()`
và bộ chọn câu, nên khác biệt nằm đúng ở phép đo tương đồng. Chấm trên `test`:

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-3 | 27,22 ±0,44 | 14,73 ±0,43 | 19,17 ±0,41 | 102 | 0,0% |
| LexRank (TF-IDF) | 24,81 ±0,40 | 12,52 ±0,38 | 17,26 ±0,35 | 112 | 1,6% |
| **LexRank-PhoBERT** | **23,87 ±0,38** | **11,97 ±0,37** | **16,42 ±0,35** | 125 | 1,4% |

Thua bản TF-IDF **−0,93 [−1,25, −0,60] ROUGE-1, p < 0,0001** (ROUGE-2 −0,55 [−0,85,
−0,25]), và thua Lead-3 **−3,35 [−3,76, −2,95]**.

**Chạy lại trên `val`: kết luận lặp lại trên một split độc lập.** Tuần 5 chỉ đo hệ
thống này trên `test`, nên nó đứng ngoài mọi bảng dùng `val` — kể cả bảng BERTScore.
Chạy bổ sung trên `val` (1.000 bài, 34 phút CPU):

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-3 | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 0,0% |
| LexRank (TF-IDF) | 24,94 ±0,55 | 12,75 ±0,52 | 17,45 ±0,48 | 112 | 1,7% |
| **LexRank-PhoBERT** | **23,76 ±0,51** | **12,08 ±0,50** | **16,48 ±0,47** | 126 | 1,4% |

Thua bản TF-IDF **−1,19 [−1,63, −0,74], p < 0,0001** (trên `test` là −0,93), và thua
Lead-3 **−3,69 [−4,30, −3,10]**. Hai split cho cùng một kết luận với độ lớn tương
đương — **23,76 trên `val` so với 23,87 trên `test`** — nên đây không phải hiện tượng
của riêng một tập bài.

Hai cách giải thích, đều nhất quán với số liệu:

- **Cosin PhoBERT nén vào dải hẹp.** Hai câu bất kỳ trong cùng một bài thường đạt cosin
  0,7-0,95, nên đồ thị gần như đầy đủ và độ trung tâm mất sức phân biệt. Vì thế mặc
  định là `threshold=None` chứ không phải 0,1 như bài báo gốc: ngưỡng ấy ở đây giữ lại
  gần hết cạnh, biến PageRank thành phép đếm bậc.
- **Độ trung tâm ngữ nghĩa chuộng câu khái quát và dài.** Bản PhoBERT sinh ra bản tóm
  tắt dài nhất trong ba (125 âm tiết, so với sapo thật 35), tức càng xa mục tiêu.

Một cách giải thích thứ ba đã được **kiểm và loại**: PhoBERT pretrain trên văn bản tách
bằng VnCoreNLP còn bộ này tách bằng `vitk`, nên có thể ngờ rằng từ ghép bị BPE bẻ vụn.
Đo trên 2.000 bài `train`: trong 4.000 từ ghép phổ biến nhất, PhoBERT giữ nguyên 85,7%
thành một mảnh — còn cao hơn tỷ lệ của từ đơn (84,3%) — và độ vụn trung bình là 1,19
mảnh so với 1,28. Sai lệch công cụ là có thật nhưng quá nhỏ để giải thích 0,93 điểm.

**Cả ba biến thể đồ thị đều thua lead**, dù đo tương đồng bằng từ chung, bằng TF-IDF,
hay bằng embedding. Đây là đặc trưng thể loại chứ không phải lỗi cài đặt: tin tức viết
theo tháp ngược nên câu quan trọng nhất nằm ngay đầu bài, còn "trung tâm của đồ thị
tương đồng" không phải là "đáng tóm tắt".

**Chi phí: đắt hơn bản TF-IDF khoảng nghìn lần, và con số tuyệt đối phụ thuộc máy.**
Trên `test`, 1.092,8 giây cho 2.000 bài (0,55 giây/bài) so với 1,06 giây của bản
TF-IDF. Nhưng lần chạy `val` trên một máy khác mất **2.031,9 giây cho 1.000 bài, tức
2,03 giây/bài** — chậm gấp 3,7 lần với cùng code và cùng mô hình. Không truy được
nguyên nhân, vì lần chạy `test` diễn ra **trước khi `run_baselines.py` biết ghi hồ sơ**
nên không còn gì để đối chiếu; đó đúng là lỗ hổng mà hồ sơ lần chạy sinh ra để bịt.
Điều giữ nguyên qua cả hai máy là **tỷ lệ**: khoảng nghìn lần đắt hơn bản TF-IDF, để
cho kết quả kém hơn. Không cần GPU.

**Kiểm chứng chéo, hai lần.** Lần chạy trên `test` cho `Lead-3` và `LexRank` trùng
khít bảng tuần 3b (lệch 0,0 trên từng bài, `guid` trùng khớp). Lần chạy bổ sung trên
`val` cũng vậy so với bảng `baselines_val`. Đường chấm điểm không xê dịch sau mọi thay
đổi của tuần 5 — và đây là phép đối chứng gần như miễn phí, vì `Lead-3` mất 0,1 giây
cho 1.000 bài.

**Mỗi lần chạy baseline cũng để lại hồ sơ.** `run_baselines.py` ghi
`results/tables/baselines_<tag>_run.json` cạnh bảng chỉ số, cùng khuôn với `vit5.py`:
tham số dòng lệnh, split/cỡ/`k`/`seed`, `guid` đầu và cuối, phiên bản thư viện, thời
gian sinh và chấm của **từng** hệ thống, điểm tóm lược và kết quả so cặp với Lead-3.
Với Lead/Random/Oracle thì hồ sơ gần như thừa — chạy lại mất vài giây và hoàn toàn tất
định. Với `lexrank_emb` thì không: nó phụ thuộc trọng số PhoBERT trên Hub *và* phiên
bản `transformers`, nên hồ sơ ghi kèm `torch`, `transformers` và tên mô hình. Chính nhờ
hồ sơ này mà khoảng chênh 3,7 lần về thời gian ở trên mới nhìn thấy được.

Hệ thống này **không** nằm trong danh sách mặc định của `run_baselines.py` vì nó cần
`torch` và `transformers`, hai gói không có trong `.venv` của dự án; để nó ở mặc định
thì lệnh baseline trong README sẽ chết trên một máy sạch. Chạy bằng:

```bash
~/.venvs/torch/Scripts/python.exe src/models/run_baselines.py --split test \
    --systems leadk lexrank lexrank_emb
~/.venvs/torch/Scripts/python.exe src/models/run_baselines.py --split val \
    --systems leadk lexrank lexrank_emb
```

Danh sách hệ thống khác mặc định thì tên file kết quả cũng khác, nên lần chạy này không
đè bảng sáu hệ thống ở trên.

**Phép kiểm tự động.** `src/models/selftest.py` mục 8 kiểm hệ thống này **mà không cần
`torch`**: thay `_phobert_vectors` bằng một bộ sinh vector tự chế rồi kiểm đúng phần
dùng chung với bản TF-IDF — `pagerank()`, bộ chọn câu, khử câu trùng, giữ thứ tự câu.
Chỉ phép tính vector là bị thay, và đó cũng là phần duy nhất `torch` đảm nhiệm. Nếu
đợi đến khi có `torch` mới kiểm thì hệ thống này mãi mãi không có phép kiểm nào — trong
khi nó lại chính là hệ thống mà bản tóm tắt sinh lại không chắc ra đúng như cũ.

## Kết quả tầng 3 — fine-tune ViT5 lần đầu

`VietAI/vit5-base` fine-tune trên `train_5k`, 3 epoch, `lr=3e-5`, batch hiệu dụng 16,
fp16, đầu vào 1.024 / đầu ra 80, `seed=13`. Chấm trên **`val`** (1.000 bài) vì `test`
để dành cho lần chấm cuối. Colab free T4, 63 phút.

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-3 (mốc) | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 0,0% |
| **ViT5-base, train_5k** | **31,62 ±0,93** | **17,63 ±0,81** | **25,05 ±0,85** | 30 | 10,7% |
| Oracle-3 (trần extractive) | 48,09 ±0,90 | 31,60 ±1,07 | 35,90 ±1,05 | 50 | 1,0% |

**Câu hỏi nghiên cứu số 1 có câu trả lời sơ bộ: CÓ.** ViT5 hơn Lead-3 **+4,17
[+3,29, +5,04] điểm ROUGE-1, p < 0,0001** (bootstrap ghép cặp). Khoảng cách này lớn
hơn hẳn nửa khoảng tin cậy nên không phải nhiễu. Đáng chú ý là nó thắng trong khi
bản tóm tắt **ngắn hơn ba lần** Lead-3 (30 so với 102 âm tiết) — tức thắng bằng chọn
đúng chữ chứ không phải bằng rải chữ cho trúng.

**Thêm epoch thì bão hoà rất sớm.** `eval_loss` qua ba epoch: 1,802 → 1,790 → **1,796**.
Epoch 3 còn nhỉnh hơn epoch 2 một chút, tức mô hình đã hết học từ sau epoch 1 — điều
này lặp lại y hệt ở `train_10k` và `train_20k`, nên ba epoch là đủ.

Đừng suy từ đây ra rằng thêm **dữ liệu** cũng vô ích: đường cong học đo sau đó cho thấy
20k hơn 5k +1,30 ROUGE-1, có ý nghĩa. Bão hoà theo epoch và bão hoà theo cỡ dữ liệu là
hai chuyện khác nhau.

**Vẫn còn xa trần extractive.** ViT5 đạt 31,62 trong khi Oracle-3 đạt 48,09 — nghĩa là
một bộ chọn câu hoàn hảo vẫn vượt xa mô hình sinh. Đây là lý do tầng 2 (PhoBERT chọn
câu có giám sát) và tầng 4 (lai) đáng làm: khoảng 16 điểm nằm giữa hai con số ấy là
phần mà việc chọn câu tốt hơn có thể lấy được.

**Mô hình chép nhiều hơn người rất nhiều.** Tỷ lệ 2-gram mới của ViT5 là 10,7%, trong
khi sapo do người viết là **59,5%**. Nó có viết lại thật (baseline extractive chỉ
0-2%) nhưng còn cách xa mức trừu tượng của con người. Quan sát này là nguyên liệu
trực tiếp cho câu hỏi nghiên cứu số 3: ROUGE đang trao điểm cao nhất cho một hệ thống
chép nhiều hơn hẳn tham chiếu, nên điểm ROUGE và cảm nhận người đọc có thể lệch nhau.

**File kết quả của lần chạy này không còn.** Nó chạy trên Colab trước khi có hồ sơ
`run.json` và bản sao an toàn, nên điểm từng bài, bản tóm tắt sinh ra và đường loss mất
theo phiên; bảng trên chỉ còn là số chép tay. Hệ quả: chưa so cặp được với hệ thống nào
khác. Tuần 5 chạy lại đúng cấu hình này trên Kaggle làm điểm `train_5k` của đường cong
học — số mới có thể lệch nhẹ vì khác phiên bản thư viện, và số mới là số được dùng.

**Độ dài 30 âm tiết so với sapo thật 35** — hơi ngắn, có thể đang mất recall. Đây là
việc của khảo sát tham số sinh ở tuần 5 (`length_penalty`, `min_length`), làm trên
tập `tune`.

### Chạy lại trên Kaggle — điểm `train_5k` của đường cong học

Đúng cấu hình trên, chạy bằng `notebooks/kaggle_train_vit5.ipynb`: Kaggle T4 ghim còn
một GPU, `transformers` 5.0.0, 55,9 phút huấn luyện. Chấm trên `val`.

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-3 (mốc) | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 0,0% |
| ViT5, `train_5k` — Colab, tuần 4 (mất file) | 31,62 ±0,93 | 17,63 ±0,81 | 25,05 ±0,85 | 30 | 10,7% |
| **ViT5, `train_5k` — Kaggle** | **32,09 ±0,94** | **17,99 ±0,84** | **25,24 ±0,86** | 31 | 10,0% |

**Tái lập được.** Lần chạy mới hơn Lead-3 **+4,64 [+3,77, +5,49], p < 0,0001**. Nó lệch
lần Colab +0,47 ROUGE-1, nhỏ hơn nửa khoảng tin cậy, và `eval_loss` lặp lại đúng hình
dạng cũ: 1,805 → **1,789** → 1,798 (Colab: 1,802 → 1,790 → 1,796). `load_best_model_at_end`
chọn `checkpoint-626`, tức cuối epoch 2 — chốt chặn thêm sau tuần 4 đã làm đúng việc
của nó. Từ đây **số của lần chạy Kaggle là số chính**, vì chỉ nó còn đủ file.

**Kiểm chứng trước khi nhận số.** Chấm lại từ `results/predictions/` bằng `eval.report`
trên máy cho ra đúng bốn chỉ số trong bảng (lệch 0); 1.000 `guid` khớp `data/splits/val.json`
và 1.000 tham chiếu khớp dữ liệu nạp lại trên máy; phép so với Lead-3 tính lại từ
`baselines_val.json` cho cùng con số.

**Tokenizer đi đường lui.** Với `transformers` 5.0.0, `AutoTokenizer` của ViT5 báo
`KeyError` nên `load_tokenizer()` nạp thẳng `tokenizer.json`. Soát 1.000 bản tóm tắt:
không bản nào rỗng, không có token lạ (`<...>`, `extra_id`, `▁`, U+FFFD), không lặp
3-gram, không có hai bản trùng nhau. Đường lui không làm hỏng đầu ra.

### Đường cong học — 3 / 3 điểm

Cùng cấu hình, chỉ đổi cỡ tập train; mọi lần chạy trên Kaggle T4, chấm trên `val`.

| Tập train | rouge1 | rouge2 | rougeL | `eval_loss` tốt nhất | Huấn luyện |
|---|---|---|---|---|---|
| `train_5k` | 32,09 ±0,94 | 17,99 ±0,84 | 25,24 ±0,86 | 1,789 (epoch 2) | 55,9 phút |
| `train_10k` | 32,40 ±0,94 | 18,12 ±0,85 | 25,65 ±0,87 | 1,746 (epoch 2) | 112,9 phút |
| **`train_20k`** | **33,40 ±0,99** | **19,15 ±0,91** | **26,59 ±0,92** | **1,707 (epoch 2)** | 232,4 phút |

So cặp trên cùng 1.000 bài của `val`:

| Cặp | rouge1 | rouge2 | rougeL |
|---|---|---|---|
| 10k − 5k | +0,30 [−0,47, +1,08] p = 0,44 | +0,13 p = 0,69 | +0,40 p = 0,26 |
| **20k − 10k** | **+1,00 [+0,18, +1,83] p = 0,016** | **+1,03 p = 0,004** | **+0,94 p = 0,015** |
| **20k − 5k** | **+1,30 [+0,40, +2,19] p = 0,005** | **+1,16 p = 0,005** | **+1,34 p = 0,002** |

**Đường cong CHƯA phẳng ở 20.000 bài.** Bước 5k → 10k không đo được (`val` 1.000 bài
chỉ phát hiện chênh lệch từ khoảng 0,8 điểm trở lên — xem bảng cỡ mẫu ở phần Tập con
cố định), nhưng bước 10k → 20k thì có ý nghĩa ở **cả ba** chỉ số, và `eval_loss` giảm
đều suốt: 1,789 → 1,746 → **1,707**. Mỗi lần chạy vẫn đạt tốt nhất ở cuối epoch 2.

**Điều này sửa lại nhận định "bão hoà sớm" của tuần 4.** Nhận định đó rút ra từ ba
epoch trên cùng một tập `train_5k`, nên nó chỉ đúng cho việc thêm **epoch**. Thêm
**dữ liệu** vẫn còn tác dụng: +1,30 ROUGE-1 từ 5k lên 20k. Vì vậy con số 20.000 bài
phải được trình bày trong báo cáo là **ràng buộc ngân sách GPU** (4 giờ một lần chạy),
không phải là điểm mô hình hết học. Nếu còn quota, 40k và toàn bộ 99.134 bài là hướng
kiểm chứng tiếp; đường cong hiện tại chưa cho phép đoán nó phẳng ở đâu.

`train_20k` hơn Lead-3 **+5,95 [+5,03, +6,88]**, và đây là mô hình tốt nhất hiện có —
mốc để tầng 2, tầng 4 và đối chứng BARTpho so với, cũng là mô hình dùng cho khâu dò
tham số sinh trên tập `tune`.

### Hồ sơ mỗi lần chạy

Một bảng chỉ số trả lời "được bao nhiêu điểm" nhưng không trả lời "điểm đó sinh ra
bằng cách nào". Nên mỗi lần chạy `vit5.py` ghi ra **ba** file cùng tên gốc:

| File | Nội dung |
|---|---|
| `results/tables/<tag>.json` | điểm từng bài, khoảng tin cậy, `guid` |
| `results/predictions/<tag>.json` | bản tóm tắt sinh ra, kèm `guid` và tham chiếu |
| `results/tables/<tag>_run.json` | **hồ sơ lần chạy** |

`<tag>` mang theo cấu hình đã sinh ra nó, ví dụ
`vit5-base-train_5k_val_e3_lr3e-05_bs16_in1024`. Trước đây tag chỉ gồm mô hình và tập
train, nên hai lần chạy khác `lr` ghi trùng tên và lần sau đè im lặng lên lần trước —
đúng thứ sẽ xảy ra ở tuần 5 khi đường cong học chạy nhiều cấu hình trên cùng một
split. Trùng tên thì file mới được thêm hậu tố `-2`, không bao giờ đè.

`run.json` giữ: toàn bộ tham số dòng lệnh, tham số sinh, cỡ hai tập, tên GPU, phiên
bản `torch`/`transformers`, số phút chạy, checkpoint nào được `load_best_model_at_end`
chọn kèm `eval_loss` của nó, kết quả tóm lược, so sánh với Lead-3, và
`train.log_history` — loss mỗi 25 bước cùng `eval_loss` cuối mỗi epoch. Đó là nguyên
liệu để vẽ đường cong học; trước đây nó chỉ tồn tại trên màn hình Colab và mất theo
phiên. Vẽ lại:

```python
import json
h = json.load(open("results/tables/<tag>_run.json", encoding="utf-8"))["train"]["log_history"]
train = [(r["step"], r["loss"]) for r in h if "loss" in r]
val   = [(r["step"], r["eval_loss"]) for r in h if "eval_loss" in r]
```

Nhớ rằng `loss` huấn luyện đã bị nhân với `gradient_accumulation_steps` (xem chú
thích đầu `vit5.py`); `eval_loss` mới là đường đáng tin để đọc.

**Tham số sinh giờ là cờ dòng lệnh** — `--beams`, `--length-penalty`, `--min-length`,
`--no-repeat-ngram` — và được ghi vào cả `run.json` lẫn tên file. Tuần 5 khảo sát
chúng trên `tune` bằng `--no-train --model runs/.../final`, không phải train lại.

### Chạy trên Kaggle

`notebooks/kaggle_train_vit5.ipynb` là notebook huấn luyện tầng 3 trên Kaggle. Kaggle
hơn Colab free ở hai điểm quyết định với đề tài này: **30 giờ GPU mỗi tuần** thay vì
hạn mức không công bố, và **chạy ngầm** — `Save Version → Save & Run All (Commit)` chạy
notebook trên máy khác, tối đa 12 giờ, không cần giữ trình duyệt mở. `train_20k` khoảng
4 giờ nên vẫn gọn trong một phiên.

Hai điều dễ sai:

- **Kaggle cấp hai T4.** `Trainer` thấy hai thiết bị sẽ bật DataParallel và `--batch 2`
  thành mỗi GPU, tức batch hiệu dụng 32 chứ không phải 16. Mọi lệnh trong notebook mở
  đầu bằng `CUDA_VISIBLE_DEVICES=0` để giữ đúng cấu hình đã chốt — đổi batch hiệu dụng
  giữa hai lần chạy là xoá mất chính thứ đang được so sánh.
- **Chỉ `/kaggle/working` được ghi**, nên `--out` phải trỏ vào đó, và output một phiên
  không quá 20 GB (mỗi `checkpoint-*` của ViT5-base nặng khoảng 2,7 GB).

Notebook clone repo từ GitHub, nên phải `git push` trước khi chạy.

**Các bước:**

1. Kaggle → *Create → New Notebook* → *File → Import Notebook*, tải lên
   `notebooks/kaggle_train_vit5.ipynb`.
2. *Settings*: Accelerator `GPU T4 x2`, Internet `On`.
3. Sửa **ô cấu hình** (ô code đầu tiên): `TRAIN_SPLIT`, và `MODEL` nếu chạy BARTpho.
   Mỗi version chạy một cấu hình.
4. Chạy tay đến hết ô "chạy thử đường ống" (khoảng 3 phút) để chắc môi trường ổn.
5. *Save Version → Save & Run All (Commit)*, rồi đóng trình duyệt.
6. Xong thì tải `ket_qua_<mô hình>_<tập>.zip` ở tab *Output*, giải nén tại thư mục gốc
   repo — file tự vào đúng `results/tables/` và `results/predictions/`.

**Đẩy từ máy bằng Kaggle CLI** — không cần mở trình duyệt. Cài CLI vào môi trường
riêng (`python -m venv ~/.venvs/kaggle` rồi `pip install kaggle`), lưu token tạo ở
Kaggle → Settings → API vào `~/.kaggle/access_token`, rồi:

```bash
~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k --dry   # xem trước
~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k         # CHẠY NGAY, trừ quota
~/.venvs/kaggle/Scripts/kaggle.exe kernels status minh12605/dl-summarisevn-vit5
~/.venvs/kaggle/Scripts/kaggle.exe kernels logs -f minh12605/dl-summarisevn-vit5   # log trực tiếp
~/.venvs/kaggle/Scripts/kaggle.exe kernels output minh12605/dl-summarisevn-vit5 -p out --file-pattern "ket_qua_.*\.zip$"
```

`kaggle_push.py` chép notebook sang thư mục tạm và chỉ đổi `TRAIN_SPLIT`/`MODEL` ở đó;
bản trong repo giữ nguyên. Cấu hình máy (T4, Internet, riêng tư) nằm ở
`notebooks/kernel-metadata.json`. Ba điều đã gặp thật:

- Tài khoản chỉ được **2 phiên GPU chạy ngầm cùng lúc**; quá thì push bị từ chối với
  `Maximum batch GPU session count of 2 reached` — tính cả notebook của dự án khác.
- `kernels logs` **không có `-f`** chỉ trả log của phiên đã xong; phiên đang chạy trả rỗng.
- `kernels output` **bỏ qua số version**: `.../dl-summarisevn-vit5/1` vẫn tải output của
  version mới nhất. Tải file zip ngay khi mỗi lần chạy xong; checkpoint của version cũ
  chỉ còn lấy được qua giao diện web hoặc gắn version đó làm input.

Lệnh `!python ...` bị lỗi **không** làm dừng notebook, nên một bản Commit hỏng vẫn báo
thành công mà không có kết quả nào. Mọi lệnh trong notebook đều kiểm tra `_exit_code`
ngay sau đó và dừng tại chỗ nếu lỗi; ô đầu tiên cũng dừng ngay khi thiếu GPU hay
Internet thay vì huấn luyện trên CPU hàng giờ.

### Đối chứng BARTpho — mô hình đối chứng THẮNG mô hình chính

Cùng cấu hình hệt ViT5: `train_20k`, 3 epoch, `lr=3e-5`, batch hiệu dụng 16, đầu vào
1.024, `seed=13`, Kaggle T4. Chấm trên `val`.

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới | Huấn luyện |
|---|---|---|---|---|---|---|
| Lead-3 (mốc) | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 0,0% | — |
| ViT5 `train_20k` | 33,40 ±0,99 | 19,15 ±0,91 | 26,59 ±0,92 | 30 | 11,1% | 232,4 phút |
| **BARTpho `train_20k`** | **35,23 ±1,06** | **20,48 ±0,98** | **27,93 ±0,98** | **34** | 11,3% | 156,8 phút |
| Oracle-3 (trần extractive) | 48,09 ±0,90 | 31,60 ±1,07 | 35,90 ±1,05 | 50 | 1,0% | — |

**BARTpho hơn ViT5 +1,83 [+0,79, +2,86] ROUGE-1, p = 0,0002** (ROUGE-2 +1,33 [+0,41,
+2,24], p = 0,004). Hơn Lead-3 **+7,78 [+6,79, +8,77]**, so với +5,95 của ViT5. Nó còn
**nhanh hơn**: 156,8 phút so với 232,4 phút, dù tham số nhiều hơn — vì tokenizer của nó
cắt bài thành chuỗi ngắn hơn ở cùng ngưỡng 1.024 token.

**Độ dài sinh ra là 34 âm tiết, gần sapo thật (35) nhất trong mọi hệ thống của đề tài**
— ViT5 sinh 30. Đây có thể là một phần lý do nó thắng, và nó cũng khép lại khảo sát
tham số sinh ở trên: thứ ViT5 thiếu không lấy lại được bằng `length_penalty`, nhưng một
mô hình khác thì tự có.

**`eval_loss` KHÔNG so được giữa hai mô hình.** BARTpho đạt 1,424 còn ViT5 1,707, nhưng
hai mô hình có tokenizer và từ vựng khác nhau nên cross-entropy của chúng không cùng
thang đo — con số thấp hơn ở đây **không** chứng minh điều gì. Chỉ ROUGE, chấm trên
cùng 1.000 bài qua cùng một đường chấm điểm, mới so được. `eval_loss` của BARTpho qua
ba epoch: 1,441 → **1,424** → 1,424, tức vẫn bão hoà ở epoch 2 y như ViT5.

**Hệ quả cho báo cáo.** Kế hoạch ban đầu coi BARTpho là "đối chứng" cho ViT5, nhưng số
liệu nói ngược lại: ở cùng ngân sách, BARTpho-syllable là lựa chọn tốt hơn cho bài toán
này. Mọi kết luận chung của đề tài không đổi (abstractive vượt extractive, còn xa trần
Oracle-3), nhưng phần "chọn mô hình" phải được viết lại theo hướng này, và tầng 4 nên
xây trên BARTpho chứ không phải ViT5.

### Dò tham số sinh trên `tune` — không tham số nào thắng được mặc định

`notebooks/sweep/` là notebook thứ hai, **không huấn luyện gì**: nạp checkpoint
`train_20k` rồi sinh lại 500 bản tóm tắt của `tune` với sáu bộ tham số sinh
(`length_penalty` ∈ {1,0; 1,5; 2,0} × `min_length` ∈ {0; 20}), mỗi bộ một lần chấm.
Lý do dò về phía sinh dài hơn: bản tóm tắt hiện dài 30–31 âm tiết trong khi sapo thật
dài 35, tức đang hụt recall.

```bash
~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py --dir notebooks/sweep
```

Ba điều được cài để khâu này không tự lừa mình:

- **Dò trên `tune`, không trên `val`.** `val` đã bị nhìn suốt quá trình huấn luyện; chọn
  tham số trên đó là để khâu chọn học thuộc tập theo dõi. `tune` đóng băng từ tuần 2,
  rời hẳn `val` và `test`. Cấu hình thắng mới được đem sang `val`/`test`.
- **Tách notebook.** Kaggle chạy toàn bộ ô khi Save & Run All; nhét khâu này vào
  notebook huấn luyện thì mỗi lần dò lại tốn bốn giờ huấn luyện lại `train_20k`.
  Checkpoint sang notebook mới qua `kernel_sources` trong `kernel-metadata.json`:
  Kaggle gắn output **của version mới nhất** vào
  `/kaggle/input/notebooks/<chủ>/<notebook>/`, giữ nguyên cây thư mục, nên checkpoint
  nằm ở `.../runs/VietAI_vit5-base_train_20k/final`. Notebook không ghim cứng đường dẫn
  đó mà dò bằng `glob`, và dừng kèm thông báo rõ nếu không thấy — quên gắn input là lỗi
  hay gặp nhất ở đây.

  **Cơ chế "version mới nhất" này đã hết đúng kể từ lần chạy BARTpho.** Version mới nhất
  của kernel `dl-summarisevn-vit5` bây giờ là BARTpho (12/09, sau sáu lần dò tham số),
  nên `kernel_sources` gắn vào là checkpoint BARTpho, còn `CKPT_GLOB` trong ô cấu hình
  vẫn trỏ `VietAI_vit5-base_train_20k/final`. Chạy lại notebook này y nguyên sẽ dừng ở ô
  tìm checkpoint — đúng như thiết kế, nhưng nó **không** tự lấy được checkpoint ViT5
  nữa. Muốn dùng lại checkpoint ViT5 `train_20k` phải gắn **đích danh version đó** qua
  giao diện web (*Add data → Your Work → chọn version*), vì `kaggle kernels output` chỉ
  tải version mới nhất. Máy này không giữ bản sao nào: thư mục `runs/` chỉ tồn tại trên
  Kaggle. Tầng 4 của tuần 6 xây trên BARTpho nên sẽ dùng checkpoint mặc định — tiện,
  nhưng phải sửa `CKPT_GLOB` thành `vinai_bartpho-syllable_train_20k/final`.
- **Tên file phân biệt được cấu hình.** `vit5.py --no-train` bỏ `epochs`/`lr`/`batch`
  khỏi tên (chúng không được dùng) và thêm `lp`/`min`, nên sáu cấu hình cho sáu tên
  khác nhau, không lần nào đè lần nào. Dùng `--name` để bảng kết quả mang tên mô hình
  chứ không phải tên thư mục checkpoint (`final`).

**Kết quả** — 500 bài `tune`, 25 phút GPU, số liệu ở
`results/tables/vit5-base-train_20k_tune_in1024*.json`:

| Cấu hình | rouge1 | rouge2 | rougeL | Độ dài |
|---|---|---|---|---|
| **mặc định** (`lp` 1,0) | 33,60 ±1,36 | 19,58 ±1,30 | 27,01 ±1,31 | 30 |
| `lp` 1,5 | 33,65 ±1,37 | 19,44 ±1,30 | 26,80 ±1,28 | 32 |
| `lp` 1,5 + `min` 20 | 33,70 ±1,37 | 19,47 ±1,29 | 26,82 ±1,28 | 32 |
| `lp` 2,0 | 33,58 ±1,35 | 19,31 ±1,27 | 26,70 ±1,28 | 32 |
| `lp` 2,0 + `min` 20 | 33,67 ±1,34 | 19,35 ±1,26 | 26,75 ±1,28 | 32 |
| `min` 20 | 33,62 ±1,36 | 19,58 ±1,29 | 27,01 ±1,31 | 30 |

**Không cấu hình nào hơn mặc định.** So cặp từng cấu hình với mặc định trên cùng 500
bài: ROUGE-1 chênh từ −0,02 đến +0,10, mọi p ≥ 0,47. ROUGE-2 thì mọi cấu hình có
`length_penalty` đều **âm** (−0,11 đến −0,27), tuy cũng chưa đủ bằng chứng. Cấu hình
cao nhất so với thấp nhất chỉ là +0,12 [−0,12, +0,41], p = 0,40.

**`min_length = 20` không ràng buộc gì.** Đo bằng chính tokenizer ViT5: bản tóm tắt mặc
định dài trung bình 36 token, p5 là 24, và **chỉ 3 / 500 bản ngắn hơn 20 token**. Nên
`min20` trùng 495 / 500 bản với mặc định — ngưỡng đặt ra chưa chạm tới phân phối thật.

**`length_penalty` là đòn bẩy yếu ở đây.** `lp` 2,0 chỉ kéo độ dài từ 30 lên 32 âm tiết
(36 → 38 token), vẫn dưới 34,3 âm tiết của sapo thật, trong khi ROUGE-2 hơi giảm: dài
thêm mà không thêm đúng chữ thì chỉ mất precision. Giả thuyết "đang hụt recall vì sinh
ngắn" của tuần 4 vì vậy **không được số liệu ủng hộ**.

**Một cặp báo "có ý nghĩa", và vì sao không nên tin nó.** `lp2_min20` hơn `lp2` +0,09
[+0,02, +0,19], p = 0,008, ổn định qua bốn hạt giống bootstrap. Nhưng hai cấu hình chỉ
khác nhau ở **9 / 500 bản tóm tắt**, 491 bản còn lại giống hệt, nên bootstrap ghép cặp
gần như không thấy phương sai và p nhỏ đi vì lý do kỹ thuật chứ không phải vì khác biệt
lớn. Thêm nữa đây là 1 trong 15 cặp được so: ở mức 5%, kỳ vọng đã có khoảng 0,75 cặp
báo nhầm. Chênh 0,09 điểm cũng nhỏ hơn mọi khoảng cách đáng quan tâm của đề tài.

**Quyết định: giữ tham số mặc định** (`num_beams=4`, `no_repeat_ngram_size=3`,
`length_penalty=1.0`) cho `val` và `test`. Kết quả âm này vẫn là một kết quả: nó nói
rằng dư địa không nằm ở khâu sinh, mà ở việc chọn nội dung — khoảng cách tới Oracle-3
(48,1) là việc của tầng 2 và tầng 4.

### Dò lại trên BARTpho — mặc định nằm trên một vùng phẳng

Khảo sát ở trên chạy trên ViT5, nên kết luận của nó chỉ đúng cho ViT5. BARTpho là mô
hình tốt nhất nên được dò lại bằng `notebooks/sweep_bartpho/`: cùng tập `tune`, cùng
checkpoint `train_20k`, 7 cấu hình, 52,9 phút GPU Kaggle.

**Lưới đối xứng, không phải lưới một phía.** ViT5 sinh 30 âm tiết nên chỉ cần dò về
phía dài hơn; BARTpho sinh 34, sát sapo thật (35), nên câu hỏi đổi thành "mặc định đã
nằm ở điểm tốt nhất chưa" — và câu đó phải dò cả hai phía. `length_penalty` chạy từ
0,6 tới 2,0; `min_length` chỉ giữ một cấu hình để xác nhận nó không ràng buộc.

| Cấu hình | rouge1 | rouge2 | rougeL | Độ dài | Trùng mặc định |
|---|---|---|---|---|---|
| `lp` 0,6 | 35,18 ±1,48 | 20,49 ±1,39 | 28,09 ±1,41 | 32,3 | 410/500 |
| `lp` 0,8 | 35,41 ±1,48 | 20,58 ±1,38 | 28,14 ±1,40 | 33,1 | 453/500 |
| **mặc định** (`lp` 1,0) | **35,35 ±1,45** | **20,50 ±1,37** | **28,00 ±1,39** | 33,7 | — |
| `lp` 1,2 | 35,39 ±1,43 | 20,44 ±1,32 | 27,97 ±1,34 | 34,3 | 460/500 |
| `lp` 1,5 | 35,40 ±1,44 | 20,41 ±1,32 | 27,84 ±1,35 | 34,8 | 419/500 |
| `lp` 2,0 | 35,57 ±1,44 | 20,51 ±1,32 | 27,87 ±1,34 | 35,6 | 364/500 |
| `lp` 1,0 + `min` 20 | 35,34 ±1,46 | 20,49 ±1,37 | 27,98 ±1,39 | 33,8 | 498/500 |

**Không cấu hình nào khác mặc định một cách đo được.** So cặp với mặc định trên cùng
500 bài: ROUGE-1 chênh từ −0,17 đến +0,22, ROUGE-2 từ −0,09 đến +0,08, mọi p ≥ 0,27.
Cao nhất là `lp` 2,0 với +0,22 [−0,27, +0,72], p = 0,39.

**Nhưng phải đọc cho đúng: đây là vùng phẳng, không phải đỉnh.** Nếu mặc định là điểm
tối ưu thì hai phía phải cùng tụt điểm; thực tế cả hai phía đều đi ngang.
`length_penalty` vẫn điều khiển độ dài rất đều — 32,3 → 33,1 → 33,7 → 34,3 → 34,8 →
35,6 âm tiết — nhưng đổi độ dài trong khoảng ấy không đổi được điểm. Kết luận đúng là
**"không có đỉnh nào để leo"**, chứ không phải "mặc định là tốt nhất".

**`min_length = 20` lại không ràng buộc gì:** trùng mặc định 498/500 bản, đúng như dự
đoán từ ViT5 (495/500) — BARTpho còn sinh dài hơn nên ngưỡng ấy càng xa phân phối thật.

**Hai mô hình cho cùng một câu trả lời, và giờ nó là phép đo chứ không phải suy diễn.**
Giữ tham số mặc định cho cả ViT5 lẫn BARTpho ở `val` và `test`.

**Kiểm chứng.** Chấm lại cả 7 cấu hình từ file dự đoán ra đúng bảng (lệch 0); `guid`
khớp `data/splits/tune.json`; mọi `run.json` ghi `--model` là
`vinai_bartpho-syllable_train_20k/final` gắn qua `kernel_sources`. Dòng
`Checkpoint: .../sweep/sinh_lai/final` trong log Kaggle là thư mục đầu ra mà `vit5.py`
in ở cuối mỗi lần chạy, không phải mô hình được nạp.

**Kết quả phụ: BARTpho hơn ViT5 lặp lại trên `tune`.** Cùng 500 bài, cùng tham số mặc
định: ROUGE-1 **+1,75 [+0,40, +3,11], p = 0,011**, so với +1,83 trên `val`. ROUGE-2
(+0,92, p = 0,15) và ROUGE-L (+0,98, p = 0,14) cùng chiều nhưng chưa đủ bằng chứng ở
500 bài. Độ dài 33,7 so với 30,2 âm tiết.

## BERTScore — thước đo thứ hai, và nó nói gì về ROUGE

Chấm bằng `src/eval/run_bertscore.py` trên **chính các file dự đoán đã lưu**, không
chạy lại mô hình nào. Bộ mã hoá `xlm-roberta-base` (đọc thẳng âm tiết; lý do không dùng
PhoBERT nằm ở đầu `src/eval/bertscore.py`). Chín hệ thống, tập `val`, 1.000 bài.

| Hệ thống | BERTScore | ROUGE-1 |
|---|---|---|
| Oracle-3 | 89,02 | 48,09 |
| **BARTpho `train_20k`** | **87,31** | **35,23** |
| **ViT5 `train_20k`** | **87,09** | **33,40** |
| Lead-1 | 85,57 | 27,70 |
| Lead-3 | 85,55 | 27,45 |
| LexRank | 85,30 | 24,94 |
| LexRank-PhoBERT | 85,05 | 23,76 |
| TextRank | 85,02 | 23,24 |
| Random-3 | 84,80 | 23,63 |

Tám hệ thống đầu chấm trong một lần (9 phút CPU); `LexRank-PhoBERT` chấm sau, ở một
lần riêng, vì bản tóm tắt của nó trên `val` sinh sau. Không gộp được vào một file:
`run_bertscore.py` từ chối khi một tên hệ thống xuất hiện ở hai file dự đoán, đúng như
thiết kế. Nhờ vậy `Lead-3` và `LexRank` được chấm **hai lần độc lập** và cho kết quả
lệch nhau nhiều nhất **1,8 × 10⁻⁵ điểm** trên từng bài — sai số dấu phẩy động do hai
lần chia lô khác nhau, không phải khác biệt thật.

**Kết luận chính của đề tài được thước đo thứ hai xác nhận.** ViT5 hơn Lead-3
**+1,54 [+1,37, +1,72] điểm BERTScore, p < 0,0001** và BARTpho hơn Lead-3
**+1,76 [+1,57, +1,95]** — cùng chiều và cùng mức ý nghĩa với ROUGE. Lead-1 ngang
Lead-3 (+0,02, p = 0,78), đúng như ROUGE đã nói ở tầng 0.

**BARTpho hơn ViT5 theo cả hai thước đo, nhưng biên rất khác nhau.** BERTScore cho
**+0,22 [+0,03, +0,41], p = 0,025**, trong khi ROUGE-1 cho +1,83 [+0,79, +2,86],
p = 0,0002. Cùng kết luận, nhưng ở BERTScore khoảng tin cậy chỉ vừa đủ rời khỏi 0 —
đúng như dải điểm hẹp của nó báo trước. Trên từng bài, hai thước đo đồng ý ở **85,7%**
khi hỏi "bài này BARTpho hay ViT5 tốt hơn", và chỗ bất đồng chia gần đều hai phía
(131 bài: 59 nghiêng về BARTpho theo BERTScore, 72 theo ROUGE; 12 bài còn lại hoà ở cả
hai thước đo) — tức không có dấu hiệu thước đo nào thiên vị một mô hình. Tương quan từng
bài giữa hai thước đo cao nhất ở BARTpho: **+0,938**.

**Nhưng BERTScore tương phản kém hơn nhiều.** Toàn bộ khoảng cách từ Random-3 lên
Oracle-3 chỉ **4,22 điểm** BERTScore, trong khi ROUGE-1 trải **24,85 điểm** — rộng gấp
5,9 lần. Năm hệ thống extractive nằm gọn trong 0,8 điểm BERTScore. Với 50 bài đầu tiên
tôi đã tưởng nó "không phân biệt được gì"; đủ 1.000 bài thì khoảng tin cậy hẹp lại và
mọi khác biệt đều có ý nghĩa, trừ cặp Lead-1/Lead-3. Bài học: dải điểm hẹp không đồng
nghĩa với không phân biệt được, nhưng nó khiến mọi kết luận phụ thuộc nặng vào cỡ mẫu.

**Một chỗ hai thước đo đảo thứ hạng.** ROUGE-1 xếp Random-3 (23,63) **trên** TextRank
(23,24), còn BERTScore xếp ngược lại (84,80 so với 85,02). Random-3 ghép ba câu rút
ngẫu nhiên nên bản tóm tắt đứt mạch; ROUGE chỉ đếm n-gram trùng nên không thấy điều đó,
còn bộ mã hoá ngữ cảnh thì có. Đây là bằng chứng cụ thể cho câu hỏi nghiên cứu số 3:
ROUGE bỏ sót thứ mà người đọc chắc chắn nhận ra.

**Trên từng bài, hai thước đo đồng thuận nhưng không trùng khít.** Tương quan Pearson
giữa ROUGE-1 và BERTScore của cùng một hệ thống là +0,87 (Lead-3) đến +0,92 (ViT5,
Oracle-3). Khi hỏi "bài này ViT5 hay Lead-3 tốt hơn", hai thước đo **đồng ý ở 83,8%**
số bài; trong 162 bài còn lại, BERTScore nghiêng về ViT5 ở 122 bài còn ROUGE nghiêng về
ViT5 ở 40 bài — tức chỗ bất đồng cũng lệch về phía ViT5.

**LexRank-PhoBERT thua cả ở thước đo thứ hai.** 85,05 so với 85,30 của bản TF-IDF —
**−0,25 [−0,35, −0,15], p < 0,0001**, cùng chiều với ROUGE (−1,19). Điều này đáng nói
vì BERTScore chấm bằng một bộ mã hoá ngữ cảnh, tức là *sân nhà* của cách tiếp cận
nhúng: nếu việc thay TF-IDF bằng vector PhoBERT có bắt được thứ gì mà ROUGE bỏ sót thì
đây là chỗ nó phải lộ ra. Nó không lộ ra. Kết luận "độ trung tâm ngữ nghĩa không phải
là đáng tóm tắt" vì thế không phải là hiện tượng của riêng phép đếm n-gram.

Cũng lưu ý một chỗ hai thước đo bất đồng về biên độ: ROUGE-1 xếp LexRank-PhoBERT trên
TextRank khá rõ (23,76 so với 23,24) trong khi BERTScore coi hai hệ thống gần như ngang
nhau (85,05 so với 85,02).

**Giới hạn phải nêu trong báo cáo.** BERTScore cũng chỉ là một phép xấp xỉ bằng mô hình,
không phải người đọc; nó được dùng ở đây để *đối chiếu* với ROUGE chứ không thay thế.
Khâu chấm blind ở tuần 7 mới là phép đo có người thật, và chính nó sẽ nói hai thước đo
tự động này bỏ sót cái gì.

```bash
~/.venvs/torch/Scripts/python.exe src/eval/run_bertscore.py \
    baselines_val vit5-base-train_20k_val_e3_lr3e-05_bs16_in1024 \
    bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024
```

Truyền nhiều file thì các hệ thống được gộp lại, nhưng `guid` của chúng phải trùng khớp
mới cho ghép — cùng chốt chặn mà `report.same_articles()` dựng cho ROUGE, vì ViT5 và
Lead-3 nằm ở hai file dự đoán do hai lần chạy khác nhau ghi ra.

## Câu hỏi 2 — cắt bài ở 1.024 token mất bao nhiêu

`src/eval/truncation.py` trên ViT5 `train_5k`, tập `val`. Không cần GPU và không chạy
lại mô hình: đếm token bằng đúng tokenizer của ViT5, rồi chia điểm từng bài đã lưu
thành hai nhóm bị cắt / không bị cắt.

Bị cắt **93 / 1.000 bài (9,3%)**, khớp mức 10,2% đo trên mẫu `train_20k` lúc chốt ngưỡng.
Bài bị cắt dài trung bình 1.173 token; phần bị cắt chiếm trung bình 11,9% số token.

| Nhóm | n | ViT5 | Lead-3 | ViT5 − Lead-3 |
|---|---|---|---|---|
| Không bị cắt | 907 | 32,76 ±0,98 | 27,99 ±0,62 | +4,77 ±0,91 |
| Bị cắt | 93 | 25,60 ±2,57 | 22,20 ±1,70 | +3,40 ±2,76 |

**Bài bị cắt khó với mọi hệ thống, không riêng ViT5.** ViT5 sụt 7,16 điểm ở nhóm bị
cắt, nhưng Lead-3 — không bao giờ đọc tới phần bị cắt — cũng sụt 5,79. Phần lớn cú sụt
đến từ việc bài dài, không phải từ việc cắt. So thẳng ViT5 giữa hai nhóm sẽ đổ oan toàn
bộ 7 điểm cho việc cắt.

**Phần thực sự do cắt: −1,37 [−4,28, +1,58], p = 0,36 — chưa đủ bằng chứng.** Đó là
hiệu của hiệu: khoảng cách ViT5 − Lead-3 ở nhóm bị cắt trừ ở nhóm còn lại. Khoảng tin
cậy rộng vì chỉ có 93 bài bị cắt, nên phép đo này chỉ bắt được mất mát cỡ 3 điểm trở
lên. Kết luận đúng là **"chưa phân biệt được với 0"**, không phải "không mất gì".

**Trần mất mát nhỏ.** Trong số âm tiết của sapo có mặt trong bài, trung bình chỉ
**2,4%** nằm riêng ở phần bị cắt (p50 0%, p90 6,9%), và 63% số bài bị cắt không mất gì.
Tin tức viết theo tháp ngược nên phần đuôi ít khi mang thông tin của sapo. Nhưng đây
chỉ là trần mất mát tính theo **chữ của sapo**; số đo trên `train_20k` ngay dưới cho
thấy thiệt hại thật lớn hơn nhiều lần con số ấy.

**Kiểm chứng.** Vị trí cắt script tính ra trùng từng token với cách tokenizer tự cắt
(`truncation=True, max_length=1024`, đúng như `vit5.py`) trên cả 93 bài.

**Mô hình càng mạnh, cái giá của việc cắt càng lộ ra — và ở `train_20k` thì đo được.**

| Mô hình | Bài bị cắt | Không bị cắt | Bị cắt | Hiệu của hiệu |
|---|---|---|---|---|
| ViT5 `train_5k` | 93 (9,3%) | 32,76 ±0,98 | 25,60 ±2,57 | −1,37 [−4,28, +1,58], p = 0,36 |
| ViT5 `train_10k` | 93 (9,3%) | 33,16 ±1,01 | 24,94 ±2,41 | −2,43 [−5,27, +0,44], p = 0,10 |
| **ViT5 `train_20k`** | 93 (9,3%) | **34,31 ±1,04** | **24,50 ±2,59** | **−4,02 [−6,84, −1,12], p = 0,007** |
| **BARTpho `train_20k`** | **102 (10,2%)** | **36,19 ±1,09** | **26,80 ±2,95** | **−4,04 [−7,33, −0,79], p = 0,017** |

**Toàn bộ phần lợi của việc thêm dữ liệu rơi vào nhóm bài không bị cắt**: 32,76 → 33,16
→ 34,31 ở nhóm vừa cửa sổ, trong khi nhóm bị cắt đứng yên (25,60 → 24,94 → 24,50). Mô
hình càng giỏi thì phần nó không được đọc càng thành nút thắt — đúng chiều phải thấy
nếu việc cắt có giá thật. **Ba phép đo trên ViT5 dùng chung đúng 93 bài bị cắt ấy**, nên
chúng không độc lập với nhau; thứ đáng tin ở đây là **cả hướng lẫn độ lớn đều tăng đơn
điệu** theo sức mạnh của mô hình, và đến `train_20k` thì khoảng tin cậy đã rời khỏi 0.
Hàng BARTpho đứng ngoài dãy đó: tokenizer của nó cắt ở chỗ khác nên nhóm bị cắt là 102
bài, không phải 93 — không so trực tiếp với ba hàng trên được, nhưng nó độc lập xác nhận
cùng một kết luận trên một mô hình khác.

**Trả lời câu hỏi 2, trên mô hình tốt nhất hiện có — BARTpho `train_20k`:** cắt bài ở
1.024 token lấy đi khoảng **4 điểm ROUGE-1 [0,8; 7,3] ở 10,2% số bài** — tính ra toàn
tập là khoảng **0,41 điểm**. Nhỏ so với 7,78 điểm mà BARTpho hơn Lead-3, nhưng không
còn là nhiễu, và nó sẽ lớn dần nếu mô hình còn mạnh lên. Trên ViT5 `train_20k` con số
tương ứng là −4,02 [−6,84, −1,12] ở 9,3% số bài, tức khoảng 0,37 điểm toàn tập — hai mô
hình cho cùng một độ lớn dù tokenizer khác nhau.

**Một nghịch lý cần giải thích trong báo cáo:** chỉ khoảng 2,5% chữ của sapo nằm riêng ở
phần bị cắt (2,4% với ViT5, 2,5% với BARTpho), mà thiệt hại đo được lại tới 4 điểm. Vậy
thứ mất đi chủ yếu **không** phải chữ của sapo nằm ở đuôi bài, mà nhiều khả năng là ngữ
cảnh giúp mô hình chọn ý và diễn đạt.
Đây là giả thuyết; tầng 4 đã kiểm nó bằng cách lọc câu lúc suy luận — xem mục Tầng 4.

Chạy lại cho mô hình khác bằng `--system <tag>`. Thêm dữ liệu train không làm hẹp
khoảng tin cậy — vẫn là 93 bài bị cắt ấy (102 với BARTpho); muốn hẹp hơn phải chấm trên
nhiều bài hơn, ví dụ gộp `tune` vào hoặc chấm trên `test` ở lần chấm cuối.

## Tầng 2 — PhoBERT chọn câu có giám sát

Tầng 0–1 chọn câu mà không học gì; Oracle-3 cho thấy trần của việc chọn 3 câu là 48,09
trong khi Lead-3 chỉ 27,45. Tầng 2 đo xem một bộ chọn **học được** lấy lại bao nhiêu
trong khoảng cách ấy. Cài đặt ở `src/models/phobert_sent.py`, chạy bằng
`notebooks/tang2/`: PhoBERT mã hoá cả bài một lần, mỗi câu lấy trung bình token của nó
rồi qua một lớp tuyến tính để cho điểm, nhãn là bộ câu của `oracle_indices()`, và bản
tóm tắt là 3 câu điểm cao nhất ghép theo thứ tự bài. Huấn luyện trên `train_20k`,
3 epoch, `lr=2e-5`, batch 8, `pos_weight` 6,35 (nhãn dương 13,6%), 52,9 phút trên
Kaggle T4; không có tập theo dõi trong lúc huấn luyện, dùng mô hình cuối epoch 3. Chấm
trên `val`.

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-3 | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 0,0% |
| **PhoBERT chọn câu** | **28,69 ±0,61** | **15,49 ±0,58** | **20,25 ±0,54** | 100 | 1,3% |
| Oracle-3 trong cửa sổ 256 token | 43,94 ±0,89 | — | — | — | — |
| Oracle-3 cả bài | 48,09 ±0,90 | 31,60 ±1,07 | 35,90 ±1,05 | 50 | 1,0% |

**Hệ thống extractive tốt nhất của đề tài lúc báo cáo, nhưng biên nhỏ** — nay đã bị bản
chọn 2 câu (`k2`, 31,00) vượt, xem mục chọn số câu linh hoạt. Hơn Lead-3
**+1,24 [+0,78, +1,71], p < 0,0001** (ROUGE-2 +0,81 [+0,38, +1,23]), hơn LexRank +3,75,
hơn LexRank-PhoBERT +4,94 và hơn TextRank +5,46 — hệ thống extractive đầu tiên của đề
tài vượt được lead (không kể Oracle-3, vốn đọc sapo) — nhưng theo ROUGE thì **không** vượt
được mốc Lead-2 thêm sau (−0,39, p = 0,19), dù BERTScore nói có (+0,23); xem mục chọn số
câu linh hoạt. Nó chỉ hơn Lead-1
**+0,99 [+0,12, +1,87], p = 0,026**, và vẫn thua cả hai mô hình sinh: ViT5 −4,70,
BARTpho −6,54.

**Lấy lại rất ít khoảng cách tới trần.** Trong 20,64 điểm từ Lead-3 lên Oracle-3, tầng 2
lấy lại 1,24 điểm, tức **6,0%**; tính theo trần mà nó thật sự với tới được (43,94, xem
dưới) thì **7,5%**. Phần lớn khoảng cách mà "chọn câu khéo hơn" hứa hẹn vẫn chưa được bộ
chọn này chạm tới.

**Cái giá của cửa sổ 256 token: 4,14 điểm trần.** Đo trên đủ 1.000 bài `val` bằng chính
tokenizer PhoBERT, một cửa sổ nhìn thấy trung bình **62,8%** số câu — trùng khít con số
mà mô hình tự ghi vào `run.json` lúc chạy trên Kaggle — và Oracle-3 bị giới hạn trong cửa
sổ ấy chỉ đạt **43,94**, so với 48,09 khi được đọc cả bài. Ước lượng trước đó trên 300
bài là 44,4 và 60,6%.

**Mô hình học một thiên lệch câu đầu mạnh hơn cả nhãn.** Tầng 2 giữ câu đầu tiên ở
**71,8%** số bài, trong khi Oracle-3 giới hạn trong cùng cửa sổ chỉ chọn câu đầu ở
39,6%. Phân bố vị trí chung thì gần nhau — 57,3% số câu được chọn nằm ở vị trí 0–2, so
với 53,3% của oracle trong cửa sổ; vị trí trung bình 2,6 so với 2,8 — tức phần lớn độ
"dồn về đầu bài" là do cửa sổ, còn phần thật sự do học là việc gần như luôn giữ câu đầu.
12,6% bản tóm tắt trùng khít Lead-3.

**Một bất lợi cấu trúc: luôn lấy đúng 3 câu.** 995/1.000 bản tóm tắt của tầng 2 có đủ 3
câu, dài trung bình 100 âm tiết. Oracle-3 dừng sớm khi thêm câu không tăng điểm nên chỉ
128/1.000 bản có đủ 3 câu, dài 50 âm tiết — gần sapo thật (35) hơn nhiều. Với F1, chữ
thừa làm mất precision; đây cũng là lý do Lead-1 (36 âm tiết) ngang Lead-3 ở tầng 0.
Chọn số câu linh hoạt thay vì cố định k = 3 là hướng cải thiện rẻ nhất — đã làm, xem
mục "Chọn số câu linh hoạt" dưới đây: lấy 2 câu thay vì 3 được +2,30.

**Kiểm chứng.** Chấm lại từ file dự đoán ra đúng bảng (lệch 0); `guid` khớp
`data/splits/val.json`; không bản tóm tắt nào rỗng. Loss huấn luyện (trung bình mỗi 50
bước) giảm từ 1,20 xuống khoảng 0,8 và phẳng dần ở epoch cuối.

**Thước đo thứ hai đồng ý, và ở đây biên còn rõ hơn ROUGE.** BERTScore trên cùng 1.000
bài (`xlm-roberta-base`, chấm từ file dự đoán đã lưu): tầng 2 đạt **85,94**, hơn Lead-3
**+0,40 [+0,29, +0,50]**, hơn Lead-1 **+0,38 [+0,22, +0,54]**, hơn LexRank +0,64
[+0,52, +0,76], mọi p < 0,0001; thua BARTpho −1,36 [−1,55, −1,18]. Với ROUGE-1, biên so
với Lead-1 chỉ vừa đủ (p = 0,026); với BERTScore khoảng tin cậy rời hẳn khỏi 0. Tầng 2
đứng đầu mọi hệ thống extractive ở cả hai thước đo, trừ Oracle-3. Số liệu ghép cặp:
`results/tables/bertscore_tang2_tang4_val_sosanh.json`.

**Một lỗi đã sửa trước khi dùng lại checkpoint.** Checkpoint tầng 2 gồm hai phần —
PhoBERT lưu bằng `save_pretrained`, lớp cho điểm lưu riêng ở `head.pt` — nhưng
`_build_model()` chỉ nạp phần đầu. Lần chạy trên Kaggle **không** dính lỗi này vì nó
huấn luyện rồi sinh trong cùng một tiến trình; nhưng mọi lần `--no-train` về sau sẽ cho
điểm bằng một lớp khởi tạo ngẫu nhiên mà không báo gì. Giờ `head.pt` được nạp, và
`--no-train` từ chối chạy khi thiếu nó. Đã kiểm trên một mô hình RoBERTa tí hon: trọng số
lớp cho điểm nạp đúng, hai lần nạp cho cùng điểm, thiếu `head.pt` thì lớp khác đi.
Checkpoint thật vẫn còn ở output của kernel `dl-summarisevn-tang2` (`tang2/phobert_sent/final`).

### Chọn số câu linh hoạt — lấy 2 câu, +2,30

`src/models/phobert_select.py` tách làm hai bước. `score` chạy PhoBERT một lần cho mỗi
split và ghi điểm từng câu ra
`results/predictions/phobert-sent-train_20k_<split>_len256_scores.json`. Bước này chạy
trên kernel `dl-summarisevn-tang2-score` và gắn checkpoint của `dl-summarisevn-tang2`, vì
tải 540 MB về máy đứt nhiều lần; mỗi split mất 7–14 giây GPU. `select` dò 10 quy tắc
trên `tune` (`k1`–`k3`, và `pT` = mọi câu có xác suất ≥ T, tối đa 3, ít nhất 1) rồi chỉ
đem **một** quy tắc thắng sang `val`. `selftest.py` mục 11 kiểm `k3` trùng
`pick_indices()` và hành vi của ngưỡng.

**Checkpoint nạp đúng.** `k3` dựng lại **1.000/1.000** bản tóm tắt của lần chạy Kaggle
trên `val`, từng chữ. Tức là `head.pt` được nạp và điểm câu trùng lần huấn luyện.

**Trên `tune` (500 bài), `k2` thắng**: 31,11, so với `p0.7` 30,77, `p0.6` 30,60, `k1`
29,49 và `k3` 28,62. So với `k3` là +2,49 [+1,93, +3,05]. Họ ngưỡng `pT` đi đúng hướng
(bớt câu thì điểm tăng) nhưng không vượt được một con số cố định.

| Trên `val` | rouge1 | rouge2 | rougeL | Độ dài | Số câu |
|---|---|---|---|---|---|
| Lead-1 | 27,70 ±0,92 | 14,94 ±0,83 | 21,02 ±0,81 | 35 | 1 |
| Lead-3 | 27,45 ±0,60 | 14,68 ±0,58 | 19,20 ±0,55 | 102 | 3 |
| Tầng 2, `k3` (đã báo cáo) | 28,69 ±0,61 | 15,49 ±0,58 | 20,25 ±0,54 | 100 | 3 |
| Lead-2 (mốc thêm sau) | 29,09 ±0,72 | 15,35 ±0,67 | 20,92 ±0,63 | 70 | 2 |
| **Tầng 2, `k2`** | **31,00 ±0,75** | **16,66 ±0,72** | **22,39 ±0,67** | 67 | 2 |

`k2` hơn `k3` **+2,30 [+1,91, +2,70]**, hơn Lead-3 +3,54 [+2,95, +4,16] và hơn Lead-1
+3,29 [+2,42, +4,19]; mọi p < 0,0001, ROUGE-2 và ROUGE-L cùng chiều. Vẫn thua ViT5 (33,40)
và BARTpho (35,23), nhưng khoảng cách tới BARTpho đã hẹp từ 6,54 còn 4,23.

**Mốc Lead-2 — và nó sửa lại một câu ở trên.** `k2` dài 67 âm tiết, nằm giữa Lead-1 và
Lead-3, nên phải hỏi mức tăng đến từ chọn câu hay chỉ từ độ dài. Lead-2 trả lời: nó đạt
29,09, **hơn cả Lead-3 (+1,63) lẫn Lead-1 (+1,38)**, và `k2` vẫn hơn nó **+1,91 [+1,26,
+2,57]** ở gần cùng độ dài (67 so với 70). Tức khoảng 1,9 trong 3,5 điểm hơn Lead-3 là
nhờ chọn câu, phần còn lại nhờ độ dài. Nhưng cũng chính Lead-2 cho thấy, **theo ROUGE**,
tầng 2 `k3` không hơn được Lead-2 (−0,39 [−0,98, +0,21], p = 0,19; BERTScore nói ngược lại,
xem dưới). Câu "hệ thống extractive đầu tiên vượt lead" ở trên, xét bằng ROUGE, chỉ đúng
với Lead-1 và Lead-3; với `k2` thì đúng với cả ba. Lead-2 được thêm
**sau khi đã thấy** `val`, không dò trên `tune`, nên nó là mốc tham chiếu chứ không phải hệ
thống dự thi. 189/1.000 bản của `k2` trùng khít Lead-2, 581 bản giữ câu đầu tiên.

**BERTScore xác nhận `k2`, nhưng không đồng ý về `k3` với Lead-2.** `k2` đạt **86,26**,
hơn Lead-2 (85,71) +0,55 [+0,42, +0,68], hơn `k3` (85,94) +0,31 [+0,24, +0,38], hơn Lead-3
+0,71 và Lead-1 +0,69; thua BARTpho −1,05 [−1,24, −0,86]; mọi p < 0,0001. Riêng `k3` so với
Lead-2 thì hai thước đo ngược nhau: ROUGE-1 −0,39 [−0,98, +0,21] (không đủ bằng chứng) còn
BERTScore **+0,23 [+0,12, +0,35]**, p < 0,0001. Nên chỉ nói được: theo ROUGE, `k3` không vượt
Lead-2; theo BERTScore thì có. `k2` vượt Lead-2 ở cả hai. So ghép cặp theo `guid` giữa
`bertscore_baselines_val_k2_leadk+phobert-sent-train_20k_val_len256_k2.json` và hai bảng
BERTScore cũ.

Hiện vật: `results/tables/phobert-sent-train_20k_tune_len256_rules.json` (lưới trên
`tune`), `phobert-sent-train_20k_val_len256_k2.json` và `_run.json` (ghi luôn phép kiểm
1000/1000), `results/tables/baselines_val_k2_leadk.json` (Lead-2). Chạy lại:

```bash
.venv/Scripts/python.exe src/models/phobert_select.py select
.venv/Scripts/python.exe src/models/run_baselines.py --split val --k 2 --systems leadk   # Lead-2
```

## Tầng 4 — lọc câu trước, abstractive viết lại

Câu hỏi 2 để lại một giả thuyết: cắt bài lấy đi khoảng 4 điểm ROUGE-1 ở 10,2% số bài,
nhưng chỉ khoảng 2,5% chữ của sapo nằm riêng ở phần bị cắt — nên thứ mất đi có thể là
ngữ cảnh chứ không phải chữ. Tầng 4 kiểm giả thuyết ấy theo cách rẻ nhất: với bài vượt
ngân sách, thay vì đưa 1.024 token **đầu bài**, đưa 1.022 token **chọn lọc từ toàn
bài** cho chính checkpoint BARTpho `train_20k`, không huấn luyện lại. Cài đặt ở
`src/models/hybrid.py` và cờ `vit5.py --filter`; chạy bằng `notebooks/tang4/`, 30 phút
GPU Kaggle, tập `val`.

**Hai chiến lược.** `lexrank` xếp hạng mọi câu theo độ trung tâm LexRank rồi lấy dần
cho tới khi đầy ngân sách; `lead_lexrank` bảo đảm 3 câu đầu trước rồi mới để LexRank lấp
phần còn lại — vì tin tức viết theo tháp ngược, một bộ lọc thuần LexRank có thể vứt mất
đúng câu đang có giá trị nhất. Cả hai giữ nguyên thứ tự câu và chỉ đụng tới đúng
**102/1.000 bài** vượt ngân sách, khớp con số 10,2% của câu hỏi 2. Bài sau lọc dài
trung bình 1.013 token, giữ khoảng 25,8 câu và bỏ khoảng 5,3 câu.

| Nhóm 102 bài bị lọc | rouge1 | rouge2 | rougeL |
|---|---|---|---|
| BARTpho, cắt thô (mốc) | 26,80 ±2,95 | 13,59 ±2,80 | 20,24 ±2,59 |
| + lọc `lexrank` | 25,98 ±2,69 | 12,92 ±2,58 | 20,08 ±2,52 |
| + lọc `lead_lexrank` | 26,67 ±2,81 | 13,53 ±2,68 | 20,50 ±2,55 |
| Lead-3 | 22,66 ±1,56 | — | — |

**Lọc lúc suy luận không lấy lại được thiệt hại do cắt.** So cặp trong nhóm bị lọc:
`lexrank` −0,83 [−3,00, +1,39], p = 0,47; `lead_lexrank` −0,13 [−2,21, +2,12], p = 0,91.
Khoảng tin cậy rộng vì chỉ có 102 bài, nhưng cận trên của chiến lược tốt hơn là +2,12:
nếu thiệt hại thật đúng cỡ 4 điểm như ước lượng ở câu hỏi 2 (−4,04 [−7,33, −0,79]) thì
lọc lúc suy luận lấy lại được nhiều nhất khoảng một nửa. Toàn tập `val`: 35,13 và 35,20
so với 35,23 của bản không lọc — không khác biệt.

**Giữ câu đầu tốt hơn, nhưng chưa đủ bằng chứng.** `lead_lexrank` hơn `lexrank`
+0,70 [−0,73, +2,13], p = 0,33 — cùng chiều với cấu trúc tháp ngược, nhưng 102 bài
chưa đủ để kết luận.

**Chưa được kết luận giả thuyết "mất ngữ cảnh" là sai.** Mô hình được huấn luyện trên
bài cắt thô, nay nhận văn bản đã lọc — tức lệch phân phối. Hai cách đọc đều nhất quán
với số liệu: hoặc ngữ cảnh ở đuôi bài không phải thứ bị mất, hoặc mô hình chưa từng học
cách dùng văn bản đã lọc. Tách được hai cách đọc này cần vòng 2: huấn luyện lại BARTpho
trên đầu vào đã lọc (~2,6 giờ GPU). **Đã chạy** — xem "Vòng 2" dưới đây: cách đọc thứ hai
cũng không được ủng hộ.

**Đối chứng nội tại, và một giả thuyết đã bị bác.** Trên 898 bài không bị lọc, 783 bản
tóm tắt trùng khít bản gốc và ROUGE-1 chênh −0,02 [−0,22, +0,18], p = 0,85 — nhóm này
không đổi điểm, đúng như phải thấy. 115 bản khác nhau cũng không phải nhiễu: hai lần chạy
lọc trùng nhau **898/898** trên nhóm này và khác bản gốc ở đúng cùng 115 bài. Giả thuyết
đầu tiên là lô bị đệm khác đi khi 102 bài đổi độ dài, nhưng tỷ lệ bản khác nhau ở lô có
bài bị lọc (12,3%) và lô không có (14,4%) gần như bằng nhau, nên giả thuyết ấy bị bác.
Nguyên nhân còn lại là **đường chạy**: bản gốc sinh ngay sau huấn luyện, bản lọc nạp
`final` từ đĩa với `--no-train`. Nó không đổi điểm, nhưng một đối chứng sạch tuyệt đối
cần thêm một lần chạy `--no-train` **không lọc** trên `val` (~15 phút GPU). **Đã chạy** và
xác nhận đúng nguyên nhân này — xem đoạn "Đối chứng sạch" ngay dưới.

**BERTScore cũng không thấy gì.** Trong 102 bài bị lọc, so với cắt thô: `lexrank`
+0,08 [−0,31, +0,49], `lead_lexrank` +0,07 [−0,31, +0,47]; hai chiến lược với nhau −0,01.
Toàn tập, cả ba bản đều 87,31. Chiều ở đây **ngược** với ROUGE-1 (−0,83 và −0,13) nhưng cả
hai thước đo đều nằm gọn trong nhiễu, nên kết luận không đổi: lọc lúc suy luận không
lấy lại được gì đo được. Trên 898 bài không bị lọc, 783 bài trùng điểm và chênh lệch là
−0,01 [−0,05, +0,03]. Nhóm 102 bài được dựng lại bằng tokenizer BARTpho theo đúng tiêu
chí của `apply_filter()` (quá 1.022 token) và kiểm bằng ROUGE-1 của nhóm — ra đúng 26,80 /
25,98 / 26,67 như bảng trên. Chạy chung ba file bằng cú pháp đổi tên mới của
`run_bertscore.py` (`TAG:TÊN`), vì cả ba cùng mang tên hệ thống `bartpho-syllable-train_20k`.

**Đối chứng sạch: `--no-train` không lọc.** `notebooks/tang4/` nạp đúng checkpoint `final`
theo đúng đường `--no-train` như hai bản lọc, chỉ bỏ bộ lọc. Nó đạt 35,18 trên toàn tập
và **trùng khít cả hai bản lọc 898/898** ở nhóm không bị lọc, trong khi chỉ trùng bản
gốc 874/1.000 (783 ở nhóm ấy). Vậy 115 bản khác nhau nói ở trên đúng là do đường chạy
(sinh ngay sau huấn luyện, hay nạp từ đĩa), không phải do bộ lọc. So với đối chứng này
trong 102 bài bị lọc: `lexrank` −0,49 [−2,67, +1,70], `lead_lexrank` +0,21 [−1,90,
+2,49]. Kết luận của vòng 1 không đổi. Đối chứng trừ bản gốc trên toàn tập: −0,05
[−0,24, +0,14], p = 0,58.

### Vòng 2 — huấn luyện lại trên đầu vào đã lọc: cũng không lấy lại được

`notebooks/tang4_train/` (kernel riêng `dl-summarisevn-tang4-train`) huấn luyện lại
BARTpho `train_20k` với `--filter lead_lexrank` áp lên **cả** train lẫn `val`. Mọi tham số
dòng lệnh trùng bản gốc, trừ `--filter` (đối chiếu từ hai `run.json`). Bộ lọc đụng
2.071/20.000 bài train (10,36%) và đúng 102/1.000 bài `val` như vòng 1. 142,9 phút trên
T4.

| Nhóm | Bản gốc (cắt thô) | Vòng 1 `lead_lexrank` | Vòng 2 `lead_lexrank` |
|---|---|---|---|
| 102 bài bị lọc | 26,80 | 26,67 | 27,36 |
| 898 bài không bị lọc | 36,19 | 36,17 | 36,92 |
| Toàn `val` (ROUGE-1) | 35,23 ±1,06 | 35,20 | 35,95 ±1,02 |
| ROUGE-2 / ROUGE-L toàn tập | 20,48 / 27,93 | — | 20,78 / 28,58 |

**Có tăng, nhưng tăng ở cả những bài không hề bị lọc.** So ghép cặp với bản gốc, ROUGE-1:
toàn tập +0,72 [−0,08, +1,51], p = 0,076; nhóm bị lọc **+0,56** [−1,43, +2,56]; nhóm
không bị lọc **+0,73** [−0,13, +1,58]. 898 bài kia nhận **đúng cùng đầu vào** ở cả hai mô
hình, nên chênh lệch ở đó chỉ đo độ dao động giữa hai lần huấn luyện. Lợi ích riêng của bộ
lọc là phần nhóm bị lọc vượt lên trên mức ấy: **−0,17 [−2,35, +2,00]**, p = 0,89
(`group_diff`, lấy mẫu độc lập hai nhóm). Hiệu hai hiệu theo ROUGE-2 là −0,00 [−1,89,
+1,90], theo ROUGE-L −0,26 [−2,31, +1,78]. Lấy đối chứng `--no-train` làm mốc thay cho
bản gốc thì ROUGE-1 là +0,14 [−2,09, +2,40]. Không thước đo nào thấy gì.

**BERTScore đồng ý rằng không có gì.** Toàn tập: bản gốc 87,31, đối chứng 87,29 (−0,02
[−0,05, +0,02]), vòng 2 **87,44** (+0,14 [−0,01, +0,29], p = 0,07). Tách theo nhóm: bị lọc
+0,35 [−0,03, +0,73], không lọc +0,11 [−0,05, +0,27], hiệu hai hiệu **+0,23 [−0,17,
+0,66]**, p = 0,28. Chiều ngược với ROUGE-1 (−0,17) nhưng cả hai đều nằm gọn trong nhiễu —
đúng như ở vòng 1.

**Nên cách đọc thứ hai của vòng 1 cũng không được ủng hộ.** Vòng 1 để ngỏ khả năng mô
hình chỉ "chưa học cách dùng văn bản đã lọc". Nay mô hình đã được học đúng điều đó, và
nhóm bị lọc vẫn không nhích hơn phần còn lại. Nhưng đây là **không thấy**, không phải
**chứng minh không có**. Cận trên +2,00 vẫn chứa khoảng một nửa thiệt hại 4,04 do cắt đo
ở câu hỏi 2, vì 102 bài quá ít để thấy hiệu ứng dưới ~2 điểm.

**Ba điều phải nói kèm khi báo cáo.**

- **Khác checkpoint được chọn.** `load_best_model_at_end` theo `eval_loss`: bản gốc lấy
  epoch 2 (1,4237, epoch 3 là 1,4239), vòng 2 lấy epoch 3 (1,4285). Hai mô hình vì thế
  khác nhau cả số epoch hiệu dụng. Đây là ứng viên cho mức +0,7 ở nhóm không lọc, bên
  cạnh dao động ngẫu nhiên. Phép hiệu hai hiệu chỉ khử được ảnh hưởng này **nếu** thêm
  một epoch tác động lên hai nhóm như nhau — giả định hợp lý nhưng chưa kiểm; muốn kiểm
  phải sinh lại từ `checkpoint-2500` của vòng 2 (~15 phút GPU, checkpoint vẫn nằm trong
  output kernel). `eval_loss` hai bên cũng không so trực tiếp được, vì một bên đo trên
  `val` đã lọc.
- **`lead_lexrank` được chọn trên `val`** (vòng 1), nên vòng 2 trên `val` không hoàn toàn
  độc lập với phép chọn ấy. Kết quả là âm tính nên sai lệch này không làm phồng kết luận.
- **Một lần huấn luyện, một seed.** Vòng 2 sinh bản tóm tắt dài hơn một chút (34,7 so với
  34,0 âm tiết) và nhiều 2-gram mới hơn (12,8% so với 11,3%); chỉ 158/1.000 bản trùng
  khít bản gốc.

Tái lập mọi số trong mục này: `~/.venvs/torch/Scripts/python.exe src/eval/tang4_sosanh.py`
(cần `sentencepiece` cho tokenizer BARTpho), ghi `results/tables/tang4_vong2_val_sosanh.json`
kèm danh sách `guid` của 102 bài bị lọc. Script tự kiểm lại 26,80 / 25,98 / 26,67 trước.

## Tuần 7 — chấm blind có người thật (câu hỏi 3)

Phiếu đã dựng nhưng **không có người chấm đủ 50 bài**: điểm hiện có là **hai lượt chấm bằng
mô hình ngôn ngữ** theo đúng phiếu này — xem mục "Kết quả" cuối phần — và chúng đã được kiểm
chứng bằng một mẫu 12 bài do 2 người chấm thật, ở mục cuối phần này. Các đoạn thiết kế dưới
đây vẫn áp dụng nguyên vẹn nếu sau này có người chấm đủ 50 bài.

Phương án "3 người chấm 50 bài" **đã bỏ**, nên ba phiếu trống `cham_nguoi1..3.csv` không còn
trong cây làm việc; chúng vẫn nằm trong commit `2638a78` nếu cần dựng lại.

```bash
.venv/Scripts/python.exe src/eval/human_eval.py prepare    # đã chạy; từ chối chạy lại khi khoa.json đã có
.venv/Scripts/python.exe src/eval/human_eval.py analyze    # sau khi thu đủ cham_nguoi1..3.csv
```

**Thiết kế.** 50 bài rút ngẫu nhiên từ `val` (`seed=13`), 3 người chấm, thang 1–5 cho ba
tiêu chí: **đầy đủ** (nắm được ý chính), **trung thực** (mọi thông tin có trong bài gốc)
và **trôi chảy**. Chấm trên `val` vì chỉ `val` có bản tóm tắt của mọi tầng. Cái giá là
`val` đã được dùng để chọn `lead_lexrank` và mốc Lead-2 — nhưng không hệ thống nào trong
phiếu được *chọn* bằng `val` (`k2` chọn trên `tune`).

**Người chấm chỉ dựa vào phiếu.** Họ không xem ROUGE, tên hệ thống hay sapo dưới danh
nghĩa "đáp án" — mọi đánh giá so với **bài gốc**; biết điểm tự động thì phép so ở câu hỏi
3 thành vòng lặp. Đầu `phieu_doc.html` có tám quy tắc chung (trong đó: không cộng điểm chỉ
vì bản dài; thông tin bài gốc không nhắc tới vẫn bị trừ điểm trung thực) và **bảng mô tả
từng mức 1–5 cho cả ba tiêu chí**, để "3 điểm" có cùng nghĩa với mọi người chấm. Bảng và
quy tắc được lưu nguyên văn vào `khoa.json` (`RUBRIC`, `RULES` trong `human_eval.py`).

**Bốn hệ thống, mỗi hệ thống trả lời một câu hỏi.**

| Trong phiếu | Vì sao |
|---|---|
| Sapo tham chiếu | trần của thang điểm người chấm; ẩn danh như mọi hệ thống khác |
| Lead-3 | mốc extractive ngây thơ mà mọi tầng phải vượt |
| Tầng 2 `k2` | extractive tốt nhất (31,00) |
| BARTpho `train_20k` | abstractive tốt nhất (35,23) — cặp `k2`/BARTpho trả lời câu hỏi 1 |

Bỏ ViT5 vì cùng vai trò với BARTpho mà thêm 25% tải chấm (`--systems` thêm được); bỏ
tầng 4 vì nó không khác BARTpho một cách đo được, người chấm sẽ tốn công mà không phân
biệt được gì. Mỗi người chấm 199 bản và đọc 50 bài gốc (524 âm tiết mỗi bài).

**Chốt chặn cho tính blind**, đều đã kiểm trên phiếu thật:

- Mọi văn bản qua `for_scoring()`: 0/199 bản còn gạch dưới, 0 bản còn dấu cách trước dấu
  câu, không tên hệ thống nào xuất hiện trong phiếu.
- Thứ tự nhãn xáo riêng từng bài; không hệ thống nào ở cùng một nhãn quá 19/50 bài.
- Bản trùng chữ chỉ hiện một nhãn. Ở bài B06, BARTpho sinh **trùng khít sapo** nên hai hệ
  thống chung nhãn C và nhận chung một điểm — hiện hai bản giống hệt nhau sẽ để lộ việc
  có hai hệ thống trùng nhau.
- Đối chiếu từng nhãn trong phiếu với file dự đoán gốc qua khoá: 0 lệch.

**Không chốt được: độ dài.** Trung bình Lead-3 dài 104,8 âm tiết, `k2` 64,6, BARTpho 34,5
và sapo 35,0. Người chấm tinh ý đoán được bản dài nhất là Lead-3. Phải nêu trong báo cáo.

**Khoá không vào git.** `results/human_eval/khoa.json` nằm trong `.gitignore`, vì người
chấm đọc được repo thì quy trình blind hỏng. Mất khoá cũng không sao: `prepare` tất định
— chạy lại vào một thư mục khác ra `phieu_doc.html` và ba file CSV trùng từng byte, khoá
trùng mọi trường trừ `created_at`. `analyze` từ chối chạy nếu phiếu đọc đã bị sửa (so
bằng SHA-256 lưu trong khoá).

**BARTpho có chép sapo không — không phải rò rỉ.** Bản trùng khít ở B06 dẫn tới một lượt
kiểm. Trên cả 1.000 bài `val`, BARTpho trùng khít sapo 4 lần, ViT5 2 lần. Không bài
`val` nào có sapo hay 300 ký tự đầu bài trùng một bài `train_20k`. Hai trong bốn lần là
hợp lệ, vì sapo nằm nguyên văn trong bài (B06: sapo chính là câu mở đầu bài). Guid 6748
có một tin **cùng sự kiện** trong `train_20k` (Jaccard âm tiết 0,40); guid 17129 chưa giải
thích được. Cả hai không nằm trong 50 bài của phiếu. Lượt kiểm này cũng nhắc lại một cạm
bẫy đã ghi: 181 `guid` của `val` "trùng" `train_20k`, nhưng cả 181 là bài khác hẳn — `guid`
đánh số riêng theo split.

**`analyze` tính gì.** Điểm từng hệ thống (trung bình các người chấm, rồi bootstrap theo
bài); so cặp đôi theo bài; đồng thuận giữa người chấm bằng Krippendorff's alpha (dữ liệu
khoảng); và cho câu hỏi 3: Spearman giữa điểm người chấm và ROUGE-1/2/L, BERTScore trên
từng bản, Kendall tau-b trong từng bài (hai thước đo có xếp các hệ thống của cùng một bài
giống nhau không), và xếp hạng cấp hệ thống. Sapo không có ROUGE nên không vào phần này.
`selftest.py` mục 7 kiểm alpha bằng giá trị tính tay (0,85), Spearman, Kendall và việc gộp
nhãn. Chạy thử trên phiếu **giả** (điểm là hàm của ROUGE-1 cộng nhiễu, CSV kiểu Excel bản
Việt dùng dấu chấm phẩy): đọc được, alpha ≈ 0,8, tương quan dương như cài vào; phiếu còn ô
trống thì bị chặn, trừ khi chạy với `--bo-qua-loi`. Số của lần chạy thử **không** phải kết
quả và không được lưu vào `results/`.

### Kết quả — hai lượt chấm bằng mô hình ngôn ngữ, **không phải người chấm**

Không có người chấm, nên phiếu được chấm **hai lượt** bằng mô hình ngôn ngữ (Claude), theo
đúng tám quy tắc và bảng mức điểm, mỗi điểm 1–2 kèm lý do:

| Lượt | Ai chấm | Được đọc gì | File |
|---|---|---|---|
| 1 | phiên làm việc đã dựng phiếu | bài gốc và các bản gắn nhãn xuất từ `phieu_doc.html` | `results/human_eval/llm_judge.csv` |
| 2 | một tác tử con **độc lập** | **chỉ** 5 file xuất đó cùng bảng tiêu chí; không đọc repo, khoá hay điểm lượt 1 | `results/human_eval/llm_judge_2.csv` |

Khoá chỉ được ghép vào sau khi cả hai lượt chấm xong:
`human_eval.py analyze --phieu "llm_judge*.csv" --ten llm_judge_2luot` →
`results/tables/llm_judge_2luot_val.json` (từng lượt riêng: `--phieu llm_judge.csv`, tức
`llm_judge_val.json`). **Không được trình bày là đánh giá của người.**

Trước khi có lượt 2, lượt 1 đã được rà lại và sửa **bốn** điểm trung thực áp thang không
nhất quán với chính quy tắc "không nhắc tới → 3, sai chi tiết quan trọng → 2, phần lớn bịa
→ 1" (B04A 2→3, B31A 2→1, B28D 2→3, B50D 3→2). Sửa cả bốn không đổi kết luận nào (BARTpho
trừ Lead-3 về trung thực: −0,64 → −0,62; Spearman với ROUGE-1: +0,145 → +0,150).

**Hai lượt đồng thuận cao, và xếp hạng hệ thống trùng nhau ở cả ba tiêu chí.**

| Tiêu chí | Trùng khít | Lệch ≤ 1 điểm | Krippendorff's alpha | TB lượt 1 / lượt 2 |
|---|---|---|---|---|
| Đầy đủ | 68% | 100% | 0,82 | 3,31 / 3,05 |
| Trung thực | 90% | 99% | 0,92 | 4,59 / 4,56 |
| Trôi chảy | 61% | 99% | 0,76 | 3,88 / 4,23 |

Lượt 2 khắt khe hơn về đầy đủ và dễ hơn về trôi chảy, nhưng lệch đều ở mọi hệ thống nên
thứ hạng không đổi. Chỉ **3/597** điểm lệch từ 2 trở lên, và đối chiếu lại bài gốc thì cả
ba là **lượt 1 chấm quá tay**:

- **B19B** (sapo), trung thực 2 / 4: chính bài gốc lẫn người phát ngôn — đoạn mở bằng Đại sứ
  rồi gắn câu trích cho ông Shamsulddin — nên "gán nhầm phát ngôn" là quá nặng.
- **B31A** (sapo), trung thực 1 / 3: tiếng nhạc, bài "Chị tôi" và sinh tố đều có gợi ý trong
  bài, chỉ không nói thẳng; điểm 1 (chính là một trong bốn chỗ đã sửa ở trên) quá nặng, 3
  mới đúng thang.
- **B39A** (Lead-3), trôi chảy 3 / 5: bản dài nhưng đọc trôi — lượt 1 đã trừ vì độ dài.

Điểm gốc của cả hai lượt được giữ nguyên: sửa một lượt sau khi đã thấy lượt kia là xoá mất
tính độc lập mà alpha đo.

**Điểm trung bình hai lượt:**

| Hệ thống | Đầy đủ | Trung thực | Trôi chảy |
|---|---|---|---|
| Sapo tham chiếu | 3,41 ±0,26 | 4,03 ±0,28 | **4,62** ±0,17 |
| Lead-3 | **3,48** ±0,23 | **5,00** ±0,00 | 3,83 ±0,17 |
| Tầng 2 `k2` | 3,11 ±0,21 | 4,96 ±0,06 | 3,37 ±0,21 |
| BARTpho | 2,72 ±0,23 | 4,32 ±0,31 | 4,41 ±0,22 |

**Mỗi hướng mạnh ở một tiêu chí — đúng thứ ROUGE không tách ra được.**

- **BARTpho viết trôi chảy nhất trong các hệ thống máy**: hơn Lead-3 +0,58 [+0,29, +0,86]
  và `k2` +1,04 [+0,77, +1,31], ngang sapo (−0,21 [−0,51, +0,09]).
- **Nhưng thiếu ý và kém trung thực hơn extractive.** Đầy đủ: kém Lead-3 −0,76 [−1,05,
  −0,47], kém `k2` −0,39 [−0,68, −0,09]. Trung thực: kém Lead-3 −0,68 [−0,99, −0,38], kém
  `k2` −0,64 [−0,94, −0,35]. BARTpho bị 1–2 điểm đầy đủ ở 21/50 (lượt 1) và 24/50 (lượt 2)
  bản — bản ngắn (34 âm tiết) bỏ mất đúng ý chính: thương vong, mức phạt, việc ai làm gì.
  Bị 1–2 điểm trung thực ở 7/50 và 9/50 bản, và cả hai lượt ghi cùng những lỗi cụ thể: sai
  địa điểm ("quận 12" thay cho huyện Hóc Môn), đảo chủ thể (tàu định vị thành tàu được định
  vị), nhầm người (chị Liễu thay cho chồng chị), gán phát ngôn cho nhầm người, nhầm đơn vị
  (cả trung tâm thay cho một chi nhánh), sai giá (30 triệu thay cho 450.000 đồng), gán hành
  vi cho nhầm nhóm người.
- **Extractive gần như luôn trung thực** (Lead-3 5,00, `k2` 4,96) vì chép nguyên câu; lỗi
  của nó là mạch văn: câu mất tiền đề ("Theo cách này", "Do đó, công ty này", "Tương tự như
  vậy"), lẫn chú thích ảnh và tiêu đề phụ, ngoặc kép vỡ. **Mức nặng nhẹ thì tuỳ người
  chấm**: lượt 1 cho 14/50 bản của `k2` điểm trôi chảy 1–2, lượt 2 chỉ 1/50 — thứ hạng giữ
  nguyên, còn con số "bao nhiêu bản hỏng" thì không nên trích. Lead-3 hơn `k2` cả ở đầy đủ
  (+0,37 [+0,13, +0,62]) lẫn trôi chảy (+0,46 [+0,20, +0,73]).
- **Sapo cũng bị trừ điểm trung thực** (4,03) vì chứa thông tin bài gốc không có — cùng
  hiện tượng mục "Đây có phải bài toán abstractive thật không?" đã đo bằng n-gram. Một phần
  lỗi "bịa" của BARTpho có thể là nó học đúng thói quen ấy của sapo.

**Câu hỏi 3 — ROUGE gần như không nói gì về điểm này.** Trên 150 bản của ba hệ thống máy,
Spearman giữa điểm trung bình ba tiêu chí (hai lượt) và ROUGE-1 chỉ **+0,12** (ROUGE-2
+0,17, ROUGE-L +0,18, BERTScore +0,16). Trong từng bài — ba bản của cùng một bài có được xếp
cùng thứ tự không — Kendall tau trung bình **−0,04** với ROUGE-1 và −0,15 với BERTScore,
tức không hơn ngẫu nhiên. Ở cấp hệ thống, ROUGE xếp BARTpho > `k2` > Lead-3, còn điểm trung
bình ba tiêu chí xếp Lead-3 > BARTpho > `k2` (Spearman −0,50). Lượt 1 một mình cho cùng bức
tranh (ROUGE-1 +0,15). Cách đọc nhất quán với các lỗi đã ghi: ROUGE thưởng việc trùng chữ
với sapo nhưng không phạt thông tin sai, không phạt thiếu ý chính, và không phạt câu mất
tiền đề.

**Giới hạn — phải nêu khi dùng bất kỳ con số nào ở trên.**

- **Không phải người đọc.** Câu hỏi 3 ở đây thành "ROUGE so với đánh giá bằng mô hình ngôn
  ngữ theo bảng tiêu chí", không phải "so với cảm nhận người đọc".
- **Hai lượt cùng một họ mô hình.** Alpha 0,76–0,92 cho thấy kết quả không phải may rủi của
  một lượt chấm, nhưng hai lượt có thể chung thiên lệch — nó là **cận trên** của độ tin cậy,
  không tương đương đồng thuận giữa hai người thật.
- **Lượt 1 không hoàn toàn blind**: đã biết bài B06 là bản trùng sapo và bản dài nhất thường
  là Lead-3. Lượt 2 không có thông tin này, và hai lượt vẫn xếp hạng như nhau.
- **Tiêu chí trung thực có lợi sẵn cho extractive**: chép nguyên câu thì gần như không thể
  sai. Đọc từng cột thay vì gộp ba tiêu chí thành một điểm.

### Phân loại lỗi — bốn họ, phân bố rất khác nhau theo tầng

Mục trên liệt kê lỗi lẻ; mục này phân loại chúng có hệ thống trên **190 ghi chú**: 115 của
hai lượt máy chấm (50 bài) và 75 của hai người chấm (mẫu 12 bài), ghép với hệ thống thật
qua `khoa.json`. Mọi ví dụ đều kèm mã bài để mở phiếu kiểm lại.

**Hồ sơ lỗi theo tầng.** Tỷ lệ bản bị chấm ≤3 điểm, trong ngoặc là ≤2. Cột "máy" gộp hai
lượt (n = 100 mỗi hệ thống), cột "người" gộp hai người trên mẫu 12 bài (n = 24):

| Hệ thống | đầy đủ (máy) | đầy đủ (người) | trung thực (máy) | trung thực (người) | trôi chảy (máy) | trôi chảy (người) |
|---|---|---|---|---|---|---|
| sapo | 49% (17%) | 29% (25%) | 32% (6%) | 21% (8%) | 6% (1%) | 17% (8%) |
| Lead-3 | 51% (9%) | 46% (0%) | **0% (0%)** | **0% (0%)** | 34% (2%) | 17% (0%) |
| `k2` | 66% (25%) | 50% (17%) | 2% (0%) | 4% (0%) | 58% (15%) | 17% (4%) |
| BARTpho | **79% (45%)** | **88% (54%)** | 23% (16%) | 21% (17%) | 15% (3%) | 12% (4%) |

**Họ 1 — thiếu ý chính.** Phổ biến nhất ở mọi tầng và là lỗi nặng nhất của BARTpho: 45%
số bản bị 1–2 điểm đầy đủ theo máy, 54% theo người. Nguyên nhân cơ học là độ dài — BARTpho
sinh 34 âm tiết nên thường chỉ giữ được một mệnh đề, bỏ mất thương vong (B48), mức phạt
(B03), quy mô (B34). Extractive thiếu ý theo cách khác: chọn đúng câu nhưng là câu bối cảnh
chứ không phải câu kết luận (B30, B21).

**Họ 2 — sai sự thật.** Chia ba loại nhỏ, và **chỉ abstractive cùng sapo mắc phải**;
Lead-3 đúng 0% ở cả máy lẫn người.

| Loại | Ví dụ | Hệ thống |
|---|---|---|
| Gán nhầm chủ thể hoặc phát ngôn | B45 (câu của ông Nguyễn Xuân Anh gán cho ông Dũng), B27 (chồng chị Liễu thành chị Liễu), B15 (đảo tàu định vị và tàu được định vị) | BARTpho |
| Sai con số, địa điểm, đơn vị | B36 (30 triệu thay cho 450.000 đồng), B05 (quận 12 thay cho huyện Hóc Môn), B42 (cả trung tâm thay cho một chi nhánh), B08 (5h30 trái bài gốc) | BARTpho |
| Bịa chi tiết không có trong bài | B31 (tivi, sinh tố mít), B11 (Instagram, Twitter), B37 (6 không gian, 8.500 euro) | **cả ba đều là sapo** |

Dòng cuối đáng chú ý: ba ví dụ "bịa" rõ nhất đều thuộc **sapo** — bản do nhà báo viết, tức
tham chiếu vàng. Đây là bằng chứng trực tiếp cho giả thuyết nêu ở mục trên, rằng thói quen
thêm thông tin ngoài bài là học được từ dữ liệu huấn luyện chứ không phải tật riêng của mô
hình. Lỗi sai sự thật của BARTpho thì khác về chất: nó **hoán đổi** chi tiết có sẵn trong
bài (ai làm gì, bao nhiêu, ở đâu), chứ không thêm chi tiết mới.

**Họ 3 — mạch văn đứt.** Đặc trưng của extractive, vì câu bị bứng khỏi ngữ cảnh:

- **Mất tiền đề** — câu mở đầu bằng từ nối hoặc đại từ không có gì đứng trước: "Theo cách
  này" (B14), "Tương tự như vậy" (B16), "Do đó, công ty này" (B29), "Kiều" chưa được giới
  thiệu (B32), "họ" không rõ (B21), "vị trí trên" (B18). Tất cả đều là `k2`.
- **Ghép hai câu rời** thành một bản tóm tắt không liền mạch (B12, B32).

BARTpho phần lớn miễn nhiễm họ này, nhưng **không tuyệt đối**: B37 mở đầu bằng "Đó là thông
tin" mà không có tiền đề, B16 để lại "không ngoại lệ" thiếu vế trước — mô hình sinh vẫn học
được cách mở câu của văn bản gốc.

**Họ 4 — rác kế thừa từ dữ liệu, không phải lỗi của tầng nào.** Chú thích ảnh lọt vào bản
tóm tắt (B47 ở cả Lead-3 lẫn `k2`, B05, B15, B27), tiêu đề phụ dạng câu hỏi (B11, B32),
ngoặc kép vỡ (B21, B32, B12), và rõ nhất là **B34: chính sapo bị mất chữ đầu** — "àn
Novaland" thay vì "Tập đoàn Novaland". Vì lỗi này xuất hiện ngay trong tham chiếu, nó thuộc
về khâu làm sạch dữ liệu. Extractive hứng nhiều nhất chỉ vì nó chép nguyên văn.

**Hai hồ sơ lỗi bù trừ nhau, và đó là kết quả chính của phân tích này.** Extractive gần như
không thể sai sự thật (0–4%) nhưng hỏng mạch văn và bỏ sót ý; abstractive đọc trôi chảy
nhưng thiếu ý nặng nhất và là tầng duy nhất hoán đổi chi tiết. Không tầng nào thắng ở cả ba
tiêu chí, nên **gộp ba tiêu chí thành một điểm duy nhất sẽ xoá mất chính sự khác biệt này**
— đọc theo cột, đừng đọc theo tổng.

**Người chấm xác nhận độc lập bốn lỗi cụ thể.** Hai người chấm mẫu, không biết nhãn nào là
hệ thống nào và không trao đổi với nhau, cùng chỉ ra đúng một chỗ: B37 bịa "8.500 euro"
(sapo), B42 nhầm chi nhánh Bỉm Sơn thành cả trung tâm Thanh Hoá (BARTpho), B43 gán việc xây
dựng cho nhóm người Trung Quốc (BARTpho), B34 mất chữ "àn Novaland" (sapo). Trùng khớp ở
mức từng lỗi như vậy là bằng chứng mạnh hơn nhiều so với trùng khớp ở điểm trung bình.

Ba họ đầu đều là thứ **ROUGE không phạt**, đúng như mục trên đã đo: trùng chữ với sapo
không đòi hỏi đúng chủ thể, đủ ý, hay liền mạch.

### Kiểm chứng máy chấm bằng một mẫu người chấm — đã chấm xong

Hai lượt máy chấm đồng thuận với nhau, nhưng điều đó không chứng minh chúng đồng thuận với
**người**. Để giữ được chữ "người đọc" trong câu hỏi 3 mà không cần 3 người × 50 bài, 2
người chấm một mẫu nhỏ, và điểm của họ được so với điểm máy trên **đúng những bản đó**.

```bash
.venv/Scripts/python.exe src/eval/human_eval.py prepare-mau    # đã chạy; từ chối chạy lại khi mau.json đã có
.venv/Scripts/python.exe src/eval/human_eval.py so-sanh        # sau khi thu cham_mau_nguoi1..2.csv
```

**Mẫu.** 12 bài rút ngẫu nhiên (hạt giống `13-mau`) trong chính 50 bài của phiếu máy đã
chấm: B01, B04, B09, B12, B17, B33, B34, B37, B38, B42, B43, B47 — 48 bản, mỗi hệ thống
đúng 12, không có nhãn gộp. Mỗi người chấm khoảng 45–60 phút. Người chấm nhận
`phieu_doc_mau.html` và **một** file `cham_mau_nguoi<i>.csv`; cùng tám quy tắc và bảng mức
điểm như phiếu máy đã dùng. `mau.json` ghi danh sách bài và mã băm phiếu.

**Mẫu là một phần của đúng phiếu máy đã chấm, không phải phiếu gần giống.** Trước khi
rút, `prepare-mau` dựng lại 50 bài và đòi trùng khít `phieu_doc.html` đã phát lẫn
`khoa.json`. Đã kiểm trên file sinh ra: mã bài và nhãn A–D giữ nguyên nên điểm người ghép
thẳng với điểm máy theo (bài, nhãn); từng bài trong phiếu mẫu trùng khít phần tương ứng của
phiếu lớn (chỉ khác ký tự xuống dòng cuối trang); phần hướng dẫn và bảng mức điểm giống hệt;
không lộ tên hệ thống. Việc tách `build_bai()` ra để dùng chung không đổi gì: `prepare` chạy
lại vào thư mục khác vẫn ra phiếu 50 bài và ba CSV trùng từng byte.

**`so-sanh` đo gì.** Trên 48 bản của mẫu, theo từng tiêu chí: alpha người–người, alpha
người–máy (trung bình người so với trung bình máy), Spearman người–máy, máy chấm cao hay
thấp hơn người bao nhiêu; thứ hạng bốn hệ thống theo người và theo máy; **chiều** của các
kết luận chính (BARTpho − Lead-3, BARTpho − `k2`, Lead-3 − `k2`) khi người chấm và khi máy
chấm; và Spearman với ROUGE-1/BERTScore tính riêng cho người và cho máy. Ghi
`results/tables/nguoi_vs_may_val.json`. Chạy thử trên phiếu **giả** (điểm máy cộng nhiễu,
CSV kiểu Excel bản Việt): chạy hết, phiếu còn ô trống bị chặn trừ khi `--bo-qua-loi`. Số
của lần chạy thử **không** phải kết quả và không được lưu vào `results/`.

**Đọc kết quả thế nào — quyết định trước khi thấy số.** Nếu alpha người–máy xấp xỉ alpha
người–người và các kết luận chính cùng chiều, được viết "điểm máy trên 50 bài đã được kiểm
chứng trên một mẫu người chấm 12 bài". Nếu người–máy thấp hơn rõ ở một tiêu chí, kết luận
về tiêu chí đó chỉ được nêu theo người chấm trên mẫu, và độ lệch ấy tự nó là một phát hiện
về máy chấm. 12 bài là ít: khoảng tin cậy trên mẫu rộng, nên mẫu dùng để kiểm **chiều** và
**mức đồng thuận**, không để thay số của 50 bài.

**Kết quả.** 2 người chấm đủ 48/48 bản của 12 bài, độc lập với nhau và với máy chấm. Số
đầy đủ ở `results/tables/nguoi_vs_may_val.json`.

| Tiêu chí | alpha người–người | alpha máy–máy | alpha người–máy | Spearman người–máy | máy − người | lệch ≤1 |
|---|---|---|---|---|---|---|
| `day_du` | 0,747 | 0,773 | **0,789** | +0,822 | −0,21 | 96% |
| `trung_thuc` | 0,568 | 0,949 | **0,764** | +0,541 | −0,19 | 94% |
| `troi_chay` | 0,585 | 0,747 | **0,504** | +0,653 | −0,47 | 81% |

**`day_du` và `trung_thuc` đạt chuẩn đã chốt trước.** Alpha người–máy (0,789 và 0,764)
**cao hơn** alpha người–người (0,747 và 0,568), tức máy chấm gần với trung bình hai người
hơn là hai người gần nhau. Với hai tiêu chí này, điểm máy trên 50 bài được coi là đã kiểm
chứng trên mẫu người chấm.

**`troi_chay` không đạt, và chính chỗ không đạt là một phát hiện.** Alpha người–máy 0,504
thấp hơn người–người 0,585, lệch trung bình lớn nhất (−0,47) và tỷ lệ lệch ≤1 thấp nhất
(81%). Đáng kể hơn con số alpha: **máy phóng đại khoảng cách trôi chảy giữa các hệ thống**.
Cả 9 kết luận chính đều **cùng chiều** ở người và máy, nhưng ba trong số đó máy tìm ra
khác biệt có ý nghĩa còn người thì không — cả ba đều thuộc `troi_chay`:

| Cặp, tiêu chí `troi_chay` | Người chấm | Máy chấm |
|---|---|---|
| bartpho − lead3 | +0,12 [−0,38, +0,58] | +0,54 [+0,04, +1,08] |
| bartpho − `k2` | +0,21 [−0,42, +0,79] | **+1,17 [+0,62, +1,71]** |
| lead3 − `k2` | +0,08 [−0,17, +0,33] | +0,62 [+0,21, +1,08] |

Chỗ lệch dồn vào `k2`: người cho 4,50 điểm trôi chảy, máy cho 3,42. Máy phạt nặng việc
ghép hai câu rời nhau, người chấm gần như không bận tâm. **Hệ quả cho báo cáo:** phát biểu
"BARTpho trôi chảy hơn hẳn extractive" chỉ được nêu **theo máy chấm**; theo người chấm trên
mẫu, khác biệt trôi chảy giữa bốn hệ thống không đo được.

**Phần lõi của câu hỏi 1 thì đứng vững.** BARTpho kém đầy đủ hơn Lead-3 (−1,38 [−1,83,
−0,83] theo người; −1,00 [−1,54, −0,50] theo máy) và kém hơn `k2` (−1,08 [−1,58, −0,50]
theo người), đồng thời kém trung thực hơn cả hai — bốn phát biểu này có ý nghĩa ở **cả**
người lẫn máy. Kết luận "abstractive trôi chảy hơn nhưng thiếu ý và kém trung thực hơn
extractive" không phụ thuộc vào việc ai chấm.

**Máy khắt khe hơn người ở cả ba tiêu chí** (−0,19 đến −0,47), nên điểm tuyệt đối trong
bảng 50 bài nên đọc như cận dưới, không phải mức người đọc thật sự cảm nhận.

**Máy phạt chính bản do người viết.** Xếp hạng bốn hệ thống trùng khít ở `day_du` và
`troi_chay`; chỉ khác ở `trung_thuc`, và khác đúng tại vị trí của sapo: người xếp sapo
trên BARTpho (4,42 so với 4,25), máy xếp ngược lại (3,83 so với 4,12). Sapo là tham chiếu
vàng do nhà báo viết, nên đây là điểm yếu của máy chấm chứ không phải của sapo — cách giải
thích nhất quán với dữ liệu là máy coi phần sapo **diễn đạt lại** (59,5% bigram của sapo
vốn không có trong bài) là "thông tin không có trong bài gốc".

**Câu hỏi 3 nay có bằng chứng từ người thật.** Trên 36 bản (bỏ sapo vì nó là tham chiếu),
Spearman giữa điểm trung bình ba tiêu chí và điểm tự động:

| | ROUGE-1 | BERTScore |
|---|---|---|
| Người chấm | −0,02 | −0,12 |
| Máy chấm | +0,05 | +0,10 |

Cả bốn con số đều quanh 0. Trước đây kết luận "ROUGE không phản ánh cảm nhận người đọc"
chỉ dựa trên máy chấm (+0,12 trên 50 bài); giờ một mẫu người chấm thật cho cùng câu trả
lời, và đó là chỗ dựa vững hơn hẳn cho câu hỏi 3.

**Giới hạn phải nêu.** 12 bài, 2 người chấm, khoảng tin cậy trên mẫu rộng. Alpha
người–người ở `trung_thuc` chỉ 0,568 — chính hai người cũng chưa thống nhất cao, nên
"máy gần người" ở tiêu chí ấy một phần vì mốc so sánh vốn đã lỏng. Ngược lại alpha máy–máy
ở `trung_thuc` là 0,949: hai lượt máy giống nhau hơn hai người giống nhau, tức máy **nhất
quán** chứ chưa chắc **đúng**.

## Tuần 8 — kết quả cuối trên `test`

Chấm **một lần duy nhất** trên 2.000 bài, bootstrap ghép cặp 10.000 lần, `seed=13`; số sau
dấu ± là nửa khoảng tin cậy 95%. Bảng sinh thẳng từ file trong `results/tables/`.

| Hệ thống | rouge1 | rouge2 | rougeL | Độ dài | 2-gram mới |
|---|---|---|---|---|---|
| Lead-1 (tầng 0) | 27,03 ±0,65 | 14,77 ±0,58 | 20,73 ±0,58 | 36 | 0,0% |
| Lead-3 (tầng 0) | 27,22 ±0,44 | 14,73 ±0,43 | 19,17 ±0,41 | 102 | 0,0% |
| Tầng 2 — PhoBERT `k2` | 31,43 ±0,56 | 17,29 ±0,54 | 22,89 ±0,52 | 67 | 1,2% |
| **Tầng 3 — BARTpho `train_20k`** | **34,55 ±0,76** | **19,83 ±0,72** | **27,26 ±0,72** | 34 | 11,0% |
| Tầng 4 — lọc `lexrank` rồi viết lại | 34,61 ±0,75 | 19,87 ±0,71 | 27,28 ±0,72 | 34 | 11,1% |
| Oracle-3 (trần extractive) | 48,14 ±0,63 | 31,89 ±0,76 | 36,21 ±0,76 | 50 | 1,1% |

**Thứ tự Lead-3 < tầng 2 < tầng 3 đứng vững trên tập kiểm định độc lập**, ở cả ba chỉ số,
mọi p < 0,0001:

| Cặp | rouge1 | rouge2 | rougeL |
|---|---|---|---|
| tầng 2 − Lead-3 | +4,21 [+3,77, +4,64] | +2,56 [+2,17, +2,95] | +3,73 [+3,33, +4,13] |
| tầng 3 − tầng 2 | +3,12 [+2,44, +3,81] | +2,54 [+1,90, +3,17] | +4,36 [+3,70, +5,01] |
| tầng 3 − Lead-3 | +7,33 [+6,65, +8,01] | +5,10 [+4,48, +5,72] | +8,09 [+7,45, +8,73] |

**Tầng 4 không hơn tầng 3:** +0,06 [−0,12, +0,24] ROUGE-1, p = 0,50 (rouge2 p = 0,70,
rougeL p = 0,74). Lặp lại đúng kết luận trên `val`: lọc câu trước khi sinh **không** lấy lại
được phần mất do cắt bài ở 1.024 token. Trên một tập độc lập, đây là kết quả âm tính vững
chứ không phải nhiễu.

**BERTScore xác nhận bằng một thước đo độc lập** (`xlm-roberta-base`, 2.000 bài, 9 hệ thống):

| Hệ thống | BERTScore |
|---|---|
| TextRank | 84,97 ±0,10 |
| LexRank | 85,23 ±0,10 |
| Lead-1 | 85,44 ±0,13 |
| Lead-3 | 85,51 ±0,10 |
| Tầng 2 — PhoBERT `k2` | 86,40 ±0,11 |
| **Tầng 3 — BARTpho** | **87,21 ±0,14** |
| Tầng 4 — lọc `lexrank` | 87,22 ±0,14 |
| Oracle-3 | 89,05 ±0,13 |

| Cặp | BERTScore | p |
|---|---|---|
| tầng 2 − Lead-3 | +0,89 [+0,80, +0,98] | < 0,0001 |
| tầng 3 − tầng 2 | +0,82 [+0,69, +0,95] | < 0,0001 |
| tầng 3 − Lead-3 | +1,71 [+1,58, +1,84] | < 0,0001 |
| tầng 4 − tầng 3 | +0,01 [−0,02, +0,04] | 0,58 |

Cùng một thứ tự, cùng một kết luận âm tính cho tầng 4 — nên cả hai phát hiện chính không phải
hiện tượng riêng của ROUGE. BERTScore còn tách được hai phương pháp đồ thị xuống **dưới**
Lead-3 một cách có ý nghĩa (TextRank −0,54, LexRank −0,27, p < 0,0001), khớp kết luận tuần 3b.
Khoảng cách tuyệt đối nhỏ hơn ROUGE vì BERTScore nén vào dải hẹp (mọi hệ thống đều 85–89).

**Cùng tầm với `val`**, không có dấu hiệu nhầm split: tầng 2 `k2` 31,00 trên `val` → 31,43
trên `test`; BARTpho 35,23 → 34,55.

**Kỷ luật "`test` dùng một lần".** `vit5.py` và `phobert_select.py` vốn từ chối `test`; chốt
được **giữ nguyên** và chỉ mở bằng cờ tường minh `--cho-phep-test`, nên mọi lần gõ nhầm vẫn bị
chặn. Quy tắc `k2` của tầng 2 được **đọc** từ bảng dò trên `tune` (hồ sơ ghi
`chosen_on: tune`), không dò lại trên `test`. Tầng 3–4 chạy bằng kernel
`dl-summarisevn-test-final`, chỉ *đọc* checkpoint qua `kernel_sources` nên không thể ghi đè
output chứa BARTpho.

**Không có ViT5 trên `test`.** Checkpoint ViT5 `train_20k` đã bị ghi đè khi chính kernel
`dl-summarisevn-vit5` huấn luyện BARTpho (12/09), và `kaggle kernels output` không truy cập
được version cũ — xin version 999 vẫn trả về bản mới nhất. Huấn luyện lại tốn ~232 phút GPU
mà không câu hỏi nghiên cứu nào cần tới; kết luận BARTpho hơn ViT5 dẫn từ bảng `val` ở trên.

```bash
~/.venvs/torch/Scripts/python.exe src/models/phobert_select.py score \
    --model <thư mục checkpoint tầng 2> --split test --cho-phep-test
.venv/Scripts/python.exe src/models/phobert_select.py cham-test --cho-phep-test
~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py --dir notebooks/test_final
```

## Hướng mới — một hệ thống duy nhất: đủ ý và không sai sự thật

Xem demo trên bài thật cho thấy mỗi tầng đều bỏ sót sự kiện chính theo một kiểu khác nhau.
Từ đây đề tài đổi mục tiêu: **một** hệ thống, với định nghĩa bản tóm tắt tốt là **đủ ý chính
và không sai sự thật — không cần càng ngắn càng tốt**. Các tầng ở trên trở thành phần phân
tích dẫn tới hệ thống đó.

### Giai đoạn 0 — chốt thước đo TRƯỚC khi thử bất kỳ cấu hình nào

**Vì sao không dùng ROUGE F1 làm chỉ số chính nữa.** F1 trộn precision vào, mà precision
giảm theo độ dài, nên F1 phạt bản tóm tắt dài hơn kể cả khi nó đủ ý hơn. Chính vì vậy BARTpho
— một câu, 34 âm tiết — đứng đầu bảng F1 dù thiếu ý nhiều nhất.

Thước đo mới nằm ở `src/eval/chinh_xac.py`, đếm âm tiết bằng đúng `syllables()` của ROUGE:

| Tiêu chí | Chỉ số | Ý nghĩa |
|---|---|---|
| Đủ ý | **ROUGE-1 recall** | bao nhiêu chữ của sapo có trong bản tóm tắt |
| Đủ ý | **Độ phủ chi tiết** | bao nhiêu tên riêng và con số của sapo có trong bản tóm tắt |
| Không sai sự thật | **Tỷ lệ có chi tiết lạ** | % bản tóm tắt có tên riêng hoặc con số không có trong bài gốc |

Con số phải khớp **kèm ngữ cảnh** (ít nhất một âm tiết bên cạnh giống bài gốc), vì chỉ kiểm
có mặt thì "giá 30 triệu" vẫn lọt khi bài có "30 năm" ở chỗ khác. Cụm tên riêng được ghép từ
các đoạn liền nhau có trong bài, để "Cẩm Lệ Đà Nẵng" không bị coi là bịa khi bài viết
"Cẩm Lệ, TP Đà Nẵng".

**Kiểm chứng bộ đo trước khi tin nó.** Hai lần sửa đều do phép kiểm dưới đây phát hiện:

- **Đối chứng âm** — extractive chép nguyên câu thì không thể có chi tiết lạ. Bản đầu tiên
  gắn cờ nhầm 5–10% bài của Lead-3 và PhoBERT, vì tách `48` ra khỏi `48D`, `8` ra khỏi `8h`
  trong khi ROUGE giữ nguyên chúng. Sau khi sửa: Lead-1, Lead-3, PhoBERT **0/1000**; Oracle-3
  1/1000, do rác sẵn trong dữ liệu ("St. Xem thêm > >").
- **Đối chứng dương** — các lỗi trung thực của BARTpho mà người và máy chấm đã ghi nhận:
  chi tiết bịa hoặc đổi (B05 "quận 12", B08 "5h30", B28 "ngày 8-8") **3/3 bắt được**; gán
  nhầm đối tượng hoặc đảo chủ thể (B36, B42, B45, B27, B43) **0/5**. B36 viết "giá 30 triệu"
  cho nón lá — con số có thật trong bài nhưng là giá của xe đạp, nên so khớp chuỗi về bản chất
  không bắt được.
- **Soi tay 15 cờ ngẫu nhiên của BARTpho**: 8 lỗi sự thật rõ ràng ("hưởng thọ 74 tuổi", "Tuy
  Đức (Đắk Lắk)", bịa chức danh, thêm năm không có trong bài), 4 ca thêm thông tin ngoài bài
  hoặc chưa phân định được, 3 ca bắt nhầm do cách viết ("G 20"/"G20", "năm 2014"/"3/2014",
  "Malaysia Najib Razak"). Tức bộ đo **hơi bắt thừa** — chấp nhận được với mục tiêu không sai
  sự thật, và phải nêu kèm mỗi khi trích con số.

**Mốc trên `val`** (1.000 bài, `results/tables/chinh_xac_val_moc.json`):

| Hệ thống | ROUGE-1 recall | Phủ chi tiết | Có chi tiết lạ | ROUGE-1 F1 | Độ dài |
|---|---|---|---|---|---|
| Lead-1 | 28,9 | 45,2 | 0,0% | 27,7 | 1,0 câu, 35 âm tiết |
| Lead-2 | 44,7 | 60,5 | 0,0% | 29,1 | 2,0 câu, 70 âm tiết |
| Lead-3 | 54,4 | **67,6** | 0,0% | 27,5 | 3,0 câu, 102 âm tiết |
| LexRank | 52,3 | 60,4 | 0,3% | 24,9 | 3,1 câu, 112 âm tiết |
| PhoBERT 3 câu | **55,6** | 66,3 | 0,0% | 28,7 | 3,0 câu, 100 âm tiết |
| PhoBERT `k2` | 46,2 | 58,5 | 0,0% | 31,0 | 2,0 câu, 67 âm tiết |
| BARTpho | 35,6 | 50,9 | **11,8%** | **35,2** | 1,1 câu, 34 âm tiết |
| ViT5 | 31,8 | 48,9 | 7,8% | 33,4 | 1,0 câu, 30 âm tiết |
| Oracle-3 | 57,7 | 66,8 | 0,1% | 48,1 | 1,7 câu, 50 âm tiết |

Theo tiêu chí mới, bảng xếp hạng **đảo ngược**: hệ thống đứng đầu F1 thiếu ý nhất và có chi
tiết lạ nhiều nhất. Oracle-3 **không** phải trần của độ phủ chi tiết ở đây (66,8 < 67,6 của
Lead-3), vì nó được dựng để tối đa F1 nên dừng sớm ở 1,7 câu.

**Recall tự tăng theo độ dài** — chép nguyên cả bài là đạt 100%. Nên so sánh "đủ ý" chỉ có
nghĩa ở **cùng ngân sách độ dài**; đó là lý do quy tắc dưới đây ràng buộc độ dài và chọn mốc
là hai hệ thống ở đúng ngân sách ấy.

**Quy tắc quyết định — chốt trước khi thử:**

1. **Không sai sự thật (ràng buộc cứng):** tỷ lệ bản có chi tiết lạ **≤ 1,0%**. Phần chép
   nguyên câu tự đạt; mọi câu do mô hình sinh viết ra phải qua phép kiểm này.
2. **Độ dài (ràng buộc cứng):** trung bình **≤ 110 âm tiết** và **tối đa 4 câu** mỗi bản.
3. **Đủ ý (mục tiêu):** độ phủ chi tiết **và** ROUGE-1 recall phải **cao hơn cả Lead-3 lẫn
   PhoBERT 3 câu** — hai hệ thống mạnh nhất ở ngân sách khoảng 100 âm tiết — một cách có ý
   nghĩa (bootstrap ghép cặp, khoảng tin cậy 95% không chứa 0).
4. **Luôn báo cáo kèm ROUGE-1 F1**, kể cả khi nó thấp hơn BARTpho — sự đánh đổi phải được nêu.
5. **Dò cấu hình trên `tune`, xác nhận trên `val`, chấm `test` đúng một lần** ở cuối. Trước
   `test`, hệ thống cuối được chấm thêm theo tiêu chí `day_du`/`trung_thuc` trên một mẫu `val`.

```bash
.venv/Scripts/python.exe src/eval/cham_chinh_xac.py --split val --ten moc \
    baselines_val:Lead-1 baselines_val_k2_leadk:Lead-2 baselines_val:Lead-3 baselines_val:LexRank \
    phobert-sent-train_20k_val_len256:phobert-sent phobert-sent-train_20k_val_len256_k2 \
    bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024 vit5-base-train_20k_val_e3_lr3e-05_bs16_in1024 \
    baselines_val:Oracle-3
```

### Giai đoạn 1 — chọn câu có chủ đích

`src/models/chon_cau.py`. Chép nguyên câu của bài nên **không thể bịa**; việc còn lại là
chọn câu cho đủ ý trong ngân sách độ dài. Chạy trên CPU, không cần GPU: điểm câu PhoBERT của
tầng 2 đã lưu sẵn (`phobert-sent-train_20k_{tune,val}_len256_scores.json`).

**Cách chọn.** Mỗi câu có điểm = xác suất PhoBERT + `w_vt` × ưu tiên vị trí (1/(1+k)) +
`w_lex` × độ trung tâm LexRank. Hai thành phần sau có cho **mọi** câu, còn PhoBERT chỉ thấy
256 token đầu (49% số câu nằm ngoài). Chọn tham lam, cộng `lam` × tỷ lệ âm tiết **chưa**
được phủ, bỏ câu mới < 30% âm tiết mới; dừng khi hết ngân sách âm tiết hoặc đủ 4 câu. Tuỳ
chọn: kéo câu đứng trước cho câu mở đầu bằng "Tuy nhiên", "Theo đó"... (`noi_tien_de`);
loại chú thích ảnh, "Xem thêm", mảnh câu (`loc_rac`). Câu bị cắt nhầm ở chữ viết tắt
("chị Th.", "TS.", "GS.TS.", "St.") được **ghép** với câu sau.

**Dò trên `tune`** (500 bài, 216 cấu hình = ngân sách {90, 100, 110} × `w_vt` {0; 0,5; 1} ×
`w_lex` {0; 0,5} × `lam` {0; 0,5; 1} × `noi_tien_de` × `loc_rac`), mục tiêu chốt trước khi dò:
tối đa phủ chi tiết + recall, với ≤ 110 âm tiết trung bình và ≤ 4 câu. Thắng: **ngân sách
110, `w_vt` 1, `w_lex` 0,5, `lam` 1, không nối tiền đề, có lọc rác**
(`results/tables/chon_cau_tune_do.json`). Bỏ từng thành phần khỏi cấu hình thắng, trên `tune`:

| Cấu hình | Phủ chi tiết | ROUGE-1 recall | Âm tiết |
|---|---|---|---|
| **Thắng** | **70,4** | **58,0** | 100,9 |
| bỏ ưu tiên vị trí (`w_vt` 0) | 69,2 | 57,7 | 100,8 |
| bỏ LexRank (`w_lex` 0) | 69,9 | 56,8 | 100,2 |
| bỏ thưởng phủ ý mới (`lam` 0) | 70,0 | 57,6 | 100,9 |
| bật nối tiền đề | 69,5 | 57,6 | 100,7 |
| bỏ lọc rác | 70,4 | 57,9 | 100,9 |
| chỉ PhoBERT + tham lam theo ngân sách | 67,9 | 56,2 | 99,0 |
| *mốc:* Lead-3 / PhoBERT 3 câu | 67,2 / 65,9 | 52,8 / 55,3 | 102,4 / 99,4 |

Mỗi thành phần góp ít (≤ 1,2 điểm) và các chênh lệch trên `tune` **chưa kiểm ý nghĩa**; nhận
xét chắc chắn được là không có "một mẹo" nào — phần hơn đến từ cộng dồn. Nối tiền đề **làm
giảm** cả hai chỉ số (tốn ngân sách cho câu ít thông tin), nên bị tắt. Lọc rác gần như không
đổi chỉ số, giữ vì là cấu hình thắng và bản tóm tắt sạch hơn. Cấu hình thắng được chọn giữa
216 cấu hình trên chính `tune`, nên số `tune` lạc quan — `val` dưới đây là con số thật.

**Xác nhận trên `val`** (1.000 bài, `results/tables/chinh_xac_val_gd1.json`, khoảng tin
cậy 95% bootstrap):

| Hệ thống | ROUGE-1 recall | Phủ chi tiết | Có chi tiết lạ | ROUGE-1 F1 | Độ dài |
|---|---|---|---|---|---|
| **Chọn câu (giai đoạn 1)** | **57,2** [56,0, 58,3] | **69,1** [67,1, 71,1] | **0,0%** | 28,6 | 3,4 câu, 100 âm tiết |
| Lead-3 | 54,4 [53,2, 55,6] | 67,6 [65,6, 69,7] | 0,0% | 27,5 | 3,0 câu, 102 âm tiết |
| PhoBERT 3 câu | 55,6 [54,4, 56,8] | 66,3 [64,3, 68,4] | 0,0% | 28,7 | 3,0 câu, 100 âm tiết |

So cặp (`paired_bootstrap`, 10.000 lần; phủ chi tiết trên 957 bài có sapo chứa chi tiết):

| So sánh | Recall | Phủ chi tiết | F1 (báo kèm) |
|---|---|---|---|
| hơn Lead-3 | +2,79 [+1,94, +3,64], p < 0,0001 | +1,47 [+0,19, +2,74], p = 0,021 | +1,16 [+0,74, +1,57] |
| hơn PhoBERT 3 câu | +1,61 [+0,80, +2,40], p = 0,0002 | +2,75 [+1,31, +4,20], p = 0,0004 | −0,09 [−0,50, +0,33], p = 0,66 |

**Theo quy tắc đã chốt: đạt cả bốn điều** — 0,0% chi tiết lạ (≤ 1,0%), 100 âm tiết và tối đa
4 câu, hơn cả hai mốc ở cả hai chỉ số đủ ý với khoảng tin cậy không chứa 0. F1 ngang PhoBERT
3 câu, vẫn thấp xa BARTpho (35,2) — đúng sự đánh đổi đã nêu ở giai đoạn 0. **Phần hơn Lead-3
về phủ chi tiết mỏng** (cận dưới +0,19): Lead-3 là mốc rất mạnh về tên riêng và con số, vì
tin tức dồn chúng vào đầu bài.

**Phần hơn đó nhạy với cách kiểm.** Quy tắc đã chốt xét từng phép so riêng (khoảng tin cậy
95% không chứa 0) và giữ nguyên như vậy — không đổi quy tắc sau khi thấy kết quả. Nhưng có
**bốn** phép so cùng lúc; nếu hiệu chỉnh Bonferroni (ngưỡng 0,05 / 4 = 0,0125) thì ba phép
vẫn qua, còn **phủ chi tiết hơn Lead-3 (p = 0,021) thì không**. Khi trình bày phải nói đúng
như vậy: chắc chắn hơn cả hai mốc về recall, hơn PhoBERT 3 câu về phủ chi tiết, còn hơn Lead-3
về phủ chi tiết chỉ ở mức gợi ý.

**Một lỗi phát hiện trên `val`, đã sửa và chạy lại.** Lần chấm `val` đầu tiên cho 0,1% chi
tiết lạ — 1/1.000, vô lý với hệ thống chép nguyên câu. Soi bài đó (guid 14423): không phải
bịa, mà bộ tách câu cắt sau học hàm ở dạng tách từ ("trao đổi nhanh với TS . Đào_Trọng_Tứ"),
bản tóm tắt nhận mảnh cụt "...trao đổi nhanh với TS." rồi nối sang câu khác, và bộ đo đọc
"TS Chuyện" thành tên lạ. Tức bộ đo đúng khi nghi ngờ, và lỗi thật là **mất tên chuyên gia**
— chính chi tiết sapo cần. Ghép mảnh câu trước đó chỉ biết chữ viết tắt tên người ("Tr.",
"N."); nay thêm `TS`, `GS`, `PGS`, `ThS`, `TSKH` và `St` ("St. Petersburg"). **Cố ý không
ghép** `NSND`, `NSƯT`, `CN`: trên `tune`/`val` chúng thường đứng cuối câu thật ("...danh hiệu
NSƯT." rồi "Trong đó, ..."). `models/selftest.py` mục 12 kiểm cả hai chiều.

Vì lỗi thấy trên `val`, việc sửa là **sửa tiền xử lý, không chỉnh cấu hình theo `val`**, và
toàn bộ quy trình được chạy lại đúng thứ tự: dò lại 216 cấu hình trên `tune` (chỉ 2/500 bài
`tune` dính lỗi này, số liệu đổi ≤ 0,06, cấu hình thắng giữ nguyên), sinh lại và chấm lại
`val`. Bản tóm tắt đổi ở đúng 5/1.000 bài, cả 5 đều là bài có học hàm hoặc "St.". Trước khi
sửa: recall 57,15, phủ chi tiết 69,03, 0,1% chi tiết lạ — mọi kết luận như sau khi sửa.

**Giới hạn đã biết, mang sang giai đoạn sau:**

- **Câu treo.** 152/1.000 bản `val` có một câu mở đầu bằng từ nối/đại từ ("Tuy nhiên",
  "Trong đó", "Họ"...) mà câu đứng trước nó không được chọn; ở 43 bản đó là câu đầu tiên.
  Danh sách từ rộng nên đây là cận trên, nhưng chắc chắn có ca đọc khó hiểu. Nối tiền đề giải
  quyết được nhưng thua về chỉ số (ở trên). Đây là việc giai đoạn 2 (viết lại) và giai đoạn 3
  (`day_du`/`trung_thuc` do người/máy chấm) phải đo — ROUGE và độ phủ không thấy lỗi mạch văn.
- **"Tối đa 4 câu" tính theo đơn vị câu của hệ thống**, tức câu của bộ tách câu dữ liệu sau
  khi ghép mảnh. Bộ chấm (`sentences_raw`) tách mịn hơn — ở "...", ở chữ viết tắt đã ghép —
  nên đếm > 4 câu ở 23/1.000 bản, tối đa 7. Không phải vi phạm: chính Lead-3, luôn đúng 3 đơn
  vị, cũng bị bộ chấm đếm > 4 câu ở 8/1.000 bài, tối đa 7.
- **Ngân sách 110 âm tiết không chặn câu đầu tiên**: câu được chọn đầu tiên luôn được giữ,
  kể cả khi một mình nó đã dài hơn ngân sách — thà một câu dài còn hơn bản rỗng. Trên `val`
  có 5/1.000 bản vượt 110 âm tiết, dài nhất 145; không vi phạm vì ràng buộc là **trung bình**
  (100). Phân bố vẫn chặt hơn hẳn hai mốc: trung vị 103 và p95 110, so với Lead-3 p95 167,
  tối đa 406, và PhoBERT 3 câu p95 170, tối đa 252 (340 và 319 bản vượt 110).
- **`val` đã được chấm hai lần** (trước và sau khi sửa lỗi học hàm). Cấu hình thắng không
  đổi nên rủi ro thấp, nhưng từ giai đoạn 2, lỗi nhìn thấy trên `val` không nên dẫn tới sửa
  hệ thống nữa — dò và soi lỗi trên `tune`.
- Bản tóm tắt `val` **không commit**, cùng chính sách với baseline: sinh lại tất định trong
  5 giây trên CPU từ điểm PhoBERT đã commit (đã kiểm: sinh lại trùng khít 1.000/1.000).

```bash
.venv/Scripts/python.exe src/models/chon_cau.py do                 # dò trên tune, ~2 phút
.venv/Scripts/python.exe src/models/chon_cau.py sinh --split val   # đọc cấu hình thắng từ file dò
.venv/Scripts/python.exe src/eval/cham_chinh_xac.py --split val --ten gd1 \
    chon-cau_val baselines_val:Lead-3 phobert-sent-train_20k_val_len256:phobert-sent
```

**PhoBERT góp bao nhiêu** — câu hỏi "deep learning nằm ở đâu" khi hệ thống cuối là chọn câu.
Lưới 216 cấu hình luôn bật điểm PhoBERT, nên đối chứng là **tắt PhoBERT rồi dò lại cả lưới**
(công bằng cho bên không có nó), so cặp trên 500 bài `tune`
(`results/tables/chon_cau_phobert_dong_gop.json`):

| | Có PhoBERT | Tắt PhoBERT, dò lại | Hiệu [KTC 95%] |
|---|---|---|---|
| ROUGE-1 recall | 58,03 | 55,66 | **+2,37** [+1,37, +3,40], p < 0,0001 |
| Phủ chi tiết | 70,44 | 69,17 | +1,27 [−0,51, +3,09], p = 0,16 |

PhoBERT là thành phần góp recall **lớn nhất** (các thành phần khác ≤ 1,2 điểm khi bỏ riêng);
không có nó, hệ thống chỉ ngang PhoBERT 3 câu (recall 55,7 so với 55,3). Về phủ chi tiết thì
chưa thấy đóng góp có ý nghĩa — vị trí câu đã mang phần lớn tín hiệu tên riêng và con số.

```bash
.venv/Scripts/python.exe src/models/chon_cau.py do --khong-phobert   # ~2 phút
.venv/Scripts/python.exe src/models/chon_cau.py so-phobert
```

### Giai đoạn 2 — câu treo: hai thử nghiệm BARTpho thất bại, thay bằng luật tất định

Chỗ hụt của giai đoạn 1 là **mạch văn**, không phải ý: 152/1.000 bản `val` có câu mở đầu bằng
từ nối/đại từ mà câu đứng trước không được chọn. Kế hoạch ban đầu là cho BARTpho viết lại có
kiểm soát. Cả hai cách thử đều thất bại, và đó là kết quả cần báo cáo.

**Thử nghiệm âm 1 — ghép câu BARTpho vào đầu bản chọn câu** (`tune`, dùng bản BARTpho đã sinh
từ tuần 5; ngân sách còn lại lấp bằng chọn câu):

| | Recall | Phủ chi tiết | Có chi tiết lạ | Âm tiết |
|---|---|---|---|---|
| **Chọn câu (giai đoạn 1)** | **58,0** | 70,4 | 0,0% | 101 |
| Chỉ BARTpho | 35,6 | 50,7 | 11,8% | 34 |
| BARTpho + chọn câu | 56,6 | **71,7** | 12,2% | 105 |
| BARTpho + chọn câu, bỏ câu BARTpho có chi tiết lạ | 55,6 | 69,2 | 0,4% | 104 |

Câu BARTpho chiếm ~34 âm tiết ngân sách mà phần lớn lặp ý các câu chọn đã có: không lọc thì
bịa, lọc thì thua cả hai chỉ số. Và lọc bằng chính `chi_tiet_la` là tự chấm mình — ràng buộc
≤ 1% gần như đạt theo cấu trúc, trong khi bộ đo đã biết là bắt 0/5 lỗi gán nhầm đối tượng.

**Thử nghiệm âm 2 — BARTpho viết lại câu treo** (`src/models/viet_lai.py`): đầu vào là câu
đứng trước + câu treo. Trước khi sinh đã kiểm BARTpho trên CPU máy này tái lập **đúng từng ký
tự** 6/6 bản `tune` đã sinh trên Kaggle. 67 câu treo trên `tune`, 8,7 phút CPU:

| Trên 67 câu viết lại | |
|---|---|
| có chi tiết lạ | **22** (33%) |
| giữ được ý câu treo (ROUGE-1 recall so với câu treo ≥ 70) | 15 |
| qua cả ba (không chi tiết lạ, không còn treo, giữ ý) | 13 |

BARTpho được huấn luyện để viết sapo cho cả bài, nên với hai câu nó vẫn viết **một câu mở đầu
bản tin mới** chứ không viết lại câu treo. Nguy hiểm hơn là lỗi bộ đo **không bắt được**:
"Iran tuyên bố cắt đứt quan hệ với Qatar" (bài: Saudi, UAE... cắt quan hệ, Iran thì không — #27);
đảo chủ thể "máy bay ném bom tân tiến của nước này [Trung Quốc]" khi bài nói của Nga (#17 —
**nằm trong 13 câu "qua cả ba"**); "Tuyền và Lắm được bổ nhiệm làm Giám đốc Công an TP HCM" —
bịa hoàn toàn (#63). 13 câu qua được phần lớn chỉ là **bỏ từ nối** ("Tuy nhiên, giới quan sát
nhận định..." → "Giới quan sát nhận định...") — việc một luật làm được mà không thể bịa.

**Kết luận: không dùng mô hình sinh trong hệ thống cuối.** BARTpho vẫn là một phần của dự án —
nó thắng ROUGE F1 (tuần 5–8) — nhưng với tiêu chí đủ ý và không sai sự thật, nó hỏng đúng ở chỗ
quan trọng nhất, và hai lần thử dùng nó có kiểm soát đều không qua.

**Cách làm thay thế — hai luật tất định** (`chon_cau.py`, soi trên `tune`):

1. **Bỏ từ nối** (`bo_noi`) khi câu mở đầu bằng "Tuy nhiên, ", "Ngoài ra, ", "Bên cạnh đó, ",
   "Như vậy, ", "Cụ thể, ", "Do đó, ", "Vì vậy, ", "Vì thế, ", "Theo đó, ", "Hơn nữa, ",
   "Thậm chí, " — chỉ khi có dấu phẩy ngay sau, để không cắt nhầm "Cụ thể hoá...". **Cố ý
   không bỏ** "Trước đó", "Sau đó", "Trong khi đó", "Khi đó", "Lúc đó": tiếng Việt không chia
   thì, nhiều khi đó là dấu thời gian duy nhất — "Trước đó, trên mảnh đất có 4 doanh nghiệp" bỏ
   đi thành nói về hiện tại, tức **sai**. "Trong đó", "Tương tự", "Cũng theo" và đại từ cũng
   giữ. Soi cả 98 câu bị bỏ từ nối trên `tune`: đọc tự nhiên, không đổi nghĩa sự kiện.
2. **Trừ điểm câu treo còn lại** (`w_treo`) khi chọn câu, để bộ chọn ưu tiên câu tự đứng được.

**Quy tắc chọn, chốt trước khi dò:** trong các cấu hình mà phủ chi tiết và recall mỗi thứ giảm
không quá 0,5 điểm so với giai đoạn 1 trên `tune` (vẫn ≤ 110 âm tiết, ≤ 4 câu), lấy cấu hình ít
bản còn câu treo nhất. Dò `w_treo` {0; 0,25; 0,5; 1; 2; 5} × bỏ từ nối có/không
(`results/tables/chon_cau_gd2_tune_do.json`):

| `w_treo` | Bỏ từ nối | Phủ chi tiết | Recall | Bản có câu treo (/500) |
|---|---|---|---|---|
| 0 | không (giai đoạn 1) | 70,44 | 58,03 | 82 |
| 0 | có | 70,44 | 57,98 | 40 |
| 0,5 | có | 70,06 | 57,80 | 15 |
| **1** | **có** | **70,09** | **57,54** | **8** |
| 2 | có | 70,09 | 57,47 | 5 — loại: recall −0,56 |

Thắng: `w_treo` 1, bỏ từ nối — recall −0,49, sát ngưỡng 0,5.

**Xác nhận trên `val`, một lần** (`results/tables/chinh_xac_val_gd2.json`):

| Hệ thống | Recall | Phủ chi tiết | Chi tiết lạ | F1 | Bản có câu treo |
|---|---|---|---|---|---|
| Giai đoạn 1 | 57,2 [56,0, 58,3] | 69,1 [67,1, 71,1] | 0,0% | 28,6 | 152 |
| **Giai đoạn 2** | 56,9 [55,8, 58,1] | 68,8 [66,8, 70,8] | 0,0% | 28,6 | **21** |
| Lead-3 | 54,4 | 67,6 | 0,0% | 27,5 | — |
| PhoBERT 3 câu | 55,6 | 66,3 | 0,0% | 28,7 | — |

| Giai đoạn 2 so với | Recall | Phủ chi tiết |
|---|---|---|
| giai đoạn 1 | −0,21 [−0,50, +0,07], p = 0,14 | −0,29 [−0,81, +0,23], p = 0,28 |
| Lead-3 | +2,58 [+1,71, +3,45], p < 0,0001 | **+1,18 [−0,14, +2,52], p = 0,08** |
| PhoBERT 3 câu | +1,40 [+0,58, +2,20], p = 0,001 | +2,46 [+1,00, +3,93], p = 0,0006 |

Câu treo giảm **152 → 21**, không mất chi tiết lạ nào, độ dài giữ 100 âm tiết (6 bản > 110,
dài nhất 145). Nhưng **giai đoạn 2 trượt một điều của quy tắc giai đoạn 0**: phủ chi tiết hơn
Lead-3 không còn có ý nghĩa (khoảng tin cậy chứa 0) — đúng chỗ đã mỏng từ giai đoạn 1 (p = 0,021).
So với giai đoạn 1, mức giảm nhỏ và không có ý nghĩa, nhưng đủ để chạm 0.

**Quyết định: giữ cả hai, không đổi quy tắc sau khi thấy kết quả.** Giai đoạn 1 là hệ thống đạt
quy tắc; giai đoạn 2 là biến thể dễ đọc hơn, trượt một điều kiện. Giai đoạn 3 chấm cả hai và
chọn theo quy tắc chốt trước dưới đây.

**Ghi nhận cho minh bạch:** sau khi đã thấy kết quả `val`, có đếm trên `val` biến thể *chỉ bỏ từ
nối, không trừ điểm* — nó giữ nguyên lựa chọn câu của giai đoạn 1 nên không mất phủ chi tiết, và
câu treo còn 87/1.000. Con số này tìm ra **sau** khi nhìn `val`, nên **không** được chọn làm cấu
hình; ghi lại để giai đoạn 3 biết có phương án đó.

**Quy tắc chọn giữa giai đoạn 1 và 2 ở giai đoạn 3 — chốt trước khi chấm:** chấm cả hai trên
cùng một mẫu `val` theo `day_du`, `trung_thuc`, `troi_chay` (thang 1–5, blind, so cặp theo bài).
Chọn **giai đoạn 2** nếu `troi_chay` cao hơn có ý nghĩa **và** `day_du`, `trung_thuc` không kém
hơn quá 0,25 điểm (cận dưới khoảng tin cậy 95% của hiệu > −0,25); ngược lại chọn **giai đoạn 1**.
Lưu ý từ tuần 7: `troi_chay` do LLM chấm **không** khớp người chấm (máy phóng đại) — tiêu chí này
cần người chấm, hoặc phải kiểm máy chấm lại trên mẫu người trước khi dùng.

**Giới hạn đã biết:**

- Bộ nhận diện câu treo chỉ nhìn **từ đầu câu**. Câu cần ngữ cảnh kiểu khác vẫn lọt, ví dụ gặp
  khi soi `tune`: "- Theo tôi vì nhiều lý do." (trích phỏng vấn, không rõ "tôi"), "...- ông nói."
  (không rõ "ông"). Số 21 là cận dưới của số bản hụt mạch.
- 21 bản còn câu treo là loại không bỏ được: "Đó là...", "Sau đó...", "Trước đó...".
- Bản tóm tắt `val` không commit — sinh lại tất định trong 6 giây.

```bash
.venv/Scripts/python.exe src/models/viet_lai.py thu-ghep                   # thử nghiệm âm 1, tune
~/.venvs/demo/Scripts/python.exe src/models/viet_lai.py ung-vien --split tune  # thử nghiệm âm 2, ~9 phút CPU
.venv/Scripts/python.exe src/models/viet_lai.py phan-tich --split tune     # đếm lại, --in-het để xem từng câu
.venv/Scripts/python.exe src/models/chon_cau.py do-treo                    # dò trên tune
.venv/Scripts/python.exe src/models/chon_cau.py sinh --gd 2 --split val
.venv/Scripts/python.exe src/eval/cham_chinh_xac.py --split val --ten gd2 \
    chon-cau-gd2_val chon-cau_val baselines_val:Lead-3 phobert-sent-train_20k_val_len256:phobert-sent
```

### Giai đoạn 3–4 — kế hoạch: ghép hai hướng thành một câu chuyện

Rà soát 17/09 cho thấy hai hướng đã nối nhau về logic nhưng còn bốn chỗ hở. Giai đoạn 3–4 được
mở rộng để lấp đúng bốn chỗ đó:

| Chỗ hở | Việc | Giai đoạn |
|---|---|---|
| Người đọc chưa chấm hệ thống cuối; tuần 7 chỉ có các hệ thống cũ | Phiếu chấm gồm **giai đoạn 1, giai đoạn 2, BARTpho, Lead-3** trên cùng bài — vừa chọn phiên bản, vừa để người đọc xác nhận cú đảo chiều | 3 |
| Bảng đảo chiều mới có trên `val`; `test` chỉ có ROUGE/BERTScore | Chấm thước đo mới trên `test` cho **mọi hệ thống** (BARTpho, ViT5 nếu có, Lead-3, PhoBERT, hệ thống cuối); `cham_chinh_xac.py` cần mở khoá `test` bằng cờ tường minh | 4 |
| Demo kể câu chuyện cũ (bốn tầng, cố ý không có ô "tốt nhất") | Demo hệ thống cuối **cạnh BARTpho, tô màu chi tiết lạ** | 4 |
| README và báo cáo mở đầu bằng khung cũ; tầng 4 chưa có chỗ trong mạch | Viết lại phần mở đầu thành "phát hiện → giải quyết", thêm câu hỏi nghiên cứu 4; tầng 4 thành một phát hiện ("lọc rồi viết lại cũng không cứu được") | 4 |

**Giai đoạn 3 — thiết kế.** Chỉ 245/1.000 bài `val` có giai đoạn 1 khác giai đoạn 2 (ở các bài
còn lại hai bản trùng chữ, phiếu tự gộp thành một nhãn), và chỉ 16/50 bài của phiếu tuần 7 thuộc
số đó — nên cần phiếu mới, rút theo hai tầng: bài ngẫu nhiên toàn `val` (đại diện, cho so sánh
hệ thống cuối với BARTpho và Lead-3) và bài ngẫu nhiên trong 245 bài hai phiên bản khác nhau (cho
việc chọn phiên bản). Khung chấm dùng lại nguyên tuần 7: ba tiêu chí, bảng mức điểm, tám quy tắc,
blind, xáo nhãn từng bài. Quy tắc chọn giai đoạn 1/2 đã chốt ở cuối mục giai đoạn 2.

### Giai đoạn 3 — chấm blind, dùng lại điểm tuần 7 (chốt trước khi có điểm)

**Đổi so với thiết kế trên (quyết định 17/09):** không rút phiếu mới mà **dùng lại 50 bài tuần
7**, vốn đã có điểm máy hai lượt và điểm hai người (12 bài) cho Lead-3, `k2`, BARTpho, sapo. Chỉ
chấm **bản mới**: giai đoạn 1 và giai đoạn 2 của từng bài, trừ bản trùng từng chữ với một bản đã
chấm (dùng thẳng điểm cũ). `src/eval/cham_gd3.py`:

| | Máy chấm | Người chấm |
|---|---|---|
| Bài | 46 (4 bài mọi bản mới đều trùng bản đã chấm) | 11 trong 12 bài mẫu tuần 7 |
| Bản cần chấm | 69 = 59 bản mới + **10 bản mốc** | 19 = 13 bản mới + 6 bản mốc |
| Bản mới dùng lại điểm cũ | 11 | — |

**Bản mốc** là bản BARTpho và Lead-3 đã chấm ở tuần 7, trộn vào phiếu mới và chấm blind lại, để
đo **độ trôi giữa hai đợt chấm** — nếu đợt mới khắt khe hay dễ hơn, độ lệch ấy lẫn vào mọi hiệu
"hệ thống cuối − BARTpho". Mỗi bài giờ có 1–3 bản thay vì 3–4, nên người chấm mất các bản kia
làm mốc ngầm; bản mốc đo đúng hệ quả này.

**Ai chấm.** Hai lượt máy, **cả hai** do tác tử con độc lập chỉ được đọc năm file
`phieu_phan<k>.md` (không đọc repo, khoá, điểm tuần 7 hay lượt kia). Khác tuần 7, lượt 1 **không**
do phiên dựng phiếu chấm: phiên này đã đọc các bản giai đoạn 1/2 khi làm giai đoạn 2, nên không
còn blind. Hai người chấm mẫu tuần 7 chấm `phieu_doc_mau.html`.

**Kiểm trước khi dựng thật** (chạy thử vào thư mục tạm với điểm giả, không lưu vào `results/`):
phiếu không chứa tên hệ thống nào; 99/99 cặp (nhãn, hệ thống) trong khoá trỏ đúng văn bản; dựng
lại 50 bài tuần 7 trùng khít phiếu và khoá đã phát; `analyze` chạy hết bốn phần.

**Quy tắc phân tích — chốt trước khi có điểm:**

1. **Độ trôi** (10 bản mốc, điểm mới − điểm cũ, trung bình hai lượt máy mỗi đợt): ghép điểm hai
   đợt là hợp lệ cho một tiêu chí khi |trôi trung bình| ≤ 0,25 **và** alpha mới–cũ ≥ 0,667. Không
   đạt thì mọi so sánh chéo đợt ở tiêu chí đó chỉ nêu kèm độ trôi, không viết thành kết luận.
2. **Cú đảo chiều** (50 bài, điểm máy): so cặp theo bài giai đoạn 1 và 2 với BARTpho, Lead-3,
   `k2`, ba tiêu chí.
3. **Chọn phiên bản** (16 bài có giai đoạn 1 ≠ 2, điểm máy): giai đoạn 2 khi `troi_chay` hơn có
   ý nghĩa **và** cận dưới khoảng tin cậy của `day_du`, `trung_thuc` đều > −0,25; ngược lại giai
   đoạn 1. `troi_chay` của máy chưa được kiểm chứng ở tuần 7 — nêu kèm. Không đủ bằng chứng thì
   viết "chưa chứng minh được giai đoạn 2 trôi chảy hơn".
4. **Người chấm** (12 bài): cùng so sánh ở mục 2, dùng để kiểm **chiều** so với máy.

```bash
.venv/Scripts/python.exe src/eval/cham_gd3.py prepare    # đã chạy; từ chối chạy lại khi gd3/khoa.json đã có
.venv/Scripts/python.exe src/eval/cham_gd3.py analyze    # sau khi thu llm_judge_1..2.csv (và cham_mau_nguoi1..2.csv)
```

#### Kết quả phần máy chấm (người chấm mới: không làm — xem cuối mục)

Hai tác tử con chấm đủ 69/69 bản mỗi lượt (`gd3/llm_judge_1.csv`, `_2.csv`), không lượt nào thấy
lượt kia. Lượt 2 có ghi một script tạm ra thư mục cha của thư mục chấm để viết file; theo báo cáo
của nó, không đọc gì ngoài năm phiếu. Kết quả: `results/tables/cham_gd3_val.json`.

**Hai lượt mới đồng thuận cao, ngang tuần 7:** alpha `day_du` 0,90, `trung_thuc` 0,92,
`troi_chay` 0,85 (tuần 7: 0,82 / 0,92 / 0,76).

**1. Độ trôi giữa hai đợt** (10 bản mốc, đã tính tay lại khớp):

| Tiêu chí | Trôi (mới − cũ) | alpha mới–cũ | Ghép hai đợt |
|---|---|---|---|
| `day_du` | **−0,55** (đợt mới khắt khe hơn) | 0,76 | **không đạt** |
| `trung_thuc` | −0,25 (đúng ngưỡng) | 0,96 | đạt |
| `troi_chay` | **+0,35** (đợt mới dễ hơn) | 0,74 | **không đạt** |

Theo quy tắc đã chốt, so sánh chéo đợt ở `day_du` và `troi_chay` **chỉ được nêu kèm độ trôi,
không viết thành kết luận**. Mỗi bài giờ chỉ có 1–3 bản thay vì 3–4 — nguyên nhân khả dĩ, chưa
kiểm được.

**2. Hệ thống cuối so với các hệ tuần 7** (50 bài, điểm máy trung bình hai lượt):

| Hệ thống | `day_du` | `trung_thuc` | `troi_chay` |
|---|---|---|---|
| sapo | 3,41 | 4,03 | 4,62 |
| Lead-3 | 3,48 | 5,00 | 3,83 |
| PhoBERT `k2` | 3,11 | 4,96 | 3,37 |
| BARTpho | 2,72 | 4,32 | 4,41 |
| Giai đoạn 1 | 3,23\* | 4,87 | 3,54\* |
| Giai đoạn 2 | 3,23\* | 4,87 | 3,56\* |

\* chấm ở đợt mới, trôi vượt ngưỡng — không so thẳng với các dòng trên.

| Giai đoạn 1 so với | `day_du` | `trung_thuc` (ghép hợp lệ) | `troi_chay` |
|---|---|---|---|
| BARTpho | +0,51 [+0,21, +0,80] \* | **+0,55 [+0,26, +0,85], p < 0,0001** | −0,87 [−1,18, −0,56] \* |
| Lead-3 | −0,25 [−0,47, −0,03] \* | −0,13 [−0,26, −0,04], p = 0,001 | −0,29 [−0,54, −0,06] \* |
| PhoBERT `k2` | +0,12 [−0,10, +0,34] \* | −0,09 [−0,16, −0,03], p = 0,001 | +0,17 [−0,10, +0,45] \* |

Đọc bảng:

- **Trung thực hơn BARTpho: là kết luận** (ghép hợp lệ, p < 0,0001) — người đọc (máy chấm đã kiểm
  chứng ở tuần 7 cho tiêu chí này) xác nhận vế "không sai sự thật" của cú đảo chiều.
- **Đầy đủ hơn BARTpho, kém trôi chảy hơn BARTpho: chưa là kết luận** vì trôi. Nhưng trôi đi
  **ngược chiều** với cả hai hiệu (đợt mới khắt khe hơn về đầy đủ mà hệ thống cuối vẫn hơn; dễ hơn
  về trôi chảy mà vẫn kém) — gợi ý hiệu thật không nhỏ hơn. Đây là nhận xét, không phải phép kiểm
  đã chốt, và bản mốc chỉ gồm BARTpho/Lead-3 nên độ trôi của bản chọn câu có thể khác.
- **Kém Lead-3 về trung thực** −0,13 dù ghép hợp lệ — nhưng nhỏ hơn chính độ trôi (−0,25) ở tiêu
  chí đó, nên chưa phân biệt được với độ trôi. Kém Lead-3 về đầy đủ −0,25 thì trôi −0,55 còn lớn
  hơn — không kết luận được, và **mâu thuẫn chưa giải với chỉ số tự động** (recall, phủ chi tiết
  hơn Lead-3 có ý nghĩa).

**3. Chọn phiên bản** (16 bài giai đoạn 1 ≠ 2): `day_du` +0,00, `trung_thuc` +0,00, `troi_chay`
+0,06 [−0,56, +0,59] → **theo quy tắc: giai đoạn 1**. Chưa chứng minh được giai đoạn 2 trôi chảy
hơn — máy chấm gần như không thấy khác biệt ở 16 bài này.

**Phát hiện: chép nguyên câu vẫn có thể làm hiểu sai.** 7 bản giai đoạn 1 bị trừ trung thực ở ít
nhất một lượt; theo ghi chú, lý do chính là **ghép câu mất ngữ cảnh**, không phải bịa: B43 "9h
cùng ngày" đứng sau câu "Sáng 13/3" nên đọc thành 13/3 trong khi bài là 9/3 (2/3 điểm); B42 "Nhóm
cán bộ này" treo nên dễ hiểu nhầm đối tượng; B16 "ông nói", "điều đó" không rõ chỉ ai. Nên câu
"chép nguyên câu nên **không thể sai sự thật**" (docstring `chon_cau.py`) là **quá mạnh** — đúng là không bịa
chi tiết (0% chi tiết lạ), nhưng sắp đặt câu vẫn tạo được hàm ý sai. Bộ nhận diện câu treo (chỉ
nhìn đầu câu) không bắt được "cùng ngày" nằm giữa câu. Phải nói đúng như vậy khi trình bày:
hệ thống cuối **không bịa**, chứ không phải "không thể sai" (docstring `chon_cau.py` đã sửa theo).

#### Quyết định: không chấm người mới, dùng lại người chấm tuần 7 trong phạm vi của nó

Người chấm mới **không làm** (quyết định 17/09); phiếu 19 bản `gd3/phieu_doc_mau.html` và
`cham_mau_nguoi1..2.csv` vẫn để trống trong repo, có người chấm thì bổ sung sau và `analyze` tự
thêm phần 4. Phiếu người tuần 7 **không thay được**: nó chấm sapo, Lead-3, `k2`, BARTpho — chỉ 3/12
bài có bản giai đoạn 1 trùng chữ với một bản đã chấm. Nó được dùng lại đúng trong phạm vi sau:

| Dùng lại cho | Vì sao được |
|---|---|
| Phần **phát hiện**: BARTpho trôi chảy nhưng thiếu ý và kém trung thực hơn extractive; ROUGE không phản ánh người đọc | chính là kết quả người chấm tuần 7 |
| Chỗ dựa cho **"hệ thống cuối trung thực hơn BARTpho"** | chuỗi kiểm chứng: máy tuần 7 khớp người ở `trung_thuc` (alpha người–máy 0,76) → máy giai đoạn 3 khớp máy tuần 7 trên bản mốc (alpha 0,96, trôi −0,25 trong ngưỡng) |

| Không dùng lại được cho | Vì sao |
|---|---|
| "Người đọc chấm hệ thống cuối" | 3 bài, không tính được gì |
| `day_du`, `troi_chay` của hệ thống cuối | chuỗi đứt: máy mới lệch máy cũ vượt ngưỡng (−0,55, +0,35) |
| Mâu thuẫn "kém Lead-3 về đầy đủ" (máy) với recall/phủ chi tiết hơn Lead-3 (tự động) | không có điểm người để phân xử — **để mở**, nêu khi trình bày |

**Cách nói khi trình bày — đúng mức bằng chứng:**

- ✅ "Người chấm (tuần 7) xác nhận BARTpho kém trung thực và thiếu ý; ROUGE không phản ánh người đọc."
- ✅ "Hệ thống cuối trung thực hơn BARTpho (máy chấm, p < 0,0001); tiêu chí này của máy chấm đã
  được kiểm chứng với người chấm."
- ⚠️ "Máy chấm cho thấy hệ thống cuối đầy đủ hơn nhưng kém trôi chảy hơn BARTpho — chưa kiểm chứng
  bằng người, và hai đợt chấm lệch nhau ở hai tiêu chí này."
- ❌ Không nói "người đọc đánh giá hệ thống cuối tốt hơn".
- ❌ Không nói "không thể sai sự thật" — chỉ "không bịa chi tiết".

### Giai đoạn 4 — chấm `test` một lần (chốt trước khi mở `test`)

**Hệ thống cuối: giai đoạn 1**, cấu hình đọc từ `chon_cau_tune_do.json` (chọn trên `tune`). Giai đoạn
2 **không** chấm trên `test` — nó không được chọn, và chấm thêm trên `test` là mở cửa cho việc chọn
lại theo `test`.

**Các hệ thống trên cùng bảng** (2.000 bài `test`, `cham_chinh_xac.py --split test --cho-phep-test`):
hệ thống cuối; Lead-1, Lead-3, LexRank, Oracle-3 (`baselines_test`); PhoBERT 3 câu (dựng lại từ
điểm đã lưu bằng `phobert_select.py sinh-k3` — trên `val` trùng từng chữ 1.000/1.000 file Kaggle);
PhoBERT `k2`; BARTpho (tầng 3); tầng 4 (lọc `lexrank` rồi viết lại). Không có ViT5 trên `test`
(tuần 8, mục 5).

**Quy tắc — y nguyên giai đoạn 0, không đổi:** (1) có chi tiết lạ ≤ 1,0%; (2) ≤ 110 âm tiết trung
bình, ≤ 4 câu; (3) recall **và** phủ chi tiết hơn **cả** Lead-3 lẫn PhoBERT 3 câu, bootstrap ghép
cặp, khoảng tin cậy 95% không chứa 0; (4) báo cáo kèm F1. Kết quả được ghi **như nó ra**, kể cả khi
trượt; không sửa hệ thống sau khi thấy `test`. Kèm theo là **bảng đảo chiều trên `test`**: xếp hạng
theo F1 so với theo recall/phủ chi tiết/chi tiết lạ, cho mọi hệ thống trên.

**Chốt khoá đã kiểm:** `cham_chinh_xac.py`, `chon_cau.py sinh`, `phobert_select.py sinh-k3` đều từ
chối `test` khi thiếu `--cho-phep-test`; `cham_chinh_xac.py` từ chối chấm lại khi bảng `test` đã
có; bộ chấm sau khi thêm khoá ra trùng khít bảng giai đoạn 1 trên `val`.

```bash
.venv/Scripts/python.exe src/models/chon_cau.py sinh --gd 1 --split test --cho-phep-test
.venv/Scripts/python.exe src/models/phobert_select.py sinh-k3 --split test --cho-phep-test
.venv/Scripts/python.exe src/eval/cham_chinh_xac.py --split test --cho-phep-test \
    chon-cau_test baselines_test:Lead-1 baselines_test:Lead-3 baselines_test:LexRank \
    phobert-sent-train_20k_test_len256:phobert-sent phobert-sent-train_20k_test_len256_k2 \
    bartpho-syllable-train_20k_test_in1024 bartpho-syllable-train_20k_test_in1024_loc-lexrank:bartpho-syllable-train_20k=tang4 \
    baselines_test:Oracle-3
```

## Demo Gradio — bốn tầng chạy cạnh nhau trên máy

Dán một bài báo, xem bốn hướng tóm tắt nó khác nhau thế nào. Chạy hoàn toàn trên CPU.

```bash
~/.venvs/demo/Scripts/python.exe app/app.py     # mở http://127.0.0.1:7860
```

Demo cố ý **không có ô "kết quả tốt nhất"**: mục đích của nó là cho thấy hai hồ sơ lỗi
bù trừ nhau đúng như mục phân loại lỗi ở trên — extractive gần như không sai sự thật
nhưng đứt mạch và bỏ sót ý, abstractive trôi chảy nhưng thiếu ý và có thể hoán đổi chi tiết.

**Môi trường riêng** `~/.venvs/demo`, để `.venv` chính vẫn sạch `torch` đúng thiết kế:
`requirements.txt` + torch (CPU) + `transformers==5.0.0` + `sentencepiece` + `gradio` +
`underthesea`. Bản đã chạy được: torch 2.14.0+cpu, transformers 5.0.0, gradio 6.27.0,
underthesea 9.5.0.

**Checkpoint để ngoài repo**, ở `~/.cache/dl-summarisevn/` — 2,46 GB: BARTpho `train_20k`
1,91 GB và tầng 2 540 MB kèm `head.pt`.

### Lấy checkpoint về: `kernels output` trả file trọng số 0 byte

`kaggle kernels output` **không** phục vụ file lớn. Nó tải về đầy đủ mọi file nhỏ —
`config.json`, `head.pt`, `bpe.codes`, `dict.txt` — nhưng `model.safetensors` là **0 byte**,
và với BARTpho nó còn bỏ sót hẳn thư mục `final/` mà các kernel khác vẫn dùng. Không có
thông báo lỗi nào.

Trọng số **vẫn còn nguyên bên Kaggle**: `tang4` và `tang2_score` đọc chúng qua
`kernel_sources` và chạy thành công. Nên cách lấy về là một kernel CPU đọc lại rồi chép
sang `/kaggle/working` (`notebooks/xuat_ckpt/`, kernel `dl-summarisevn-xuat-ckpt`) —
**không phải huấn luyện lại**:

```bash
~/.venvs/kaggle/Scripts/kaggle.exe kernels push -p notebooks/xuat_ckpt
~/.venvs/kaggle/Scripts/kaggle.exe kernels output minh12605/dl-summarisevn-xuat-ckpt \
    -p ~/.cache/dl-summarisevn/xuat
```

Đừng đẩy kernel này bằng `notebooks/kaggle_push.py`: script đó ghim cứng
`--accelerator NvidiaTeslaT4`, tức xin GPU cho một việc chỉ chép file.

### Hai lỗi âm thầm mà demo chặn thẳng

Cả hai đều chạy trót lọt và cho ra kết quả **trông hợp lý** nếu không chặn:

1. **Trọng số rỗng.** Kiểm sự tồn tại của `head.pt` là *chưa đủ* — bản tải hỏng vẫn có nó.
   `kiem_trong_so()` kiểm **kích thước** file trọng số. Thiếu `head.pt` thì lớp cho điểm
   tầng 2 là trọng số khởi tạo ngẫu nhiên và nó chọn câu bừa mà không báo gì.
2. **Đầu vào chưa tách từ.** `sentences()` cắt câu theo ranh giới `" . "` của bộ VietNews;
   văn bản người dùng gõ có dấu chấm dính liền chữ nên nó thấy **đúng một câu**, và mọi
   tầng extractive trả về **nguyên bài**. `chuan_bi()` báo lỗi khi văn bản có từ hai dấu
   kết câu trở lên mà chỉ cắt được một câu — và cố ý *không* báo lỗi với bài thật sự một câu.

`tim_checkpoint()` cũng ưu tiên thư mục có trọng số thật khi trên máy có nhiều bản tải:
chọn nhầm bản hỏng sẽ báo lỗi y hệt như chưa tải gì.

**Chi phí và giới hạn.** Khoảng 15–25 giây một bài trên CPU, phần lớn là nạp mô hình cho
lần chạy đầu (mô hình được giữ lại cho các lần sau). Với bài ngắn, Lead-3 và LexRank có
thể cho ra cùng một kết quả — đó là trùng hợp trên bài ít câu, không phải lỗi.

## Cấu trúc

```
data/raw/          dữ liệu tải về, không chỉnh sửa
data/processed/    đã chuẩn hoá và khử tách từ
data/splits/       file ID cố định của train/val/test
notebooks/         kaggle_train_vit5.ipynb (huấn luyện tầng 3), sweep/ và
                   sweep_bartpho/ (dò tham số sinh), tang4/ (lọc câu lúc suy luận),
                   tang4_train/ (huấn luyện trên đầu vào đã lọc), tang2/ (PhoBERT
                   chọn câu), tang2_score/ (cho điểm câu bằng checkpoint tầng 2),
                   kaggle_push.py (đẩy lên Kaggle)
src/data/          text.py (3 phép biến đổi dùng chung), splits.py (nạp),
                   make_splits.py (đóng băng), inspect_vietnews.py (kiểm tra),
                   browse.py (duyệt dữ liệu trên trình duyệt)
src/models/        extractive.py (tầng 0-1), phobert_sent.py (tầng 2, cần GPU),
                   phobert_select.py (tầng 2 chọn số câu linh hoạt),
                   vit5.py (tầng 3, cần GPU), hybrid.py (tầng 4), run_baselines.py
                   (chạy + chấm), measure_tokens.py (đo độ dài cắt), selftest.py
src/eval/          rouge.py (đã đối chiếu Google), stats.py (bootstrap),
                   report.py (khung chấm điểm), bertscore.py, selftest.py,
                   truncation.py (câu hỏi 2, cần transformers),
                   run_bertscore.py (chấm BERTScore, cần torch),
                   tang4_sosanh.py (tầng 4 vòng 1/2 và đối chứng, cần sentencepiece),
                   human_eval.py (tuần 7: dựng phiếu chấm blind, phân tích phiếu,
                   phiếu mẫu cho người chấm và so người với máy chấm)
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

**Môi trường phân tích** — cho các script cần tokenizer nhưng không cần GPU
(`truncation.py`, `measure_tokens.py`). Tách riêng khỏi `.venv` để không kéo lệch các
phiên bản đã ghim, và ghim `transformers==5.0.0` cho trùng bản Kaggle đã huấn luyện,
để số token đếm được đúng là thứ mô hình đã đọc:

```bash
python -m venv ~/.venvs/phan-tich
~/.venvs/phan-tich/Scripts/python.exe -m pip install -r requirements.txt "transformers==5.0.0"
~/.venvs/phan-tich/Scripts/python.exe src/eval/truncation.py
```

## Chạy

```bash
.venv/Scripts/python.exe src/data/inspect_vietnews.py   # kiểm tra dữ liệu
.venv/Scripts/python.exe src/data/make_splits.py        # đóng băng tập con (chạy MỘT lần)
.venv/Scripts/python.exe src/models/run_baselines.py    # baseline tầng 0-1 trên test
.venv/Scripts/python.exe src/models/measure_tokens.py   # đo độ dài cắt (cần transformers)
.venv/Scripts/python.exe src/data/browse.py             # duyệt dữ liệu: http://127.0.0.1:8009
```

`browse.py` đọc thẳng file parquet trong cache Hugging Face và phục vụ một trang có phân
trang và tìm kiếm, không xuất file trung gian. Mỗi bài hiện nhãn tập con đã đóng băng
chứa nó (`test`, `train_5k`, …). Khi tìm, dấu cách và gạch dưới được coi là một, nên gõ
"Triều Tiên" vẫn khớp với văn bản đã tách từ `Triều_Tiên`.

Tự kiểm tra, chạy lại sau mỗi lần sửa module tương ứng — cả hai đều không cần mạng và
xong trong vài giây:

```bash
.venv/Scripts/python.exe src/eval/selftest.py     # ROUGE, bootstrap
.venv/Scripts/python.exe src/models/selftest.py   # tầng 0-1
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
- [x] Tuần 3a — Khung đánh giá: ROUGE, bootstrap, bảng kết quả
- [x] Tuần 3b — Baseline tầng 0–1 (LexRank bản nhúng PhoBERT hoãn vì thiếu `torch`,
  đã làm xong ở tuần 5: thua cả bản TF-IDF)
- [x] Tuần 4 — Fine-tune ViT5 lần đầu (đối chứng BARTpho dời sang tuần 5)
- [x] Tuần 5 — Đường cong học ba điểm, dò tham số sinh trên `tune`, câu hỏi 2,
  LexRank bản PhoBERT, BERTScore, đối chứng BARTpho (BARTpho thắng ViT5)
- [x] Rà soát tuần 5 — lấp các chỗ hở phát hiện khi review (xem dưới)
- [x] Tuần 6 — Tầng 2 và tầng 4. Tầng 2: hệ thống extractive đầu tiên vượt Lead-3
  (+1,24). Tầng 4 vòng 1: lọc lúc suy luận không lấy lại được thiệt hại do cắt.
  BERTScore xác nhận cả hai kết luận.
- [x] Phần còn lại của tuần 6:
  - [x] đối chứng `--no-train` không lọc — trùng khít hai bản lọc 898/898 ở nhóm không
    lọc; kết luận vòng 1 không đổi
  - [x] tầng 4 vòng 2, huấn luyện lại trên đầu vào đã lọc — cũng không lấy lại được
    thiệt hại do cắt (hiệu hai hiệu −0,17 [−2,35, +2,00])
  - [x] tầng 2 chọn số câu linh hoạt — `k2` chọn trên `tune`, đạt 31,00 trên `val`
    (+2,30 so với `k3`, +1,91 so với mốc Lead-2 thêm sau)
- [x] Tuần 7 — Người chấm, phân tích lỗi, demo Gradio
  - [x] công cụ chấm blind: phiếu 50 bài `val` × 4 hệ thống đã dựng và kiểm (`human_eval.py`)
  - [x] chấm phiếu — **không có người chấm**; thay bằng hai lượt chấm của mô hình ngôn ngữ,
    ghi rõ là máy chấm (`llm_judge.csv`, `llm_judge_2.csv` do tác tử độc lập chấm), alpha
    0,76–0,92: BARTpho trôi chảy nhất nhưng thiếu ý và kém trung thực hơn extractive;
    tương quan với ROUGE-1 chỉ +0,12
  - [x] phiếu mẫu 12 bài cho người chấm kiểm chứng máy chấm (`prepare-mau`), đã kiểm là một
    phần trùng khít của phiếu 50 bài
  - [x] 2 người chấm điền `cham_mau_nguoi1..2.csv`, đã chạy `so-sanh`: `day_du` và
    `trung_thuc` đạt chuẩn kiểm chứng (alpha người–máy cao hơn alpha người–người),
    `troi_chay` không đạt — máy phóng đại khác biệt trôi chảy; 9/9 kết luận chính cùng
    chiều; người chấm cũng không tương quan với ROUGE-1 (−0,02), củng cố câu hỏi 3
  - [x] phân tích lỗi định tính — 190 ghi chú của 2 lượt máy và 2 người chấm, phân thành
    bốn họ lỗi (thiếu ý, sai sự thật, mạch văn đứt, rác kế thừa từ dữ liệu); hai hồ sơ lỗi
    của extractive và abstractive bù trừ nhau; ba ví dụ "bịa" rõ nhất đều thuộc sapo
  - [x] demo Gradio — `app/pipeline.py` + `app/app.py`, bốn tầng chạy trên CPU với trọng số
    thật; kèm kernel `dl-summarisevn-xuat-ckpt` lấy lại checkpoint mà `kernels output`
    trả về 0 byte
- [ ] Tuần 8 — Báo cáo, kiểm tra tái lập
  - [x] chấm tầng 2, 3, 4 trên `test` (tầng 0–1 đã có từ tuần 3b): Lead-3 27,22 < tầng 2
    31,43 < tầng 3 34,55, mọi cặp p < 0,0001; tầng 4 không hơn tầng 3 (+0,06, p = 0,50)
  - [x] BERTScore trên `test` — cùng thứ tự Lead-3 < tầng 2 < tầng 3 (mọi cặp p < 0,0001),
    tầng 4 bằng tầng 3 (+0,01, p = 0,58)
  - [ ] báo cáo và slide
  - [ ] kiểm tra tái lập trên máy sạch
- [ ] Hướng mới — một hệ thống duy nhất: đủ ý và không sai sự thật
  - [x] giai đoạn 0: chốt thước đo (`eval/chinh_xac.py`), kiểm chứng bằng đối chứng âm/dương
    và soi tay, đo mốc trên `val`, chốt quy tắc quyết định trước khi thử
  - [x] giai đoạn 1: chọn câu có chủ đích (`models/chon_cau.py`) — trên `val` recall 57,2 và
    phủ chi tiết 69,1, hơn cả Lead-3 lẫn PhoBERT 3 câu có ý nghĩa, 0,0% chi tiết lạ, 100 âm
    tiết; sửa lỗi cắt câu ở học hàm (phát hiện trên `val`) rồi chạy lại từ `tune`
  - [x] giai đoạn 2: hai thử nghiệm BARTpho có kiểm soát đều thất bại (ghép câu giảm chỉ số;
    viết lại câu treo 33% có chi tiết lạ, có lỗi đảo chủ thể bộ đo không bắt) → không dùng mô
    hình sinh; thay bằng bỏ từ nối + trừ điểm câu treo: câu treo `val` 152 → 21, nhưng phủ chi
    tiết hơn Lead-3 không còn có ý nghĩa (p = 0,08) → giữ cả hai, giai đoạn 3 chọn. PhoBERT góp
    +2,37 recall (tune, p < 0,0001)
  - [x] giai đoạn 3: chấm blind trên 50 bài tuần 7 (dùng lại điểm cũ, 10 bản mốc), hai lượt máy
    bằng tác tử con — **hệ thống cuối là giai đoạn 1**; trung thực hơn BARTpho +0,55 (p < 0,0001);
    `day_du`/`troi_chay` so chéo đợt chỉ tham khảo (trôi vượt ngưỡng); ghép câu vẫn có thể tạo hàm
    ý sai. Người chấm mới không làm — dùng lại người chấm tuần 7 trong phạm vi đã nêu
  - [ ] giai đoạn 4: chấm thước đo mới trên `test` cho mọi hệ thống (một lần); demo hệ thống
    cuối cạnh BARTpho có tô chi tiết lạ; viết lại mở đầu README/báo cáo theo khung "phát hiện →
    giải quyết", thêm câu hỏi nghiên cứu 4

### Rà soát tuần 5 — đã lấp và còn nợ

Sáu hạng mục tuyên bố của tuần 5 đều có đủ hiện vật và tái lập được từ file đã lưu.
Nhưng một lượt rà soát tìm ra sáu chỗ hở, phần lớn sinh ra từ đúng một sự kiện:
**BARTpho thắng ViT5 ở cuối tuần 5**, sau khi mọi khảo sát đã làm xong trên ViT5.

Đã lấp:

- [x] **LexRank-PhoBERT trên `val`** — trước chỉ có trên `test` nên đứng ngoài mọi
  bảng dùng `val`. Kết luận lặp lại trên split độc lập: 23,76 so với 23,87.
- [x] **BERTScore cho LexRank-PhoBERT** — bảng BERTScore giờ đủ chín hệ thống. Nó
  thua bản TF-IDF ở cả thước đo thứ hai, tức không phải hiện tượng của riêng ROUGE.
- [x] **Hồ sơ lần chạy cho baseline** — `run_baselines.py` ghi `_run.json` như
  `vit5.py`. Chính nó phát hiện khoảng chênh 3,7 lần về tốc độ giữa hai máy.
- [x] **Phép kiểm tự động cho LexRank bản nhúng** — `selftest.py` mục 8, chạy được
  không cần `torch`. Trước đó đây là hệ thống duy nhất của dự án không có phép kiểm.
- [x] **Dò tham số sinh cho BARTpho** — `notebooks/sweep_bartpho/`, 7 cấu hình, 52,9
  phút GPU. Không cấu hình nào khác mặc định; `length_penalty` là vùng phẳng hai phía.

Còn nợ:

- [ ] **ViT5 `train_2k`** (~22 phút GPU) — tập con này đã đóng băng từ tuần 2 nhưng
  chưa bao giờ dùng; đường cong học vì thế có 3 điểm chứ không phải 4. Điểm 2k nằm ở
  chỗ đường cong dốc nhất, trong khi bước 5k → 10k hiện không đo được.
  **Còn ràng buộc thứ tự:** nó đẩy lên kernel `dl-summarisevn-vit5` và khiến checkpoint
  BARTpho không còn lấy được qua `kernel_sources` — trong khi checkpoint ấy vẫn cần
  cho lần chấm `test` ở tuần 8. (Đối chứng `--no-train` không lọc đã chạy xong nên không
  còn phụ thuộc vào nó.) Hoặc chấm `test` trước, hoặc đẩy `train_2k` lên kernel riêng.
- [ ] **Đường cong học cho BARTpho** (~2 giờ GPU) — hiện chỉ có một điểm `train_20k`.
  Không bắt buộc: có thể trình bày đường cong như một khảo sát *trên ViT5*, nói rõ vậy.
- [ ] **Chấm mọi tầng trên `test`** — việc của tuần 8, đúng thiết kế. Hiện tầng 0–1
  chấm trên `test` còn tầng 3 chấm trên `val`, nên chưa có bảng nào đặt được mọi tầng
  cạnh nhau trên cùng một split.
