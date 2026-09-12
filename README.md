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
chiều ngược lại phải nhờ `underthesea`, khác công cụ với VnCoreNLP đã tách tham chiếu,
và sai khác công cụ đó chỉ giáng lên phía abstractive. Có thể báo cáo thêm ROUGE trên
dạng tách từ để đối chiếu với các bài báo trước, kèm ghi chú về bất lợi này.

Quy trình blind cũng dùng đúng dạng đó: nếu bản tóm tắt extractive còn nguyên gạch
dưới còn bản của ViT5 thì không, người chấm nhận ra ngay đâu là hệ thống nào.

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

**Còn thiếu ở tầng 1:** LexRank bản nhúng PhoBERT cần `torch` và `transformers`, chưa
có trong `.venv` trên máy này (xem `requirements.txt`) nên dời sang tuần 4, chạy cùng
lúc dựng môi trường GPU. Phần xếp hạng đã viết chung ở `pagerank()` nên chỉ cần thay
cách dựng ma trận tương đồng.

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

| Mô hình | ViT5, không bị cắt | ViT5, bị cắt | Hiệu của hiệu |
|---|---|---|---|
| `train_5k` | 32,76 ±0,98 | 25,60 ±2,57 | −1,37 [−4,28, +1,58], p = 0,36 |
| `train_10k` | 33,16 ±1,01 | 24,94 ±2,41 | −2,43 [−5,27, +0,44], p = 0,10 |
| **`train_20k`** | **34,31 ±1,04** | **24,50 ±2,59** | **−4,02 [−6,84, −1,12], p = 0,007** |

**Toàn bộ phần lợi của việc thêm dữ liệu rơi vào nhóm bài không bị cắt**: 32,76 → 33,16
→ 34,31 ở nhóm vừa cửa sổ, trong khi nhóm bị cắt đứng yên (25,60 → 24,94 → 24,50). Mô
hình càng giỏi thì phần nó không được đọc càng thành nút thắt — đúng chiều phải thấy
nếu việc cắt có giá thật. Ba phép đo dùng chung 93 bài bị cắt ấy, nên chúng không độc
lập với nhau; thứ đáng tin ở đây là **cả hướng lẫn độ lớn đều tăng đơn điệu** theo sức
mạnh của mô hình, và đến `train_20k` thì khoảng tin cậy đã rời khỏi 0.

**Trả lời câu hỏi 2, trên mô hình tốt nhất hiện có:** cắt bài ở 1.024 token lấy đi
khoảng **4 điểm ROUGE-1 [1,1; 6,8] ở 9,3% số bài** — tính ra toàn tập là khoảng 0,37
điểm. Nhỏ so với 5,95 điểm mà ViT5 hơn Lead-3, nhưng không còn là nhiễu, và nó sẽ lớn
dần nếu mô hình còn mạnh lên.

**Một nghịch lý cần giải thích trong báo cáo:** chỉ 2,4% chữ của sapo nằm riêng ở phần
bị cắt, mà thiệt hại đo được lại tới 4 điểm. Vậy thứ mất đi chủ yếu **không** phải chữ
của sapo nằm ở đuôi bài, mà nhiều khả năng là ngữ cảnh giúp mô hình chọn ý và diễn đạt.
Đây là giả thuyết, chưa kiểm; cách kiểm rẻ nhất là cho tầng 4 lọc câu trước rồi so.

Chạy lại cho mô hình khác bằng `--system <tag>`. Thêm dữ liệu train không làm hẹp
khoảng tin cậy — vẫn là 93 bài bị cắt ấy; muốn hẹp hơn phải chấm trên nhiều bài hơn,
ví dụ gộp `tune` vào hoặc chấm trên `test` ở lần chấm cuối.

## Cấu trúc

```
data/raw/          dữ liệu tải về, không chỉnh sửa
data/processed/    đã chuẩn hoá và khử tách từ
data/splits/       file ID cố định của train/val/test
notebooks/         notebook trình bày
src/data/          text.py (3 phép biến đổi dùng chung), splits.py (nạp),
                   make_splits.py (đóng băng), inspect_vietnews.py (kiểm tra),
                   browse.py (duyệt dữ liệu trên trình duyệt)
src/models/        extractive.py (tầng 0-1), vit5.py (tầng 3, cần GPU),
                   run_baselines.py (chạy + chấm), measure_tokens.py (đo
                   độ dài cắt), selftest.py (tự kiểm tra)
src/eval/          rouge.py (đã đối chiếu Google), stats.py (bootstrap),
                   report.py (khung chấm điểm), bertscore.py, selftest.py,
                   truncation.py (câu hỏi 2, cần transformers)
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
- [x] Tuần 3b — Baseline tầng 0–1 (LexRank bản nhúng PhoBERT dời sang tuần 4, cần torch)
- [x] Tuần 4 — Fine-tune ViT5 lần đầu (đối chứng BARTpho dời sang tuần 5)
- [ ] Tuần 5 — Huấn luyện đầy đủ, khảo sát tham số sinh văn bản
  (đã có: đường cong học đủ ba điểm `train_5k`/`train_10k`/`train_20k`, câu hỏi 2 sơ
  bộ; còn: dò tham số sinh trên `tune`, đối chứng BARTpho, LexRank bản PhoBERT)
- [ ] Tuần 6 — Tầng 2 và tầng 4
- [ ] Tuần 7 — Người chấm, phân tích lỗi, demo Gradio
- [ ] Tuần 8 — Báo cáo, kiểm tra tái lập
