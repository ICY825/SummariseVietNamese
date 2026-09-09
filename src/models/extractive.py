"""Tầng 0 và tầng 1: các baseline extractive, chọn câu có sẵn trong bài.

Tầng 0 không học gì (Lead-1, Lead-3, Random-3, Oracle); tầng 1 không giám sát
(TextRank, LexRank). Chúng là mốc để trả lời câu hỏi nghiên cứu số 1 — nếu ViT5
fine-tune không vượt được Lead-3 thì cả hướng abstractive mất lý do tồn tại.

Năm quy ước bắt buộc, và lý do:

**Cắt câu bằng `data.text.sentences()`, không tự viết lại.** Mọi baseline ở đây xây
trên cùng một ranh giới câu. Nếu mỗi baseline cắt một kiểu thì chênh lệch điểm giữa
chúng lẫn cả sai khác tiền xử lý, và phép so sánh có kiểm soát mất ý nghĩa.

**Trả về văn bản ĐÃ TÁCH TỪ, không tự khử tách từ.** Đầu ra tầng 0-2 vốn ở dạng tách
từ; việc đưa về dạng chuẩn là của `eval.report` qua `for_scoring()`. Nếu tầng này tự
khử tách từ, văn bản sẽ đi qua `detokenize()` hai lần và không tầng nào còn kiểm soát
được dạng văn bản lúc chấm.

**Giữ nguyên thứ tự câu trong bài.** Chọn câu 5 rồi câu 2 thì vẫn xuất ra theo thứ tự
2, 5. Bản tóm tắt đọc xuôi hơn, và ROUGE-L (LCS toàn chuỗi) thưởng cho việc giữ đúng
trật tự của tham chiếu — tham chiếu thì viết theo mạch bài.

**Đo tương đồng bằng `tokens()`, không phải `syllables()`.** Bộ dữ liệu đã tách từ sẵn
nên `Khởi_tố` là một đơn vị mang nghĩa; tách nó thành "khởi" và "tố" chỉ làm nhiễu đồ
thị tương đồng. Đây là lựa chọn của riêng tầng 1 và không ảnh hưởng khâu chấm điểm —
`eval.rouge` luôn chấm trên âm tiết của dạng thô, bất kể tầng này làm gì bên trong.

**Tầng 1 khử câu trùng nội dung, tầng 0 thì không.** 179 trong 2.000 bài của tập test
chứa sẵn câu lặp lại, mà hai bản sao thì có điểm trung tâm bằng hệt nhau nên bộ xếp
hạng vơ cả hai. Lead-k và Random-k giữ nguyên vì định nghĩa của chúng là "k câu đầu"
và "k câu rút ngẫu nhiên" — sửa đi thì chúng không còn là mốc ngây thơ nữa. Chi tiết
và số đo tác động nằm ở `_pick()`.

Không loại từ dừng: dự án không có sẵn danh sách từ dừng tiếng Việt nào đã kiểm chứng,
và một danh sách tự chế sẽ là một biến không kiểm soát nằm giữa tầng 1 và mọi tầng
khác. `eval.rouge` cũng không loại từ dừng.
"""

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring, sentences, syllables, tokens  # noqa: E402

SEED = 13  # trung voi seed cua make_splits.py va bootstrap
K = 3      # so cau mac dinh: bang Lead-3 de so sanh co kiem soat


# --------------------------------------------------------------------------
# Tien xu ly dung chung
# --------------------------------------------------------------------------

def join(sents, idx):
    """Ghép các câu đã chọn theo ĐÚNG thứ tự trong bài."""
    return " ".join(sents[i] for i in sorted(idx))


def _sent_syllables(sents):
    """Âm tiết của từng câu, đo trên dạng chuẩn — khớp với `eval.rouge`.

    Chuẩn hoá theo từng câu rồi mới nối cho ra đúng kết quả như chuẩn hoá cả bản tóm
    tắt đã nối, vì `for_scoring()` chỉ làm các phép biến đổi cục bộ (NFC, dán dấu câu)
    không bắc qua ranh giới câu. Nhờ vậy `oracle()` tối ưu đúng đại lượng mà
    `eval.rouge` sẽ chấm, chứ không phải một xấp xỉ của nó.
    """
    return [syllables(for_scoring(s)) for s in sents]


# --------------------------------------------------------------------------
# Tang 0: khong hoc
# --------------------------------------------------------------------------

def lead(article, k=K):
    """Lead-k: k câu đầu bài.

    Baseline mạnh đến mức khó chịu với tin tức: nhà báo viết theo hình tháp ngược nên
    câu đầu đã gần như là bản tóm tắt. Bất cứ hệ thống nào không vượt được Lead-3 thì
    coi như chưa học được gì.
    """
    sents = sentences(article)
    return join(sents, range(min(k, len(sents))))


def random_k(article, k=K, key="", seed=SEED):
    """Random-k: k câu ngẫu nhiên, dùng để đo phần điểm ROUGE "cho không".

    ROUGE thưởng cả những từ trùng nhau thuần tuý do cùng chủ đề, nên một bản chọn
    bừa vẫn được điểm khác 0. Không có mốc này thì không biết Lead-3 hơn ngẫu nhiên
    bao nhiêu.

    `key` (thường là guid của bài) đưa vào hạt giống để mỗi bài có một bộ câu ngẫu
    nhiên cố định: chạy lại, đổi thứ tự bài, hay chấm trên tập con đều ra đúng kết quả
    cũ. Dùng một `random.Random(seed)` chung cho cả vòng lặp thì kết quả phụ thuộc thứ
    tự duyệt và không tái lập được.
    """
    sents = sentences(article)
    n = min(k, len(sents))
    rng = random.Random(f"{seed}:{key}")
    return join(sents, rng.sample(range(len(sents)), n))


def _f1(match, n_hyp, n_ref):
    if not match or not n_hyp or not n_ref:
        return 0.0
    p, r = match / n_hyp, match / n_ref
    return 2 * p * r / (p + r)


def _ngram_f1(ref, hyp, n):
    R = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    H = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
    return _f1(sum((R & H).values()), sum(H.values()), sum(R.values()))


def oracle_indices(article, reference, k=K):
    """Chọn tham lam bộ câu tối đa hoá ROUGE-1 + ROUGE-2 so với sapo thật.

    Đây là **trần** của mọi phương pháp extractive dùng cùng ngân sách k câu, không
    phải một hệ thống chạy được: nó đọc trước đáp án. Giá trị của nó là tách đôi
    khoảng cách — phần từ tầng 1 lên oracle là do chọn câu còn kém, phần từ oracle lên
    100 là thứ mà extractive **không bao giờ** với tới và chỉ abstractive mới lấy được.
    Không có số này thì không biết nên đổ công vào đâu.

    Tham lam chứ không vét cạn: vét cạn C(17,3) mỗi bài là khả thi nhưng tham lam là
    cách làm chuẩn trong tài liệu (Nallapati 2017, Liu & Lapata 2019) nên số liệu còn
    so được với các bài báo khác. Mục tiêu ROUGE-1+ROUGE-2 cũng theo chuẩn đó.

    Dừng sớm khi thêm câu không cải thiện, nên có bài ra ít hơn k câu — đúng ý: nhồi
    thêm câu vô ích chỉ hạ độ chính xác và hạ luôn điểm F1.
    """
    sents = sentences(article)
    if not sents:
        return []
    ref = syllables(for_scoring(reference))
    per_sent = _sent_syllables(sents)

    chosen, best = [], 0.0
    for _ in range(min(k, len(sents))):
        gain = None
        for i in range(len(sents)):
            if i in chosen:
                continue
            cand = sorted(chosen + [i])
            hyp = [t for j in cand for t in per_sent[j]]
            s = _ngram_f1(ref, hyp, 1) + _ngram_f1(ref, hyp, 2)
            if gain is None or s > gain[0]:
                gain = (s, i)
        if gain is None or gain[0] <= best:
            break
        best, picked = gain
        chosen.append(picked)
    return sorted(chosen)


def oracle(article, reference, k=K):
    """Bản tóm tắt của `oracle_indices()`."""
    return join(sentences(article), oracle_indices(article, reference, k))


# --------------------------------------------------------------------------
# Tang 1: khong giam sat, xep hang cau tren do thi
# --------------------------------------------------------------------------

def pagerank(sim, damping=0.85, iters=100, tol=1e-8):
    """PageRank bằng lặp luỹ thừa trên ma trận tương đồng đối xứng.

    Dùng chung cho TextRank và LexRank: hai phương pháp chỉ khác nhau ở cách dựng
    `sim`, còn phần xếp hạng thì giống hệt. Tách ra một chỗ để khi so sánh chúng
    trong báo cáo, khác biệt chắc chắn nằm ở phép đo tương đồng chứ không phải ở
    chi tiết cài đặt bộ xếp hạng.

    Hàng toàn 0 (câu không giống câu nào — thường là câu một chữ) được rải đều thay
    vì để rò rỉ mất khối lượng: chuẩn hoá một hàng 0 sẽ chia cho 0.
    """
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0)
    m = sim.copy().astype(float)
    np.fill_diagonal(m, 0.0)
    rows = m.sum(axis=1)
    m[rows == 0] = 1.0 / n
    m = m / m.sum(axis=1, keepdims=True)

    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        nxt = (1 - damping) / n + damping * (m.T @ p)
        if np.abs(nxt - p).sum() < tol:
            return nxt
        p = nxt
    return p


def _pick(sents, scores, k):
    """Chỉ số của k câu điểm cao nhất, BỎ QUA câu trùng nội dung với câu đã chọn.

    Hoà điểm thì câu đứng trước thắng: `argsort` ổn định trên khoá âm giữ nguyên thứ
    tự bài khi điểm bằng nhau, nên kết quả tất định — quan trọng với những bài ngắn
    nơi nhiều câu cùng điểm.

    Vì sao phải khử trùng: 179 trong 2.000 bài của tập test chứa sẵn câu lặp lại
    (thường là chú thích ảnh xuất hiện hai lần). Hai bản sao của cùng một câu có điểm
    trung tâm **bằng hệt nhau**, nên bộ xếp hạng chọn cả hai và bản tóm tắt 3 câu thực
    chất chỉ còn 2 câu nội dung. Đo thử trên tập test: lỗi này xảy ra ở 2,2% bản tóm
    tắt của TextRank và 1,5% của LexRank, và sửa nó chỉ đổi ROUGE-1 thêm +0,018
    (p = 0,31) — tức **không** phải nguyên nhân khiến hai phương pháp đồ thị thua
    Lead-3. Sửa vì lý do khác: tuần 7 có khâu người chấm blind, và một bản tóm tắt lặp
    nguyên một câu thì người chấm nhận ra ngay.

    Chỉ áp dụng cho tầng 1 (xếp hạng), không áp dụng cho Lead-k và Random-k: hai
    baseline đó cố ý ngây thơ, định nghĩa của chúng là "k câu đầu" và "k câu rút ngẫu
    nhiên", sửa đi thì chúng không còn là mốc ngây thơ nữa. Oracle không cần vì thêm
    một câu trùng không làm tăng điểm nên nó tự bỏ qua.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")
    out, seen = [], set()
    for i in order:
        s = sents[int(i)]
        if s in seen:
            continue
        seen.add(s)
        out.append(int(i))
        if len(out) == k:
            break
    return sorted(out)


def textrank_indices(article, k=K):
    """TextRank (Mihalcea & Tarau 2004): tương đồng = số từ chung, chuẩn hoá theo log.

    Chia cho `log|A| + log|B|` là để câu dài không tự động thắng: câu dài chung từ với
    mọi câu khác chỉ vì nó dài. Câu một từ có log = 0 nên mẫu số có thể bằng 0 hoặc
    âm — bỏ qua cạnh đó thay vì để chia cho 0.
    """
    sents = sentences(article)
    n = len(sents)
    if n == 0:
        return []
    sets = [set(tokens(s)) for s in sents]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if not sets[i] or not sets[j]:
                continue
            denom = math.log(len(sets[i])) + math.log(len(sets[j]))
            if denom <= 0:
                continue
            v = len(sets[i] & sets[j]) / denom
            sim[i, j] = sim[j, i] = v
    return _pick(sents, pagerank(sim), min(k, n))


def _tfidf(sents):
    """Ma trận TF-IDF của các câu, đã chuẩn hoá L2. Thuần numpy, không cần sklearn.

    IDF tính trên chính các câu của bài chứ không trên toàn kho: LexRank bản một văn
    bản làm vậy, và nó đúng ý đồ — từ xuất hiện ở mọi câu của bài này thì không giúp
    phân biệt câu nào quan trọng, dù nó hiếm trong kho.

    Dạng IDF trơn `log(1 + N/df)` luôn dương, nên không từ nào bị triệt tiêu hoàn toàn
    như ở dạng `log(N/df)` khi từ có mặt ở mọi câu.
    """
    vocab = {}
    counts = []
    for s in sents:
        c = Counter(tokens(s))
        for w in c:
            vocab.setdefault(w, len(vocab))
        counts.append(c)
    n, v = len(sents), len(vocab)
    if v == 0:
        return np.zeros((n, 0))

    tf = np.zeros((n, v))
    for i, c in enumerate(counts):
        for w, f in c.items():
            tf[i, vocab[w]] = f
    df = (tf > 0).sum(axis=0)
    idf = np.log(1.0 + n / np.maximum(df, 1))
    x = tf * idf
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def lexrank_indices(article, k=K, threshold=0.1):
    """LexRank (Erkan & Radev 2004): tương đồng = cosin giữa hai vector TF-IDF.

    Khác TextRank ở chỗ có trọng số IDF — từ hiếm trong bài nặng hơn từ có ở mọi câu.
    Với tin tức tiếng Việt, khác biệt này đáng kể vì các câu đều dày từ chức năng
    chung mà dự án lại không loại từ dừng.

    `threshold` là bản LexRank cổ điển: cạnh dưới ngưỡng bị cắt hẳn để đồ thị thưa ra
    và các cụm câu thực sự gần nhau nổi lên; `threshold=None` cho bản liên tục giữ
    nguyên trọng số cosin. Ngưỡng 0,1 là giá trị trong bài báo gốc.
    """
    sents = sentences(article)
    n = len(sents)
    if n == 0:
        return []
    x = _tfidf(sents)
    sim = x @ x.T
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0] = 0.0
    if threshold is not None:
        sim = (sim >= threshold).astype(float)
        np.fill_diagonal(sim, 0.0)
    return _pick(sents, pagerank(sim), min(k, n))


def textrank(article, k=K):
    """Bản tóm tắt của `textrank_indices()`."""
    return join(sentences(article), textrank_indices(article, k))


def lexrank(article, k=K, threshold=0.1):
    """Bản tóm tắt của `lexrank_indices()`."""
    return join(sentences(article), lexrank_indices(article, k, threshold))
