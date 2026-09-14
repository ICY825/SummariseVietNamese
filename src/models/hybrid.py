"""Tầng 4: lọc câu bằng extractive TRƯỚC, rồi để abstractive viết lại.

Lý do tồn tại của tầng này nằm ở câu hỏi nghiên cứu số 2. Đo trên `val`, việc cắt bài
ở 1.024 token lấy đi khoảng **4 điểm ROUGE-1 ở 10,2% số bài** (BARTpho `train_20k`) —
không còn là nhiễu. Nhưng chỉ khoảng 2,5% chữ của sapo nằm riêng ở phần bị cắt, nên
thứ mất đi chủ yếu **không phải** chữ của sapo ở đuôi bài mà nhiều khả năng là ngữ
cảnh giúp mô hình chọn ý. Tầng 4 là phép kiểm rẻ nhất cho giả thuyết ấy: thay vì đưa
1.024 token **đầu bài**, đưa 1.024 token **chọn lọc từ toàn bài**.

Ba quyết định, và lý do:

**Chỉ lọc bài thực sự vượt ngân sách.** Với bài đã vừa cửa sổ, lọc chỉ có thể bỏ đi
thông tin chứ không thêm được gì — mô hình vốn đã đọc trọn bài. Bài vừa cửa sổ được
trả về nguyên văn, không đụng tới. Nhờ vậy chênh lệch đo được quy hết về đúng nhóm bài
bị cắt, và nhóm còn lại thành đối chứng nội tại: nếu nó cũng đổi điểm thì có lỗi cài
đặt chứ không phải hiệu ứng thật.

**Giữ nguyên thứ tự câu trong bài.** Bộ chọn xếp hạng câu, nhưng đầu ra ghép lại theo
thứ tự gốc. Mô hình sinh được pretrain trên văn xuôi bình thường; đảo thứ tự câu là
đưa chuỗi ngoài phân phối huấn luyện vào, và khi ấy không phân biệt được phần mất mát
nào do lọc sai câu, phần nào do văn bản đứt mạch.

**Hai chiến lược, vì một mình LexRank có rủi ro đã biết.** Tin tức viết theo tháp
ngược nên câu đầu quan trọng nhất — chính vì thế Lead-3 thắng cả ba phương pháp đồ
thị ở tầng 1. Một bộ lọc thuần LexRank hoàn toàn có thể vứt mất câu đầu, tức làm hỏng
đúng thứ đang có giá trị nhất. Nên `lead_lexrank` bảo đảm K câu đầu rồi mới để LexRank
lấp phần còn lại. So hai chiến lược mới tách được "chọn câu theo độ trung tâm có ích
không" khỏi "giữ được câu đầu có ích không".

Bộ lọc **không** tự đếm token: nó nhận một hàm đếm từ bên ngoài, vì ngân sách phải đo
bằng chính tokenizer của mô hình sẽ đọc văn bản đó. Đếm bằng âm tiết là cận dưới và
lệch tới 20% (hệ số nở token/âm tiết là 1,20 với BARTpho).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring, sentences  # noqa: E402
from models.extractive import _tfidf, pagerank  # noqa: E402

K = 3  # so cau dau duoc bao dam o chien luoc `lead_lexrank`; trung K cua tang 0-1
STRATEGIES = ("lexrank", "lead_lexrank")


def lexrank_scores(sents, threshold=0.1):
    """Điểm trung tâm LexRank cho TỪNG câu, không cắt lấy k câu.

    `lexrank_indices()` ở tầng 1 chỉ trả về k câu thắng; ở đây cần điểm của mọi câu vì
    ngân sách tính bằng token chứ không bằng số câu, nên không biết trước lấy mấy câu.
    Dùng chung `_tfidf()` và `pagerank()` với tầng 1 để hai nơi không trôi khỏi nhau.
    """
    if not sents:
        return np.zeros(0)
    x = _tfidf(sents)
    sim = x @ x.T
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0] = 0.0
    if threshold is not None:
        sim = (sim >= threshold).astype(float)
        np.fill_diagonal(sim, 0.0)
    return pagerank(sim)


def filter_indices(sents, count, budget, strategy="lexrank"):
    """Chỉ số câu được giữ lại, đã sắp theo thứ tự bài.

    `count` là hàm đếm token của chính mô hình sẽ đọc; `budget` là ngân sách token.
    Câu trùng nội dung bị bỏ qua — hai bản sao của cùng một câu có điểm trung tâm bằng
    hệt nhau nên bộ xếp hạng sẽ vơ cả hai, ăn mất ngân sách mà không thêm thông tin.

    Luôn trả về ít nhất một câu: bài mà câu đầu tiên đã dài hơn ngân sách vẫn phải có
    đầu vào để sinh, và việc cắt câu ấy là việc của tokenizer chứ không phải của đây.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Không biết chiến lược {strategy!r}. Chọn trong: {STRATEGIES}")
    if not sents:
        return []

    # Dem tren DANG THO, khong phai dang tach tu. Mo hinh sinh doc dang tho; dang tach
    # tu ton nhieu token hon han vi dau gach duoi bi BPE be nho. Dem nham dang se uoc
    # luong thua va bo phi mot phan ba ngan sach — da do duoc dung nhu vay tren `val`.
    tho = [for_scoring(s) for s in sents]

    chon, da_dung, da_thay = [], 0, set()

    def them(i):
        nonlocal da_dung
        s = sents[i]
        if s in da_thay:
            return False
        c = count(tho[i])
        if chon and da_dung + c > budget:
            return False
        da_thay.add(s)
        chon.append(i)
        da_dung += c
        return True

    if strategy == "lead_lexrank":
        for i in range(min(K, len(sents))):
            them(i)

    diem = lexrank_scores(sents)
    # `stable` de hoa diem thi cau dung truoc thang — cung quy uoc voi `_pick()`.
    for i in np.argsort(-diem, kind="stable"):
        i = int(i)
        if i in chon:
            continue
        them(i)
    return sorted(chon)


def filter_article(article, count, budget, strategy="lexrank"):
    """Bài đã lọc, ở DẠNG THÔ — đúng thứ `article_raw` chứa.

    Xếp hạng làm trên dạng tách từ vì ranh giới câu ở đó mới đáng tin (dấu câu là
    token đứng riêng), rồi mới khử tách từ ở bước cuối. Làm ngược lại là cắt câu trên
    văn bản thô, nơi "TP." trông y hệt dấu chấm kết câu.
    """
    sents = sentences(article)
    idx = filter_indices(sents, count, budget, strategy)
    return for_scoring(" ".join(sents[i] for i in idx))


def apply_filter(rows, count, budget, strategy="lexrank", key="article_raw"):
    """Ghi đè `article_raw` cho những bài VƯỢT ngân sách; trả về thống kê.

    Bài vừa cửa sổ không bị đụng tới. Thống kê trả về là thứ phải kiểm trước khi tin
    bảng điểm: nếu số bài bị lọc không khớp tỷ lệ cắt đã đo ở câu hỏi 2 thì ngân sách
    hoặc hàm đếm đang sai.
    """
    n_loc, giu_lai, bo_di = 0, [], []
    for r in rows:
        goc = r[key]
        if count(goc) <= budget:
            continue
        sents = sentences(r["article"])
        idx = filter_indices(sents, count, budget, strategy)
        r[key] = for_scoring(" ".join(sents[i] for i in idx))
        n_loc += 1
        giu_lai.append(len(idx))
        bo_di.append(len(sents) - len(idx))
    return {
        "strategy": strategy,
        "budget": budget,
        "n": len(rows),
        "n_loc": n_loc,
        "phan_tram_loc": round(100 * n_loc / max(len(rows), 1), 2),
        "cau_giu_tb": round(float(np.mean(giu_lai)), 2) if giu_lai else 0.0,
        "cau_bo_tb": round(float(np.mean(bo_di)), 2) if bo_di else 0.0,
    }
