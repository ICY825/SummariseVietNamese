"""ROUGE cho tiếng Việt, tính trên dạng văn bản chuẩn của dự án.

Mọi hàm công khai ở đây nhận **chuỗi thô chưa xử lý** và tự đưa về dạng chuẩn bằng
`data.text.for_scoring()`. Cố ý thiết kế như vậy: đầu ra tầng 0-2 là văn bản đã tách
từ còn đầu ra tầng 3 là văn bản thô, và chấm lẫn lộn hai dạng là so sánh vô nghĩa
(`học_sinh` đếm một đơn vị, `học sinh` đếm hai — riêng việc đổi dạng đã dịch ROUGE-2
của Lead-3 từ 10,19 lên 14,21). Bằng cách không cho phép truyền vào danh sách token
đã tách sẵn, module này khiến việc làm đúng trở thành việc duy nhất làm được.

Không chuẩn hoá gốc từ và không loại từ dừng — tiếng Việt không biến hình, và ROUGE
gốc cũng không loại từ dừng theo mặc định.

`rougeL` (LCS trên toàn chuỗi) là biến thể chính của dự án, không phải `rougeLsum`.
Lý do: `rougeLsum` phải cắt câu trên **văn bản thô**, mà ranh giới câu tiếng Việt ở
dạng thô không đáng tin — "TP." trong "Công an TP. Hưng Yên" trông y hệt một dấu chấm
kết câu. Sai số đó lại lệch một chiều vì chỉ đầu ra abstractive mới cần cắt câu ở
dạng thô. `rougeLsum` vẫn được tính kèm để đối chiếu với các bài báo khác, có bộ chặn
viết tắt ở `sentences_raw()`, nhưng đừng dùng nó làm căn cứ kết luận chính.
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring, syllables  # noqa: E402

# Viet tat pho bien trong tin tuc tieng Viet: dau cham sau chung KHONG ket cau.
_ABBREV = {
    "tp", "tt", "tx", "q", "p", "h", "x", "ts", "ths", "gs", "pgs", "bs", "ks",
    "đh", "cđ", "kcn", "nxb", "vs", "st", "mr", "mrs", "no", "tr", "gđ", "cty",
}
_RAW_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences_raw(text):
    """Cắt câu trên văn bản THÔ, có chặn viết tắt. Chỉ dùng cho `rougeLsum`.

    Kém tin cậy hơn `data.text.sentences()` vốn chạy trên dạng đã tách từ nơi dấu câu
    đứng riêng. Dùng ở đây vì đầu ra của mô hình sinh không có dạng tách từ.
    """
    out = []
    for part in _RAW_SENT.split(text.strip()):
        # Noi vao doan TRUOC neu doan truoc ket thuc bang viet tat, vi dau cham
        # do khong ket cau. Phai xet duoi cua out[-1], khong phai cua `part`.
        if out:
            prev = out[-1].split()[-1]
            stem = prev[:-1].lower().strip("(\"'“‘")
            if prev.endswith((".", "!", "?")) and (
                stem in _ABBREV or (len(stem) == 1 and prev[:-1].isupper())
            ):
                out[-1] = out[-1] + " " + part
                continue
        out.append(part)
    return [s.strip() for s in out if s.strip()]


def _prf(match, n_hyp, n_ref):
    if not match or not n_hyp or not n_ref:
        return 0.0
    p, r = match / n_hyp, match / n_ref
    return 2 * p * r / (p + r)


def _ngrams(toks, n):
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _lcs_positions(a, b):
    """Chỉ số các token của `a` nằm trong một LCS của `a` và `b`."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    pos, i, j = set(), n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pos.add(i - 1)
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return pos


def _rouge_n(ref, hyp, n):
    R, H = _ngrams(ref, n), _ngrams(hyp, n)
    return _prf(sum((R & H).values()), sum(H.values()), sum(R.values()))


def _rouge_l(ref, hyp):
    return _prf(len(_lcs_positions(ref, hyp)), len(hyp), len(ref))


def _rouge_lsum(ref_sents, hyp_sents):
    """Union-LCS: với mỗi câu tham chiếu, hợp các vị trí khớp từ MỌI câu giả thuyết.

    Bước trừ dần `remain_*` là để chặn đếm trùng: một token chỉ được tính số lần
    bằng số lần nó thực sự xuất hiện. Thiếu bước này thì các câu giả thuyết chồng
    lấn nhau sẽ cộng điểm nhiều lần cho cùng một từ. Bản ROUGE gốc và `rouge_score`
    của Google đều làm vậy; giữ đúng để `rougeLsum` còn so được với các bài báo khác.
    """
    n_ref = sum(len(s) for s in ref_sents)
    n_hyp = sum(len(s) for s in hyp_sents)
    if not n_ref or not n_hyp:
        return 0.0
    remain_ref = Counter(t for s in ref_sents for t in s)
    remain_hyp = Counter(t for s in hyp_sents for t in s)
    hits = 0
    for r in ref_sents:
        union = set()
        for h in hyp_sents:
            union |= _lcs_positions(r, h)
        for i in sorted(union):
            tok = r[i]
            if remain_hyp[tok] > 0 and remain_ref[tok] > 0:
                hits += 1
                remain_hyp[tok] -= 1
                remain_ref[tok] -= 1
    return _prf(hits, n_hyp, n_ref)


def score(prediction, reference):
    """Điểm F1 của một cặp. Nhận chuỗi bất kỳ dạng nào, tự đưa về dạng chuẩn."""
    p_txt, r_txt = for_scoring(prediction), for_scoring(reference)
    hyp, ref = syllables(p_txt), syllables(r_txt)
    return {
        "rouge1": _rouge_n(ref, hyp, 1) * 100,
        "rouge2": _rouge_n(ref, hyp, 2) * 100,
        "rougeL": _rouge_l(ref, hyp) * 100,
        "rougeLsum": _rouge_lsum(
            [syllables(s) for s in sentences_raw(r_txt)],
            [syllables(s) for s in sentences_raw(p_txt)],
        ) * 100,
    }


def score_all(predictions, references):
    """Điểm từng bài, giữ nguyên thứ tự. Bootstrap cần mảng này chứ không cần trung bình."""
    if len(predictions) != len(references):
        raise ValueError(
            f"Lệch số lượng: {len(predictions)} dự đoán vs {len(references)} tham chiếu."
        )
    return [score(p, r) for p, r in zip(predictions, references)]
