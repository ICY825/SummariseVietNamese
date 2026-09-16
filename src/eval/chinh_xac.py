"""Thước đo cho hướng mới: bản tóm tắt phải ĐỦ Ý và KHÔNG SAI SỰ THẬT.

ROUGE F1 của `eval.rouge` phạt bản tóm tắt dài hơn kể cả khi nó đủ ý hơn — F1 trộn
precision vào, mà precision giảm theo độ dài. Với tiêu chí "đủ ý, không cần càng ngắn
càng tốt", chỉ số phải tách hai chiều ra:

- **Đủ ý** — `rouge_recall()`: bao nhiêu chữ của sapo có trong bản tóm tắt; và
  `do_phu_chi_tiet()`: bao nhiêu **tên riêng và con số** của sapo có trong bản tóm tắt.
  Chi tiết là thứ người đọc tin tức cần nhất (ai, bao nhiêu, khi nào, ở đâu).
- **Không sai sự thật** — `chi_tiet_la()`: tên riêng hoặc con số có trong bản tóm tắt
  nhưng **không có trong bài gốc**. Đúng kiểu lỗi đã gặp ở phần phân tích lỗi: "30 triệu"
  thay cho 450.000 đồng, "quận 12" thay cho huyện Hóc Môn.

Mọi hàm nhận chuỗi thô và tự đưa về dạng chuẩn bằng `for_scoring()`, đếm âm tiết bằng
đúng `syllables()` của ROUGE, nên recall ở đây so được với F1 của `eval.rouge`.

Giới hạn phải nêu khi dùng: `chi_tiet_la()` KHÔNG bắt được lỗi đảo chủ thể ("ai làm gì")
hay lỗi bỏ từ rào đón ("bị cáo buộc"), vì các tên ấy vẫn có trong bài; và coi "2 người"
là chi tiết lạ nếu bài chỉ viết "hai người". Nhận diện tên riêng dựa vào chữ hoa nên có
nhiễu (ví dụ "Cơ" trong "Cơ quan"); so khớp không phân biệt hoa thường nên nhiễu ấy phần
lớn vô hại.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring, syllables  # noqa: E402
from eval.rouge import _ABBREV, _ngrams, sentences_raw  # noqa: E402

# Lay tron token chu-so ("48D", "8h", "5h30", "U23"), khong chi phan chu so: `syllables()`
# giu nguyen cac token do thanh mot am tiet, nen tach rieng "48" ra se khong bao gio khop
# voi bai goc — tung lam extractive bi gan co 5-10% "chi tiet la" du chep nguyen cau.
_SO = re.compile(r"[^\W_]*\d[^\W_]*(?:[.,:/]\d[^\W_]*)*")
_VIEN = "\"'“”‘’()[]{}…-–—,;:!?."


def rouge_recall(prediction, reference):
    """ROUGE-1/2 recall (%) — phần sapo được bản tóm tắt phủ, không phạt độ dài."""
    hyp, ref = syllables(for_scoring(prediction)), syllables(for_scoring(reference))
    out = {}
    for n in (1, 2):
        R, H = _ngrams(ref, n), _ngrams(hyp, n)
        tong = sum(R.values())
        out[f"r{n}"] = 100 * sum((R & H).values()) / tong if tong else 0.0
    return out


def _la_viet_tat(tok):
    goc = tok.rstrip(".").lower().strip(_VIEN)
    return tok.endswith(".") and (goc in _ABBREV or (len(goc) == 1 and goc.isalpha()))


def chi_tiet(text):
    """Tập chi tiết kiểm được của một văn bản: con số và cụm tên riêng.

    Mỗi chi tiết là một bộ âm tiết viết thường. Cụm tên riêng là chuỗi từ viết hoa liền
    nhau, không bị ngắt bởi dấu câu. Cụm nằm ở ĐẦU CÂU thì bỏ từ đầu tiên, vì chữ hoa ở
    đó có thể chỉ là chữ đầu câu ("Theo Bộ Công an" -> "bộ công"); nhờ vậy "Hôm nay" không
    thành tên riêng, còn "Vũ Thị Dung" ở đầu câu vẫn giữ được "thị dung".
    """
    t = for_scoring(text)
    out = {tuple(syllables(m.group())) for m in _SO.finditer(t)}

    cum, dau_cau = [], True
    bat_dau_o_dau_cau = False

    def dong():
        nonlocal cum
        tu = cum[1:] if bat_dau_o_dau_cau else cum
        if tu:
            out.add(tuple(syllables(" ".join(tu))))
        cum = []

    for tok in t.split():
        loi = tok.strip(_VIEN)
        hoa = bool(loi) and loi[0].isupper() and not loi[0].isdigit()
        if hoa:
            if not cum:
                bat_dau_o_dau_cau = dau_cau
            cum.append(loi)
        elif cum:
            dong()
        viet_tat = _la_viet_tat(tok)
        het_cau = tok.endswith((".", "!", "?")) and not viet_tat
        # Dau cau (tru dau cham cua viet tat) ngat cum ten rieng.
        if cum and tok.rstrip(_VIEN) != tok and not viet_tat:
            dong()
        dau_cau = het_cau
    if cum:
        dong()
    return {c for c in out if c}


def _co_trong(chuoi, toks):
    n = len(chuoi)
    return any(tuple(toks[i : i + n]) == chuoi for i in range(len(toks) - n + 1))


def _ho_tro(c, toks):
    """Chi tiết `c` có được văn bản `toks` hỗ trợ không.

    Con số phải khớp CHÍNH XÁC — đổi số là lỗi nguy hiểm nhất. Cụm tên riêng thì được ghép
    từ các đoạn liền nhau có trong văn bản (mỗi đoạn từ 2 âm tiết, đoạn cuối lẻ một âm
    tiết cũng được): "Cẩm Lệ Đà Nẵng" được bài "Cẩm Lệ, TP Đà Nẵng" hỗ trợ, vì đó là cùng
    sự thật viết khác đi, không phải chi tiết bịa.
    """
    if any(ch.isdigit() for tu in c for ch in tu):
        return _co_trong(c, toks)
    if _co_trong(c, toks):
        return True
    i, n = 0, len(c)
    while i < n:
        dai = next((L for L in range(n - i, 1, -1) if _co_trong(c[i : i + L], toks)), 0)
        if dai:
            i += dai
        elif n - i == 1 and c[i] in toks:
            i += 1
        else:
            return False
    return True


def do_phu_chi_tiet(prediction, reference):
    """% chi tiết của sapo có mặt trong bản tóm tắt; None nếu sapo không có chi tiết nào."""
    can = chi_tiet(reference)
    if not can:
        return None
    toks = syllables(for_scoring(prediction))
    return 100 * sum(_ho_tro(c, toks) for c in can) / len(can)


def _so_la(hyp, toks):
    """Con số của bản tóm tắt không có trong bài — xét KÈM NGỮ CẢNH.

    Chỉ kiểm con số có mặt thì lọt lỗi: "giá 30 triệu" thay cho 450.000 đồng vẫn qua nếu bài
    có "30 năm" ở chỗ khác. Nên một cụm số chỉ được coi là có trong bài khi nó đi cùng ít nhất
    một âm tiết bên cạnh giống bài gốc; "làm chết 3 người" vẫn qua nhờ "3 người".
    """
    la, i = [], 0
    while i < len(hyp):
        if not any(ch.isdigit() for ch in hyp[i]):
            i += 1
            continue
        j = i
        while j < len(hyp) and any(ch.isdigit() for ch in hyp[j]):
            j += 1
        cum = tuple(hyp[i:j])
        truoc = (hyp[i - 1],) + cum if i > 0 else None
        sau = cum + (hyp[j],) if j < len(hyp) else None
        if truoc is None and sau is None:
            ok = _co_trong(cum, toks)
        else:
            ok = any(x is not None and _co_trong(x, toks) for x in (truoc, sau))
        if not ok:
            la.append(cum)
        i = j
    return la


def chi_tiet_la(prediction, article):
    """Chi tiết của bản tóm tắt KHÔNG có trong bài gốc, sắp xếp để kết quả tất định."""
    toks = syllables(for_scoring(article))
    ten = [c for c in chi_tiet(prediction)
           if not any(ch.isdigit() for tu in c for ch in tu) and not _ho_tro(c, toks)]
    so = _so_la(syllables(for_scoring(prediction)), toks)
    return sorted(set(ten) | set(so))


def cham(prediction, reference, article):
    """Mọi chỉ số của hướng mới cho một bài."""
    rc = rouge_recall(prediction, reference)
    la = chi_tiet_la(prediction, article)
    return {
        "r1_recall": rc["r1"],
        "r2_recall": rc["r2"],
        "do_phu_chi_tiet": do_phu_chi_tiet(prediction, reference),
        "so_chi_tiet_la": len(la),
        "co_chi_tiet_la": int(bool(la)),
        "so_cau": len(sentences_raw(for_scoring(prediction))),
        "am_tiet": len(syllables(for_scoring(prediction))),
    }
