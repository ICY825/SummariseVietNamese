"""Bước 1 của kế hoạch: xác minh bộ dữ liệu VietNews (VNDS) dùng được.

Script trả lời đúng 5 câu hỏi, mỗi câu là một cái bẫy đã nêu trong kế hoạch:
  1. Schema và kích thước các split có đúng như mô tả không?
  2. Tiếng Việt có bị lỗi mã hoá / chưa chuẩn hoá NFC không?
  3. Có bài trùng lặp trong cùng một split không?
  4. Có rò rỉ giữa các split không?  <-- nguy hiểm nhất
  5. Sapo (abstract) có phải là câu đầu bài chép lại không?

Văn bản của bộ này ĐÃ ĐƯỢC TÁCH TỪ sẵn bằng VnCoreNLP (`Khởi_tố`, `ma_tuý`) và
dấu câu cũng đã tách rời thành token độc lập. Hai hệ quả chi phối toàn bộ script:

  - Đếm độ dài phải phân biệt *token đã ghép* (PhoBERT ăn dạng này) với *âm tiết*
    (ViT5 và BARTpho-syllable ăn văn bản thô, tính theo âm tiết). Regex `\w+` gộp
    `Khởi_tố` thành một đơn vị nên chỉ đo được vế thứ nhất.
  - Ranh giới câu là dấu chấm đứng riêng (" . "), không phải mọi dấu chấm. Cắt câu
    bằng `(?<=[.!?])\s+` sẽ đứt ngay ở chữ viết tắt như "TP." và cho câu cụt.

Các phép biến đổi văn bản lấy từ `src/data/text.py` — cả dự án dùng chung một
cách cắt câu và một cách đếm độ dài.

Chạy:  .venv/Scripts/python.exe src/data/inspect_vietnews.py
"""

import sys

# Console Windows mac dinh cp1252 -> khong in duoc tieng Viet. Ep UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import sentences, syllables, tokens  # noqa: E402

DATASET = "nam194/vietnews"
SAMPLE = 4000  # số bài lấy mẫu cho các kiểm tra tốn thời gian
SEED = 13  # cố định để mẫu lặp lại được giữa các lần chạy

def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def lead(text, n=1):
    return " ".join(sentences(text)[:n])


def coverage(reference, candidate):
    """Tỷ lệ token của `reference` được `candidate` bao phủ (unigram recall)."""
    ref = Counter(tokens(reference))
    if not ref:
        return 0.0
    cand = Counter(tokens(candidate))
    hit = sum(min(n, cand[w]) for w, n in ref.items())
    return hit / sum(ref.values())


def percentiles(xs):
    """Trả về (trung bình, p50, p90, p95, max). Kẹp chỉ số để q=1.0 không tràn."""
    xs = sorted(xs)
    p = lambda q: xs[min(int(len(xs) * q), len(xs) - 1)]
    return statistics.mean(xs), p(0.5), p(0.9), p(0.95), xs[-1]


def main():
    rule("1. TẢI VÀ KIỂM TRA SCHEMA")
    ds = load_dataset(DATASET)
    print(ds)

    cols = ds["train"].column_names
    print(f"\nCác cột: {cols}")
    for need in ("article", "abstract"):
        status = "OK" if need in cols else "THIẾU"
        print(f"  - {need:10s} {status}")

    rule("2. XEM BẰNG MẮT 3 BÀI ĐẦU")
    for i in range(3):
        row = ds["train"][i]
        print(f"\n--- Bài {i} " + "-" * 55)
        print(f"TITLE   : {row.get('title', '')[:110]}")
        print(f"ABSTRACT: {row['abstract'][:220]}")
        print(f"ARTICLE : {row['article'][:220]}")

    # Mau NGAU NHIEN co dinh seed: bo du lieu co thu tu (theo nguon/chuyen muc) nen
    # lay 4000 bai dau se cho so lieu lech so voi toan bo split.
    sample = ds["train"].shuffle(seed=SEED).select(range(min(SAMPLE, len(ds["train"]))))

    rule("3. MÃ HOÁ VÀ CHUẨN HOÁ UNICODE")
    not_nfc = sum(
        1 for r in sample if not unicodedata.is_normalized("NFC", r["article"])
    )
    has_replacement = sum(1 for r in sample if "\ufffd" in r["article"])
    print(f"Mẫu kiểm tra          : {len(sample)} bài (ngẫu nhiên, seed={SEED})")
    print(f"Chưa chuẩn NFC        : {not_nfc}  ({not_nfc / len(sample):.1%})")
    print(f"Có ký tự lỗi U+FFFD   : {has_replacement}")
    if not_nfc:
        print("  -> BẮT BUỘC chuẩn hoá NFC ở bước tiền xử lý.")

    rule("4. TRÙNG LẶP VÀ RÒ RỈ GIỮA CÁC SPLIT")

    def hashes(split):
        return [
            hashlib.md5(
                unicodedata.normalize("NFC", r["article"]).strip().encode()
            ).hexdigest()
            for r in ds[split]
        ]

    splits = ("train", "validation", "test")
    h = {s: hashes(s) for s in splits}
    for s in splits:
        vals = h[s]
        dup = len(vals) - len(set(vals))
        print(f"{s:11s}: {len(vals):>7,} bài, trùng nội bộ {dup:>6,} ({dup / len(vals):.2%})")

    print()
    # Kiem tra DU CA BA CAP, khong chi train vs phan con lai.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leak = len(set(h[a]) & set(h[b]))
        pct = leak / len(h[b])  # theo SO BAI cua split b, khong theo so hash duy nhat
        flag = "  <-- RÒ RỈ, PHẢI XỬ LÝ" if pct > 0.005 else ""
        print(f"{a} ∩ {b:11s}: {leak:>6,} bài ({pct:.2%} của {b}){flag}")

    rule("5. SAPO CÓ PHẢI LÀ CÂU ĐẦU BÀI CHÉP LẠI?")
    for n in (1, 3):
        covs = [coverage(r["abstract"], lead(r["article"], n)) for r in sample]
        near_copy = sum(1 for c in covs if c >= 0.9)
        print(
            f"Lead-{n}: bao phủ tb {statistics.mean(covs):.3f} | "
            f"trung vị {statistics.median(covs):.3f} | "
            f"gần như chép lại (>=0.9) {near_copy} bài ({near_copy / len(covs):.1%})"
        )
        if n == 1 and near_copy / len(covs) > 0.15:
            print("  -> Lead-1 sẽ mạnh giả tạo. Phải nêu rõ tỷ lệ này trong báo cáo.")

    n_sents = [len(sentences(r["article"])) for r in sample]
    mean, p50, p90, p95, mx = percentiles(n_sents)
    print(
        f"\nSố câu mỗi bài: tb {mean:.1f} | p50 {p50} | p90 {p90} | max {mx} | "
        f"bài chỉ có 1 câu: {sum(1 for k in n_sents if k == 1)}"
    )

    rule("6. THỐNG KÊ ĐỘ DÀI")

    def show(name, xs):
        mean, p50, p90, p95, mx = percentiles(xs)
        print(f"{name:26s} tb {mean:7.1f} | p50 {p50:5d} | p90 {p90:5d} | p95 {p95:5d} | max {mx:6d}")

    art_tok = [len(tokens(r["article"])) for r in sample]
    art_syl = [len(syllables(r["article"])) for r in sample]
    abs_tok = [len(tokens(r["abstract"])) for r in sample]
    abs_syl = [len(syllables(r["abstract"])) for r in sample]

    print("Token đã tách từ (PhoBERT đọc dạng này):")
    show("  Bài gốc", art_tok)
    show("  Sapo", abs_tok)
    print("\nÂm tiết thật (ViT5 / BARTpho-syllable đọc dạng này):")
    show("  Bài gốc", art_syl)
    show("  Sapo", abs_syl)

    ratio = statistics.mean(a / b for a, b in zip(abs_tok, art_tok) if b)
    print(f"\nTỷ lệ nén trung bình (theo token): {ratio:.3f}")

    rule("KẾT LUẬN")
    print("Nếu mục 4 báo rò rỉ > 0.5% thì phải tự khử trùng lặp và chia lại split")
    print("trước khi train bất cứ mô hình nào.")
    print("Cột 'âm tiết' ở mục 6 mới là con số dùng để chọn max_input_length cho ViT5.")


if __name__ == "__main__":
    main()
