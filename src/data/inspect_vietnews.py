"""Bước 1 của kế hoạch: xác minh bộ dữ liệu VietNews (VNDS) dùng được.

Script trả lời đúng 5 câu hỏi, mỗi câu là một cái bẫy đã nêu trong kế hoạch:
  1. Schema và kích thước các split có đúng như mô tả không?
  2. Tiếng Việt có bị lỗi mã hoá / chưa chuẩn hoá NFC không?
  3. Có bài trùng lặp trong cùng một split không?
  4. Có rò rỉ giữa train và test không?  <-- nguy hiểm nhất
  5. Sapo (abstract) có phải là câu đầu bài chép lại không?

Chạy:  .venv/Scripts/python.exe src/data/inspect_vietnews.py
"""

import io
import sys

# Console Windows mac dinh cp1252 -> khong in duoc tieng Viet. Ep UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import re
import statistics
import unicodedata
from collections import Counter

from datasets import load_dataset

DATASET = "nam194/vietnews"
SAMPLE = 4000  # số bài lấy mẫu cho các kiểm tra tốn thời gian


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def norm_words(text):
    """Tách âm tiết thô, bỏ dấu câu. Đủ dùng cho việc đo độ trùng ở bước này."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else ""


def coverage(reference, candidate):
    """Tỷ lệ âm tiết của `reference` được `candidate` bao phủ (unigram recall)."""
    ref = Counter(norm_words(reference))
    if not ref:
        return 0.0
    cand = Counter(norm_words(candidate))
    hit = sum(min(n, cand[w]) for w, n in ref.items())
    return hit / sum(ref.values())


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

    rule("3. MÃ HOÁ VÀ CHUẨN HOÁ UNICODE")
    sample = ds["train"].select(range(min(SAMPLE, len(ds["train"]))))
    not_nfc = sum(
        1 for r in sample if not unicodedata.is_normalized("NFC", r["article"])
    )
    has_replacement = sum(1 for r in sample if "�" in r["article"])
    print(f"Mẫu kiểm tra          : {len(sample)} bài")
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

    h = {s: hashes(s) for s in ("train", "validation", "test")}
    for s, vals in h.items():
        dup = len(vals) - len(set(vals))
        print(f"{s:11s}: {len(vals):>7,} bài, trùng nội bộ {dup:>6,} ({dup / len(vals):.2%})")

    train_set = set(h["train"])
    for s in ("validation", "test"):
        leak = len(train_set & set(h[s]))
        pct = leak / len(set(h[s]))
        flag = "  <-- RÒ RỈ, PHẢI XỬ LÝ" if pct > 0.005 else ""
        print(f"train ∩ {s:11s}: {leak:>6,} bài ({pct:.2%}){flag}")

    rule("5. SAPO CÓ PHẢI LÀ CÂU ĐẦU BÀI CHÉP LẠI?")
    covs = [coverage(r["abstract"], first_sentence(r["article"])) for r in sample]
    near_copy = sum(1 for c in covs if c >= 0.9)
    print(f"Độ bao phủ trung bình của Lead-1 lên sapo : {statistics.mean(covs):.3f}")
    print(f"Trung vị                                  : {statistics.median(covs):.3f}")
    print(f"Số bài sapo gần như chép câu đầu (>=0.9)  : {near_copy} ({near_copy / len(covs):.1%})")
    if near_copy / len(covs) > 0.15:
        print("  -> Lead-1 sẽ mạnh giả tạo. Phải nêu rõ tỷ lệ này trong báo cáo.")

    rule("6. THỐNG KÊ ĐỘ DÀI (theo âm tiết)")
    art = [len(norm_words(r["article"])) for r in sample]
    abs_ = [len(norm_words(r["abstract"])) for r in sample]

    def show(name, xs):
        xs = sorted(xs)
        p = lambda q: xs[int(len(xs) * q)]
        print(f"{name:9s} tb {statistics.mean(xs):7.1f} | p50 {p(.5):5d} | p90 {p(.9):5d} | p95 {p(.95):5d} | max {xs[-1]:5d}")

    show("Bài gốc", art)
    show("Sapo", abs_)
    print(f"Tỷ lệ nén trung bình: {statistics.mean(a / b for a, b in zip(abs_, art) if b):.3f}")

    rule("KẾT LUẬN")
    print("Nếu mục 4 báo rò rỉ > 0.5% thì phải tự khử trùng lặp và chia lại split")
    print("trước khi train bất cứ mô hình nào.")


if __name__ == "__main__":
    main()
