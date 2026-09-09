"""Đo độ dài token THẬT bằng chính tokenizer của mô hình, để chốt độ dài cắt.

    python src/models/measure_tokens.py                          # ViT5, van ban tho
    python src/models/measure_tokens.py --model vinai/bartpho-syllable
    python src/models/measure_tokens.py --model vinai/phobert-base --form segmented

Vì sao phải chạy script này trước khi viết bất cứ dòng code huấn luyện nào:

README hiện ghi tỷ lệ bài bị cắt theo **âm tiết** — 512 cắt 37,5%, 768 cắt 15,3%,
1.024 cắt 4,5%. Đó là **cận dưới**, vì SentencePiece băm một âm tiết tiếng Việt thành
nhiều subword nên số token thật luôn lớn hơn số âm tiết. Chọn `max_input_length` bằng
những con số cận dưới ấy là tự lừa mình: tỷ lệ bài bị cắt thực tế sẽ cao hơn, và phần
bị cắt chính là thứ câu hỏi nghiên cứu số 2 phải đo.

Đây cũng là một đánh đổi ngân sách GPU chứ không chỉ là một hằng số: chi phí attention
tăng theo bình phương độ dài, nên 1.024 token tốn khoảng gấp bốn 512 ở phần attention.
Trên Colab free, chênh lệch đó quyết định `train_20k` có chạy xong trong một phiên hay
không.

**Không cần GPU.** Tokenizer chạy trên CPU; đừng đốt hạn ngạch GPU của Colab cho bước
này nếu bạn chưa định huấn luyện ngay sau đó.

Dạng văn bản phải khớp với thứ mô hình thực sự đọc, nếu không số đo vô nghĩa:

  `--form raw`       (mặc định) văn bản thô, đã khử tách từ — ViT5, BARTpho-syllable
  `--form segmented` văn bản còn gạch dưới `Khởi_tố` — PhoBERT ở tầng 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import syllables  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"

# Cac nguong dang can nhac cho max_input_length. Trung voi bang trong README de so
# duoc truc tiep con so am tiet (can duoi) voi con so token that.
THRESHOLDS = (512, 768, 1024, 1536)

# Nguong che phu toi thieu de goi la "du": duoi muc nay thi phan bi cat qua lon,
# tang 4 (pipeline lai) se phai ganh, va cau hoi nghien cuu so 2 mat y nghia.
COVERAGE = 90.0


def stats(lengths):
    a = np.asarray(lengths, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def line(name, s):
    return (
        f"  {name:<22} tb {s['mean']:7.1f} | p50 {s['p50']:7.0f} | p90 {s['p90']:7.0f}"
        f" | p95 {s['p95']:7.0f} | p99 {s['p99']:7.0f} | max {s['max']:7.0f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Đo độ dài token thật của một mô hình.")
    ap.add_argument("--model", default="VietAI/vit5-base", help="tên tokenizer trên Hub")
    ap.add_argument("--split", default="train_20k", help="tập con để lấy mẫu")
    ap.add_argument("--sample", type=int, default=2000, help="số bài lấy mẫu")
    ap.add_argument("--seed", type=int, default=13, help="hạt giống lấy mẫu")
    ap.add_argument(
        "--form",
        default="raw",
        choices=["raw", "segmented"],
        help="dạng văn bản mô hình thực sự đọc",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"Nạp tokenizer {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Tat canh bao "sequence longer than model_max_length": o day CO Y do do dai
    # that, chua cat gi ca, nen canh bao do la nhieu.
    tok.model_max_length = int(1e9)

    print(f"Nạp tập {args.split} ...")
    rows = list(load_split(args.split, add_raw=True))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)
    rows = [rows[int(i)] for i in idx]
    print(f"  lấy mẫu {len(rows)} bài (seed={args.seed}).\n")

    a_col, b_col = ("article_raw", "abstract_raw") if args.form == "raw" else ("article", "abstract")
    arts = [r[a_col] for r in rows]
    abss = [r[b_col] for r in rows]

    print("Đang tokenize ...")
    art_tok = [len(x) for x in tok(arts, add_special_tokens=True, truncation=False)["input_ids"]]
    abs_tok = [len(x) for x in tok(abss, add_special_tokens=True, truncation=False)["input_ids"]]
    art_syl = [len(syllables(t)) for t in arts]
    abs_syl = [len(syllables(t)) for t in abss]

    s_art, s_abs = stats(art_tok), stats(abs_tok)
    s_art_syl, s_abs_syl = stats(art_syl), stats(abs_syl)

    print(f"\n### {args.model} — dạng {args.form}, tập {args.split}\n")
    print("Bài (đầu vào):")
    print(line("token thật", s_art))
    print(line("âm tiết (cận dưới)", s_art_syl))
    print("\nSapo (đầu ra):")
    print(line("token thật", s_abs))
    print(line("âm tiết (cận dưới)", s_abs_syl))

    # He so no: mot am tiet tieng Viet bi bam thanh bao nhieu subword. Chinh he so
    # nay giai thich vi sao bang am tiet trong README la can duoi.
    ratio = float(np.mean(np.asarray(art_tok) / np.maximum(np.asarray(art_syl), 1)))
    print(f"\nHệ số nở token/âm tiết trên bài: {ratio:.2f}")

    print("\nTỷ lệ bài BỊ CẮT theo ngưỡng đầu vào:")
    trunc = {}
    for t in THRESHOLDS:
        by_tok = 100 * float(np.mean(np.asarray(art_tok) > t))
        by_syl = 100 * float(np.mean(np.asarray(art_syl) > t))
        trunc[t] = {"token": by_tok, "syllable": by_syl}
        print(f"  {t:>5}: {by_tok:5.1f}%  (theo âm tiết chỉ {by_syl:4.1f}% — cận dưới)")

    ok = [t for t in THRESHOLDS if 100 - trunc[t]["token"] >= COVERAGE]
    suggest_in = min(ok) if ok else max(THRESHOLDS)
    # Lam tron len boi cua 8 cho thuan voi nhan Tensor Core khi chay fp16.
    suggest_out = int(np.ceil(s_abs["p99"] / 8) * 8)

    print(
        f"\nGợi ý (quy tắc: ngưỡng NHỎ NHẤT che được ≥ {COVERAGE:.0f}% số bài):\n"
        f"  max_input_length  = {suggest_in}   -> cắt {trunc[suggest_in]['token']:.1f}% số bài\n"
        f"  max_target_length = {suggest_out}   -> p99 của sapo là {s_abs['p99']:.0f} token"
    )
    print(
        "\nĐây là GỢI Ý, không phải quyết định. Nếu ngân sách GPU không cho phép, hạ\n"
        "xuống ngưỡng thấp hơn và GHI LẠI tỷ lệ bài bị cắt — chính con số đó là dữ\n"
        "liệu cho câu hỏi nghiên cứu số 2 và cho tầng 4."
    )

    out = {
        "model": args.model,
        "form": args.form,
        "split": args.split,
        "sample": len(rows),
        "seed": args.seed,
        "article": {"token": s_art, "syllable": s_art_syl},
        "abstract": {"token": s_abs, "syllable": s_abs_syl},
        "expansion_token_per_syllable": ratio,
        "truncation_percent": {str(k): v for k, v in trunc.items()},
        "suggested": {
            "max_input_length": suggest_in,
            "max_target_length": suggest_out,
            "coverage_rule_percent": COVERAGE,
        },
    }
    slug = args.model.replace("/", "_")
    path = RESULTS / "tables" / f"token_lengths_{slug}_{args.form}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi: {path}")


if __name__ == "__main__":
    main()
