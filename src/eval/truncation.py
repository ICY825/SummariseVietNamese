"""Câu hỏi nghiên cứu số 2: cắt bài ở 1.024 token làm mất bao nhiêu?

    python src/eval/truncation.py
    python src/eval/truncation.py --system vit5-base-train_10k_val_e3_lr3e-05_bs16_in1024

Cần `transformers` để đếm token bằng đúng tokenizer của mô hình (xem README, mục cài
đặt môi trường phân tích). Không cần GPU, không cần chạy lại mô hình: mọi thứ lấy từ
điểm từng bài đã lưu trong `results/tables/`.

**Vì sao không so thẳng ViT5 trên bài bị cắt với bài không bị cắt.** Bài bị cắt là
bài DÀI, mà bài dài vốn khó tóm tắt với mọi hệ thống — sapo phải chọn lọc từ nhiều
thông tin hơn. So thẳng thì lẫn hai hiệu ứng "bài dài" và "bị cắt" vào nhau.

Lead-3 là đối chứng tự nhiên: nó chỉ đọc ba câu đầu nên **không bao giờ** chạm tới
phần bị cắt, nhưng vẫn chịu đủ cái khó của bài dài. Nên thứ cần đo là khoảng cách
`ViT5 − Lead-3` trên từng bài, rồi so khoảng cách đó giữa hai nhóm (hiệu của hiệu).
Nếu cắt bài không gây hại, khoảng cách hai nhóm như nhau; khoảng cách teo lại ở nhóm
bị cắt thì phần teo đó là cái giá của việc cắt.

Kèm theo là một số đo không dính gì tới mô hình: bao nhiêu nội dung của sapo CHỈ nằm
trong phần bị cắt — tức mô hình không thể nhìn thấy dù có hoàn hảo. Đó là trần mất
mát, và là thứ tầng 4 (extractive lọc trước) được thiết kế để lấy lại.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Console Windows mac dinh khong phai UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import syllables  # noqa: E402
from eval.report import RESULTS, same_articles  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402

MAX_INPUT = 1024   # dung nguong vit5.py da cat
SEED = 13


def group_diff(a, b, n_boot=10_000, seed=SEED, alpha=0.05):
    """Chênh lệch trung bình giữa hai nhóm bài KHÁC NHAU: mean(a) − mean(b).

    Khác `stats.paired_bootstrap()`: ở đây hai nhóm là hai tập bài rời nhau (bị cắt /
    không bị cắt), không có gì để ghép cặp, nên lấy lại mẫu ĐỘC LẬP trong từng nhóm.
    Dùng hàm ghép cặp ở đây là sai — nó còn đòi hai nhóm cùng cỡ.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    by = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    diffs = bx - by
    obs = float(x.mean() - y.mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = float((diffs <= 0).mean() if obs > 0 else (diffs >= 0).mean())
    return {"diff": obs, "lo": float(lo), "hi": float(hi), "p": min(1.0, 2 * tail),
            "significant": not (lo <= 0.0 <= hi)}


def cut_point(tok, text, max_input):
    """(số token thật, vị trí ký tự nơi mô hình ngừng đọc).

    Đếm có token đặc biệt, không cắt — đúng như `vit5.py` gọi tokenizer rồi mới cắt ở
    `max_length`. Token cuối cùng được giữ là `</s>` nên phần văn bản còn đọc được là
    `max_input − 1` token đầu; vị trí ký tự lấy từ offset của chính tokenizer chứ không
    ước lượng theo số âm tiết.
    """
    enc = tok(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n <= max_input:
        return n, len(text)
    offsets = [o for o in enc["offset_mapping"] if o[1] > o[0]]   # bo token dac biet
    return n, offsets[max_input - 2][1]


def lost_reference(reference, kept, cut):
    """% âm tiết (dạng từ, không lặp) của sapo xuất hiện trong bài CHỈ ở phần bị cắt.

    Chỉ tính những âm tiết của sapo có mặt trong bài — âm tiết mới do người viết sapo
    thêm vào thì đằng nào cũng không nằm trong bài, không phải lỗi của việc cắt.
    """
    ref = {s.lower() for s in syllables(reference)}
    k = {s.lower() for s in syllables(kept)}
    c = {s.lower() for s in syllables(cut)}
    in_article = ref & (k | c)
    if not in_article:
        return 0.0
    return 100 * len(in_article - k) / len(in_article)


def load_result(path, name=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if name is None:
        return rows[0]
    hit = [r for r in rows if r["name"] == name]
    if not hit:
        raise SystemExit(f"Không có hệ thống {name} trong {path}.")
    return hit[0]


def fmt(ci):
    mean, lo, hi = ci
    return f"{mean:5.2f} ±{(hi - lo) / 2:.2f}"


def main():
    ap = argparse.ArgumentParser(description="Đo mất mát do cắt bài (câu hỏi 2).")
    ap.add_argument("--system", default="vit5-base-train_5k_val_e3_lr3e-05_bs16_in1024",
                    help="tag của bảng kết quả trong results/tables/")
    ap.add_argument("--model", default="VietAI/vit5-base", help="tokenizer đã dùng khi huấn luyện")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-input", type=int, default=MAX_INPUT)
    ap.add_argument("--metric", default="rouge1")
    args = ap.parse_args()

    from models.measure_tokens import load_tokenizer

    tables = RESULTS / "tables"
    sys_res = load_result(tables / f"{args.system}.json")
    lead = load_result(tables / f"baselines_{args.split}.json", "Lead-3")
    # Hai bang do hai lan chay khac nhau ghi ra; ghep cap tung bai chi hop le khi
    # chung cham tren dung cung tap bai, cung thu tu.
    same_articles(sys_res, lead)

    print(f"Nạp tokenizer {args.model} ...")
    tok = load_tokenizer(args.model)
    tok.model_max_length = int(1e9)   # co y do do dai that, chua cat
    print(f"Nạp {args.split} ...")
    rows = {str(r["guid"]): r for r in load_split(args.split, add_raw=True)}

    guids = sys_res["guid"]
    n_tok, cut, lost, cut_share = [], [], [], []
    for g in guids:
        art = rows[g]["article_raw"]
        n, pos = cut_point(tok, art, args.max_input)
        n_tok.append(n)
        cut.append(n > args.max_input)
        if n > args.max_input:
            lost.append(lost_reference(rows[g]["abstract_raw"], art[:pos], art[pos:]))
            cut_share.append(100 * (n - args.max_input) / n)
    cut = np.asarray(cut)
    n_cut = int(cut.sum())

    m = args.metric
    s = np.asarray(sys_res["per_article"][m])
    l3 = np.asarray(lead["per_article"][m])
    gap = s - l3

    res = {
        "system": args.system,
        "split": args.split,
        "metric": m,
        "max_input": args.max_input,
        "n": len(guids),
        "n_truncated": n_cut,
        "truncated_percent": 100 * n_cut / len(guids),
        "tokens_mean": {"truncated": float(np.mean(np.asarray(n_tok)[cut])),
                        "full": float(np.mean(np.asarray(n_tok)[~cut]))},
        "cut_share_percent_mean": float(np.mean(cut_share)),
        "reference_lost_percent": {"mean": float(np.mean(lost)),
                                   "p50": float(np.percentile(lost, 50)),
                                   "p90": float(np.percentile(lost, 90)),
                                   "zero_percent": 100 * float(np.mean(np.asarray(lost) == 0))},
        "groups": {},
        "gap_vs_lead3": {},
    }
    for label, mask in (("truncated", cut), ("full", ~cut)):
        res["groups"][label] = {
            "system": bootstrap_ci(s[mask], seed=SEED),
            "lead3": bootstrap_ci(l3[mask], seed=SEED),
            "gap": bootstrap_ci(gap[mask], seed=SEED),
        }
    res["effect_system"] = group_diff(s[cut], s[~cut])
    res["effect_lead3"] = group_diff(l3[cut], l3[~cut])
    res["effect_truncation"] = group_diff(gap[cut], gap[~cut])

    print(f"\n### Cắt bài ở {args.max_input} token — {args.system}, tập {args.split}\n")
    print(f"Bài bị cắt: {n_cut} / {len(guids)} ({res['truncated_percent']:.1f}%), "
          f"dài tb {res['tokens_mean']['truncated']:.0f} token so với "
          f"{res['tokens_mean']['full']:.0f} ở nhóm còn lại.")
    print(f"Phần bị cắt chiếm tb {res['cut_share_percent_mean']:.1f}% số token của bài.")
    rl = res["reference_lost_percent"]
    print(f"Nội dung sapo CHỈ nằm trong phần bị cắt: tb {rl['mean']:.1f}%, p50 {rl['p50']:.1f}%, "
          f"p90 {rl['p90']:.1f}%; {rl['zero_percent']:.0f}% số bài bị cắt không mất gì.\n")
    print(f"| Nhóm | n | Hệ thống | Lead-3 | Hệ thống − Lead-3 |")
    print("|---|---|---|---|---|")
    for label, name in (("full", "Không bị cắt"), ("truncated", "Bị cắt")):
        gr, k = res["groups"][label], int((cut if label == "truncated" else ~cut).sum())
        print(f"| {name} | {k} | {fmt(gr['system'])} | {fmt(gr['lead3'])} | {fmt(gr['gap'])} |")

    def say(title, r):
        v = "CÓ ý nghĩa" if r["significant"] else "KHÔNG đủ bằng chứng"
        print(f"  {title}: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f} → {v}")

    print(f"\nBị cắt trừ không bị cắt ({m}):")
    say("hệ thống        ", res["effect_system"])
    say("Lead-3 (đối chứng)", res["effect_lead3"])
    say("hiệu của hiệu   ", res["effect_truncation"])
    print("  (hiệu của hiệu < 0 nghĩa là khoảng cách hệ thống − Lead-3 teo lại ở bài bị cắt)")

    out = RESULTS / "tables" / f"truncation_{args.system}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi: {out}")


if __name__ == "__main__":
    main()
