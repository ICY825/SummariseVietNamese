"""Chạy và chấm toàn bộ baseline tầng 0-1 trên một tập con đã đóng băng.

    .venv/Scripts/python.exe src/models/run_baselines.py                 # tap test
    .venv/Scripts/python.exe src/models/run_baselines.py --split val
    .venv/Scripts/python.exe src/models/run_baselines.py --limit 100     # chay thu

Sáu hệ thống, xếp theo thứ tự kỳ vọng tăng dần để đọc bảng là thấy ngay câu chuyện:

    Random-3   phần điểm ROUGE "cho không" do trùng chủ đề
    Lead-1     một câu đầu
    Lead-3     mốc thật sự phải vượt
    TextRank   xếp hạng câu trên đồ thị từ chung
    LexRank    như trên nhưng có trọng số IDF
    Oracle-3   trần của mọi phương pháp extractive dùng 3 câu

Ba điều script này cố ý KHÔNG làm:

**Không tự tính ROUGE.** Toàn bộ phần chấm giao cho `eval.report.evaluate()`, nên
baseline và ViT5 ở tuần 4 đi qua đúng một đường chấm điểm. Nếu tầng nào tự chấm, chênh
lệch giữa các tầng sẽ lẫn cả khác biệt cách chấm.

**Không tự lấy mẫu.** Bài nào được chấm là do `data.splits.load_split()` quyết định.

**Không khử tách từ.** Nạp với `add_raw=False`: hai cột `*_raw` chỉ để dành cho tầng 3,
còn ở đây chúng vừa thừa vừa tốn — mọi hàm của `eval` đều tự gọi `for_scoring()`.

Oracle là hệ thống duy nhất được đọc sapo. Nó không phải hệ thống chạy được mà là
thước đo trần: khoảng từ LexRank lên Oracle là phần còn cứu được bằng cách chọn câu
khéo hơn, khoảng từ Oracle lên 100 là phần extractive không bao giờ với tới và là lý
do tồn tại của tầng 3.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Phai dat ca stderr, khong chi stdout: thong bao chan ten he thong sai di ra bang
# SystemExit, tuc qua stderr — console Windows mac dinh cp1252 se bam nat tieng Viet
# va bien mot loi ro rang thanh mot dong ky tu vo nghia.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from eval.report import evaluate, compare, save, table  # noqa: E402
from models.extractive import lead, lexrank, lexrank_emb, oracle, random_k, textrank  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"

# `lexrank_emb` KHONG nam trong danh sach mac dinh: no can `torch` + `transformers`,
# ma `.venv` cua du an khong co hai goi do (xem requirements.txt). De no o mac dinh thi
# lenh `run_baselines.py` trong README se chet ngay tren may sach. Chay no bang:
#   --systems leadk lexrank lexrank_emb
DEFAULT_SYSTEMS = ["random", "lead1", "leadk", "textrank", "lexrank", "oracle"]
ALL_SYSTEMS = DEFAULT_SYSTEMS + ["lexrank_emb"]
BASELINE = "leadk"  # moc de so cap doi, vi day la baseline manh cua tin tuc


def label(key, k):
    """Tên hiển thị, LUÔN suy ra từ `k` chứ không viết cứng.

    Viết cứng "Lead-3" rồi để `k` là tham số thì chạy `--k 5` sẽ cho một bảng ghi
    "Lead-3" trong khi số liệu là của Lead-5 — sai lệch không thể phát hiện lại từ
    file kết quả. Lead-1 cố ý không phụ thuộc `k`: nó là mốc một câu, không phải
    biến thể của Lead-k.
    """
    return {
        "random": f"Random-{k}",
        "lead1": "Lead-1",
        "leadk": f"Lead-{k}",
        "textrank": "TextRank",
        "lexrank": "LexRank",
        "lexrank_emb": "LexRank-PhoBERT",
        "oracle": f"Oracle-{k}",
    }[key]


def build(key, rows, k):
    """Sinh bản tóm tắt cho một hệ thống, giữ đúng thứ tự bài của tập con."""
    if key == "lead1":
        return [lead(r["article"], 1) for r in rows]
    if key == "leadk":
        return [lead(r["article"], k) for r in rows]
    if key == "random":
        # guid lam khoa: moi bai co bo cau ngau nhien co dinh, khong phu thuoc
        # thu tu duyet hay viec dang chay tren tap con nao.
        return [random_k(r["article"], k, key=str(r["guid"])) for r in rows]
    if key == "textrank":
        return [textrank(r["article"], k) for r in rows]
    if key == "lexrank":
        return [lexrank(r["article"], k) for r in rows]
    if key == "lexrank_emb":
        return [lexrank_emb(r["article"], k) for r in rows]
    if key == "oracle":
        return [oracle(r["article"], r["abstract"], k) for r in rows]
    raise ValueError(f"Không biết hệ thống {key!r}.")


def main():
    ap = argparse.ArgumentParser(description="Baseline extractive tầng 0-1.")
    ap.add_argument("--split", default="test", help="tên tập con đã đóng băng")
    ap.add_argument("--k", type=int, default=3, help="số câu mỗi bản tóm tắt")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N bài đầu (chạy thử)")
    ap.add_argument(
        "--systems", nargs="+", default=DEFAULT_SYSTEMS, metavar="TÊN",
        help=f"hệ thống cần chạy, chọn trong {ALL_SYSTEMS}; mặc định là sáu hệ thống "
             "không cần torch",
    )
    ap.add_argument("--n-boot", type=int, default=10_000, help="số lần lấy lại mẫu")
    args = ap.parse_args()

    la = [s for s in args.systems if s not in ALL_SYSTEMS]
    if la:
        raise SystemExit(f"Không biết hệ thống {la}. Chọn trong: {ALL_SYSTEMS}")

    print(f"Nạp tập {args.split} ...")
    ds = load_split(args.split, add_raw=False)
    rows = list(ds)
    if args.limit:
        rows = rows[: args.limit]
        print(f"  CHẠY THỬ trên {len(rows)} bài đầu — số liệu KHÔNG dùng để báo cáo.")
    refs = [r["abstract"] for r in rows]
    arts = [r["article"] for r in rows]
    guids = [str(r["guid"]) for r in rows]
    print(f"  {len(rows)} bài.\n")

    results, preds = [], {}
    for key in args.systems:
        name = label(key, args.k)
        t0 = time.time()
        p = build(key, rows, args.k)
        gen = time.time() - t0

        t0 = time.time()
        r = evaluate(name, p, refs, arts, guids=guids, n_boot=args.n_boot)
        r["seconds"] = {"generate": gen, "evaluate": time.time() - t0}
        results.append(r)
        preds[name] = p
        c = r["corpus"]
        print(
            f"  {name:<9} R1 {c['rouge1']['mean']:5.2f}  R2 {c['rouge2']['mean']:5.2f}  "
            f"RL {c['rougeL']['mean']:5.2f}  ({gen:.0f}s sinh + {r['seconds']['evaluate']:.0f}s chấm)"
        )

    md = table(results)
    print("\n" + md)

    base_name = label(BASELINE, args.k)
    if BASELINE not in args.systems:
        # Khong tu nap moc tu file cu: bang Lead-3 cu do MOT LAN CHAY KHAC ghi ra, va
        # ghep cap voi no chi hop le khi hai ben cham tren dung cung nhung bai do.
        print(f"\nBỏ qua so cặp: {base_name} không nằm trong --systems lần này.")
    else:
        print(f"\nSo cặp đôi với {base_name} (bootstrap ghép cặp, seed=13):")
        base = next(r for r in results if r["name"] == base_name)
        for r in results:
            if r["name"] != base_name:
                print("  " + compare(r, base, "rouge1", n_boot=args.n_boot))

    # k phai nam trong ten file khi khac mac dinh, neu khong mot lan chay --k 5 se
    # ghi de im lang len bang ket qua k=3 dang duoc trich trong README.
    tag = args.split
    if args.k != 3:
        tag += f"_k{args.k}"
    # Danh sach he thong khac mac dinh thi tag phai khac, neu khong mot lan chay
    # `--systems lexrank_emb` se ghi de bang sau he thong dang duoc trich trong README.
    if list(args.systems) != DEFAULT_SYSTEMS:
        tag += "_" + "-".join(args.systems)
    if args.limit:
        tag += f"_thu{args.limit}"
    path = save(results, f"baselines_{tag}.json")
    md_path = RESULTS / "tables" / f"baselines_{tag}.md"
    md_path.write_text(md + "\n", encoding="utf-8")

    pred_dir = RESULTS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"baselines_{tag}.json"
    pred_path.write_text(
        json.dumps(
            {"guid": [r["guid"] for r in rows], "reference": refs, **preds},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã ghi:\n  {path}\n  {md_path}\n  {pred_path}")


if __name__ == "__main__":
    main()
