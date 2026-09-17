"""Chấm các bản tóm tắt ĐÃ sinh sẵn theo hướng mới: đủ ý và không sai sự thật.

    .venv/Scripts/python.exe src/eval/cham_chinh_xac.py --split val \
        baselines_val:Lead-3 phobert-sent-train_20k_val_len256_k2 \
        bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024

Mỗi tham số là tên file trong `results/predictions/` (không đuôi `.json`); thêm `:Tên` để
chỉ lấy một hệ thống trong file có nhiều hệ thống, bỏ trống thì lấy hết. Ghi
`results/tables/chinh_xac_<split>.json` với điểm TỪNG BÀI kèm `guid`, cùng khuôn với các
bảng khác nên so cặp lại được mà không phải chấm lại.

Không chạy mô hình nào, chỉ cần `.venv`. Thước đo định nghĩa ở `eval.chinh_xac`.
"""

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from eval.chinh_xac import cham  # noqa: E402
from eval.rouge import score  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
CHI_SO = ("r1_recall", "r2_recall", "do_phu_chi_tiet", "co_chi_tiet_la", "r1_f1", "so_cau", "am_tiet")


def nap_he_thong(specs):
    """[(tên, {guid: bản tóm tắt})] theo đúng thứ tự tham số."""
    out = []
    for spec in specs:
        tag, _, ten = spec.partition(":")
        ten, _, nhan = ten.partition("=")  # "tag:ten=nhan": doi ten khi hai file cung ten he thong
        d = json.loads((RESULTS / "predictions" / f"{tag}.json").read_text(encoding="utf-8"))
        guid = [str(g) for g in d["guid"]]
        cac_ten = [ten] if ten else [k for k in d if k not in ("guid", "reference", "article")]
        for t in cac_ten:
            if t not in d:
                raise SystemExit(f"{tag}.json không có hệ thống {t!r}. Có: {list(d)}")
            out.append((nhan or t, dict(zip(guid, d[t]))))
    trung = sorted({t for t, _ in out if [x for x, _ in out].count(t) > 1})
    if trung:
        raise SystemExit(f"Hai hệ thống cùng tên {trung} — đổi tên bằng tag:ten=nhan.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Chấm đủ ý và không sai sự thật.")
    ap.add_argument("tag", nargs="+")
    ap.add_argument("--split", default="val", choices=["tune", "val", "test"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--ten", default="", help="hậu tố tên file kết quả")
    ap.add_argument("--cho-phep-test", action="store_true", help="MỞ khoá `test` — chỉ ở lần chấm cuối")
    args = ap.parse_args()
    if args.split == "test" and not args.cho_phep_test:
        raise SystemExit("Từ chối chấm trên `test`: tập này dùng MỘT lần. Thêm --cho-phep-test khi chấm lần cuối.")
    ten_file = f"chinh_xac_{args.split}" + (f"_{args.ten}" if args.ten else "") + ".json"
    if args.split == "test" and (RESULTS / "tables" / ten_file).exists():
        raise SystemExit(f"{ten_file} đã có. `test` chấm MỘT lần — xoá tay nếu thật sự muốn chấm lại.")

    rows = list(load_split(args.split, add_raw=True))
    guid = [str(r["guid"]) for r in rows]
    ket_qua = []
    for ten, pr in nap_he_thong(args.tag):
        if set(pr) != set(guid):
            raise SystemExit(f"{ten}: tập bài khác data/splits/{args.split}.json.")
        per = {k: [] for k in CHI_SO}
        for r in rows:
            p = pr[str(r["guid"])]
            m = cham(p, r["abstract"], r["article"])
            m["r1_f1"] = score(p, r["abstract"])["rouge1"]
            for k in CHI_SO:
                per[k].append(m[k])
        corpus = {}
        for k in CHI_SO:
            vals = [v for v in per[k] if v is not None]
            if k == "co_chi_tiet_la":
                vals = [100 * v for v in vals]
            mean, lo, hi = bootstrap_ci(vals, n_boot=args.n_boot, seed=13)
            corpus[k] = {"mean": mean, "lo": lo, "hi": hi, "n": len(vals)}
        ket_qua.append({"name": ten, "n": len(rows), "guid": guid, "per_article": per, "corpus": corpus})
        c = corpus
        print(f"{ten:28s} R1-recall {c['r1_recall']['mean']:5.1f} | phủ chi tiết {c['do_phu_chi_tiet']['mean']:5.1f}"
              f" | có chi tiết lạ {c['co_chi_tiet_la']['mean']:4.1f}% | R1-F1 {c['r1_f1']['mean']:5.1f}"
              f" | {c['so_cau']['mean']:.1f} câu, {c['am_tiet']['mean']:.0f} âm tiết")

    dest = RESULTS / "tables" / ten_file
    dest.write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


if __name__ == "__main__":
    main()
