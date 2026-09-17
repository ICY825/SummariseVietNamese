"""So cặp theo bài trên một bảng `chinh_xac_<split>*.json` đã chấm — không chấm lại gì.

    .venv/Scripts/python.exe src/eval/so_cap_chinh_xac.py chinh_xac_test chon-cau Lead-3 phobert-sent \
        bartpho-syllable-train_20k

Hệ thống đầu tiên được so với từng hệ thống sau, trên recall, phủ chi tiết và F1 (bootstrap ghép
cặp, 10.000 lần, hạt giống mặc định của `eval.stats`). Phủ chi tiết chỉ tính trên bài có sapo chứa
chi tiết. Ghi `results/tables/<bảng>_so_cap.json`. Tách khỏi `cham_chinh_xac.py` vì bảng `test`
chỉ được chấm một lần, còn phép so thì phải chạy lại được.
"""

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.stats import paired_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
CHI_SO = ("r1_recall", "do_phu_chi_tiet", "r1_f1")


def main():
    ap = argparse.ArgumentParser(description="So cặp theo bài trên bảng chinh_xac đã chấm.")
    ap.add_argument("bang", help="tên bảng trong results/tables, không đuôi .json")
    ap.add_argument("he_thong", nargs="+", help="hệ thống chính, rồi các hệ thống để so")
    args = ap.parse_args()

    bang = {e["name"]: e for e in json.loads((RESULTS / "tables" / f"{args.bang}.json").read_text(encoding="utf-8"))}
    chinh, *khac = args.he_thong
    kq = {"bang": args.bang, "he_thong": chinh, "so_voi": {}}
    for b in khac:
        if bang[chinh]["guid"] != bang[b]["guid"]:
            raise SystemExit(f"{chinh} và {b} không chấm trên cùng thứ tự bài.")
        kq["so_voi"][b] = {}
        for m in CHI_SO:
            cap = [(x, y) for x, y in zip(bang[chinh]["per_article"][m], bang[b]["per_article"][m])
                   if x is not None and y is not None]
            r = paired_bootstrap([x for x, _ in cap], [y for _, y in cap])
            kq["so_voi"][b][m] = {"n": len(cap), **r}
            print(f"{chinh} − {b:28s} {m:16s} n={len(cap)} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                  f"p={r['p']:.4f}")
    dest = RESULTS / "tables" / f"{args.bang}_so_cap.json"
    dest.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {dest}")


if __name__ == "__main__":
    main()
