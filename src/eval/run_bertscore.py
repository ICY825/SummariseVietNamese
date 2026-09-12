"""Chấm BERTScore cho các bản tóm tắt ĐÃ sinh sẵn — không chạy lại mô hình nào.

    ~/.venvs/torch/Scripts/python.exe src/eval/run_bertscore.py baselines_val --limit 50
    ~/.venvs/torch/Scripts/python.exe src/eval/run_bertscore.py \
        baselines_val vit5-base-train_20k_val_e3_lr3e-05_bs16_in1024

Đọc `results/predictions/<tag>.json`, chấm mọi hệ thống có trong đó, ghi
`results/tables/bertscore_<tag>.json` với điểm TỪNG BÀI kèm `guid` — cùng khuôn với
bảng ROUGE, nên `eval.report.compare()` dùng lại được mà không phải sinh lại gì.

**Vì sao tách khỏi `run_baselines.py` và `vit5.py`.** BERTScore cần `torch` và
`bert-score`, hai gói không có trong `.venv` của dự án. Gắn nó vào đường chạy chính sẽ
làm mọi lệnh trong README chết trên một máy sạch. Ở đây nó là một bước riêng, chạy trên
file dự đoán đã lưu, nên chấm lại bao nhiêu lần cũng được mà không tốn GPU.

**Vì sao ROUGE vẫn là chỉ số chính.** BERTScore đọc bằng một mô hình đa ngữ và cho
điểm theo độ gần ngữ nghĩa; nó bổ sung cho ROUGE ở câu hỏi nghiên cứu số 3 (ROUGE có
phản ánh cảm nhận người đọc không) chứ không thay thế, vì chính nó cũng là một phép
xấp xỉ chứ không phải người đọc.

CHẠY TRÊN CPU RẤT CHẬM. Hãy `--limit 50` một lần trước để ước lượng thời gian.
"""

import argparse
import json
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.bertscore import DEFAULT_MODEL, bert_score  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main():
    ap = argparse.ArgumentParser(description="Chấm BERTScore trên file dự đoán đã có.")
    ap.add_argument(
        "tag", nargs="+",
        help="một hoặc nhiều tên file trong results/predictions/ (không cần đuôi .json). "
             "Nhiều file thì các hệ thống được gộp lại và guid phải trùng khớp",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="bộ mã hoá, đọc thẳng âm tiết")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N bài đầu (ước lượng thời gian)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--baseline", default="Lead-3", help="hệ thống làm mốc khi so cặp")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    # Gop nhieu file: ViT5 nam o file rieng con Lead-3 nam trong `baselines_<split>`,
    # nen muon so hai he thong do thi phai doc ca hai. Ghep chi hop le khi hai ben cham
    # tren DUNG cung nhung bai do, cung thu tu — cung chot chan ma `report.same_articles()`
    # dung cho ROUGE, khong duoc bo qua chi vi o day tien tay.
    data, guids, refs, systems = {}, None, None, []
    for tag in args.tag:
        path = RESULTS / "predictions" / f"{tag}.json"
        if not path.exists():
            có = sorted(p.stem for p in (RESULTS / "predictions").glob("*.json"))
            raise SystemExit(f"Không có {path}.\nHiện có: {có}")
        d = json.loads(path.read_text(encoding="utf-8"))
        g = [str(x) for x in d["guid"]]
        if guids is None:
            guids, refs = g, d["reference"]
        elif g != guids:
            lệch = sum(1 for a, b in zip(g, guids) if a != b) + abs(len(g) - len(guids))
            raise SystemExit(
                f"{tag} chấm trên tập bài KHÁC với {args.tag[0]} ({lệch} vị trí lệch guid) "
                "— không được ghép. Chấm lại cả hai trên cùng tập đã đóng băng."
            )
        for k, v in d.items():
            if k in ("guid", "reference"):
                continue
            if k in data:
                raise SystemExit(f"Hệ thống {k!r} xuất hiện ở hai file — tên phải duy nhất.")
            data[k] = v
            systems.append(k)
    path = RESULTS / "predictions" / f"{args.tag[0]}.json"
    if args.limit:
        guids, refs = guids[: args.limit], refs[: args.limit]
        print(f"CHẠY THỬ {args.limit} bài — số liệu KHÔNG dùng để báo cáo.")
    print(f"{len(systems)} hệ thống, {len(refs)} bài, bộ mã hoá {args.model}\n")

    out = []
    for name in systems:
        preds = data[name][: len(refs)]
        t0 = time.time()
        per = bert_score(preds, refs, model=args.model, batch_size=args.batch_size)
        mean, lo, hi = bootstrap_ci(per, n_boot=args.n_boot)
        out.append({
            "name": name,
            "n": len(per),
            "guid": guids,
            "per_article": {"bertscore": per},
            "corpus": {"bertscore": {"mean": mean, "lo": lo, "hi": hi}},
        })
        print(f"  {name:<22} BERTScore {mean:5.2f} [{lo:5.2f}, {hi:5.2f}]  ({time.time() - t0:.0f}s)")

    base = next((r for r in out if r["name"] == args.baseline), None)
    if base is None:
        print(f"\nBỏ qua so cặp: không có hệ thống {args.baseline!r} trong file này.")
    else:
        print(f"\nSo cặp đôi với {args.baseline} (bootstrap ghép cặp, seed=13):")
        for r in out:
            if r["name"] == args.baseline:
                continue
            # Cung mot file du doan nen chac chan cung tap bai, cung thu tu — dieu ma
            # `report.compare()` phai kiem bang guid khi hai bang den tu hai lan chay.
            d = paired_bootstrap(r["per_article"]["bertscore"], base["per_article"]["bertscore"],
                                 n_boot=args.n_boot)
            v = "CÓ ý nghĩa" if d["significant"] else "KHÔNG đủ bằng chứng"
            print(f"  {r['name']} − {args.baseline}: {d['diff']:+.2f} "
                  f"[{d['lo']:+.2f}, {d['hi']:+.2f}] p={d['p']:.4f} → {v}")

    name = "bertscore_" + "+".join(args.tag) + (f"_thu{args.limit}" if args.limit else "")
    dest = RESULTS / "tables" / f"{name}.json"
    dest.write_text(json.dumps(
        {"model": args.model, "n": len(refs), "source": list(args.tag),
         "guid": guids, "systems": out},
        ensure_ascii=False, indent=1,
    ), encoding="utf-8")
    print(f"\nĐã ghi: {dest}")


if __name__ == "__main__":
    main()
