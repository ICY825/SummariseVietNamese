"""Tầng 2, bước hai: chọn SỐ CÂU linh hoạt thay vì luôn lấy đúng 3.

    # 1. Cho điểm từng câu — cần torch, chạy được trên CPU
    ~/.venvs/torch/Scripts/python.exe src/models/phobert_select.py score --model <final> --split tune
    ~/.venvs/torch/Scripts/python.exe src/models/phobert_select.py score --model <final> --split val
    # 2. Dò quy tắc trên `tune`, đem quy tắc thắng sang `val` — không cần torch
    .venv/Scripts/python.exe src/models/phobert_select.py select

Vì sao: 995/1.000 bản tóm tắt của tầng 2 có đủ 3 câu, dài 100 âm tiết, trong khi sapo
dài 35 và Oracle-3 dừng sớm nên chỉ 128/1.000 bản có đủ 3 câu. Với F1, chữ thừa làm mất
precision — cũng là lý do Lead-1 ngang Lead-3 ở tầng 0.

**Tách hai bước.** Bước tốn kém (chạy PhoBERT) chỉ làm một lần cho mỗi split và ghi điểm
từng câu ra đĩa; mọi quy tắc chọn đọc lại cùng một bộ điểm, nên dò bao nhiêu quy tắc cũng
không tốn thêm lần chạy mô hình nào, và các quy tắc so với nhau là so đúng quy tắc.

**Dò trên `tune`, không trên `val`** — cùng lý do như khâu dò tham số sinh ở tuần 5.
`val` chỉ nhận đúng MỘT quy tắc: quy tắc thắng trên `tune`.

**Không huấn luyện lại.** Mô hình giữ nguyên, chỉ đổi cách biến điểm thành bản tóm tắt.
Nhờ vậy `k3` trên `val` phải tái tạo **từng chữ** file dự đoán của lần chạy Kaggle — đó
là phép kiểm rằng checkpoint được nạp đúng (kể cả `head.pt`) trước khi tin quy tắc mới.

Hai họ quy tắc:

- `kN` — N câu điểm cao nhất. `k3` chính là tầng 2 như đã báo cáo.
- `pT` — mọi câu có xác suất ≥ T, tối đa 3, và luôn giữ ít nhất câu điểm cao nhất: một
  bản tóm tắt rỗng không bao giờ tốt hơn một câu. Xác suất ở đây đã bị `pos_weight` 6,35
  lúc huấn luyện đẩy lên cao, nên lưới T trải rộng chứ không dừng quanh 0,5.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import sentences  # noqa: E402
from eval.report import compare, evaluate, save, table  # noqa: E402
from models.extractive import join  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
NAME = "phobert-sent"
BASE = "phobert-sent-train_20k_{split}_len256"
K_MAX = 3
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RULES = ("k1", "k2", "k3") + tuple(f"p{t:g}" for t in TAUS)


def pick_rule(scores, sents, rule):
    """Chỉ số câu được chọn theo `rule`, trả về theo thứ tự bài.

    Cùng quy ước với `phobert_sent.pick_indices()`: hoà điểm thì câu đứng trước thắng,
    hai bản sao của cùng một câu không được chọn cả hai. `k3` trùng hệt hàm ấy.
    """
    if rule.startswith("k"):
        k, tau = int(rule[1:]), None
    elif rule.startswith("p"):
        k, tau = K_MAX, float(rule[1:])
    else:
        raise ValueError(f"Không biết quy tắc {rule!r}")
    s = np.asarray(scores, dtype=float)
    out, seen = [], set()
    for i in np.argsort(-s, kind="stable"):
        i = int(i)
        if sents[i] in seen:
            continue
        # Da xep giam dan, nen cau dau tien duoi nguong thi moi cau sau cung duoi.
        if tau is not None and out and 1.0 / (1.0 + np.exp(-s[i])) < tau:
            break
        seen.add(sents[i])
        out.append(i)
        if len(out) == k:
            break
    return sorted(out)


def scores_path(split):
    return RESULTS / "predictions" / f"{BASE.format(split=split)}_scores.json"


def cmd_score(args):
    import torch
    from transformers import AutoTokenizer

    from models.phobert_sent import _build_model, encode_rows, score_examples

    if not (Path(args.model) / "head.pt").exists():
        raise SystemExit(f"{args.model} không có head.pt — không phải checkpoint tầng 2.")
    dest = scores_path(args.split)
    if dest.exists():
        raise SystemExit(f"{dest} đã có. Xoá tay nếu thật sự muốn tính lại — không đè.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = list(load_split(args.split, add_raw=False))
    ex = encode_rows(rows, tok, with_labels=False, max_len=args.max_len)
    model = _build_model(args.model, device)
    print(f"Cho điểm {len(rows)} bài {args.split} trên {device} ...")
    t0 = time.time()
    scores = score_examples(model, ex, tok.pad_token_id, device, args.batch)
    secs = round(time.time() - t0, 1)
    dest.write_text(json.dumps({
        "model": args.model, "split": args.split, "max_len": args.max_len,
        "device": device, "torch": torch.__version__, "seconds": secs,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "guid": [str(r["guid"]) for r in rows], "scores": scores,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"Xong sau {secs}s. Đã ghi {dest}")


def load_scores(split):
    path = scores_path(split)
    if not path.exists():
        raise SystemExit(f"Chưa có {path}. Chạy `score --split {split}` trước.")
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = list(load_split(split, add_raw=False))
    if d["guid"] != [str(r["guid"]) for r in rows]:
        raise SystemExit(f"{path.name} chấm trên tập bài khác data/splits/{split}.json.")
    return rows, d


def summaries(rows, scores, rule):
    preds, n_cau = [], []
    for r, s in zip(rows, scores):
        if not s:
            preds.append("")
            n_cau.append(0)
            continue
        sents = sentences(r["article"])
        idx = pick_rule(s, sents[: len(s)], rule)
        preds.append(join(sents, idx))
        n_cau.append(len(idx))
    return preds, n_cau


def _eval(rows, scores, rule, n_boot):
    preds, n_cau = summaries(rows, scores, rule)
    res = evaluate(f"{NAME}-{rule}", preds, [r["abstract"] for r in rows],
                   [r["article"] for r in rows], guids=[str(r["guid"]) for r in rows],
                   n_boot=n_boot)
    res["so_cau_tb"] = round(float(np.mean(n_cau)), 3)
    return res, preds


def cmd_select(args):
    tune_rows, tune = load_scores("tune")
    print(f"Dò {len(RULES)} quy tắc trên tune ({len(tune_rows)} bài) ...")
    sweep = [_eval(tune_rows, tune["scores"], r, args.n_boot)[0] for r in RULES]
    print("\n" + table(sweep))
    print("\nSố câu trung bình: " + ", ".join(f"{r['name']} {r['so_cau_tb']}" for r in sweep))
    by = {r["name"]: r for r in sweep}
    best = max(sweep, key=lambda r: r["corpus"]["rouge1"]["mean"])
    rule = best["name"].removeprefix(f"{NAME}-")
    print(f"\nThắng trên tune: {rule}")
    tune_vs = {m: compare(best, by[f"{NAME}-k3"], m, n_boot=args.n_boot)
               for m in ("rouge1", "rouge2", "rougeL")} if rule != "k3" else {}
    for v in tune_vs.values():
        print("  " + v)
    save(sweep, f"{BASE.format(split='tune')}_rules.json")

    val_rows, val = load_scores("val")
    # Kiem chung truoc khi tin: `k3` phai tai tao dung ban tom tat cua lan chay Kaggle.
    cu = json.loads((RESULTS / "predictions" / f"{BASE.format(split='val')}.json")
                    .read_text(encoding="utf-8"))
    k3, _ = summaries(val_rows, val["scores"], "k3")
    trung = sum(a == b for a, b in zip(k3, cu[NAME]))
    print(f"\nKiểm chứng: k3 tái tạo {trung}/{len(k3)} bản tóm tắt của lần chạy Kaggle trên val.")
    if rule == "k3":
        print("Quy tắc thắng là k3 — trùng tầng 2 đã báo cáo, không có gì mới để chấm trên val.")
        return

    res, preds = _eval(val_rows, val["scores"], rule, args.n_boot)
    print("\n" + table([res]) + f"\nSố câu trung bình: {res['so_cau_tb']}")
    goc = json.loads((RESULTS / "tables" / f"{BASE.format(split='val')}.json")
                     .read_text(encoding="utf-8"))[0]
    moc = {r["name"]: r for r in json.loads(
        (RESULTS / "tables" / "baselines_val.json").read_text(encoding="utf-8"))}
    versus = {}
    for ten, doi in (("k3", goc), ("Lead-3", moc["Lead-3"]), ("Lead-1", moc["Lead-1"])):
        for m in ("rouge1", "rouge2", "rougeL"):
            versus[f"{ten} {m}"] = compare(res, doi, m, n_boot=args.n_boot)
            print("  " + versus[f"{ten} {m}"])

    tag = f"{BASE.format(split='val')}_{rule}"
    save([res], f"{tag}.json")
    (RESULTS / "predictions" / f"{tag}.json").write_text(json.dumps(
        {"guid": [str(r["guid"]) for r in val_rows],
         "reference": [r["abstract"] for r in val_rows], res["name"]: preds},
        ensure_ascii=False), encoding="utf-8")
    (RESULTS / "tables" / f"{tag}_run.json").write_text(json.dumps({
        "tag": tag, "rule": rule, "rules_tried": list(RULES), "chosen_on": "tune",
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scores_files": {s: str(scores_path(s)) for s in ("tune", "val")},
        "scoring_env": {s: {k: d[k] for k in ("model", "device", "torch", "seconds")}
                        for s, d in (("tune", tune), ("val", val))},
        "tune": {r["name"]: {"rouge1": r["corpus"]["rouge1"]["mean"],
                             "so_cau_tb": r["so_cau_tb"]} for r in sweep},
        "tune_vs_k3": tune_vs,
        "k3_reproduces_kaggle_val": f"{trung}/{len(k3)}",
        "val": {"corpus": res["corpus"], "length": res["length"], "so_cau_tb": res["so_cau_tb"]},
        "versus": versus,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi results/tables/{tag}.json, _run.json và results/predictions/{tag}.json")


def main():
    ap = argparse.ArgumentParser(description="Tầng 2: chọn số câu linh hoạt.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="cho điểm từng câu bằng checkpoint tầng 2 (cần torch)")
    s.add_argument("--model", required=True, help="thư mục final có head.pt")
    s.add_argument("--split", required=True, choices=["tune", "val"])
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--max-len", type=int, default=256)
    s.set_defaults(fn=cmd_score)
    c = sub.add_parser("select", help="dò quy tắc trên tune, áp quy tắc thắng lên val")
    c.add_argument("--n-boot", type=int, default=10_000)
    c.set_defaults(fn=cmd_select)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
