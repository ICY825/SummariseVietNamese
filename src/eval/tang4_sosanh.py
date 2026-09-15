"""Tầng 4: so ghép cặp vòng 1, vòng 2 và đối chứng `--no-train` trên `val`.

    ~/.venvs/torch/Scripts/python.exe src/eval/tang4_sosanh.py

Cần `transformers` và `sentencepiece` (tokenizer BARTpho) để dựng lại nhóm bài bị lọc,
nên không chạy trong `.venv` của dự án. Không chạy mô hình nào: chỉ đọc bảng điểm và
file dự đoán đã lưu, ghi `results/tables/tang4_vong2_val_sosanh.json`.

**Nhóm bị lọc dựng lại thế nào.** Đúng tiêu chí của `hybrid.apply_filter()`: bài có số
token (tokenizer BARTpho, không token đặc biệt) vượt `1.024 − 2`. Trước khi tin mọi con
số khác, script in ROUGE-1 của nhóm ấy cho ba hệ thống vòng 1 — phải ra 26,80 / 25,98 /
26,67 như README.

**Vì sao có hiệu của hai hiệu.** Vòng 2 là một lần HUẤN LUYỆN LẠI, nên nó khác bản gốc
cả ở 898 bài không hề bị lọc. Chênh lệch ở nhóm ấy đo độ dao động giữa hai lần huấn
luyện; lợi ích thật của bộ lọc là phần nhóm 102 bài vượt lên trên mức đó. Hai nhóm rời
nhau, khác cỡ, nên dùng `group_diff()` (lấy mẫu độc lập từng nhóm), không ghép cặp.
"""

import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from eval.stats import paired_bootstrap  # noqa: E402
from eval.truncation import group_diff  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
MODEL = "vinai/bartpho-syllable"
BUDGET = 1024 - 2
TAGS = {
    "goc": "bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024",
    "doichung": "bartpho-syllable-train_20k_val_in1024",
    "v1_lexrank": "bartpho-syllable-train_20k_val_in1024_loc-lexrank",
    "v1_lead": "bartpho-syllable-train_20k_val_in1024_loc-lead_lexrank",
    "v2_lead": "bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024_loc-lead_lexrank",
}


def fmt(d):
    return f"{d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] p={d['p']:.4f}"


def main():
    tab, pred = {}, {}
    for k, tag in TAGS.items():
        tab[k] = json.loads((RESULTS / "tables" / f"{tag}.json").read_text(encoding="utf-8"))[0]
        p = json.loads((RESULTS / "predictions" / f"{tag}.json").read_text(encoding="utf-8"))
        # bang ghi guid dang chuoi, file du doan dang so
        if list(map(str, p["guid"])) != list(map(str, tab[k]["guid"])):
            raise SystemExit(f"{tag}: guid của bảng và file dự đoán lệch nhau.")
        pred[k] = p[tab[k]["name"]]
    guid = [str(g) for g in tab["goc"]["guid"]]
    for k in TAGS:
        if [str(g) for g in tab[k]["guid"]] != guid:
            raise SystemExit(f"{TAGS[k]} chấm trên tập bài khác bản gốc.")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = {str(r["guid"]): r["article_raw"] for r in load_split("val", add_raw=True)}
    if set(raw) != set(guid):
        raise SystemExit("Bảng điểm không chấm trên đúng data/splits/val.json.")
    loc = np.array([len(tok(raw[g], add_special_tokens=False)["input_ids"]) > BUDGET for g in guid])
    print(f"{len(guid)} bài, {int(loc.sum())} bài bị lọc\n")

    def r(k, m="rouge1"):
        return np.asarray(tab[k]["per_article"][m], dtype=float)

    print("ROUGE-1 theo nhóm (kiểm: vòng 1 phải ra 26,80 / 25,98 / 26,67 ở nhóm bị lọc)")
    nhom = {}
    for k in TAGS:
        nhom[k] = {"loc": r(k)[loc].mean(), "khong_loc": r(k)[~loc].mean(), "tat_ca": r(k).mean()}
        print(f"  {k:11s} bị lọc {nhom[k]['loc']:.2f} | không lọc {nhom[k]['khong_loc']:.2f} | "
              f"tất cả {nhom[k]['tat_ca']:.2f}")

    print("\nĐối chứng --no-train không lọc")
    trung = np.array([a == b for a, b in zip(pred["doichung"], pred["goc"])])
    doichung = {
        "trung_khit_goc": {"tat_ca": int(trung.sum()), "khong_loc": int(trung[~loc].sum()),
                           "loc": int(trung[loc].sum())},
        "trung_khit_vong1_tren_khong_loc": {},
        "doichung_tru_goc_rouge1": paired_bootstrap(r("doichung"), r("goc")),
        "vong1_tru_doichung_rouge1_nhom_loc": {},
    }
    print(f"  trùng khít bản gốc {trung.sum()}/{len(guid)} "
          f"(không lọc {trung[~loc].sum()}, bị lọc {trung[loc].sum()})")
    for k in ("v1_lexrank", "v1_lead"):
        s = np.array([a == b for a, b in zip(pred["doichung"], pred[k])])
        doichung["trung_khit_vong1_tren_khong_loc"][k] = f"{s[~loc].sum()}/{(~loc).sum()}"
        d = paired_bootstrap(r(k)[loc], r("doichung")[loc])
        doichung["vong1_tru_doichung_rouge1_nhom_loc"][k] = d
        print(f"  trùng khít {k} trên nhóm không lọc: {s[~loc].sum()}/{(~loc).sum()}; "
              f"{k} − đối chứng, nhóm bị lọc: {fmt(d)}")
    print(f"  đối chứng − gốc, tất cả: {fmt(doichung['doichung_tru_goc_rouge1'])}")

    print("\nVòng 2 (huấn luyện lại trên đầu vào đã lọc)")
    vong2 = {}
    for ref in ("goc", "doichung", "v1_lead"):
        for m in ("rouge1", "rouge2", "rougeL"):
            a, b = r("v2_lead", m), r(ref, m)
            res = {"tat_ca": paired_bootstrap(a, b), "loc": paired_bootstrap(a[loc], b[loc]),
                   "khong_loc": paired_bootstrap(a[~loc], b[~loc]),
                   "loc_tru_khong_loc": group_diff((a - b)[loc], (a - b)[~loc])}
            vong2[f"v2_tru_{ref}_{m}"] = res
            print(f"  v2 − {ref:8s} {m:6s} tất cả {fmt(res['tat_ca'])} | bị lọc {fmt(res['loc'])} | "
                  f"không lọc {fmt(res['khong_loc'])} | hiệu hai hiệu {fmt(res['loc_tru_khong_loc'])}")
    t2 = sum(a == b for a, b in zip(pred["v2_lead"], pred["goc"]))
    print(f"  vòng 2 trùng khít bản gốc: {t2}/{len(guid)}")

    # BERTScore cham rieng bang run_bertscore.py (cung ba file, doi ten TAG:TEN); co thi
    # tach theo nhom giong het ROUGE, chua co thi bo qua chu khong chan phan ROUGE.
    bs_path = RESULTS / "tables" / ("bertscore_" + "+".join(TAGS[k] for k in ("goc", "doichung", "v2_lead")) + ".json")
    bertscore = None
    if bs_path.exists():
        bs = json.loads(bs_path.read_text(encoding="utf-8"))
        if [str(g) for g in bs["guid"]] != guid:
            raise SystemExit(f"{bs_path.name} chấm trên tập bài khác.")
        he = {s["name"]: np.asarray(s["per_article"]["bertscore"], dtype=float) for s in bs["systems"]}
        a, b = he["BARTpho-vong2-lead_lexrank"], he["BARTpho-goc"]
        bertscore = {"tat_ca": paired_bootstrap(a, b), "loc": paired_bootstrap(a[loc], b[loc]),
                     "khong_loc": paired_bootstrap(a[~loc], b[~loc]),
                     "loc_tru_khong_loc": group_diff((a - b)[loc], (a - b)[~loc])}
        print(f"  BERTScore v2 − goc: tất cả {fmt(bertscore['tat_ca'])} | bị lọc {fmt(bertscore['loc'])} | "
              f"không lọc {fmt(bertscore['khong_loc'])} | hiệu hai hiệu {fmt(bertscore['loc_tru_khong_loc'])}")
    else:
        print(f"  (chưa có {bs_path.name} — bỏ qua BERTScore)")

    dest = RESULTS / "tables" / "tang4_vong2_val_sosanh.json"
    dest.write_text(json.dumps({
        "split": "val", "n": len(guid), "n_loc": int(loc.sum()), "tokenizer": MODEL, "budget": BUDGET,
        "guid_loc": [g for g, x in zip(guid, loc) if x], "sources": TAGS, "rouge1_theo_nhom": nhom,
        "doichung": doichung, "vong2": vong2, "vong2_trung_khit_goc": t2,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


if __name__ == "__main__":
    main()
