"""Hướng mới, giai đoạn 2 — THỬ NGHIỆM ÂM: cho BARTpho viết lại câu treo của bản chọn câu.

    ~/.venvs/demo/Scripts/python.exe src/models/viet_lai.py ung-vien --split tune  # sinh, cần torch, ~9 phút CPU
    .venv/Scripts/python.exe src/models/viet_lai.py phan-tich --split tune           # đếm lại các con số của README

KẾT LUẬN (xem README, giai đoạn 2): KHÔNG dùng. BARTpho được huấn luyện để viết sapo cho cả bài
nên với đầu vào hai câu nó vẫn viết một câu mở đầu bản tin mới, không phải viết lại câu treo:
22/67 câu có chi tiết lạ, chỉ 15/67 giữ được ý câu treo, và có lỗi đảo chủ thể/bịa chức vụ mà
bộ đo không bắt được. File này được giữ lại làm bằng chứng cho kết luận đó.

Giai đoạn 1 đủ ý và không bịa, nhưng khoảng 15% bản có một câu mở đầu bằng từ nối hoặc đại
từ ("Tuy nhiên", "Trong đó", "Họ"...) mà câu đứng trước nó không được chọn — đọc lên hụt
mạch. Kéo cả câu đứng trước vào (`noi_tien_de`) tốn ngân sách và làm giảm chỉ số. Ở đây
BARTpho đọc **câu đứng trước + câu treo** và viết lại thành một câu tự đứng được, thay cho câu
treo. Mọi câu khác vẫn chép nguyên.

Vì sao chỉ viết lại câu treo: thử ghép một câu BARTpho vào đầu bản chọn câu (trên `tune`) làm
GIẢM cả recall lẫn độ phủ chi tiết — câu sinh chiếm ngân sách mà phần lớn lặp ý đã có. Chỗ
chọn câu thật sự hụt là mạch văn, không phải ý.

`ung-vien` ghi MỌI câu viết lại kèm đầu vào, để soi được mà không phải sinh lại.
"""

import argparse
import json
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.chon_cau import chon, chuan_bi_bai, nap  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
MAX_INPUT = 1024
MAX_TARGET = 80
GEN = {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}  # mac dinh, tuan 5 do khong cau hinh nao hon


def cau_hinh_gd1():
    th = json.loads((RESULTS / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))["thang"]
    return [th[k] for k in ("ngan_sach", "w_vt", "w_lex", "lam", "noi_tien_de", "loc_rac")]


def cau_treo_can_viet(b, idx):
    """Các đơn vị treo trong bản chọn câu mà câu đứng trước KHÔNG được chọn."""
    chon_ = set(idx)
    return [k for k in idx if b["treo"][k] and k > 0 and (k - 1) not in chon_]

# --------------------------------------------------------------------------
# Buoc 1: sinh ung vien (can torch)
# --------------------------------------------------------------------------

def cmd_ung_vien(args):
    if args.split == "test" and not args.cho_phep_test:
        raise SystemExit("Từ chối sinh trên `test`: tập này dùng MỘT lần. Thêm --cho-phep-test khi chấm lần cuối.")
    dest = RESULTS / "predictions" / f"viet-lai_{args.split}_ung-vien.json"
    if dest.exists():
        raise SystemExit(f"{dest} đã có — không đè. Xoá tay nếu thật sự muốn sinh lại.")

    import torch
    from transformers import AutoModelForSeq2SeqLM

    sys.path.insert(0, str(RESULTS.parent / "app"))
    from pipeline import kiem_trong_so, tim_checkpoint  # noqa: E402
    from models.measure_tokens import load_tokenizer
    from models.vit5 import generate

    cfg = cau_hinh_gd1()
    rows, scores = nap(args.split)
    viec = []
    for r, s in zip(rows, scores):
        b = chuan_bi_bai(r["article"], s)
        idx = chon(b, *cfg)
        for k in cau_treo_can_viet(b, idx):
            viec.append({"guid": str(r["guid"]), "don_vi": int(k), "chon": [int(i) for i in idx],
                         "truoc": b["dv"][k - 1][1], "treo": b["dv"][k][1]})
    print(f"{len(viec)} câu treo cần viết lại trên {len(rows)} bài {args.split}")

    duong = tim_checkpoint("bartpho")
    kiem_trong_so(duong)
    tok = load_tokenizer(str(duong))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(duong))
    t0 = time.time()
    ra = generate(model, tok, [{"article_raw": v["truoc"] + " " + v["treo"]} for v in viec],
                  MAX_INPUT, MAX_TARGET, batch=args.batch, gen=GEN)
    for v, o in zip(viec, ra):
        v["viet_lai"] = o
    giay = round(time.time() - t0, 1)
    dest.write_text(json.dumps({
        "split": args.split, "cau_hinh_gd1": cfg, "checkpoint": duong.as_posix(), "gen": GEN,
        "max_target": MAX_TARGET, "torch": torch.__version__, "giay": giay, "ung_vien": viec,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Sinh {len(viec)} câu trong {giay}s. Đã ghi {dest}")


# --------------------------------------------------------------------------
# Buoc 2: dem lai cac con so cua ket qua am (chi can .venv)
# --------------------------------------------------------------------------

def cmd_phan_tich(args):
    from data.splits import load_split
    from eval.chinh_xac import chi_tiet_la, rouge_recall
    from models.chon_cau import la_treo

    bai = {str(r["guid"]): r["article"] for r in load_split(args.split, add_raw=True)}
    uv = json.loads((RESULTS / "predictions" / f"viet-lai_{args.split}_ung-vien.json")
                    .read_text(encoding="utf-8"))["ung_vien"]
    la = giu = qua = 0
    for i, v in enumerate(uv):
        o = v["viet_lai"]
        co_la = bool(chi_tiet_la(o, bai[v["guid"]]))
        rc = rouge_recall(o, v["treo"])["r1"]
        la += co_la
        giu += rc >= 70
        qua += (not co_la) and (not la_treo(o)) and rc >= 70
        if args.in_het:
            print(f"#{i} guid {v['guid']} | lạ {co_la} | recall câu treo {rc:.0f}\n"
                  f"  TRƯỚC: {v['truoc']}\n  TREO : {v['treo']}\n  VIẾT : {o}")
    print(f"{len(uv)} câu viết lại | có chi tiết lạ {la} | giữ ý câu treo (recall ≥ 70) {giu} | "
          f"qua cả ba điều kiện {qua}")


def cmd_thu_ghep(args):
    """Thử nghiệm âm thứ nhất: đặt câu BARTpho (đã sinh sẵn) lên đầu, lấp ngân sách còn lại bằng chọn câu.

    Không cần torch: dùng bản tóm tắt BARTpho `train_20k` trên `tune` đã commit từ tuần 5.
    """
    import numpy as np
    from data.text import for_scoring, syllables
    from eval.chinh_xac import cham, chi_tiet_la
    from models.chon_cau import van_ban

    d = json.loads((RESULTS / "predictions" / "bartpho-syllable-train_20k_tune_in1024.json").read_text(encoding="utf-8"))
    bart = dict(zip([str(g) for g in d["guid"]], d["bartpho-syllable-train_20k"]))
    cfg = cau_hinh_gd1()
    rows, scores = nap("tune")
    kq = {"chọn câu (giai đoạn 1)": [], "chỉ BARTpho": [], "BARTpho + chọn câu": [],
          "BARTpho + chọn câu, lọc chi tiết lạ": []}
    for r, s in zip(rows, scores):
        b = chuan_bi_bai(r["article"], s)
        ext = van_ban(b, chon(b, *cfg))
        a = bart[str(r["guid"])]
        con_lai = max(cfg[0] - len(syllables(for_scoring(a))), 0)
        ghep = a + " " + van_ban(b, chon(b, con_lai, *cfg[1:]))
        for ten, t in zip(kq, (ext, a, ghep, ext if chi_tiet_la(a, r["article"]) else ghep)):
            kq[ten].append(cham(t, r["abstract"], r["article"]))
    for ten, v in kq.items():
        phu = [x["do_phu_chi_tiet"] for x in v if x["do_phu_chi_tiet"] is not None]
        print(f"{ten:38s} recall {np.mean([x['r1_recall'] for x in v]):5.1f} | phủ {np.mean(phu):5.1f} | "
              f"chi tiết lạ {100 * np.mean([x['co_chi_tiet_la'] for x in v]):4.1f}% | "
              f"{np.mean([x['am_tiet'] for x in v]):5.1f} âm tiết")


def main():
    ap = argparse.ArgumentParser(description="Hướng mới, giai đoạn 2: hai thử nghiệm âm với BARTpho.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("thu-ghep", help="ghép câu BARTpho vào đầu bản chọn câu (tune)").set_defaults(fn=cmd_thu_ghep)
    p = sub.add_parser("phan-tich", help="đếm lại kết quả từ file ứng viên đã sinh")
    p.add_argument("--split", default="tune", choices=["tune", "val"])
    p.add_argument("--in-het", action="store_true", help="in từng câu viết lại")
    p.set_defaults(fn=cmd_phan_tich)
    u = sub.add_parser("ung-vien", help="sinh câu viết lại cho mọi câu treo (cần torch)")
    u.add_argument("--split", required=True, choices=["tune", "val", "test"])
    u.add_argument("--batch", type=int, default=1)  # 1: tranh padding lam lech so voi ban da bao cao
    u.add_argument("--cho-phep-test", action="store_true")
    u.set_defaults(fn=cmd_ung_vien)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
