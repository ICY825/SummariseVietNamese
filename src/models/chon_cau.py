"""Hướng mới, giai đoạn 1: chọn câu có chủ đích — đủ ý, không sai sự thật.

    .venv/Scripts/python.exe src/models/chon_cau.py do               # dò cấu hình trên tune
    .venv/Scripts/python.exe src/models/chon_cau.py sinh --split val # sinh bằng cấu hình đã chốt

Chép nguyên câu nên **không thể sai sự thật**; việc còn lại là chọn câu cho đủ ý. Bốn khuyết
tật của các tầng cũ, và cách xử lý — mỗi cái đều dựa trên dữ liệu `tune` chứ không phỏng đoán:

- **Câu ngoài cửa sổ PhoBERT không có điểm.** 49% số câu nằm ngoài 256 token đầu, và ở 16%
  số bài câu khớp sapo nhất nằm từ vị trí 10 trở đi. Điểm câu vì vậy cộng thêm độ trung tâm
  LexRank và ưu tiên vị trí, hai thứ có cho MỌI câu.
- **Lặp ý.** Chọn tham lam, cộng điểm cho phần âm tiết CHƯA được phủ, bỏ câu gần như trùng
  ý đã chọn — lỗi của bài "trốn thuế" trong demo, nơi hai câu cùng nói "không kê khai, nộp
  thuế" còn việc bị bắt tạm giam thì mất.
- **Câu treo.** Câu mở đầu bằng "Tuy nhiên", "Theo đó", "Điều này"... (hàng trăm câu trên
  `tune`) được kéo kèm câu đứng trước; không đủ ngân sách thì bỏ.
- **Rác và mảnh câu.** Chú thích ảnh, "Xem thêm", dòng tên tác giả bị loại; câu bị cắt nhầm
  ở chữ viết tắt tên người ("...đưa cháu Tr.") được GHÉP với câu sau thay vì bỏ.

Mục tiêu chọn cấu hình, chốt trước khi dò: tối đa (độ phủ chi tiết + ROUGE-1 recall) với
độ dài trung bình ≤ 110 âm tiết và ≤ 4 câu mỗi bản.
"""

import argparse
import itertools
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import for_scoring, sentences, syllables  # noqa: E402
from eval.chinh_xac import _ho_tro, chi_tiet  # noqa: E402
from models.hybrid import lexrank_scores  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
TOI_DA_CAU = 4
NGAN_SACH_TB = 110
TREO = ("tuy nhiên", "theo đó", "ngoài ra", "trước đó", "sau đó", "trong khi đó", "đây là",
        "điều này", "đó là", "trong đó", "cụ thể", "bên cạnh đó", "do đó", "vì vậy", "vì thế",
        "tương tự", "theo cách này", "việc này", "hơn nữa", "thậm chí", "như vậy", "khi đó",
        "lúc đó", "cũng theo", "họ ", "nó ")
RAC = re.compile(r"ảnh\s*:|\(ảnh\s*:|ảnh minh hoạ|ảnh minh họa|xem thêm|nguồn\s*:|video\s*:", re.I)
# Chi mot chu hoa ("T.", "N.V.") hoac phu am dau ghep that cua ten nguoi Viet ("Tr.", "Ng.").
# Cho phep moi tu hai chu cai thi ghep nham ca dia danh cuoi cau: "Hoang Sa.", "Hoa An.".
# Hoc ham dung truoc ten ("TS.", "GS.TS.", "PGS.TS.", "ThS.") va "St." ("St. Petersburg")
# cung bi cat nham: mot bai val mat ten chuyen gia, con lai manh "...trao doi nhanh voi TS.".
# KHONG ghep "NSND.", "NSƯT.", "CN.": tren tune/val chung thuong la cuoi cau that
# ("...danh hieu NSƯT." roi "Trong do, ...").
VIET_TAT_TEN = re.compile(r"(?:^|\s)(?:(?:TSKH|PGS|ThS|GS|TS|St|Tr|Th|Ng|Nh|Ch|Kh|Ph|Gi|Qu|[A-ZĐ])\.)+\s*$")


# --------------------------------------------------------------------------
# Tien xu ly cau
# --------------------------------------------------------------------------

def don_vi(article):
    """Danh sách câu đã ghép mảnh cắt nhầm; mỗi phần tử: (chỉ số câu gốc, văn bản thô)."""
    goc = [for_scoring(s) for s in sentences(article)]
    out, i = [], 0
    while i < len(goc):
        idx, t = [i], goc[i]
        while VIET_TAT_TEN.search(t) and idx[-1] + 1 < len(goc):
            idx.append(idx[-1] + 1)
            t = t + " " + goc[idx[-1]]
        out.append((idx, t))
        i = idx[-1] + 1
    return out


def la_rac(t):
    am = len(syllables(t))
    # Dau hieu chu thich chi loai cau NGAN: du lieu co cho dinh dong ghi nguon anh vao dau mot
    # cau noi dung dai ("Anh: Internet De day nhanh tien do, Pho Thu tuong yeu cau...").
    if RAC.search(t) and am <= 30:
        return True
    ket = t.rstrip().endswith((".", "!", "?", '"', "»", "”"))
    if not ket and am <= 15:
        return True
    # KHONG loai cau co so ngoac kep le: loi trich dai trai qua nhieu cau nen tung cau le
    # ngoac la binh thuong — luat do tung loai oan hang loat cau trich co noi dung.
    return am <= 4 or (t.count('"') % 2 == 1 and am <= 6)


def la_treo(t):
    dau = t.lower().lstrip('"“ ')
    return dau.startswith(TREO)


# --------------------------------------------------------------------------
# Chon cau
# --------------------------------------------------------------------------

def chuan_bi_bai(article, diem_phobert):
    """Mọi thứ không phụ thuộc cấu hình, tính một lần cho mỗi bài."""
    dv = don_vi(article)
    n = len(dv)
    p = np.zeros(n)
    if diem_phobert:
        for k, (idx, _) in enumerate(dv):
            vals = [diem_phobert[i] for i in idx if i < len(diem_phobert)]
            if vals:
                p[k] = 1 / (1 + np.exp(-max(vals)))
    lex = lexrank_scores([t for _, t in dv]) if n > 1 else np.ones(n)
    lex = lex / lex.max() if n and lex.max() > 0 else lex
    return {
        "dv": dv,
        "p": p,
        "lex": np.asarray(lex, dtype=float),
        "vt": np.array([1 / (1 + k) for k in range(n)]),
        "am": np.array([len(syllables(t)) for _, t in dv]),
        "tu": [set(syllables(t)) for _, t in dv],
        "rac": np.array([la_rac(t) for _, t in dv]),
        "treo": np.array([la_treo(t) for _, t in dv]),
    }


def chon(b, ngan_sach, w_vt, w_lex, lam, noi_tien_de, loc_rac):
    """Chỉ số các đơn vị được chọn, theo thứ tự bài."""
    n = len(b["dv"])
    if not n:
        return []
    diem = b["p"] + w_vt * b["vt"] + w_lex * b["lex"]
    duoc = ~b["rac"] if loc_rac else np.ones(n, dtype=bool)
    if not duoc.any():
        duoc = np.ones(n, dtype=bool)
    chon_, phu, am = [], set(), 0
    while len(chon_) < TOI_DA_CAU:
        tot, gt = None, -1e9
        for k in range(n):
            if k in chon_ or not duoc[k] or not b["tu"][k]:
                continue
            moi = len(b["tu"][k] - phu) / len(b["tu"][k])
            if chon_ and moi < 0.3:
                continue
            v = diem[k] + lam * moi
            if v > gt:
                tot, gt = k, v
        if tot is None:
            break
        them = [tot]
        if noi_tien_de and b["treo"][tot] and tot > 0 and (tot - 1) not in chon_:
            if duoc[tot - 1] or not loc_rac:
                them = [tot - 1, tot]
            else:
                duoc[tot] = False
                continue
        can = sum(b["am"][k] for k in them)
        if chon_ and (am + can > ngan_sach or len(chon_) + len(them) > TOI_DA_CAU):
            duoc[tot] = False
            continue
        chon_ += them
        am += can
        for k in them:
            phu |= b["tu"][k]
    return sorted(set(chon_))


def van_ban(b, idx):
    return " ".join(b["dv"][k][1] for k in idx)


def lead_ngan_sach(b, ngan_sach):
    idx, am = [], 0
    for k in range(len(b["dv"])):
        if idx and (am + b["am"][k] > ngan_sach or len(idx) >= TOI_DA_CAU):
            break
        idx.append(k)
        am += b["am"][k]
    return idx


# --------------------------------------------------------------------------
# Cham nhanh trong vong do (chi tiet la cua extractive = 0 theo cau truc)
# --------------------------------------------------------------------------

def cham_nhanh(tom_tat, sapo_ct, sapo_uni, sapo_n):
    toks = syllables(for_scoring(tom_tat))
    phu = None if not sapo_ct else 100 * sum(_ho_tro(c, toks) for c in sapo_ct) / len(sapo_ct)
    H = Counter(toks)
    rc = 100 * sum(min(c, H[w]) for w, c in sapo_uni.items()) / sapo_n if sapo_n else 0.0
    return phu, rc, len(toks)


def nap(split):
    rows = list(load_split(split, add_raw=True))
    sc = json.loads((RESULTS / "predictions" / f"phobert-sent-train_20k_{split}_len256_scores.json")
                    .read_text(encoding="utf-8"))
    if [str(g) for g in sc["guid"]] != [str(r["guid"]) for r in rows]:
        raise SystemExit(f"Điểm PhoBERT {split} lệch guid với data/splits/{split}.json")
    return rows, sc["scores"]


def cmd_do(args):
    rows, scores = nap("tune")
    t0 = time.time()
    bai = [chuan_bi_bai(r["article"], s) for r, s in zip(rows, scores)]
    sapo = []
    for r in rows:
        toks = syllables(for_scoring(r["abstract"]))
        sapo.append((chi_tiet(r["abstract"]), Counter(toks), len(toks)))
    print(f"Chuẩn bị {len(rows)} bài tune trong {time.time() - t0:.0f}s")

    def do(ten, ds_idx):
        phu, rc, am, cau = [], [], [], []
        for b, (ct, uni, n), idx in zip(bai, sapo, ds_idx):
            p, r, a = cham_nhanh(van_ban(b, idx), ct, uni, n)
            if p is not None:
                phu.append(p)
            rc.append(r)
            am.append(a)
            cau.append(len(idx))
        return {"ten": ten, "phu": float(np.mean(phu)), "recall": float(np.mean(rc)),
                "am": float(np.mean(am)), "cau_tb": float(np.mean(cau)), "cau_max": int(max(cau))}

    moc = [do("Lead-3", [list(range(min(3, len(b["dv"])))) for b in bai])]
    k3 = []
    for b in bai:
        idx = sorted(np.argsort(-b["p"], kind="stable")[:3].tolist()) if b["p"].any() else list(range(min(3, len(b["dv"]))))
        k3.append(idx)
    moc.append(do("PhoBERT 3 câu", k3))
    for B in args.ngan_sach:
        moc.append(do(f"Lead theo ngân sách {B}", [lead_ngan_sach(b, B) for b in bai]))
    for m in moc:
        print(f"  MỐC {m['ten']:26s} phủ {m['phu']:5.1f} | recall {m['recall']:5.1f} | "
              f"{m['am']:5.1f} âm tiết | {m['cau_tb']:.1f} câu")

    luoi = list(itertools.product(args.ngan_sach, (0.0, 0.5, 1.0), (0.0, 0.5), (0.0, 0.5, 1.0), (False, True), (False, True)))
    print(f"\nDò {len(luoi)} cấu hình ...")
    kq = []
    for B, w_vt, w_lex, lam, tien_de, rac in luoi:
        m = do("", [chon(b, B, w_vt, w_lex, lam, tien_de, rac) for b in bai])
        m.update(ngan_sach=B, w_vt=w_vt, w_lex=w_lex, lam=lam, noi_tien_de=tien_de, loc_rac=rac,
                 muc_tieu=m["phu"] + m["recall"])
        kq.append(m)
    hop_le = [m for m in kq if m["am"] <= NGAN_SACH_TB and m["cau_max"] <= TOI_DA_CAU]
    hop_le.sort(key=lambda m: -m["muc_tieu"])
    print(f"{len(hop_le)}/{len(kq)} cấu hình thoả ràng buộc độ dài. 10 cấu hình tốt nhất:")
    for m in hop_le[:10]:
        print(f"  phủ {m['phu']:5.1f} | recall {m['recall']:5.1f} | {m['am']:5.1f} âm tiết | {m['cau_tb']:.1f} câu"
              f" | B={m['ngan_sach']} vt={m['w_vt']} lex={m['w_lex']} lam={m['lam']}"
              f" tiền đề={m['noi_tien_de']} rác={m['loc_rac']}")
    dest = RESULTS / "tables" / "chon_cau_tune_do.json"
    dest.write_text(json.dumps({"moc": moc, "cau_hinh": kq, "thang": hop_le[0] if hop_le else None},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


def cmd_sinh(args):
    """Sinh bản tóm tắt bằng cấu hình THẮNG trên `tune` — đọc từ file dò, không gõ tay."""
    if args.split == "test" and not args.cho_phep_test:
        raise SystemExit("Từ chối sinh trên `test`: tập này dùng MỘT lần. Thêm --cho-phep-test khi chấm lần cuối.")
    do_ = json.loads((RESULTS / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))
    th = do_["thang"]
    cfg = {k: th[k] for k in ("ngan_sach", "w_vt", "w_lex", "lam", "noi_tien_de", "loc_rac")}
    tag = f"chon-cau_{args.split}"
    dest = RESULTS / "predictions" / f"{tag}.json"
    if dest.exists():
        raise SystemExit(f"{dest} đã có — không đè. Xoá tay nếu thật sự muốn sinh lại.")
    rows, scores = nap(args.split)
    t0 = time.time()
    preds = []
    for r, s in zip(rows, scores):
        b = chuan_bi_bai(r["article"], s)
        preds.append(van_ban(b, chon(b, cfg["ngan_sach"], cfg["w_vt"], cfg["w_lex"], cfg["lam"],
                                     cfg["noi_tien_de"], cfg["loc_rac"])))
    rong = sum(1 for p in preds if not p.strip())
    dest.write_text(json.dumps({"guid": [str(r["guid"]) for r in rows],
                                "reference": [r["abstract"] for r in rows], "chon-cau": preds},
                               ensure_ascii=False), encoding="utf-8")
    (RESULTS / "tables" / f"{tag}_run.json").write_text(json.dumps({
        "tag": tag, "split": args.split, "cau_hinh": cfg, "chon_tren": "tune",
        "diem_phobert": f"phobert-sent-train_20k_{args.split}_len256_scores.json",
        "giay": round(time.time() - t0, 1), "ban_rong": rong,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Cấu hình (thắng trên tune): {cfg}")
    print(f"Sinh {len(preds)} bản {args.split} trong {time.time() - t0:.0f}s, {rong} bản rỗng. Đã ghi {dest}")


def main():
    ap = argparse.ArgumentParser(description="Hướng mới, giai đoạn 1: chọn câu có chủ đích.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("do", help="dò cấu hình trên tune")
    d.add_argument("--ngan-sach", type=int, nargs="+", default=[90, 100, 110])
    d.set_defaults(fn=cmd_do)
    s = sub.add_parser("sinh", help="sinh bản tóm tắt bằng cấu hình thắng trên tune")
    s.add_argument("--split", required=True, choices=["tune", "val", "test"])
    s.add_argument("--cho-phep-test", action="store_true", help="MỞ khoá `test` — chỉ ở lần chấm cuối")
    s.set_defaults(fn=cmd_sinh)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
