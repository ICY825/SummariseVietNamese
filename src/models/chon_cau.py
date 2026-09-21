"""Hướng mới, giai đoạn 1 và 2: chọn câu có chủ đích — đủ ý, không sai sự thật.

    .venv/Scripts/python.exe src/models/chon_cau.py do                        # gđ 1: dò trên tune
    .venv/Scripts/python.exe src/models/chon_cau.py do --khong-phobert        # gđ 1: đối chứng tắt PhoBERT
    .venv/Scripts/python.exe src/models/chon_cau.py so-phobert                # gđ 1: PhoBERT góp bao nhiêu
    .venv/Scripts/python.exe src/models/chon_cau.py do-treo                   # gđ 2: dò xử lý câu treo trên tune
    .venv/Scripts/python.exe src/models/chon_cau.py sinh --gd 2 --split val   # sinh bằng cấu hình đã chốt

Chép nguyên câu nên **không bịa** chi tiết — dù ghép câu vẫn có thể tạo hàm ý sai (giai đoạn 3).
Việc còn lại là chọn câu cho đủ ý. Bốn khuyết tật của các tầng cũ, và cách xử lý — mỗi cái đều dựa trên dữ liệu `tune` chứ không phỏng đoán:

- **Câu ngoài cửa sổ PhoBERT không có điểm.** 49% số câu nằm ngoài 256 token đầu, và ở 20%
  số bài câu khớp sapo nhất (ROUGE-1 F1 của từng câu với sapo) nằm từ vị trí 10 trở đi.
  Điểm câu vì vậy cộng thêm độ trung tâm LexRank và ưu tiên vị trí, hai thứ có cho MỌI câu.
- **Lặp ý.** Chọn tham lam, cộng điểm cho phần âm tiết CHƯA được phủ, bỏ câu gần như trùng
  ý đã chọn — lỗi của bài "trốn thuế" trong demo, nơi hai câu cùng nói "không kê khai, nộp
  thuế" còn việc bị bắt tạm giam thì mất.
- **Câu treo** — câu mở đầu bằng "Tuy nhiên", "Theo đó", "Điều này"... (827 câu trên `tune`)
  mà câu đứng trước không được chọn. Giai đoạn 1 thử kéo kèm câu đứng trước (`noi_tien_de`)
  nhưng thua về chỉ số nên tắt. Giai đoạn 2: câu mở đầu bằng TỪ NỐI bỏ được thì bỏ từ nối
  (`bo_noi`); câu treo còn lại bị trừ điểm khi chọn (`w_treo`). Không dùng mô hình sinh.
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
# Giai doan 2: tu noi BO DUOC ma khong doi nghia su kien — chi mat sac thai noi cau. Chi bo khi
# ngay sau la dau phay, de khong cat nham "Cu the hoa...". CO Y GIU (khong bo, chi tru diem):
# "truoc do", "sau do", "trong khi do", "khi do", "luc do" — tieng Viet khong chia thi, nhieu khi
# day la dau thoi gian duy nhat: "Truoc do, tren manh dat co 4 doanh nghiep" bo di thanh noi
# ve hien tai, tuc SAI. "Trong do", "tuong tu", "cung theo" va dai tu tro vao cau truoc.
TU_NOI_BO_DUOC = ("tuy nhiên", "ngoài ra", "bên cạnh đó", "như vậy", "cụ thể", "do đó", "vì vậy",
                  "vì thế", "theo đó", "hơn nữa", "thậm chí")
BO_NOI = re.compile(r'^(\s*["“]?\s*)(?:' + "|".join(TU_NOI_BO_DUOC) + r")\s*,\s*", re.I)
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


def bo_tu_noi(t):
    """Bỏ từ nối bỏ được ở đầu câu, viết hoa chữ đầu phần còn lại; không khớp thì trả nguyên."""
    m = BO_NOI.match(t)
    if not m or m.end() >= len(t):
        return t
    con = t[m.end():]
    return m.group(1) + con[0].upper() + con[1:]


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
        # treo ma bo tu noi cung khong cuu duoc — doi tuong cua `w_treo`
        "treo_cung": np.array([la_treo(bo_tu_noi(t)) for _, t in dv]),
    }


def chon(b, ngan_sach, w_vt, w_lex, lam, noi_tien_de, loc_rac, w_treo=0.0):
    """Chỉ số các đơn vị được chọn, theo thứ tự bài. `w_treo` = 0 là đúng giai đoạn 1."""
    n = len(b["dv"])
    if not n:
        return []
    diem = b["p"] + w_vt * b["vt"] + w_lex * b["lex"] - w_treo * b["treo_cung"]
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


def van_ban(b, idx, bo_noi=False):
    return " ".join(bo_tu_noi(b["dv"][k][1]) if bo_noi else b["dv"][k][1] for k in idx)


def so_cau_treo(b, idx, bo_noi=False):
    """Số câu trong bản tóm tắt vẫn còn treo: mở đầu bằng từ nối/đại từ mà câu trước không có."""
    chon_ = set(idx)
    return sum(1 for k in idx
               if la_treo(bo_tu_noi(b["dv"][k][1]) if bo_noi else b["dv"][k][1]) and (k - 1) not in chon_)


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


GD1 = ("ngan_sach", "w_vt", "w_lex", "lam", "noi_tien_de", "loc_rac")


def chuan_bi_tune(khong_phobert=False):
    rows, scores = nap("tune")
    t0 = time.time()
    bai = [chuan_bi_bai(r["article"], None if khong_phobert else s) for r, s in zip(rows, scores)]
    sapo = []
    for r in rows:
        toks = syllables(for_scoring(r["abstract"]))
        sapo.append((chi_tiet(r["abstract"]), Counter(toks), len(toks)))
    print(f"Chuẩn bị {len(rows)} bài tune trong {time.time() - t0:.0f}s"
          + (" — ĐÃ TẮT điểm PhoBERT" if khong_phobert else ""))
    return rows, bai, sapo


def do_nhanh(bai, sapo, ten, ds_idx, bo_noi=False):
    phu, rc, am, cau, treo = [], [], [], [], 0
    for b, (ct, uni, n), idx in zip(bai, sapo, ds_idx):
        p, r, a = cham_nhanh(van_ban(b, idx, bo_noi), ct, uni, n)
        if p is not None:
            phu.append(p)
        rc.append(r)
        am.append(a)
        cau.append(len(idx))
        treo += so_cau_treo(b, idx, bo_noi) > 0
    return {"ten": ten, "phu": float(np.mean(phu)), "recall": float(np.mean(rc)),
            "am": float(np.mean(am)), "cau_tb": float(np.mean(cau)), "cau_max": int(max(cau)),
            "ban_co_treo": treo}


def cmd_do(args):
    rows, bai, sapo = chuan_bi_tune(args.khong_phobert)

    def do(ten, ds_idx):
        return do_nhanh(bai, sapo, ten, ds_idx)

    moc = [do("Lead-3", [list(range(min(3, len(b["dv"])))) for b in bai])]
    if not args.khong_phobert:
        k3 = [sorted(np.argsort(-b["p"], kind="stable")[:3].tolist()) if b["p"].any()
              else list(range(min(3, len(b["dv"])))) for b in bai]
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
    ten_file = "chon_cau_tune_do_khong-phobert.json" if args.khong_phobert else "chon_cau_tune_do.json"
    dest = RESULTS / "tables" / ten_file
    dest.write_text(json.dumps({"moc": moc, "cau_hinh": kq, "thang": hop_le[0] if hop_le else None},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


def cmd_so_phobert(args):
    """PhoBERT góp bao nhiêu: cấu hình thắng CÓ điểm PhoBERT so với cấu hình thắng khi TẮT nó.

    Công bằng cho bên không có PhoBERT: nó được dò lại cả lưới, chứ không chỉ tắt điểm ở cấu hình
    vốn được chọn cho bên có PhoBERT. Chấm bằng bộ chấm chính thức, so cặp trên cùng 500 bài `tune`.
    """
    from eval.chinh_xac import cham
    from eval.stats import paired_bootstrap

    def thang(ten_file):
        return json.loads((RESULTS / "tables" / ten_file).read_text(encoding="utf-8"))["thang"]

    co, khong = thang("chon_cau_tune_do.json"), thang("chon_cau_tune_do_khong-phobert.json")
    rows, scores = nap("tune")
    per = {"co": [], "khong": []}
    for r, s in zip(rows, scores):
        for ten, th, diem in (("co", co, s), ("khong", khong, None)):
            b = chuan_bi_bai(r["article"], diem)
            per[ten].append(cham(van_ban(b, chon(b, *[th[k] for k in GD1])), r["abstract"], r["article"]))
    kq = {"co_phobert": {k: co[k] for k in GD1}, "khong_phobert": {k: khong[k] for k in GD1}}
    for m in ("r1_recall", "do_phu_chi_tiet"):
        cap = [(a[m], b[m]) for a, b in zip(per["co"], per["khong"]) if a[m] is not None and b[m] is not None]
        x, y = [a for a, _ in cap], [b for _, b in cap]
        kq[m] = {"co": float(np.mean(x)), "khong": float(np.mean(y)), "n": len(cap), **paired_bootstrap(x, y)}
        print(f"{m:16s} có {kq[m]['co']:5.2f} | không {kq[m]['khong']:5.2f} | hiệu {kq[m]['diff']:+.2f} "
              f"[{kq[m]['lo']:+.2f}, {kq[m]['hi']:+.2f}] p = {kq[m]['p']:.4f}")
    dest = RESULTS / "tables" / "chon_cau_phobert_dong_gop.json"
    dest.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {dest}")


# Quy tac chon cua giai doan 2, CHOT TRUOC KHI DO: trong cac cau hinh giu duoc chi so (phu chi
# tiet va recall moi thu giam khong qua 0,5 diem so voi giai doan 1 tren tune, van <= 110 am
# tiet va <= 4 cau), lay cau hinh co IT BAN CON CAU TREO nhat; hoa thi lay w_treo nho hon, roi
# khong bo tu noi (don gian hon).
DUNG_SAI = 0.5
W_TREO = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)


def cmd_do_treo(args):
    rows, bai, sapo = chuan_bi_tune()
    th = json.loads((RESULTS / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))["thang"]
    cfg = [th[k] for k in GD1]
    kq = []
    for w, bo in itertools.product(W_TREO, (False, True)):
        m = do_nhanh(bai, sapo, "", [chon(b, *cfg, w_treo=w) for b in bai], bo)
        m.update({k: th[k] for k in GD1}, w_treo=w, bo_noi=bo)
        kq.append(m)
        print(f"  w_treo {w:4.2f} bỏ từ nối {bo!s:5s} | phủ {m['phu']:5.2f} | recall {m['recall']:5.2f} | "
              f"{m['am']:5.1f} âm tiết | {m['cau_max']} câu tối đa | bản có câu treo {m['ban_co_treo']}")
    goc = kq[0]
    hop_le = [m for m in kq if m["am"] <= NGAN_SACH_TB and m["cau_max"] <= TOI_DA_CAU
              and m["phu"] >= goc["phu"] - DUNG_SAI and m["recall"] >= goc["recall"] - DUNG_SAI]
    hop_le.sort(key=lambda m: (m["ban_co_treo"], m["w_treo"], m["bo_noi"]))
    th2 = hop_le[0]
    print(f"\nGiai đoạn 1: {goc['ban_co_treo']} bản có câu treo. Thắng: w_treo {th2['w_treo']}, "
          f"bỏ từ nối {th2['bo_noi']} → {th2['ban_co_treo']} bản "
          f"(phủ {th2['phu'] - goc['phu']:+.2f}, recall {th2['recall'] - goc['recall']:+.2f})")
    dest = RESULTS / "tables" / "chon_cau_gd2_tune_do.json"
    dest.write_text(json.dumps({"dung_sai": DUNG_SAI, "giai_doan_1": goc, "cau_hinh": kq, "thang": th2},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {dest}")


def cmd_sinh(args):
    """Sinh bản tóm tắt bằng cấu hình THẮNG trên `tune` — đọc từ file dò, không gõ tay."""
    if args.split == "test" and not args.cho_phep_test:
        raise SystemExit("Từ chối sinh trên `test`: tập này dùng MỘT lần. Thêm --cho-phep-test khi chấm lần cuối.")
    if args.gd == 1:
        th = json.loads((RESULTS / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))["thang"]
        th, ten = dict(th, w_treo=0.0, bo_noi=False), "chon-cau"
    else:
        th = json.loads((RESULTS / "tables" / "chon_cau_gd2_tune_do.json").read_text(encoding="utf-8"))["thang"]
        ten = "chon-cau-gd2"
    cfg = {k: th[k] for k in GD1 + ("w_treo", "bo_noi")}
    tag = f"{ten}_{args.split}"
    dest = RESULTS / "predictions" / f"{tag}.json"
    if dest.exists():
        raise SystemExit(f"{dest} đã có — không đè. Xoá tay nếu thật sự muốn sinh lại.")
    rows, scores = nap(args.split)
    t0 = time.time()
    preds, treo = [], 0
    for r, s in zip(rows, scores):
        b = chuan_bi_bai(r["article"], s)
        idx = chon(b, *[cfg[k] for k in GD1], w_treo=cfg["w_treo"])
        preds.append(van_ban(b, idx, cfg["bo_noi"]))
        treo += so_cau_treo(b, idx, cfg["bo_noi"]) > 0
    rong = sum(1 for p in preds if not p.strip())
    dest.write_text(json.dumps({"guid": [str(r["guid"]) for r in rows],
                                "reference": [r["abstract"] for r in rows], ten: preds},
                               ensure_ascii=False), encoding="utf-8")
    run = {"tag": tag, "split": args.split, "cau_hinh": cfg, "chon_tren": "tune",
           "diem_phobert": f"phobert-sent-train_20k_{args.split}_len256_scores.json",
           "giay": round(time.time() - t0, 1), "ban_rong": rong}
    if args.gd == 2:
        run["ban_co_cau_treo"] = treo
    (RESULTS / "tables" / f"{tag}_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Cấu hình (thắng trên tune): {cfg}")
    print(f"Sinh {len(preds)} bản {args.split} trong {time.time() - t0:.0f}s, {rong} bản rỗng, "
          f"{treo} bản có câu treo. Đã ghi {dest}")


# Bon thanh phan cua diem chon cau; moi cai la MOT TRUC cua luoi 216 cau hinh da do tren
# `tune`, nen "do lai luoi khi tat mot thanh phan" chi la loc lai file cu chu khong phai chay
# lai. PhoBERT KHONG nam trong danh sach nay: tat no doi chinh diem tung cau nen phai do bang
# mot lan chay rieng (`do --khong-phobert`) — xem `cmd_so_phobert`.
# Moi muc: (ten, {truc: gia tri khi TAT}, nhan). Muc cuoi tat CA BA trong so cung luc — can
# thiet vi ba thu nay thay the duoc cho nhau (vi tri va LexRank deu la diem cho MOI cau, ke ca
# cau nam ngoai cua so 256 token cua PhoBERT), nen tat tung cai mot se luon ra so nho.
THANH_PHAN = (("w_vt", {"w_vt": 0.0}, "ưu tiên vị trí"),
              ("w_lex", {"w_lex": 0.0}, "độ trung tâm LexRank"),
              ("lam", {"lam": 0.0}, "thưởng phủ ý mới"),
              ("loc_rac", {"loc_rac": False}, "lọc rác, ghép mảnh câu"),
              ("ba_trong_so", {"w_vt": 0.0, "w_lex": 0.0, "lam": 0.0}, "cả ba trọng số cùng tắt"))


def cmd_so_thanh_phan(args):
    """Tung thanh phan gop bao nhieu vao he thong cuoi, do tren `tune`.

    Cong bang cho ben bi tat, dung cach da dung o `so-phobert`: ben khong co thanh phan duoc
    DO LAI ca luoi (lay cau hinh tot nhat TRONG SO cac cau hinh tat thanh phan ay) chu khong
    phai chi tat no o cau hinh von duoc chon cho ben day du. Neu khong se tinh vong vao phan
    ma cac thanh phan con lai bu duoc — tuc phong dai dong gop.

    Cham bang bo cham chinh thuc `eval.chinh_xac.cham`, so cap bootstrap tren cung 500 bai
    `tune`. Khong dung `val` hay `test`.
    """
    from eval.chinh_xac import cham
    from eval.stats import paired_bootstrap

    luoi = json.loads((RESULTS / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))
    hop_le = [m for m in luoi["cau_hinh"] if m["am"] <= NGAN_SACH_TB and m["cau_max"] <= TOI_DA_CAU]
    day_du = max(hop_le, key=lambda m: m["muc_tieu"])
    if [day_du[k] for k in GD1] != [luoi["thang"][k] for k in GD1]:
        raise SystemExit("Cấu hình tốt nhất lọc lại không trùng `thang` đã ghi — dừng, đừng đọc bảng này")

    rows, scores = nap("tune")
    t0 = time.time()
    bai = [chuan_bi_bai(r["article"], s) for r, s in zip(rows, scores)]

    def cham_cau_hinh(th):
        return [cham(van_ban(b, chon(b, *[th[k] for k in GD1])), r["abstract"], r["article"])
                for b, r in zip(bai, rows)]

    nen = cham_cau_hinh(day_du)
    kq = {"day_du": {k: day_du[k] for k in GD1}, "n_hop_le": len(hop_le), "n_bai": len(rows),
          "thanh_phan": {}}
    for khoa, tat_gt, nhan in THANH_PHAN:
        ung = [m for m in hop_le if all(m[k] == v for k, v in tat_gt.items())]
        if not ung:
            raise SystemExit(f"Lưới không có cấu hình nào tắt `{khoa}` — không dò lại được")
        tat = max(ung, key=lambda m: m["muc_tieu"])
        per = cham_cau_hinh(tat)
        muc = {"nhan": nhan, "cau_hinh_tat": {k: tat[k] for k in GD1}, "n_ung_vien": len(ung)}
        for m in ("r1_recall", "do_phu_chi_tiet"):
            cap = [(a[m], b[m]) for a, b in zip(nen, per) if a[m] is not None and b[m] is not None]
            x, y = [a for a, _ in cap], [b for _, b in cap]
            muc[m] = {"co": float(np.mean(x)), "khong": float(np.mean(y)), "n": len(cap),
                      **paired_bootstrap(x, y)}
        kq["thanh_phan"][khoa] = muc
        r, f = muc["r1_recall"], muc["do_phu_chi_tiet"]
        print(f"  {nhan:22s} recall {r['diff']:+5.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p = {r['p']:.4f}"
              f" | phủ {f['diff']:+5.2f} [{f['lo']:+.2f}, {f['hi']:+.2f}] p = {f['p']:.4f}")
    kq["giay"] = round(time.time() - t0, 1)
    dest = RESULTS / "tables" / "chon_cau_thanh_phan_dong_gop.json"
    dest.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {dest}")


def main():
    ap = argparse.ArgumentParser(description="Hướng mới, giai đoạn 1 và 2: chọn câu có chủ đích.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("do", help="giai đoạn 1: dò cấu hình trên tune")
    d.add_argument("--ngan-sach", type=int, nargs="+", default=[90, 100, 110])
    d.add_argument("--khong-phobert", action="store_true", help="đối chứng: tắt điểm PhoBERT, dò lại cả lưới")
    d.set_defaults(fn=cmd_do)
    sub.add_parser("so-phobert", help="giai đoạn 1: PhoBERT góp bao nhiêu (tune)").set_defaults(fn=cmd_so_phobert)
    sub.add_parser("so-thanh-phan", help="giai đoạn 1: từng thành phần góp bao nhiêu (tune)").set_defaults(fn=cmd_so_thanh_phan)
    sub.add_parser("do-treo", help="giai đoạn 2: dò xử lý câu treo trên tune").set_defaults(fn=cmd_do_treo)
    s = sub.add_parser("sinh", help="sinh bản tóm tắt bằng cấu hình thắng trên tune")
    s.add_argument("--split", required=True, choices=["tune", "val", "test"])
    s.add_argument("--gd", type=int, default=1, choices=[1, 2], help="1: chọn câu; 2: thêm xử lý câu treo")
    s.add_argument("--cho-phep-test", action="store_true", help="MỞ khoá `test` — chỉ ở lần chấm cuối")
    s.set_defaults(fn=cmd_sinh)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
