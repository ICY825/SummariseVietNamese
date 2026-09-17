"""Hướng mới, giai đoạn 3: chấm blind hệ thống cuối, DÙNG LẠI điểm tuần 7.

    .venv/Scripts/python.exe src/eval/cham_gd3.py prepare     # một lần: phiếu bản mới + bản mốc
    .venv/Scripts/python.exe src/eval/cham_gd3.py analyze     # sau khi thu đủ phiếu

**Dùng lại gì.** Tuần 7 đã chấm Lead-3, PhoBERT `k2`, BARTpho và sapo trên 50 bài `val`
(`results/human_eval/`): máy chấm hai lượt đủ 50 bài, hai người chấm 12 bài. Giai đoạn 3 giữ
nguyên 50 bài và mã bài B01–B50, và chỉ chấm **bản mới**: bản giai đoạn 1 (`gd1`) và giai đoạn
2 (`gd2`) của từng bài, trừ khi bản đó **trùng từng chữ** với một bản đã chấm ở tuần 7 — khi ấy
điểm cũ dùng thẳng. gd1 và gd2 trùng chữ nhau thì chỉ một nhãn, như tuần 7.

**Bản mốc — đo độ trôi giữa hai đợt chấm.** Điểm BARTpho chấm từ tuần 7, điểm hệ thống cuối chấm
bây giờ: nếu đợt mới khắt khe hay dễ hơn đợt cũ, độ lệch đó lẫn vào hiệu hệ thống cuối − BARTpho.
Nên phiếu mới trộn **10 bản đã chấm ở tuần 7** (5 BARTpho, 5 Lead-3; 6 trong số đó thuộc 12 bài
người chấm), chấm blind như mọi bản, rồi so với điểm cũ của chính chúng.

**Không chốt được:** mỗi bài giờ chỉ có 1–3 bản thay vì 3–4, nên người chấm không còn thấy các bản
kia để làm mốc ngầm. Bản mốc đo được đúng hệ quả của việc này.

`prepare` ghi vào `results/human_eval/gd3/` — cùng quy ước với tuần 7:

| File | Cho ai |
|---|---|
| `phieu_doc.html`, `phieu_phan<k>.md` | máy chấm — 50 bài (HTML) hoặc chia năm phần (Markdown) |
| `llm_judge_1.csv`, `llm_judge_2.csv` | hai lượt máy chấm, mỗi lượt một tác tử con độc lập |
| `phieu_doc_mau.html`, `cham_mau_nguoi<i>.csv` | người chấm — chỉ 12 bài của mẫu tuần 7 |
| `khoa.json` | **chỉ người phân tích** (không commit) |

**Phân tích — chốt TRƯỚC khi có điểm** (README, giai đoạn 3):

1. *Độ trôi*: trên 10 bản mốc, điểm mới (trung bình hai lượt máy) − điểm cũ (trung bình hai lượt
   máy tuần 7), theo tiêu chí. Ghép điểm hai đợt được coi là hợp lệ cho tiêu chí đó khi |trôi
   trung bình| ≤ 0,25 **và** alpha mới–cũ ≥ 0,667. Không đạt thì so sánh chéo đợt ở tiêu chí ấy
   chỉ được nêu kèm độ trôi, không được viết là kết luận.
2. *Cú đảo chiều*: trên 50 bài, so cặp theo bài gd1 − BARTpho, gd1 − Lead-3, gd1 − `k2` (và gd2
   như vậy), ba tiêu chí, điểm máy trung bình hai lượt, bootstrap ghép cặp.
3. *Chọn gd1/gd2*: trên các bài gd1 ≠ gd2, chọn gd2 khi `troi_chay` gd2 − gd1 > 0 có ý nghĩa **và**
   cận dưới khoảng tin cậy 95% của `day_du` và `trung_thuc` gd2 − gd1 đều > −0,25; ngược lại gd1.
   Tính trên điểm máy (người chấm có quá ít bài gd1 ≠ gd2) — `troi_chay` của máy chưa được kiểm
   chứng ở tuần 7, phải nêu kèm. Mẫu nhỏ không đủ bằng chứng thì mặc định gd1 và viết "chưa chứng
   minh được gd2 trôi chảy hơn", không viết "gd2 không trôi chảy hơn".
4. *Người chấm* (12 bài): cùng các so sánh ở mục 2 trên điểm người, để kiểm **chiều**; và người
   so với máy trên các bản mới.
"""

import argparse
import csv
import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring  # noqa: E402
from eval import human_eval as H  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

OUT = H.OUT / "gd3"
DEST = H.RESULTS / "tables" / "cham_gd3_val.json"
MOI = {"gd1": ("chon-cau_val", "chon-cau"), "gd2": ("chon-cau-gd2_val", "chon-cau-gd2")}
SEED = f"{H.SEED}-gd3"
MOC = {"bartpho": (3, 2), "lead3": (3, 2)}  # (so ban trong 12 bai mau, so ban ngoai mau)
TRUOT_TOI_DA, ALPHA_TOI_THIEU, DUNG_SAI = 0.25, 0.667, 0.25
PHAN = 5


def bai_tuan7():
    """50 bài tuần 7 dựng lại, kiểm trùng khít phiếu và khoá đã phát."""
    khoa = json.loads((H.OUT / "khoa.json").read_text(encoding="utf-8"))
    bai = H.build_bai(khoa["systems"], khoa["n"])
    if (H.OUT / "phieu_doc.html").read_text(encoding="utf-8") != H.render_html(bai):
        raise SystemExit("Dựng lại 50 bài tuần 7 không ra đúng phieu_doc.html — không dùng lại điểm được.")
    if [(b["ma_bai"], b["guid"], {x["nhan"]: x["he_thong"] for x in b["nhan"]}) for b in bai] != \
            [(b["ma_bai"], b["guid"], b["nhan"]) for b in khoa["bai"]]:
        raise SystemExit("Dựng lại 50 bài tuần 7 không khớp khoa.json.")
    mau = json.loads((H.OUT / "mau.json").read_text(encoding="utf-8"))
    return khoa, bai, set(mau["ma_bai"])


def ban_moi():
    out = {}
    for ten, (tag, key) in MOI.items():
        d = json.loads((H.RESULTS / "predictions" / f"{tag}.json").read_text(encoding="utf-8"))
        out[ten] = dict(zip(map(str, d["guid"]), d[key]))
    return out


def dung_phieu():
    """[bài của phiếu mới], {(mã bài, hệ mới): hệ tuần 7 trùng chữ} — tất định."""
    khoa7, bai7, mau = bai_tuan7()
    moi = ban_moi()
    rng = random.Random(SEED)
    # Ban moc: chon truoc, trong cac ban cu KHONG trung chu voi ban moi cua cung bai
    ung_vien = {(s, trong): [] for s in MOC for trong in (True, False)}
    for b in bai7:
        tt_moi = {for_scoring(moi[h][b["guid"]]) for h in MOI}
        for x in b["nhan"]:
            for s in MOC:
                if s in x["he_thong"] and x["van_ban"] not in tt_moi:
                    ung_vien[(s, b["ma_bai"] in mau)].append((b["ma_bai"], s, x["van_ban"]))
    moc = []
    for s, (n_trong, n_ngoai) in MOC.items():
        moc += rng.sample(ung_vien[(s, True)], n_trong) + rng.sample(ung_vien[(s, False)], n_ngoai)
    moc_theo_bai = {}
    for mb, s, t in moc:
        moc_theo_bai.setdefault(mb, []).append((s, t))

    phieu, dung_lai = [], {}
    for b in bai7:
        cu = {x["van_ban"]: x["he_thong"] for x in b["nhan"]}
        texts = {}
        for h in MOI:
            t = for_scoring(moi[h][b["guid"]])
            if t in cu:
                dung_lai[(b["ma_bai"], h)] = cu[t]
            else:
                texts.setdefault(t, []).append(h)
        for s, t in moc_theo_bai.get(b["ma_bai"], []):
            texts.setdefault(t, []).append(f"moc:{s}")
        if not texts:
            continue
        items = sorted(texts.items())
        random.Random(f"{SEED}-{b['guid']}").shuffle(items)
        phieu.append({**{k: b[k] for k in ("ma_bai", "guid", "tieu_de", "bai_goc")},
                      "nhan": [{"nhan": chr(ord("A") + i), "van_ban": t, "he_thong": sorted(hs)}
                               for i, (t, hs) in enumerate(items)]})
    return phieu, dung_lai, mau


def xuat_markdown(phieu):
    """Chia phiếu thành PHAN file Markdown cho máy chấm — cùng hướng dẫn, bảng mức, quy tắc."""
    dau = ["# Phiếu chấm tóm tắt — hướng dẫn", "",
           "Mỗi bài có một bài gốc và một đến ba bản tóm tắt gắn nhãn A, B, C. Chấm **từng nhãn** theo "
           "thang 1–5 (5 là tốt nhất) cho ba tiêu chí.", ""]
    dau += [f"- **{k}**: {v}" for k, v in H.CRITERIA.items()] + ["", "## Quy tắc chung", ""]
    dau += [f"{i}. {r.replace('<b>', '**').replace('</b>', '**').replace('<code>', '`').replace('</code>', '`')}"
            for i, r in enumerate(H.RULES, 1)]
    dau += ["", "## Mô tả từng mức điểm", "", "| Điểm | " + " | ".join(H.CRITERIA) + " |",
            "|---|" + "---|" * len(H.CRITERIA)]
    dau += [f"| {m} | " + " | ".join(H.RUBRIC[c][m] for c in H.CRITERIA) + " |" for m in range(5, 0, -1)]
    k = -(-len(phieu) // PHAN)
    for p in range(PHAN):
        phan = phieu[p * k:(p + 1) * k]
        dong = dau + [""]
        for b in phan:
            dong += ["---", "", f"## {b['ma_bai']}" + (f" — {b['tieu_de']}" if b["tieu_de"] else ""), "",
                     "**Bài gốc:**", "", b["bai_goc"], ""]
            dong += [f"**{x['nhan']}.** {x['van_ban']}" + "\n" for x in b["nhan"]]
        (OUT / f"phieu_phan{p + 1}.md").write_text("\n".join(dong), encoding="utf-8")


def cmd_prepare(args):
    khoa_path = OUT / "khoa.json"
    if khoa_path.exists():
        raise SystemExit(f"{khoa_path} đã có. Phiếu có thể đã phát — xoá tay nếu thật sự muốn rút lại.")
    phieu, dung_lai, mau = dung_phieu()
    _, bai7_goc, _ = bai_tuan7()
    moi = ban_moi()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phieu_doc.html").write_text(H.render_html(phieu), encoding="utf-8")
    xuat_markdown(phieu)
    for luot in (1, 2):
        with open(OUT / f"llm_judge_{luot}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ma_bai", "nhan", *H.CRITERIA, "ghi_chu"])
            for b in phieu:
                for x in b["nhan"]:
                    w.writerow([b["ma_bai"], x["nhan"], "", "", "", ""])
    sub = [b for b in phieu if b["ma_bai"] in mau]
    (OUT / "phieu_doc_mau.html").write_text(H.render_html(sub), encoding="utf-8")
    H_OUT = H.OUT
    H.OUT = OUT  # write_sheets ghi vao H.OUT
    try:
        H.write_sheets(sub, "cham_mau_nguoi", 2)
    finally:
        H.OUT = H_OUT
    khoa = {
        "tu_khoa_tuan7": "results/human_eval/khoa.json", "seed": SEED, "moi": MOI, "moc": MOC,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phieu_sha256": H.sha256(OUT / "phieu_doc.html"), "mau_sha256": H.sha256(OUT / "phieu_doc_mau.html"),
        "ma_bai_mau": sorted(mau),
        "gd1_khac_gd2": [b["ma_bai"] for b in bai7_goc if moi["gd1"][b["guid"]] != moi["gd2"][b["guid"]]],
        "dung_lai": [{"ma_bai": mb, "he": h, "he_tuan7": hs} for (mb, h), hs in sorted(dung_lai.items())],
        "bai": [{"ma_bai": b["ma_bai"], "guid": b["guid"], "nhan": {x["nhan"]: x["he_thong"] for x in b["nhan"]}}
                for b in phieu],
    }
    khoa_path.write_text(json.dumps(khoa, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(b["nhan"]) for b in phieu)
    n_mau = sum(len(b["nhan"]) for b in sub)
    print(f"Phiếu máy: {len(phieu)} bài, {n} bản (gồm {sum(MOC[s][0] + MOC[s][1] for s in MOC)} bản mốc); "
          f"{len(dung_lai)} bản mới dùng lại điểm tuần 7.")
    print(f"Phiếu người: {len(sub)} bài, {n_mau} bản mỗi người.")
    print(f"Đã ghi {OUT} (KHÔNG gửi khoa.json cho người chấm).")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _doc(thu_muc, mau_ten, khoa, chi_bai=None):
    out = {}
    for p in sorted(thu_muc.glob(mau_ten)):
        d, loi = H.read_sheet(p, khoa, chi_bai)
        if loi:
            raise SystemExit(f"{p.name}: {len(loi)} lỗi — {loi[:5]}")
        out[p.stem] = d
    if not out:
        raise SystemExit(f"Không có phiếu {mau_ten} trong {thu_muc}.")
    return out


def _tb(phieu, ma_bai, nhan, c):
    v = [phieu[p][(ma_bai, nhan)][c] for p in phieu if (ma_bai, nhan) in phieu[p]]
    return float(np.mean(v)) if v else np.nan


def bang_diem(khoa7, khoa, cu, moi, chi_bai=None):
    """{he: {tieu_chi: {ma_bai: diem trung binh cac luot}}} cho he tuan 7, gd1, gd2 va ban moc."""
    diem = {}
    for b in khoa7["bai"]:
        if chi_bai is not None and b["ma_bai"] not in chi_bai:
            continue
        for lab, hs in b["nhan"].items():
            for h in hs:
                for c in H.CRITERIA:
                    diem.setdefault(h, {}).setdefault(c, {})[b["ma_bai"]] = _tb(cu, b["ma_bai"], lab, c)
    for x in khoa["dung_lai"]:
        if chi_bai is not None and x["ma_bai"] not in chi_bai:
            continue
        for c in H.CRITERIA:
            diem.setdefault(x["he"], {}).setdefault(c, {})[x["ma_bai"]] = diem[x["he_tuan7"][0]][c][x["ma_bai"]]
    for b in khoa["bai"]:
        if chi_bai is not None and b["ma_bai"] not in chi_bai:
            continue
        for lab, hs in b["nhan"].items():
            for h in hs:
                for c in H.CRITERIA:
                    diem.setdefault(h, {}).setdefault(c, {})[b["ma_bai"]] = _tb(moi, b["ma_bai"], lab, c)
    return diem


def so_cap(diem, a, b, c, bai=None):
    chung = sorted(set(diem[a][c]) & set(diem[b][c]) if bai is None else bai)
    x = [diem[a][c][m] for m in chung]
    y = [diem[b][c][m] for m in chung]
    return {"n": len(chung), **paired_bootstrap(x, y)}


def cmd_analyze(args):
    khoa7 = json.loads((H.OUT / "khoa.json").read_text(encoding="utf-8"))
    khoa = json.loads((OUT / "khoa.json").read_text(encoding="utf-8"))
    if H.sha256(OUT / "phieu_doc.html") != khoa["phieu_sha256"]:
        raise SystemExit("gd3/phieu_doc.html đã khác bản lúc phát.")
    kq = {"quy_tac": {"truot_toi_da": TRUOT_TOI_DA, "alpha_toi_thieu": ALPHA_TOI_THIEU, "dung_sai": DUNG_SAI}}

    may_cu = _doc(H.OUT, "llm_judge*.csv", khoa7)
    may_moi = _doc(OUT, "llm_judge_*.csv", khoa)
    D = bang_diem(khoa7, khoa, may_cu, may_moi)
    print(f"Máy chấm: tuần 7 {list(may_cu)} | giai đoạn 3 {list(may_moi)}")

    # 1. Do troi tren ban moc
    print("\n1. Độ trôi giữa hai đợt chấm — bản mốc, điểm mới − điểm cũ (trung bình hai lượt máy):")
    kq["do_troi"] = {}
    for c in H.CRITERIA:
        cap = [(D[f"moc:{s}"][c][mb], D[s][c][mb]) for s in MOC for mb in D[f"moc:{s}"][c]]
        lech = np.array([m - o for m, o in cap])
        a = H.krippendorff_alpha([[m, o] for m, o in cap])
        ok = bool(abs(lech.mean()) <= TRUOT_TOI_DA and a is not None and a >= ALPHA_TOI_THIEU)
        kq["do_troi"][c] = {"n": len(cap), "troi_tb": float(lech.mean()), "alpha_moi_cu": a,
                            "lech_toi_da_1": float(np.mean(np.abs(lech) <= 1)), "ghep_hop_le": ok}
        print(f"  {c:10s} n={len(cap)} trôi {lech.mean():+.2f} | alpha mới–cũ {H._fmt(a)} | "
              f"lệch ≤1 {np.mean(np.abs(lech) <= 1):.0%} | ghép hai đợt: {'HỢP LỆ' if ok else 'KHÔNG'}")

    # 2. Cu dao chieu
    print("\n2. Hệ thống cuối so với các hệ tuần 7 — 50 bài, điểm máy:")
    he = ["sapo", "lead3", "tang2_k2", "bartpho", "gd1", "gd2"]
    kq["he_thong"] = {h: {c: dict(zip(("mean", "lo", "hi"), bootstrap_ci(list(D[h][c].values()))))
                          for c in H.CRITERIA} for h in he}
    for h in he:
        print(f"  {h:9s} " + " | ".join(f"{c} {kq['he_thong'][h][c]['mean']:.2f}" for c in H.CRITERIA))
    kq["so_cap"] = {}
    for a_ in ("gd1", "gd2"):
        for b_ in ("bartpho", "lead3", "tang2_k2"):
            for c in H.CRITERIA:
                r = so_cap(D, a_, b_, c)
                kq["so_cap"][f"{a_} − {b_} {c}"] = r
                canh_bao = "" if kq["do_troi"][c]["ghep_hop_le"] else "  (trôi hai đợt KHÔNG đạt — chỉ tham khảo)"
                print(f"  {a_} − {b_:8s} {c:10s} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                      f"p={r['p']:.4f}{canh_bao}")

    # 3. Chon gd1 / gd2
    khac = khoa["gd1_khac_gd2"]
    kq["chon"] = {c: so_cap(D, "gd2", "gd1", c, khac) for c in H.CRITERIA}
    t, d, tt = kq["chon"]["troi_chay"], kq["chon"]["day_du"], kq["chon"]["trung_thuc"]
    chon_gd2 = bool(t["diff"] > 0 and t["significant"] and d["lo"] > -DUNG_SAI and tt["lo"] > -DUNG_SAI)
    kq["chon"]["bai"], kq["chon"]["ket_qua"] = khac, "gd2" if chon_gd2 else "gd1"
    print(f"\n3. Chọn phiên bản — {len(khac)} bài có gd1 ≠ gd2, gd2 − gd1 (điểm máy):")
    for c in H.CRITERIA:
        r = kq["chon"][c]
        print(f"  {c:10s} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")
    print(f"  → theo quy tắc đã chốt: {kq['chon']['ket_qua']}")

    # 4. Nguoi cham
    mau = set(khoa["ma_bai_mau"])
    try:
        nguoi_cu = _doc(H.OUT, "cham_mau_nguoi*.csv", khoa7, mau)
        nguoi_moi = _doc(OUT, "cham_mau_nguoi*.csv", khoa, mau)
    except SystemExit as e:
        print(f"\n4. Người chấm: chưa đủ phiếu ({e}) — bỏ qua phần này.")
    else:
        N = bang_diem(khoa7, khoa, nguoi_cu, nguoi_moi, mau)
        kq["nguoi"] = {"do_troi": {}, "so_cap": {}}
        print(f"\n4. Người chấm, {len(mau)} bài:")
        for c in H.CRITERIA:
            cap = [(N[f"moc:{s}"][c][mb], N[s][c][mb]) for s in MOC for mb in N.get(f"moc:{s}", {}).get(c, {})]
            if cap:
                lech = np.array([m - o for m, o in cap])
                kq["nguoi"]["do_troi"][c] = {"n": len(cap), "troi_tb": float(lech.mean())}
                print(f"  độ trôi {c:10s} n={len(cap)} {lech.mean():+.2f}")
        for b_ in ("bartpho", "lead3", "tang2_k2"):
            for c in H.CRITERIA:
                rn = so_cap(N, "gd1", b_, c)
                rm = so_cap(D, "gd1", b_, c, sorted(mau))
                cung = bool(np.sign(round(rn["diff"], 6)) == np.sign(round(rm["diff"], 6)))
                kq["nguoi"]["so_cap"][f"gd1 − {b_} {c}"] = {"nguoi": rn, "may": rm, "cung_chieu": cung}
                print(f"  gd1 − {b_:8s} {c:10s} người {rn['diff']:+.2f} [{rn['lo']:+.2f}, {rn['hi']:+.2f}] | "
                      f"máy {rm['diff']:+.2f} | {'cùng chiều' if cung else 'NGƯỢC CHIỀU'}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DEST.write_text(json.dumps(kq, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\nĐã ghi {DEST}")


def main():
    global OUT, DEST
    ap = argparse.ArgumentParser(description="Giai đoạn 3: chấm blind hệ thống cuối, dùng lại điểm tuần 7.")
    ap.add_argument("--thu-muc", help="chạy thử: ghi phiếu/khoá/kết quả vào thư mục này, không đụng results/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="dựng phiếu bản mới + bản mốc").set_defaults(fn=cmd_prepare)
    sub.add_parser("analyze", help="ghép điểm, đo độ trôi, so sánh, chọn phiên bản").set_defaults(fn=cmd_analyze)
    args = ap.parse_args()
    if args.thu_muc:
        OUT = Path(args.thu_muc)
        DEST = OUT / "cham_gd3_thu.json"
    args.fn(args)


if __name__ == "__main__":
    main()
