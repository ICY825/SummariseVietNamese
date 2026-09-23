"""Thước đo phủ ý: bản tóm tắt lấy được bao nhiêu ý chính của bài.

    .venv/Scripts/python.exe src/eval/phu_y.py chuan-bi     # một lần: phiếu liệt kê ý
    .venv/Scripts/python.exe src/eval/phu_y.py phieu-phu    # sau khi đã liệt kê ý
    .venv/Scripts/python.exe src/eval/phu_y.py phan-tich    # sau khi đã đánh dấu phủ

**Vì sao cần thêm một thước đo nữa.** Tuần 7 và giai đoạn 3 đã chấm `day_du` theo thang
1–5. Hai chỗ thang ấy không trả lời được:

1. *Độ trôi giữa hai đợt chấm.* Hệ thống cuối chấm ở đợt mới, BARTpho và Lead-3 chấm ở
   tuần 7; trôi `day_du` là −0,55, vượt ngưỡng 0,25 đã chốt trước, nên mọi so sánh chéo
   đợt ở tiêu chí này **không được viết thành kết luận** (README, giai đoạn 3). Thước đo
   ở đây chấm mọi hệ thống trong **một lượt trên cùng danh sách ý**, nên không có đợt để
   mà trôi.
2. *Thang 1–5 không nói ý nào bị bỏ.* "3 điểm đầy đủ" không cho biết hệ thống rụng ý ở
   đầu bài hay cuối bài, rụng ý chính hay ý phụ. Đếm theo từng ý thì biết, và biết rồi
   mới sửa được.

Thước đo này cũng nhắm vào **mâu thuẫn chưa giải** đã ghi trong README: chỉ số tự động
nói hệ thống cuối phủ chi tiết hơn Lead-3, còn máy chấm nói kém `day_du` hơn −0,25.

**Hai pha, và vì sao phải tách.** Danh sách ý viết ở pha 1 khi người gán **chưa nhìn thấy
bản tóm tắt nào**. Nhìn trước rồi mới liệt kê thì danh sách ý có xu hướng uốn theo bản
mình có thiện cảm, và thước đo mất giá trị. `phieu-phu` từ chối chạy khi `gan_y.json`
chưa có đủ ý cho mọi bài.

**Không chốt được: tính blind.** Người gán nhận ra hệ thống qua hình thức văn bản —
extractive chép nguyên câu, BARTpho viết một câu ngắn, sapo là tít dẫn. Nhãn vẫn xáo
riêng từng bài để không có "A luôn là Lead-3", nhưng đây là che nhãn chứ không phải chấm
mù thật sự. Phải nêu trong báo cáo, cùng với việc ai là người gán.

**Dùng lại mẫu tuần 7.** Cùng 50 bài `val` và cùng mã bài B01–B50, nên điểm phủ ý ghép
thẳng theo bài với `day_du` đã có. Mặc định chỉ lấy 12 bài mà **người** đã chấm ở tuần 7,
để đối chiếu được với điểm người chứ không chỉ điểm máy; `--tat-ca` lấy đủ 50 bài.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import for_scoring  # noqa: E402
from eval.human_eval import krippendorff_alpha  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "human_eval" / "phu_y"
KHOA_TUAN7 = RESULTS / "human_eval" / "khoa.json"
MAU_TUAN7 = RESULTS / "human_eval" / "mau.json"
SEED = 13

# ten ngan -> (file trong results/predictions, ten cot trong file do)
HE_THONG = {
    "sapo": ("baselines_val", "reference"),
    "lead3": ("baselines_val", "Lead-3"),
    "bartpho": ("bartpho-syllable-train_20k_val_in1024", "bartpho-syllable-train_20k"),
    "chon-cau": ("chon-cau_val", "chon-cau"),
}

QUY_TAC = [
    "Đọc bài gốc trước, và viết danh sách ý **trước khi** xem bất kỳ bản tóm tắt nào.",
    "Mỗi ý một dòng, dạng chủ thể – hành động – đối tượng.",
    "Một câu có thể chứa hai ý, và hai câu có thể cùng nói một ý. Đừng để danh sách ý "
    "biến thành danh sách câu.",
    "Chú thích ảnh, tít phụ và tên tác giả không phải nội dung bài — không tính là ý.",
    "Ý chỉ có trong sapo mà thân bài không nhắc thì không tính, vì không hệ thống trích "
    "rút nào lấy được.",
    "Một ý tính là PHỦ khi bản tóm tắt nói được phần cốt lõi (chủ thể và hành động); "
    "thiếu chi tiết phụ vẫn tính là phủ.",
    "Nói đúng ý nhưng gắn sai chi tiết (sai quận, sai giờ) vẫn tính là phủ — sai sự thật "
    "là việc của thước đo chi tiết lạ, không phải của thước đo này.",
]


# --------------------------------------------------------------------------
# Nap du lieu
# --------------------------------------------------------------------------

def _doc_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def nap_ban_tom_tat():
    """{tên hệ thống: {guid: bản tóm tắt}} — đọc thẳng từ results/predictions."""
    ra = {}
    for ten, (tep, cot) in HE_THONG.items():
        d = _doc_json(RESULTS / "predictions" / f"{tep}.json")
        if cot not in d:
            raise KeyError(f"{tep}.json không có cột {cot!r}; có: {list(d)}")
        ra[ten] = {str(g): s for g, s in zip(d["guid"], d[cot])}
    return ra


def chon_bai(tat_ca):
    """[(mã bài, guid)] — mẫu tuần 7, mặc định chỉ 12 bài người đã chấm."""
    khoa = _doc_json(KHOA_TUAN7)
    bai = [(b["ma_bai"], str(b["guid"])) for b in khoa["bai"]]
    if tat_ca:
        return bai
    ma_nguoi = set(_doc_json(MAU_TUAN7)["ma_bai"])
    return [x for x in bai if x[0] in ma_nguoi]


# --------------------------------------------------------------------------
# chuan-bi: phieu liet ke y (chi co bai goc)
# --------------------------------------------------------------------------

def cmd_chuan_bi(args):
    bai = chon_bai(args.tat_ca)
    ds = {str(r["guid"]): r for r in load_split("val")}
    thieu = [g for _, g in bai if g not in ds]
    if thieu:
        raise SystemExit(f"Không thấy {len(thieu)} guid trong split val: {thieu[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)

    dong = ["# Phiếu liệt kê ý — pha 1", "",
            "Viết danh sách ý cho từng bài **trước khi** xem bất kỳ bản tóm tắt nào.", ""]
    dong += [f"{i}. {q}" for i, q in enumerate(QUY_TAC, 1)] + [""]
    for ma, g in bai:
        dong += [f"## {ma} (guid {g})", "", for_scoring(ds[g]["article_raw"]), ""]
    (OUT / "phieu_y.md").write_text("\n".join(dong), encoding="utf-8")

    mau = {ma: {"guid": g, "y": [], "phu": {}} for ma, g in bai}
    tep = OUT / "gan_y.json"
    if tep.exists():
        cu = _doc_json(tep)
        for ma in mau:
            if ma in cu:
                mau[ma] = cu[ma]                       # giu nguyen phan da gan
    tep.write_text(json.dumps(mau, ensure_ascii=False, indent=1), encoding="utf-8")

    (OUT / "mau.json").write_text(json.dumps(
        {"split": "val", "seed": SEED, "n": len(bai), "tat_ca": bool(args.tat_ca),
         "he_thong": list(HE_THONG), "tu_mau_tuan7": str(KHOA_TUAN7.relative_to(ROOT)),
         "quy_tac": QUY_TAC, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(bai)} bài -> {OUT/'phieu_y.md'} và {OUT/'gan_y.json'}")


# --------------------------------------------------------------------------
# phieu-phu: phieu danh dau phu y (y + cac ban tom tat gan nhan)
# --------------------------------------------------------------------------

def _nhan(ma, tens):
    """Nhãn A, B, C… xáo riêng từng bài với hạt giống cố định: không có 'A luôn là Lead-3'."""
    import random

    ten = sorted(tens)
    random.Random(f"{SEED}-{ma}").shuffle(ten)
    return {chr(ord("A") + i): t for i, t in enumerate(ten)}


def cmd_phieu_phu(args):
    gan = _doc_json(OUT / "gan_y.json")
    chua = [ma for ma, v in gan.items() if not v["y"]]
    if chua:
        raise SystemExit(
            f"{len(chua)} bài chưa có danh sách ý ({', '.join(chua[:6])}…). "
            "Pha 1 phải xong trước: nhìn bản tóm tắt rồi mới liệt kê ý thì danh sách ý "
            "sẽ uốn theo bản mình có thiện cảm.")
    tt = nap_ban_tom_tat()
    ds = {str(r["guid"]): r for r in load_split("val")}

    khoa, dong = {}, ["# Phiếu đánh dấu phủ ý — pha 2", "",
                      "Với mỗi bản, đánh dấu ý nào bản đó nói được. Quy tắc:", ""]
    dong += [f"- {q}" for q in QUY_TAC[5:]] + [""]
    for ma, v in gan.items():
        g = v["guid"]
        nhan = _nhan(ma, [t for t in HE_THONG if g in tt[t]])
        khoa[ma] = {"guid": g, "nhan": nhan}
        dong += [f"## {ma} (guid {g})", "", "**Bài gốc.** " + for_scoring(ds[g]["article_raw"]),
                 "", "**Ý chính.**", ""]
        dong += [f"{i}. {y}" for i, y in enumerate(v["y"], 1)] + [""]
        for n, t in nhan.items():
            dong += [f"**Bản {n}.** " + for_scoring(tt[t][g]), ""]
    (OUT / "phieu_phu.md").write_text("\n".join(dong), encoding="utf-8")
    (OUT / "khoa.json").write_text(json.dumps(
        {"seed": SEED, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "bai": khoa},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(gan)} bài -> {OUT/'phieu_phu.md'} (khoá ở {OUT/'khoa.json'})")


# --------------------------------------------------------------------------
# phan-tich: ghep nhan voi he thong, ra bang
# --------------------------------------------------------------------------

def _diem(gan, khoa):
    """{hệ thống: [tỷ lệ phủ từng bài]} và [mã bài] — theo cùng một thứ tự bài."""
    ma_bai = [ma for ma in gan if gan[ma]["phu"]]
    diem = {t: [] for t in HE_THONG}
    for ma in ma_bai:
        v, nhan = gan[ma], khoa["bai"][ma]["nhan"]
        n = len(v["y"])
        for n_nhan, he in nhan.items():
            dau = v["phu"].get(n_nhan)
            if dau is None:
                raise SystemExit(f"{ma}: thiếu đánh dấu cho bản {n_nhan}.")
            if len(dau) != n:
                raise SystemExit(
                    f"{ma} bản {n_nhan}: {len(dau)} dấu nhưng bài có {n} ý.")
            diem[he].append(100 * sum(bool(x) for x in dau) / n)
    return {t: v for t, v in diem.items() if v}, ma_bai


def cmd_phan_tich(args):
    gan, khoa = _doc_json(OUT / "gan_y.json"), _doc_json(OUT / "khoa.json")
    diem, ma_bai = _diem(gan, khoa)
    if not diem:
        raise SystemExit("Chưa bài nào được đánh dấu phủ — chạy pha 2 trước.")
    so_y = [len(gan[ma]["y"]) for ma in ma_bai]

    bang = {"n_bai": len(ma_bai), "ma_bai": ma_bai,
            "so_y": {"tong": sum(so_y), "tb": sum(so_y) / len(so_y),
                     "min": min(so_y), "max": max(so_y)},
            "he_thong": {}, "so_cap": {}}
    for t, v in diem.items():
        m, lo, hi = bootstrap_ci(v, seed=SEED)
        bang["he_thong"][t] = {"phu_y": round(m, 1), "lo": round(lo, 1), "hi": round(hi, 1)}
    # Cap nao co ket luan can kiem: he thong cuoi so voi tung moc, va so voi sapo.
    for a, b in (("chon-cau", "lead3"), ("chon-cau", "bartpho"),
                 ("lead3", "bartpho"), ("chon-cau", "sapo")):
        if a in diem and b in diem:
            r = paired_bootstrap(diem[a], diem[b], seed=SEED)
            bang["so_cap"][f"{a} - {b}"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in r.items()}
    tep = RESULTS / "tables" / "phu_y_val.json"
    tep.write_text(json.dumps(bang, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(ma_bai)} bài, trung bình {bang['so_y']['tb']:.1f} ý/bài "
          f"(thấp nhất {bang['so_y']['min']}, cao nhất {bang['so_y']['max']})\n")
    print(f"{'hệ thống':12}{'phủ ý (%)':>12}   khoảng tin cậy 95%")
    for t, v in sorted(bang["he_thong"].items(), key=lambda x: -x[1]["phu_y"]):
        print(f"{t:12}{v['phu_y']:>12.1f}   [{v['lo']:.1f}, {v['hi']:.1f}]")
    print()
    for k, v in bang["so_cap"].items():
        sao = "có ý nghĩa" if v["significant"] else "chưa phân biệt được với 0"
        print(f"  {k:24} {v['diff']:+6.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]  p={v['p']:.4f}  {sao}")
    print(f"\n-> {tep}")


# --------------------------------------------------------------------------
# phieu-nguoi / so-sanh: nguoi gan doc lap, roi do do dong thuan
# --------------------------------------------------------------------------

def _tep_nguoi(i):
    return OUT / f"cham_phu_y_nguoi{i}.csv"


def cmd_phieu_nguoi(args):
    """Phiếu CSV trống để người gán đánh dấu phủ, dùng lại đúng danh sách ý đã có.

    Người gán dùng **cùng danh sách ý** nên độ đồng thuận đo đúng bước đánh dấu, tách khỏi
    bước liệt kê ý. Muốn kiểm luôn bước liệt kê thì viết danh sách ý riêng rồi so tay —
    hai bước hỏng theo hai kiểu khác nhau và không nên trộn vào một con số.
    """
    gan = _doc_json(OUT / "gan_y.json")
    chua = [ma for ma, v in gan.items() if not v["y"]]
    if chua:
        raise SystemExit(f"{len(chua)} bài chưa có danh sách ý: {', '.join(chua[:6])}…")
    tep = _tep_nguoi(args.nguoi)
    if tep.exists() and not args.ghi_de:
        raise SystemExit(f"{tep} đã có. Dùng --ghi-de nếu thật sự muốn viết đè.")
    nhan = sorted({n for v in _doc_json(OUT / "khoa.json")["bai"].values() for n in v["nhan"]})
    with tep.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ma_bai", "stt", "y"] + nhan + ["ghi_chu"])
        for ma, v in gan.items():
            for i, y in enumerate(v["y"], 1):
                w.writerow([ma, i, y] + [""] * len(nhan) + [""])
    print(tep)
    print(f"  Điền 1 (bản đó nói được ý này) hoặc 0 vào các cột {', '.join(nhan)}.")
    print(f"  Đọc bài và các bản ở {OUT/'phieu_phu.md'}.")
    print("  KHÔNG mở gan_y.json — trong đó có sẵn bản đánh dấu của lượt trước.")


def _doc_phieu_nguoi(tep, gan):
    """{mã bài: {nhãn: [đánh dấu]}} — chỉ những bài đã điền xong.

    Bài để trống hoàn toàn thì bỏ qua, để còn làm dở vài bài rồi đo thử trước khi bỏ công
    làm hết. Nhưng bài điền DỞ DANG thì báo lỗi: một ô bỏ sót lặng lẽ tính thành "không phủ"
    sẽ hạ điểm của đúng hệ thống mà dòng đó rơi vào.
    """
    hang = list(csv.DictReader(tep.open(encoding="utf-8-sig")))
    nhan = [c for c in hang[0] if c not in ("ma_bai", "stt", "y", "ghi_chu")]
    theo_bai = {}
    for h in hang:
        theo_bai.setdefault(h["ma_bai"], []).append(h)

    ra = {}
    for ma, hs in theo_bai.items():
        o = [(h[n] or "").strip() for h in hs for n in nhan]
        if not any(o):
            continue                                   # chua lam bai nay
        for h in hs:
            for n in nhan:
                x = (h[n] or "").strip()
                if x not in ("0", "1"):
                    raise SystemExit(
                        f"{tep.name} — {ma} ý {h['stt']} cột {n}: {x!r} (phải là 0 hoặc 1). "
                        "Bài đã điền dở thì phải điền cho hết, hoặc xoá trắng cả bài.")
                ra.setdefault(ma, {}).setdefault(n, []).append(x == "1")
        for n, dau in ra[ma].items():
            if len(dau) != len(gan[ma]["y"]):
                raise SystemExit(f"{tep.name} — {ma} cột {n}: {len(dau)} dòng, cần {len(gan[ma]['y'])}.")
    if not ra:
        raise SystemExit(f"{tep.name} chưa điền bài nào.")
    return ra


def cmd_so_sanh(args):
    gan, khoa = _doc_json(OUT / "gan_y.json"), _doc_json(OUT / "khoa.json")
    tep = _tep_nguoi(args.nguoi)
    if not tep.exists():
        raise SystemExit(f"Chưa có {tep}. Chạy `phieu-nguoi` rồi điền vào đó trước.")
    nguoi = _doc_phieu_nguoi(tep, gan)

    # Do dong thuan tren TUNG luot danh dau: moi (bai, nhan, y) la mot don vi hai gia tri.
    don_vi, trung, tong = [], 0, 0
    for ma, v in nguoi.items():
        for n, dau in v.items():
            for i, x in enumerate(dau):
                y = bool(gan[ma]["phu"][n][i])
                don_vi.append([float(x), float(y)])
                trung += x == y
                tong += 1
    alpha = krippendorff_alpha(don_vi)

    # Ket luan co doi chieu khong: tinh lai bang cua nguoi gan doc lap.
    diem_n = {t: [] for t in HE_THONG}
    for ma, v in nguoi.items():
        for n, dau in v.items():
            diem_n[khoa["bai"][ma]["nhan"][n]].append(100 * sum(dau) / len(dau))
    diem_m, _ = _diem({ma: gan[ma] for ma in nguoi}, khoa)

    print(f"{tong} lượt đánh dấu trên {len(nguoi)} bài")
    print()
    print(f"  trùng nhau           {100*trung/tong:.1f}%")
    print(f"  Krippendorff alpha   {alpha:.3f}" if alpha is not None
          else "  Krippendorff alpha   không tính được (mọi lượt đánh dấu giống nhau)")
    print("  (chuẩn đã dùng ở tuần 7: alpha >= 0,667 thì coi là đủ đồng thuận)")
    print()
    print(f"{'hệ thống':12}{'người gán':>12}{'lượt trước':>12}{'chênh':>9}")
    for t in sorted(diem_n, key=lambda k: -(sum(diem_n[k]) / len(diem_n[k]) if diem_n[k] else 0)):
        if not diem_n[t]:
            continue
        a = sum(diem_n[t]) / len(diem_n[t])
        b = sum(diem_m[t]) / len(diem_m[t])
        print(f"{t:12}{a:>11.1f}%{b:>11.1f}%{a-b:>+9.1f}")
    print()
    for a, b in (("chon-cau", "lead3"), ("chon-cau", "bartpho"), ("chon-cau", "sapo")):
        if diem_n[a] and diem_n[b]:
            r = paired_bootstrap(diem_n[a], diem_n[b], seed=SEED)
            cu = paired_bootstrap(diem_m[a], diem_m[b], seed=SEED)
            doi = "GIU NGUYEN" if r["significant"] == cu["significant"] else "**ĐỔI CHIỀU**"
            print(f"  {a} - {b:10} người {r['diff']:+6.1f} (p={r['p']:.4f}) | "
                  f"lượt trước {cu['diff']:+6.1f} (p={cu['p']:.4f}) -> kết luận {doi}")
    tep_ra = RESULTS / "tables" / "phu_y_nguoi_vs_may_val.json"
    tep_ra.write_text(json.dumps(
        {"n_bai": len(nguoi), "n_luot": tong, "trung_nhau": round(100 * trung / tong, 1),
         "alpha": alpha, "nguoi": {t: round(sum(v) / len(v), 1) for t, v in diem_n.items() if v},
         "luot_truoc": {t: round(sum(v) / len(v), 1) for t, v in diem_m.items() if v}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"-> {tep_ra}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    s = p.add_subparsers(dest="lenh", required=True)
    c = s.add_parser("chuan-bi", help="dựng phiếu liệt kê ý (pha 1)")
    c.add_argument("--tat-ca", action="store_true",
                   help="lấy đủ 50 bài tuần 7 thay vì 12 bài người đã chấm")
    c.set_defaults(fn=cmd_chuan_bi)
    s.add_parser("phieu-phu", help="dựng phiếu đánh dấu phủ ý (pha 2)").set_defaults(fn=cmd_phieu_phu)
    s.add_parser("phan-tich", help="ghép nhãn với hệ thống, ra bảng").set_defaults(fn=cmd_phan_tich)
    n = s.add_parser("phieu-nguoi", help="phiếu CSV trống cho người gán độc lập")
    n.add_argument("--nguoi", type=int, default=1, help="người gán thứ mấy (mặc định 1)")
    n.add_argument("--ghi-de", action="store_true", help="viết đè phiếu đã có")
    n.set_defaults(fn=cmd_phieu_nguoi)
    c = s.add_parser("so-sanh", help="đo đồng thuận giữa người gán và lượt gán trước")
    c.add_argument("--nguoi", type=int, default=1)
    c.set_defaults(fn=cmd_so_sanh)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
