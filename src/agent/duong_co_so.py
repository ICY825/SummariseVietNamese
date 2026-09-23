"""Giai đoạn 6b: chạy các hệ thống cũ trên ngữ liệu tin mới, lấy mốc trước khi có agent.

    ~/.venvs/demo/Scripts/python.exe src/agent/duong_co_so.py --gioi-han 3   # thử
    ~/.venvs/demo/Scripts/python.exe src/agent/duong_co_so.py                # cả 40 bài

**Vì sao phải làm bước này trước khi dựng agent.** Không có mốc thì sau này không ai biết
agent hơn được cái gì. Và vì ngữ liệu mới là tin bây giờ còn BARTpho huấn luyện trên tin
2016–2018, bước này còn trả lời luôn một câu hỏi riêng: mô hình chịu được dịch chuyển miền
tới đâu.

**Dùng lại đúng đường của báo cáo.** Các tầng gọi qua `app/pipeline.py` — cùng hàm mà demo
và bảng `test` đã dùng — nên số ở đây so thẳng được với `results/tables/chinh_xac_test.json`,
không phải một đường chạy riêng dựng lại.

**Sapo làm tham chiếu, và đã biết trước nó yếu.** Giai đoạn 5 đo được sapo chỉ phủ 15,7% số
ý của thân bài, nên ROUGE với sapo **không** đo được "đủ ý". Ở đây vẫn chấm ROUGE để so với
bảng cũ, nhưng thước đo chính cho hướng agent phải là phủ ý gán tay (giai đoạn 6c trở đi).

Ghi `results/predictions/tin_moi.json` (bản tóm tắt từng hệ thống) và
`results/tables/chinh_xac_tin_moi.json` (điểm từng bài + tổng hợp, cùng khuôn bảng cũ).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import pipeline as P  # noqa: E402
from eval.chinh_xac import cham  # noqa: E402
from eval.rouge import score  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402

TIN = ROOT / "data" / "tin_moi" / "tin.json"
RESULTS = ROOT / "results"

# Ten he thong giu NGUYEN nhu trong bang `test`, de ghep hai bang lai la so duoc ngay.
HE_THONG = ["Lead-1", "Lead-3", "LexRank", "chon-cau", "bartpho-syllable-train_20k"]
# Cung danh sach va cung thu tu voi cham_chinh_xac.py, de hai bang ghep lai duoc.
CHI_SO = ["r1_recall", "r1_f1", "r2_recall", "do_phu_chi_tiet", "so_chi_tiet_la",
          "co_chi_tiet_la", "so_cau", "am_tiet"]


def _sinh(tho, seg):
    """{tên hệ thống: bản tóm tắt} cho một bài. Lỗi của một tầng không làm hỏng cả lượt chạy."""
    ra = {}
    for ten, ham in (
        ("Lead-1", lambda: P.tang01(seg, "lead", k=1)),
        ("Lead-3", lambda: P.tang01(seg, "lead", k=3)),
        ("LexRank", lambda: P.tang01(seg, "lexrank", k=3)),
        ("chon-cau", lambda: P.he_thong_cuoi(seg)),
        ("bartpho-syllable-train_20k", lambda: P.tang3(tho)),
    ):
        try:
            ra[ten] = ham()
        except Exception as e:
            ra[ten] = ""
            print(f"      [{ten}] LỖI {type(e).__name__}: {e}")
    return ra


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gioi-han", type=int, help="chỉ chạy N bài đầu (để thử)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--cham-lai", action="store_true",
                    help="chấm lại từ bản đã sinh, không chạy lại mô hình")
    a = ap.parse_args()

    tin = json.loads(TIN.read_text(encoding="utf-8"))
    if a.gioi_han:
        tin = tin[: a.gioi_han]
    print(f"{len(tin)} bài\n")

    ban = {t["ma"]: {} for t in tin}
    t0 = time.time()
    if a.cham_lai:
        # Sinh lai mat ~18 phut CPU. Ban tom tat da luu roi, chi can tach tu lai de co `seg`.
        pr = json.loads((RESULTS / "predictions" / "tin_moi.json").read_text(encoding="utf-8"))
        theo_ma = {g: i for i, g in enumerate(pr["guid"])}
        for t in tin:
            j = theo_ma[t["ma"]]
            ban[t["ma"]] = {ten: pr[ten][j] for ten in HE_THONG}
            ban[t["ma"]]["_seg"] = P.tach_tu(t["than_bai"])
        print(f"  đọc lại {len(tin)} bài đã sinh, tách từ lại trong {time.time()-t0:.0f}s")
    for i, t in enumerate([] if a.cham_lai else tin, 1):
        # Tach tu bang underthesea: dung khau ma demo dung cho bai nguoi dung dan vao.
        # Khac bo tach tu cua VietNews, va khac biet do phai neu khi so voi bang cu.
        seg, _ = P.chuan_bi(t["than_bai"])
        ban[t["ma"]] = _sinh(t["than_bai"], seg)
        ban[t["ma"]]["_seg"] = seg
        print(f"  [{i:2}/{len(tin)}] {t['ma']} {t['bao']:11} {time.time()-t0:6.0f}s  {t['tieu_de'][:48]}")

    ket_qua = []
    for ten in HE_THONG:
        per, ma_bai = {k: [] for k in CHI_SO}, []
        for t in tin:
            du_doan = ban[t["ma"]].get(ten, "")
            if not du_doan:
                continue
            m = cham(du_doan, t["sapo"], ban[t["ma"]]["_seg"])
            m["r1_f1"] = score(du_doan, t["sapo"])["rouge1"]
            for k in CHI_SO:
                per[k].append(m[k])
            ma_bai.append(t["ma"])
        if not ma_bai:
            continue
        corpus = {}
        for k in CHI_SO:
            # `do_phu_chi_tiet` la None khi sapo khong co chi tiet nao de doi chieu -- bo di,
            # khong thi trung binh thanh nan. `co_chi_tiet_la` nhan 100 cho ra don vi phan tram,
            # dung nhu cham_chinh_xac.py, neu khong hai bang khac thang ma nhin nhu giong nhau.
            vals = [v for v in per[k] if v is not None]
            if k == "co_chi_tiet_la":
                vals = [100 * v for v in vals]
            m, lo, hi = bootstrap_ci(vals, n_boot=a.n_boot, seed=13)
            corpus[k] = {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}
        ket_qua.append({"name": ten, "n": len(ma_bai), "guid": ma_bai,
                        "per_article": per, "corpus": corpus})

    (RESULTS / "tables" / "chinh_xac_tin_moi.json").write_text(
        json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / "predictions" / "tin_moi.json").write_text(json.dumps(
        {"guid": [t["ma"] for t in tin], "reference": [t["sapo"] for t in tin],
         **{ten: [ban[t["ma"]].get(ten, "") for t in tin] for ten in HE_THONG}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    cot = ("r1_recall", "r1_f1", "r2_recall", "do_phu_chi_tiet", "co_chi_tiet_la", "so_cau", "am_tiet")
    print(f"\n{'hệ thống':30}" + "".join(f"{c:>17}" for c in cot))
    for r in ket_qua:
        print(f"{r['name']:30}" + "".join(f"{r['corpus'][c]['mean']:>17.1f}" for c in cot))
    print(f"\n-> results/tables/chinh_xac_tin_moi.json và results/predictions/tin_moi.json")


if __name__ == "__main__":
    main()
