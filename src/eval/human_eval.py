"""Tuần 7: chấm blind — dựng phiếu, phân tích phiếu, và kiểm chứng máy chấm bằng người chấm.

    .venv/Scripts/python.exe src/eval/human_eval.py prepare            # một lần: phiếu 50 bài
    .venv/Scripts/python.exe src/eval/human_eval.py analyze            # sau khi thu phiếu
    .venv/Scripts/python.exe src/eval/human_eval.py prepare-mau        # một lần: phiếu mẫu 12 bài
    .venv/Scripts/python.exe src/eval/human_eval.py so-sanh            # người chấm mẫu so với máy chấm

`prepare` ghi vào `results/human_eval/`:

| File | Cho ai |
|---|---|
| `phieu_doc.html` | người chấm — bài gốc và các bản tóm tắt gắn nhãn A, B, C, … |
| `cham_nguoi<i>.csv` | người chấm thứ i — mỗi dòng một (bài, nhãn), điền ba cột điểm |
| `khoa.json` | **chỉ người phân tích** — nhãn nào là hệ thống nào |

**Vì sao hệ thống nào có mặt.** Mỗi hệ thống trả lời một câu hỏi nghiên cứu: Lead-3 là
mốc extractive ngây thơ; tầng 2 `k2` là extractive tốt nhất; BARTpho là abstractive tốt
nhất — cặp này trả lời câu hỏi 1 ("vượt ở khía cạnh nào"). Sapo tham chiếu được trộn vào
như một hệ thống ẩn danh để có trần của thang điểm người chấm. ViT5 thêm được bằng
`--systems`, nhưng mặc định bỏ vì cùng vai trò với BARTpho mà làm tải người chấm tăng 25%.

**Ba chốt chặn cho tính blind.**

- Mọi văn bản đi qua `for_scoring()`: bản extractive còn gạch dưới thì người chấm nhận ra
  ngay (xem README, mục Đánh giá).
- Thứ tự nhãn xáo **riêng từng bài** với hạt giống cố định, nên không có "A luôn là Lead-3".
- Hai hệ thống ra **cùng một chữ** ở một bài thì chỉ hiện một nhãn, và nhãn đó trỏ tới cả
  hai trong `khoa.json`. Hiện hai bản giống hệt nhau vừa tốn công chấm vừa để lộ rằng có
  hai hệ thống trùng nhau.

Không chốt được: **độ dài**. Lead-3 dài gấp ba sapo, người chấm tinh ý đoán được. Phải nêu
trong báo cáo.

`prepare` từ chối chạy khi `khoa.json` đã có: rút mẫu lại sau khi đã phát phiếu là mất
khớp giữa phiếu trong tay người chấm và khoá.

**Kiểm chứng máy chấm bằng một mẫu người chấm.** Phiếu 50 bài được chấm bằng mô hình ngôn
ngữ (`llm_judge*.csv`), vì không có người chấm đủ 50 bài. `prepare-mau` rút 12 bài trong
chính 50 bài ấy và giữ nguyên mã bài lẫn nhãn A–D, nên điểm người và điểm máy ghép thẳng
theo (bài, nhãn). Trước khi rút, nó dựng lại 50 bài và đòi trùng khít phiếu đã phát — mẫu
phải là một phần của đúng phiếu máy đã chấm, không phải một phiếu gần giống. `so-sanh` đo
người–người, người–máy, và kiểm xem các kết luận rút ra từ máy có cùng chiều khi người chấm.
"""

import argparse
import csv
import hashlib
import html
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
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
OUT = RESULTS / "human_eval"
SPLIT = "val"
SEED = 13
REFERENCE = "sapo"
# ten ngan -> (tag trong results/predictions va results/tables, ten he thong trong file)
SYSTEMS = {
    REFERENCE: ("baselines_val", "reference"),
    "lead3": ("baselines_val", "Lead-3"),
    "tang2_k2": ("phobert-sent-train_20k_val_len256_k2", "phobert-sent-k2"),
    "vit5": ("vit5-base-train_20k_val_e3_lr3e-05_bs16_in1024", "vit5-base-train_20k"),
    "bartpho": ("bartpho-syllable-train_20k_val_e3_lr3e-05_bs16_in1024", "bartpho-syllable-train_20k"),
}
DEFAULT = ["sapo", "lead3", "tang2_k2", "bartpho"]
CRITERIA = {
    "day_du": "Đầy đủ — bản tóm tắt nắm được các ý chính của bài.",
    "trung_thuc": "Trung thực — mọi thông tin đều có trong bài gốc, không bịa, không sai.",
    "troi_chay": "Trôi chảy — đọc tự nhiên, đúng ngữ pháp, không lặp, không cụt.",
}
SCALE = (1, 5)
# Mo ta tung muc diem. Chi mot dong mo ta cho moi tieu chi thi moi nguoi hieu "3 diem" mot
# kieu va do dong thuan thap vi ly do khong lien quan toi ban tom tat.
RUBRIC = {
    "day_du": {
        5: "Nêu đủ các ý chính của bài (ai, việc gì, kết quả hay điều quan trọng nhất); không bỏ sót ý quan trọng.",
        4: "Đủ ý chính, chỉ thiếu một chi tiết phụ.",
        3: "Có ý chính nhưng thiếu một ý quan trọng, hoặc dành nhiều chỗ cho chi tiết phụ.",
        2: "Chỉ nắm được một phần nhỏ, bỏ sót phần lớn ý chính.",
        1: "Không nêu được ý chính của bài, hoặc chỉ nói về chuyện phụ.",
    },
    "trung_thuc": {
        5: "Mọi thông tin đều đúng và đều có trong bài gốc.",
        4: "Có một chỗ diễn đạt lại hơi lệch nhưng không làm sai nghĩa.",
        3: "Có một chi tiết phụ sai, hoặc có thông tin bài gốc không nhắc tới.",
        2: "Một chi tiết quan trọng sai (tên người, con số, thời gian, địa điểm), hoặc nhiều chi tiết phụ sai.",
        1: "Sai ý chính, hoặc phần lớn nội dung là bịa.",
    },
    "troi_chay": {
        5: "Đọc tự nhiên như người viết, đúng ngữ pháp.",
        4: "Đọc trôi, chỉ có một chỗ hơi vụng.",
        3: "Có chỗ lặp, câu cụt, hoặc các câu ghép rời rạc làm khó theo dõi.",
        2: "Nhiều chỗ khó hiểu, lặp hoặc đứt mạch.",
        1: "Gần như không đọc hiểu được.",
    },
}
RULES = [
    "Luôn đọc bài gốc trước. Mọi đánh giá so với <b>bài gốc</b>, không so các bản tóm tắt với nhau.",
    "Chấm từng bản <b>độc lập</b>: không xếp hạng, không cố cho các bản khác điểm nhau.",
    "<b>Không cộng điểm chỉ vì bản dài.</b> Một bản ngắn mà đủ ý chính vẫn được 5 điểm đầy đủ.",
    "Thông tin đúng ngoài đời nhưng <b>bài gốc không nhắc tới</b> vẫn bị trừ điểm trung thực.",
    "Ba tiêu chí chấm tách rời: một bản sai sự thật vẫn có thể được 5 điểm trôi chảy.",
    "Không đoán bản nào do máy hay người viết. Một số bài có ít hơn 4 bản — cứ chấm bình thường.",
    "Phân vân giữa hai mức thì chọn mức mô tả gần nhất; mức 2 và 4 dùng cho trường hợp nằm giữa.",
    "Cho điểm 1–2 ở tiêu chí nào thì ghi ngắn lý do vào cột <code>ghi_chu</code> (ví dụ: \"sai số tiền\", \"lặp ý\").",
]
# Cap he thong co ket luan chinh can kiem chieu khi nguoi cham (so-sanh)
KET_LUAN = [("bartpho", "lead3"), ("bartpho", "tang2_k2"), ("lead3", "tang2_k2")]


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def load_summaries(names):
    """{ten_ngan: [ban tom tat theo thu tu guid]} va danh sach guid chung, da kiem khop."""
    out, guids = {}, None
    for name in names:
        tag, key = SYSTEMS[name]
        d = json.loads((RESULTS / "predictions" / f"{tag}.json").read_text(encoding="utf-8"))
        g = [str(x) for x in d["guid"]]
        if guids is None:
            guids = g
        elif g != guids:
            raise SystemExit(f"{tag}: guid khác thứ tự với file đầu — không ghép được.")
        out[name] = d[key]
    return out, guids


def label_article(texts, rng):
    """Xáo thứ tự các hệ thống của MỘT bài, gộp bản trùng chữ; trả về [(nhan, van_ban, [he])]."""
    groups = {}
    for name, text in texts.items():
        groups.setdefault(text, []).append(name)
    items = list(groups.items())
    rng.shuffle(items)
    return [(chr(ord("A") + i), text, sorted(names)) for i, (text, names) in enumerate(items)]


def build_bai(names, n):
    """Rút `n` bài và gắn nhãn — tất định, nên gọi lại ra đúng những bài và nhãn ấy."""
    from data.splits import load_split

    summaries, guids = load_summaries(names)
    rows = {str(r["guid"]): r for r in load_split(SPLIT, add_raw=True)}
    if set(rows) != set(guids):
        raise SystemExit(f"File dự đoán không chấm trên đúng data/splits/{SPLIT}.json.")

    rng = random.Random(SEED)
    chon = sorted(rng.sample(range(len(guids)), n))
    bai = []
    for stt, i in enumerate(chon, 1):
        g = guids[i]
        # hat giong rieng tung bai (theo guid) de xao lai mot bai khong keo theo bai khac
        rng_bai = random.Random(f"{SEED}-{g}")
        texts = {name: for_scoring(summaries[name][i]) for name in names}
        if any(not t for t in texts.values()):
            raise SystemExit(f"Bài {g} có bản tóm tắt rỗng — không đưa vào phiếu được.")
        bai.append({
            "ma_bai": f"B{stt:02d}", "guid": g,
            "tieu_de": for_scoring(rows[g]["title"]) if rows[g].get("title") else "",
            "bai_goc": rows[g]["article_raw"],
            "nhan": [{"nhan": lab, "van_ban": t, "he_thong": hs} for lab, t, hs in label_article(texts, rng_bai)],
        })
    return bai


def write_sheets(bai, prefix, nguoi):
    """Phiếu điểm trống cho từng người chấm: `<prefix><i>.csv`."""
    for k in range(1, nguoi + 1):
        with open(OUT / f"{prefix}{k}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ma_bai", "nhan", *CRITERIA, "ghi_chu"])
            for b in bai:
                for x in b["nhan"]:
                    w.writerow([b["ma_bai"], x["nhan"], "", "", "", ""])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_prepare(args):
    names = args.systems
    bad = [n for n in names if n not in SYSTEMS]
    if bad:
        raise SystemExit(f"Không biết hệ thống {bad}. Chọn trong: {list(SYSTEMS)}")
    khoa_path = OUT / "khoa.json"
    if khoa_path.exists():
        raise SystemExit(f"{khoa_path} đã có. Phiếu có thể đã phát — xoá tay nếu thật sự muốn rút lại.")

    bai = build_bai(names, args.n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phieu_doc.html").write_text(render_html(bai), encoding="utf-8")
    write_sheets(bai, "cham_nguoi", args.nguoi)
    khoa = {
        "split": SPLIT, "seed": SEED, "n": args.n, "nguoi": args.nguoi, "systems": names,
        "sources": {n: SYSTEMS[n][0] for n in names}, "criteria": CRITERIA, "scale": SCALE,
        "rubric": RUBRIC, "rules": RULES,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # bam cua phieu doc: phan tich tu choi neu phieu da bi sua sau khi phat
        "phieu_sha256": sha256(OUT / "phieu_doc.html"),
        "bai": [{"ma_bai": b["ma_bai"], "guid": b["guid"],
                 "nhan": {x["nhan"]: x["he_thong"] for x in b["nhan"]}} for b in bai],
    }
    khoa_path.write_text(json.dumps(khoa, ensure_ascii=False, indent=1), encoding="utf-8")

    n_ban = sum(len(b["nhan"]) for b in bai)
    gop = sum(1 for b in bai for x in b["nhan"] if len(x["he_thong"]) > 1)
    print(f"{args.n} bài {SPLIT}, {len(names)} hệ thống, {n_ban} bản cần chấm mỗi người "
          f"({gop} nhãn gộp bản trùng chữ).")
    print(f"Đã ghi {OUT / 'phieu_doc.html'}, cham_nguoi1..{args.nguoi}.csv và khoa.json "
          "(KHÔNG gửi khoa.json cho người chấm).")


def cmd_prepare_mau(args):
    khoa_path, mau_path = OUT / "khoa.json", OUT / "mau.json"
    if not khoa_path.exists():
        raise SystemExit("Chưa có khoa.json — chạy `prepare` trước.")
    if mau_path.exists():
        raise SystemExit(f"{mau_path} đã có. Phiếu mẫu có thể đã phát — xoá tay nếu thật sự muốn rút lại.")
    khoa = json.loads(khoa_path.read_text(encoding="utf-8"))

    bai = build_bai(khoa["systems"], khoa["n"])
    # So van ban da doc (read_text quy doi xuong dong) chu khong so bam: tren Windows
    # write_text ghi \r\n, nen bam cua chuoi khong bang bam cua file.
    if (OUT / "phieu_doc.html").read_text(encoding="utf-8") != render_html(bai):
        raise SystemExit("Dựng lại 50 bài không ra đúng phieu_doc.html đã phát — không rút mẫu từ phiếu khác.")
    if sha256(OUT / "phieu_doc.html") != khoa["phieu_sha256"]:
        raise SystemExit("phieu_doc.html đã khác bản lúc phát phiếu — khoá không còn khớp.")
    if [(b["ma_bai"], b["guid"], {x["nhan"]: x["he_thong"] for x in b["nhan"]}) for b in bai] != \
            [(b["ma_bai"], b["guid"], b["nhan"]) for b in khoa["bai"]]:
        raise SystemExit("Dựng lại 50 bài không khớp khoa.json.")
    if not 1 <= args.n <= len(bai):
        raise SystemExit(f"--n phải trong 1..{len(bai)}.")

    rng = random.Random(f"{SEED}-mau")
    sub = [bai[i] for i in sorted(rng.sample(range(len(bai)), args.n))]
    (OUT / "phieu_doc_mau.html").write_text(render_html(sub), encoding="utf-8")
    write_sheets(sub, "cham_mau_nguoi", args.nguoi)
    mau = {
        "tu_khoa": "khoa.json", "seed": f"{SEED}-mau", "n": args.n, "nguoi": args.nguoi,
        "ma_bai": [b["ma_bai"] for b in sub],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phieu_sha256": sha256(OUT / "phieu_doc_mau.html"),
    }
    mau_path.write_text(json.dumps(mau, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ban = sum(len(b["nhan"]) for b in sub)
    print(f"Mẫu {args.n} bài {mau['ma_bai']}, {n_ban} bản mỗi người.")
    print(f"Đã ghi phieu_doc_mau.html, cham_mau_nguoi1..{args.nguoi}.csv và mau.json.")


def render_html(bai):
    e = html.escape
    tieu_chi = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in CRITERIA.items())
    muc = range(SCALE[1], SCALE[0] - 1, -1)
    bang = ("<table class='rubric'><tr><th>Điểm</th>" + "".join(f"<th>{e(c)}</th>" for c in CRITERIA) + "</tr>"
            + "".join(f"<tr><td><b>{m}</b></td>" + "".join(f"<td>{e(RUBRIC[c][m])}</td>" for c in CRITERIA)
                      + "</tr>" for m in muc) + "</table>")
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Phiếu đọc — chấm tóm tắt</title>",
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:60rem;margin:auto;padding:0 1rem}"
        "section{border-top:2px solid #444;margin-top:2.5rem}.goc{background:#f4f4f4;padding:.8rem}"
        ".tt{border-left:4px solid #888;padding:.2rem .8rem;margin:.8rem 0}"
        ".rubric{border-collapse:collapse;font-size:14px}.rubric td,.rubric th{border:1px solid #999;"
        "padding:.3rem .5rem;vertical-align:top}.rubric th{background:#e8e8e8}"
        ".nhac{font-size:13px;color:#555}"
        "@media print{section{page-break-before:always}}</style>",
        "<h1 id='huong-dan'>Phiếu đọc — hướng dẫn chấm</h1>",
        f"<p>Có {len(bai)} bài. Mỗi bài có một bài gốc và vài bản tóm tắt gắn nhãn A, B, C, …. Chấm "
        f"<b>từng nhãn</b> theo thang {SCALE[0]}–{SCALE[1]} ({SCALE[1]} là tốt nhất) cho ba tiêu chí, ghi "
        "vào file CSV của mình: mỗi dòng là một (bài, nhãn), cột <code>ma_bai</code> và <code>nhan</code> "
        "đã điền sẵn, chỉ điền số nguyên vào ba cột điểm.</p>",
        f"<ul>{tieu_chi}</ul>",
        "<h2>Quy tắc chung</h2>",
        "<ol>" + "".join(f"<li>{r}</li>" for r in RULES) + "</ol>",
        "<h2>Mô tả từng mức điểm</h2>",
        bang,
    ]
    for b in bai:
        parts.append(f"<section><h2>{e(b['ma_bai'])}{' — ' + e(b['tieu_de']) if b['tieu_de'] else ''}</h2>")
        parts.append("<p class='nhac'>Đọc bài gốc trước · so với bài gốc, không so các bản với nhau · "
                     "<a href='#huong-dan'>xem lại bảng mức điểm</a></p>")
        parts.append(f"<div class='goc'>{e(b['bai_goc'])}</div>")
        for x in b["nhan"]:
            parts.append(f"<div class='tt'><b>{e(x['nhan'])}.</b> {e(x['van_ban'])}</div>")
        parts.append("</section>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def krippendorff_alpha(units):
    """Krippendorff's alpha, dữ liệu khoảng; `units` là danh sách điểm của từng đơn vị chấm.

    Đơn vị có dưới hai điểm không ghép cặp được nên bị bỏ, đúng định nghĩa. Dùng khoảng
    chứ không thứ bậc vì thang 1–5 được phân tích bằng trung bình ở mọi chỗ khác.
    Trả về None khi mọi điểm bằng nhau (không có phương sai kỳ vọng để chia).
    """
    units = [np.asarray(u, dtype=float) for u in units if len(u) >= 2]
    if not units:
        return None
    # tong (a-b)^2 tren moi cap co thu tu i != j trong mot tap = 2m*sum(v^2) - 2*(sum v)^2
    pair = lambda v: 2 * v.size * float((v ** 2).sum()) - 2 * float(v.sum()) ** 2
    n = sum(u.size for u in units)
    d_o = sum(pair(u) / (u.size - 1) for u in units) / n
    d_e = pair(np.concatenate(units)) / (n * (n - 1))
    return None if d_e == 0 else 1.0 - d_o / d_e


def rankdata(x):
    """Hạng trung bình cho giá trị hoà — đủ cho Spearman, khỏi kéo scipy vào `.venv`."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a, b):
    """Kendall tau-b trên một nhóm nhỏ (các hệ thống của MỘT bài); None khi không xếp được."""
    conc = disc = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    den = ((conc + disc + ta) * (conc + disc + tb)) ** 0.5
    return None if den == 0 else (conc - disc) / den


def read_sheet(path, khoa, chi_bai=None):
    """{(ma_bai, nhan): {tieu_chi: diem}} của một người chấm, kiểm đủ và đúng thang.

    `chi_bai` giới hạn vào một tập mã bài: dòng của bài khác bị bỏ qua (không tính là lỗi),
    vì phiếu máy chấm đủ 50 bài được đọc lại chỉ ở phần trùng với phiếu mẫu.
    """
    raw = path.read_text(encoding="utf-8-sig")
    # Excel ban dia phuong Viet luu CSV bang dau cham phay
    dialect = csv.Sniffer().sniff(raw.splitlines()[0], delimiters=",;")
    rows = list(csv.DictReader(raw.splitlines(), dialect=dialect))
    can = {(b["ma_bai"], lab) for b in khoa["bai"] for lab in b["nhan"]
           if chi_bai is None or b["ma_bai"] in chi_bai}
    out, loi = {}, []
    for r in rows:
        k = (r["ma_bai"].strip(), r["nhan"].strip())
        if chi_bai is not None and k[0] not in chi_bai:
            continue
        if k not in can:
            loi.append(f"dòng lạ {k}")
            continue
        diem = {}
        for c in CRITERIA:
            v = (r.get(c) or "").strip()
            if not v.isdigit() or not SCALE[0] <= int(v) <= SCALE[1]:
                loi.append(f"{k} {c}={v!r}")
            else:
                diem[c] = int(v)
        out[k] = diem
    thieu = can - set(out)
    if thieu:
        loi.append(f"thiếu {len(thieu)} dòng, ví dụ {sorted(thieu)[:3]}")
    return out, loi


def per_article_metric(tag, name, metric, guids):
    """Điểm tự động từng bài theo `guid` từ results/tables; None nếu không tìm thấy."""
    if metric == "bertscore":
        for p in sorted((RESULTS / "tables").glob("bertscore_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for s in d["systems"]:
                if s["name"] == name and set(map(str, d["guid"])) >= set(guids):
                    m = dict(zip(map(str, d["guid"]), s["per_article"]["bertscore"]))
                    return [m[g] for g in guids]
        return None
    for s in json.loads((RESULTS / "tables" / f"{tag}.json").read_text(encoding="utf-8")):
        if s["name"] == name:
            m = dict(zip(map(str, s["guid"]), s["per_article"][metric]))
            return [m[g] for g in guids]
    return None


def cmd_analyze(args):
    khoa = json.loads((OUT / "khoa.json").read_text(encoding="utf-8"))
    if sha256(OUT / "phieu_doc.html") != khoa["phieu_sha256"]:
        raise SystemExit("phieu_doc.html đã khác bản lúc phát phiếu — khoá không còn khớp.")
    sheets = sorted(OUT.glob(args.phieu))
    if not sheets:
        raise SystemExit(f"Không có phiếu nào khớp {args.phieu} trong {OUT}.")
    diem, loi_tong = {}, 0
    for p in sheets:
        d, loi = read_sheet(p, khoa)
        if loi:
            loi_tong += len(loi)
            print(f"{p.name}: {len(loi)} lỗi — {loi[:5]}")
        diem[p.stem] = d
    if loi_tong and not args.bo_qua_loi:
        raise SystemExit("Phiếu còn ô trống hoặc sai thang. Sửa phiếu, hoặc chạy lại với --bo-qua-loi.")
    nguoi = list(diem)
    names, guids = khoa["systems"], [b["guid"] for b in khoa["bai"]]
    print(f"{len(nguoi)} người chấm, {len(guids)} bài, hệ thống {names}\n")

    # diem[he][tieu_chi] = ma tran (bai, nguoi); nhan gop nhieu he thi cung diem cho ca hai
    M = {n: {c: np.full((len(guids), len(nguoi)), np.nan) for c in CRITERIA} for n in names}
    units = {c: [] for c in CRITERIA}
    for i, b in enumerate(khoa["bai"]):
        for lab, hs in b["nhan"].items():
            for c in CRITERIA:
                vals = [diem[p][(b["ma_bai"], lab)][c] for p in nguoi
                        if c in diem[p].get((b["ma_bai"], lab), {})]
                units[c].append(vals)
                for j, p in enumerate(nguoi):
                    v = diem[p].get((b["ma_bai"], lab), {}).get(c)
                    for h in hs:
                        M[h][c][i, j] = np.nan if v is None else v

    ket_qua = {"nguoi": nguoi, "n_bai": len(guids), "systems": names, "he_thong": {}, "so_cap": {},
               "dong_thuan": {}, "cau_hoi_3": {}}
    print("Đồng thuận (Krippendorff's alpha, khoảng):")
    for c in CRITERIA:
        a = krippendorff_alpha(units[c])
        ket_qua["dong_thuan"][c] = a
        print(f"  {c:10s} {'—' if a is None else f'{a:.3f}'}")

    tb = {n: {c: np.nanmean(M[n][c], axis=1) for c in CRITERIA} for n in names}
    print("\nĐiểm trung bình theo bài (trung bình các người chấm), khoảng tin cậy bootstrap:")
    print("| Hệ thống | " + " | ".join(CRITERIA) + " |")
    print("|---|" + "---|" * len(CRITERIA))
    for n in names:
        cis = {c: bootstrap_ci(tb[n][c]) for c in CRITERIA}
        ket_qua["he_thong"][n] = {c: {"mean": m, "lo": lo, "hi": hi} for c, (m, lo, hi) in cis.items()}
        print(f"| {n} | " + " | ".join(f"{m:.2f} ±{(hi - lo) / 2:.2f}" for m, lo, hi in cis.values()) + " |")

    print("\nSo cặp đôi theo bài:")
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            for c in CRITERIA:
                d = paired_bootstrap(tb[names[x]][c], tb[names[y]][c])
                ket_qua["so_cap"][f"{names[x]} − {names[y]} {c}"] = d
                print(f"  {names[x]} − {names[y]} {c}: {d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] "
                      f"p={d['p']:.4f}")

    # Cau hoi 3: ROUGE co phan anh nguoi doc khong. Sapo khong co ROUGE (no la tham chieu).
    print("\nCâu hỏi 3 — điểm người chấm (trung bình ba tiêu chí) so với điểm tự động:")
    may = [n for n in names if n != REFERENCE]
    nguoi_tb = {n: np.mean([tb[n][c] for c in CRITERIA], axis=0) for n in may}
    for metric in ("rouge1", "rouge2", "rougeL", "bertscore"):
        auto = {n: per_article_metric(*SYSTEMS[n], metric, guids) for n in may}
        if any(v is None for v in auto.values()):
            print(f"  {metric}: thiếu điểm của {[n for n, v in auto.items() if v is None]} — bỏ qua")
            continue
        h = np.concatenate([nguoi_tb[n] for n in may])
        a = np.concatenate([auto[n] for n in may])
        rho = spearman(h, a)
        taus = [kendall_tau_b([nguoi_tb[n][i] for n in may], [auto[n][i] for n in may])
                for i in range(len(guids))]
        taus = [t for t in taus if t is not None]
        he = spearman([nguoi_tb[n].mean() for n in may], [np.mean(auto[n]) for n in may])
        ket_qua["cau_hoi_3"][metric] = {"spearman_ban": rho, "kendall_trong_bai_tb": float(np.mean(taus)),
                                        "n_bai_xep_duoc": len(taus), "spearman_he_thong": he}
        print(f"  {metric:9s} Spearman trên {len(h)} bản: {rho:+.3f} | Kendall trong từng bài (tb "
              f"{len(taus)} bài): {np.mean(taus):+.3f} | xếp hạng {len(may)} hệ thống: {he:+.3f}")

    ket_qua["phieu"] = [p.name for p in sheets]
    dest = RESULTS / "tables" / f"{args.ten}_{khoa['split']}.json"
    dest.write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


# --------------------------------------------------------------------------
# so-sanh: nguoi cham mau so voi may cham
# --------------------------------------------------------------------------

def _doc_nhom(pattern, khoa, chi_bai):
    sheets = sorted(OUT.glob(pattern))
    if not sheets:
        raise SystemExit(f"Không có phiếu nào khớp {pattern} trong {OUT}.")
    out, loi_tong = {}, 0
    for p in sheets:
        d, loi = read_sheet(p, khoa, chi_bai)
        if loi:
            loi_tong += len(loi)
            print(f"{p.name}: {len(loi)} lỗi — {loi[:5]}")
        out[p.stem] = d
    return out, loi_tong


def _fmt(v, nd=2):
    return "—" if v is None else f"{v:+.{nd}f}"


def cmd_so_sanh(args):
    khoa = json.loads((OUT / "khoa.json").read_text(encoding="utf-8"))
    mau = json.loads((OUT / "mau.json").read_text(encoding="utf-8"))
    if sha256(OUT / "phieu_doc_mau.html") != mau["phieu_sha256"]:
        raise SystemExit("phieu_doc_mau.html đã khác bản lúc phát — không so được.")
    chi = set(mau["ma_bai"])
    nguoi, loi_n = _doc_nhom(args.phieu_nguoi, khoa, chi)
    may, loi_m = _doc_nhom(args.phieu_may, khoa, chi)
    if set(nguoi) & set(may):
        raise SystemExit(f"Một phiếu vừa được tính là người vừa là máy: {set(nguoi) & set(may)}.")
    if (loi_n or loi_m) and not args.bo_qua_loi:
        raise SystemExit("Phiếu còn ô trống hoặc sai thang. Sửa phiếu, hoặc chạy lại với --bo-qua-loi.")

    units = [(b["ma_bai"], lab, hs) for b in khoa["bai"] if b["ma_bai"] in chi for lab, hs in b["nhan"].items()]
    print(f"Mẫu {len(chi)} bài, {len(units)} bản | người: {list(nguoi)} | máy: {list(may)}\n")

    def mat(scores, c):
        return np.array([[scores[p].get((mb, lab), {}).get(c, np.nan) for p in scores]
                         for mb, lab, _ in units], dtype=float)

    kq = {"mau": mau["ma_bai"], "nguoi": list(nguoi), "may": list(may), "n_ban": len(units),
          "tieu_chi": {}, "he_thong": {}, "chieu_ket_luan": {}, "cau_hoi_3": {}}
    h_tb, m_tb = {}, {}
    print("Theo tiêu chí (điểm từng bản, trung bình người vs trung bình máy):")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in CRITERIA:
            H, M = mat(nguoi, c), mat(may, c)
            h, m = np.nanmean(H, axis=1), np.nanmean(M, axis=1)
            h_tb[c], m_tb[c] = h, m
            ok = ~np.isnan(h) & ~np.isnan(m)
            a_nn = krippendorff_alpha([[v for v in row if not np.isnan(v)] for row in H]) if H.shape[1] >= 2 else None
            a_mm = krippendorff_alpha([[v for v in row if not np.isnan(v)] for row in M]) if M.shape[1] >= 2 else None
            a_nm = krippendorff_alpha([[x, y] for x, y in zip(h[ok], m[ok])])
            rho = spearman(h[ok], m[ok])
            lech = m[ok] - h[ok]
            tung_luot = {p: spearman(h[ok], M[ok, j]) for j, p in enumerate(may)}
            kq["tieu_chi"][c] = {
                "alpha_nguoi_nguoi": a_nn, "alpha_may_may": a_mm, "alpha_nguoi_may": a_nm,
                "spearman_nguoi_may": rho, "spearman_nguoi_tung_luot_may": tung_luot,
                "may_tru_nguoi_tb": float(lech.mean()), "lech_tuyet_doi_tb": float(np.abs(lech).mean()),
                "ty_le_lech_toi_da_1": float(np.mean(np.abs(lech) <= 1)),
                "tb_nguoi": float(h[ok].mean()), "tb_may": float(m[ok].mean()), "n": int(ok.sum()),
            }
            print(f"  {c:10s} alpha người–người {_fmt(a_nn)} | máy–máy {_fmt(a_mm)} | người–máy {_fmt(a_nm)} "
                  f"| Spearman người–máy {_fmt(rho)} | máy − người {lech.mean():+.2f} "
                  f"| lệch ≤1: {np.mean(np.abs(lech) <= 1):.0%}")

    print("\nTrung bình từng hệ thống trên mẫu (người / máy):")
    for s in khoa["systems"]:
        idx = [i for i, (_, _, hs) in enumerate(units) if s in hs]
        kq["he_thong"][s] = {c: {"nguoi": float(np.nanmean(h_tb[c][idx])), "may": float(np.nanmean(m_tb[c][idx]))}
                             for c in CRITERIA}
        print(f"  {s:9s} " + " | ".join(f"{c} {kq['he_thong'][s][c]['nguoi']:.2f}/{kq['he_thong'][s][c]['may']:.2f}"
                                         for c in CRITERIA))
    for c in CRITERIA:
        xh_n = sorted(khoa["systems"], key=lambda s: -kq["he_thong"][s][c]["nguoi"])
        xh_m = sorted(khoa["systems"], key=lambda s: -kq["he_thong"][s][c]["may"])
        kq["tieu_chi"][c]["xep_hang_nguoi"], kq["tieu_chi"][c]["xep_hang_may"] = xh_n, xh_m
        print(f"  xếp hạng {c}: người {xh_n} | máy {xh_m} | {'TRÙNG' if xh_n == xh_m else 'KHÁC'}")

    # Chieu cac ket luan chinh o cap bai: cung mot cap he thong, nguoi va may co cung dau khong
    print("\nChiều các kết luận chính (so cặp theo bài trên mẫu):")
    ma_bai = [mb for mb in mau["ma_bai"]]

    def theo_bai(tb_c, s):
        return np.array([next(tb_c[i] for i, (mb2, _, hs) in enumerate(units) if mb2 == mb and s in hs)
                         for mb in ma_bai])

    for s1, s2 in KET_LUAN:
        if s1 not in khoa["systems"] or s2 not in khoa["systems"]:
            continue
        for c in CRITERIA:
            dn = paired_bootstrap(theo_bai(h_tb[c], s1), theo_bai(h_tb[c], s2))
            dm = paired_bootstrap(theo_bai(m_tb[c], s1), theo_bai(m_tb[c], s2))
            cung = bool(np.sign(round(dn["diff"], 6)) == np.sign(round(dm["diff"], 6)))
            kq["chieu_ket_luan"][f"{s1} − {s2} {c}"] = {"nguoi": dn, "may": dm, "cung_chieu": cung}
            print(f"  {s1} − {s2} {c:10s} người {dn['diff']:+.2f} [{dn['lo']:+.2f}, {dn['hi']:+.2f}] | "
                  f"máy {dm['diff']:+.2f} [{dm['lo']:+.2f}, {dm['hi']:+.2f}] | {'cùng chiều' if cung else 'NGƯỢC CHIỀU'}")

    # Cau hoi 3 tren mau: ROUGE so voi nguoi, va so voi may, tren dung cung nhung ban
    print("\nCâu hỏi 3 trên mẫu — Spearman giữa điểm trung bình ba tiêu chí và điểm tự động:")
    guid_of = {b["ma_bai"]: b["guid"] for b in khoa["bai"]}
    cap = [(i, s) for i, (_, _, hs) in enumerate(units) for s in hs if s != REFERENCE]
    avg_n = np.mean([h_tb[c] for c in CRITERIA], axis=0)
    avg_m = np.mean([m_tb[c] for c in CRITERIA], axis=0)
    for metric in ("rouge1", "bertscore"):
        auto = {}
        for s in {s for _, s in cap}:
            gs = [guid_of[mb] for mb in ma_bai]
            vals = per_article_metric(*SYSTEMS[s], metric, gs)
            auto[s] = None if vals is None else dict(zip(gs, vals))
        if any(v is None for v in auto.values()):
            print(f"  {metric}: thiếu điểm — bỏ qua")
            continue
        a = [auto[s][guid_of[units[i][0]]] for i, s in cap]
        rn = spearman([avg_n[i] for i, _ in cap], a)
        rm = spearman([avg_m[i] for i, _ in cap], a)
        kq["cau_hoi_3"][metric] = {"spearman_nguoi": rn, "spearman_may": rm, "n": len(cap)}
        print(f"  {metric:9s} trên {len(cap)} bản: người {_fmt(rn)} | máy {_fmt(rm)}")

    dest = RESULTS / "tables" / f"{args.ten}_{khoa['split']}.json"
    dest.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {dest}")


def main():
    ap = argparse.ArgumentParser(description="Chấm blind có người thật.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="rút bài, dựng phiếu đọc, phiếu điểm và khoá")
    p.add_argument("--n", type=int, default=50, help="số bài")
    p.add_argument("--nguoi", type=int, default=3, help="số người chấm")
    p.add_argument("--systems", nargs="+", default=DEFAULT, metavar="TÊN", help=f"trong {list(SYSTEMS)}")
    p.set_defaults(fn=cmd_prepare)
    a = sub.add_parser("analyze", help="đọc phiếu đã điền, tính điểm, đồng thuận, câu hỏi 3")
    a.add_argument("--bo-qua-loi", action="store_true", help="vẫn phân tích khi phiếu còn ô trống")
    # Diem cua mo hinh ngon ngu (LLM-as-judge) nam o file rieng va ghi ra bang rieng, de
    # khong bao gio lan voi diem nguoi cham that.
    a.add_argument("--phieu", default="cham_nguoi*.csv", help="mẫu tên phiếu trong results/human_eval/")
    a.add_argument("--ten", default="human_eval", help="tiền tố bảng kết quả trong results/tables/")
    a.set_defaults(fn=cmd_analyze)
    pm = sub.add_parser("prepare-mau", help="rút phiếu mẫu cho người chấm từ chính phiếu 50 bài")
    pm.add_argument("--n", type=int, default=12, help="số bài trong mẫu")
    pm.add_argument("--nguoi", type=int, default=2, help="số người chấm mẫu")
    pm.set_defaults(fn=cmd_prepare_mau)
    ss = sub.add_parser("so-sanh", help="so điểm người chấm mẫu với điểm máy chấm trên cùng những bản")
    ss.add_argument("--phieu-nguoi", default="cham_mau_nguoi*.csv", help="phiếu người chấm mẫu")
    ss.add_argument("--phieu-may", default="llm_judge*.csv", help="phiếu máy chấm (đủ 50 bài)")
    ss.add_argument("--bo-qua-loi", action="store_true", help="vẫn so khi phiếu người còn ô trống")
    ss.add_argument("--ten", default="nguoi_vs_may", help="tiền tố bảng kết quả trong results/tables/")
    ss.set_defaults(fn=cmd_so_sanh)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
