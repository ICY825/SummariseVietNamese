r"""Trang duyệt toàn bộ VietNews ngay trên máy — 143.816 bài, không xuất file trung gian.

Vì sao cần script này: dữ liệu nằm trong ba file `.parquet` ở cache HuggingFace,
là định dạng nén theo cột nên không mở trực tiếp được. Xuất ra CSV thì được 475 MB
và Excel bò rất chậm ở cỡ 99.134 dòng có cột văn bản dài. Nên thay vì xuất, script
mở sẵn dữ liệu trong bộ nhớ rồi phục vụ qua HTTP với phân trang và tìm kiếm.

Đọc THẲNG từ parquet chứ không gọi `load_dataset`: mở nhanh hơn hẳn và không sinh
thêm cache Arrow (thư mục `datasets/` của HF vốn đã 588 MB).

Điểm khác biệt so với việc mở parquet bằng extension của VS Code: script biết
`data/splits/`, nên mỗi bài hiện luôn nhãn cho biết nó thuộc tập con đã đóng băng
nào (`test`, `train_5k`, …). Đó là thứ quyết định bài đó có được chấm điểm hay không.

Tìm kiếm chạy ở tầng C++ bằng `pyarrow.compute` chứ không lặp bằng Python — quét
99.134 bài mất chưa tới một giây. Dấu cách và gạch dưới được coi là một, nên gõ
"Triều Tiên" hay "Triều_Tiên" đều ra cùng kết quả; đây là bộ dữ liệu đã tách từ
sẵn, nếu khớp nguyên văn thì gõ có dấu cách sẽ ra 0 bài.

Chạy:  .venv/Scripts/python.exe src/data/browse.py
       .venv/Scripts/python.exe src/data/browse.py --split train --port 8010
"""

import argparse
import html
import json
import os
import pathlib
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Console Windows mac dinh cp1252 -> khong in duoc tieng Viet. Ep UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyarrow.compute as pc
import pyarrow.parquet as pq

SNAPSHOTS = (
    pathlib.Path(os.environ["USERPROFILE"])
    / ".cache/huggingface/hub/datasets--nam194--vietnews/snapshots"
)
SPLITS_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "splits"
PAGE = 25

# Ten split cua HuggingFace; `val`/`tune` cua du an deu cat ra tu `validation`.
HF_SPLITS = ("train", "validation", "test")


def parquet_path(split):
    """File parquet cua mot split HF. Bao loi ro rang neu cache chua co."""
    try:
        return next(SNAPSHOTS.rglob(f"{split}-*.parquet"))
    except StopIteration:
        raise SystemExit(
            f"Chưa có dữ liệu {split} trong cache HuggingFace.\n"
            f"  Đã tìm ở: {SNAPSHOTS}\n"
            f"  Tải về bằng: .venv/Scripts/python.exe src/data/inspect_vietnews.py"
        )


def load_subset_labels():
    """guid -> tên các tập con đã đóng băng chứa nó, tra theo từng split HF.

    guid CHI duy nhat trong pham vi mot split HF, nen khoa phai la cap
    (split, guid). Gop chung mot bang se gan nham nhan cho bai khac split.
    """
    labels = {s: {} for s in HF_SPLITS}
    for path in sorted(SPLITS_DIR.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        bucket = labels[meta["split"]]
        for g in meta["guid"]:
            bucket.setdefault(str(g), []).append(meta["name"])
    return labels


class Data:
    """Giu bang parquet da nap, nap lai khi doi split."""

    def __init__(self):
        self.split = None
        self.table = None
        self.labels = load_subset_labels()

    def use(self, split):
        if split != self.split:
            print(f"  nạp {split}...", flush=True)
            self.table = pq.read_table(parquet_path(split))
            self.split = split
        return self.table

    def search(self, split, q, field):
        """Chi so cac bai khop `q`. Rong thi tra ve toan bo."""
        t = self.use(split)
        if not q:
            return None  # None = khong loc, tranh dung mot mang 99.134 phan tu
        # Van ban da tach tu san ("Trieu_Tien"), nen tim "Trieu Tien" bang
        # match_substring se ra 0 bai — nguoi dung tuong khong co bai nao.
        # Coi dau cach va gach duoi la MOT: moi khoang trang trong cau tim thanh
        # `[ _]+`. Phan con lai phai escape vi day la regex.
        pat = "[ _]+".join(re.escape(w) for w in q.split())
        mask = pc.match_substring_regex(t.column(field), pat, ignore_case=True)
        return pc.indices_nonzero(mask).to_pylist()


def mark(text, q):
    """Thoat HTML roi to dam cum tim kiem.

    To dam phai dung DUNG luat khop cua `Data.search` — cung coi dau cach va
    gach duoi la mot — neu khong se co truong hop tim ra bai nhung khong cho
    dam cho nao, trong nhu bi loi.

    Escape HTML TRUOC roi moi chen the <mark>, de van ban trong du lieu khong
    tro thanh the HTML.
    """
    out = html.escape(text)
    if not q:
        return out
    pat = "[ _]+".join(re.escape(html.escape(w)) for w in q.split())
    return re.sub(pat, lambda m: f"<mark>{m.group(0)}</mark>", out, flags=re.I)


STYLE = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b68;--line:#e3e3e0;--card:#fff;--acc:#b5501f}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--dim:#9a9a96;
--line:#33322f;--card:#232320;--acc:#e0855a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
form{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
input,select,button{font:inherit;padding:7px 10px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--fg)}
input[type=text]{flex:1;min-width:200px}
button{cursor:pointer;background:var(--acc);color:#fff;border-color:transparent}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:14px 16px;margin-bottom:12px}
.gid{font:12px ui-monospace,Consolas,monospace;color:var(--dim)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;
border:1px solid var(--acc);color:var(--acc);margin-left:6px}
.ttl{font-weight:600;margin:6px 0}
.abs{margin:8px 0;padding-left:11px;border-left:3px solid var(--acc)}
details summary{cursor:pointer;color:var(--dim);font-size:13px}
.art{margin-top:8px;color:var(--dim);font-size:14px;white-space:pre-wrap}
mark{background:#ffd9a8;color:#1a1a19}
nav{display:flex;gap:10px;align-items:center;justify-content:center;margin-top:20px}
nav a{color:var(--acc)}
.none{color:var(--dim);padding:32px;text-align:center}
"""


def render(d, split, q, field, page):
    idx = d.search(split, q, field)
    t = d.table
    total = t.num_rows if idx is None else len(idx)
    pages = max(1, -(-total // PAGE))
    page = max(1, min(page, pages))
    lo = (page - 1) * PAGE
    rows = range(lo, min(lo + PAGE, total)) if idx is None else idx[lo : lo + PAGE]

    labels = d.labels[split]
    cards = []
    for i in rows:
        guid = str(t.column("guid")[i].as_py())
        tags = "".join(f'<span class="tag">{n}</span>' for n in labels.get(guid, []))
        art = t.column("article")[i].as_py()
        cards.append(
            f'<div class="card"><span class="gid">#{i} · guid {guid}</span>{tags}'
            f'<div class="ttl">{mark(t.column("title")[i].as_py(), q)}</div>'
            f'<div class="abs">{mark(t.column("abstract")[i].as_py(), q)}</div>'
            f"<details><summary>Toàn văn — {len(art.split()):,} token</summary>"
            f'<div class="art">{mark(art, q)}</div></details></div>'
        )
    body = "".join(cards) or '<div class="none">Không có bài nào khớp.</div>'

    def link(p, txt):
        qs = urllib.parse.urlencode(
            {"split": split, "q": q, "field": field, "page": p}
        )
        return f'<a href="/?{qs}">{txt}</a>'

    nav = " ".join(
        [
            link(page - 1, "← Trước") if page > 1 else "",
            f"Trang {page:,} / {pages:,}",
            link(page + 1, "Sau →") if page < pages else "",
        ]
    )
    opts = "".join(
        f'<option value="{s}"{" selected" if s == split else ""}>{s}</option>'
        for s in HF_SPLITS
    )
    fopts = "".join(
        f'<option value="{f}"{" selected" if f == field else ""}>{lbl}</option>'
        for f, lbl in (
            ("title", "tiêu đề"),
            ("abstract", "sapo"),
            ("article", "toàn văn"),
        )
    )
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VietNews — {split}</title><style>{STYLE}</style></head><body><div class="wrap">
<h1>VietNews — {split}</h1>
<div class="sub">{total:,} bài{"" if idx is None else " khớp"} ·
nhãn màu cho biết bài thuộc tập con đã đóng băng nào trong <code>data/splits/</code></div>
<form><select name="split">{opts}</select>
<input type="text" name="q" value="{html.escape(q)}" placeholder="Tìm trong...">
<select name="field">{fopts}</select>
<button>Tìm</button></form>
{body}<nav>{nav}</nav></div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test", choices=HF_SPLITS, help="split mở sẵn")
    ap.add_argument("--port", type=int, default=8009)
    args = ap.parse_args()

    d = Data()
    d.use(args.split)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path != "/":
                self.send_error(404)
                return
            p = urllib.parse.parse_qs(u.query)
            page = p.get("page", ["1"])[0]
            out = render(
                d,
                p.get("split", [args.split])[0],
                p.get("q", [""])[0],
                p.get("field", ["title"])[0],
                int(page) if page.isdigit() else 1,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass  # khong do log request ra man hinh

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"\n  Mở trình duyệt tại  http://127.0.0.1:{args.port}")
    print("  Ctrl+C để dừng.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
