"""Giai đoạn 6a: thu thập và **đóng băng** một ngữ liệu tin mới để còn đo được.

    .venv/Scripts/python.exe src/agent/thu_thap.py thu-thap --n 40
    .venv/Scripts/python.exe src/agent/thu_thap.py kiem

**Vì sao phải đóng băng.** Agent tìm kiếm trực tiếp trên web thì không lần chạy nào giống
lần nào, và cả đồ án mất khả năng tái lập — thứ đắt nhất mà repo này đang có. Nên tách đôi:
agent *demo* đi tìm trực tiếp, còn agent *được chấm* chạy trên một bộ tin đã tải về và khoá
lại. Mọi con số trong báo cáo đến từ bộ khoá đó.

**Chọn nguồn theo robots.txt, không theo tiện.** `vnexpress.net` ghi `Disallow: /` cho
ClaudeBot, anthropic-ai, GPTBot, CCBot và các bot thu thập cho AI khác (chỉ cho phép loại
truy cập do người dùng chủ động). Thu thập hàng loạt để dựng ngữ liệu đúng là thứ họ từ
chối, nên **không lấy vnexpress**; `dantri.com.vn` cũng nêu tên claudebot nên tránh luôn.
Ba nguồn dùng ở đây đều có `User-agent: *` → `Allow: /` và không có điều khoản riêng cho AI.
Kiểm lại bằng `kiem` trước mỗi đợt thu thập mới, vì robots.txt có thể đổi.

**Lấy vừa đủ, và chậm.** Mỗi bài một yêu cầu, nghỉ `NGHI` giây giữa hai yêu cầu, tổng chỉ
vài chục bài. Dùng RSS do chính toà soạn phát hành để lấy danh sách bài thay vì dò sitemap.

**Toàn văn KHÔNG commit.** `tin.json` nằm ngoài git (cùng lối với checkpoint), chỉ `kê_khai.json`
— URL, thời điểm, độ dài, SHA-256 — được commit. Đủ để đối chiếu rằng bộ dữ liệu không bị
sửa giữa chừng, mà không phát tán lại toàn văn báo chí.
"""

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import syllables  # noqa: E402
from eval.rouge import sentences_raw  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "tin_moi"
HTML = OUT / "html"
UA = "BTL-DL-research/0.1 (do an sinh vien; thu thap ngu lieu hoc thuat quy mo nho)"
NGHI = 2.5          # giay nghi giua hai yeu cau -- lich su voi may chu, va du cham de khong bi chan
HET_GIO = 25

# Than bai cua tung nguon. Tuoi Tre va Thanh Nien dung chung CMS nen chung mot quy tac.
# `muc`: {slug tren trang do: ten chuyen muc dung chung}. Moi toa soan dat ten mot kieu --
# Thanh Nien khong co `kinh-doanh` hay `phap-luat` (404), ma la `kinh-te` va `doi-song`.
# Gom ve ten chung de con dem duoc phan bo chuyen muc tren ca bo ngu lieu.
NGUON = {
    "tuoitre": {"ten": "Tuổi Trẻ", "rss": "https://tuoitre.vn/rss/{}.rss",
                "than": r'itemprop="articleBody"',
                "muc": {"thoi-su": "thoi-su", "the-gioi": "the-gioi", "kinh-doanh": "kinh-doanh",
                        "phap-luat": "phap-luat", "giao-duc": "giao-duc"}},
    "thanhnien": {"ten": "Thanh Niên", "rss": "https://thanhnien.vn/rss/{}.rss",
                  "than": r'itemprop="articleBody"',
                  "muc": {"thoi-su": "thoi-su", "the-gioi": "the-gioi", "kinh-te": "kinh-doanh",
                          "doi-song": "doi-song", "giao-duc": "giao-duc"}},
    "vietnamnet": {"ten": "VietnamNet", "rss": "https://vietnamnet.vn/rss/{}.rss",
                   "than": r'id="maincontent"',
                   "muc": {"thoi-su": "thoi-su", "the-gioi": "the-gioi", "kinh-doanh": "kinh-doanh",
                           "phap-luat": "phap-luat", "giao-duc": "giao-duc"}},
}

# Nguong loai bai: bai anh, bai video va tin van khong dung de do tom tat.
CAU_TOI_THIEU = 5
AM_TIET_TOI_THIEU = 150


def _tai(u):
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=HET_GIO)
    return r.read().decode("utf-8-sig", "replace")


def _chu(h):
    """HTML -> văn bản thuần: bỏ script/style, đổi thẻ khối thành khoảng trắng, gỡ thực thể."""
    h = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", h)
    h = re.sub(r"(?is)<br\s*/?>|</(p|div|h\d|li)>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    return re.sub(r"[ \t]+", " ", html.unescape(h)).strip()


def _cat_khoi(h, moc):
    """Cắt khối HTML chứa thân bài, bắt đầu từ `moc` và dừng ở chỗ hết bài.

    Cắt thô theo mốc kết thúc thay vì dựng cây DOM: các trang này nhét rất nhiều khối
    "tin liên quan" phía sau bài, và mọi mốc dưới đây đều nằm SAU câu cuối của bài.
    """
    i = h.find(moc) if moc in h else -1
    if i < 0:
        m = re.search(moc, h)
        if not m:
            return ""
        i = m.start()
    # Moc (`itemprop="articleBody"`, `id="maincontent"`) nam BEN TRONG the mo, nen phai
    # nhay qua dau '>' dong the -- khong thi phan con lai cua the roi thang vao van ban.
    dong = h.find(">", i)
    con = h[dong + 1:] if dong > 0 else h[i:]
    # `sendstarauthor`/`formreactdetail`: khoi tang sao va binh luan cuoi bai cua Tuoi Tre.
    # `summary__content`: the bai lien quan cuoi bai cua VietnamNet.
    for het in ('class="detail-tab-list', 'class="box-tinlienquan', 'class="article-relate',
                'class="detail-tag', 'id="admzone', 'class="box-category', 'class="relate',
                'class="sendstarauthor', 'class="formreactdetail', 'class="summary__content',
                'class="footer', "</body>"):
        j = con.find(het)
        if j > 0:
            con = con[:j]
    return con


def _go_khoi_div(h, mau):
    """Gỡ trọn các khối `<div>` mà thẻ mở khớp `mau`, đếm lồng nhau để cắt đúng thẻ đóng.

    Không dùng regex `<div.*?</div>` được: các thẻ này lồng nhiều tầng, regex sẽ cắt ở thẻ
    đóng đầu tiên và để lại nửa khối rác trong văn bản.
    """
    while True:
        m = re.search(mau, h)
        if not m:
            return h
        i, sau, sau_do = m.start(), 1, m.end()
        for t in re.finditer(r"(?i)<(/?)div", h[m.end():]):
            sau += -1 if t.group(1) else 1
            if sau == 0:
                sau_do = m.end() + t.end()
                j = h.find(">", sau_do - 1)
                sau_do = j + 1 if j > 0 else sau_do
                break
        else:
            return h[:i]                      # khong thay the dong: cat tu day tro di
        h = h[:i] + " " + h[sau_do:]


# Dong tac gia va chu thich anh: khong phai noi dung bai. Bo o day de bo do phu y khong
# phai dem chung nhu mot y -- dung bai hoc tu bai mau "bat ca" o demo.
BO_DONG = re.compile(
    r"(?i)^\s*(ảnh\s*:|nguồn\s*:|video\s*:|clip\s*:|xem thêm|theo\s+\w+\s*$|"
    r"tin liên quan|đọc thêm|chia sẻ|bình luận)")


def _than_bai(h, moc):
    khoi = _cat_khoi(h, moc)
    if not khoi:
        return ""
    # Bo chu thich anh truoc khi go the, vi sau khi go thi khong con phan biet duoc.
    khoi = re.sub(r"(?is)<figcaption[^>]*>.*?</figcaption>", " ", khoi)
    khoi = re.sub(r"(?is)<table[^>]*>.*?</table>", " ", khoi)
    # The bai lien quan nhung GIUA than bai (VietnamNet: <article class="ck-cms-insert-news">).
    # Phai go khoi chu khong cat duoi: no nam giua bai, cat la mat phan con lai cua bai.
    khoi = re.sub(r"(?is)<article[^>]*>.*?</article>", " ", khoi)
    # Tuoi Tre / Thanh Nien nhung the "bai lien quan" giua than bai bang mot div long nhieu
    # tang (`type="RelatedOneNews"`). Tieu de + sapo cua bai KHAC, khong phai noi dung bai nay.
    khoi = _go_khoi_div(khoi, r'(?i)<div[^>]*type="RelatedOneNews"[^>]*>')
    dong = [d.strip() for d in _chu(khoi).split("\n")]
    giu = [d for d in dong if len(d) > 40 and not BO_DONG.match(d) and not d.endswith((":",))]
    return " ".join(giu)


def _muc_rss(t):
    """[(tiêu đề, link)] từ một feed RSS."""
    ra = []
    for m in re.findall(r"(?is)<item>(.*?)</item>", t):
        td = re.search(r"(?is)<title>(.*?)</title>", m)
        lk = re.search(r"(?is)<link>(.*?)</link>", m)
        if not (td and lk):
            continue
        bo = lambda s: html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", s).strip())
        ra.append((bo(td.group(1)), bo(lk.group(1))))
    return ra


def _ten_html(url):
    return HTML / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".html")


def _mot_bai(url, cau_hinh, tai_lai=True):
    """(sapo, thân bài) — tải về thì lưu HTML thô, và lần sau đọc thẳng từ đĩa.

    Bộ trích nội dung chắc chắn phải sửa vài vòng. Giữ HTML thô để mỗi vòng sửa chỉ là
    chạy lại `trich` trên đĩa, chứ không phải tải lại từ máy chủ của toà soạn.
    """
    tep = _ten_html(url)
    if tep.exists() and not tai_lai:
        h = tep.read_text(encoding="utf-8", errors="replace")
    else:
        h = _tai(url)
        HTML.mkdir(parents=True, exist_ok=True)
        tep.write_text(h, encoding="utf-8")
    og = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]*)"', h)
    sapo = html.unescape(og.group(1)).strip() if og else ""
    return sapo, _than_bai(h, cau_hinh["than"])


def cmd_thu_thap(args):
    OUT.mkdir(parents=True, exist_ok=True)
    moi_nguon = -(-args.n // len(NGUON))          # chia deu, lam tron len

    # Han muc cho TUNG chuyen muc, khong chi cho tung nguon: khong the ca bo ngu lieu deu la
    # `thoi-su`, vi tin thoi su nghieng han ve tai nan va hanh chinh, khong dai dien cho bao chi.
    moi_muc = -(-moi_nguon // max(len(v["muc"]) for v in NGUON.values()))
    tin, bo_qua = [], []
    for khoa, ch in NGUON.items():
        lay = 0
        for slug, cm in ch["muc"].items():
            if lay >= moi_nguon or len(tin) >= args.n:
                break
            lay_muc = 0
            try:
                muc = _muc_rss(_tai(ch["rss"].format(slug)))
            except Exception as e:
                print(f"  [{khoa}/{slug}] không lấy được RSS: {type(e).__name__}")
                time.sleep(NGHI)
                continue
            time.sleep(NGHI)
            for tieu_de, url in muc:
                if lay >= moi_nguon or lay_muc >= moi_muc or len(tin) >= args.n:
                    break
                da_co = _ten_html(url).exists()
                if any(t["url"] == url for t in tin):
                    continue
                try:
                    sapo, than = _mot_bai(url, ch, tai_lai=not _ten_html(url).exists())
                except Exception as e:
                    bo_qua.append((url, f"{type(e).__name__}"))
                    time.sleep(NGHI)
                    continue
                if not da_co:
                    time.sleep(NGHI)
                n_cau, n_am = len(sentences_raw(than)), len(syllables(than))
                if not sapo or n_cau < CAU_TOI_THIEU or n_am < AM_TIET_TOI_THIEU:
                    bo_qua.append((url, f"quá ngắn: {n_cau} câu, {n_am} âm tiết"))
                    continue
                lay += 1
                lay_muc += 1
                tin.append({
                    "ma": f"T{len(tin) + 1:03d}", "bao": ch["ten"], "khoa_bao": khoa,
                    "chuyen_muc": cm, "url": url, "tieu_de": tieu_de, "sapo": sapo,
                    "than_bai": than, "so_cau": n_cau, "am_tiet": n_am,
                    "sha256": hashlib.sha256(than.encode("utf-8")).hexdigest(),
                    "thoi_diem_crawl": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                print(f"  [{len(tin):3}] {ch['ten']:11} {n_cau:2} câu {n_am:4} âm tiết  {tieu_de[:58]}")

    (OUT / "tin.json").write_text(json.dumps(tin, ensure_ascii=False, indent=1), encoding="utf-8")
    ke = {
        "tao_luc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(tin), "nguon": {k: v["ten"] for k, v in NGUON.items()},
        "chuyen_muc": {k: v["muc"] for k, v in NGUON.items()}, "nghi_giay": NGHI, "user_agent": UA,
        "loai_tru": {"vnexpress.net": "robots.txt Disallow: / cho ClaudeBot/anthropic-ai/GPTBot/CCBot",
                     "dantri.com.vn": "robots.txt nêu tên claudebot"},
        "nguong": {"cau_toi_thieu": CAU_TOI_THIEU, "am_tiet_toi_thieu": AM_TIET_TOI_THIEU},
        "bo_qua": [{"url": u, "ly_do": r} for u, r in bo_qua],
        "bai": [{k: t[k] for k in ("ma", "bao", "chuyen_muc", "url", "tieu_de",
                                   "so_cau", "am_tiet", "sha256", "thoi_diem_crawl")} for t in tin],
    }
    (OUT / "ke_khai.json").write_text(json.dumps(ke, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(tin)} bài -> {OUT/'tin.json'} (ngoài git) và {OUT/'ke_khai.json'} (commit)")
    if bo_qua:
        print(f"bỏ qua {len(bo_qua)} bài; lý do ghi trong kê khai")


def cmd_trich(args):
    """Trích lại toàn văn từ HTML đã lưu — không gọi mạng lần nào.

    Bộ trích nội dung phải sửa vài vòng mới sạch. Tách hẳn khâu này khỏi khâu tải để mỗi
    vòng sửa không thành một đợt gõ cửa máy chủ của toà soạn.
    """
    ke = json.loads((OUT / "ke_khai.json").read_text(encoding="utf-8"))
    cu_tin = {t["url"]: t for t in json.loads((OUT / "tin.json").read_text(encoding="utf-8"))}
    tin, hong = [], []
    for b in ke["bai"]:
        tep = _ten_html(b["url"])
        if not tep.exists():
            hong.append((b["ma"], "chưa lưu HTML"))
            continue
        ch = NGUON[{v["ten"]: k for k, v in NGUON.items()}[b["bao"]]]
        sapo, than = _mot_bai(b["url"], ch, tai_lai=False)
        n_cau, n_am = len(sentences_raw(than)), len(syllables(than))
        t = dict(cu_tin.get(b["url"], {}), **{
            "ma": b["ma"], "bao": b["bao"], "chuyen_muc": b["chuyen_muc"], "url": b["url"],
            "tieu_de": b["tieu_de"], "sapo": sapo, "than_bai": than,
            "so_cau": n_cau, "am_tiet": n_am,
            "sha256": hashlib.sha256(than.encode("utf-8")).hexdigest(),
            "thoi_diem_crawl": b["thoi_diem_crawl"]})
        tin.append(t)
        cu = cu_tin.get(b["url"], {}).get("am_tiet")
        doi = f"  ({cu} -> {n_am})" if cu and cu != n_am else ""
        print(f"  {b['ma']} {b['bao']:11} {n_cau:2} câu {n_am:4} âm tiết{doi}")
    (OUT / "tin.json").write_text(json.dumps(tin, ensure_ascii=False, indent=1), encoding="utf-8")
    for b in ke["bai"]:
        t = next((x for x in tin if x["ma"] == b["ma"]), None)
        if t:
            b.update({k: t[k] for k in ("so_cau", "am_tiet", "sha256")})
    (OUT / "ke_khai.json").write_text(json.dumps(ke, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"{len(tin)} bài trích lại từ đĩa" + (f"; hỏng {len(hong)}: {hong}" if hong else ""))


def cmd_kiem(args):
    """Đối chiếu toàn văn với kê khai, và kiểm lại robots.txt của các nguồn."""
    tin = json.loads((OUT / "tin.json").read_text(encoding="utf-8"))
    ke = json.loads((OUT / "ke_khai.json").read_text(encoding="utf-8"))
    theo_ma = {b["ma"]: b for b in ke["bai"]}
    lech = [t["ma"] for t in tin
            if theo_ma.get(t["ma"], {}).get("sha256")
            != hashlib.sha256(t["than_bai"].encode("utf-8")).hexdigest()]
    print(f"{len(tin)} bài | khớp kê khai: {len(tin) - len(lech)}/{len(tin)}"
          + (f" | LỆCH: {lech}" if lech else ""))
    am = sorted(t["am_tiet"] for t in tin)
    cau = sorted(t["so_cau"] for t in tin)
    if am:
        print(f"  âm tiết: {am[0]}–{am[-1]}, trung vị {am[len(am)//2]}")
        print(f"  số câu : {cau[0]}–{cau[-1]}, trung vị {cau[len(cau)//2]}")
        from collections import Counter
        print("  theo báo:", dict(Counter(t["bao"] for t in tin)))
        print("  chuyên mục:", dict(Counter(t["chuyen_muc"] for t in tin)))

    if not args.bo_qua_mang:
        print("\nKiểm lại robots.txt (có thể đổi sau ngày thu thập):")
        for khoa in NGUON:
            u = f"https://{ {'tuoitre': 'tuoitre.vn', 'thanhnien': 'thanhnien.vn', 'vietnamnet': 'vietnamnet.vn'}[khoa] }/robots.txt"
            try:
                t = _tai(u)
            except Exception as e:
                print(f"  {khoa}: không đọc được ({type(e).__name__})")
                continue
            ai = sorted({a.strip().lower() for a in re.findall(r"(?im)^user-agent:\s*(.+)$", t)
                         if any(s in a.lower() for s in ("claude", "anthropic", "gpt", "ccbot"))})
            print(f"  {khoa}: bot AI được nêu tên: {ai or 'không có'}")
            time.sleep(NGHI)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    s = p.add_subparsers(dest="lenh", required=True)
    t = s.add_parser("thu-thap", help="tải và đóng băng ngữ liệu")
    t.add_argument("--n", type=int, default=40, help="số bài cần lấy (chia đều ba nguồn)")
    t.set_defaults(fn=cmd_thu_thap)
    s.add_parser("trich", help="trích lại toàn văn từ HTML đã lưu (không gọi mạng)").set_defaults(fn=cmd_trich)
    k = s.add_parser("kiem", help="đối chiếu toàn văn với kê khai, kiểm lại robots.txt")
    k.add_argument("--bo-qua-mang", action="store_true", help="không kiểm lại robots.txt")
    k.set_defaults(fn=cmd_kiem)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
