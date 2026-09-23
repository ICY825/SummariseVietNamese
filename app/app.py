"""Demo Gradio: hệ thống cuối cạnh BARTpho, và bốn tầng cũ làm phần phân tích.

    ~/.venvs/demo/Scripts/python.exe app/app.py

Tab chính kể đúng câu chuyện của đề tài: BARTpho đứng đầu ROUGE nhưng có thể bịa chi tiết;
hệ thống cuối (PhoBERT chọn câu có chủ đích) chép nguyên câu nên không bịa và đủ ý hơn. Chi tiết
lạ được **tô vàng** theo đúng bộ đo `chi_tiet_la` dùng để chấm — kèm giới hạn của bộ đo, vì demo
không được hứa nhiều hơn bộ đo làm được.

Tab "Bốn tầng" giữ nguyên demo tuần 7: các hướng cũ chạy cạnh nhau, không có ô "tốt nhất".

**Bài mẫu chạy trên đúng dạng tách từ của bộ dữ liệu** (`vi_du.json`), nên hệ thống cuối ra đúng
bản đã báo cáo, và BARTpho ra đúng bản `val_in1024` (nạp checkpoint từ đĩa — cùng đường với lần
chấm `test`). Bản BARTpho trong phiếu chấm tuần 7 sinh ngay sau huấn luyện nên khác chữ ở 126/1.000
bài (README, "Đối chứng sạch"); ở B08 hai bản chỉ khác chữ "địa bàn", cùng lỗi "5h30". Bài dán vào được tách từ bằng underthesea — khác bộ tách từ của dữ liệu — nên hệ thống cuối
có thể chọn câu khác bản báo cáo (40 bài `val`: 18/40 trùng, recall 59,5 so với 60,4).

Phần trình bày nằm ở `app/giao_dien.css`; mọi con số trên trang đọc thẳng từ
`results/tables/chinh_xac_test.json`, không gõ tay, để giao diện không bao giờ lệch báo cáo.
"""

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import gradio as gr  # noqa: E402

import pipeline as P  # noqa: E402

VI_DU = (
    "Công an TP Long Khánh (Đồng Nai) cho biết đang tạm giữ hình sự đối với Quách Lục "
    "Tuấn (41 tuổi) để điều tra về hành vi trộm cắp và cố ý gây thương tích. Trước đó, "
    "ngày 9/8, Tuấn đột nhập nhà ông Thái Ngọc Lục (59 tuổi) để trộm cắp. Hành động của "
    "tên trộm nhanh chóng bị phát giác. Khi chủ nhà lớn tiếng hô hoán thì Tuấn rút dao "
    "giấu sẵn trong người đâm khiến ông Lục bị thương. Nghe tiếng tri hô, một số người "
    "dân xung quanh đã đến hỗ trợ vây bắt và bàn giao Tuấn cho công an phường. Ông Lục "
    "được người nhà đưa đi cấp cứu trong tình trạng bị đâm ở chân và tay, hiện đã bình "
    "phục và sẽ xuất viện trong vài ngày tới."
)
# Bai mau cua tab chinh: bai `val` co loi BARTpho da duoc nguoi/may cham xac nhan o tuan 7
VI_DU_CUOI = json.loads((ROOT / "app" / "vi_du.json").read_text(encoding="utf-8"))
TACH_TU_MAU = {x["tho"]: x["tach_tu"] for x in VI_DU_CUOI}
# Truc thu hai cua tab chinh: bai nay co bao nhieu y, ban tom tat lay duoc may.
# Danh sach y do NGUOI gan tay trong `vi_du.json`, khong tu sinh -- nen chi bai mau moi co.
# Quy tac gan: mot y tinh la phu khi ban tom tat noi duoc phan cot loi (chu the + hanh dong);
# thieu chi tiet phu van tinh la phu. Cau chu thich anh khong phai noi dung bai nen khong tinh y.
Y_CHINH = {x["tho"]: x["y_chinh"] for x in VI_DU_CUOI if x.get("y_chinh")}

# Khoa lay thang tu pipeline.TEN_TANG de hai noi khong bao gio lech nhau.
MO_TA = dict(zip(P.TEN_TANG, (
    "Tầng 0 — ba câu đầu bài. Mốc ngây thơ, nhưng tin tức viết theo tháp ngược nên rất khó vượt.",
    "Tầng 1 — xếp hạng câu theo độ trung tâm. Thua Lead-3 trên bộ này.",
    "Có giám sát, chọn 2 câu. PhoBERT đã fine-tune cho điểm từng câu.",
    "Sinh câu mới. Đứng đầu ROUGE, trôi chảy, nhưng thiếu ý và có thể sai chi tiết.",
    "Lai: lọc câu cho vừa 1.024 token rồi để BARTpho viết lại.",
)))
assert len(MO_TA) == len(P.TEN_TANG), "Số mô tả không khớp số tầng."

CSS = (ROOT / "app" / "giao_dien.css").read_text(encoding="utf-8")
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Be+Vietnam+Pro:wght@400;500;600;700&'
    'family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">'
)
# Font phai boc trong gr.themes.Font: Gradio 6 so sanh theme voi cac theme dung san khi khoi
# dong, va chuoi thuan lam phep so sanh do vo ("'str' object has no attribute 'name'").
# Font() khong tu tai ve — chu da duoc nap bang the <link> trong HEAD.
_F = gr.themes.Font
THEME = gr.themes.Base(
    font=[_F("Be Vietnam Pro"), _F("Segoe UI"), _F("system-ui"), _F("sans-serif")],
    font_mono=[_F("ui-monospace"), _F("Consolas"), _F("monospace")],
)

# Nen sang la mac dinh -- bang mau hong la thiet ke chinh; nen toi chi khi nguoi xem tu chon.
# Dat class tren ca <html> lan <body> vi moi ban Gradio gan class `dark` mot cho khac nhau.
JS_NAP = """
() => {
  let toi = false;
  try { toi = localStorage.getItem('dl-nen') === 'toi'; } catch (e) {}
  for (const el of [document.documentElement, document.body]) el.classList.toggle('dark', toi);
}
"""
JS_DOI_NEN = """
() => {
  const toi = !document.documentElement.classList.contains('dark');
  for (const el of [document.documentElement, document.body]) el.classList.toggle('dark', toi);
  try { localStorage.setItem('dl-nen', toi ? 'toi' : 'sang'); } catch (e) {}
}
"""


# --------------------------------------------------------------------------
# So lieu tren `test`: doc tu bang ket qua, khong go tay
# --------------------------------------------------------------------------

def _so_test():
    """Chỉ số `test` của hệ thống cuối và BARTpho, lấy từ bảng đã chấm.

    Đọc file thay vì gõ số vào giao diện: bảng chấm lại thì trang tự đổi theo, và không có
    đường nào để demo nói một đằng còn báo cáo một nẻo.
    """
    mac_dinh = {"cuoi": {"la": 0.15, "recall": 57.4}, "bartpho": {"la": 10.0, "recall": 35.1}}
    try:
        bang = json.loads(
            (ROOT / "results" / "tables" / "chinh_xac_test.json").read_text(encoding="utf-8"))
        hang = {r["name"]: r["corpus"] for r in bang}
        lay = lambda t: {"la": hang[t]["co_chi_tiet_la"]["mean"],
                         "recall": hang[t]["r1_recall"]["mean"]}
        return {"cuoi": lay("chon-cau"), "bartpho": lay("bartpho-syllable-train_20k")}
    except Exception:
        return mac_dinh


SO = _so_test()


def _so_vn(x, chu_so=None):
    """Số kiểu Việt Nam. Mặc định: dưới 1 lấy hai chữ số thập phân, từ 1 trở lên lấy một."""
    if chu_so is None:
        chu_so = 2 if abs(x) < 1 else 1
    return f"{x:.{chu_so}f}".replace(".", ",")


def _hero():
    """Kết luận một câu, rồi hai thanh vẽ **đúng tỷ lệ** — chênh 67 lần phải nhìn thấy được."""
    bart, cuoi = SO["bartpho"]["la"], SO["cuoi"]["la"]
    rong = lambda v: max(v / max(bart, cuoi, 1e-9) * 100, 0.35)
    hang = lambda ten, v, sach: (
        f"<div class='thanh-hang{' sach' if sach else ''}'>"
        f"<span class='thanh-ten'>{ten}</span>"
        f"<span class='thanh-ray'><span class='thanh-day' style='width:{rong(v):.2f}%'></span></span>"
        f"<span class='thanh-so'>{_so_vn(v)}%</span></div>"
    )
    return (
        "<section class='hero'>"
        "<h1>Điểm cao nhất chưa chắc là bản tóm tắt tốt nhất</h1>"
        "<p class='dan'>Trên 2.000 bài <code>test</code>, BARTpho đã fine-tune đứng đầu ROUGE. "
        "Nhưng cứ mười bản tóm tắt của nó thì một bản có tên riêng hoặc con số "
        "<b>không hề có trong bài gốc</b>. Hệ thống cuối chọn câu từ chính bài báo, nên không có "
        "chỗ để bịa.</p>"
        "<div class='thanh'>"
        "<p class='thanh-de'>Tỷ lệ bản tóm tắt chứa chi tiết không có trong bài gốc — vẽ đúng tỷ lệ</p>"
        + hang("BARTpho", bart, False)
        + hang("Hệ thống cuối", cuoi, True)
        + f"<p class='thanh-nguon'>Cùng 2.000 bài đó, độ đủ ý (ROUGE-1 recall): hệ thống cuối "
          f"{_so_vn(SO['cuoi']['recall'], 1)} so với BARTpho {_so_vn(SO['bartpho']['recall'], 1)}. "
          f"Số đọc từ <code>results/tables/chinh_xac_test.json</code>.</p>"
        "</div></section>"
    )


GIOI_HAN = (
    "<div class='ghi'>"
    "<p><b>Đọc vệt vàng thế nào.</b> <span class='vd'>Vệt vàng</span> = tên riêng hoặc con số "
    "<i>không có trong bài gốc</i>, theo đúng bộ đo dùng để chấm.</p>"
    "<p>Bộ đo <b>bắt thừa</b> đôi chỗ (hai con số hai bên dấu chấm bị ghép làm một) và "
    "<b>không bắt được</b> lỗi gán nhầm đối tượng — con số có thật trong bài nhưng gắn sai chỗ.</p>"
    "<p>Hệ thống cuối chép nguyên câu nên <b>không bịa chi tiết</b>. Nhưng ghép câu vẫn có thể làm "
    "mất ngữ cảnh: chữ &ldquo;cùng ngày&rdquo; có thể bị đọc thành một ngày khác.</p>"
    "<p><b>Đọc &ldquo;phủ x/y ý&rdquo; thế nào.</b> Danh sách ý của mỗi bài do người gán tay nên "
    "chỉ có ở bài mẫu. Một ý tính là phủ khi bản tóm tắt nói được phần cốt lõi; thiếu chi tiết phụ "
    "vẫn tính là phủ.</p>"
    "<p>Đây là trục <i>độc lập</i> với vệt vàng, và nó chấm cả hệ thống cuối: một bản "
    "<b>không bịa gì vẫn có thể bỏ sót quá nửa số ý</b>. Trích rút chép nguyên câu nên phải trả "
    "trọn giá một câu để lấy một ý, và hết ngân sách trước khi đọc tới cuối bài.</p>"
    "</div>"
)


# --------------------------------------------------------------------------
# Cac manh giao dien
# --------------------------------------------------------------------------

def _rong(chu):
    return f"<div class='rong'>{chu}</div>"


def _the(ten, cach, x, phu=None):
    """Một bản tóm tắt, chấm trên HAI trục.

    Viền dọc và verdict mang trục thứ nhất (có bịa chi tiết không); vệt vàng nằm sẵn trong
    `x['html']`. `phu` là trục thứ hai — (số ý phủ được, tổng số ý, danh sách ý bỏ sót) — và
    chỉ có ở bài mẫu vì danh sách ý phải gán tay.

    Hai trục cố tình tách rời nhau: bản không bịa gì vẫn có thể bỏ sót quá nửa số ý, và demo
    phải nói ra điều đó thay vì chỉ khoe mặt mạnh của hệ thống cuối.
    """
    la = x["so_la"]
    lop, dau = ("bp--sach", "Không có chi tiết lạ") if la == 0 else ("bp--la", f"{la} chi tiết lạ")
    chip = f"<span class='chip'>{x['am_tiet']} âm tiết</span>"
    sot_html = ""
    if phu:
        n, tong, sot = phu
        # Chip dam hon khi con thieu y: danh dau mot khang dinh, nhung KHONG dung mau bao hoa --
        # mau do/xanh da danh cho truc chi tiet la, dung de hai truc lan mau nhau.
        chip += (f"<span class='chip{'' if n == tong else ' chip--thieu'}'>"
                 f"Phủ {n}/{tong} ý</span>")
        if sot:
            sot_html = ("<p class='bp-thieu'><b>Bỏ sót:</b> "
                        + "; ".join(html.escape(y) for y in sot) + ".</p>")
    return (
        f"<article class='bp {lop}'>"
        f"<header class='bp-dau'><h3 class='bp-ten'>{ten}</h3>"
        f"<span class='bp-cach'>{cach}</span>"
        f"<span class='bp-verdict'>{dau}</span></header>"
        f"<p class='bp-van'>{x['html']}</p>"
        f"<div class='bp-chan'>{chip}</div>{sot_html}"
        f"</article>"
    )


def _phu(bai, he):
    """(số ý phủ được, tổng số ý, ý bỏ sót) của tầng `he` trên `bai` — None nếu bài chưa gán ý."""
    ys = Y_CHINH.get(bai)
    if not ys:
        return None
    sot = [y["y"] for y in ys if not y[he]]
    return len(ys) - len(sot), len(ys), sot


def _the_cho(ten, cach):
    """Khung chờ: chạy mất 15–25 giây trên CPU, không được để màn hình đứng im."""
    return (
        f"<article class='bp bp--cho'>"
        f"<header class='bp-dau'><h3 class='bp-ten'>{ten}</h3>"
        f"<span class='bp-cach'>{cach}</span>"
        f"<span class='bp-verdict'>Đang chạy trên CPU</span></header>"
        f"<p class='bp-van'>Đang nạp mô hình và tóm tắt bài báo. Lần chạy đầu lâu nhất vì phải "
        f"nạp trọng số từ đĩa; những lần sau mô hình được giữ lại trong bộ nhớ.</p>"
        f"<div class='bp-chan'><span class='chip'>25–35 giây</span></div>"
        f"</article>"
    )


def _the_loi(ten, e):
    """Lỗi hiện ngay trong khung kết quả, kèm cách sửa — không để traceback rơi ra toast."""
    thieu_ckpt = isinstance(e, FileNotFoundError) or "0 byte" in str(e) or "RỖNG" in str(e)
    cach_sua = (
        "Tải trọng số về <code>~/.cache/dl-summarisevn/</code> — xem README, mục "
        "&ldquo;Demo Gradio&rdquo;, phần lấy checkpoint bằng kernel <code>xuat-ckpt</code>."
        if thieu_ckpt else
        "Kiểm tra môi trường <code>~/.venvs/demo</code>: torch bản CPU, "
        "<code>transformers==5.0.0</code>, <code>sentencepiece</code>, <code>underthesea</code>."
    )
    ly_do = "Chưa nạp được mô hình" if thieu_ckpt else "Chạy hỏng giữa chừng"
    return (
        f"<article class='bp bp--loi'>"
        f"<header class='bp-dau'><h3 class='bp-ten'>{ten}</h3>"
        f"<span class='bp-verdict'>{ly_do}</span></header>"
        f"<p class='bp-van bp-van--loi'>{cach_sua}</p>"
        f"<div class='bp-chan'><span class='chip'>{type(e).__name__}</span></div>"
        f"</article>"
    )


def _ma(chu):
    """Escape rồi cho `mã` trong dấu huyền thành <code> — ghi chú có nhắc tên tệp và tên split."""
    return re.sub(r"`([^`]+)`", r"<code></code>", html.escape(chu))


def _ghi_nguon(bai, seg):
    nguon = (
        "Bài mẫu, chạy trên đúng dạng tách từ của bộ dữ liệu: hệ thống cuối trùng bản đã báo cáo, "
        "BARTpho trùng bản nạp checkpoint từ đĩa — cùng đường với lần chấm <code>test</code>."
        if seg else
        "Bài tự dán, tách từ bằng underthesea. Khác bộ tách từ của dữ liệu nên hệ thống cuối có thể "
        "chọn câu khác bản báo cáo. <b>Trục phủ ý không hiện ở bài tự dán</b> vì danh sách ý phải "
        "gán tay."
    )
    bai_mau = next((x for x in VI_DU_CUOI if x["tho"] == bai), {})
    loi = bai_mau.get("loi_da_xac_nhan", "")
    them = f"<p><b>Lỗi đã được xác nhận ở bài này:</b> {html.escape(loi)}</p>" if loi else ""
    ghi_y = bai_mau.get("y_ghi_chu", "")
    if ghi_y:
        them += f"<p><b>Cách gán ý ở bài này:</b> {html.escape(ghi_y)}</p>"
    # Noi thang vi sao bai nay nam trong bo mau. Bo mau ba bai thi de bi ngo la chon bai co loi
    # cho minh, nen ly do chon phai hien ngay tren trang chu khong nam trong dau nguoi lam.
    vi_sao = bai_mau.get("vi_sao_chon", "")
    if vi_sao:
        them += f"<p><b>Vì sao bài này nằm trong bộ mẫu:</b> {_ma(vi_sao)}</p>"
    return f"<div class='ghi'><p>{nguon}</p>{them}</div>"


TEN_CUOI = "Hệ thống cuối"
CACH_CUOI = "PhoBERT chọn câu, chép nguyên văn từ bài"
TEN_BART = "BARTpho"
CACH_BART = "Mô hình sinh, đứng đầu ROUGE"


def _kiem(bai):
    if not bai or not bai.strip():
        raise gr.Error("Hãy dán một bài báo vào ô bên trái.")
    if len(bai.split()) < 30:
        raise gr.Error("Bài quá ngắn để tóm tắt — hãy dán một bài dài hơn (từ 30 chữ).")


def cho_cuoi(bai):
    """Bước một của tab chính: kiểm đầu vào rồi dựng khung chờ, trước khi mô hình khởi động."""
    _kiem(bai)
    return _the_cho(TEN_CUOI, CACH_CUOI), _the_cho(TEN_BART, CACH_BART), ""


def chay_cuoi(bai):
    seg = TACH_TU_MAU.get(bai)
    try:
        kq = P.so_sanh_cuoi(bai, seg)
    except Exception as e:  # loi hien trong khung ket qua, kem cach sua
        return _the_loi(TEN_CUOI, e), _the_loi(TEN_BART, e), ""
    return (_the(TEN_CUOI, CACH_CUOI, kq["cuoi"], _phu(bai, "cuoi")),
            _the(TEN_BART, CACH_BART, kq["bartpho"], _phu(bai, "bartpho")),
            _ghi_nguon(bai, seg))


def _hang_tang(so, ten, mo, than):
    return (f"<div class='tang'><span class='tang-so'>{so}</span><div>"
            f"<h3 class='tang-ten'>{ten}</h3><p class='tang-mo'>{mo}</p>{than}</div></div>")


def _tang_rong(so, ten, mo):
    return _hang_tang(so, ten, mo, "<p class='tang-van tang-van--rong'>Chưa chạy</p>")


def _tang_cho(so, ten, mo):
    return _hang_tang(so, ten, mo, "<p class='tang-van tang-van--rong'>Đang chạy…</p>")


def _tang_ket_qua(so, ten, mo, van):
    loi = van.startswith("[lỗi]")
    lop = "tang-van tang-van--loi" if loi else "tang-van"
    return _hang_tang(so, ten, mo, f"<p class='{lop}'>{html.escape(van)}</p>")


# Ten tang trong TEN_TANG co san tien to "Tang N — "; so tang da nam o le trai nen bo di khi hien.
def _ten_gon(ten):
    return re.sub(r"^Tầng \d+ — ", "", ten)


def cho_bon_tang(bai):
    _kiem(bai)
    return [_tang_cho(i, _ten_gon(t), MO_TA[t]) for i, t in enumerate(MO_TA)]


def chay_bon_tang(bai):
    # `tom_tat_tat_ca` bat loi cua tung tang, nhung khau tach tu chay TRUOC vong lap do:
    # thieu underthesea la hong ca nam tang, nen bat o day de bao trong tung hang.
    try:
        kq = P.tom_tat_tat_ca(bai)
    except Exception as e:
        kq = {t: f"[lỗi] {type(e).__name__}: {e}" for t in MO_TA}
    return [_tang_ket_qua(i, _ten_gon(t), MO_TA[t], kq.get(t, ""))
            for i, t in enumerate(MO_TA)]


# Gradio 6 chuyen css/head/theme/js tu Blocks sang launch(); dat o day thi con canh bao.
with gr.Blocks(title="Tóm tắt tin tức tiếng Việt — đủ ý và không bịa") as demo:
    with gr.Row(equal_height=False):
        with gr.Column(scale=9):
            gr.HTML(_hero())
        with gr.Column(scale=1, min_width=128):
            nut_nen = gr.Button("Sáng / tối", size="sm", elem_classes=["nut-nen"])
            nut_nen.click(fn=None, js=JS_DOI_NEN)

    with gr.Tab("Hệ thống cuối và BARTpho"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                c_bai = gr.Textbox(label="Bài báo", lines=18,
                                   placeholder="Dán nội dung bài báo vào đây…")
                with gr.Row():
                    c_nut = gr.Button("Tóm tắt và đối chiếu", variant="primary", scale=3)
                    gr.ClearButton(c_bai, value="Xoá", scale=1)
                gr.Examples([[x["tho"]] for x in VI_DU_CUOI], inputs=c_bai,
                            example_labels=[x["tieu_de"] for x in VI_DU_CUOI],
                            label="Bài mẫu từ `val` — hai bài BARTpho bịa chi tiết, một bài nó sạch")
            with gr.Column(scale=5):
                c_cuoi = gr.HTML(_rong(
                    "<b>Hai bản tóm tắt sẽ hiện ở đây.</b><br>Dán một bài báo, hoặc chọn một bài "
                    "mẫu bên trái, rồi bấm <b>Tóm tắt và đối chiếu</b>."))
                c_bart = gr.HTML()
                c_ghi = gr.HTML()
                gr.HTML(GIOI_HAN)
        c_nut.click(cho_cuoi, inputs=c_bai, outputs=[c_cuoi, c_bart, c_ghi],
                    show_progress="hidden").then(
            chay_cuoi, inputs=c_bai, outputs=[c_cuoi, c_bart, c_ghi], show_progress="hidden")

    with gr.Tab("Bốn tầng (phần phân tích)"):
        gr.HTML(
            "<div class='ghi'><p>Năm hướng đã thử ở tuần 1–8, chạy cạnh nhau trên cùng một bài. "
            "<b>Không ô nào là &ldquo;đáp án đúng&rdquo;</b>: extractive gần như không sai sự thật "
            "nhưng đứt mạch và bỏ sót ý, abstractive trôi chảy nhưng thiếu ý và có thể hoán đổi "
            "chi tiết.</p></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                o_bai = gr.Textbox(label="Bài báo", lines=18,
                                   placeholder="Dán nội dung bài báo vào đây…")
                with gr.Row():
                    nut = gr.Button("Chạy tất cả các tầng", variant="primary", scale=3)
                    gr.ClearButton(o_bai, value="Xoá", scale=1)
                gr.Examples([[VI_DU]], inputs=o_bai,
                            example_labels=["Vụ trộm ở TP Long Khánh, Đồng Nai"],
                            label="Bài mẫu")
            with gr.Column(scale=5):
                o_ra = [gr.HTML(_tang_rong(i, _ten_gon(t), MO_TA[t]))
                        for i, t in enumerate(MO_TA)]
        nut.click(cho_bon_tang, inputs=o_bai, outputs=o_ra, show_progress="hidden").then(
            chay_bon_tang, inputs=o_bai, outputs=o_ra, show_progress="hidden")


if __name__ == "__main__":
    demo.launch(css=CSS, head=HEAD, theme=THEME, js=JS_NAP)
