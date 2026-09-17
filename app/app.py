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
"""

import json
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

# Khoa lay thang tu pipeline.TEN_TANG de hai noi khong bao gio lech nhau.
MO_TA = dict(zip(P.TEN_TANG, (
    "Tầng 0 — ba câu đầu bài. Mốc ngây thơ, nhưng tin tức viết theo tháp ngược nên rất khó vượt.",
    "Tầng 1 — xếp hạng câu theo độ trung tâm. Thua Lead-3 trên bộ này.",
    "Có giám sát, chọn 2 câu. PhoBERT đã fine-tune cho điểm từng câu.",
    "Sinh câu mới. Đứng đầu ROUGE, trôi chảy, nhưng thiếu ý và có thể sai chi tiết.",
    "Lai: lọc câu cho vừa 1.024 token rồi để BARTpho viết lại.",
)))
assert len(MO_TA) == len(P.TEN_TANG), "Số mô tả không khớp số tầng."

GIOI_HAN = (
    "**Đọc phần tô vàng thế nào.** Tô vàng = tên riêng hoặc con số *không có trong bài gốc*, theo "
    "đúng bộ đo dùng để chấm. Bộ đo **bắt thừa** đôi chỗ (hai con số hai bên dấu chấm bị ghép làm "
    "một) và **không bắt được** lỗi gán nhầm đối tượng (con số có thật trong bài nhưng gắn sai chỗ). "
    "Hệ thống cuối chép nguyên câu nên **không bịa chi tiết** — nhưng ghép câu vẫn có thể làm mất "
    "ngữ cảnh (\"cùng ngày\" đọc thành một ngày khác)."
)


def _kiem(bai):
    if not bai or not bai.strip():
        raise gr.Error("Hãy dán một bài báo vào ô bên trái.")
    if len(bai.split()) < 30:
        raise gr.Error("Bài quá ngắn để tóm tắt — hãy dán một bài dài hơn (từ 30 chữ).")


def _o(ten, x):
    dem = "không có chi tiết lạ" if x["so_la"] == 0 else f"<b>{x['so_la']} chi tiết lạ</b> (tô vàng)"
    return (f"<div class='o'><h3>{ten}</h3><p class='tt'>{x['html']}</p>"
            f"<p class='nho'>{x['am_tiet']} âm tiết · {dem}</p></div>")


def chay_cuoi(bai):
    _kiem(bai)
    seg = TACH_TU_MAU.get(bai)
    kq = P.so_sanh_cuoi(bai, seg)
    nguon = ("Bài mẫu — chạy trên đúng dạng tách từ của bộ dữ liệu: hệ thống cuối trùng bản đã báo cáo; "
             "BARTpho trùng bản nạp checkpoint từ đĩa (cùng đường với lần chấm `test`)."
             if seg else "Bài tự dán — tách từ bằng underthesea, có thể lệch nhẹ so với số liệu báo cáo.")
    loi = next((x["loi_da_xac_nhan"] for x in VI_DU_CUOI if x["tho"] == bai), "")
    return (_o("Hệ thống cuối — PhoBERT chọn câu có chủ đích", kq["cuoi"]),
            _o("BARTpho — mô hình sinh, đứng đầu ROUGE", kq["bartpho"]),
            f"*{nguon}*" + (f"\n\n**Lỗi đã được xác nhận ở bài này:** {loi}" if loi else ""))


def chay_bon_tang(bai):
    _kiem(bai)
    kq = P.tom_tat_tat_ca(bai)
    return [kq.get(t, "") for t in MO_TA]


CSS = (".o{border:1px solid #bbb;border-radius:8px;padding:.6rem 1rem;margin-bottom:.8rem}"
       ".o h3{margin:.2rem 0 .5rem}.tt{font-size:1.05rem;line-height:1.6}"
       ".nho{color:#666;font-size:.9rem}mark{background:#ffe066;padding:0 .15rem;border-radius:3px}")

with gr.Blocks(title="DL-SummariseVN — tóm tắt tin tức tiếng Việt", css=CSS) as demo:
    gr.Markdown(
        "# Tóm tắt tin tức tiếng Việt — đủ ý và không bịa\n"
        "Mô hình sinh (BARTpho) đứng đầu ROUGE, nhưng trên `test` **10%** bản tóm tắt của nó có tên "
        "riêng hoặc con số không có trong bài. Hệ thống cuối chọn câu có chủ đích: đủ ý hơn và "
        "chỉ **0,15%**."
    )
    with gr.Tab("Hệ thống cuối và BARTpho"):
        with gr.Row():
            with gr.Column(scale=1):
                c_bai = gr.Textbox(label="Bài báo", lines=18, placeholder="Dán nội dung bài báo…")
                with gr.Row():
                    c_nut = gr.Button("Tóm tắt", variant="primary")
                    gr.ClearButton(c_bai, value="Xoá")
                gr.Examples([[x["tho"]] for x in VI_DU_CUOI], inputs=c_bai,
                            label="Bài mẫu (val) — BARTpho có lỗi đã được xác nhận")
            with gr.Column(scale=1):
                c_cuoi = gr.HTML()
                c_bart = gr.HTML()
                c_ghi = gr.Markdown()
                gr.Markdown(GIOI_HAN)
        c_nut.click(chay_cuoi, inputs=c_bai, outputs=[c_cuoi, c_bart, c_ghi])

    with gr.Tab("Bốn tầng (phần phân tích)"):
        gr.Markdown("Các hướng đã thử ở tuần 1–8, chạy cạnh nhau trên cùng một bài. "
                    "*Không có ô nào là \"đáp án đúng\": mỗi hướng mạnh và yếu ở chỗ khác nhau.*")
        with gr.Row():
            with gr.Column(scale=1):
                o_bai = gr.Textbox(label="Bài báo", lines=18, placeholder="Dán nội dung bài báo…")
                with gr.Row():
                    nut = gr.Button("Tóm tắt", variant="primary")
                    gr.ClearButton(o_bai, value="Xoá")
                gr.Examples([[VI_DU]], inputs=o_bai, label="Bài mẫu")
            with gr.Column(scale=1):
                o_ra = [gr.Textbox(label=t, info=MO_TA[t], lines=3) for t in MO_TA]
        nut.click(chay_bon_tang, inputs=o_bai, outputs=o_ra)


if __name__ == "__main__":
    demo.launch()
