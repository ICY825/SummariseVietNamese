"""Demo Gradio: dán một bài báo tiếng Việt, xem bốn tầng tóm tắt nó khác nhau thế nào.

    ~/.venvs/demo/Scripts/python.exe app/app.py

Mục đích của demo là cho thấy **hồ sơ lỗi của hai hướng bù trừ nhau** — đúng kết luận
của tuần 7: extractive gần như không sai sự thật nhưng đứt mạch văn và bỏ sót ý, còn
abstractive đọc trôi chảy nhưng thiếu ý nhiều nhất và là hướng duy nhất hoán đổi chi
tiết. Vì thế các tầng hiện cạnh nhau chứ không có một ô "kết quả tốt nhất".
"""

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

# Khoa lay thang tu pipeline.TEN_TANG de hai noi khong bao gio lech nhau.
MO_TA = dict(zip(P.TEN_TANG, (
    "Tầng 0 — ba câu đầu bài. Mốc ngây thơ, nhưng tin tức viết theo tháp ngược nên rất khó vượt.",
    "Tầng 1 — xếp hạng câu theo độ trung tâm. Thua Lead-3 trên bộ này.",
    "Có giám sát, chọn 2 câu. Hệ extractive tốt nhất của đề tài.",
    "Sinh câu mới. Trôi chảy nhất, nhưng thiếu ý nhiều nhất và có thể sai chi tiết.",
    "Lai: lọc câu cho vừa 1.024 token rồi để BARTpho viết lại.",
)))
assert len(MO_TA) == len(P.TEN_TANG), "Số mô tả không khớp số tầng."


def chay(bai):
    if not bai or not bai.strip():
        raise gr.Error("Hãy dán một bài báo vào ô bên trái.")
    if len(bai.split()) < 30:
        raise gr.Error("Bài quá ngắn để tóm tắt — hãy dán một bài dài hơn (từ 30 chữ).")
    kq = P.tom_tat_tat_ca(bai)
    return [kq.get(t, "") for t in MO_TA]


with gr.Blocks(title="DL-SummariseVN — tóm tắt tin tức tiếng Việt") as demo:
    gr.Markdown(
        "# Tóm tắt tin tức tiếng Việt — bốn tầng\n"
        "Dán một bài báo rồi bấm **Tóm tắt**. Bốn hệ thống chạy trên cùng một bài để so sánh.\n\n"
        "*Không có ô nào là \"đáp án đúng\": mỗi hướng mạnh và yếu ở chỗ khác nhau.*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            o_bai = gr.Textbox(label="Bài báo", lines=18, placeholder="Dán nội dung bài báo…")
            with gr.Row():
                nut = gr.Button("Tóm tắt", variant="primary")
                gr.ClearButton(o_bai, value="Xoá")
            gr.Examples([[VI_DU]], inputs=o_bai, label="Bài mẫu")
        with gr.Column(scale=1):
            o_ra = [gr.Textbox(label=t, info=MO_TA[t], lines=3)
                    for t in MO_TA]

    nut.click(chay, inputs=o_bai, outputs=o_ra)


if __name__ == "__main__":
    demo.launch()
