"""Tự kiểm tra module đánh giá. Chạy lại sau mỗi lần sửa `rouge.py` hay `stats.py`.

    .venv/Scripts/python.exe src/eval/selftest.py

Các giá trị kỳ vọng ở đây tính được bằng tay trên giấy, nên file này không phụ thuộc
thư viện ngoài nào. Bốn chỉ số đã được đối chiếu khớp tuyệt đối với `rouge_score` của
Google trên 600 cặp thật (tokenizer tuỳ biến, vì bộ tách token mặc định của họ xoá sạch
dấu tiếng Việt); phép đối chiếu đó không nằm trong đây vì `rouge-score` không phải phụ
thuộc của dự án.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.rouge import score, sentences_raw  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

fails = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'HỎNG'}] {name}: {got:.4f} (kỳ vọng {want:.4f})")
    if not ok:
        fails.append(name)


def check_true(name, cond):
    print(f"  [{'OK ' if cond else 'HỎNG'}] {name}")
    if not cond:
        fails.append(name)


print("1. ROUGE tính tay được")
# ref 5 am tiet, hyp 5 am tiet, trung 4 unigram / 3 bigram / LCS dai 4
s = score("con mèo ngồi trên chiếu", "con mèo ngồi trên thảm")
check("ROUGE-1 (4/5 khớp)", s["rouge1"], 80.0)
check("ROUGE-2 (3/4 khớp)", s["rouge2"], 75.0)
check("ROUGE-L (LCS=4)", s["rougeL"], 80.0)
check("Giống hệt nhau -> 100", score("con mèo", "con mèo")["rouge1"], 100.0)
check("Không trùng gì -> 0", score("xe đạp", "con mèo")["rouge1"], 0.0)

print("\n2. BẤT BIẾN CỐT LÕI: dạng tách từ và dạng thô phải cho ĐIỂM Y HỆT")
seg = "Học_sinh trường Trần_Phú đã khởi_hành ."
raw = "Học sinh trường Trần Phú đã khởi hành."
check("tách từ vs tách từ", score(seg, seg)["rouge1"], 100.0)
check("thô vs tách từ  ", score(raw, seg)["rouge1"], 100.0)
check("tách từ vs thô  ", score(seg, raw)["rouge1"], 100.0)
ref = "Cảnh_sát đã khởi_tố vụ án ."
check_true(
    "mọi tổ hợp dạng cho cùng một điểm",
    len({round(score(h, r)["rouge2"], 9)
         for h in (seg, raw) for r in (ref, ref.replace("_", " "))}) == 1,
)

print("\n3. Cắt câu thô chặn được viết tắt")
check_true(
    "'TP. Hưng Yên' không bị tách đôi",
    sentences_raw("Công an TP. Hưng Yên cho biết. Vụ án đã khởi tố.")
    == ["Công an TP. Hưng Yên cho biết.", "Vụ án đã khởi tố."],
)

print("\n4. Bootstrap")
vals = [10.0] * 100
m, lo, hi = bootstrap_ci(vals, n_boot=500)
check("hằng số -> khoảng tin cậy suy biến", hi - lo, 0.0)
r = paired_bootstrap([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_boot=500)
check("so với chính nó -> chênh lệch 0", r["diff"], 0.0)
check_true("so với chính nó -> không có ý nghĩa", not r["significant"])
r = paired_bootstrap([9.0] * 60, [1.0] * 60, n_boot=500)
check("chênh lệch rõ ràng", r["diff"], 8.0)
check_true("chênh lệch rõ ràng -> có ý nghĩa", r["significant"] and r["p"] < 0.01)
try:
    paired_bootstrap([1.0, 2.0], [1.0])
    check_true("lệch số bài thì phải báo lỗi", False)
except ValueError:
    check_true("lệch số bài thì phải báo lỗi", True)

print("\n" + ("THẤT BẠI: " + ", ".join(fails) if fails else "TẤT CẢ ĐỀU ĐẠT."))
raise SystemExit(1 if fails else 0)
