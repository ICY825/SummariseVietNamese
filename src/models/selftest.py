"""Tự kiểm tra tầng 0 và tầng 1. Chạy lại sau mỗi lần sửa `extractive.py`.

    .venv/Scripts/python.exe src/models/selftest.py

Không nạp dữ liệu thật: mọi bài kiểm tra ở đây dùng văn bản tí hon tự chế, và mọi giá
trị kỳ vọng đều tính tay được trên giấy — kể cả điểm PageRank. Nhờ vậy file này chạy
trong một giây và không phụ thuộc mạng, nên chạy được sau từng lần sửa.

Trọng tâm không phải "hàm có chạy không" mà là ba bất biến dễ vỡ nhất, hỏng cái nào
cũng khiến bảng kết quả sai một cách khó nhận ra:

  1. Thứ tự câu trong bài phải được giữ nguyên ở đầu ra.
  2. Đầu ra phải còn ở dạng TÁCH TỪ (còn gạch dưới) — khử tách từ là việc của
     `eval.report`, làm hai lần thì hỏng dạng chấm điểm.
  3. Mọi bộ chọn phải tất định: chạy lại phải ra đúng kết quả cũ.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import sentences  # noqa: E402
from models.extractive import (  # noqa: E402
    _tfidf,
    join,
    lead,
    lexrank_indices,
    oracle,
    oracle_indices,
    pagerank,
    random_k,
    textrank,
    textrank_indices,
)

fails = []


def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'HỎNG'}] {name}: {got:.6f} (kỳ vọng {want:.6f})")
    if not ok:
        fails.append(name)


def check_true(name, cond):
    print(f"  [{'OK ' if cond else 'HỎNG'}] {name}")
    if not cond:
        fails.append(name)


# Van ban thu o dang DA TACH TU, dau cau dung rieng - dung nhu bo VietNews.
A = "Cảnh_sát bắt nghi_phạm . Toà tuyên_án hôm_qua . Nghi_phạm khai nhận ."
SA = sentences(A)

print("1. Lead-k và phép ghép câu")
check_true("cắt được đúng 3 câu", len(SA) == 3)
check_true("Lead-1 lấy câu đầu", lead(A, 1) == SA[0])
check_true("Lead-2 lấy hai câu đầu", lead(A, 2) == SA[0] + " " + SA[1])
check_true("k lớn hơn số câu -> lấy hết", lead(A, 99) == A)
check_true("ghép luôn theo thứ tự bài", join(SA, [2, 0]) == SA[0] + " " + SA[2])
check_true("đầu ra GIỮ dạng tách từ (còn gạch dưới)", "_" in lead(A, 1))
check_true("bài rỗng không làm vỡ", lead("", 3) == "")

print("\n2. Random-k tất định theo từng bài")
check_true("cùng key -> cùng kết quả", random_k(A, 2, key="g1") == random_k(A, 2, key="g1"))
check_true("lấy đúng k câu", len(sentences(random_k(A, 2, key="g1"))) == 2)
check_true(
    "câu chọn ra đều là câu của bài",
    set(sentences(random_k(A, 2, key="g7"))) <= set(SA),
)
check_true(
    "giữ thứ tự bài (không đảo)",
    [SA.index(s) for s in sentences(random_k(A, 2, key="g7"))] == sorted(
        [SA.index(s) for s in sentences(random_k(A, 2, key="g7"))]
    ),
)
check_true(
    "key khác nhau -> không phải lúc nào cũng cùng một câu",
    len({random_k(A, 1, key=f"g{i}") for i in range(30)}) >= 2,
)

print("\n3. Oracle bám đúng đáp án")
B = "Trời mưa to . Cảnh_sát bắt nghi_phạm ma_tuý ở Hà_Nội . Giá vàng tăng ."
REF = "Cảnh_sát bắt nghi_phạm ma_tuý ở Hà_Nội ."
check_true("chọn đúng câu trùng khít sapo", oracle_indices(B, REF, 3) == [1])
check_true("dừng sớm, không nhồi thêm câu vô ích", oracle(B, REF, 3) == sentences(B)[1])
check_true("k=1 vẫn ra đúng câu đó", oracle_indices(B, REF, 1) == [1])
check_true("bài rỗng không làm vỡ", oracle("", REF) == "")

print("\n4. PageRank (tính tay được)")
p = pagerank(np.ones((4, 4)))
check("đồ thị đầy đủ -> đều nhau", float(p[0]), 0.25, tol=1e-6)
check("tổng bằng 1", float(p.sum()), 1.0, tol=1e-9)
p0 = pagerank(np.zeros((4, 4)))
check("không có cạnh nào -> đều nhau", float(p0[0]), 0.25, tol=1e-6)
# Do thi hinh sao: nut 0 noi voi 1, 2, 3; cac la chi noi voi 0.
#   c = 0,0375 + 2,55 l   va   l = 0,0375 + c/3 * 0,85
#   -> c = 0,133125 / 0,2775 = 0,479730
star = np.zeros((4, 4))
star[0, 1:] = star[1:, 0] = 1.0
ps = pagerank(star)
check("hình sao: tâm", float(ps[0]), 0.479730, tol=1e-5)
check("hình sao: lá", float(ps[1]), 0.173423, tol=1e-5)
check_true("tâm cao hơn lá", ps[0] > ps[1])
check_true("ma trận rỗng không làm vỡ", pagerank(np.zeros((0, 0))).size == 0)

print("\n5. TF-IDF")
X = _tfidf(["an ninh", "an ninh", "giá vàng"])
check("hàng đã chuẩn hoá L2", float(np.linalg.norm(X[0])), 1.0, tol=1e-9)
check("hai câu giống hệt -> cosin 1", float(X[0] @ X[1]), 1.0, tol=1e-9)
check("hai câu không chung từ -> cosin 0", float(X[0] @ X[2]), 0.0, tol=1e-9)
check_true("không có từ nào -> không làm vỡ", _tfidf([". ."]).shape[0] == 1)

print("\n6. TextRank và LexRank")
C = ("cảnh_sát bắt nghi_phạm . cảnh_sát bắt nghi_phạm . "
     "cảnh_sát bắt nghi_phạm . giá vàng tăng .")
check_true("TextRank bỏ câu lạc đề", textrank_indices(C, 3) == [0, 1, 2])
check_true("LexRank bỏ câu lạc đề", lexrank_indices(C, 3) == [0, 1, 2])
check_true("TextRank tất định", textrank(C, 3) == textrank(C, 3))
check_true("k lớn hơn số câu -> lấy hết", textrank_indices(A, 99) == [0, 1, 2])
D = "một hai . một hai . một hai . một hai ."
check_true("hoà điểm -> ưu tiên câu đứng trước", textrank_indices(D, 2) == [0, 1])
check_true("bài rỗng không làm vỡ", textrank("", 3) == "" and lexrank_indices("", 3) == [])
check_true("đầu ra TextRank giữ dạng tách từ", "_" in textrank(C, 1))

print("\n" + ("THẤT BẠI: " + ", ".join(fails) if fails else "TẤT CẢ ĐỀU ĐẠT."))
raise SystemExit(1 if fails else 0)
