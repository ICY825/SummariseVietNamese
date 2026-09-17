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
from data.text import for_scoring, sentences  # noqa: E402
from models.extractive import (  # noqa: E402
    _tfidf,
    join,
    lead,
    lexrank,
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
C = ("cảnh_sát bắt nghi_phạm ma_tuý . cảnh_sát khám_xét nhà nghi_phạm . "
     "nghi_phạm ma_tuý bị bắt hôm_qua . giá vàng tăng .")
check_true("TextRank bỏ câu lạc đề", textrank_indices(C, 3) == [0, 1, 2])
check_true("LexRank bỏ câu lạc đề", lexrank_indices(C, 3) == [0, 1, 2])
check_true("TextRank tất định", textrank(C, 3) == textrank(C, 3))
check_true("k lớn hơn số câu -> lấy hết", textrank_indices(A, 99) == [0, 1, 2])
# Tam giac doi xung: ba cau khac nhau nhung diem trung tam bang het nhau.
D = "một hai . hai ba . ba một ."
check_true("hoà điểm -> ưu tiên câu đứng trước", textrank_indices(D, 2) == [0, 1])
check_true("bài rỗng không làm vỡ", textrank("", 3) == "" and lexrank_indices("", 3) == [])
check_true("đầu ra TextRank giữ dạng tách từ", "_" in textrank(C, 1))

print("\n7. Khử câu trùng nội dung — chỉ ở tầng 1, không ở tầng 0")
# Cau 0 va cau 2 giong het nhau: chung co diem trung tam bang nhau nen bo xep hang
# se vo ca hai neu khong khu trung.
E = "cảnh_sát bắt nghi_phạm . giá vàng tăng mạnh . cảnh_sát bắt nghi_phạm ."
check_true("TextRank không chọn hai bản sao", len(textrank_indices(E, 3)) == 2)
check_true("LexRank không chọn hai bản sao", len(lexrank_indices(E, 3)) == 2)
for nhan, ra in (("TextRank", textrank(E, 3)), ("LexRank", lexrank(E, 3))):
    c = sentences(ra)
    check_true(f"{nhan}: đầu ra không lặp câu", len(c) == len(set(c)))
check_true(
    "Lead-3 CỐ Ý giữ nguyên câu lặp (mốc ngây thơ)",
    len(sentences(lead(E, 3))) == 3 and len(set(sentences(lead(E, 3)))) == 2,
)

print("\n8. LexRank bản nhúng — kiểm phần KHÔNG cần torch")
# `lexrank_emb_indices()` chi khac ban TF-IDF o dung mot cho: ma tran tuong dong lay
# tu PhoBERT thay vi tu TF-IDF. Phan con lai — pagerank(), _pick(), khu cau trung, giu
# thu tu cau — dung chung code voi ban TF-IDF. Do la phan kiem duoc o day: thay
# `_phobert_vectors` bang mot bo sinh vector tu che roi doi chieu voi ket qua tinh tay.
#
# Vi sao phai gia lap: PhoBERT can `torch` + `transformers`, ma `.venv` cua du an co y
# khong co hai goi do de lenh baseline chay duoc tren mot may sach. Neu doi den khi co
# torch moi kiem thi he thong nay mai mai khong co phep kiem tu dong nao — trong khi no
# lai la he thong DUY NHAT cua du an ma ban tom tat sinh lai khong chac ra dung nhu cu
# (phu thuoc trong so PhoBERT tren Hub va phien ban `transformers`).
import models.extractive as ex  # noqa: E402

_goc = ex._phobert_vectors
try:
    # Ba cau, vector da chuan hoa L2 san: cau 0 va 1 gan nhau (cosin 0,995), cau 2
    # truc giao voi ca hai (cosin 0). Bo xep hang phai bo dung cau 2.
    ex._phobert_vectors = lambda sents, *a, **k: np.array(
        [[1.0, 0.0], [0.995, 0.0998], [0.0, 1.0]]
    )[: len(sents)]
    F = "cảnh_sát bắt nghi_phạm . cảnh_sát khám nhà . giá vàng tăng ."
    check_true("bỏ câu lạc đề (vector tự chế)", ex.lexrank_emb_indices(F, 2) == [0, 1])
    check_true("giữ nguyên thứ tự câu", ex.lexrank_emb_indices(F, 3) == [0, 1, 2])
    check_true(
        "tất định: chạy lại ra đúng kết quả cũ",
        ex.lexrank_emb_indices(F, 2) == ex.lexrank_emb_indices(F, 2),
    )
    check_true("đầu ra giữ dạng tách từ", "_" in ex.lexrank_emb(F, 1))
    check_true("k lớn hơn số câu -> lấy hết", len(ex.lexrank_emb_indices(F, 9)) == 3)

    # Khu cau trung noi dung phai co y nhu ban TF-IDF: hai ban sao cua cung mot cau co
    # vector bang het nhau nen bo xep hang se vo ca hai neu khong khu.
    ex._phobert_vectors = lambda sents, *a, **k: np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    )[: len(sents)]
    G = "cảnh_sát bắt nghi_phạm . giá vàng tăng mạnh . cảnh_sát bắt nghi_phạm ."
    check_true("không chọn hai bản sao", len(ex.lexrank_emb_indices(G, 3)) == 2)
    check_true(
        "đầu ra không lặp câu",
        len(sentences(ex.lexrank_emb(G, 3))) == len(set(sentences(ex.lexrank_emb(G, 3)))),
    )

    # Bai rong phai ve som, KHONG duoc goi toi bo ma hoa: nap PhoBERT cho mot bai rong
    # la tra gia hang giay cho viec khong lam gi.
    def _no(*a, **k):
        raise AssertionError("bài rỗng không được gọi tới PhoBERT")

    ex._phobert_vectors = _no
    check_true("bài rỗng: không vỡ và không gọi PhoBERT", ex.lexrank_emb_indices("", 3) == [])
finally:
    ex._phobert_vectors = _goc
check_true("đã trả `_phobert_vectors` về bản thật", ex._phobert_vectors is _goc)

print("\n9. Tầng 4 — bộ lọc câu trước khi đưa cho mô hình sinh")
# Dem token bang so tu tren van ban THO — dung dang ma bo loc dem, va dung dang ma
# mo hinh sinh doc. Tokenizer that chi doi CON SO, khong doi logic, nen khong can keo
# `transformers` vao `.venv` sach.
from models.hybrid import K as HK  # noqa: E402
from models.hybrid import apply_filter, filter_article, filter_indices  # noqa: E402

dem = lambda s: len(s.split())  # noqa: E731

# Cau 0 va 1 cung chu de, cau 2 lac de han. Do dai: 4 / 4 / 4 tu.
H = "cảnh_sát bắt nghi_phạm . cảnh_sát khám nhà . giá vàng tăng ."
HS = sentences(H)
check_true("ngân sách rộng -> giữ hết câu", filter_indices(HS, dem, 999) == [0, 1, 2])
check_true("ngân sách chỉ đủ 2 câu (đếm dạng thô) -> bỏ câu lạc đề", filter_indices(HS, dem, 9) == [0, 1])
check_true("đầu ra luôn theo thứ tự bài", filter_indices(HS, dem, 9) == sorted(filter_indices(HS, dem, 9)))

# `lead_lexrank` phai bao dam K cau dau ngay ca khi chung khong phai cau trung tam nhat.
# Bai duoi: cau 0,1,2 moi cau mot chu de; cau 3 va 4 lap lai chu de cua cau 4 -> theo
# do trung tam thi cum cuoi thang, nhung cau 0-2 van phai co mat.
I = ("alpha alpha alpha . beta beta beta . gamma gamma gamma . "
     "delta delta delta . delta delta delta khac .")
IS = sentences(I)
thuan = filter_indices(IS, dem, 12, "lexrank")
bao_dam = filter_indices(IS, dem, 12, "lead_lexrank")
check_true(f"lead_lexrank giữ đủ {HK} câu đầu", set(range(HK)) <= set(bao_dam))
check_true("hai chiến lược cho kết quả khác nhau", thuan != bao_dam)

# Cau trung noi dung khong duoc chon hai lan — no an ngan sach ma khong them thong tin.
J = "cảnh_sát bắt nghi_phạm . giá vàng tăng mạnh . cảnh_sát bắt nghi_phạm ."
JS = sentences(J)
idx = filter_indices(JS, dem, 999)
check_true("không chọn hai bản sao của cùng một câu", len(idx) == 2)
check_true("bỏ bản sao thì giữ bản ĐẦU", 0 in idx and 2 not in idx)

# Truong hop bien.
check_true("bài rỗng -> không câu nào", filter_indices([], dem, 100) == [])
check_true("câu đầu đã dài hơn ngân sách -> vẫn trả về 1 câu",
           filter_indices(HS, dem, 1) == [0])
try:
    filter_indices(HS, dem, 10, "khong_co_that")
    check_true("chiến lược lạ -> phải báo lỗi", False)
except ValueError:
    check_true("chiến lược lạ -> phải báo lỗi", True)

# Dau ra phai o DANG THO: mo hinh sinh doc van xuoi binh thuong, khong doc gach duoi.
check_true("đầu ra bộ lọc ở dạng thô", "_" not in filter_article(H, dem, 8))

# `apply_filter` chi dung toi bai VUOT ngan sach; bai vua cua so phai nguyen van.
rows = [
    {"article": H, "article_raw": for_scoring(H)},      # ngan, khong bi dung toi
    {"article": I, "article_raw": for_scoring(I)},      # dai, se bi loc
]
goc0 = rows[0]["article_raw"]
tk = apply_filter(rows, dem, 12, "lexrank")
check_true("bài vừa cửa sổ giữ nguyên văn", rows[0]["article_raw"] == goc0)
check_true("bài vượt ngân sách bị lọc", rows[1]["article_raw"] != for_scoring(I))
check_true("thống kê đếm đúng số bài bị lọc", tk["n_loc"] == 1 and tk["n"] == 2)
check_true("thống kê ghi lại chiến lược và ngân sách",
           tk["strategy"] == "lexrank" and tk["budget"] == 12)

print("\n10. Tầng 2 — dựng ví dụ cho PhoBERT (phần KHÔNG cần torch)")
# Bo ma hoa gia: moi tu mot token. Du de kiem logic cua so va can le span; tokenizer
# that chi doi CON SO chu khong doi logic, nen khong keo `torch` vao `.venv` sach.
from models.phobert_sent import build_spans, labels_for, pick_indices  # noqa: E402

ma_hoa = lambda s: [0] * len(s.split())  # noqa: E731
M = "cảnh_sát bắt nghi_phạm . giá vàng tăng mạnh . cảnh_sát khám nhà ."
MS = sentences(M)

ids, spans = build_spans(MS, ma_hoa, max_len=99)
check_true("cửa sổ rộng -> mọi câu đều lọt", len(spans) == len(MS))
check_true("span bắt đầu từ 1, chừa chỗ token mở đầu", spans[0][0] == 1)
check_true("các span liền nhau, không chồng lấn",
           all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)))
check_true("span cuối khớp độ dài chuỗi", spans[-1][1] == 1 + len(ids))
check_true("độ dài span = số token của câu",
           all(z - a == len(ma_hoa(s)) for (a, z), s in zip(spans, MS)))

# Cua so hep: phai cat theo CAU chu khong cat giua cau.
ids2, spans2 = build_spans(MS, ma_hoa, max_len=10)
check_true("cửa sổ hẹp -> bỏ bớt câu, không cắt giữa câu", len(spans2) == 1)
check_true("câu lọt vào vẫn nguyên vẹn", spans2[0][1] - spans2[0][0] == len(ma_hoa(MS[0])))
check_true("không câu nào lọt -> không span nào", build_spans(MS, ma_hoa, max_len=3)[1] == [])
check_true("bài rỗng không làm vỡ", build_spans([], ma_hoa, max_len=99) == ([], []))

# Nhan lay tu oracle cua CA BAI, chi cat lay phan nhin thay duoc.
nhan = labels_for(MS, "cảnh_sát bắt nghi_phạm .", len(MS), k=1)
check_true("nhãn đúng số câu nhìn thấy", len(nhan) == len(MS))
check_true("oracle k=1 chọn đúng câu khớp sapo", nhan == [1.0, 0.0, 0.0])
check_true("cửa sổ hẹp thì nhãn cũng ngắn theo", len(labels_for(MS, MS[0], 1, k=1)) == 1)

# Bo chon: cung quy uoc voi `extractive._pick()`.
check_true("chọn theo điểm, trả về theo thứ tự bài", pick_indices([0.1, 0.9, 0.5], MS, 2) == [1, 2])
check_true("k lớn hơn số câu -> lấy hết", len(pick_indices([0.1, 0.9, 0.5], MS, 9)) == 3)
D2 = ["a .", "b .", "a ."]
check_true("không chọn hai bản sao", pick_indices([0.9, 0.1, 0.95], D2, 2) == [1, 2])
check_true("hoà điểm -> câu đứng trước thắng", pick_indices([0.5, 0.5, 0.5], MS, 1) == [0])

print("\n11. Tầng 2 — chọn số câu linh hoạt")
# `k3` phai trung HET `pick_indices(k=3)`: no la bang chung rang quy tac moi doc cung
# bo diem voi tang 2 da bao cao, nen so voi nhau la so dung quy tac.
from models.phobert_select import RULES, pick_rule  # noqa: E402

N6 = ["a .", "b .", "c .", "d .", "b .", "e ."]
for diem in ([0.3, 2.0, -1.0, 1.5, 2.0, 0.0], [0.0] * 6, [5, 4, 3, 2, 1, 0]):
    check_true(f"k3 trùng pick_indices với điểm {diem}",
               pick_rule(diem, N6, "k3") == pick_indices(diem, N6, 3))
check_true("k1 lấy đúng một câu", pick_rule([0.1, 3.0, 2.0], MS, "k1") == [1])
# logit 2,0 -> xac suat 0,88; 0,0 -> 0,5; -3,0 -> 0,047
check_true("pT giữ các câu trên ngưỡng", pick_rule([2.0, 0.0, 2.1], MS, "p0.6") == [0, 2])
check_true("pT không vượt 3 câu", len(pick_rule([5, 5, 5, 5, 5, 5], N6, "p0.3")) == 3)
check_true("pT luôn giữ ít nhất câu điểm cao nhất",
           pick_rule([-3.0, -2.0, -4.0], MS, "p0.9") == [1])
check_true("pT không chọn hai bản sao", pick_rule([0, 9, 0, 0, 9, 0], N6, "p0.9") == [1])
check_true("mọi quy tắc trong lưới đều chạy", all(pick_rule([1.0, 0.0, 2.0], MS, r) for r in RULES))

print("\n12. Hướng mới, giai đoạn 1 — ghép mảnh câu bị cắt nhầm ở chữ viết tắt")
from models.chon_cau import VIET_TAT_TEN, don_vi  # noqa: E402

# Dang tach tu nhu du lieu that: chi o dang nay bo tach cau moi cat nham sau "TS ."
check_true("học hàm 'TS.' được ghép với tên phía sau",
           [i for i, _ in don_vi("PV trao đổi với TS . Đào_Trọng_Tứ , giám_đốc . Ông nói .")] == [[0, 1], [2]])
for duoi in ("với TS.", "ông GS.TS.", "bà PGS.TS.", "chị ThS.", "tại St.", "cháu Tr.", "anh N.V."):
    check_true(f"'{duoi}' là mảnh cần ghép", bool(VIET_TAT_TEN.search(duoi)))
for duoi in ("danh hiệu NSƯT.", "được phong NSND.", "vào ngày CN.", "ở Hoàng Sa.", "tại Hòa An.", "vậy."):
    check_true(f"'{duoi}' là cuối câu thật, không ghép", not VIET_TAT_TEN.search(duoi))

print("\n" + ("THẤT BẠI: " + ", ".join(fails) if fails else "TẤT CẢ ĐỀU ĐẠT."))
raise SystemExit(1 if fails else 0)
