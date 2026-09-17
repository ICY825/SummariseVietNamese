"""Đường suy luận cho demo: một bài văn thô người dùng dán vào -> bản tóm tắt bốn tầng.

Chạy bằng `~/.venvs/demo`. Checkpoint nằm ngoài repo, ở `~/.cache/dl-summarisevn/`.

Hai lỗi âm thầm mà module này CHẶN thay vì để chạy tiếp, vì cả hai đều cho ra kết quả
trông hợp lý mà sai hoàn toàn:

1. **Thiếu `head.pt`** — `from_pretrained` chỉ nạp phần mã hoá; lớp cho điểm của tầng 2
   khi ấy là trọng số khởi tạo ngẫu nhiên, và bản tóm tắt là ba câu lấy bừa.
2. **Đầu vào chưa tách từ** — `sentences()` cắt câu theo ranh giới " . " của bộ VietNews.
   Văn bản thô có dấu chấm dính liền chữ nên nó thấy đúng MỘT câu, và mọi tầng extractive
   trả về nguyên bài thay vì k câu.
"""

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.text import for_scoring, normalize, sentences  # noqa: E402
from models import extractive as E  # noqa: E402
from models.hybrid import filter_article  # noqa: E402
from models.phobert_select import pick_rule  # noqa: E402

CKPT = Path.home() / ".cache" / "dl-summarisevn"
MAX_INPUT = 1024          # da chot o tuan 2, dung cho tang 3 va ngan sach tang 4
MAX_TARGET = 80
GEN = {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True}
RULE = "k2"               # quy tac tang 2 thang tren `tune`
K = 3                     # so cau cua tang 0-1, trung bao cao


def _co_trong_so(duong):
    return any((duong / t).exists() and (duong / t).stat().st_size > 0
               for t in ("model.safetensors", "pytorch_model.bin"))


def tim_checkpoint(ten):
    """Thư mục checkpoint dùng được của tầng `ten`, dò khắp `CKPT`.

    Dò cả cây thay vì chỉ `CKPT/<ten>` vì cùng một tầng có thể có nhiều bản tải về nằm ở
    các bố cục khác nhau (`<ten>/runs/.../checkpoint-N`, `xuat/ckpt/<ten>`...). Quan
    trọng hơn: bản tải bằng `kernels output` có trọng số **0 byte**, nên phải **ưu tiên
    thư mục có trọng số thật** — chọn nhầm bản hỏng sẽ báo lỗi y hệt như chưa tải gì,
    và rất dễ kết luận nhầm là bản tải mới cũng thất bại.
    """
    if not CKPT.exists():
        raise FileNotFoundError(f"Chưa tải checkpoint nào về {CKPT}.")
    ung_vien = [p.parent for p in CKPT.rglob("config.json") if ten in p.parent.as_posix()]
    if not ung_vien:
        raise FileNotFoundError(f"Không thấy checkpoint nào của {ten} dưới {CKPT}.")
    tot = [d for d in ung_vien if _co_trong_so(d)]
    # Khong co ban nao dung duoc thi van tra ve mot ban de `kiem_trong_so` bao loi ro rang.
    return sorted(tot or ung_vien, key=lambda p: len(p.parts))[0]


def kiem_trong_so(duong):
    """Báo lỗi nếu thư mục checkpoint không có file trọng số dùng được.

    Kaggle `kernels output` trả về file trọng số **0 byte** cho checkpoint lớn: mọi file
    nhỏ (config, tokenizer, `head.pt`) đều thật, chỉ riêng `model.safetensors` rỗng. Chỉ
    kiểm sự tồn tại là không đủ — phải kiểm kích thước.
    """
    ten = ("model.safetensors", "pytorch_model.bin")
    co = [duong / t for t in ten if (duong / t).exists()]
    if not co:
        raise FileNotFoundError(f"{duong} không có file trọng số nào trong {ten}.")
    if all(p.stat().st_size == 0 for p in co):
        raise RuntimeError(
            f"Trọng số ở {duong} là file RỖNG (0 byte) — Kaggle không trả về file lớn "
            "qua `kernels output`. Phải xuất lại checkpoint từ kernel (xem NHAT_KY)."
        )


def tach_tu(tho):
    """Văn bản thô -> dạng tách từ của bộ VietNews (`Khởi_tố`, dấu câu đứng rời).

    Đây là bước bắt buộc: thiếu nó thì `sentences()` chỉ thấy một câu.
    """
    from underthesea import word_tokenize

    return word_tokenize(normalize(tho), format="text")


def chuan_bi(tho):
    """(dạng tách từ, danh sách câu) — báo lỗi nếu khâu tách từ không cho ra câu nào.

    Phép kiểm ở đây là chốt chặn cho lỗi âm thầm số 2: nếu văn bản có nhiều dấu kết câu
    mà chỉ cắt ra một câu thì khâu tách từ đã hỏng, và mọi tầng phía sau sẽ trả về
    nguyên bài mà không báo gì.
    """
    seg = tach_tu(tho)
    cau = sentences(seg)
    if len(cau) == 1 and sum(tho.count(d) for d in ".!?") >= 2:
        raise RuntimeError(
            "Tách từ hỏng: văn bản có nhiều câu nhưng chỉ cắt được một. "
            "Kiểm tra `underthesea` đã cài đúng chưa."
        )
    return seg, cau


# --------------------------------------------------------------------------
# Tang 0-1: khong can mo hinh, chay tren CPU trong vai mili giay
# --------------------------------------------------------------------------

def tang01(seg, he="lead", k=K):
    ham = {"lead": E.lead, "textrank": E.textrank, "lexrank": E.lexrank}[he]
    return for_scoring(ham(seg, k=k))


# --------------------------------------------------------------------------
# Tang 2: PhoBERT cho diem tung cau, lay 2 cau cao nhat
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tang2():
    import torch
    from transformers import AutoTokenizer
    from models.phobert_sent import _build_model

    duong = tim_checkpoint("tang2")
    if not (duong / "head.pt").exists():
        raise FileNotFoundError(
            f"{duong} không có head.pt. Nạp thiếu nó thì lớp cho điểm là trọng số "
            "ngẫu nhiên và tầng 2 chọn câu bừa mà không báo lỗi."
        )
    kiem_trong_so(duong)
    tok = AutoTokenizer.from_pretrained(str(duong))
    model = _build_model(str(duong), torch.device("cpu"))
    return tok, model


def tang2(seg):
    from models.phobert_sent import encode_rows, score_examples
    from models.extractive import join

    tok, model = _tang2()
    vd = encode_rows([{"article": seg}], tok, with_labels=False)
    if vd[0] is None:
        raise RuntimeError("Không câu nào lọt cửa sổ 256 token của PhoBERT.")
    diem = score_examples(model, vd, tok.pad_token_id, "cpu")[0]
    cau = vd[0]["sents"][: vd[0]["n_visible"]]
    return for_scoring(join(cau, pick_rule(diem, cau, RULE)))


# --------------------------------------------------------------------------
# Tang 3 va 4: BARTpho train_20k
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _bartpho():
    from transformers import AutoModelForSeq2SeqLM
    from models.measure_tokens import load_tokenizer

    duong = tim_checkpoint("bartpho")
    kiem_trong_so(duong)
    tok = load_tokenizer(str(duong))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(duong))
    model.eval()
    return tok, model


def _sinh(tho):
    from models.vit5 import generate

    tok, model = _bartpho()
    return generate(model, tok, [{"article_raw": tho}], MAX_INPUT, MAX_TARGET,
                    batch=1, gen=GEN)[0]


def tang3(tho):
    return _sinh(normalize(tho))


def tang4(seg, chien_luoc="lead_lexrank"):
    """Lọc câu cho vừa ngân sách token rồi mới để BARTpho viết lại."""
    tok, _ = _bartpho()
    # Dem tren tung cau o dang THO, dung thu BARTpho thuc su doc.
    dem = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    return _sinh(filter_article(seg, dem, MAX_INPUT, chien_luoc))


# --------------------------------------------------------------------------
# He thong cuoi (huong moi): chon cau co chu dich — cau hinh doc tu file do tren tune
# --------------------------------------------------------------------------

def cau_hinh_cuoi():
    """Cấu hình thắng trên `tune` — đọc từ file dò, không gõ tay, để demo không lệch báo cáo."""
    import json

    th = json.loads((ROOT / "results" / "tables" / "chon_cau_tune_do.json").read_text(encoding="utf-8"))["thang"]
    return [th[k] for k in ("ngan_sach", "w_vt", "w_lex", "lam", "noi_tien_de", "loc_rac")]


def diem_phobert(seg):
    """Logit PhoBERT từng câu nhìn thấy được — đúng đường tính điểm đã lưu cho `val`/`test`."""
    from models.phobert_sent import encode_rows, score_examples

    tok, model = _tang2()
    vd = encode_rows([{"article": seg}], tok, with_labels=False)
    return None if vd[0] is None else score_examples(model, vd, tok.pad_token_id, "cpu")[0]


def he_thong_cuoi(seg):
    from models.chon_cau import chon, chuan_bi_bai, van_ban

    b = chuan_bi_bai(seg, diem_phobert(seg))
    return van_ban(b, chon(b, *cau_hinh_cuoi()))


def to_chi_tiet_la(tom_tat, seg):
    """(HTML tô vàng chi tiết lạ, số chi tiết lạ) — chi tiết lạ theo đúng bộ đo `chi_tiet_la`.

    Bộ đo trả về chuỗi âm tiết đã hạ chữ thường; ở đây tìm lại chúng trong văn bản gốc của bản
    tóm tắt, cho phép dấu câu/khoảng trắng xen giữa các âm tiết ("5h30", "TP. HCM").
    """
    import html
    import re

    from eval.chinh_xac import chi_tiet_la

    la = chi_tiet_la(tom_tat, seg)
    ra = html.escape(tom_tat)
    for cum in sorted(la, key=lambda c: -sum(map(len, c))):
        mau = r"(?<![^\W_])" + r"[\W_]*".join(re.escape(html.escape(s)) for s in cum) + r"(?![^\W_])"
        ra = re.sub(mau, lambda m: f"<mark>{m.group(0)}</mark>", ra, flags=re.IGNORECASE)
    return ra, len(la)


def so_sanh_cuoi(tho, seg=None):
    """{hệ thống cuối, BARTpho}: bản tóm tắt, HTML tô chi tiết lạ, số chi tiết lạ, số âm tiết.

    `seg` là dạng tách từ có sẵn (bài mẫu lấy từ bộ dữ liệu); không có thì tách bằng underthesea.
    """
    from data.text import syllables

    if seg is None:
        seg, _ = chuan_bi(tho)
    ra = {}
    for ten, ham in (("cuoi", lambda: he_thong_cuoi(seg)), ("bartpho", lambda: tang3(tho))):
        t = ham()
        html_, n = to_chi_tiet_la(t, seg)
        ra[ten] = {"van_ban": t, "html": html_, "so_la": n, "am_tiet": len(syllables(t))}
    return ra


# Ten va thu tu cac tang, dung chung cho ca giao dien. Mot nguon su that duy nhat:
# chep tay o hai noi thi lech mot ky tu la giao dien hien o trong ma khong bao gi.
TEN_TANG = (
    "Lead-3",
    "LexRank",
    "Tầng 2 — PhoBERT k2",
    "Tầng 3 — BARTpho",
    "Tầng 4 — lọc rồi viết lại",
)


def tom_tat_tat_ca(tho):
    """Chạy cả bốn tầng, trả về {tên tầng: bản tóm tắt hoặc thông báo lỗi}."""
    seg, _ = chuan_bi(tho)
    ham = {
        "Lead-3": lambda: tang01(seg, "lead"),
        "LexRank": lambda: tang01(seg, "lexrank"),
        "Tầng 2 — PhoBERT k2": lambda: tang2(seg),
        "Tầng 3 — BARTpho": lambda: tang3(tho),
        "Tầng 4 — lọc rồi viết lại": lambda: tang4(seg),
    }
    ra = {}
    for ten in TEN_TANG:
        try:
            ra[ten] = ham[ten]()
        except Exception as e:
            ra[ten] = f"[lỗi] {type(e).__name__}: {e}"
    return ra
