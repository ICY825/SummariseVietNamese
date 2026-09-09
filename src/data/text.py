r"""Ba phép biến đổi văn bản dùng chung cho TOÀN BỘ dự án.

Bộ VietNews lưu văn bản đã tách từ sẵn bằng VnCoreNLP: các âm tiết của một từ ghép
nối bằng dấu gạch dưới (`Khởi_tố`) và dấu câu đứng rời thành token riêng (`hoa ,`).
Ba tầng mô hình lại cần ba dạng khác nhau, nên mọi chỗ trong dự án phải gọi chung
các hàm ở đây thay vì tự viết lại — nếu mỗi tầng cắt câu hoặc khử tách từ theo một
kiểu, chênh lệch điểm giữa các tầng sẽ lẫn cả sai khác tiền xử lý và phép so sánh
có kiểm soát mất ý nghĩa.

Ba quy tắc, và lý do:

`sentences()` — dấu câu là token đứng riêng, nên ranh giới câu là " . " chứ không
phải mọi dấu chấm. Cắt bằng `(?<=[.!?])\s+` sẽ đứt ngay ở viết tắt như "TP." và
trả về câu cụt.

`detokenize()` — ViT5 và BARTpho-syllable được pretrain trên văn bản bình thường.
Chuỗi " ," nằm ngoài phân phối huấn luyện của chúng và bị SentencePiece băm thành
thêm subword vô ích, vừa hạ chất lượng vừa ăn mất ngân sách 1.024 token. Chỉ
`replace("_", " ")` là chưa đủ, phải dán lại dấu câu.

`for_scoring()` — đầu ra tầng 0-2 là văn bản tách từ, đầu ra tầng 3 là văn bản thô.
Chấm ROUGE trực tiếp hai thứ đó với nhau là so sánh vô nghĩa: `học_sinh` đếm một
đơn vị còn `học sinh` đếm hai, và riêng việc đổi dạng đã dịch ROUGE-2 của Lead-3
đi 4 điểm. Chuẩn chính của dự án là **dạng thô**, vì khử tách từ là chiều tất định
mà hệ thống nào cũng làm được; chiều ngược lại phải nhờ `underthesea`, khác công cụ
với VnCoreNLP đã tách tham chiếu, và sai khác đó chỉ giáng lên phía abstractive.
Mọi bản tóm tắt — của mọi tầng, kể cả tham chiếu — phải đi qua hàm này trước khi
chấm điểm.
"""

import re
import unicodedata

# Ranh gioi cau: dau cau dung rieng (co khoang trang truoc no).
_SENT = re.compile(r"(?<=\s[.!?])\s+")

# Dau cau dan vao chu ben TRAI (khong co khoang trang truoc).
_LEFT = set(",.;:!?%)]}…”’") | {"...", "..", "...."}
# Dau cau dan vao chu ben PHAI (khong co khoang trang sau).
_RIGHT = set("([{“‘")

_WS = re.compile(r"\s+")


def normalize(text):
    """Chuẩn hoá NFC và gom khoảng trắng.

    4,5% số bài chưa ở dạng NFC. Nghe nhỏ, nhưng trong những bài đó thì 61% token
    đổi khi chuẩn hoá, và sai lệch này **một chiều**: mô hình neural sinh ra dạng
    NFC còn tham chiếu thì không, nên chỉ phía abstractive chịu phạt khi so chuỗi.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def sentences(text):
    """Cắt câu trên văn bản đã tách từ, không đứt ở viết tắt kiểu `TP.`."""
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def detokenize(text):
    """Văn bản đã tách từ -> văn bản thô: bỏ gạch dưới và dán lại dấu câu.

    An toàn khi gọi trên văn bản vốn đã thô (người dùng nhập ở demo Gradio), vì
    lúc đó dấu câu đã dính liền chữ nên không token nào khớp các tập dấu câu.
    """
    out, quote_open = [], False
    for tok in text.replace("_", " ").split():
        if tok == '"':
            # Nhay kep thang khong phan biet mo/dong -> luan phien.
            out.append(('' if quote_open else ' ') + '"')
            quote_open = not quote_open
        elif tok in _LEFT or (len(tok) > 1 and set(tok) == {"."}):
            out.append(tok)
        elif tok in _RIGHT:
            out.append(" " + tok)
        elif out and (out[-1][-1] in _RIGHT or (out[-1].endswith('"') and quote_open)):
            out.append(tok)
        else:
            out.append(" " + tok)
    return _WS.sub(" ", "".join(out)).strip()


def for_scoring(text):
    """Dạng chuẩn duy nhất để chấm ROUGE / BERTScore và để in phiếu cho người chấm.

    Người chấm cũng phải đọc dạng này: nếu bản tóm tắt extractive còn nguyên gạch
    dưới còn bản của ViT5 thì không, người chấm nhận ra ngay đâu là hệ thống nào và
    quy trình blind hỏng hoàn toàn.
    """
    return normalize(detokenize(text))


def tokens(text):
    """Token theo cách tách từ sẵn có: `Khởi_tố` là MỘT đơn vị (dạng PhoBERT ăn)."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def syllables(text):
    """Âm tiết thật: tách cả dấu gạch dưới (dạng ViT5 / BARTpho-syllable ăn)."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
