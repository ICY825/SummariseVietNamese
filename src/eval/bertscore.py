"""BERTScore cho tiếng Việt. Bọc mỏng quanh `bert-score`, nạp lười.

**Vì sao không dùng PhoBERT làm bộ mã hoá.** PhoBERT chỉ nhận văn bản đã tách từ,
trong khi dạng chấm điểm chuẩn của dự án là văn bản thô. Muốn dùng PhoBERT thì phải
tách từ lại đầu ra của ViT5 bằng `underthesea` — khác công cụ với VnCoreNLP đã tách
tham chiếu, và sai khác công cụ đó chỉ giáng lên phía abstractive, đúng thứ thiên lệch
mà cả dự án đang tránh. Nên mặc định dùng một mô hình đọc thẳng âm tiết.

Đã chạy thật trên `val` — 1.000 bài, 8 hệ thống, 9 phút CPU; kết quả và cách đọc nằm ở
mục "BERTScore" của README. `torch` và `bert-score` **không** nằm trong `.venv` của dự
án (xem `requirements.txt`), nên gọi qua `src/eval/run_bertscore.py` bằng môi trường
riêng `~/.venvs/torch`, trên chính các file dự đoán đã lưu chứ không sinh lại gì.
"""

DEFAULT_MODEL = "xlm-roberta-base"  # doc thang am tiet, khong can tach tu


def bert_score(predictions, references, model=DEFAULT_MODEL, batch_size=32, device=None):
    """Trả về F1 BERTScore của **từng bài** (nhân 100), để bootstrap dùng được.

    Cần `pip install bert-score torch`.
    """
    try:
        import bert_score as _bs
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "Thiếu `bert-score`. Cài bằng: pip install bert-score torch\n"
            "Trên Colab nhớ chọn runtime có GPU, chạy trên CPU rất chậm."
        ) from e

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.text import for_scoring

    # Cung mot phieu loc dang van ban nhu ROUGE - khong duoc lech.
    preds = [for_scoring(p) for p in predictions]
    refs = [for_scoring(r) for r in references]
    _, _, f1 = _bs.score(
        preds, refs, model_type=model, batch_size=batch_size, device=device,
        lang="vi", rescale_with_baseline=False, verbose=False,
    )
    return [float(x) * 100 for x in f1]
