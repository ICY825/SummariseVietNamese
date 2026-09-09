"""Nạp các tập con đã đóng băng trong `data/splits/`.

Đây là phía ĐỌC; `make_splits.py` là phía GHI và chỉ chạy một lần. Từ tuần 3 trở
đi mọi tầng mô hình nạp dữ liệu qua đây, không tự lấy mẫu lại — có vậy tầng 0 và
tầng 4 mới được chấm trên đúng cùng một tập bài.

    from data.splits import load_split
    test = load_split("test")            # 2.000 bài, đã loại bài rò rỉ
    train = load_split("train_20k")      # 20.000 bài

Trả về một `datasets.Dataset` đã lọc, thêm hai cột dẫn xuất sẵn ở dạng thô
(`article_raw`, `abstract_raw`) để tầng 3 và khâu chấm điểm dùng thẳng.
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring  # noqa: E402

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


def available():
    """Tên các tập con đã đóng băng."""
    return sorted(p.stem for p in SPLITS_DIR.glob("*.json"))


def manifest(name):
    """Siêu dữ liệu của một tập con: seed, cỡ, ngày tạo, ghi chú."""
    path = SPLITS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Chạy `src/data/make_splits.py` trước. "
            f"Hiện có: {available()}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(name, add_raw=True):
    """Nạp tập con `name`. `add_raw=False` để bỏ qua bước khử tách từ cho nhanh."""
    meta = manifest(name)
    wanted = set(meta["guid"])
    ds = load_dataset(meta["dataset"])[meta["split"]]
    sub = ds.filter(lambda r: r["guid"] in wanted, desc=f"lọc {name}")

    if len(sub) != meta["size"]:
        raise RuntimeError(
            f"{name}: chờ {meta['size']} bài nhưng lọc ra {len(sub)}. "
            "Bộ dữ liệu trên Hub có thể đã đổi — phải chạy lại make_splits.py "
            "và chấm điểm lại toàn bộ."
        )
    if not add_raw:
        return sub
    return sub.map(
        lambda r: {
            "article_raw": for_scoring(r["article"]),
            "abstract_raw": for_scoring(r["abstract"]),
        },
        desc=f"khử tách từ {name}",
    )
