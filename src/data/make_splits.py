"""Đóng băng các tập con cố định vào `data/splits/`.

Phải chạy script này TRƯỚC khi chạy bất cứ baseline nào. Đổi tập con về sau đồng
nghĩa với chấm điểm lại toàn bộ mọi tầng, nên danh sách ID cần cố định từ đầu.

Vì sao dùng tập con thay vì toàn bộ dữ liệu:

  Test 2.000 bài thay vì 22.498 — đo trên tập test thật, độ lệch chuẩn ROUGE-1
  giữa các bài là 9,8 và độ lệch chuẩn của hiệu khi so cặp đôi là 12,5. Với n=2.000
  nửa khoảng tin cậy 95% là ±0,55 điểm khi so hai hệ thống. Khoảng cách giữa các
  tầng trong đề tài này là 2-5 điểm nên mức đó dư dùng; chạy toàn bộ test chỉ siết
  xuống ±0,16 nhưng tốn gấp 11 lần, chưa kể mỗi hệ thống neural phải sinh lại từng
  ấy bản tóm tắt.

  Train tối đa 20.000 bài — một epoch trên 99.134 bài với ViT5-base ở đầu vào 1.024
  token vượt quá giới hạn một phiên Colab free. ViT5 đã được pretrain nên fine-tune
  tóm tắt bão hoà sớm; phần GPU tiết kiệm được đổ vào đường cong học và khảo sát
  tham số sinh sẽ cho ra kết quả có nội dung hơn là thêm 1-2 điểm ROUGE.

  Các tập train LỒNG NHAU (2k ⊂ 5k ⊂ 10k ⊂ 20k) — nếu lấy bốn mẫu ngẫu nhiên độc
  lập, chênh lệch trên đường cong học sẽ lẫn cả dao động do lấy mẫu và không đọc
  được gì.

  Tập dò tham số tách rời tập validation — tuần 5 khảo sát tham số sinh văn bản.
  Dò trên chính tập dùng để theo dõi huấn luyện sẽ làm hỏng cả hai vai trò.

Chạy:  .venv/Scripts/python.exe src/data/make_splits.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import random
import unicodedata
from datetime import date
from pathlib import Path

from datasets import load_dataset

DATASET = "nam194/vietnews"
SEED = 13  # trung voi seed cua inspect_vietnews.py
OUT = Path(__file__).resolve().parents[2] / "data" / "splits"

TRAIN_SIZES = [2_000, 5_000, 10_000, 20_000]  # long nhau, cho duong cong hoc
VAL_SIZE = 1_000      # theo doi qua tung epoch
TUNE_SIZE = 500       # do tham so sinh van ban (tuan 5), roi khoi VAL
TEST_SIZE = 2_000     # cham diem cuoi cung, dung mot lan


def fingerprint(text):
    return hashlib.md5(unicodedata.normalize("NFC", text).strip().encode()).hexdigest()


def write(name, split, guids, note):
    """Ghi một tập con. Lưu cả `split` vì `guid` chỉ duy nhất TRONG một split."""
    path = OUT / f"{name}.json"
    payload = {
        "name": name,
        "dataset": DATASET,
        "split": split,
        "seed": SEED,
        "size": len(guids),
        "created": date.today().isoformat(),
        "note": note,
        "guid": sorted(guids),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {name:16s} {len(guids):>6,} bài  <- {path.relative_to(OUT.parents[1])}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET)
    rng = random.Random(SEED)

    print("Tìm bài rò rỉ train ∩ test để loại khỏi tập chấm điểm...")
    train_fp = {fingerprint(a) for a in ds["train"]["article"]}
    test_guid = ds["test"]["guid"]
    leaked = [
        g
        for g, a in zip(test_guid, ds["test"]["article"])
        if fingerprint(a) in train_fp
    ]
    print(f"  Loại {len(leaked)} bài rò rỉ: guid {leaked}")

    print("\nĐóng băng các tập con:")

    # --- TRAIN: long nhau. Xao mot lan roi cat tien to. ---
    train_pool = list(ds["train"]["guid"])
    rng.shuffle(train_pool)
    for n in TRAIN_SIZES:
        write(
            f"train_{n // 1000}k",
            "train",
            train_pool[:n],
            f"Tập con lồng nhau cho đường cong học; train_{n // 1000}k ⊂ các tập lớn hơn.",
        )

    # --- VALIDATION: 1000 theo doi + 500 do tham so, ROI NHAU. ---
    val_pool = list(ds["validation"]["guid"])
    rng.shuffle(val_pool)
    write("val", "validation", val_pool[:VAL_SIZE], "Theo dõi loss/ROUGE qua từng epoch.")
    write(
        "tune",
        "validation",
        val_pool[VAL_SIZE : VAL_SIZE + TUNE_SIZE],
        "Dò tham số sinh văn bản ở tuần 5. Rời hẳn khỏi `val` và `test`.",
    )

    # --- TEST: loai bai ro ri TRUOC khi lay mau. ---
    test_pool = [g for g in test_guid if g not in set(leaked)]
    rng.shuffle(test_pool)
    write(
        "test",
        "test",
        test_pool[:TEST_SIZE],
        f"Chấm điểm cuối cùng, dùng một lần. Đã loại {len(leaked)} bài rò rỉ "
        f"train ∩ test. Nửa khoảng tin cậy 95% khi so cặp đôi: ±0,55 điểm ROUGE.",
    )

    # --- Kiem chung lai nhung gi vua ghi ---
    print("\nKiểm chứng:")
    loaded = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*.json")}

    sets = {k: set(v["guid"]) for k, v in loaded.items()}
    ok = True
    for a, b in zip(TRAIN_SIZES, TRAIN_SIZES[1:]):
        na, nb = f"train_{a // 1000}k", f"train_{b // 1000}k"
        nested = sets[na] <= sets[nb]
        ok &= nested
        print(f"  {na} ⊂ {nb}: {'OK' if nested else 'HỎNG'}")
    for a, b in (("val", "tune"),):
        disjoint = not (sets[a] & sets[b])
        ok &= disjoint
        print(f"  {a} ∩ {b} rỗng: {'OK' if disjoint else 'HỎNG'}")
    no_leak = not (sets["test"] & set(leaked))
    ok &= no_leak
    print(f"  test không chứa bài rò rỉ: {'OK' if no_leak else 'HỎNG'}")
    for k, v in sorted(loaded.items()):
        ok &= len(v["guid"]) == len(set(v["guid"]))
    print(f"  không tập nào trùng lặp nội bộ: {'OK' if ok else 'HỎNG'}")

    print("\nXONG." if ok else "\nCÓ KIỂM CHỨNG THẤT BẠI — xem lại trước khi dùng.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
