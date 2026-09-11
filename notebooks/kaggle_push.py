"""Đẩy notebook huấn luyện lên Kaggle với một cấu hình, không sửa bản trong repo.

    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k --dry
    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k
    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_5k --model vinai/bartpho-syllable

CẢNH BÁO: push là CHẠY NGAY — Kaggle tạo version mới, chạy toàn bộ notebook và trừ
quota GPU. `--dry` chỉ dựng bản sẽ đẩy rồi in phần cấu hình ra để xem trước.

Vì sao không sửa thẳng `TRAIN_SPLIT` trong notebook rồi push: bản trong repo là bản
Kaggle clone về và là bản người đọc repo nhìn thấy. Sửa nó cho mỗi lần chạy thì mặc
định của repo trôi theo lần chạy gần nhất, và rất dễ lỡ tay commit. Script chép sang
một thư mục tạm, đổi đúng hai dòng cấu hình ở đó, rồi đẩy từ thư mục tạm.

Cần: Kaggle CLI (`pip install kaggle` vào một môi trường riêng, không vào `.venv` của
dự án) và token ở `~/.kaggle/access_token`.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "kaggle_train_vit5.ipynb"
METADATA = HERE / "kernel-metadata.json"
SPLITS = HERE.parent / "data" / "splits"


def set_config(text, split, model):
    """Đổi giá trị `TRAIN_SPLIT` và `MODEL` trong ô cấu hình, mỗi dòng đúng MỘT lần.

    Làm trên chuỗi JSON thô chứ không nạp lại rồi ghi ra, để bản đẩy lên khác bản
    trong repo đúng hai dòng — `diff` giữa hai bản là bằng chứng không có gì khác lọt vào.
    """
    for key, value in (("TRAIN_SPLIT", split), ("MODEL", model)):
        pat = re.compile(rf'("{key} = \\")[^"\\]*(\\")')
        text, n = pat.subn(rf"\g<1>{value}\g<2>", text)
        if n != 1:
            raise SystemExit(f"Tìm thấy {n} dòng `{key} = ...` trong notebook, phải đúng 1. Ô cấu hình đã đổi?")
    return text


def main():
    ap = argparse.ArgumentParser(description="Đẩy notebook huấn luyện lên Kaggle.")
    ap.add_argument("split", help="tập train, ví dụ train_20k")
    ap.add_argument("--model", default="VietAI/vit5-base")
    ap.add_argument("--dry", action="store_true", help="chỉ dựng và in cấu hình, không đẩy")
    args = ap.parse_args()

    have = sorted(p.stem for p in SPLITS.glob("train_*.json"))
    if args.split not in have:
        raise SystemExit(f"Không có tập train `{args.split}`. Hiện có: {have}")

    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(METADATA, tmp / METADATA.name)
        (tmp / NOTEBOOK.name).write_text(
            set_config(NOTEBOOK.read_text(encoding="utf-8"), args.split, args.model),
            encoding="utf-8", newline="\n",
        )
        # Doc lai ban SE DAY (khong phai ban trong repo) va in dung o cau hinh — o
        # code dau tien — de thu nguoi dung duyet la thu Kaggle se chay.
        cells = json.loads((tmp / NOTEBOOK.name).read_text(encoding="utf-8"))["cells"]
        config = next(c for c in cells if c["cell_type"] == "code")
        print("Cấu hình sẽ đẩy:")
        for line in config["source"]:
            if re.match(r"(MODEL|TRAIN_SPLIT|EPOCHS|LR|DRY_RUN|RESUME) = ", line):
                print("  " + line.rstrip())
        if args.dry:
            print("--dry: không đẩy.")
            return 0
        out = subprocess.run([kaggle, "kernels", "push", "-p", str(tmp), "--accelerator", "NvidiaTeslaT4"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        msg = (out.stdout + out.stderr).strip()
        print(msg)
        # CLI bao loi (vd "Maximum batch GPU session count of 2 reached") ma van tra
        # ma thoat 0, nen phai doc noi dung chu khong chi nhin ma thoat.
        if out.returncode != 0 or "error" in msg.lower() or "successfully pushed" not in msg:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
