"""Đẩy một notebook của dự án lên Kaggle, không sửa bản trong repo.

    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k --dry
    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_20k
    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py train_5k --model vinai/bartpho-syllable
    ~/.venvs/kaggle/Scripts/python.exe notebooks/kaggle_push.py --dir notebooks/sweep

CẢNH BÁO: push là CHẠY NGAY — Kaggle tạo version mới, chạy toàn bộ notebook và trừ
quota GPU. `--dry` chỉ dựng bản sẽ đẩy rồi in phần cấu hình ra để xem trước.

Vì sao không sửa thẳng `TRAIN_SPLIT` trong notebook rồi push: bản trong repo là bản
Kaggle clone về và là bản người đọc repo nhìn thấy. Sửa nó cho mỗi lần chạy thì mặc
định của repo trôi theo lần chạy gần nhất, và rất dễ lỡ tay commit. Script chép sang
một thư mục tạm, đổi đúng hai dòng cấu hình ở đó, rồi đẩy từ thư mục tạm.

`--dir` trỏ tới thư mục chứa `kernel-metadata.json`; notebook lấy theo khoá `code_file`
trong đó. Notebook huấn luyện có dòng `TRAIN_SPLIT`/`MODEL` nên bắt buộc truyền tập
train; notebook dò tham số (`notebooks/sweep`) không có, nên đẩy nguyên bản.

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
SPLITS = HERE.parent / "data" / "splits"
KEYS = ("MODEL", "TRAIN_SPLIT", "EPOCHS", "LR", "DRY_RUN", "RESUME",
        "EVAL_SPLIT", "NAME", "CKPT_GLOB")


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
    ap = argparse.ArgumentParser(description="Đẩy notebook lên Kaggle.")
    ap.add_argument("split", nargs="?", help="tập train, ví dụ train_20k (notebook huấn luyện)")
    ap.add_argument("--dir", default=str(HERE), help="thư mục chứa kernel-metadata.json")
    ap.add_argument("--model", default="VietAI/vit5-base")
    ap.add_argument("--dry", action="store_true", help="chỉ dựng và in cấu hình, không đẩy")
    args = ap.parse_args()

    folder = Path(args.dir).resolve()
    meta_path = folder / "kernel-metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"Không có {meta_path}.")
    notebook = folder / json.loads(meta_path.read_text(encoding="utf-8"))["code_file"]
    if not notebook.exists():
        raise SystemExit(f"Metadata trỏ tới {notebook.name} nhưng không có file đó trong {folder}.")

    src = notebook.read_text(encoding="utf-8")
    # Notebook nao co dong TRAIN_SPLIT thi phai duoc noi chay tap nao; notebook do
    # tham so thi khong — bat nham hai truong hop nay se day di mot cau hinh khac
    # voi cai nguoi dung nghi minh day.
    can_split = '"TRAIN_SPLIT = ' in src
    if can_split and not args.split:
        raise SystemExit(f"{notebook.name} cần tập train. Ví dụ: kaggle_push.py train_20k")
    if args.split and not can_split:
        raise SystemExit(f"{notebook.name} không có dòng TRAIN_SPLIT — bỏ tham số `{args.split}`.")
    if args.split:
        have = sorted(p.stem for p in SPLITS.glob("train_*.json"))
        if args.split not in have:
            raise SystemExit(f"Không có tập train `{args.split}`. Hiện có: {have}")

    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(meta_path, tmp / meta_path.name)
        (tmp / notebook.name).write_text(
            set_config(src, args.split, args.model) if args.split else src,
            encoding="utf-8", newline="\n",
        )
        # Doc lai ban SE DAY (khong phai ban trong repo) va in dung o cau hinh — o
        # code dau tien — de thu nguoi dung duyet la thu Kaggle se chay.
        cells = json.loads((tmp / notebook.name).read_text(encoding="utf-8"))["cells"]
        config = next(c for c in cells if c["cell_type"] == "code")
        print(f"Notebook: {notebook.name} -> {json.loads(meta_path.read_text(encoding='utf-8'))['id']}")
        print("Cấu hình sẽ đẩy:")
        for line in config["source"]:
            if re.match(rf"({'|'.join(KEYS)}) = ", line):
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
