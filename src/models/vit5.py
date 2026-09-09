"""Tầng 3: fine-tune mô hình sinh văn bản (ViT5, đối chứng BARTpho) cho tóm tắt.

    python src/models/vit5.py --train-split train_2k --epochs 1     # chay thu
    python src/models/vit5.py --train-split train_10k --epochs 3
    python src/models/vit5.py --model vinai/bartpho-syllable --train-split train_10k

Phải chạy trên GPU. Trên Colab free (T4) nhớ cho `--out` trỏ vào Google Drive, vì
phiên bị ngắt bất chợt và checkpoint nằm trong `/content` sẽ mất sạch.

Sáu quyết định đã chốt, và lý do:

**Đầu vào là `article_raw`, đầu ra là `abstract_raw`.** ViT5 và BARTpho được pretrain
trên văn bản bình thường; đưa `Khởi_tố` vào là đưa chuỗi ngoài phân phối huấn luyện,
vừa hạ chất lượng vừa ăn mất ngân sách token. `load_split(add_raw=True)` dựng sẵn hai
cột này bằng `for_scoring()`, đúng dạng mà khâu chấm điểm dùng.

**`max_input_length = 1024`, đo bằng chính tokenizer chứ không đoán.** Đo trên 2.000
bài của `train_20k` (`seed=13`): hệ số nở token/âm tiết là 1,18 với ViT5 và 1,20 với
BARTpho — tức bảng theo âm tiết trong README là cận dưới, tỷ lệ cắt thật ở ngưỡng
1.024 là 10,2% (ViT5) và 11,2% (BARTpho), không phải 4,2%. Hai mô hình gần như trùng
nhau nên dùng chung một ngưỡng, và nhờ vậy so ViT5 với BARTpho là so mô hình chứ
không phải so ngân sách đầu vào.

Không chọn 1.536 dù nó hạ tỷ lệ cắt xuống 0,4%: dài gấp 1,5 lần thì chi phí attention
gấp khoảng 2,25 lần, đủ để `train_20k` không chạy xong trong một phiên Colab free.
Quan trọng hơn, **việc cắt bài chính là đối tượng nghiên cứu** của câu hỏi số 2 và là
lý do tồn tại của tầng 4 — xoá nó đi bằng cửa sổ dài hơn là xoá luôn câu hỏi.

**`max_target_length = 80`.** Sapo có p99 là 74 token (ViT5) và 78 (BARTpho); 80 che
được 99% và là bội của 8, thuận cho fp16. Bài dài nhất 144 token là ngoại lệ, không
đáng để nới ngưỡng cho cả tập.

**Không bao giờ chạm vào `test`.** Mặc định theo dõi trên `val`. `test` để dành chấm
một lần ở cuối, và tập `tune` mới là chỗ dò tham số sinh văn bản ở tuần 5.

**Tham số sinh cố định ở tuần 4.** `num_beams=4`, `no_repeat_ngram_size=3`. Khảo sát
là việc của tuần 5 và phải làm trên `tune`; dò trên `val` sẽ làm hỏng vai trò theo dõi
của chính tập đó.

**Chấm điểm qua `eval.report`.** Mô hình sinh ra văn bản thô còn baseline ra văn bản
tách từ; `for_scoring()` đã lo việc quy về một dạng. Tự tính ROUGE ở đây là phá phép
so sánh có kiểm soát mà tầng 0-1 đã dựng.

CẢNH BÁO RÒ RỈ: trên Hub có checkpoint ViT5 đã fine-tune sẵn cho tóm tắt VietNews.
Điểm khởi đầu phải là bản pretrain thuần (`VietAI/vit5-base`). Dùng bản đã fine-tune
là để mô hình nhìn trước tập test, và mọi con số sau đó vô nghĩa.
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from eval.report import compare, evaluate, save, table  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"

MAX_INPUT = 1024   # do that: cat 10,2% so bai (ViT5) / 11,2% (BARTpho)
MAX_TARGET = 80    # p99 cua sapo la 74-78 token
SEED = 13


def _pick_kwarg(cls, options, value):
    """Chọn tên tham số mà phiên bản thư viện hiện tại thực sự nhận.

    `transformers` đã đổi tên vài tham số giữa các bản 4.x và 5.x
    (`evaluation_strategy` -> `eval_strategy`, `tokenizer` -> `processing_class`).
    Dò theo chữ ký thay vì ghim một tên, để notebook không chết chỉ vì Colab vừa nâng
    cấp thư viện. Nếu không tên nào khớp thì trả về dict rỗng và báo ra màn hình, chứ
    KHÔNG im lặng bỏ qua — một tham số bị nuốt mất có thể đổi hẳn kết quả huấn luyện.
    """
    names = set(inspect.signature(cls.__init__).parameters)
    for opt in options:
        if opt in names:
            return {opt: value}
    print(f"  CẢNH BÁO: {cls.__name__} không nhận tham số nào trong {options}.")
    return {}


def tokenize(rows, tok, max_input, max_target, prefix=""):
    """Văn bản thô -> tensor. Cắt ở đây, đúng ngưỡng đã chốt."""
    x = tok(
        [prefix + r["article_raw"] for r in rows],
        max_length=max_input,
        truncation=True,
    )
    y = tok(
        [r["abstract_raw"] for r in rows],
        max_length=max_target,
        truncation=True,
    )
    return [
        {"input_ids": a, "attention_mask": m, "labels": b}
        for a, m, b in zip(x["input_ids"], x["attention_mask"], y["input_ids"])
    ]


def generate(model, tok, rows, max_input, max_target, batch=16, beams=4, prefix=""):
    """Sinh bản tóm tắt cho cả tập, giữ nguyên thứ tự bài."""
    import torch

    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        chunk = [prefix + r["article_raw"] for r in rows[i : i + batch]]
        enc = tok(
            chunk,
            max_length=max_input,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            ids = model.generate(
                **enc,
                max_length=max_target,
                num_beams=beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        out.extend(tok.batch_decode(ids, skip_special_tokens=True))
        done = min(i + batch, len(rows))
        print(f"\r  sinh {done}/{len(rows)}", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description="Fine-tune tầng 3.")
    ap.add_argument("--model", default="VietAI/vit5-base")
    ap.add_argument("--train-split", default="train_2k")
    ap.add_argument("--eval-split", default="val", help="KHÔNG dùng test ở tuần 4-5")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2, help="mỗi thiết bị")
    ap.add_argument("--grad-accum", type=int, default=8, help="batch hiệu dụng = batch * cái này")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-input", type=int, default=MAX_INPUT)
    ap.add_argument("--max-target", type=int, default=MAX_TARGET)
    ap.add_argument("--prefix", default="", help='tiền tố kiểu T5, ví dụ "vietnews: "')
    ap.add_argument("--eval-limit", type=int, default=0, help="chỉ sinh N bài đầu khi chạy thử")
    ap.add_argument("--out", default="runs", help="thư mục lưu; trên Colab hãy trỏ vào Drive")
    ap.add_argument("--no-train", action="store_true", help="chỉ nạp và sinh, để thử đường ống")
    args = ap.parse_args()

    if args.eval_split == "test":
        raise SystemExit(
            "Từ chối chấm trên `test`. Tập này dùng MỘT lần ở cuối dự án; "
            "tuần 4-5 theo dõi trên `val`, dò tham số trên `tune`."
        )

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    from models.measure_tokens import load_tokenizer

    set_seed(SEED)
    if not torch.cuda.is_available():
        print("CẢNH BÁO: không thấy GPU. Huấn luyện trên CPU sẽ mất nhiều giờ.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Nạp {args.model} ...")
    tok = load_tokenizer(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print(f"Nạp {args.train_split} và {args.eval_split} ...")
    train_rows = list(load_split(args.train_split, add_raw=True))
    eval_rows = list(load_split(args.eval_split, add_raw=True))
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
        print(f"  CHẠY THỬ: chỉ sinh {len(eval_rows)} bài — số liệu KHÔNG dùng báo cáo.")
    print(f"  train {len(train_rows)} bài, eval {len(eval_rows)} bài.\n")

    out_dir = Path(args.out) / f"{args.model.replace('/', '_')}_{args.train_split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_train:
        train_ds = tokenize(train_rows, tok, args.max_input, args.max_target, args.prefix)
        eval_ds = tokenize(eval_rows, tok, args.max_input, args.max_target, args.prefix)

        targs = dict(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,   # doi toc do lay VRAM, can o 1.024 token
            fp16=torch.cuda.is_available(),  # T4 khong ho tro bf16
            logging_steps=50,
            save_total_limit=1,
            seed=SEED,
            report_to=[],
        )
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("eval_strategy", "evaluation_strategy"), "epoch"))
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("save_strategy",), "epoch"))

        trainer_kwargs = dict(
            model=model,
            args=Seq2SeqTrainingArguments(**targs),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForSeq2Seq(tok, model=model),
        )
        trainer_kwargs.update(_pick_kwarg(Seq2SeqTrainer, ("processing_class", "tokenizer"), tok))

        print("Bắt đầu huấn luyện ...")
        t0 = time.time()
        Seq2SeqTrainer(**trainer_kwargs).train()
        mins = (time.time() - t0) / 60
        print(f"Huấn luyện xong sau {mins:.1f} phút.")
        model.save_pretrained(out_dir / "final")
        tok.save_pretrained(out_dir / "final")

    print("\nSinh bản tóm tắt trên tập đánh giá ...")
    preds = generate(
        model, tok, eval_rows, args.max_input, args.max_target,
        batch=args.gen_batch, prefix=args.prefix,
    )

    name = f"{args.model.split('/')[-1]}-{args.train_split}"
    refs = [r["abstract_raw"] for r in eval_rows]
    arts = [r["article_raw"] for r in eval_rows]
    result = evaluate(name, preds, refs, arts)

    tag = f"{name}_{args.eval_split}" + (f"_thu{args.eval_limit}" if args.eval_limit else "")
    print("\n" + table([result]))

    base_path = RESULTS / "tables" / f"baselines_{args.eval_split}.json"
    if base_path.exists() and not args.eval_limit:
        base = [r for r in json.loads(base_path.read_text(encoding="utf-8")) if r["name"] == "Lead-3"]
        if base and base[0]["n"] == result["n"]:
            print("\n" + compare(result, base[0], "rouge1"))

    save([result], f"{tag}.json")
    pred_dir = RESULTS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{tag}.json").write_text(
        json.dumps(
            {"guid": [r["guid"] for r in eval_rows], "reference": refs, name: preds},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã ghi kết quả vào {RESULTS / 'tables' / (tag + '.json')}")
    print(f"Checkpoint: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
