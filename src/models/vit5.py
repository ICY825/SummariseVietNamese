"""Tầng 3: fine-tune mô hình sinh văn bản (ViT5, đối chứng BARTpho) cho tóm tắt.

    python src/models/vit5.py --train-split train_2k --epochs 1     # chay thu
    python src/models/vit5.py --train-split train_10k --epochs 3
    python src/models/vit5.py --model vinai/bartpho-syllable --train-split train_10k

Tuần 5 dò tham số sinh: nạp checkpoint đã có rồi chỉ sinh lại trên `tune`, không train.
`--name` để bảng kết quả mang tên mô hình chứ không phải tên thư mục checkpoint:

    python src/models/vit5.py --no-train --eval-split tune \
        --model runs/VietAI_vit5-base_train_20k/final --name vit5-base-train_20k \
        --length-penalty 2.0 --min-length 20

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

**Mỗi lần chạy để lại `run.json`.** Siêu tham số, đường cong loss từng bước
(`trainer.state.log_history`), checkpoint được chọn, tham số sinh, phiên bản thư
viện và tên GPU — tất cả ghi ra đĩa cạnh bảng chỉ số. Trước đây những thứ ấy chỉ
tồn tại trên màn hình Colab: đóng tab là một bảng kết quả không còn tự nói được nó
sinh ra từ cấu hình nào, và đường cong học không vẽ lại được.

**Chấm điểm qua `eval.report`.** Mô hình sinh ra văn bản thô còn baseline ra văn bản
tách từ; `for_scoring()` đã lo việc quy về một dạng. Tự tính ROUGE ở đây là phá phép
so sánh có kiểm soát mà tầng 0-1 đã dựng.

CẢNH BÁO RÒ RỈ: trên Hub có checkpoint ViT5 đã fine-tune sẵn cho tóm tắt VietNews.
Điểm khởi đầu phải là bản pretrain thuần (`VietAI/vit5-base`). Dùng bản đã fine-tune
là để mô hình nhìn trước tập test, và mọi con số sau đó vô nghĩa.

ĐỪNG HOẢNG VÌ `loss` IN RA QUÁ CAO. Bản `transformers` trên Colab ghi loss huấn
luyện **đã nhân với `gradient_accumulation_steps`**. Đo thực tế trên `train_2k`:

    grad_accum = 8  ->  loss in ra 18,4   eval_loss 1,97
    grad_accum = 1  ->  loss in ra  4,1   eval_loss 3,28

Cùng một mô hình, cùng dữ liệu, chỉ khác cách gộp gradient. Đây là lỗi chuẩn hoá khi
báo cáo, không đụng đến gradient nên không ảnh hưởng chất lượng. Dấu hiệu nhận ra:
loss huấn luyện cao gấp đúng `grad_accum` lần so với `eval_loss`. **Luôn nhìn
`eval_loss` để phán đoán**, vì nó không đi qua đường tích luỹ gradient. Mốc lành
mạnh cho T5 fine-tune tóm tắt là khoảng 1,8-3,5; trên 10 là thật sự có vấn đề (từ
vựng ~36k nên đoán bừa đã là ln(36000) ≈ 10,5).
"""

import argparse
import inspect
import json
import platform
import shutil
import sys
import time
from pathlib import Path

# Console Windows mac dinh khong phai UTF-8. Phai dat ca stderr, khong chi stdout:
# thong bao chan `test` di ra bang SystemExit, tuc qua stderr — mot canh bao quan
# trong ma doc khong noi thi coi nhu khong co.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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


def gen_kwargs(args):
    """Tham số sinh, gom một chỗ để vừa truyền vào `generate()` vừa ghi vào `run.json`."""
    return {
        "num_beams": args.beams,
        "no_repeat_ngram_size": args.no_repeat_ngram,
        "length_penalty": args.length_penalty,
        "min_length": args.min_length,
        "early_stopping": True,
    }


def run_tag(args, name):
    """Tên file mang theo cấu hình đã sinh ra nó.

    Trước đây tag chỉ gồm mô hình và tập train, nên hai lần chạy khác `lr` hay khác
    `epochs` ghi trùng tên và lần sau ĐÈ IM LẶNG lên lần trước. Tuần 5 chạy đường
    cong học với nhiều cấu hình trên cùng một split nên chắc chắn dính. Nhét cấu
    hình vào tên là cách rẻ nhất để một file kết quả tự khai nó là của lần chạy nào.
    """
    bits = [name, args.eval_split, f"in{args.max_input}"]
    # Luong `--no-train` khong dung epochs/lr/batch, nen nhet chung vao ten la ghi
    # vao file mot cau hinh chua he chay — doc bang ket qua tuan 5 se tuong lan chay
    # do co train. Nhung mieng phan biet cac lan do tham so sinh nam o duoi.
    if not getattr(args, "no_train", False):
        bits[2:2] = [f"e{args.epochs:g}", f"lr{args.lr:g}", f"bs{args.batch * args.grad_accum}"]
    if args.beams != 4:
        bits.append(f"beam{args.beams}")
    if args.length_penalty != 1.0:
        bits.append(f"lp{args.length_penalty:g}")
    if args.min_length:
        bits.append(f"min{args.min_length}")
    if args.eval_limit:
        bits.append(f"thu{args.eval_limit}")
    return "_".join(bits)


def free_tag(tag, dirs):
    """Tag chưa bị dùng ở bất kỳ thư mục nào trong `dirs`.

    Một lần chạy 60 phút trên Colab không bao giờ được phép đè kết quả cũ, kể cả khi
    cấu hình trùng khít. Trùng thì thêm hậu tố, không hỏi và không đè.
    """
    candidate, i = tag, 2
    while any((d / f"{candidate}.json").exists() for d in dirs):
        candidate, i = f"{tag}-{i}", i + 1
    if candidate != tag:
        print(f"  {tag}.json đã có — ghi thành {candidate}.json để không đè kết quả cũ.")
    return candidate


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


def generate(model, tok, rows, max_input, max_target, batch=16, prefix="", gen=None):
    """Sinh bản tóm tắt cho cả tập, giữ nguyên thứ tự bài.

    `gen` là các tham số sinh. Truyền vào tường minh chứ không ghim trong thân hàm,
    vì tuần 5 khảo sát đúng chúng: một bảng chỉ số không nói rõ nó sinh bằng
    `length_penalty` nào thì không so được với bảng khác.
    """
    import torch

    gen = dict(gen or {"num_beams": 4, "no_repeat_ngram_size": 3, "early_stopping": True})
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
            ids = model.generate(**enc, max_length=max_target, **gen)
        out.extend(tok.batch_decode(ids, skip_special_tokens=True))
        done = min(i + batch, len(rows))
        print(f"\r  sinh {done}/{len(rows)}", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description="Fine-tune tầng 3.")
    ap.add_argument("--model", default="VietAI/vit5-base")
    ap.add_argument(
        "--name", default="",
        help="tên hệ thống trong bảng kết quả; mặc định suy từ --model và --train-split. "
             "Cần khi --model là đường dẫn checkpoint, vì khi đó tên suy ra là `final`",
    )
    ap.add_argument("--train-split", default="train_2k")
    ap.add_argument("--eval-split", default="val", help="KHÔNG dùng test ở tuần 4-5")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=2, help="mỗi thiết bị")
    ap.add_argument("--grad-accum", type=int, default=8, help="batch hiệu dụng = batch * cái này")
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=4, help="số tia khi sinh")
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument(
        "--length-penalty", type=float, default=1.0,
        help="tuần 5 khảo sát trên `tune`; lớn hơn 1 thì sinh dài hơn",
    )
    ap.add_argument("--min-length", type=int, default=0, help="tuần 5 khảo sát trên `tune`")
    ap.add_argument("--max-input", type=int, default=MAX_INPUT)
    ap.add_argument("--max-target", type=int, default=MAX_TARGET)
    ap.add_argument("--prefix", default="", help='tiền tố kiểu T5, ví dụ "vietnews: "')
    ap.add_argument("--eval-limit", type=int, default=0, help="chỉ sinh N bài đầu khi chạy thử")
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=0, help="dừng sớm, để chẩn đoán")
    ap.add_argument(
        "--fp16",
        default="auto",
        choices=["auto", "on", "off"],
        help="auto = BẬT (nhanh hơn 33%%); off nếu eval_loss thành NaN",
    )
    ap.add_argument("--out", default="runs", help="thư mục lưu; trên Colab hãy trỏ vào Drive")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="chạy tiếp từ checkpoint gần nhất trong --out, khi Colab ngắt giữa chừng",
    )
    ap.add_argument("--no-train", action="store_true", help="chỉ nạp và sinh, để thử đường ống")
    args = ap.parse_args()

    if args.eval_split == "test":
        raise SystemExit(
            "Từ chối chấm trên `test`. Tập này dùng MỘT lần ở cuối dự án; "
            "tuần 4-5 theo dõi trên `val`, dò tham số trên `tune`."
        )

    import torch
    import transformers
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
    # `Trainer` tu day model len GPU khi huan luyen, nhung luong `--no-train` thi
    # KHONG ai lam viec do: model nam nguyen tren CPU va sinh 1.000 bai mat hang gio.
    # Tuan 5 dung dung luong nay — nap checkpoint da luu roi chi sinh lai voi tham so
    # sinh khac — nen phai chuyen thiet bi ngay tu day.
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Nạp {args.train_split} và {args.eval_split} ...")
    train_rows = list(load_split(args.train_split, add_raw=True))
    eval_rows = list(load_split(args.eval_split, add_raw=True))
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
        print(f"  CHẠY THỬ: chỉ sinh {len(eval_rows)} bài — số liệu KHÔNG dùng báo cáo.")
    print(f"  train {len(train_rows)} bài, eval {len(eval_rows)} bài.\n")

    out_dir = Path(args.out) / f"{args.model.replace('/', '_')}_{args.train_split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_info = None   # con None khi chay --no-train

    # fp16 BAT mac dinh vi do duoc nhanh hon 33% tren T4 (4,0 so voi 3,0 mau/giay),
    # va do thuc te khong thay bat on: eval_loss lanh manh o ca fp16 lan fp32.
    #
    # Rui ro co that nhung chua xay ra o day: ho T5 duoc pretrain o bfloat16, ma fp16
    # co dai so mu hep hon nen ve ly thuyet co the tran va cho NaN o nhung lan chay
    # dai. No se lo ra ngay tai eval_loss cuoi moi epoch. Thay NaN thi chay lai voi
    # `--fp16 off` (T4 khong ho tro bf16 nen duong lui la fp32, cham hon 33%).
    use_fp16 = torch.cuda.is_available() and args.fp16 in ("auto", "on")
    print(f"  fp16 = {use_fp16}")

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
            fp16=use_fp16,
            logging_steps=args.logging_steps,
            save_total_limit=2,   # can giu >1 de con checkpoint tot nhat
            seed=SEED,
            report_to=[],
        )
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("eval_strategy", "evaluation_strategy"), "epoch"))
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("save_strategy",), "epoch"))
        # Lan chay train_5k cho eval_loss 1,802 -> 1,790 -> 1,796: epoch cuoi TE HON
        # epoch 2. Lay checkpoint cuoi la lay ban da bat dau qua khop. Ba tham so
        # duoi bao Trainer nap lai ban co eval_loss thap nhat truoc khi sinh van ban.
        # Chi hoat dong khi save_strategy va eval_strategy trung nhau - o day deu la
        # "epoch" - va save_total_limit > 1.
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("load_best_model_at_end",), True))
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("metric_for_best_model",), "eval_loss"))
        targs.update(_pick_kwarg(Seq2SeqTrainingArguments, ("greater_is_better",), False))
        if args.max_steps:
            targs["max_steps"] = args.max_steps

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
        # Colab free ngat phien bat chot. `save_strategy="epoch"` da ghi checkpoint
        # vao --out sau moi epoch, nen --resume cho phep chay tiep thay vi lam lai
        # tu dau. Trainer tu tim checkpoint moi nhat khi truyen True.
        trainer = Seq2SeqTrainer(**trainer_kwargs)
        trainer.train(resume_from_checkpoint=args.resume or None)
        mins = (time.time() - t0) / 60
        print(f"Huấn luyện xong sau {mins:.1f} phút.")
        # `load_best_model_at_end` khong in ra dau vet nao trong log cua Colab, nen
        # khong the biet ban tom tat duoc sinh bang trong so cua epoch nao. In ra
        # day de lan chay sau tu ghi lai dieu do vao ket qua.
        best = getattr(trainer.state, "best_model_checkpoint", None)
        best_loss = getattr(trainer.state, "best_metric", None)
        print(f"  Checkpoint tốt nhất: {best or 'KHÔNG có — dùng trọng số cuối'}"
              f" (eval_loss {best_loss})")
        # `log_history` giu loss moi `logging_steps` buoc va `eval_loss` cuoi moi
        # epoch — day la nguyen lieu duy nhat de ve duong cong hoc trong bao cao.
        # No song trong bo nho cua tien trinh: het `main()` la mat, nen phai ghi ra
        # dia chu khong chi in len man hinh Colab.
        train_info = {
            "minutes": round(mins, 1),
            "best_checkpoint": best,
            "best_eval_loss": best_loss,
            "epochs_ran": getattr(trainer.state, "epoch", None),
            "global_step": getattr(trainer.state, "global_step", None),
            "log_history": list(getattr(trainer.state, "log_history", [])),
        }
        model.save_pretrained(out_dir / "final")
        tok.save_pretrained(out_dir / "final")

    print("\nSinh bản tóm tắt trên tập đánh giá ...")
    gen = gen_kwargs(args)
    print(f"  tham số sinh: {gen}")
    preds = generate(
        model, tok, eval_rows, args.max_input, args.max_target,
        batch=args.gen_batch, prefix=args.prefix, gen=gen,
    )

    name = args.name or f"{args.model.split('/')[-1]}-{args.train_split}"
    refs = [r["abstract_raw"] for r in eval_rows]
    arts = [r["article_raw"] for r in eval_rows]
    guids = [str(r["guid"]) for r in eval_rows]
    result = evaluate(name, preds, refs, arts, guids=guids)

    tables_dir = RESULTS / "tables"
    pred_dir = RESULTS / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    tag = free_tag(run_tag(args, name), (tables_dir, pred_dir, out_dir))
    print("\n" + table([result]))

    versus = None
    base_path = tables_dir / f"baselines_{args.eval_split}.json"
    if base_path.exists() and not args.eval_limit:
        base = [r for r in json.loads(base_path.read_text(encoding="utf-8")) if r["name"] == "Lead-3"]
        if base:
            # Bang Lead-3 nay do MOT LAN CHAY KHAC ghi ra, co the la truoc khi
            # `data/splits/` duoc sinh lai. `compare()` doi chieu guid nen se bao
            # loi neu hai ben khac tap bai — do la tin tuc that su can biet chu
            # khong phai phien toai. Bat lai vi khoi so sanh nay chay TRUOC buoc
            # ghi file: de no nem loi la mat trang mot phien Colab 60 phut.
            try:
                versus = compare(result, base[0], "rouge1")
                print("\n" + versus)
            except ValueError as e:
                versus = f"BỎ QUA — {e}"
                print(f"\nBỎ QUA so sánh với Lead-3 — {e}")

    table_path = save([result], f"{tag}.json")
    pred_path = pred_dir / f"{tag}.json"
    pred_path.write_text(
        json.dumps(
            {"guid": [r["guid"] for r in eval_rows], "reference": refs, name: preds},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Ho so cua lan chay. Bang chi so tra loi "duoc bao nhieu diem"; file nay tra loi
    # "diem do sinh ra bang cach nao" — sieu tham so, duong cong loss, checkpoint nao
    # duoc chon, tham so sinh, GPU, phien ban thu vien. Do la thu can co khi trinh
    # bay, va la thu duy nhat chung minh mot bang ket qua thuoc ve cau hinh nao.
    run_path = tables_dir / f"{tag}_run.json"
    record = {
        "tag": tag,
        "name": name,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "generation": gen,
        "data": {
            "train_split": args.train_split,
            "n_train": len(train_rows),
            "eval_split": args.eval_split,
            "n_eval": len(eval_rows),
            "max_input": args.max_input,
            "max_target": args.max_target,
            "seed": SEED,
        },
        "env": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "fp16": use_fp16,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "train": train_info,
        "scores": {
            "corpus": result["corpus"],
            "length": result["length"],
            "novel": result.get("novel"),
        },
        "versus_lead3": versus,
        "files": {
            "table": str(table_path),
            "predictions": str(pred_path),
            "checkpoint": args.model if args.no_train else str(out_dir / "final"),
        },
    }
    run_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # `RESULTS` nam trong ban clone cua repo. Tren Colab cho nay LA TAM THOI: ngat
    # phien la mat sach, va mot lan chay 62 phut ma mat ket qua thi phai chay lai tu
    # dau. Chep them mot ban vao `--out` (thuong tro vao Drive) de ket qua song sot.
    # Chep chu khong doi cho ghi: doc so lieu van o dung noi ma README trich dan.
    #
    # PHAI them tien to thu muc cha. `table_path` va `pred_path` dung CHUNG mot
    # `tag` nen basename cua chung giong het nhau; chep thang bang `src_path.name`
    # thi ban predictions de len ban bang chi so, va thu duy nhat song sot qua mot
    # phien Colab bi ngat lai la thu KHONG chua diem tung bai, khoang tin cay hay
    # guid. Dung cai ma ban sao an toan sinh ra de bao ve.
    saved = []
    for src_path in (table_path, pred_path, run_path):
        dest = out_dir / f"{src_path.parent.name}_{src_path.name}"
        shutil.copy2(src_path, dest)
        saved.append(dest)

    print(f"\nĐã ghi kết quả:\n  {table_path}\n  {pred_path}\n  {run_path}  <- hồ sơ lần chạy")
    print("Bản sao an toàn (sống sót khi ngắt phiên):")
    for dest in saved:
        print(f"  {dest}")
    print(f"Checkpoint: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
