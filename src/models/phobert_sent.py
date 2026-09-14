"""Tầng 2: PhoBERT phân loại câu, kiểu BERTSum. Cần GPU.

    python src/models/phobert_sent.py --train-split train_20k --eval-split val
    python src/models/phobert_sent.py --no-train --model runs/phobert_sent/final

Tầng 0-1 chọn câu mà không học gì; tầng 3 sinh chữ mới. Tầng 2 nằm giữa: vẫn chỉ
**chọn câu có sẵn**, nhưng học cách chọn từ dữ liệu. Nó trả lời một câu hỏi cụ thể mà
Oracle-3 đã đặt ra — Oracle-3 đạt 48,09 còn Lead-3 chỉ 27,45, tức khoảng **21 điểm**
nằm trong tầm với chỉ nhờ chọn câu khéo hơn. Tầng 2 đo xem một bộ chọn học được lấy
lại bao nhiêu trong 21 điểm ấy.

**Nhãn lấy từ `oracle_indices()`**, đúng hàm đã dựng trần extractive ở tầng 0. Câu nào
nằm trong bộ oracle thì nhãn 1, còn lại nhãn 0. Dùng lại hàm ấy chứ không viết bộ sinh
nhãn riêng: nếu nhãn và trần đến từ hai cách tính khác nhau thì khoảng cách giữa tầng 2
và Oracle-3 không còn đọc được.

**Đầu vào là dạng TÁCH TỪ.** PhoBERT được pretrain trên văn bản đã tách từ, nên ở đây
dùng thẳng `sentences(article)` chứ không khử gạch dưới. Đầu ra vì thế cũng ở dạng tách
từ, giống hệt tầng 0-1 — khâu chấm điểm tự quy về dạng thô qua `for_scoring()`.

**Giới hạn 256 token là ràng buộc lớn nhất của tầng này, và nó đã được đo.** PhoBERT
chỉ có 256 vị trí. Đo trên 300 bài `val` bằng chính tokenizer PhoBERT: **80% số bài
vượt 256 token**, và một cửa sổ chỉ nhìn thấy trung bình **60,6% số câu**. Hệ quả là
trần của tầng này không phải 49,8 mà là **44,4** — Oracle-3 bị giới hạn trong đúng cửa
sổ ấy. Vẫn còn hơn Lead-3 khoảng 16 điểm, tức thừa dư địa để tầng 2 có nghĩa.

Không làm cửa sổ trượt ở bản đầu: đó là thêm một biến chưa đo vào một tầng chưa chạy.
Con số 44,4 được báo cáo như một trần thứ hai, và chính khoảng cách giữa 44,4 với 49,8
là một kết quả — nó đo cái giá của việc chọn một bộ mã hoá chỉ đọc được 256 token.

**Biểu diễn câu là trung bình token của câu đó**, không phải `[CLS]` chèn trước mỗi câu
như BERTSum gốc. Lý do giống chỗ `extractive._phobert_vectors()` đã ghi: `[CLS]` của
một mô hình chỉ pretrain MLM không hề được huấn luyện để làm vector câu. Cả bài vẫn
được mã hoá MỘT lần nên mỗi câu vẫn thấy ngữ cảnh của các câu khác — đó là điểm khác
căn bản so với LexRank bản nhúng, nơi mỗi câu được mã hoá riêng lẻ.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.splits import load_split  # noqa: E402
from data.text import sentences  # noqa: E402
from eval.report import compare, evaluate, save, table  # noqa: E402
from models.extractive import join, oracle_indices  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
K = 3            # so cau moi ban tom tat, trung tang 0-1
MAX_LEN = 256    # gioi han vi tri cua PhoBERT, khong phai lua chon


def build_spans(sents, encode, max_len=MAX_LEN):
    """Ghép câu thành một chuỗi và ghi lại vị trí từng câu trong đó.

    Trả về `(ids, spans)`: `ids` chưa có token đặc biệt, `spans[i]` là `(dau, cuoi)`
    của câu thứ i **tính cả token mở đầu** sẽ được thêm ở ngoài. Câu nào không lọt vào
    cửa sổ thì không có span — nó vô hình với mô hình, và cũng không được gán nhãn.

    Cắt theo CÂU chứ không cắt giữa câu: một câu cụt nửa chừng vẫn chiếm chỗ mà biểu
    diễn của nó thì không còn đáng tin, nên thà bỏ hẳn.
    """
    ids, spans = [], []
    for s in sents:
        t = encode(s)
        if not t:
            continue
        # +1 cho token mo dau, +1 cho token ket thuc se duoc them o ngoai.
        if 1 + len(ids) + len(t) + 1 > max_len:
            break
        spans.append((1 + len(ids), 1 + len(ids) + len(t)))
        ids.extend(t)
    return ids, spans


def labels_for(sents, reference, n_visible, k=K):
    """Nhãn nhị phân cho các câu NHÌN THẤY ĐƯỢC, lấy từ bộ oracle của cả bài.

    Oracle chạy trên **toàn bài** chứ không chỉ trong cửa sổ: câu tốt nhất nằm ngoài
    cửa sổ thì mô hình không học được nó, và đó đúng là cái giá của giới hạn 256 token
    chứ không phải thứ nên giấu đi bằng cách đổi định nghĩa nhãn.
    """
    idx = set(oracle_indices(" ".join(sents), reference, k))
    return [1.0 if i in idx else 0.0 for i in range(n_visible)]


def pick_indices(scores, sents, k=K):
    """k câu điểm cao nhất, bỏ qua câu trùng nội dung, trả về theo thứ tự bài.

    Cùng quy ước với `extractive._pick()`: hoà điểm thì câu đứng trước thắng, và hai
    bản sao của cùng một câu không được chọn cả hai.
    """
    out, seen = [], set()
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        i = int(i)
        if sents[i] in seen:
            continue
        seen.add(sents[i])
        out.append(i)
        if len(out) == k:
            break
    return sorted(out)


# --------------------------------------------------------------------------
# Phan duoi day can `torch`; nap luoi de `selftest.py` chay duoc tren `.venv` sach.
# --------------------------------------------------------------------------

def _build_model(name, device):
    import torch
    from transformers import AutoModel

    class SentenceScorer(torch.nn.Module):
        """PhoBERT + một lớp tuyến tính cho điểm từng câu."""

        def __init__(self, name):
            super().__init__()
            self.enc = AutoModel.from_pretrained(name)
            self.drop = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask, span_mask):
            # span_mask: (B, S_cau, L) — trung binh co mat na, mot hang cho moi cau.
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            w = span_mask.unsqueeze(-1)
            v = (h.unsqueeze(1) * w).sum(2) / w.sum(2).clamp(min=1e-9)
            return self.head(self.drop(v)).squeeze(-1)

    model = SentenceScorer(name)
    # Checkpoint da huan luyen gom HAI phan: `enc` luu bang `save_pretrained`, con lop
    # cho diem luu rieng o `head.pt`. `from_pretrained` chi nap phan dau; khong nap
    # `head.pt` thi lop cho diem la trong so khoi tao NGAU NHIEN va `--no-train` sinh ra
    # ban tom tat vo nghia ma khong bao loi gi.
    head = Path(name) / "head.pt"
    if head.exists():
        model.head.load_state_dict(torch.load(head, map_location="cpu"))
    return model.to(device)


def _collate(batch, pad_id, device):
    import torch

    L = max(len(b["ids"]) for b in batch)
    S = max(len(b["spans"]) for b in batch)
    n = len(batch)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    span_mask = torch.zeros((n, S, L))
    y = torch.zeros((n, S))
    y_mask = torch.zeros((n, S))
    for i, b in enumerate(batch):
        ids = b["ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
        for j, (a, z) in enumerate(b["spans"]):
            span_mask[i, j, a:z] = 1.0
            y_mask[i, j] = 1.0
            if b["labels"]:
                y[i, j] = b["labels"][j]
    return (input_ids.to(device), attn.to(device), span_mask.to(device),
            y.to(device), y_mask.to(device))


def encode_rows(rows, tok, with_labels=True, k=K, max_len=MAX_LEN):
    """Bài -> ví dụ huấn luyện. Bỏ bài không có câu nào lọt cửa sổ."""
    bos, eos = tok.cls_token_id, tok.sep_token_id
    out = []
    for r in rows:
        sents = sentences(r["article"])
        ids, spans = build_spans(
            sents, lambda s: tok(s, add_special_tokens=False)["input_ids"], max_len)
        if not spans:
            out.append(None)
            continue
        lab = labels_for(sents, r["abstract"], len(spans), k) if with_labels else None
        out.append({"ids": [bos] + ids + [eos], "spans": spans, "labels": lab,
                    "sents": sents, "n_visible": len(spans)})
    return out


def score_examples(model, examples, pad_id, device, batch=8):
    """Logit của từng câu nhìn thấy được, mỗi bài một danh sách; bài không có câu nào -> None.

    Tách khỏi `main()` để `phobert_select.py` dùng lại đúng đường tính điểm này: mọi
    quy tắc chọn câu phải đọc cùng một bộ điểm thì so với nhau mới có nghĩa.
    """
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(examples), batch):
            chunk = examples[i : i + batch]
            thuc = [e for e in chunk if e]
            if thuc:
                ii, am, sm, _, _ = _collate(thuc, pad_id, device)
                logits = model(ii, am, sm).float().cpu().numpy()
            it = iter(range(len(thuc)))
            for e in chunk:
                out.append(logits[next(it), : e["n_visible"]].tolist() if e else None)
            print(f"\r  {min(i + batch, len(examples))}/{len(examples)}", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description="Tầng 2: PhoBERT phân loại câu.")
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--name", default="phobert-sent")
    ap.add_argument("--train-split", default="train_20k")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--train-limit", type=int, default=0,
                    help="chỉ lấy N bài train đầu — để chạy thử đường ống, "
                         "số liệu KHÔNG dùng báo cáo")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    if args.no_train and not (Path(args.model) / "head.pt").exists():
        raise SystemExit(
            f"--no-train cần thư mục checkpoint có head.pt, mà {args.model} không có. "
            "Thiếu nó thì lớp cho điểm là trọng số ngẫu nhiên.")

    import torch
    from transformers import AutoTokenizer

    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CẢNH BÁO: không thấy GPU. Huấn luyện trên CPU sẽ mất hàng giờ.")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Nạp {args.eval_split}{'' if args.no_train else ' và ' + args.train_split} ...")
    train_rows = [] if args.no_train else list(load_split(args.train_split, add_raw=False))
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    eval_rows = list(load_split(args.eval_split, add_raw=False))
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
    print(f"  train {len(train_rows)} bài, eval {len(eval_rows)} bài.\n")

    print("Dựng ví dụ (sinh nhãn oracle — chậm nhất ở khâu này) ...")
    t0 = time.time()
    train_ex = [e for e in encode_rows(train_rows, tok, True, args.k, args.max_len) if e]
    eval_ex = encode_rows(eval_rows, tok, True, args.k, args.max_len)
    print(f"  {len(train_ex)} ví dụ train, {sum(1 for e in eval_ex if e)} eval "
          f"({time.time() - t0:.0f}s)")
    nhin_thay = [e["n_visible"] / max(len(e["sents"]), 1) for e in eval_ex if e]
    print(f"  cửa sổ {args.max_len} token nhìn thấy trung bình "
          f"{100 * float(np.mean(nhin_thay)):.1f}% số câu\n")

    model = _build_model(args.model, device)
    out_dir = Path(args.out) / "phobert_sent"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_info = None
    if not args.no_train:
        pos = sum(sum(e["labels"]) for e in train_ex)
        tong = sum(len(e["labels"]) for e in train_ex)
        pos_weight = torch.tensor((tong - pos) / max(pos, 1.0), device=device)
        print(f"Nhãn dương {pos:.0f}/{tong:.0f} ({100 * pos / tong:.1f}%) "
              f"-> pos_weight {pos_weight.item():.2f}")
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        n_step = int(args.epochs * (len(train_ex) + args.batch - 1) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(n_step, 1), pct_start=0.1)
        hist, step = [], 0
        t0 = time.time()
        for ep in range(int(args.epochs)):
            model.train()
            rng = np.random.default_rng(13 + ep)
            order = rng.permutation(len(train_ex))
            tot = 0.0
            for i in range(0, len(order), args.batch):
                batch = [train_ex[j] for j in order[i : i + args.batch]]
                ii, am, sm, y, ym = _collate(batch, tok.pad_token_id, device)
                logits = model(ii, am, sm)
                loss = (lossf(logits, y) * ym).sum() / ym.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                tot += loss.item()
                step += 1
                if step % 50 == 0:
                    hist.append({"step": step, "loss": round(tot / 50, 4)})
                    print(f"\r  epoch {ep + 1} bước {step}/{n_step} loss {tot / 50:.4f}",
                          end="", flush=True)
                    tot = 0.0
            print()
        train_info = {"minutes": round((time.time() - t0) / 60, 1), "steps": step,
                      "log_history": hist, "pos_rate": round(pos / tong, 4)}
        model.enc.save_pretrained(out_dir / "final")
        tok.save_pretrained(out_dir / "final")
        torch.save(model.head.state_dict(), out_dir / "final" / "head.pt")
        print(f"Đã lưu {out_dir / 'final'}")

    print("\nSinh bản tóm tắt trên", args.eval_split, "...")
    scores = score_examples(model, eval_ex, tok.pad_token_id, device, args.batch)
    preds = [
        "" if s is None
        else join(e["sents"], pick_indices(s, e["sents"][: e["n_visible"]], args.k))
        for e, s in zip(eval_ex, scores)
    ]

    refs = [r["abstract"] for r in eval_rows]
    arts = [r["article"] for r in eval_rows]
    guids = [str(r["guid"]) for r in eval_rows]
    res = evaluate(args.name, preds, refs, arts, guids=guids, n_boot=args.n_boot)
    print("\n" + table([res]))

    # So cap voi moc chi hop le khi hai ben cham tren DUNG cung nhung bai do. Lan chay
    # thu (`--eval-limit`) thi khong, va khi ay bo qua chu khong duoc do vo: `compare()`
    # nem ValueError, ma mot lan chay 40 phut GPU khong duoc chet o buoc in bang.
    moc_path = RESULTS / "tables" / f"baselines_{args.eval_split}.json"
    versus = {}
    if moc_path.exists():
        for ten in ("Lead-3", "Oracle-3"):
            m = next((r for r in json.loads(moc_path.read_text(encoding="utf-8"))
                      if r["name"] == ten), None)
            if m is None:
                continue
            if m.get("guid") != res["guid"]:
                print(f"  Bỏ qua so cặp với {ten}: khác tập bài "
                      f"({len(m.get('guid', []))} so với {len(res['guid'])}).")
                continue
            versus[ten] = compare(res, m, "rouge1", n_boot=args.n_boot)
            print("  " + versus[ten])

    tag = f"{args.name}-{args.train_split}_{args.eval_split}_len{args.max_len}"
    if args.eval_limit or args.train_limit:
        tag += f"_thu{args.eval_limit or args.train_limit}"
    # `--no-train` cho ra dung ten cua lan chay da huan luyen; khong co dong nay thi no
    # de im lang len bang ket qua that. Trung ten thi them hau to, khong bao gio de.
    from models.vit5 import free_tag
    tag = free_tag(tag, (RESULTS / "tables", RESULTS / "predictions"))
    path = save([res], f"{tag}.json")
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS / "predictions" / f"{tag}.json").write_text(
        json.dumps({"guid": guids, "reference": refs, args.name: preds}, ensure_ascii=False),
        encoding="utf-8")
    run = {
        "tag": tag, "name": args.name,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "data": {"train_split": args.train_split, "n_train": len(train_rows),
                 "eval_split": args.eval_split, "n_eval": len(eval_rows),
                 "max_len": args.max_len, "k": args.k, "seed": 13},
        "window": {"ty_le_cau_nhin_thay": round(float(np.mean(nhin_thay)), 4)},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "train": train_info,
        "scores": {"corpus": res["corpus"], "length": res["length"], "novel": res["novel"]},
        "versus": versus,
    }
    (RESULTS / "tables" / f"{tag}_run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi:\n  {path}\n  {RESULTS / 'tables' / f'{tag}_run.json'}")


if __name__ == "__main__":
    main()
