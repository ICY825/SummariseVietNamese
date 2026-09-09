"""Bộ khung chấm điểm: từ dự đoán thô ra bảng kết quả có khoảng tin cậy.

Dùng chung cho cả năm tầng. Mỗi tầng chỉ cần sinh ra một danh sách chuỗi tóm tắt theo
đúng thứ tự của tập test, phần còn lại do đây lo — nhờ vậy không tầng nào tự chọn dạng
văn bản, tự tính ROUGE hay tự báo cáo trung bình trần không kèm khoảng tin cậy.

    from eval.report import evaluate, compare, table
    guids   = [str(r["guid"]) for r in rows]        # BAT BUOC neu con muon compare()
    r_lead3 = evaluate("Lead-3", preds_lead3, refs, articles, guids=guids)
    r_vit5  = evaluate("ViT5",   preds_vit5,  refs, articles, guids=guids)
    print(table([r_lead3, r_vit5]))
    print(compare(r_vit5, r_lead3, "rouge1"))
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text import for_scoring, syllables  # noqa: E402
from eval.rouge import score_all  # noqa: E402
from eval.stats import bootstrap_ci, paired_bootstrap  # noqa: E402

METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
RESULTS = Path(__file__).resolve().parents[2] / "results"


def novel_ngram_rate(summary, article, n):
    """Tỷ lệ n-gram của bản tóm tắt không xuất hiện trong bài gốc.

    Phân biệt "viết lại thật" với "chép câu". Tầng extractive theo định nghĩa phải ra
    gần 0; nếu một hệ thống abstractive cũng gần 0 thì nó chỉ đang chép, dù ROUGE cao.
    """
    s, a = syllables(for_scoring(summary)), syllables(for_scoring(article))
    S = {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}
    if not S:
        return 0.0
    A = {tuple(a[i : i + n]) for i in range(len(a) - n + 1)}
    return 100 * len(S - A) / len(S)


def evaluate(name, predictions, references, articles=None, guids=None,
             n_boot=10_000, seed=13):
    """Chấm một hệ thống. Giữ lại điểm TỪNG BÀI vì `compare()` cần chúng.

    `guids` là ID của các bài, theo ĐÚNG thứ tự của `predictions`. Luôn truyền vào:
    `compare()` dựa vào nó để từ chối ghép cặp hai hệ thống chấm trên hai tập bài
    khác nhau. Không có nó thì bảng kết quả ghi ra đĩa không còn tự chứng minh được
    mình chấm trên bài nào, và một lần chấm sai cặp sẽ không để lại dấu vết nào.
    """
    per_article = score_all(predictions, references)
    out = {"name": name, "n": len(predictions), "per_article": {}, "corpus": {}}
    if guids is not None:
        if len(guids) != len(predictions):
            raise ValueError(
                f"{name}: {len(guids)} guid nhưng {len(predictions)} dự đoán."
            )
        out["guid"] = [str(g) for g in guids]

    for m in METRICS:
        vals = [x[m] for x in per_article]
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        out["per_article"][m] = vals
        out["corpus"][m] = {"mean": mean, "lo": lo, "hi": hi}

    lens = [len(syllables(for_scoring(p))) for p in predictions]
    out["length"] = {
        "mean_syllables": sum(lens) / len(lens),
        "min": min(lens),
        "max": max(lens),
        "empty": sum(1 for x in lens if x == 0),
    }
    if articles is not None:
        out["novel"] = {
            f"{n}gram": sum(
                novel_ngram_rate(p, a, n) for p, a in zip(predictions, articles)
            ) / len(predictions)
            for n in (1, 2, 4)
        }
    return out


def same_articles(a, b):
    """Chặn việc ghép cặp hai hệ thống đã chấm trên HAI TẬP BÀI khác nhau.

    `paired_bootstrap()` chỉ kiểm được số lượng bài, mà cỡ bằng nhau hoàn toàn không
    có nghĩa là cùng tập bài — thử hai hệ thống chấm trên hai tập rời nhau, nó vẫn
    vui vẻ trả về `+100,00 [+100,00, +100,00] p=0,0000 CÓ ý nghĩa`.

    Đường đi nguy hiểm có thật: `vit5.py` nạp Lead-3 từ
    `results/tables/baselines_<split>.json` do một lần chạy KHÁC, ở thời điểm KHÁC
    ghi ra. Nếu `data/splits/` bị sinh lại giữa hai lần chạy thì hai bên vẫn cùng
    cỡ 1.000 bài nhưng khác bài, và bootstrap ghép cặp sẽ cho ra một khoảng tin cậy
    sai mà không báo gì. Sai kiểu đó không lộ ra ở bất cứ đâu trong bảng kết quả,
    nên phải chặn ngay tại chỗ ghép cặp.
    """
    ga, gb = a.get("guid"), b.get("guid")
    if ga is None or gb is None:
        thieu = ", ".join(r["name"] for r, g in ((a, ga), (b, gb)) if g is None)
        raise ValueError(
            f"Thiếu danh sách guid ở: {thieu}. Không kiểm chứng được hai hệ thống có "
            "chấm trên cùng tập bài hay không, mà cỡ bằng nhau thì không bảo đảm điều "
            "đó. Chấm lại bằng `evaluate(..., guids=...)` để sinh lại bảng."
        )
    if ga != gb:
        lech = sum(1 for x, y in zip(ga, gb) if x != y) + abs(len(ga) - len(gb))
        raise ValueError(
            f"{a['name']} và {b['name']} chấm trên HAI TẬP BÀI khác nhau "
            f"({lech} vị trí lệch guid) — không được ghép cặp. Thường là do "
            "`data/splits/` đã bị sinh lại sau khi bảng cũ được ghi; phải chấm lại "
            "cả hai hệ thống trên cùng tập bài đã đóng băng."
        )


def compare(a, b, metric="rouge1", n_boot=10_000, seed=13):
    """So sánh cặp đôi hai kết quả của `evaluate()` trên cùng tập bài."""
    same_articles(a, b)
    r = paired_bootstrap(
        a["per_article"][metric], b["per_article"][metric], n_boot=n_boot, seed=seed
    )
    verdict = "CÓ ý nghĩa" if r["significant"] else "KHÔNG đủ bằng chứng"
    return (
        f"{a['name']} − {b['name']} ({metric}): {r['diff']:+.2f} "
        f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f} → {verdict}"
    )


def table(results, metric_fmt="{mean:.2f} ±{half:.2f}"):
    """Bảng markdown, mỗi chỉ số kèm nửa khoảng tin cậy 95%."""
    head = "| Hệ thống | " + " | ".join(METRICS) + " | Độ dài | 2-gram mới |"
    sep = "|" + "---|" * (len(METRICS) + 3)
    rows = [head, sep]
    for r in results:
        cells = []
        for m in METRICS:
            c = r["corpus"][m]
            cells.append(metric_fmt.format(mean=c["mean"], half=(c["hi"] - c["lo"]) / 2))
        novel = f"{r['novel']['2gram']:.1f}%" if "novel" in r else "—"
        rows.append(
            f"| {r['name']} | " + " | ".join(cells)
            + f" | {r['length']['mean_syllables']:.0f} | {novel} |"
        )
    return "\n".join(rows)


def save(results, filename):
    """Ghi kết quả ra `results/tables/`. Điểm từng bài giữ lại để chấm lại không cần chạy mô hình."""
    path = RESULTS / "tables" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return path
