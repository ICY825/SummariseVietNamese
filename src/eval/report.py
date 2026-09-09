"""Bộ khung chấm điểm: từ dự đoán thô ra bảng kết quả có khoảng tin cậy.

Dùng chung cho cả năm tầng. Mỗi tầng chỉ cần sinh ra một danh sách chuỗi tóm tắt theo
đúng thứ tự của tập test, phần còn lại do đây lo — nhờ vậy không tầng nào tự chọn dạng
văn bản, tự tính ROUGE hay tự báo cáo trung bình trần không kèm khoảng tin cậy.

    from eval.report import evaluate, compare, table
    r_lead3 = evaluate("Lead-3", preds_lead3, refs, articles)
    r_vit5  = evaluate("ViT5",   preds_vit5,  refs, articles)
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


def evaluate(name, predictions, references, articles=None, n_boot=10_000, seed=13):
    """Chấm một hệ thống. Giữ lại điểm TỪNG BÀI vì `compare()` cần chúng."""
    per_article = score_all(predictions, references)
    out = {"name": name, "n": len(predictions), "per_article": {}, "corpus": {}}

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


def compare(a, b, metric="rouge1", n_boot=10_000, seed=13):
    """So sánh cặp đôi hai kết quả của `evaluate()` trên cùng tập bài."""
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
