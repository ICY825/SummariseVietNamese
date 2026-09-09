"""Bootstrap: khoảng tin cậy và kiểm định ý nghĩa cho so sánh giữa các hệ thống.

Đo trên tập test thật, độ lệch chuẩn ROUGE-1 giữa các bài là 9,8 — lớn hơn chính
khoảng cách giữa các hệ thống. Nghĩa là **không được** kết luận A hơn B chỉ vì trung
bình của A cao hơn; phải kèm khoảng tin cậy.

Điểm mấu chốt của `paired_bootstrap()`: lấy lại mẫu **chỉ số bài**, rồi áp đúng bộ chỉ
số đó cho cả hai hệ thống. Hai hệ thống chấm trên cùng một bài có điểm tương quan mạnh
(bài dễ thì hệ nào cũng cao), nên giữ nguyên ghép cặp sẽ khử phần lớn phương sai do độ
khó của bài. Lấy mẫu độc lập cho hai hệ thống là vứt bỏ đúng thông tin đó và cho khoảng
tin cậy rộng vô ích.
"""

import numpy as np


def bootstrap_ci(values, n_boot=10_000, seed=13, alpha=0.05):
    """Khoảng tin cậy percentile cho điểm trung bình của một hệ thống.

    Trả về `(trung_bình, cận_dưới, cận_trên)`.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("Danh sách điểm rỗng.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot=10_000, seed=13, alpha=0.05):
    """So sánh cặp đôi A trừ B trên cùng tập bài.

    Trả về dict: `diff` (chênh lệch trung bình), `lo`/`hi` (khoảng tin cậy của chênh
    lệch), `p` (giá trị p hai phía), `significant` (khoảng tin cậy có chứa 0 không).

    Giá trị p theo lối lấy lại mẫu của Koehn (2004): tỷ lệ lần lấy lại mẫu mà chênh
    lệch đổi dấu so với quan sát, nhân đôi cho hai phía.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Hai hệ thống phải chấm trên cùng số bài: {x.shape} vs {y.shape}.")
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))  # cung bo chi so cho ca hai
    diffs = d[idx].mean(axis=1)
    obs = float(d.mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = float((diffs <= 0).mean() if obs > 0 else (diffs >= 0).mean())
    return {
        "diff": obs,
        "lo": float(lo),
        "hi": float(hi),
        "p": min(1.0, 2 * tail),
        "significant": not (lo <= 0.0 <= hi),
    }
