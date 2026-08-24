"""
MAPIE 校準可行性測試（Day 4，決策層方案 B 提前驗證）

對應 PROJECT_PLAN.md 第 9 節第 7 點：MAPIE 校準的分數是「共識一致率」，不是 Gemini
自報的 confidence。這支腳本不是正式校準——正式校準排在 8/28 標註集擴充到 80-100+ 之後，
現在樣本數（37 筆）太小，conformalize/test 切分後兩邊都個位數，統計上不穩定。

這裡只驗證兩件事：
1. MAPIE 1.5.0 的 API（SplitConformalClassifier + prefit=True + 自訂 proba 估計器）能不能
   接上我們「共識投票 -> 偽機率矩陣」的資料格式，整條路走不走得通。
2. 拿一致率算出的偽機率矩陣餵進 conformal predictor，看 prediction set 的行為合不合理：
   一致率高的 claim 該收斂成單一 verdict（可自主判定），一致率低的該產生多 verdict 的
   棄權集合（對應 CANNOT_DETERMINE）。

用法：
    pip install mapie
    python eval/mapie_feasibility.py --calibration labeled_set/calibration_data.json
"""

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from mapie.classification import SplitConformalClassifier

CLASSES = ["CANNOT_DETERMINE", "MATCH", "MISMATCH"]


class PrefitProbaClassifier:
    """MAPIE 需要一個有 predict_proba 的 sklearn 相容估計器；我們沒有訓練真的模型，
    機率是直接從共識投票算出來的，所以這裡包一層 passthrough：傳進去的 X 本身就是機率矩陣，
    predict_proba 原樣傳回，不做任何轉換。"""

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.asarray(X)

    def predict(self, X):
        X = np.asarray(X)
        return self.classes_[np.argmax(X, axis=1)]


def votes_to_proba(votes):
    counts = np.array([votes.count(c) for c in CLASSES], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.ones(len(CLASSES)) / len(CLASSES)
    return counts / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="labeled_set/calibration_data.json")
    parser.add_argument("--confidence-level", type=float, default=0.85,
                         help="MAPIE 目標覆蓋率（1-alpha），先用 0.85 對照 verify.py 現在的固定門檻")
    parser.add_argument("--conformalize-fraction", type=float, default=0.6,
                         help="標註集裡有多少比例拿去算 conformal 門檻，剩下當測試集看實際覆蓋率")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    rows = [r for r in rows if r["agreement_rate"] is not None]
    if len(rows) < 10:
        print(f"⚠️  只有 {len(rows)} 筆有完整共識一致率的資料，樣本數太小——這次先確認流程跑得通，"
              f"數字不用當真，等 8/28 標註集擴充後重跑才是真正有意義的校準。\n")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    split = int(len(rows) * args.conformalize_fraction)
    conf_idx, test_idx = idx[:split], idx[split:]

    X = np.array([votes_to_proba(r["votes"]) for r in rows])
    y = np.array([r["ground_truth"] for r in rows])

    X_conf, y_conf = X[conf_idx], y[conf_idx]
    X_test = X[test_idx]

    estimator = PrefitProbaClassifier(CLASSES)
    mapie_clf = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=args.confidence_level,
        conformity_score="lac",
        prefit=True,
    )
    mapie_clf.conformalize(X_conf, y_conf)

    y_pred, y_sets = mapie_clf.predict_set(X_test)

    print(f"conformalize 用 {len(conf_idx)} 筆，測試用 {len(test_idx)} 筆，目標信心水準 {args.confidence_level:.0%}")
    print(f"y_sets 型狀: {np.asarray(y_sets).shape}（用來確認這版 MAPIE 實際回傳格式）")
    print("=" * 70)

    y_sets_arr = np.asarray(y_sets)
    covered = 0
    single_verdict_count = 0
    for i, row_i in enumerate(test_idx):
        r = rows[row_i]
        mask = y_sets_arr[i].reshape(-1) if y_sets_arr[i].ndim == 1 else y_sets_arr[i, :, 0]
        pred_set = [CLASSES[j] for j, in_set in enumerate(mask) if in_set]
        in_set = r["ground_truth"] in pred_set
        covered += in_set
        single_verdict_count += len(pred_set) == 1
        mark = "✅" if in_set else "❌"
        print(f"{mark} [{r['claim_id']}] GT={r['ground_truth']:16s} agreement={r['agreement_rate']:.2f}  "
              f"prediction_set={pred_set}")

    n = len(test_idx)
    print("-" * 70)
    print(f"實際覆蓋率（ground truth 落在 prediction set 裡的比例）: {covered / n:.1%}（目標 {args.confidence_level:.0%}）")
    print(f"單一 verdict（可直接自主判定、不用棄權）比例: {single_verdict_count / n:.1%}")


if __name__ == "__main__":
    main()
