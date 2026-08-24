"""
驗證引擎精確率測試

對應 pain-point-analysis/ai-影片生成重跑浪費/analysis.md 第 8 節「實驗 1」：
用 20-30 顆標註鏡頭量測驗證引擎精確率，判準 精確率 >= 85%（誤報率 <= 15%）。

用法：
    這個腳本走 Vertex AI 模式（不是 Google AI Studio 的 API key 模式），
    因為只有 Vertex AI 的用量會實際算進 GCP 專案的 credit（$300 試用額度／hackathon 抵用金），
    API key 模式是另一個完全無關的免費額度池，用了也不會扣到你申請的 credit。

    正式跑之前先做好：
        gcloud auth application-default login
        gcloud config set project <你的專案ID>
        pip install "google-cloud-aiplatform[agent_engines,adk]>=1.101.0"

    然後：
        set GOOGLE_CLOUD_PROJECT=<你的專案ID>      (PowerShell: $env:GOOGLE_CLOUD_PROJECT="...")
        set GOOGLE_CLOUD_LOCATION=us-central1        (PowerShell: $env:GOOGLE_CLOUD_LOCATION="...")
        python run_precision_eval.py --labels labeled_set/template.csv

    還沒設定好、只想先測腳本本身（CSV 讀取／ffmpeg 抽格／精確率計算邏輯）：
        python run_precision_eval.py --labels labeled_set/template.csv --dry-run

--dry-run 模式不會呼叫 Gemini，會用假的固定 verdict 跑完整套流程，讓你在拿到金鑰前
就能確認腳本本身沒問題，金鑰一到位只要拿掉 --dry-run 就能跑真的。

標註集格式與建置方式見 labeled_set/README.md。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from engine.verify import VALID_VERDICTS, CONSENSUS_ROUNDS, call_gemini_verify_with_consensus, extract_frames

PRECISION_THRESHOLD = 0.85


def load_labels(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("ground_truth_verdict", "").strip():
                continue
            gt = row["ground_truth_verdict"].strip().upper()
            if gt not in VALID_VERDICTS:
                print(f"⚠️  跳過 {row.get('claim_id')}：ground_truth_verdict 值 '{gt}' 不合法，"
                      f"必須是 {VALID_VERDICTS} 其中之一")
                continue
            row["ground_truth_verdict"] = gt
            row["verifiable"] = row.get("verifiable", "TRUE").strip().upper() == "TRUE"
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="labeled_set/template.csv")
    parser.add_argument("--frames-dir", default="frames")
    parser.add_argument("--dry-run", action="store_true",
                         help="不呼叫 Gemini，只測試腳本本身的邏輯")
    parser.add_argument("--claim-ids", default=None,
                         help="只跑指定的 claim_id，逗號分隔，例如 c_0016,c_0024（除錯/迭代 prompt 用，省錢）")
    parser.add_argument("--force-consensus", action="store_true",
                         help="不管自報信心分數多高，每條 claim 都強制跑滿 3 次獨立呼叫——"
                              "MAPIE 校準需要真正的共識一致率，用這個 flag 才有乾淨資料可用")
    parser.add_argument("--export-calibration", default=None,
                         help="把每條 claim 的一致率/多數決結果/人工標註寫成 JSON，供 MAPIE 校準腳本讀取，"
                              "例如 --export-calibration labeled_set/calibration_data.json")
    args = parser.parse_args()

    rows = load_labels(args.labels)
    if args.claim_ids:
        wanted = {c.strip() for c in args.claim_ids.split(",")}
        rows = [r for r in rows if r["claim_id"] in wanted]
        missing = wanted - {r["claim_id"] for r in rows}
        if missing:
            print(f"⚠️  這些 claim_id 在標註集裡找不到：{missing}")
    if not rows:
        print("標註集裡沒有任何已填 ground_truth_verdict 的列，先照 labeled_set/README.md 標好資料。")
        sys.exit(1)

    print(f"讀到 {len(rows)} 條已標註的 claim{'（dry-run 模式）' if args.dry_run else ''}")

    results = []
    for row in rows:
        video_path = Path(args.labels).parent / row["video_path"]
        if not video_path.exists():
            print(f"⚠️  跳過 {row['claim_id']}：找不到影片檔 {video_path}")
            continue

        frame_out_dir = Path(args.frames_dir) / row["claim_id"]
        frames = extract_frames(video_path, frame_out_dir, row.get("temporal", "static"), row["claim_id"])
        if not frames:
            print(f"⚠️  跳過 {row['claim_id']}：抽格失敗")
            continue

        verdict_data = call_gemini_verify_with_consensus(
            row["claim_text"], frames, args.dry_run, force_consensus=args.force_consensus
        )
        system_verdict = verdict_data.get("verdict", "CANNOT_DETERMINE")

        results.append({
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "ground_truth": row["ground_truth_verdict"],
            "system_verdict": system_verdict,
            "observed": verdict_data.get("observed", ""),
            "checkable_components": verdict_data.get("checkable_components", []),
            "frame_observations": verdict_data.get("frame_observations", ""),
            "all_required_subjects_fully_visible": verdict_data.get("all_required_subjects_fully_visible"),
            "artifacts_affect_judgment": verdict_data.get("artifacts_affect_judgment"),
            "event_causal_order": verdict_data.get("event_causal_order"),
            "consensus_calls": verdict_data.get("consensus_calls"),
            "consensus_votes": verdict_data.get("consensus_votes"),
        })

    if not results:
        print("沒有任何一條 claim 完整跑完（可能是影片檔都不存在），先補齊 eval/frames/ 底下的素材。")
        sys.exit(1)

    report(results, args.dry_run)

    if args.export_calibration:
        export_calibration_data(results, args.export_calibration, args.force_consensus)


def export_calibration_data(results, out_path, force_consensus):
    """把每條 claim 的共識一致率、多數決結果、人工標註寫成 JSON，供 MAPIE 校準腳本讀取。

    對應 PROJECT_PLAN.md 第 9 節第 7 點：MAPIE 校準的分數是「共識一致率」（3 次獨立呼叫裡
    有幾次判斷一致），不是 Gemini 自報的 confidence。只有真的跑滿 `CONSENSUS_ROUNDS` 次的
    claim（force_consensus=True，或剛好自報信心低於門檻觸發了完整共識）才有意義的一致率；
    只打過 1 次的 claim，agreement_rate 標記為 None，不能拿 1/1=100% 冒充「完全一致」，
    那只是沒比對過、不是真的一致。
    """
    calibration_rows = []
    for r in results:
        votes = r.get("consensus_votes") or []
        calls = r.get("consensus_calls") or len(votes) or 1
        full_consensus = calls >= CONSENSUS_ROUNDS
        if votes and full_consensus:
            top_verdict, top_count = Counter(votes).most_common(1)[0]
            agreement_rate = top_count / len(votes)
        else:
            top_verdict = Counter(votes).most_common(1)[0][0] if votes else r["system_verdict"]
            agreement_rate = None
        calibration_rows.append({
            "claim_id": r["claim_id"],
            "ground_truth": r["ground_truth"],
            "majority_verdict": top_verdict,
            "agreement_rate": agreement_rate,
            "consensus_calls": calls,
            "votes": votes,
        })

    incomplete = sum(1 for row in calibration_rows if row["agreement_rate"] is None)
    if incomplete and not force_consensus:
        print(f"\n⚠️  {incomplete}/{len(calibration_rows)} 筆沒有真正的共識一致率（自報信心夠高沒觸發多次呼叫，"
              f"只打了 1 次），這批匯出的校準資料不完整——建議加 --force-consensus 重跑一次拿到乾淨資料。")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibration_rows, f, ensure_ascii=False, indent=2)
    print(f"\n校準資料已匯出：{out_path}（{len(calibration_rows)} 筆，"
          f"{len(calibration_rows) - incomplete} 筆有完整一致率資料）")


def report(results, dry_run):
    print("\n" + "=" * 60)
    print("驗證結果" + ("（dry-run，數字沒有意義，只是測試腳本）" if dry_run else ""))
    print("=" * 60)

    confusion = Counter((r["ground_truth"], r["system_verdict"]) for r in results)
    for r in results:
        mark = "✅" if r["ground_truth"] == r["system_verdict"] else "❌"
        consensus_note = f"  [共識 {r['consensus_calls']} 次：{r['consensus_votes']}]" if r.get("consensus_calls", 1) > 1 else ""
        print(f"{mark} [{r['claim_id']}] GT={r['ground_truth']:16s} "
              f"系統={r['system_verdict']:16s}  {r['claim_text']}{consensus_note}")
        if mark == "❌":
            print(f"      fully_visible={r.get('all_required_subjects_fully_visible')}  "
                  f"artifacts_affect_judgment={r.get('artifacts_affect_judgment')}  "
                  f"event_causal_order={r.get('event_causal_order')}")
            print(f"      checkable_components={r.get('checkable_components')}")
            print(f"      frame_observations: {r.get('frame_observations', '')[:400]}")

    predicted_mismatch = [r for r in results if r["system_verdict"] == "MISMATCH"]
    correct_mismatch = [r for r in predicted_mismatch if r["ground_truth"] == "MISMATCH"]
    ground_truth_match = [r for r in results if r["ground_truth"] == "MATCH"]
    false_alarms = [r for r in predicted_mismatch if r["ground_truth"] == "MATCH"]
    ground_truth_mismatch = [r for r in results if r["ground_truth"] == "MISMATCH"]
    missed_mismatch = [r for r in ground_truth_mismatch if r["system_verdict"] == "MATCH"]

    precision = len(correct_mismatch) / len(predicted_mismatch) if predicted_mismatch else None
    false_alarm_rate = len(false_alarms) / len(ground_truth_match) if ground_truth_match else None
    missed_mismatch_rate = len(missed_mismatch) / len(ground_truth_mismatch) if ground_truth_mismatch else None
    cannot_determine_rate = sum(1 for r in results if r["system_verdict"] == "CANNOT_DETERMINE") / len(results)

    print("\n" + "-" * 60)
    print(f"總計 {len(results)} 條 claim")
    print(f"系統判 MISMATCH 的 {len(predicted_mismatch)} 條裡，"
          f"{len(correct_mismatch)} 條跟人工標註一致")
    print(f"精確率（MISMATCH 判斷正確率）: "
          f"{precision:.1%}" if precision is not None else "精確率: 無法計算（系統沒判過任何 MISMATCH）")
    print(f"誤報率（人工標 MATCH 但系統判 MISMATCH）: "
          f"{false_alarm_rate:.1%}" if false_alarm_rate is not None else "誤報率: 無法計算（沒有 ground-truth MATCH 樣本）")
    print(f"漏抓率（人工標 MISMATCH 但系統判 MATCH，真的有問題卻被放行）: "
          + (f"{missed_mismatch_rate:.1%}（{len(missed_mismatch)}/{len(ground_truth_mismatch)}）"
             if missed_mismatch_rate is not None else "無法計算（沒有 ground-truth MISMATCH 樣本）"))
    if missed_mismatch:
        for r in missed_mismatch:
            print(f"    ⚠️  [{r['claim_id']}] {r['claim_text']}")
    print(f"CANNOT_DETERMINE 比例: {cannot_determine_rate:.1%}")

    print("\n判準（analysis.md 第 8 節實驗 1）：精確率 >= 85%（誤報率 <= 15%）")
    print("注意：精確率/誤報率只衡量『系統喊有問題時準不準』，不衡量『真的有問題時系統抓不抓得到』——")
    print("      漏抓率才是對應「AI 生成瑕疵被誤判放行」這個核心痛點的指標，兩者都要看，不能只看精確率通過就結案。")
    if dry_run:
        print("→ dry-run 模式，跳過通過/未通過判定，先確認上面的流程跑得動")
    elif precision is not None:
        passed = precision >= PRECISION_THRESHOLD
        print(f"→ {'✅ 通過' if passed else '❌ 未通過'}"
              f"（{'繼續往方案 A/B/C 推進' if passed else '先修 claim 抽取與比對 prompt，不要往下推進'}）")
    else:
        print("→ 樣本不足以判斷，擴充標註集或檢查是否所有 claim 都被判成 CANNOT_DETERMINE")


if __name__ == "__main__":
    main()
