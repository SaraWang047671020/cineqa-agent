import json
from datetime import datetime
from database.clickhouse_init import get_client

def log_verification_ledger(scene_id, take_num, ledger):
    try:
        client = get_client()
        
        rows = []
        for entry in ledger:
            rows.append([
                datetime.now(),
                str(scene_id),
                int(take_num),
                str(entry.get("claim_id", "")),
                str(entry.get("type", "")),
                str(entry.get("tier", "")),
                str(entry.get("temporal", "")),
                str(entry.get("claim_text", "")),
                str(entry.get("verdict", "")),
                float(entry.get("confidence", 0.0)),
                float(entry.get("conformal_set_size", 0.0)),
                int(entry.get("conformal_autonomous", 0)),
                json.dumps(entry.get("consensus_votes", [])),
                str(entry.get("observed", ""))[:2000]
            ])
            
        client.insert('verification_ledger', rows, column_names=[
            'ts', 'scene_id', 'take_num', 'claim_id', 'claim_type', 'tier', 
            'temporal', 'claim_text', 'verdict', 'confidence', 'conformal_set_size', 
            'is_autonomous', 'consensus_votes', 'observed'
        ])
        print(f"[ClickHouse] Inserted {len(rows)} claims into verification_ledger.")
    except Exception as e:
        print(f"[ClickHouse] Failed to insert ledger: {e}")

def log_remediation_history(claim_type, claim_text, failure_observed, refined_positive_prompt, negative_prompt, take_num_before, take_num_after, outcome_verdict):
    try:
        client = get_client()
        row = [
            datetime.now(),
            str(claim_type),
            str(claim_text),
            str(failure_observed)[:2000],
            str(refined_positive_prompt),
            str(negative_prompt),
            int(take_num_before),
            int(take_num_after),
            str(outcome_verdict)
        ]
        
        client.insert('remediation_history', [row], column_names=[
            'ts', 'claim_type', 'claim_text', 'failure_observed', 
            'refined_positive_prompt', 'negative_prompt', 
            'take_num_before', 'take_num_after', 'outcome_verdict'
        ])
        print(f"[ClickHouse] Logged remediation history for {claim_type} -> {outcome_verdict}.")
    except Exception as e:
        print(f"[ClickHouse] Failed to insert remediation: {e}")

def log_remediation_history_batch(prev_ledger, curr_ledger, plan, take_num_before, take_num_after):
    """Record one row per claim that FAILED in the previous take, labelled with whether
    the remediation actually fixed it in the current take."""
    try:
        from database.clickhouse_init import get_client
        from datetime import datetime
        client = get_client()

        # 把前一輪失敗的 claim 建索引
        prev_failed = {}
        for e in (prev_ledger or []):
            if e.get("verdict") == "MISMATCH":
                key = e.get("claim_id") or e.get("claim_text", "")
                prev_failed[key] = e

        if not prev_failed:
            return  # 前一輪沒有失敗，就沒有修正案例可記錄

        plan = plan or {}
        surgeries = plan.get("targeted_token_surgery", []) or []
        negative_prompt = plan.get("negative_prompt", "")
        fallback_fix = plan.get("refined_positive_prompt", "")

        rows = []
        for e in (curr_ledger or []):
            key = e.get("claim_id") or e.get("claim_text", "")
            prev = prev_failed.get(key)
            if prev is None:
                continue  # 這條上一輪沒失敗，沒有做過修正，不該進記憶庫

            claim_type = e.get("type", "unknown")

            # 優先取針對這個 claim type 的具體 token surgery
            fix_text = fallback_fix
            for s in surgeries:
                if s.get("failure_claim_type") == claim_type:
                    fix_text = (
                        f"{s.get('original_phrase', '')} -> {s.get('repaired_phrase', '')}"
                        f" | rationale: {s.get('rationale', '')}"
                    )
                    break

            rows.append([
                datetime.now(),
                str(claim_type),
                str(e.get("claim_text", "")),
                str(prev.get("observed", ""))[:2000],   # ← 真正的失敗原因
                str(fix_text)[:4000],                   # ← 針對性的修法
                str(negative_prompt),
                int(take_num_before),
                int(take_num_after),
                str(e.get("verdict", "UNKNOWN")),       # ← 這次修好了沒
            ])

        if rows:
            client.insert('remediation_history', rows, column_names=[
                'ts', 'claim_type', 'claim_text', 'failure_observed',
                'refined_positive_prompt', 'negative_prompt',
                'take_num_before', 'take_num_after', 'outcome_verdict'
            ])
            print(f"[ClickHouse] Logged {len(rows)} remediation cases "
                  f"(take {take_num_before} -> {take_num_after}).")
    except Exception as e:
        print(f"[ClickHouse] Failed to batch insert remediation: {e}")


def log_tweak_suggestions(suggestions, session_id, take_num):
    """
    Records newly generated tweak suggestions to ClickHouse (if available).
    """
    try:
        from database.clickhouse_init import get_client
        client = get_client()

        # Ensure table exists
        client.command("""
        CREATE TABLE IF NOT EXISTS tweak_suggestions (
            ts DateTime,
            suggestion_id String,
            session_id String,
            take_num UInt32,
            issue String,
            tweak_instruction String,
            severity String,
            was_pasted UInt8,
            was_edited_before_send UInt8,
            led_to_new_take UInt8
        ) ENGINE = MergeTree()
        ORDER BY (session_id, take_num, ts);
        """)

        rows = []
        for s in suggestions:
            rows.append([
                datetime.now(),
                str(s.get("suggestion_id", "")),
                str(session_id),
                int(take_num),
                str(s.get("issue", "")),
                str(s.get("tweak_instruction", "")),
                str(s.get("severity", "medium")),
                0, 0, 0
            ])

        if rows:
            client.insert('tweak_suggestions', rows, column_names=[
                'ts', 'suggestion_id', 'session_id', 'take_num',
                'issue', 'tweak_instruction', 'severity',
                'was_pasted', 'was_edited_before_send', 'led_to_new_take'
            ])
            print(f"[ClickHouse] Logged {len(rows)} tweak suggestions for take {take_num}.")
    except Exception as e:
        print(f"[ClickHouse] Skipped logging tweak suggestions: {e}")


def update_tweak_suggestion(suggestion_id, was_pasted=None, was_edited_before_send=None, led_to_new_take=None):
    """
    Updates the telemetry status of a tweak suggestion in ClickHouse.
    """
    if not suggestion_id:
        return
    try:
        from database.clickhouse_init import get_client
        client = get_client()
        
        updates = []
        if was_pasted is not None:
            updates.append(f"was_pasted = {1 if was_pasted else 0}")
        if was_edited_before_send is not None:
            updates.append(f"was_edited_before_send = {1 if was_edited_before_send else 0}")
        if led_to_new_take is not None:
            updates.append(f"led_to_new_take = {1 if led_to_new_take else 0}")

        if updates:
            sql = f"ALTER TABLE tweak_suggestions UPDATE {', '.join(updates)} WHERE suggestion_id = '{suggestion_id}'"
            client.command(sql)
            print(f"[ClickHouse] Updated suggestion {suggestion_id}: {updates}")
    except Exception as e:
        print(f"[ClickHouse] Skipped updating tweak suggestion: {e}")

def log_guidance_event(
    session_id: str,
    event_type: str,
    axis: str = "",
    option_label: str = "",
    option_fragment: str = "",
    was_recommended: int = 0,
    tweak_text: str = "",
    axes_asked: list[str] = None,
    scene_summary: str = "",
    axis_confidence: float = 1.0
):
    """Logs an event into ClickHouse guidance_events table."""
    try:
        from database.clickhouse_init import get_client
        client = get_client()
        row = [
            datetime.now(),
            str(session_id or ""),
            str(event_type or ""),
            str(axis or ""),
            str(option_label or ""),
            str(option_fragment or ""),
            int(1 if was_recommended else 0),
            str(tweak_text or ""),
            list(axes_asked or []),
            str(scene_summary or ""),
            float(axis_confidence)
        ]
        client.insert('guidance_events', [row], column_names=[
            'ts', 'session_id', 'event_type', 'axis',
            'option_label', 'option_fragment', 'was_recommended',
            'tweak_text', 'axes_asked', 'scene_summary', 'axis_confidence'
        ])
        print(f"[ClickHouse] Logged guidance_event: {event_type} (axis: {axis}, conf: {axis_confidence})")
    except Exception as e:
        print(f"[ClickHouse] Skipped guidance_event log: {e}")


