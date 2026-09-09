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
                str(entry.get("claim_type") or entry.get("type") or ""),
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
    import time
    for attempt in range(2):
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
            return
        except Exception as e:
            from database.clickhouse_init import reset_client
            reset_client()
            if attempt == 0:
                print(f"[ClickHouse] guidance_event retrying after connection issue: {e}")
                time.sleep(1)
                continue
            print(f"[ClickHouse] Skipped guidance_event log: {e}")


