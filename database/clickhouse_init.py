import os
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    host = os.environ.get("CLICKHOUSE_HOST", "localhost")
    port = int(os.environ.get("CLICKHOUSE_PORT", 8443))
    user = os.environ.get("CLICKHOUSE_USER", "default")
    password = os.environ.get("CLICKHOUSE_PASSWORD", "")
    secure = host != "localhost"
    _client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        secure=secure,
        connect_timeout=5,        # seconds, fail fast
        send_receive_timeout=10,  # seconds, interrupt slow queries
    )
    return _client

def init_tables():
    """Initializes tables in ClickHouse if they do not exist."""
    try:
        client = get_client()
        client.command("""
        CREATE TABLE IF NOT EXISTS guidance_events (
            ts              DateTime,
            session_id      String,
            event_type      String,          -- 'option_shown' | 'option_chosen' | 'freetext' | 'tweak_requested'
            axis            String,          -- 五個維度之一，tweak 事件則為推測出的軸
            option_label    String,
            option_fragment String,
            was_recommended UInt8,           -- 這個選項是否為 agent 推薦的那一個
            tweak_text      String,          -- 僅 tweak_requested 事件使用
            axes_asked      Array(String),   -- 該 session 實際問過哪些軸
            scene_summary   String,          -- 原始 prompt 的簡短摘要，供日後分類
            axis_confidence Float32 DEFAULT 1.0
        ) ENGINE = MergeTree()
        ORDER BY (axis, ts);
        """)
        client.command("""
            ALTER TABLE guidance_events
            ADD COLUMN IF NOT EXISTS axis_confidence Float32 DEFAULT 1.0
        """)
        print("[ClickHouse] guidance_events table and axis_confidence column ensured.")
    except Exception as e:
        print(f"[ClickHouse] init_tables failed (will retry or ignore if offline): {e}")

