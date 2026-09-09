import os
from dotenv import load_dotenv
import clickhouse_connect

from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

_client = None

def get_client(force_new: bool = False):
    global _client
    if _client is not None and not force_new:
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
        connect_timeout=15,       # allow cloud instance wakeup
        send_receive_timeout=30,  # allow cold-start queries to complete
    )
    return _client

def reset_client():
    global _client
    _client = None

def init_tables():
    """Initializes the two core ClickHouse tables: guidance_events and verification_ledger."""
    try:
        client = get_client()

        # 1. guidance_events: Tracks director interview choices & user tweak regrets (Adaptive Memory via MCP)
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

        # 2. verification_ledger: Immutable audit ledger of all claim evaluations per take
        client.command("""
        CREATE TABLE IF NOT EXISTS verification_ledger (
            ts                  DateTime,
            scene_id            String,
            take_num            UInt8,
            claim_id            String,
            claim_type          String,
            tier                String,
            temporal            String,
            claim_text          String,
            verdict             String,
            confidence          Float32,
            conformal_set_size  Float32,
            is_autonomous       UInt8,
            consensus_votes     String,
            observed            String
        ) ENGINE = MergeTree()
        ORDER BY (scene_id, ts);
        """)

        print("[ClickHouse] Core tables ensured: guidance_events, verification_ledger.")
    except Exception as e:
        print(f"[ClickHouse] init_tables failed (will retry or ignore if offline): {e}")

