from mcp.server.mcpserver import MCPServer
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.clickhouse_init import get_client

mcp = MCPServer("CineQA Memory")




STRONG_N = 5


@mcp.tool()
def get_axis_priority(scene_summary: str) -> str:
    """Return which creative dimensions users most often end up regretting (asking to tweak
afterwards) and how often the agent's recommendations get overridden, to decide
which dimension to ask about first."""
    try:
        client = get_client()
        regrets = client.query("""
            SELECT axis, count() AS regrets
            FROM guidance_events
            WHERE event_type = 'tweak_requested'
              AND NOT has(axes_asked, axis)
              AND axis_confidence >= 0.6
              AND axis != 'unknown'
            GROUP BY axis
            HAVING regrets >= 2
            ORDER BY regrets DESC
            LIMIT 5
            """).result_rows
        acceptance = client.query("""
            SELECT axis,
                   countIf(event_type = 'option_chosen' AND was_recommended = 1) AS accepted,
                   countIf(event_type = 'option_shown' AND was_recommended = 1) AS offered
            FROM guidance_events
            GROUP BY axis
            HAVING offered >= 5
            ORDER BY accepted / offered ASC
            LIMIT 5
            """).result_rows

        strong = [(a, n) for a, n in regrets if n >= STRONG_N]
        weak = [(a, n) for a, n in regrets if n < STRONG_N]

        out = "Historical guidance signals:\n"
        if strong:
            out += "\nSTRONG SIGNAL -- dimensions users frequently ask to fix afterwards:\n"
            for axis, n in strong:
                out += f"- {axis}: {n} regrets (n={n}, reliable)\n"
        if weak:
            out += "\nWEAK SIGNAL -- low sample size, use only as a tiebreaker, do not override your own judgement:\n"
            for axis, n in weak:
                out += f"- {axis}: {n} regrets (n={n}, NOT yet statistically meaningful)\n"
        if not strong and not weak and not acceptance:
            out += "No historical guidance data yet. Fall back to the default order: action first, then shot_framing_and_motion, then location, then lighting, then style.\n"
        if acceptance:
            out += "\nRecommendation acceptance rate (low rate means director advice gets overridden):\n"
            for axis, acc, off in acceptance:
                rate = acc / off if off else 0
                out += f"- {axis}: {acc}/{off} = {rate:.0%} (n={off})\n"
        return out
    except Exception as e:
        return f"No historical data available ({e}). Fall back to default order."


if __name__ == "__main__":
    mcp.run()
