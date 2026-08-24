"""
Claim 抽取 prompt 測試（Day 1 任務：拿真實腳本片段測分類法夠不夠用）

對應 technical_spec_verification_engine.md 第 5 節步驟 2、第 4 節 Claim schema。

用法：
    python test_claim_extraction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from engine.claims import extract_claims

TEST_SCENES = [
    {
        "name": "Hell Grind - 訓練室場景",
        "text": """EXACT 3 CHARACTERS — NO DUPLICATES: ROCO, JAX, REIN. Underground base, training hall, day. ROCO has
been drilling alone for hours; JAX and REIN come in late with food and find the room wrecked. One
continuous 12-second shot, no cuts.

ROCO — bare-chested, the crystal sheathing his right arm from wrist to shoulder, blood dried under
his nose. JAX — carrying two food trays. REIN — tablet in her left hand, screen alive.

The round training mat sits at the center of the hall under one hard overhead light. The door is in
the far wall at frame-LEFT, about eight metres from the mat. Five smashed mannequins lie scattered
at CENTER-RIGHT, one still rocking on its base. A bench with two trays stands at frame-RIGHT.

ROCO planted at the center of the mat, torso angled to frame-LEFT, gaze down on the broken
mannequins; the open door at frame-LEFT with JAX and REIN just inside it, trays in hand, two metres
apart.

Exactly three people in the hall, and no one else. Exactly ONE crystal arm, on ROCO's right arm,
wrist to shoulder — never on the left, never spreading past the shoulder. FIVE smashed mannequins,
never re-rendered as intact, never multiplied. Two trays, never more.""",
    },
    {
        "name": "Hell Grind - 走廊對話場景",
        "text": """JAX and REIN walk the corridor toward the lens, in step. JAX talks with his eyes up on
the ceiling lights, one hand patting his stomach; REIN's thumb keeps scrolling the tablet, her pace
unchanged, she never looks up at him.

The distant THUD from the training room lands: REIN's thumb STOPS on the glass, and only
then her head turns to the door — the interrupted work is the accent of the beat. JAX's grin drops
half a second later.""",
    },
    {
        "name": "補測 color/relative_size 場景（自寫）",
        "text": """Interior, cargo bay, night. A single red emergency light pulses overhead, washing the
metal walls in dull crimson. KESTREL stands at the center wearing a faded green flight jacket, its
left sleeve torn at the elbow. Beside her, a small silver drone hovers, no bigger than a dinner
plate, dwarfed by the massive black shipping crate looming behind it — the crate alone is easily
three times the drone's height. Two yellow warning stripes run along the floor, framing the crate on
both sides. Near the far wall, a second crate — identical in shape but painted plain grey, roughly
the same size as the black one — sits unopened.""",
    },
]


def main():
    for scene in TEST_SCENES:
        print("=" * 70)
        print(f"場景：{scene['name']}")
        print("=" * 70)
        claims = extract_claims(scene["text"])
        print(f"抽出 {len(claims)} 條 claim\n")
        for i, c in enumerate(claims, 1):
            print(f"{i}. [{c['type']:20s}] {c['claim_text']}")
            print(f"   verifiable={c['verifiable']}  temporal={c['temporal']}  entities={c['entities']}")
        print()


if __name__ == "__main__":
    main()
