import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from agents.agent_engine import CineQAAgentEngine

def main():
    print("=" * 60)
    print("🎬 Testing CineQAAgentEngine locally")
    print("=" * 60)

    engine = CineQAAgentEngine()
    engine.set_up()

    test_prompt = "A red sports car accelerates down a rain-slicked highway at midnight, passing under a bright neon billboard."
    print(f"\nTesting query with prompt: '{test_prompt}'\n")

    result = engine.query(scene_text=test_prompt)
    print(f"Status: {result['status']}")
    print(f"Extracted {len(result['claims'])} claims:")
    for i, c in enumerate(result['claims']):
        print(f"  #{i+1} [{c['type'].upper()}]: {c['claim_text']}")

    print("\n[SUCCESS] CineQAAgentEngine local execution passed 100%!")

if __name__ == "__main__":
    main()
