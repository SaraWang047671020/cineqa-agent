from google.cloud import bigquery
import time
from datetime import datetime, timedelta
import random

client = bigquery.Client()
table_id = f"{client.project}.cineqa_telemetry.verification_logs"

# Drop table to refresh schema
client.delete_table(table_id, not_found_ok=True)
print("Table dropped.")

# Recreate table with new schema
schema = [
    bigquery.SchemaField("timestamp", "TIMESTAMP"),
    bigquery.SchemaField("take_num", "INTEGER"),
    bigquery.SchemaField("video_engine", "STRING"),
    bigquery.SchemaField("pass_rate", "FLOAT"),
    bigquery.SchemaField("original_prompt", "STRING"),
    bigquery.SchemaField("healed_prompt", "STRING"),
    bigquery.SchemaField("verdicts_json", "STRING"),
    bigquery.SchemaField("avg_set_size", "FLOAT"),
    bigquery.SchemaField("failed_defect_types", "STRING"),
    bigquery.SchemaField("time_saved_s", "INTEGER"),
    bigquery.SchemaField("dollars_saved", "FLOAT")
]
table = bigquery.Table(table_id, schema=schema)
client.create_table(table)
print("Table created.")

time.sleep(2) # wait for creation to propagate

# Mock data generation
SCENES = [
    ("A man walking a dog in the park", ["Action", "State"]),
    ("A coffee cup shattering on the floor", ["Topology", "Action"]),
    ("A red car drifting around a tight corner", ["Direction", "Action"]),
    ("A robot serving a glowing blue drink", ["State", "Direction"]),
    ("A bowling ball hitting 10 pins", ["Action", "Topology"]),
    ("A rocket launching into the night sky", ["Direction", "Action"]),
    ("Water freezing into ice cubes", ["Topology", "State"]),
    ("A dancer spinning exactly 3 times", ["State", "Action"]),
    ("A book falling off a wooden desk", ["Action", "Direction"]),
    ("A cat jumping over a high fence", ["Action", "Direction"]),
    ("A glass window cracking but not breaking", ["Topology", "State"]),
    ("Two swords clashing with sparks", ["Action", "State"]),
    ("A balloon popping into confetti", ["Topology", "Action"]),
    ("A character walking backwards through a door", ["Direction", "Action"]),
    ("A clock spinning backwards rapidly", ["Direction", "State"])
]

rows_to_insert = []
base_time = datetime.utcnow() - timedelta(days=2)

for i, (prompt, defect_risks) in enumerate(SCENES):
    scene_time = base_time + timedelta(hours=i*3)
    
    pass_rate_1 = random.choice([25.0, 50.0, 66.6, 75.0])
    failed_defects_1 = ",".join(random.sample(defect_risks, k=random.randint(1, len(defect_risks)))) if pass_rate_1 < 100 else ""
    set_size_1 = random.choice([1.0, 1.5, 2.0, 3.0])
    
    rows_to_insert.append({
        "timestamp": scene_time.isoformat(),
        "take_num": 1,
        "video_engine": "veo-3.1",
        "pass_rate": pass_rate_1,
        "original_prompt": prompt,
        "healed_prompt": "",
        "verdicts_json": "[]",
        "avg_set_size": set_size_1,
        "failed_defect_types": failed_defects_1,
        "time_saved_s": 0,
        "dollars_saved": 0.0
    })
    
    if pass_rate_1 < 100:
        scene_time += timedelta(minutes=random.randint(1, 5))
        pass_rate_2 = pass_rate_1 + random.choice([25.0, 33.3, 50.0])
        if pass_rate_2 > 100: pass_rate_2 = 100.0
        failed_defects_2 = random.choice(defect_risks) if pass_rate_2 < 100 else ""
        set_size_2 = random.choice([1.0, 1.0, 1.25])
        
        rows_to_insert.append({
            "timestamp": scene_time.isoformat(),
            "take_num": 2,
            "video_engine": "veo-3.1",
            "pass_rate": pass_rate_2,
            "original_prompt": prompt,
            "healed_prompt": f"[HEALED] {prompt} MUST explicitly...",
            "verdicts_json": "[]",
            "avg_set_size": set_size_2,
            "failed_defect_types": failed_defects_2,
            "time_saved_s": 45,
            "dollars_saved": 0.05
        })
        
        if pass_rate_2 < 100 and random.random() > 0.3:
            scene_time += timedelta(minutes=random.randint(1, 5))
            rows_to_insert.append({
                "timestamp": scene_time.isoformat(),
                "take_num": 3,
                "video_engine": "veo-3.1",
                "pass_rate": 100.0,
                "original_prompt": prompt,
                "healed_prompt": f"[HEALED] [PACING] {prompt}",
                "verdicts_json": "[]",
                "avg_set_size": 1.0,
                "failed_defect_types": "",
                "time_saved_s": 45,
                "dollars_saved": 0.05
            })

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors:
    print(f"Errors: {errors}")
else:
    print(f"Successfully seeded {len(rows_to_insert)} records!")
