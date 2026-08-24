# 標註集負樣本 Prompt 清單（Day 2）

用 `scripts/test_veo_generation.py` 生成，每次只跑一組，跑完自己看結果再決定 `ground_truth_verdict`。
故意挑「生成模型已知容易出錯」的主張類型，讓負樣本命中率比亂猜高，但**不保證每次都會生成錯**——
Veo 也可能剛好生對，生對了就直接收進標註集當正樣本（`MATCH`），一樣有用，不是浪費。

## 怎麼用

1. 打開 `scripts/test_veo_generation.py`，把 `TEST_PROMPT` 換成下面某一組的 `prompt`
2. 跑 `python scripts/test_veo_generation.py`，看到預估花費按 Enter 確認
3. 下載影片，自己看畫面，對照該組列出的 `claim_text`，判斷 `MATCH` / `MISMATCH` / `CANNOT_DETERMINE`
4. 把結果填進 `labeled_set/template.csv`（`source_text` 填 `prompt` 原文，`claim_text`/`type`/`temporal` 照抄下面欄位，`video_path` 填你存檔的路徑）
5. 一組跑完再跑下一組，不要一次全跑——先看第一組結果，若失敗率太低（一直生對）可以再調 prompt 加大難度，避免浪費 credit 生一堆用不上的正樣本

---

## 1. count（已知弱點：精確計數）

**prompt：**
```
A wide static shot of a loading dock. Exactly four masked robbers stand in a straight line
facing a steel vault door, shoulder to shoulder, and no one else is in frame. Overhead
fluorescent light, no camera movement, 4 seconds.
```

**claim_text：** "There are exactly four people in frame, standing in a line, and no one else."
**type：** count　**temporal：** static

---

## 2. relative_size（已知弱點：物件間比例）

**prompt：**
```
A tiny grey mouse sits at the base of a massive oak tree trunk, the trunk at least ten times
the mouse's height, filling most of the frame behind it. Static wide shot, daylight, 4 seconds.
```

**claim_text：** "The tree trunk is at least ten times taller than the mouse."
**type：** relative_size　**temporal：** static

---

## 3. relative_position（已知弱點：左右空間關係一致性）

**prompt：**
```
A red bicycle leans against a brick wall on the left side of the frame. A blue bicycle stands
upright on the right side of the frame, roughly six meters away from the red one. Static wide
shot, daylight, 4 seconds.
```

**claim_text：** "The red bicycle is on the left side of frame, the blue bicycle is on the right side."
**type：** relative_position　**temporal：** static

---

## 4. action（已知弱點：物理破壞動作，對應 Hell Grind「門不會破」已知失敗模式）

**prompt：**
```
A heavy wooden door is kicked open with force, splinters flying outward from the impact point,
the door swinging fully open and slamming against the interior wall. Handheld camera, dim
hallway lighting, 4 seconds.
```

**claim_text：** "The door breaks open, with visible splinters flying from the impact."
**type：** action　**temporal：** sequential

---

## 5. color（已知弱點：整段時間內顏色一致性）

**prompt：**
```
A woman wearing a bright yellow raincoat runs down a flight of concrete stairs. Her raincoat
stays bright yellow throughout the entire shot, with no color shift or flicker as she moves
through changing shadow and light. Tracking shot, 4 seconds.
```

**claim_text：** "The woman's raincoat is bright yellow for the entire duration of the shot, with no color change."
**type：** color　**temporal：** sequential

---

## 第二輪：補 MATCH 案例（2026-08-19）

第一輪 5 組刻意挑容易出錯的場景，跑完 MATCH 只有 3 筆（`c_0007`/`c_0009`/`c_0012`），比例偏低。
這輪反過來，刻意挑「構圖簡單、模型通常做得到」的場景，目標是拿到更多明顯正確的案例，
同時補上 `state` 類型還沒有 Veo 自己生成的真實資料（`c_0014`-`c_0017` 的 state 資料來自 T2V-CompBench，不是 Veo）。

### 6. state（簡單版，目標拿到 MATCH）

**prompt：**
```
A man stands still in an empty warehouse, wearing a bright orange safety helmet, hands empty
at his sides. Static wide shot, daylight, 4 seconds.
```

**claim_text：** "The man is wearing an orange safety helmet."
**type：** state　**temporal：** static

### 7. direction（簡單版，目標拿到 MATCH——目前 direction 只有 MISMATCH 沒有 MATCH）

**prompt：**
```
A red car drives across the frame from right to left on a straight empty highway. Static wide
shot, daylight, 4 seconds.
```

**claim_text：** "The car moves from right to left."
**type：** direction　**temporal：** sequential

### 8. count（簡單版，小數字，目標拿到 MATCH——目前 count 只有 MISMATCH 沒有 MATCH）

**prompt：**
```
A static wide shot of exactly three red apples arranged in a row on a wooden table, and
nothing else on the table. Daylight, 4 seconds.
```

**claim_text：** "There are exactly three apples on the table, and nothing else."
**type：** count　**temporal：** static

## 第三輪：relative_size 補中等比例案例（2026-08-21，Day 4）

現有 4 筆 relative_size（`c_0005`/`c_0006` 樹幹/老鼠 10 倍、`c_0039` 沙灘球/貓 5 倍、`c_0040` 貨車/信箱 8 倍）比例都偏極端，且都遇到透視/構圖爭議（見 `PROJECT_PLAN.md` 第 12 節風險表、`collection_report.md` 第 9 節 `c_0040` 案例）。這輪換中等比例（2-4 倍，人眼較難一眼判斷，更貼近實際校準需要的邊界案例），並明確要求「同景深」降低透視干擾，其中一組改比直徑而非高度，增加維度多樣性。`relative_size` 在 T2V-CompBench 免費資料集裡沒有對應分類（見 `collection_report.md` 第 10 節），只能靠 Veo 生成補。

### 9. relative_size（中等比例，同景深控制）

**prompt：**
```
A golden retriever standing beside a fire hydrant on a sidewalk, the dog at least twice as
tall as the hydrant, both positioned at the same distance from the camera. Static wide shot,
daylight, 4 seconds.
```

**claim_text：** "The dog is at least twice as tall as the fire hydrant."
**type：** relative_size　**temporal：** static

### 10. relative_size（直徑比較，非高度）

**prompt：**
```
A basketball resting next to a tennis ball on a wooden floor, the basketball at least three
times the diameter of the tennis ball, both fully visible and at the same distance from the
camera. Static close-up shot, daylight, 4 seconds.
```

**claim_text：** "The basketball is at least three times the diameter of the tennis ball."
**type：** relative_size　**temporal：** static

### 11. relative_size（室內場景，中等比例）

**prompt：**
```
A grandfather clock standing next to a small wooden stool, the clock at least four times
taller than the stool, both fully visible from base to top, at the same distance from the
camera. Static wide shot, indoor lighting, 4 seconds.
```

**claim_text：** "The grandfather clock is at least four times taller than the stool."
**type：** relative_size　**temporal：** static

## 涵蓋度檢查

跑完這 5 組，對照 `eval/labeled_set/README.md` 的維度表：
- 七類裡已涵蓋 count / relative_size / relative_position / action / color 五類，`direction`、`state` 還沒覆蓋，之後可以再補
- static 3 組、sequential 2 組，符合「static/sequential 各半」的建議
- 全部都是「刻意設計成容易錯」的案例，記得再混一些「明顯正確」「邊界模糊」的案例（例如直接照抄劇本正常描述去生成），不要標註集全部都是刁鑽案例，不然精確率數字會失真
