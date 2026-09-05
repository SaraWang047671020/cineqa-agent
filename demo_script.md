# 🎬 CineQA Agent - 駭客松 Demo 劇本與操作指南

這份文件包含您在進行 Live Demo 或錄製 Demo 影片時的「實機操作劇本 (Demo Flow)」以及完整的「英文旁白腳本 (Voiceover Script)」。

---

## 💻 第一部分：Live Demo 實機操作劇本 (Demo Flow)

這是一套為您量身打造的測試案例，能完美觸發我們剛剛修正的所有高階技術亮點（運鏡尾幀約束、嚴格動態驗證、MCP 記憶與 Omni 修補）。

### 📍 步驟 1：輸入神級 Prompt
在 UI 上輸入以下這段經過設計的 Prompt：

> **Base Prompt:** 
> "The camera pans rightward along the alley to track a running subject. A masked traveler sprints from left to right through a neon-lit cyberpunk alley. While sprinting, the traveler draws a glowing blue energy katana from its scabbard. The blade emits a bright blue laser light that illuminates the surrounding brick walls."

### 📍 步驟 2：展示 Claim Extraction (4-Tier)
1. 點擊 **"Step 1: Extract 4-Tier Critical Path Claims"**。
2. **💡 展演亮點**：向評審展示系統如何萃取出 `action` (draws a katana)、`direction` (left to right)、`color` (glowing blue light) 等獨立的驗證節點。
3. **💡 展演亮點**：切換到 Last Frame 生成區塊，指給評審看 `[CAMERA/FRAMING DIRECTIVE]` 的系統指令，說明這保證了尾幀一定會呈現 "Pan rightward" 後的新視角。

### 📍 步驟 3：生成 Take 1 與觸發驗證 (Inspector)
1. 生成 Take 1（通常 Veo 在第一輪會把光劍畫成普通的金屬劍，而且可能只有拿著劍跑步，沒有「拔」的動作）。
2. 啟動 Auto-QA 驗證。
3. **💡 展演亮點**：展示進度條與紅色的 `❌ MISMATCH`。
4. 打開錯誤原因，展示 Gemini 給出的嚴格判定：「雖然他拿著劍，但**缺乏拔劍的動力學過程 (Kinematics)**」或「**這是一把普通金屬劍，不是發光的光劍**」。順便強調這背後是 Split-Conformal (LAC) 的共形預測演算法在防堵 AI 幻覺。

### 📍 步驟 4：啟動 Auto-Heal (MCP + Omni V2V)
1. 讓系統自動進入 Auto-Heal 流程。
2. **💡 展演亮點**：在終端機或 UI 上展示系統呼叫了 `search_clickhouse_memory` 進行 Vector Search，從歷史記憶中找出「光劍」的正確修復 Prompt。
3. **💡 展演亮點**：展示系統生成了時空遮罩 (Mask)，並透過繞過 SDK 的 REST API 原生呼叫，將影片與遮罩同時傳給 **Omni Interactions API**。
4. 展示 Take 2 完美修復的成果！

---

## 🎙️ 第二部分：3 分鐘英文旁白腳本 (Hardcore Tech Version)

**建議時間**：約 3 分鐘 (180 秒)
**旁白語氣**：技術研討會風格，語速緊湊，專注於架構設計與演算法決策

### 🕒 [0:00 - 0:45] 模組 1：語意解構與首尾幀雙向約束 (Decomposition & Frame Constraints)
"Current AI video generation is a black box. CineQA transforms it into a deterministic pipeline. It starts with the Decomposer Agent. We don't just pass raw text to the generator; we parse it into a JSON-based 4-Tier Critical Path, extracting causal actions, spatial geometry, and temporal states while strictly enforcing a saliency budget. 

To guarantee temporal consistency, we use a Keyframe-Driven approach. We generate the First Frame as the starting state. For the Last Frame, the system dynamically injects the extracted spatial claims and, crucially, a 'Camera/Framing Directive'. This forces the Image model to render the exact ending perspective—such as a shifted FOV for a 'Pan Right'—preventing the video model from ignoring camera constraints."

### 🕒 [0:45 - 1:15] 模組 2：動態路由與精確抽幀 (Routing & Exact Frame Extraction)
"Based on the density of the claims, our Router dynamically determines the optimal generation strategy—directing the payload to Vertex AI's Veo for Image-to-Video tasks. 

Once the video is generated, the verification phase begins. Standard frame extraction often misses the exact final state due to rapid keyframe seeking. We engineered a precise slow-seek FFmpeg pipeline that forcefully clears cache directories and extracts frames linearly up to exactly `duration minus 0.01 seconds`, ensuring the ultimate state of the physics is captured without EOF corruption."

### 🕒 [1:15 - 2:10] 模組 3：共形預測與動力學稽核 (Conformal Verification & Kinematics)
"Now, the Inspector Agent takes over using Gemini 1.5 Pro. To prevent AI visual hallucinations, we embedded strict 'Action vs. State' and 'Anti-Morphing' topology rules into the core system prompt. The model is forced to differentiate between a subject simply holding a sword, versus the kinematic execution of drawing it.

But LLMs can be uncertain. That’s why we implemented Split-Conformal Prediction via LAC. We run parallel consensus threads with varying temperatures. If the resulting prediction set size is greater than one, the system autonomously flags the verdict as ambiguous. It catches topological drift and morphing that human eyes might miss."

### 🕒 [2:10 - 2:50] 模組 4：MCP 向量記憶與 Omni 局部修復 (MCP Memory & Omni V2V Healing)
"When a failure is detected, we don't discard the video. The Remediator Agent queries our FastMCP server, which uses `text-embedding-004` to perform a cosine-distance vector search in our ClickHouse database, retrieving historical, proven fix strategies.

Armed with the correct negative prompts, the system generates a spatio-temporal mask and bypasses standard SDKs to directly interface with the Gemini Omni Interactions REST API. By passing the source video and mask natively, Omni performs targeted Video-to-Video inpainting—healing only the defective physics, without regenerating the entire timeline."

### 🕒 [2:50 - 3:00] 總結 (Conclusion)
"From prompt decomposition and exact keyframe constraints, to mathematically-backed conformal verification and MCP-driven Omni healing. CineQA is a fully autonomous, self-correcting engineering pipeline for the future of deterministic AI video. Thank you."
