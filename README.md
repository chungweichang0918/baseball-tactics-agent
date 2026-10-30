# 多模態棒球戰術決策 Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chungweichang0918/baseball-tactics-agent/blob/main/notebooks/colab_demo.ipynb)

上傳 10–15 秒的棒球短影片並填入賽況。Agent 先用 YOLO 分析投手或打者的動作（投手另外追蹤球軌跡），再結合 RE24 得分期望值與 Gemini 推理，產出戰術建議報告。完整設計見上一層目錄的 `系統規劃書.md`。

## Diamond Studio：整合投打與守備工作台

網站部署：接入既有 `cwchang.me` 的反向代理、服務與內嵌範本見 [deploy/README.md](deploy/README.md)。目前是部署準備，尚未上傳到遠端主機。

啟動方式不變：`uv run python scripts/app.py`，開啟 `http://127.0.0.1:7860`。
已建立本機虛擬環境時，也可直接執行 `.venv/Scripts/python.exe scripts/app.py`（Windows）。

1. 上傳 MP4、MOV 或 AVI。介面會顯示影片播放、長度、幀率與可能的切鏡。
2. 在「片段選取與鏡頭切換」選取連續鏡頭，或設定開始／結束秒數。60 秒內預設分析整段影片，超過時先選前 60 秒。建議選一個完整 play；多鏡頭重播不會自動去重。
3. 「完整投打」模式依序執行投手與打者分支，分別保存指標、關鍵幀、曲線、報告與審查紀錄；也可只選投球或打擊。
4. 多人轉播建議先展開「球員設定與人物辨識」，辨識人物並依框線指定角色。自動選擇目前依人物大小、出現率與手腕活動量評分，**不是 MLB 角色分類器**；投打選到相同人物時，總覽和報告會提示，不能把這種結果當成完整投打重建。
5. 「動作時間軸」顯示原影片時間，點選事件可查看對應原始幀；支援 `playback_position` 的 Gradio 版本也會同步播放器位置。切換上傳或分析片段會清除舊結果與人物選擇。
6. 「守備模擬」提供九名守備員、三種假設球路、三種配置和 0–100% 回放滑桿。球與接球員位置依預設路線移動，壘包與 RE24 隨使用者賽況更新。

**能力範圍：** 投打指標是骨架估計；打擊時間以手腕速度峰值作代理，並未偵測球棒碰球。守備戰術板是手動情境模擬，尚未實作從轉播還原九人位置、真實落點、跑壘或 3D 球場。球速僅在可信球軌跡、固定側拍及換算條件成立時顯示。未知值顯示「—」，不補造數據。局數、球數、比分和壘況目前由使用者設定。

輸出保存在 `runs/studio/<session>/pitcher/` 與 `batter/`。下載區另提供 `session.json`，記錄來源檔名、分析區間、原片時間軸及人物識別限制。分析片段快取於 `runs/clips/`，不改動原影片。

介面程式：`src/bbagent/ui/workspace.py`；呈現與片段工具：`studio.py`；樣式：`studio.css`。保留原 `scripts/app.py` 與 Colab 的啟動入口，支援 Gradio 5/6 的參數位置差異。

## 環境建置（macOS，Apple Silicon）

需要 [uv](https://docs.astral.sh/uv/)（`brew install uv`）。

```bash
uv sync --extra dev
cp .env.example .env              # 填入 GEMINI_API_KEY
uv run python scripts/check_env.py --list-models --ping-gemini
uv run python scripts/download_models.py
```

`check_env.py` 會確認：

- Python 是 arm64 版本
- MPS 可用
- 套件都已安裝
- 金鑰有效
- `config.yaml` 中的 Gemini 型號目前可用

不可用的型號請依它列出的清單更新 `config.yaml`。

## 階段 0：基準測試與標註

1. 把測試影片放進 `data/samples/`，命名與規格見 [data/samples/README.md](data/samples/README.md)。
2. 跑 MPS 與 CPU 的速度和一致性比較：

   ```bash
   uv run python scripts/benchmark_mps.py data/samples/p_side_01.mp4
   ```

   報告（`report.md`、`report.json`）和偵測示意圖會輸出到 `runs/benchmark/<時間>/`。若結果判定不一致，表示 MPS 的輸出和 CPU 有落差，請先回報再繼續。

3. 標註每段影片的動作階段幀號：

   ```bash
   uv run python scripts/annotate_samples.py data/samples/p_side_01.mp4 --subject pitcher --view side
   ```

**完成標準**：

- `check_env.py` 全部通過。
- 基準測試報告顯示 mps 與 cpu 一致。
- 記下 mps 的 FPS，供後續估算整段影片的處理時間。
- `data/samples/labels/` 有 6–8 個標註檔。

## 階段 1：資料結構與 RE24

- `src/bbagent/schemas.py`：賽況（GameState）、計畫（Plan）、工具結果（ToolResult）、事件（AgentEvent）。
- `data/re24.json`：RE24 數值表與各戰術的限制條件。數值取自 [Tom Tango 的 MLB 2010–2015 表格](https://tangotiger.net/re24.html)（資料來自 Retrosheet），已逐項核對。
- `src/bbagent/tools/re24.py`：查表（`re24_lookup`）、戰術評估（`re24_evaluate`）、損益平衡成功率、戰術可行性檢查。

執行測試：

```bash
uv run pytest
```

## 階段 2：投手視覺流程

```bash
uv run python scripts/analyze_video.py data/samples/p_side_01.mp4
```

流程：影片探測（含場景切換）→ Pass 1 粗掃（YOLO pose + ByteTrack，結果快取在 `cache/`）→ 合併斷開的追蹤 → 選出投手 → 6 個動作階段 → 投手指標。`data/samples/labels/` 有同名標註檔時，會自動比對每個階段的誤差。

輸出到 `runs/analyze/<影片檔名>/`：

- `result.json`：完整結果
- `signals.png`：動作能量、投球手腕速度、前導腳膝蓋高度、前導腳踝速度的時間曲線。實線是偵測結果，虛線是標註，用來調整門檻

分段門檻集中在 `src/bbagent/analysis/segmentation.py` 的 `PitchSegParams`。預設值是用合成資料調的，要用標註過的真實影片再校正。

## 階段 4：Gemini 工具

- `src/bbagent/llm/gemini_client.py`：Gemini 呼叫的統一入口。
  - **重試**：429 與 5xx 指數退避；免費層級額度為 0 時不重試，直接說明原因。
  - **結構化輸出**：回傳結果以 Pydantic 驗證。
  - **串流**：文字逐段產生。
  - **回應快取**：`config.llm.cache_mode`，`auto` 時同樣的輸入不重複消耗額度。
- `src/bbagent/tools/scout_overview.py`：看 4 張均勻取樣的畫面，判斷視角、主體、鏡頭是否移動、是否全身入鏡，並讀取記分板或測速螢幕。
- `src/bbagent/tools/gemini_inspect.py`：看關鍵幀回答觀察重點。每項觀察都附信心等級與依據的圖片編號。
- 提示詞放在 `src/bbagent/prompts/*.md`。

試跑（需要 `.env` 中的金鑰）：

```bash
uv run python scripts/try_gemini.py overview data/samples/p_other_01.mp4
uv run python scripts/try_gemini.py inspect data/samples/p_other_01.mp4 --frames 115,139,144 --focus "出手時投球手臂的角度"
uv run python scripts/try_gemini.py stream "用三句話說明什麼是 RE24"
```

## 階段 5：Agent 迴圈

```bash
uv run python scripts/run_agent.py data/samples/p_other_01.mp4 --inning 7 --half bottom --outs 1 --bases 1 --role offense --hand R
uv run python scripts/run_agent.py data/samples/p_other_01.mp4 --outs 1 --bases 1 --hand R --demo-fault   # 示範 Critic 退回
```

流程（`src/bbagent/agent/`）：

1. **偵察**：`probe_video` → `scout_overview`，規劃前固定執行。
2. **Planner**：Gemini 決定工具、觀察重點與候選戰術；程式檢查工具是否存在、前置條件、必要步驟與戰術是否合法。不通過就附上錯誤重新規劃，Gemini 無法使用時改用預設計畫。
3. **Executor**：依相依順序在背景執行緒呼叫工具，即時回報進度。關鍵工具失敗時重新規劃一次。
4. **Synthesizer**：依證據與關鍵幀串流撰寫報告，並引用證據編號。
5. **Critic**：先跑規則檢查（`critic_rules.py`：章節、引用、RE24 數字、戰術合法性、Tipping 用語、拍攝限制），通過後再請 Gemini 檢查推論品質。不通過就退回修正，最多 2 次。

第 6 節「Agent 執行摘要」由程式產生。報告與執行紀錄存在 `runs/agent/<時間>_<影片>/`（`report.md`、`trace.json`、`keyframes/`）。

新增工具：`camera_motion`（OpenCV 光流量測鏡頭移動）、`tipping_compare`（各球動作差異）、`render_keyframes`（關鍵幀疊圖）。

## 階段 6：Gradio 介面

```bash
uv run python scripts/app.py            # 開啟 http://127.0.0.1:7860
uv run python scripts/app.py --share    # 產生公開連結
```

左邊上傳影片、填入賽況，下方的局面圖會即時更新；右邊分成 4 個分頁：

| 分頁 | 內容 |
|---|---|
| Agent 過程 | 計畫表（每個步驟的工具、依賴、執行狀態、證據編號）＋ 思考與工具呼叫紀錄（工具執行中會顯示進度，完成後可展開看結果） |
| 報告 | 報告逐字串流、下載 `report.md` 與 `trace.json`、Critic 每一輪的審查結果 |
| 關鍵幀 | 每一球關鍵階段的骨架疊圖 |
| 動作曲線 | 分段用的 4 條訊號與偵測到的動作階段 |

開始撰寫報告時會自動切到「報告」分頁。勾選「進階選項 → 示範模式」可以看到 Critic 退回修正的過程。

介面只消費 `run_agent` 的事件：`src/bbagent/ui/view.py` 把事件整理成畫面狀態（不依賴 Gradio，可單獨測試），`app.py` 負責版面。

## 階段 7：打者分支

影片主角是打者時（自動判斷，或用 `--target batter`、介面的「分析對象」指定），Agent 改走打者分支：

- `src/bbagent/analysis/swing.py`：每次揮棒的 5 個階段：準備姿勢、引棒、跨步著地、擊球、收棒。擊球以手腕速度峰值代替，看球不揮的球不會被偵測。
- `src/bbagent/analysis/batting_metrics.py`：引棒到擊球時間、著地到擊球時間、跨步長度（側拍）、頭部移動量、最大手速、擊球時手的高度。
- 新工具 `batting_metrics`、`swing_compare`（各次揮棒是否一致）。Planner 會檢查工具是否適用於所選分支，例如打者計畫不能用 `tipping_compare`。

```bash
uv run python scripts/run_agent.py data/samples/b_side_01.mp4 --target batter --role defense --batter-hand L
```

## 階段 3：球軌跡

新工具 `track_ball`（只用於投手分支）：

1. `src/bbagent/vision/ball_scan.py`：只在每一球出手前 0.05 秒到出手後 0.6 秒逐幀找球（Pass 2 細掃，`config.vision.fine_scan`），結果快取在 `cache/ball_scan/`。
2. `src/bbagent/analysis/ball_track.py`：過濾誤報（離手腕太近的、在同一位置重複出現的），再用 RANSAC 擬合軌跡（x 等速、y 二次曲線），至少 5 幀、0.06 秒、速度夠快才採用。
3. 側拍且鏡頭固定時，用投手身高（假設 1.8 公尺）把像素換成公尺，粗估球速與飛行時間，得到「到本壘時間（球軌跡估計）」。非側拍時只回報軌跡。

軌跡圖會出現在介面的「關鍵幀」分頁。球在手機影片中常常只有幾個像素，找不到軌跡是正常的，Agent 會退回固定 0.42 秒飛行時間的估計。

## 揮棒偵測評估（MLB-YouTube）

用 [MLB-YouTube 資料集](https://github.com/piergiaj/mlb-youtube)（Piergiovanni & Ryoo, 2018）的標註，量測打者揮棒偵測在 MLB 轉播畫面上的準確度。資料集只有標註，影片要自己取得；這些是 MLB 轉播畫面，有版權，只下載少量做自己的測試，**不要放進 repo**（`cache/`、`runs/` 已在 `.gitignore`）。

```bash
uv run python scripts/eval_mlb_youtube.py plan --labels ~/Downloads/mlb-youtube-master.zip --games 2
# 把列出的整場比賽存成 <YouTube影片id>.mp4，放在同一個資料夾
uv run python scripts/eval_mlb_youtube.py run --labels ~/Downloads/mlb-youtube-master.zip --videos ~/mlb_games --limit 40
```

- 片段用 OpenCV 依標註自動切好（資料集附的切片腳本是 Python 2，無法直接使用），快取在 `cache/mlb_clips/`。
- 抽樣時有揮棒與沒揮棒各半。
- 比較三種方式：Agent 原本的規則（分數最高的人）、轉播規則（排除身形最大的人，也就是中外野視角下的投手），以及轉播改良：先從投手骨架找出手時間，只看出手後 0.15–0.75 秒，並要求雙手同時快速移動、彼此靠近、移動距離夠長。改良版的門檻是事先設定的，要確認效果請換一批片段（`--seed 1` 或另一場比賽）再評估。
- 結果在 `runs/eval/mlb_youtube/<時間>/`：`summary.md` 有準確率、精確率、召回率，以及「擊中球」與「揮空」分開的召回率；`results.csv` 是每段片段的結果。

轉播是中外野視角、常有切換鏡頭，和本專案設計的側拍差很多，這份數字只代表轉播畫面上的表現。

## Colab

`notebooks/colab_demo.ipynb` 是作業繳交用的教材式筆記本：一段說明配一小格程式，依序展示偵察 → Planner（含計畫檢查）→ Tool Use → Synthesizer → Critic（含注入錯誤的規則檢查），再介紹設計模式與參考文獻，最後才用 `run_agent()` 串起全部並啟動 Gradio 工作台。

- 預設從本專案的 GitHub repo 取得程式碼；也可以改用公開的 Drive 連結或自己的 Drive。
- 金鑰放在 Colab 的 Secrets（`GEMINI_API_KEY`），不要寫在筆記本裡；沒有金鑰也能跑完（Gemini 步驟改用預設流程）。
- 會額外消耗 API 額度或公開連結的格子都預設關閉（`RUN_…`、`START_DEMO` 改成 `True` 才執行）。
- `data/samples/*.mp4` 不進 Git；找不到範例影片時，筆記本會請你上傳。

打包專案（在專案的上一層資料夾執行）：

```bash
zip -r baseball-tactics-agent.zip baseball-tactics-agent -x "*/.venv/*" "*/.uv-cache/*" "*/models/*" "*/runs/*" "*/cache/*" "*/.env" "*/__pycache__/*" "*/.pytest_cache/*"
```

## 目錄

```
config.yaml          模型、抽幀參數、Gemini 型號、路徑
src/bbagent/
  agent/             Orchestrator、Planner、Executor、Synthesizer、Critic、工具註冊表
  vision/            影片讀取、YOLO 載入、裝置選擇、粗掃、找球細掃、快取、鏡頭移動、繪圖、圖片讀寫
  analysis/          骨架時間序列、主體識別、投手／打者動作分段與指標、球軌跡擬合、標註比對、動作曲線圖
  ui/                Gradio 介面、事件整理（view.py）、局面圖（diagram.py）
  llm/               Gemini 客戶端（重試、結構化輸出、串流、快取）、提示詞載入
  prompts/           提示詞（.md）
  tools/             給 Agent 呼叫的工具：RE24、scout_overview、gemini_inspect、關鍵幀、球軌跡
  schemas.py         共用資料結構
tests/               單元測試（synthetic.py 產生合成投手骨架，fake_gemini.py 模擬 Gemini）
scripts/             check_env / download_models / benchmark_mps / annotate_samples / analyze_video / try_gemini / run_agent / app
notebooks/           Colab 筆記本
data/samples/        測試影片與標註
models/ cache/ runs/ 權重、YOLO 快取、執行輸出（不進版本控制）
```
