# iRent 智能車況管家

**2026 和泰 AI 黑客松 · 題目 5：運用 AI 打造 iRent 智能車況管家：優化租還車體驗與營運效率**

> **拍對 → 判對 → 派對 → 判得公平**

用戶拍照的當下就確認照片合格（拍對）。AI 比對取車照與還車照，找出「這次租車產生的新車損」，並依信心分流（判對）。異常自動變成工單，驗收通過才重新上架（派對）。AI 只負責分流，不直接計費，人工確認後才通知用戶，用戶可以一鍵申訴（判得公平）。

這個 repo 收錄我們的 demo app 和實驗程式。所有模型都在本機推論，**官方資料、照片和訓練出來的權重都沒有放進 repo**，原因見下方〈資料保密〉。

---

## 這個 repo 裡有什麼

| 題目環節 | 我們做的事 | 程式 |
|---|---|---|
| **環節一 即時防呆**：現場拍照的即時分析 | 手機取景框疊上半透明的範例圖層。手機瀏覽器每 0.3 秒跑一次 TF.js COCO-SSD 車輛偵測（上一次還沒算完就跳過），提示「往左／往右／太遠」。拍下的瞬間檢查模糊、亮度、直橫。伺服器逐張回饋，並用車牌 OCR 確認是這台車、用儀表板里程 OCR 對照車聯網里程 | `app/static/app.html`、`app/static/overlay.js`、`app/vision/quality.py`、`app/vision/plate.py` |
| **環節二 新車損判讀**：本機模型打分 | 用 **Jev 式讀法**讓本機開源視覺語言模型（VLM）判斷「還車照有沒有取車照沒有的新傷」：不生成文字，一次前向，直接讀 Yes/No 答案 token 的機率。比較了 10 多種本機模型與設定 | `experiments/vlm_option_probs.py`、`experiments/jev_like_probs.py` |
| **環節二** **Jev-like 模型訓練** | 用大模型當老師，在無標註照片上蒸餾出一個只有 0.8B 參數的快速分類器（LoRA） | `experiments/jev_student.py` |
| **環節三 營運閉環** | 車輛狀態機、車損紀錄狀態機、工單、驗收、通知、申訴、後台工作台 demo | `app/fsm.py`、`app/pipeline.py`、`app/static/admin.html` |

demo 怎麼啟動、怎麼錄影，見 [docs/DEMO.md](docs/DEMO.md)。所有實驗的完整數字與方法見 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

---

## 核心結果

評估資料：官方的 141 對取車／還車照片，其中 21 對有索賠（新車損）、120 對正常，共 50 台車。所有數字都是本機推論；硬體為一張 RTX 4070 Ti SUPER（16 GB）；每對秒數包含讀圖。

| 方法 | AUC [95% CI，以車為單位 bootstrap] | 每對秒數 | 要不要訓練 |
|---|---|---|---|
| **Qwen3.5-4B，Jev 式讀 Yes/No 機率（原始解析度，單一問法）** | **0.954** [0.91, 0.98] | **0.53** | 不用 |
| 同上，三個問法平均 | 0.962 [0.93, 0.99] | 1.57 | 不用 |
| **自訓 Jev-like 學生：Qwen3.5-0.8B + LoRA（從 4B 蒸餾）** | **0.941** [0.89, 0.98] | **0.32** | 只用無標註照片 |
| 傳統做法：訓練的影像分類器（ResNet-50、EfficientNet-B0、ConvNeXt-T） | 0.61–0.84 | — | 要 |
| 傳統做法：CLIP 向量相似度（不訓練） | 0.84 | — | 不用 |

- **在原始解析度下，用 Jev 式讀法的 VLM 高於傳統訓練出來的影像分類器**：
  - 4B（P1，0.954）比最好的訓練分類器 ConvNeXt（0.840）高 **+0.113** [+0.02, +0.22]，比 CLIP 高 **+0.117** [+0.02, +0.23]，兩個信賴區間都不含 0。
  - 原始解析度是比較 4 種解析度後才選定的。如果換成 512、768、1024，4B 仍然比較高（+0.06 到 +0.10），但信賴區間包含 0。
- **兩張同時給，模型會受取車照影響**：同一張還車照放兩次時，141 對全部回答「沒有新傷」。不過只看還車照就有 0.942，在這批資料上加入取車照的增益是 +0.012 [−0.03, +0.06]，還不顯著（見〈誠實聲明〉）。
- **自訓的 0.8B 學生**：跟老師（4B P1）只差 −0.012 [−0.04, +0.01]，速度快 1.6 倍，訓練時的顯示記憶體峰值 2.4 GiB。

### 轉人工比例（用來估人力效益）

門檻設在「21 對已知索賠全部抓到」時：

| 計算單位 | 要人工看的比例 |
|---|---|
| 單一角落的照片配對（3,850 對無標註配對：1,352 對是真正的取車照對還車照，2,498 對是同一台車相鄰兩筆訂單的還車照） | 約 17% |
| 一整筆還車（四個角齊全，任一角超過門檻就送人工；500 組補充照片中的 338 筆） | 約 50% |

也就是說，在不漏掉已知索賠的設定下，**AI 可以直接放行大約一半的還車**，人工只看另一半，而且是附上 AI 標記和前後對照圖再看。

---

## Jev 的原理怎麼用在影像上

TypeSafe AI 的 Jev（[2026-09-15 發布](https://typesafe.ai/blog/introducing-system-one-models-and-jev)）是「決策模型」：不寫文字，直接回傳型別化的答案和機率，再依信心決定自動執行或轉人工。依[官方文件](https://docs.typesafe.ai/models.md)，Jev 只吃文字與 JSON、只能透過雲端 API 使用；而競賽資料不得送往第三方。所以我們**沒有使用 Jev 本身，而是把它的原理搬到本機開源 VLM 上**：

| Jev 的原則 | 我們的做法 |
|---|---|
| 不生成文字，回傳型別化答案 | 問「還車照有沒有取車照沒有的新傷？回答 Yes 或 No」，只讀答案位置 Yes/No 兩個 token 的 log 機率，得到連續分數 |
| 一次呼叫就完成 | 一次前向，`logits_to_keep=1`；不生成任何文字 |
| 校準過的機率 | 學生用 soft-target BCE（適當評分規則）去貼近老師的機率。**目前還沒有對真實比例做校準**：老師在 E1 上有 40% 回答 Yes，但實際索賠只有 15%。上線前需要用試點資料做溫度或 Platt 校準 |
| 依信心路由 | 分數低於 T1 自動放行；高於 T1 送人工；T1 依「索賠召回率」設定 |
| 可以自己訓練 | 大模型當老師、小模型當學生，在無標註照片上蒸餾（見下節） |

我們也在本機實測了 Hugging Face 上別人訓練好的 Jev-like 影像模型：imajev-4b、Intern-Decision-4B、vev-4b。推論程式依照它們的文件**自行重寫**，不執行原作者的程式碼，權重只接受 safetensors。三個模型的 AUC 都和原版 Qwen3.5-4B 打平（0.95–0.97）。imajev 的速度慢 2–3 倍，Intern、vev 跟 4B 差不多；但三者在照片對調後分數仍高（0.72–0.85），比較不會分辨哪張是取車照。

## 自訓 Jev-like 快速分類器（蒸餾）

1. **老師**：Qwen3.5-4B（Jev 式讀法，原始解析度，P1），在 **3,850 對無標註**配對上給出 P(新車損)。其中 1,352 對來自官方 500 組補充照片，是真正的取車照對還車照；2,498 對來自主辦單位上傳清單，是同一台車相鄰兩筆訂單的還車照。老師的設定（解析度、問法）是在 E1 上比較後選的。
2. **學生**：Qwen3.5-0.8B，只在語言模型那一側加 LoRA（r=16，可訓練參數約 1,000 萬），用 soft-target BCE 去學老師的機率。訓練 1 個 epoch，75 分鐘。
3. **評估**：用官方 141 對評估。學生的訓練沒有用到這 141 對，也沒有用到任何標註。我們做了兩項重疊檢查：
   - 三個資料來源之間沒有完全相同的檔案（MD5 比對）。
   - 上傳清單的部分用車牌排除了 E1 的車：找到 1 台車、2 對配對，已經移除。
   - 500 組補充照片沒有車牌資訊，所以無法確認車輛不重疊。

結果：

| | AUC | 照片對調後 AUC | 每對秒數 |
|---|---|---|---|
| 未訓練的 0.8B（P1） | 0.899 | 0.74 | 0.25 |
| **蒸餾後的 0.8B 學生** | **0.941** | **0.35** | 0.32 |
| 老師 4B（P1） | 0.954 | 0.49 | 0.53 |

- 蒸餾後，學生的分數對照片順序變得敏感：照片對調後 AUC 從 0.74 掉到 0.35。
- 但兩張照片相同時，學生的 AUC 仍有 0.805（老師是 0.602），所以**還不能說它學會了排除舊傷**。
- AUC 增益 +0.042 [−0.01, +0.11]：方向是進步，但 21 對正樣本還不足以證明顯著。

---

## 誠實聲明與限制

- **正樣本只有 21 對。**AUC 的 95% 信賴區間約 ±0.03–0.04，方法之間 0.01–0.04 的差距分不出好壞。要宣稱「召回 ≥ 95%」，至少需要 59 對全部抓到；21/21 的單側 95% 信賴下界是 86.7%。
- **有些設定是在同一批資料上比較後才選的**，包括影像解析度、問法、集成組合，因此數字可能偏樂觀。
  - 原始解析度比縮到 1024 好（0.962 vs 0.915），三種大小的 Qwen3.5 方向一致，但 4B 的差距 +0.047 [−0.003, +0.12] 不顯著。
  - 三模型集成（4B + Gemma 4 + imajev）有 0.979，但組合是事後挑的，列為待驗證的假設。
- **轉人工比例的門檻是看過答案才設的**，由 21 對中分數最低的那一對決定；對新案件的期望召回約 95%，不是 100%。
- 21 對索賠多半是明顯的傷，小刮痕還沒有驗證過。
- 只看還車照就有 0.942，加入取車照的增益（+0.012）在這批資料上還不顯著。以「取車照本來就有傷的正常配對」來看（模型判斷，n=15）：只看還車照會有 10 對被判有傷，兩張一起看降到 3 對。也就是誤報減少，但沒有消失。
- 我們用「客人拍的車損近照」推測新傷，做了兩份弱標籤評估：E2 用 500 組補充照片，E3 用上傳清單。推測用的模型就是 Qwen3.5-4B。這種標籤的雜訊太大：E2 落在 0.43–0.57、E3 落在 0.63–0.67，信賴區間約 ±0.1–0.15，分不出方法好壞。下一步需要人工標註更多真標籤。
- 主要結論都經過三個獨立的 AI 審查代理，從原始分數重算並專門找漏洞，再依結果修正措辭。修正內容列在 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 最後。

## 資料保密

官方資料包規定只能用於本次競賽、不得對外提供。因此：

- **repo 裡沒有**任何官方照片、上傳清單、從照片衍生的檔案（例如含檔名的分數快取、向量、範例疊圖），也沒有用這些照片訓練出來的權重。`.gitignore` 也排除了這些路徑。
- **所有推論與訓練都在本機 GPU 完成**，沒有把照片送到任何雲端 API。
- `results/final_summary.json` 只有彙總後的數字，不含任何檔名或車牌。

要重現實驗，需要自己準備官方資料包，再用環境變數指向它，見下節。

## 如何重現

環境：Windows 11、Python 3.12、NVIDIA GPU（16 GB）。

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-experiments.txt
```

需要的環境變數：

- `IRENT_DATA`：指向含 `進行索賠資料/`、`沒有進行索賠/` 的官方資料夾。
- `IRENT_PIC500`：指向 500 組補充照片的 `Hims_Pic500/`。
- `IRENT_XLSX_PHOTOS`：放上傳清單照片的資料夾，只有建弱標籤評估集和蒸餾訓練池時需要。
  - 照片要自己依官方上傳清單下載。下載程式含官方儲存位址，所以沒有公開。
  - 資料夾裡要有 `manifest.csv`，欄位是 `order_number, CarNo, ImageType, Image, status, bytes`，照片放在 `<order_number>/<Image>`。
- `IRENT_MODELS`（預設 `~/irent_models`）：Jev-like 權重與學生 LoRA 的位置。
- `IRENT_CACHE`（預設 `~/irent_cache`）：影像快取。

```powershell
# 本機 VLM 用 Jev 式讀法打分（141 對；P1-P3 加上對照組 P4/P5/P1S/P6；--max-side 1536 = 不縮圖）
python experiments/vlm_option_probs.py --model Qwen/Qwen3.5-4B --max-side 1536
python experiments/vlm_option_probs.py --model google/gemma-4-12B-it --quant 4bit --max-side 1536
# HF 上的 Jev-like 模型（safetensors 權重需先下載到 %IRENT_MODELS%\hf）
python experiments/jev_like_probs.py --model imajev-4b --size 400k --slots P1,P2,P3
python experiments/jev_like_probs.py --model intern-4b --size native
python experiments/jev_like_probs.py --model vev-4b --size native
# 建立弱標籤評估集（E2/E3）與蒸餾訓練池（需要上傳清單照片）
python experiments/xlsx_pairs.py
python experiments/weak_eval.py xdmg
python experiments/weak_eval.py build
# 老師在訓練池上打分，再訓練 Jev-like 學生（並跑照片對調、同一張放兩次的對照組）
python experiments/vlm_option_probs.py --model Qwen/Qwen3.5-4B --max-side 1536 --pool json --pairs-json %IRENT_XLSX_PHOTOS%\train_pool.json --prompts P1
python experiments/jev_student.py --student Qwen/Qwen3.5-0.8B --tag s08-native --teacher-size 1536 --max-side 1536 --slots P1,P1S,P6
# 弱標籤評估：先用各模型對 e2_pairs.json / e3_pairs.json 打分（--pool json --prompts P1），再
python experiments/weak_eval.py eval
# 傳統基準（訓練分類器；CLIP 基準需先跑 claim_classifier.py 產生向量）與彙總
python experiments/finetune_classifier.py --tasks q2 --seeds 20261003,1,2
python experiments/claim_classifier.py
python experiments/final_summary.py
```

## 檔案結構

```
app/                     demo（FastAPI + SQLite）
  static/app.html        手機端：取景框、即時引導、上傳、結果、通知、申訴
  static/overlay.js      取景框範例圖層
  static/admin.html      後台工作台：轉人工佇列、工單、車況
  vision/                品質檢查、車牌／里程 OCR、取還車比對、車內整潔
  fsm.py, pipeline.py    車輛／車損狀態機、三個環節的流程
experiments/
  vlm_option_probs.py    本機 VLM 用 Jev 式讀法打分（多種模型、解析度、對照組）
  jev_like_probs.py      HF 上的 Jev-like 模型（自行重寫的推論）
  jev_student.py         Jev-like 學生蒸餾訓練與評估
  weak_eval.py           弱標籤評估集（E2/E3）與蒸餾訓練池
  xlsx_pairs.py          上傳清單 → 同車連續訂單的配對
  finetune_classifier.py 傳統影像分類器（凍結特徵與 fine-tune）
  metadata_baseline.py   只看檔案資訊的基準（檢查捷徑）
  compare_pairs.py, final_summary.py, score_variants.py   比較表與彙總
  claim_classifier*.py   第一版基準（CLIP／DINOv2）
results/final_summary.json   彙總數字（不含檔名）
docs/DEMO.md            demo 啟動與錄影
docs/EXPERIMENTS.md     完整實驗紀錄
```

## 使用的模型與授權

| 模型 | 授權（以模型卡為準） | 用途 |
|---|---|---|
| [Qwen3.5-0.8B／2B／4B／9B](https://huggingface.co/Qwen/Qwen3.5-4B) | Apache-2.0 | 主要模型、老師、學生 |
| [Gemma 4 12B-it](https://huggingface.co/google/gemma-4-12B-it) | Apache-2.0 | 比較 |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 | 比較 |
| [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | MIT | 比較 |
| [imajev-4b](https://huggingface.co/mohit67890/imajev-4b)、[Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B) | Apache-2.0 | Jev-like 比較 |
| [vev-4b-lora](https://huggingface.co/CountingSheep/vev-4b-lora) | CC BY-NC 4.0 | **只做研究比較，不用在提案方案裡** |
| TF.js COCO-SSD | Apache-2.0 | 手機端即時位置引導 |
