# 一拍即安

2026 和泰 AI 黑客松，題目 5「運用 AI 打造 iRent 智能車況管家：優化租還車體驗與營運效率」的參賽作品。

還車檢查分成三步：

1. 拍照防呆：用戶拍照時，畫面上有範例圖層和即時提示，拍完馬上檢查照片能不能用。
2. 模型判讀：把取車照和還車照一起給模型，看這次租車有沒有多出新的車損，給出一個分數。
3. 分流分工：新車損分數低的直接放行，其他的轉給人工看。有問題的車會開工單，處理完、驗收通過才重新上架。

模型只負責打分和分流，不會直接扣款。人工確認之後才通知用戶，用戶也可以一鍵申訴。

這裡放的是 demo 和實驗程式。所有模型都在本機跑，官方資料、照片和訓練出來的權重都沒有放上來，原因寫在下面的「資料保密」。

## 這個 repo 裡有什麼

| 題目環節 | 做了什麼 | 程式 |
|---|---|---|
| 環節一 即時防呆 | 手機取景框疊上半透明的範例圖層。瀏覽器每 0.3 秒用 TF.js COCO-SSD 偵測一次車輛（上一次還沒算完就跳過），提示往左、往右或太遠。拍下時檢查模糊、亮度和直橫，伺服器逐張回饋，再用車牌 OCR 確認是同一台車、用儀表板里程 OCR 對照車聯網里程 | `app/static/app.html`、`app/static/overlay.js`、`app/vision/quality.py`、`app/vision/plate.py` |
| 環節二 新車損判讀 | 讓本機的開源視覺語言模型（VLM）判斷還車照有沒有取車照沒有的新傷。做法參考 Jev：不讓模型寫字，只讀它回答 Yes 或 No 的機率。總共比較了 10 多種模型和設定 | `experiments/vlm_option_probs.py`、`experiments/jev_like_probs.py` |
| 環節二 自訓小模型 | 用大模型當老師，在沒有標註的照片上蒸餾出 0.8B 的小模型（LoRA），簡報裡叫 Classifier-0.8B | `experiments/jev_student.py` |
| 環節三 營運閉環 | 車輛和車損紀錄的狀態機、工單、驗收、通知、申訴，以及後台工作台 | `app/fsm.py`、`app/pipeline.py`、`app/static/admin.html` |

demo 怎麼啟動、怎麼錄影寫在 [docs/DEMO.md](docs/DEMO.md)，所有實驗的完整數字和方法在 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 結果

評估用官方的 141 對取車／還車照片，其中 21 對有索賠（新車損）、120 對正常，共 50 台車。所有數字都是在本機一張 RTX 4070 Ti SUPER（16 GB）上跑的，每對秒數包含讀圖。

| 方法 | AUC [95% CI，以車為單位 bootstrap] | 每對秒數 | 要不要訓練 |
|---|---|---|---|
| Qwen3.5-4B，讀 Yes/No 機率（原始解析度，單一問法） | 0.954 [0.91, 0.98] | 0.53 | 不用 |
| 同上，三個問法平均 | 0.962 [0.93, 0.99] | 1.57 | 不用 |
| 自訓小模型：Qwen3.5-0.8B + LoRA（從 4B 蒸餾） | 0.941 [0.89, 0.98] | 0.32 | 只用沒有標註的照片 |
| 傳統做法：訓練影像分類器（ResNet-50、EfficientNet-B0、ConvNeXt-T） | 0.61–0.84 | — | 要 |
| 傳統做法：CLIP 向量相似度（不訓練） | 0.84 | — | 不用 |

- 在原始解析度下，4B（0.954）比最好的訓練分類器 ConvNeXt（0.840）高 0.113 [+0.02, +0.22]，比 CLIP 高 0.117 [+0.02, +0.23]，兩個信賴區間都不含 0。不過原始解析度是比過 4 種解析度才選的，換成 512、768、1024 時 4B 還是比較高（+0.06 到 +0.10），但信賴區間包含 0。
- 兩張照片一起給時，模型會受取車照影響：同一張還車照放兩次，141 對全部回答沒有新傷。但只看還車照也有 0.942，加入取車照只多 +0.012 [−0.03, +0.06]，在這批資料上還不顯著（見「限制」）。
- 自訓的 0.8B 小模型跟老師（4B 單一問法）只差 −0.012 [−0.04, +0.01]，速度快 1.6 倍，訓練時顯示記憶體最多用到 2.4 GiB。

### 要轉人工的比例

門檻設在 21 對已知索賠全部抓到的位置（用 4B 的分數）：

- 以單一角落的照片配對來算，約 17% 要人工看。這裡用的是 3,850 對沒有標註的配對：1,352 對是取車照對還車照，2,498 對是同一台車前後兩筆訂單的還車照。
- 以一整筆還車來算，約 50% 要人工看。這裡用的是 500 組補充照片裡四個角都有的 338 筆，任一角超過門檻就送人工。

所以在不漏掉已知索賠的情況下，大約一半的還車可以直接放行。另一半由人工看，看的時候會附上分數和前後對照圖。

## 參考 Jev 的做法

TypeSafe AI 在 2026-09-15 [發表了 Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。它不寫文字，直接回傳答案和機率，再依信心決定自動處理還是轉人工。不過依照[官方文件](https://docs.typesafe.ai/models.md)，Jev 只吃文字和 JSON，也只能透過雲端 API 使用，而比賽資料不能送給第三方。所以我們沒有直接用 Jev，改成把同樣的做法套在本機的開源 VLM 上（簡報裡的「System One 讀法」）：

- 不生成文字：問模型「還車照有沒有取車照沒有的新傷？回答 Yes 或 No」，只讀答案位置 Yes 和 No 兩個 token 的 log 機率，當成連續的分數。
- 一次就完成：只跑一次前向（`logits_to_keep=1`），不生成任何文字。
- 校準機率：小模型用 soft-target BCE 去貼近老師的機率。不過目前還沒有對真實比例做校準，老師在 E1 上有 40% 回答 Yes，實際索賠只有 15%，上線前要用試點資料做溫度或 Platt 校準。
- 依信心分流：分數低於 T1 自動放行，高於 T1 送人工，T1 依索賠的召回率來設。
- 可以自己訓練：大模型當老師、小模型當學生，在沒有標註的照片上蒸餾（見下一節）。

我們也在本機試了 Hugging Face 上別人訓練好的 Jev-like 影像模型：imajev-4b、Intern-Decision-4B、vev-4b。推論程式是照它們的文件自己重寫的，沒有執行原作者的程式碼，權重也只用 safetensors。三個模型的 AUC 都跟原版 Qwen3.5-4B 差不多（0.95–0.97）。imajev 慢 2–3 倍，Intern 和 vev 跟 4B 差不多快；但三個模型在照片對調後分數還是很高（0.72–0.85），比較分不出哪張是取車照。

## 自訓小模型（蒸餾）

1. 老師：Qwen3.5-4B（原始解析度，問法 P1），在 3,850 對沒有標註的配對上打出新車損的機率。其中 1,352 對來自官方 500 組補充照片，是真正的取車照對還車照；2,498 對來自主辦單位的上傳清單，是同一台車前後兩筆訂單的還車照。老師用的解析度和問法是在 E1 上比較後選的。
2. 學生：Qwen3.5-0.8B，只在語言模型那一側加 LoRA（r=16，可訓練參數約 1,000 萬），用 soft-target BCE 學老師的機率。訓練 1 個 epoch，花 75 分鐘。
3. 評估：用官方的 141 對。學生訓練時沒有用到這 141 對，也沒有用到任何標註。重疊的檢查：
   - 三個資料來源之間沒有完全相同的檔案（MD5 比對）。
   - 上傳清單用車牌排除了 E1 的車，找到 1 台車、2 對配對，已經拿掉。
   - 500 組補充照片沒有車牌資訊，沒辦法確認車子有沒有重疊。

| | AUC | 照片對調後 AUC | 每對秒數 |
|---|---|---|---|
| 未訓練的 0.8B（P1） | 0.899 | 0.74 | 0.25 |
| 蒸餾後的 0.8B | 0.941 | 0.35 | 0.32 |
| 老師 4B（P1） | 0.954 | 0.49 | 0.53 |

- 蒸餾之後，小模型對照片順序變得敏感，照片對調後 AUC 從 0.74 掉到 0.35。
- 但兩張放同一張照片時，小模型的 AUC 還有 0.805（老師是 0.602），所以還不能說它學會了排除舊傷。
- AUC 多了 0.042 [−0.01, +0.11]，方向是進步，但只有 21 對正樣本，還不足以證明顯著。

## 限制

- 正樣本只有 21 對，AUC 的 95% 信賴區間大約 ±0.03–0.04，方法之間差 0.01–0.04 分不出好壞。要說召回率至少 95%，需要 59 對全部抓到；21/21 的單側 95% 信賴下界是 86.7%。
- 有些設定是在同一批資料上比過才選的，包括解析度、問法和集成組合，所以數字可能偏樂觀。
  - 原始解析度比縮到 1024 好（0.962 vs 0.915），三種大小的 Qwen3.5 方向一致，但 4B 的差距 +0.047 [−0.003, +0.12] 不顯著。
  - 三個模型集成（4B + Gemma 4 + imajev）有 0.979，但組合是事後挑的，只能算待驗證的假設。
- 轉人工比例的門檻是看過答案才設的，由 21 對裡分數最低的那一對決定。對新的案件，預期召回率大約 95%，不是 100%。
- 21 對索賠大多是明顯的傷，小刮痕還沒驗證過。
- 只看還車照就有 0.942，加入取車照的增益（+0.012）在這批資料上還不顯著。另外看取車照本來就有傷的正常配對（由模型判斷，n=15）：只看還車照有 10 對被判有傷，兩張一起看降到 3 對，誤報變少但沒有消失。
- 我們也試過用客人拍的車損近照來推測新傷，做了兩份弱標籤評估：E2 用 500 組補充照片，E3 用上傳清單，推測用的模型就是 Qwen3.5-4B。這種標籤雜訊太大，E2 落在 0.43–0.57、E3 落在 0.63–0.67，信賴區間約 ±0.1–0.15，分不出方法好壞。之後需要更多人工標註的真標籤。
- 主要數字都從原始分數重新算過，找到並修正的錯誤列在 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 第 9 節。

## 資料保密

官方資料包規定只能用在這次比賽，不能對外提供，所以：

- repo 裡沒有官方照片、上傳清單，也沒有從照片衍生的檔案（例如含檔名的分數快取、向量、範例疊圖），或用這些照片訓練出來的權重。`.gitignore` 也排除了這些路徑。
- 推論和訓練都在本機 GPU 上做，照片沒有送到任何雲端 API。
- `results/final_summary.json` 只有彙總後的數字，沒有檔名或車牌。

要重現實驗，需要自己準備官方資料包，再用環境變數指過去，見下一節。

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
| [vev-4b-lora](https://huggingface.co/CountingSheep/vev-4b-lora) | CC BY-NC 4.0 | 只拿來做研究比較，沒有用在提案的方案裡 |
| TF.js COCO-SSD | Apache-2.0 | 手機端即時位置引導 |
