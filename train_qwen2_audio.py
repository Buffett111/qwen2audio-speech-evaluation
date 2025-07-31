"""
訓練腳本：使用 Qwen2‑Audio‑7B‑Instruct 與 LoRA 在第二語言口語評量資料集上進行微調

本程式遵循《Assessment of L2 Oral Proficiency using Speech Large Language Models》一文所述的設定，
將 Qwen2‑Audio 7B 模型視為基準模型，並透過 Low‑Rank Adaptation (LoRA) 在編碼器與解碼器層
插入階數為 16 的適配器。模型僅更新少數參數，其餘權重保持凍結。根據用戶提供的輸入，
程式可選擇三種不同的損失函式：

1. **交叉熵分類（CE）**：將 CEFR 等級映射為字母 A–F，使用單一標籤計算交叉熵。
2. **公平平均（FA）**：以 softmax 機率加權六個候選等級對應的數值（1–6），與參考分數
   做均方差比較。此方法符合論文提出的「fair average」損失【357919207058670†L155-L170】。
3. **迴歸（Reg）**：在模型最後加入線性層，直接預測實數評分，並以均方誤差為損失。

使用者需事先透過 `huggingface‑cli login` 登入 Hugging Face 以存取私人資料集。
資料集以 [`datasets.load_dataset`](https://huggingface.co/docs/datasets) 讀取，並期望每筆樣本
至少包含 `audio` 欄（內含音訊檔案與取樣率）與 `score` 欄（浮點數，1 至 6，允許半分）。
若欄位名稱不同，請修改 `ScoreDataset` 類別中取得分數的部分。

程式將把每段音訊切割為長度不超過 30 秒的片段，並針對每個片段建立一個 ChatML
對話，包含系統提示、使用者的音訊以及模型要產生的一個字母（A–F）作為標籤。
之後透過 Hugging Face `AutoProcessor.apply_chat_template` 將對話轉為模型輸入，
使用模型產生的最後一個 token 的 logits 來計算損失。程式支援批次訓練與驗證，並可輸出
RMSE、皮爾森相關與斯皮爾曼相關等評估指標。

參考文獻：文中指出在 Linguaskill 與 S&I 資料集上採用階數 16 的 LoRA，
訓練 2 個 epoch，學習率 1e‑4，S&I 批次大小為 8【357919207058670†L278-L295】。

使用方式（範例）：

```bash
python train_qwen2_audio.py \
    --dataset_name ntnu-smil/sla-p1 \
    --output_dir ./finetuned_qwen2audio \
    --loss_type FA \
    --num_train_epochs 2 \
    --per_device_batch_size 8
```

此程式僅為研究用途，不會實際下載或發佈任何資料或模型權重。若要在本地執行，請確保
具備適當的計算資源（建議使用具備較大顯存的 GPU），並安裝 `transformers>=4.40.0`、
`peft`、`datasets`、`torch`、`librosa` 等套件。
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (AutoProcessor, Qwen2AudioForConditionalGeneration,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, TaskType


def set_seed(seed: int = 42) -> None:
    """為可重複實驗設定隨機種子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_score_to_letter(score: float) -> str:
    """將數值分數 (1–6) 映射至 CEFR 等級字母 A–F。最高分 6 對應 A，最低分 1 對應 F。"""
    # 四捨五入到最接近的整數
    s_int = int(round(score))
    # 限定範圍
    s_int = max(1, min(6, s_int))
    mapping = {6: "A", 5: "B", 4: "C", 3: "D", 2: "E", 1: "F"}
    return mapping[s_int]


def map_letter_to_score(letter: str) -> float:
    """將 CEFR 字母轉回數值分數。"""
    reverse = {"A": 6.0, "B": 5.0, "C": 4.0, "D": 3.0, "E": 2.0, "F": 1.0}
    return reverse[letter]


def compute_metrics(preds: List[float], labels: List[float]) -> Dict[str, float]:
    """計算 RMSE、皮爾森相關與斯皮爾曼相關。"""
    preds = np.array(preds)
    labels = np.array(labels)
    rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
    # Pearson correlation
    if len(preds) > 1:
        pcc = float(np.corrcoef(preds, labels)[0, 1])
    else:
        pcc = 0.0
    # Spearman correlation
    if len(preds) > 1:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(preds, labels)
        src = float(rho)
    else:
        src = 0.0
    return {"rmse": rmse, "pcc": pcc, "src": src}


PROMPT = (
    "Predict the overall score of the speech using the options provided below.\n\n"
    "Option A: Can produce clear, smoothly flowing, well-structured discourse with an effective logical structure which helps the recipient to notice and remember significant points.\n"
    "Option B: Can produce extended, coherent stretches of language. May demonstrate natural, unforced errors in their speech.\n"
    "Option C: Shows reasonably accurate control over a range of frequently used grammatical structures. Speech may occasionally be disrupted by errors.\n"
    "Option D: Can produce simple connected text on topics that are familiar or of personal interest. Some hesitation and repetition may be present.\n"
    "Option E: Can produce simple descriptions or narrations. Fluency is limited and wording may be awkward.\n"
    "Option F: Can produce simple, mainly isolated phrases about people and places.\n"
    "Please select the letter corresponding to the most appropriate option, and only output that letter without any additional comments or text.\n\n"
    "Result is Option:"
)


@dataclass
class ScoreSample:
    """封裝單一訓練樣本"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    audio_values: List[torch.Tensor]
    label_letter: str
    numeric_score: float


class ScoreDataset(Dataset):
    """將 HuggingFace 資料集轉為可供微調的樣本。

    每筆資料可能包含較長的音訊，會切割為多個 30 秒內的片段，每片段皆視為一個樣本。
    """

    def __init__(self, hf_dataset: Any, processor: AutoProcessor, max_duration: float = 30.0):
        self.samples: List[ScoreSample] = []
        self.processor = processor
        sr = processor.feature_extractor.sampling_rate
        max_len = int(max_duration * sr)
        for item in hf_dataset:
            # 取得音訊陣列與取樣率。若 dataset 存放 audio 記錄為字典，這裡將存取 audio['array']。
            audio = item.get("audio")
            if audio is None:
                raise ValueError("Dataset example must contain an 'audio' field with audio data.")
            # audio 可能為 dict {"array": np.ndarray, "sampling_rate": int}
            if isinstance(audio, dict):
                array = audio.get("array")
                sr_example = audio.get("sampling_rate")
                if array is None or sr_example is None:
                    raise ValueError("Audio field must contain 'array' and 'sampling_rate'.")
                # 若取樣率不同，重採樣
                if sr_example != sr:
                    import librosa  # 延遲匯入

                    array = librosa.resample(array, orig_sr=sr_example, target_sr=sr)
            else:
                # 若 audio 欄直接是一維陣列，視為已經正確取樣率
                array = np.asarray(audio)
            # 分數欄位名稱可能為 "score" 或 "label"。嘗試讀取。
            score = None
            for key in ["score", "score_float", "label", "labels", "targets"]:
                if key in item:
                    score = float(item[key])
                    break
            if score is None:
                raise ValueError("Dataset example must contain a numeric 'score' field.")
            # 切分長度超過 30s 的音訊
            start = 0
            total_length = len(array)
            while start < total_length:
                end = min(start + max_len, total_length)
                segment = array[start:end]
                # 填補至 30s 長度以避免批次中長度不一，使用 0 填充
                if len(segment) < max_len:
                    pad_width = max_len - len(segment)
                    segment = np.pad(segment, (0, pad_width), mode="constant")
                letter = map_score_to_letter(score)
                # 構建對話
                conversation = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": segment.tolist()},
                    ]},
                    {"role": "assistant", "content": letter},
                ]
                # 使用 apply_chat_template 生成 chat 文字，包含 <im_start> 等特殊符號
                text = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                # 產生 token 與 audio feature
                processed = processor(text=text, audios=[segment], return_tensors="pt", padding=True)
                input_ids = processed.input_ids.squeeze(0)
                attention_mask = processed.attention_mask.squeeze(0)
                # audio_values 為 list，取第一個
                audio_values = processed.audios
                # 儲存樣本
                self.samples.append(
                    ScoreSample(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        audio_values=audio_values,
                        label_letter=letter,
                        numeric_score=score,
                    )
                )
                start += max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ScoreSample:
        return self.samples[idx]


def collate_fn(batch: List[ScoreSample], processor: AutoProcessor) -> Dict[str, Any]:
    """將批次樣本組合，使用 processor 進行 padding。"""
    input_ids = [b.input_ids for b in batch]
    attention_masks = [b.attention_mask for b in batch]
    # 音訊值仍以 list 儲存，processor 會自動堆疊
    audios = [b.audio_values[0] for b in batch]
    # 使用 processor 進行 padding
    padded = processor.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks, "audios": audios},
        return_tensors="pt"
    )
    labels = [b.label_letter for b in batch]
    numeric_scores = torch.tensor([b.numeric_score for b in batch], dtype=torch.float)
    return {
        "input_ids": padded.input_ids,
        "attention_mask": padded.attention_mask,
        "audios": padded.audios,
        "labels": labels,
        "scores": numeric_scores,
    }


class RegressionHead(nn.Module):
    """在最後一個 token 隱藏狀態上套用線性層以預測實值分數。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(hidden_state)
        return self.mlp(x)


def train(args: argparse.Namespace) -> None:
    """主訓練流程"""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 下載模型與處理器
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # 啟用 LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.to(device)
    model.train()
    # 如果使用迴歸，需要額外的 regression head
    reg_head: Optional[RegressionHead] = None
    if args.loss_type.lower() == "reg":
        hidden_size = model.config.hidden_size
        reg_head = RegressionHead(hidden_size).to(device)
    # 讀取資料集
    train_ds = load_dataset(args.dataset_name, split="train")
    val_ds = load_dataset(args.dataset_name, split="validation") if args.val_split else None
    train_dataset = ScoreDataset(train_ds, processor, max_duration=args.max_duration)
    val_dataset = ScoreDataset(val_ds, processor, max_duration=args.max_duration) if val_ds is not None else None
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, processor),
        )
    # 優化器與學習率排程
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.num_train_epochs * math.ceil(len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    # 訓練迴圈
    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audios = batch["audios"].to(device)
            labels_letters = batch["labels"]
            scores = batch["scores"].to(device)
            # 前向傳播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audios=audios,
                output_hidden_states=True,
            )
            # 取最後一個 token 的 logits
            logits = outputs.logits[:, -1, :]
            # 建立損失
            if args.loss_type.lower() == "ce":
                # CE：計算交叉熵；將目標字母轉成 token id
                class_tokens = []
                target_indices = []
                for letter in ["A", "B", "C", "D", "E", "F"]:
                    token_id = processor.tokenizer.convert_tokens_to_ids(letter)
                    class_tokens.append(token_id)
                # 構建目標索引
                for letter in labels_letters:
                    token_id = processor.tokenizer.convert_tokens_to_ids(letter)
                    # 計算在 class_tokens 中的索引
                    target_indices.append(class_tokens.index(token_id))
                target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
                # 從 logits 中取出對應 class token 的 logits
                class_logits = logits[:, class_tokens]
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(class_logits, target_tensor)
                preds = class_logits.argmax(dim=-1)
                pred_scores = [map_letter_to_score(["A", "B", "C", "D", "E", "F"][int(p)]) for p in preds.cpu().numpy()]
            elif args.loss_type.lower() == "fa":
                # FA：計算 fair average 的 MSE 損失
                class_tokens = []
                for letter in ["A", "B", "C", "D", "E", "F"]:
                    token_id = processor.tokenizer.convert_tokens_to_ids(letter)
                    class_tokens.append(token_id)
                class_logits = logits[:, class_tokens]
                probs = torch.nn.functional.softmax(class_logits, dim=-1)
                # 根據分數映射值計算加權平均
                value_tensor = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], device=device)
                pred_values = (probs * value_tensor).sum(dim=-1)
                loss = torch.nn.functional.mse_loss(pred_values, scores)
                pred_scores = pred_values.detach().cpu().tolist()
            elif args.loss_type.lower() == "reg":
                assert reg_head is not None
                # 取最後一個 token 的隱藏狀態，使用 regression head 預測
                hidden_state = outputs.hidden_states[-1][:, -1, :]
                pred_values = reg_head(hidden_state).squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred_values, scores)
                pred_scores = pred_values.detach().cpu().tolist()
            else:
                raise ValueError(f"Unknown loss type {args.loss_type}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
            global_step += 1
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{args.num_train_epochs}, training loss: {avg_loss:.4f}")
        # 驗證
        if val_loader is not None:
            model.eval()
            all_preds: List[float] = []
            all_labels: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    audios = batch["audios"].to(device)
                    labels_letters = batch["labels"]
                    scores = batch["scores"].to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        audios=audios,
                        output_hidden_states=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    if args.loss_type.lower() == "ce":
                        class_tokens = [processor.tokenizer.convert_tokens_to_ids(l) for l in ["A", "B", "C", "D", "E", "F"]]
                        class_logits = logits[:, class_tokens]
                        probs = torch.nn.functional.softmax(class_logits, dim=-1)
                        pred_classes = probs.argmax(dim=-1)
                        pred_scores = [map_letter_to_score(["A", "B", "C", "D", "E", "F"][int(p)]) for p in pred_classes.cpu().numpy()]
                    elif args.loss_type.lower() == "fa":
                        class_tokens = [processor.tokenizer.convert_tokens_to_ids(l) for l in ["A", "B", "C", "D", "E", "F"]]
                        class_logits = logits[:, class_tokens]
                        probs = torch.nn.functional.softmax(class_logits, dim=-1)
                        value_tensor = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], device=device)
                        pred_vals = (probs * value_tensor).sum(dim=-1)
                        pred_scores = pred_vals.detach().cpu().tolist()
                    else:  # reg
                        hidden_state = outputs.hidden_states[-1][:, -1, :]
                        pred_vals = reg_head(hidden_state).squeeze(-1)
                        pred_scores = pred_vals.detach().cpu().tolist()
                    all_preds.extend(pred_scores)
                    all_labels.extend(scores.cpu().tolist())
            metrics = compute_metrics(all_preds, all_labels)
            print(f"Validation metrics: RMSE={metrics['rmse']:.4f}, PCC={metrics['pcc']:.4f}, SRC={metrics['src']:.4f}")
    # 儲存模型和參數
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # 儲存 LoRA 權重
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        if reg_head is not None:
            torch.save(reg_head.state_dict(), os.path.join(args.output_dir, "reg_head.pt"))
        print(f"Model saved to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune Qwen2Audio for speech scoring")
    parser.add_argument("--dataset_name", type=str, default="ntnu-smil/sla-p1", help="HuggingFace 資料集名稱，例如 ntnu-smil/sla-p1")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="預訓練模型名稱或路徑")
    parser.add_argument("--output_dir", type=str, default="./output", help="模型儲存路徑")
    parser.add_argument("--loss_type", type=str, choices=["CE", "FA", "Reg", "ce", "fa", "reg"], default="FA", help="損失函式類型：CE、FA 或 Reg")
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="單 GPU 的批次大小")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="訓練 epoch 數")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="學習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="權重衰減")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="學習率排程 warmup 比例")
    parser.add_argument("--max_duration", type=float, default=30.0, help="單片段最大長度（秒）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--use_lora", action="store_true", help="是否使用 LoRA 微調")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 階數 r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--val_split", action="store_true", help="是否在資料集包含驗證集。若為 False 則不進行驗證。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)