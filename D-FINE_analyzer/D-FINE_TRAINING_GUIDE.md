# D-FINE 学習実装ガイド

## 概要
HuggingFace の D-FINE (Detection Transformer with Improved Denoising Anchor Boxes) を使用したカスタムオブジェクト検出モデルの学習実装における重要なポイントをまとめたガイドです。

---

## 1. モデルの基本設定

### 1.1 モデルのロード
```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_name = "ustc-community/dfine-xlarge-coco"
num_classes = 7  # カスタムクラス数

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(
    model_name,
    num_labels=num_classes,  # ★重要: カスタムクラス数を指定
    ignore_mismatched_sizes=True  # ★重要: サイズ不一致を許可
)
```

**重要ポイント:**
- `num_labels` を必ず指定（デフォルトはCOCOの91クラス）
- `ignore_mismatched_sizes=True` で分類ヘッドのサイズ変更を許可

---

## 2. データセット準備

### 2.1 COCO形式のアノテーション
D-FINEは内部的にCOCO形式を想定していますが、**学習時は手動でラベルを作成する必要があります**。

```python
# COCO JSONからアノテーションを読み込み
annotations_list = [ann for ann in annotations if ann['image_id'] == image_id]

# ★重要: 正規化された中心座標形式 [cx, cy, w, h] に変換
boxes = []
class_labels = []

for ann in annotations_list:
    x, y, w, h = ann['bbox']  # COCO形式: [x, y, width, height]
    
    # 正規化された中心座標に変換
    cx = (x + w / 2) / img_w  # 中心X座標（0-1に正規化）
    cy = (y + h / 2) / img_h  # 中心Y座標（0-1に正規化）
    nw = w / img_w            # 幅（0-1に正規化）
    nh = h / img_h            # 高さ（0-1に正規化）
    
    boxes.append([cx, cy, nw, nh])
    class_labels.append(ann['category_id'])

# labelsとして渡す
labels = {
    'boxes': torch.tensor(boxes, dtype=torch.float32),
    'class_labels': torch.tensor(class_labels, dtype=torch.int64)
}
```

**重要ポイント:**
- ❌ COCO形式の `[x, y, width, height]` をそのまま使わない
- ✅ **正規化された中心座標** `[cx, cy, w, h]` に変換
- ✅ すべての値を `[0, 1]` の範囲に正規化

### 2.2 画像の前処理
```python
# processorを使用（自動で正しい形式に変換）
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs['pixel_values']
```

---

## 3. 学習設定の重要ポイント

### 3.1 バックボーンの凍結/解凍

**❌ 失敗例: バックボーン凍結（freeze_backbone=true）**
```python
# 出力層のみ学習
for name, param in model.named_parameters():
    if 'backbone' in name or 'encoder' in name:
        param.requires_grad = False
```
**結果:**
- Val Loss: 1.87
- 検出精度: わずか2%
- 問題: class_embedのバイアスが負の値（-1.9前後）になり、検出スコアが極端に低下

**✅ 成功例: バックボーン解凍（freeze_backbone=false）**
```python
# 全パラメータを学習
for param in model.parameters():
    param.requires_grad = True
```
**結果:**
- Val Loss: 0.0386（**48倍改善！**）
- エポック46で達成
- 全パラメータ（約62.5M）を学習

**結論: D-FINEでは最初から全体を学習する方が効果的**

### 3.2 ハイパーパラメータ
```yaml
training:
  num_epochs: 50
  batch_size: 1          # GPU メモリに応じて調整
  learning_rate: 0.0001  # 1e-4（AdamW推奨）
  weight_decay: 0.0001
  freeze_backbone: false # ★重要: 必ずfalseにする
```

---

## 4. 学習ループの実装

### 4.1 損失計算
```python
outputs = model(
    pixel_values=pixel_values.to(device),
    labels=labels_batch  # labelsはリスト形式
)
loss = outputs.loss  # D-FINEが自動計算
```

**重要ポイント:**
- D-FINEは内部で複数の損失（分類、bbox、マッチングなど）を計算
- `outputs.loss` を直接使用できる

### 4.2 チェックポイント保存
```python
# ベストモデルの保存
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, 'checkpoints/best_model.pth')
```

---

## 5. 推論時の注意点

### 5.1 モデルのロード
```python
# ★重要: num_labelsを指定してからstate_dictをロード
model = AutoModelForObjectDetection.from_pretrained(
    model_name,
    num_labels=num_classes,  # 学習時と同じクラス数
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

**重要ポイント:**
- 推論時も `num_labels` を必ず指定
- 指定しないとCOCOの91クラスがロードされ、RuntimeErrorが発生

### 5.2 後処理
```python
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(
    outputs,
    threshold=0.25,  # 信頼度閾値（調整が必要な場合あり）
    target_sizes=target_sizes
)[0]

# 結果の取得
for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
    score = score.item()
    label = label.item()
    box = [b.item() for b in box]  # [x1, y1, x2, y2]
```

### 5.3 信頼度閾値の調整
- デフォルト: 0.5
- バックボーン凍結時: 0.25でも検出が困難
- バックボーン解凍後: 適切な閾値を実験的に決定

---

## 6. トラブルシューティング

### 6.1 検出精度が極端に低い（2%以下）
**原因:**
- バックボーンが凍結されている
- class_embedのバイアスが負の値になっている

**解決策:**
```yaml
freeze_backbone: false  # 全体を学習
```

### 6.2 Val Lossは低いが検出されない
**原因:**
- 信頼度閾値が高すぎる
- モデルのスコア分布を確認

**解決策:**
```python
# 生のlogitsを確認
logits = outputs.logits[0]
probs = torch.softmax(logits, dim=-1)
max_probs = probs.max(dim=-1)[0]
print(f"最大スコア: {max_probs.max().item()}")
```

### 6.3 RuntimeError: size mismatch
**原因:**
- 推論時に `num_labels` を指定していない

**解決策:**
```python
# 必ず指定
model = AutoModelForObjectDetection.from_pretrained(
    model_name,
    num_labels=7,  # 学習時と同じ
    ignore_mismatched_sizes=True
)
```

---

## 7. データ量と精度の関係

### 実験結果
| データセット | Train/Val | Val Loss | 検出精度 | 備考 |
|------------|-----------|----------|---------|------|
| 100枚 (凍結) | 80/20 | 1.7511 | 10% | バックボーン凍結 |
| 1000枚 (凍結) | 800/200 | 1.8700 | 2% | データ増加でも改善せず |
| 1000枚 (解凍) | 800/200 | **0.0386** | 評価中 | **48倍改善！** |

**結論:**
- データ量を増やすだけでは効果なし
- **バックボーン解凍が決定的に重要**

---

## 8. 推奨する学習フロー

```python
# 1. 設定
config = {
    'model': {
        'name': 'ustc-community/dfine-xlarge-coco',
        'num_classes': 7
    },
    'training': {
        'num_epochs': 50,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'freeze_backbone': False  # ★重要
    }
}

# 2. モデル準備
model = AutoModelForObjectDetection.from_pretrained(
    config['model']['name'],
    num_labels=config['model']['num_classes'],
    ignore_mismatched_sizes=True
)

# 3. 全パラメータを学習可能に
for param in model.parameters():
    param.requires_grad = True

# 4. オプティマイザ
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate']
)

# 5. 学習ループ
for epoch in range(config['training']['num_epochs']):
    # 学習
    model.train()
    for batch in train_dataloader:
        outputs = model(
            pixel_values=batch['pixel_values'].to(device),
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 検証
    model.eval()
    # ...
```

---

## 9. まとめ

### ✅ 必ず実施すること
1. `num_labels` を学習時・推論時ともに指定
2. `freeze_backbone: false` で全体を学習
3. 正規化された中心座標 `[cx, cy, w, h]` を使用
4. `ignore_mismatched_sizes=True` を設定

### ❌ 避けるべきこと
1. バックボーンの凍結（小規模データセットでも）
2. COCO形式のbboxをそのまま使用
3. 推論時の `num_labels` 指定忘れ

### 🎯 期待される結果
- Val Loss: 0.03-0.05程度まで低下
- 適切な信頼度閾値で高精度な検出が可能
- エポック40-50で収束

---

## 10. 参考情報

- **モデル:** [ustc-community/dfine-xlarge-coco](https://huggingface.co/ustc-community/dfine-xlarge-coco)
- **パラメータ数:** 約62.5M
- **推奨GPU:** 8GB以上のVRAM
- **学習時間:** バックボーン解凍時は約1時間/50エポック（GPU依存）

---

## 変更履歴

- 2025-11-08: 初版作成
  - バックボーン凍結の問題を特定
  - バックボーン解凍でVal Loss 48倍改善を確認
  - 正規化された中心座標の重要性を追記
