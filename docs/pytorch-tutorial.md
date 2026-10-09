# Learn PyTorch with Wind-Up

PyTorch provides tensors, automatic differentiation, and modules with trainable parameters. This project turns those pieces into a complete classifier without requiring a large language model.

## 1. A tensor is a shaped array

```python
import torch

ids = torch.tensor([12, 7, 12, 40, 3], dtype=torch.long)
offsets = torch.tensor([0, 3], dtype=torch.long)
```

The first text has feature IDs `[12, 7, 12]`; the second has `[40, 3]`. `offsets` tells PyTorch where each bag starts. IDs are integers: they select embedding rows rather than values to differentiate.

## 2. Embeddings are learned vectors

```python
from language_detector import build_model

model = build_model(languages=3, buckets=64, embedding_dim=8, hidden_dim=16, arch="mlp")
logits = model(ids, offsets)
print(logits.shape)  # torch.Size([2, 3])
```

Each of 64 embedding rows holds eight trainable numbers. Mean pooling combines each input's selected rows. A linear layer then maps eight numbers to three language scores. Scores are logits, not probabilities; their relative size determines the predicted class.

## 3. Loss measures the mistake

```python
targets = torch.tensor([0, 2], dtype=torch.long)
loss = torch.nn.CrossEntropyLoss()(logits, targets)
```

Targets are class indices: the first input belongs to class 0 and the second to class 2. Cross entropy encourages high scores for the right labels. Pass raw logits to it; adding softmax beforehand is unnecessary and changes the intended calculation.

## 4. Autograd finds how parameters affect loss

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
loss.backward()
print(model.classifier.weight.grad.shape)  # torch.Size([3, 8])
optimizer.step()
```

`backward()` computes derivatives through the linear layer and embedding mean. Adam uses those derivatives to update parameters. Gradients accumulate by default, so clear them before the next update. Only learning examples change weights; prediction does not.

## 5. Train a tiny demonstration

From the repository root:

```bash
.venv/bin/python language_detector.py train --data data/demo.csv --epochs 120 --model model-demo.pt
.venv/bin/python language_detector.py predict "Nie mogę się zalogować do konta." --model model-demo.pt
```

This small authored CSV demonstrates the training loop; it is not enough to build a dependable detector. It uses a smaller default feature configuration than the bundled release. Save it to `model-demo.pt` so the release weights remain intact.

A full training loop repeats: choose a batch, encode texts, compute logits, compute loss, clear gradients, call `backward`, and step the optimizer. Validation measures generalization on examples excluded from parameter updates. Training loss alone cannot show whether a model will work on new text.

## 6. Inference and evaluation

```python
model.eval()
with torch.inference_mode():
    predicted_indices = model(ids, offsets).argmax(dim=1)
```

`eval()` switches modules to evaluation behavior; `inference_mode()` disables gradient tracking and its overhead. This particular network has no module dropout, but using both makes inference intent clear. Feature dropout in the full trainer is applied explicitly to input feature IDs.

Read [training](training.md) next. Compare accuracy with macro recall and macro F1: a large class can dominate accuracy while smaller languages still perform poorly. Keep test data separate when selecting epochs or changing preprocessing.
