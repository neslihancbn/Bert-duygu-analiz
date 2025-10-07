import os
import time
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# 1. Cihaz ve Yapılandırma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Deterministik işlemler için
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


# 2. Veri Temizleme Fonksiyonu
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # HTML temizleme
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama temizleme
    return text.strip().lower()


# 3. Veri Yükleme ve İşleme
data_path = os.path.join("data", "NLPlabeledData.tsv")
df = pd.read_csv(data_path, delimiter="\t", quoting=3)
df = df[['review', 'sentiment']].dropna()
df['cleaned_review'] = df['review'].apply(clean_text)

# Sınıf dengesi kontrolü
print("Sınıf Dağılımı:")
print(df['sentiment'].value_counts())

# 4. Veri Bölme
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_review'].tolist(),
    df['sentiment'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

# 5. Tokenizer ve DataCollator
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 6. Dataset Sınıfı
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 7. DataLoader'lar
train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    collate_fn=data_collator
)

# 8. Model ve Optimizer
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_model.pth"
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 9. Checkpoint Yükleme
start_epoch = 0
best_f1 = 0
if os.path.exists(LAST_CHECKPOINT_PATH):
    print("⏳ Son checkpoint yükleniyor...")
    checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']
    print(f"✅ Eğitime {start_epoch}. epoch'tan devam edilecek")


# 10. Early Stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# 11. Eğitim Fonksiyonu
def train_epoch(model, loader, gradient_accumulation=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Eğitim")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation
        loss.backward()

        if (i + 1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


# 12. Değerlendirme Fonksiyonu
def evaluate(model, loader):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Değerlendirme"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='weighted'
    )
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, precision, recall, f1, preds, true_labels


# 13. Eğitim Döngüsü
if start_epoch < 10:
    early_stopping = EarlyStopping()
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, 10):
        print(f"\nEpoch {epoch + 1}/10")

        # Eğitim
        train_loss = train_epoch(model, train_loader)
        train_losses.append(train_loss)

        # Değerlendirme
        val_loss, acc, prec, rec, f1, _, _ = evaluate(model, val_loader)
        val_losses.append(val_loss)

        print(f"Eğitim Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Checkpoint Kaydet
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_f1': best_f1,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, LAST_CHECKPOINT_PATH)
        print(f"💾 Checkpoint kaydedildi: {LAST_CHECKPOINT_PATH}")

        # En iyi modeli kaydet
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("🎉 Yeni en iyi model kaydedildi!")

        # Early Stopping
        if early_stopping(val_loss):
            print("🚨 Early Stopping!")
            break

    # Loss grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Eğitim Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Değişimi')
    plt.show()
else:
    print("✅ Tüm epoch'lar tamamlandı")

# 14. Final Değerlendirme
print("\nFinal Değerlendirme Sonuçları:")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
val_loss, acc, prec, rec, f1, val_preds, val_true = evaluate(model, val_loader)

# Performans metrikleri
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

# Karışıklık matrisi
cm = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()