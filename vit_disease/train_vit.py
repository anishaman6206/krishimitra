import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Load feature extractor from pretrained ViT
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Transform for dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean,
                         std=feature_extractor.image_std),
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

# Get labels
num_labels = len(train_dataset.classes)
id2label = {i: c for i, c in enumerate(train_dataset.classes)}
label2id = {c: i for i, c in enumerate(train_dataset.classes)}

# Load model
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Collate function
def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-plant-disease",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

# Train and save model + feature extractor
trainer.train()
trainer.save_model("./vit-plant-disease")
feature_extractor.save_pretrained("./vit-plant-disease")
print("Training complete. Model and feature extractor saved.")
