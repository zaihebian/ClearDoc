import os, io, random, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from datasets import load_dataset
from transformers import (AutoImageProcessor, ConvNextV2ForImageClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import torch


# ---- Config (env overridable) ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "ClearDoc")
MODEL_ID = os.getenv("MODEL_ID", "facebook/convnextv2-tiny-1k-224")
OUT_DIR  = os.getenv("OUT_DIR", "./model")
EPOCHS   = int(os.getenv("EPOCHS", "1"))
BSZ      = int(os.getenv("BSZ", "32"))
LR       = float(os.getenv("LR", "3e-4"))
SEED     = int(os.getenv("SEED", "42"))
P_BAD    = float(os.getenv("P_BAD", "0.5"))   # chance to create a BAD sample

random.seed(SEED)
np.random.seed(SEED)


# --- Custom collator ---
def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


# --- Image degradation ---
def degrade_one(img: Image.Image) -> (Image.Image, str):
    op = random.choice(["blur", "motion", "skew", "lowlight", "jpeg"])
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5))), "blur"
    if op == "motion":
        k = random.choice([3, 5])
        img2 = img.resize((img.width, img.height))
        return img2.filter(ImageFilter.BoxBlur(k)).rotate(random.uniform(-2, 2), expand=False), "motion"
    if op == "skew":
        ang = random.uniform(-8, 8)
        return img.rotate(ang, expand=False, fillcolor="white"), "skew"
    if op == "lowlight":
        return ImageEnhance.Brightness(img).enhance(0.45), "low-light"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(25, 45), optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), "compression"


# --- Dataset builder ---
def build_datasets():
    ds = load_dataset("nielsr/funsd")  # splits: train/test
    ds = ds.cast_column("image", ds["train"].features["image"])  # ensure PIL

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    def make_transform(split):
        rng = random.Random(SEED + (0 if split == "train" else 1))

        def _tf(batch):
            images, labels = [], []
            for img in batch["image"]:
                img = img.convert("RGB")
                # inject BAD for both train and val (so metrics are meaningful)
                if rng.random() < P_BAD:
                    img, _ = degrade_one(img)
                    label = 1
                else:
                    label = 0
                images.append(img)
                labels.append(label)

            proc = processor(images=images, return_tensors="pt")
            return {
                "pixel_values": proc["pixel_values"],
                "labels": np.array(labels, dtype=np.int64),
            }

        return _tf

    ds_train = ds["train"].with_transform(make_transform("train"))
    ds_val   = ds["test"].with_transform(make_transform("val"))
    return ds_train, ds_val, processor


# --- Trainer that drops unsupported kwarg ---
class CleanTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if "num_items_in_batch" in inputs:
            inputs = dict(inputs)
            inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# --- Main training loop ---
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params({
            "model_id": MODEL_ID,
            "epochs": EPOCHS,
            "batch_size": BSZ,
            "lr": LR,
            "seed": SEED,
            "p_bad": P_BAD,
        })

        ds_train, ds_val, processor = build_datasets()

        id2label = {0: "GOOD", 1: "BAD"}
        label2id = {"GOOD": 0, "BAD": 1}

        model = ConvNextV2ForImageClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        # --- freeze backbone, train only the head ---
        for param in model.convnextv2.parameters():
            param.requires_grad = False

        def metrics_fn(pred):
            logits, labels = pred
            preds = logits.argmax(axis=1)
            return {
                "acc": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="macro"),
            }

        args = TrainingArguments(
            output_dir="./out",
            per_device_train_batch_size=BSZ,
            per_device_eval_batch_size=BSZ,
            learning_rate=LR,
            num_train_epochs=EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="acc",
            remove_unused_columns=False,
            fp16=False,
            logging_steps=20,
            report_to=[],
            seed=SEED,
        )

        trainer = CleanTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=metrics_fn,
            data_collator=collate_fn,
        )

        trainer.train()
        eval_metrics = trainer.evaluate(ds_val)
        print("Val:", eval_metrics)

        mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})

        os.makedirs(OUT_DIR, exist_ok=True)
        model.save_pretrained(OUT_DIR)
        processor.save_pretrained(OUT_DIR)
        mlflow.log_artifacts(OUT_DIR, artifact_path="model")

        print(f"Saved model to {OUT_DIR}")


if __name__ == "__main__":
    main()
