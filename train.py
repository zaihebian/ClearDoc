import os, io, random, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from datasets import load_dataset
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score

# ---- Config (env overridable) ----
MODEL_ID = os.getenv("MODEL_ID", "facebook/convnextv2-tiny-1k-224")
OUT_DIR  = os.getenv("OUT_DIR", "./model")
EPOCHS   = int(os.getenv("EPOCHS", "3"))
BSZ      = int(os.getenv("BSZ", "32"))
LR       = float(os.getenv("LR", "3e-4"))
SEED     = int(os.getenv("SEED", "42"))
P_BAD    = float(os.getenv("P_BAD", "0.5"))   # chance to create a BAD sample

random.seed(SEED)
np.random.seed(SEED)

def degrade_one(img: Image.Image) -> (Image.Image, str):
    """Return degraded image + reason."""
    op = random.choice(["blur", "motion", "skew", "lowlight", "jpeg"])
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5))), "blur"
    if op == "motion":
        # simulate by strong directional blur (box + small rotate)
        k = random.choice([3,5])
        img2 = img.resize((img.width, img.height))  # noop copy
        return img2.filter(ImageFilter.BoxBlur(k)).rotate(random.uniform(-2,2), expand=False), "motion"
    if op == "skew":
        ang = random.uniform(-8, 8)
        return img.rotate(ang, expand=False, fillcolor="white"), "skew"
    if op == "lowlight":
        return ImageEnhance.Brightness(img).enhance(0.45), "low-light"
    # jpeg heavy compression
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(25, 45), optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), "compression"

def build_datasets():
    ds = load_dataset("nielsr/funsd")  # splits: train/test
    # ensure PIL images
    ds = ds.cast_column("image", ds["train"].features["image"])

    # Map labels: 0=GOOD, 1=BAD. We inject BAD on-the-fly in transform.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    def make_transform(split):
        rng = random.Random(SEED + (0 if split=="train" else 1))
        def _tf(ex):
            img: Image.Image = ex["image"].convert("RGB")
            if split == "train" and rng.random() < P_BAD:
                img, _ = degrade_one(img)
                label = 1
            else:
                label = 0
            out = processor(images=img, return_tensors="pt")
            ex = {"pixel_values": out["pixel_values"].squeeze(0), "labels": label}
            return ex
        return _tf

    ds_train = ds["train"].with_transform(make_transform("train"))
    # Use test as val/test (no synthetic BAD there → real-world check)
    ds_val   = ds["test"].with_transform(make_transform("val"))
    return ds_train, ds_val, processor

def main():
    ds_train, ds_val, processor = build_datasets()

    id2label = {0: "GOOD", 1: "BAD"}
    label2id = {"GOOD": 0, "BAD": 1}

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, num_labels=2, id2label=id2label, label2id=label2id
    )

    def metrics_fn(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)
        return {
            "acc": accuracy_score(labels, preds),
            "f1":  f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=BSZ,
        per_device_eval_batch_size=BSZ,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="acc",
        remove_unused_columns=False,
        fp16=True,
        logging_steps=20,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=processor,
        compute_metrics=metrics_fn,
    )

    trainer.train()
    print("Val:", trainer.evaluate(ds_val))

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(f"Saved model to {OUT_DIR}")

if __name__ == "__main__":
    main()
