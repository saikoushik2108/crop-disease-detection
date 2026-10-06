import pandas as pd
from pathlib import Path

def check_splits():
    train_df = pd.read_csv("data/processed/splits/train.csv")
    val_df = pd.read_csv("data/processed/splits/val.csv")
    test_df = pd.read_csv("data/processed/splits/test.csv")

    print("=== SPLIT SIZES ===")
    print(f"Train: {len(train_df)} ({len(train_df)/10407*100:.2f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/10407*100:.2f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/10407*100:.2f}%)")

    s_train = set(train_df["image_id"])
    s_val = set(val_df["image_id"])
    s_test = set(test_df["image_id"])

    print("\n=== LEAKAGE CHECK (SET INTERSECTIONS) ===")
    print(f"Overlap Train-Val:  {len(s_train & s_val)}")
    print(f"Overlap Train-Test: {len(s_train & s_test)}")
    print(f"Overlap Val-Test:   {len(s_val & s_test)}")

    dist = pd.DataFrame({
        "Train_Count": train_df["label"].value_counts(),
        "Train_Pct": (train_df["label"].value_counts(normalize=True)*100).round(2),
        "Val_Count": val_df["label"].value_counts(),
        "Val_Pct": (val_df["label"].value_counts(normalize=True)*100).round(2),
        "Test_Count": test_df["label"].value_counts(),
        "Test_Pct": (test_df["label"].value_counts(normalize=True)*100).round(2),
    })
    print("\n=== CLASS DISTRIBUTION ACROSS SPLITS ===")
    print(dist.to_string())

    var_dist = pd.DataFrame({
        "Train": train_df["variety"].value_counts(),
        "Val": val_df["variety"].value_counts(),
        "Test": test_df["variety"].value_counts(),
    }).fillna(0).astype(int)
    print("\n=== VARIETY DISTRIBUTION ACROSS SPLITS ===")
    print(var_dist.to_string())

    print("\n=== AGE SUMMARY ACROSS SPLITS ===")
    print(f"Train age: mean={train_df['age'].mean():.2f}, std={train_df['age'].std():.2f}, min={train_df['age'].min()}, max={train_df['age'].max()}")
    print(f"Val age:   mean={val_df['age'].mean():.2f}, std={val_df['age'].std():.2f}, min={val_df['age'].min()}, max={val_df['age'].max()}")
    print(f"Test age:  mean={test_df['age'].mean():.2f}, std={test_df['age'].std():.2f}, min={test_df['age'].min()}, max={test_df['age'].max()}")

    # Compute inverse class frequencies on Train for class weighting
    class_counts = train_df["label"].value_counts().sort_index()
    total_train = len(train_df)
    n_classes = len(class_counts)
    weights = total_train / (n_classes * class_counts)
    norm_weights = weights / weights.sum() * n_classes
    print("\n=== INVERSE FREQUENCY CLASS WEIGHTS (TRAIN ONLY) ===")
    weight_df = pd.DataFrame({
        "Class": class_counts.index,
        "Train_Count": class_counts.values,
        "Inverse_Weight": weights.values.round(4),
        "Normalized_Weight": norm_weights.values.round(4)
    })
    print(weight_df.to_string(index=False))

if __name__ == "__main__":
    check_splits()
