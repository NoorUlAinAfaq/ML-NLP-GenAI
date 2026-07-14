
import os
import cv2
import csv
import mediapipe as mp
from tqdm import tqdm

# ---- CONFIG ----
DATASET_PATH = "data/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "data/landmarks.csv"
LIMIT_PER_CLASS = None  # set to e.g. 200 for a quick sanity-check run, None for full dataset

# ---- SETUP MEDIAPIPE ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # True = treat each frame independently (better for still images)
    max_num_hands=1,              # dataset is single-hand signs
    min_detection_confidence=0.5
)

# ---- BUILD CSV HEADER ----
# 21 landmarks * 3 coords (x, y, z) = 63 columns, plus 1 label column
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

classes = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(classes)} classes: {classes}")

total_written = 0
total_skipped = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(cls_path):
            continue

        img_files = [
            fn for fn in os.listdir(cls_path)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if LIMIT_PER_CLASS:
            img_files = img_files[:LIMIT_PER_CLASS]

        class_written = 0
        class_skipped = 0

        for img_file in tqdm(img_files, desc=f"Class {cls}"):
            img_path = os.path.join(cls_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                class_skipped += 1
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            if not results.multi_hand_landmarks:
                class_skipped += 1
                continue

            hand_landmarks = results.multi_hand_landmarks[0]
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            row.append(cls)

            writer.writerow(row)
            class_written += 1

        total_written += class_written
        total_skipped += class_skipped
        print(f"  {cls}: {class_written} written, {class_skipped} skipped (no hand detected / bad file)")

hands.close()

print(f"\nDone. Total rows written: {total_written}")
print(f"Total skipped: {total_skipped}")
print(f"Saved to: {OUTPUT_CSV}")