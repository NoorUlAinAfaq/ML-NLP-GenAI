import os
import cv2
import matplotlib.pyplot as plt
import random

# Path to dataset
DATASET_PATH = "data/asl_alphabet_train/asl_alphabet_train"

# Check all classes
classes = os.listdir(DATASET_PATH)
classes.sort()
print(f"✅ Total classes found: {len(classes)}")
print(f"📋 Classes: {classes}")

# Count images per class
print("\n📊 Images per class:")
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    count = len(os.listdir(cls_path))
    print(f"  {cls}: {count} images")

# Show sample images from each class
fig, axes = plt.subplots(3, 9, figsize=(18, 7))
axes = axes.flatten()

for i, cls in enumerate(classes[:27]):  # show first 27 classes
    cls_path = os.path.join(DATASET_PATH, cls)
    img_file = random.choice(os.listdir(cls_path))
    img_path = os.path.join(cls_path, img_file)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[i].imshow(img)
    axes[i].set_title(cls, fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("ASL Alphabet Dataset — Sample Images", fontsize=16)
plt.tight_layout()
plt.savefig("notebooks/dataset_preview.png")
plt.show()
print("\n✅ Dataset preview saved to notebooks/dataset_preview.png")