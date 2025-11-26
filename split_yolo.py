import os
import shutil
import random

# --- CONFIGURATION ---
# Source folders
IMAGES_SOURCE_DIR = "raw_data/images"
LABELS_SOURCE_DIR = "raw_data/labels"

# Destination folder
DEST_DIR = "dataset"

# Split ratio (0.8 = 80% train, 20% val)
SPLIT_RATIO = 0.8
# ---------------------

def split_dataset():
    # 1. Check if sources exist
    if not os.path.exists(IMAGES_SOURCE_DIR) or not os.path.exists(LABELS_SOURCE_DIR):
        print("❌ Error: Could not find 'raw_data/images' or 'raw_data/labels'")
        return

    # 2. Get list of all images
    images = [f for f in os.listdir(IMAGES_SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 3. Shuffle them randomly
    random.shuffle(images)

    # 4. Calculate split
    split_point = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # 5. Create YOLO folder structure
    for category in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, 'images', category), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', category), exist_ok=True)

    print(f"Found {len(images)} images. Splitting: {len(train_images)} Train, {len(val_images)} Val.")

    # 6. Move files function
    def move_files(file_list, split_type):
        for img_name in file_list:
            # Define source paths
            src_img = os.path.join(IMAGES_SOURCE_DIR, img_name)
            
            # Find corresponding label (change .jpg to .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(LABELS_SOURCE_DIR, label_name)

            # Define destination paths
            dst_img = os.path.join(DEST_DIR, 'images', split_type, img_name)
            dst_label = os.path.join(DEST_DIR, 'labels', split_type, label_name)

            # Copy Image
            shutil.copy(src_img, dst_img)

            # Copy Label (only if it exists)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"⚠️ Warning: No label found for {img_name}")

    # 7. Execute Move
    print("Copying Train files...")
    move_files(train_images, 'train')
    
    print("Copying Val files...")
    move_files(val_images, 'val')

    print(f"\n✅ SUCCESS! Dataset organized in '{DEST_DIR}/'")

if __name__ == "__main__":
    split_dataset()