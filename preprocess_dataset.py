import cv2
import mediapipe as mp
import os
from tqdm import tqdm

SRC_DIR = "data/asl_alphabet_train/asl_alphabet_train"
DST_DIR = "data/processed_asl"
IMG_SIZE = 224

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

os.makedirs(DST_DIR, exist_ok=True)

for label in os.listdir(SRC_DIR):
    src_label_dir = os.path.join(SRC_DIR, label)
    dst_label_dir = os.path.join(DST_DIR, label)
    os.makedirs(dst_label_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(src_label_dir), desc=f"Processing {label}"):
        img_path = os.path.join(src_label_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        h, w, _ = img.shape
        lm = result.multi_hand_landmarks[0]

        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]

        x1, x2 = max(0, min(xs)-20), min(w, max(xs)+20)
        y1, y2 = max(0, min(ys)-20), min(h, max(ys)+20)

        hand = img[y1:y2, x1:x2]
        if hand.size == 0:
            continue

        hand = cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
        save_path = os.path.join(dst_label_dir, img_name)
        cv2.imwrite(save_path, hand)

print("✅ Dataset preprocessing complete")
