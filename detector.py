"""
♻️  Live Webcam Recycling Detector
===================================
Detects objects via webcam using YOLOv8 + Deep Learning
and classifies each as RECYCLABLE or NON-RECYCLABLE.

Platform : macOS Apple Silicon (M2) — uses MPS acceleration
Author   : Generated for VS Code / Mac M2
"""

import cv2
import torch
import numpy as np
import sys
import time
import platform
from collections import deque
from ultralytics import YOLO


RECYCLING_DATABASE = {
    # ── Recyclable ────────────────────────────────────────────────────────────
    "bottle":        (True,  "Plastic/Glass", "Rinse before recycling"),
    "wine glass":    (True,  "Glass",         "Place in glass recycling bin"),
    "cup":           (True,  "Paper/Plastic", "Remove lid, rinse if dirty"),
    "fork":          (False, "Metal",         "Check local metal recycling rules"),
    "knife":         (False, "Metal",         "Wrap safely before disposal"),
    "spoon":         (False, "Metal",         "Check local metal recycling rules"),
    "bowl":          (True,  "Ceramic/Glass", "Rinse and recycle if glass"),
    "banana":        (True,  "Organic",       "Compost — great for garden!"),
    "apple":         (True,  "Organic",       "Compost it"),
    "sandwich":      (False, "Food waste",    "Compost food waste"),
    "orange":        (True,  "Organic",       "Compost peels"),
    "broccoli":      (True,  "Organic",       "Compost"),
    "carrot":        (True,  "Organic",       "Compost"),
    "hot dog":       (False, "Food waste",    "General waste / compost"),
    "pizza":         (False, "Food waste",    "Greasy box → general waste"),
    "donut":         (False, "Food waste",    "General waste"),
    "cake":          (False, "Food waste",    "General waste"),
    "book":          (True,  "Paper",         "Paper recycling bin"),
    "clock":         (False, "Electronic",    "E-waste collection point"),
    "cell phone":    (False, "Electronic",    "E-waste / phone recycling"),
    "laptop":        (False, "Electronic",    "E-waste collection — never landfill"),
    "keyboard":      (False, "Electronic",    "E-waste recycling"),
    "mouse":         (False, "Electronic",    "E-waste recycling"),
    "remote":        (False, "Electronic",    "Remove batteries → e-waste"),
    "microwave":     (False, "Appliance",     "Appliance recycling center"),
    "oven":          (False, "Appliance",     "Appliance recycling center"),
    "toaster":       (False, "Appliance",     "Appliance recycling"),
    "refrigerator":  (False, "Appliance",     "Appliance recycling center"),
    "tv":            (False, "Electronic",    "E-waste — contains hazardous materials"),
    "scissors":      (False, "Metal",         "Check local metal recycling"),
    "teddy bear":    (False, "Textile",       "Donate if good condition"),
    "hair drier":    (False, "Electronic",    "E-waste collection"),
    "toothbrush":    (False, "Plastic",       "TerraCycle toothbrush program"),
    "vase":          (True,  "Glass/Ceramic", "Glass recycling if glass"),
    "potted plant":  (True,  "Organic",       "Compost soil and plant matter"),
    "chair":         (False, "Furniture",     "Donate or bulk waste pickup"),
    "couch":         (False, "Furniture",     "Donate or council collection"),
    "bed":           (False, "Furniture",     "Mattress recycling program"),
    "dining table":  (False, "Furniture",     "Donate or bulk waste"),
    "toilet":        (False, "Ceramic",       "Construction waste disposal"),
    "sink":          (False, "Ceramic",       "Construction waste disposal"),
    "backpack":      (False, "Textile",       "Donate or textile recycling"),
    "umbrella":      (False, "Mixed",         "Check local recycling options"),
    "handbag":       (False, "Textile",       "Donate if usable"),
    "tie":           (False, "Textile",       "Donate or textile recycling"),
    "suitcase":      (False, "Mixed",         "Donate or textile recycling"),
    "sports ball":   (False, "Rubber",        "Donate or sporting goods recycling"),
    "baseball bat":  (False, "Wood/Aluminum", "Check material — aluminum recyclable"),
    "skateboard":    (False, "Mixed",         "Donate or check local rules"),
    "bottle cap":    (True,  "Plastic/Metal", "Separate from bottle to recycle"),
    "paper":         (True,  "Paper",         "Paper recycling bin"),
    "cardboard":     (True,  "Cardboard",     "Flatten and recycle"),
    "newspaper":     (True,  "Paper",         "Paper recycling"),
    "magazine":      (True,  "Paper",         "Paper recycling"),
    "can":           (True,  "Aluminum",      "Rinse and recycle — very valuable!"),
    "tin":           (True,  "Metal",         "Rinse and recycle"),
    "jar":           (True,  "Glass",         "Rinse and recycle"),
    "bag":           (False, "Plastic",       "Plastic bag return at grocery store"),
    "person":        (False, "N/A",           "Hello there! 👋"),
    "car":           (False, "Metal/Mixed",   "Auto dismantlers recycle cars"),
    "truck":         (False, "Metal/Mixed",   "Auto recycling"),
    "bicycle":       (False, "Metal",         "Donate or scrap metal recycling"),
    "motorcycle":    (False, "Metal/Mixed",   "Auto recycling"),
    "airplane":      (False, "Metal",         "Aerospace recycling"),
    "bus":           (False, "Metal/Mixed",   "Fleet disposal programs"),
    "train":         (False, "Metal",         "Scrap metal"),
    "boat":          (False, "Mixed",         "Marine recycling programs"),
    "bench":         (False, "Wood/Metal",    "Donate or bulk collection"),
    "bird":          (False, "N/A",           "It's alive! Let it fly 🐦"),
    "cat":           (False, "N/A",           "It's a cat! 🐱"),
    "dog":           (False, "N/A",           "It's a dog! 🐕"),
    "horse":         (False, "N/A",           "It's alive! 🐴"),
    "sheep":         (False, "N/A",           "Baa! 🐑"),
    "cow":           (False, "N/A",           "Moo! 🐄"),
    "elephant":      (False, "N/A",           "Wow, big animal! 🐘"),
    "bear":          (False, "N/A",           "A bear! 🐻"),
    "zebra":         (False, "N/A",           "Striped! 🦓"),
    "giraffe":       (False, "N/A",           "Tall! 🦒"),
    "fire hydrant":  (False, "Metal",         "Municipal infrastructure"),
    "stop sign":     (False, "Metal",         "Municipal infrastructure"),
    "parking meter": (False, "Metal",         "Municipal infrastructure"),
    "traffic light": (False, "Electronic",    "Municipal e-waste"),
}

DEFAULT_ENTRY = (False, "Unknown", "Check with your local recycling authority")


#  VISUAL THEME

# BGR colors for OpenCV
COLOR = {
    "recycle":    (50,  205,  50),   # Green
    "no_recycle": (50,   50, 220),   # Red
    "warning":    (50,  200, 230),   # Yellow
    "white":      (255, 255, 255),
    "black":      (0,   0,   0),
    "panel_bg":   (20,  20,  30),
    "accent":     (200, 160,  40),   # Gold
    "gray":       (120, 120, 120),
}

#
#  HELPER: draw rounded rectangle

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=0.7):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# 
#  HELPER: draw label badge on bounding box
#
def draw_detection_box(frame, x1, y1, x2, y2, label, conf, recyclable, material, tip):
    color = COLOR["recycle"] if recyclable else COLOR["no_recycle"]
    icon  = "♻️ " if recyclable else "🚫 "
    status = "RECYCLABLE" if recyclable else "NOT RECYCLABLE"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    corner_len = 18
    thickness  = 3
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

    # Label background
    font       = cv2.FONT_HERSHEY_DUPLEX
    name_text  = f"{label.upper()}  {conf:.0%}"
    stat_text  = status
    mat_text   = f"[{material}]"

    (nw, nh), _ = cv2.getTextSize(name_text, font, 0.55, 1)
    (sw, sh), _ = cv2.getTextSize(stat_text, font, 0.45, 1)
    (mw, mh), _ = cv2.getTextSize(mat_text,  font, 0.38, 1)

    pad    = 8
    box_w  = max(nw, sw, mw) + pad * 2
    box_h  = nh + sh + mh + pad * 3 + 6
    bx     = max(0, x1)
    by     = max(0, y1 - box_h - 6)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Left color bar
    cv2.rectangle(frame, (bx, by), (bx + 4, by + box_h), color, -1)

    # Text
    ty = by + pad + nh
    cv2.putText(frame, name_text, (bx + 10, ty),
                font, 0.55, COLOR["white"], 1, cv2.LINE_AA)
    ty += sh + 4
    cv2.putText(frame, stat_text, (bx + 10, ty),
                font, 0.45, color, 1, cv2.LINE_AA)
    ty += mh + 4
    cv2.putText(frame, mat_text, (bx + 10, ty),
                font, 0.38, COLOR["gray"], 1, cv2.LINE_AA)


#  HELPER: draw HUD overlay (top-left panel)
# 
def draw_hud(frame, fps, device_name, recycle_count, no_recycle_count,
             total_objects, model_name, paused):
    h, w = frame.shape[:2]
    panel_w = 290
    panel_h = 200

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (12, 14, 22), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (40, 40, 60), 1)

    font  = cv2.FONT_HERSHEY_DUPLEX
    font2 = cv2.FONT_HERSHEY_SIMPLEX
    x, y  = 22, 38

    # Title
    cv2.putText(frame, "RECYCLING DETECTOR", (x, y), font, 0.52,
                COLOR["accent"], 1, cv2.LINE_AA)
    y += 22
    cv2.line(frame, (x, y), (x + panel_w - 24, y), (40, 40, 60), 1)
    y += 16

    # FPS + device
    fps_color = (50, 200, 50) if fps > 20 else (50, 150, 230) if fps > 10 else (50, 50, 220)
    cv2.putText(frame, f"FPS: {fps:5.1f}  |  {device_name}", (x, y),
                font2, 0.38, fps_color, 1, cv2.LINE_AA)
    y += 18

    cv2.putText(frame, f"Model: {model_name}", (x, y),
                font2, 0.37, COLOR["gray"], 1, cv2.LINE_AA)
    y += 18

    cv2.line(frame, (x, y), (x + panel_w - 24, y), (40, 40, 60), 1)
    y += 14

    # Detection counts
    cv2.putText(frame, "Objects detected:", (x, y),
                font2, 0.40, COLOR["white"], 1, cv2.LINE_AA)
    y += 20

    # Recyclable bar
    cv2.circle(frame, (x + 6, y - 5), 5, COLOR["recycle"], -1)
    cv2.putText(frame, f"Recyclable     {recycle_count:3d}", (x + 16, y),
                font2, 0.40, COLOR["recycle"], 1, cv2.LINE_AA)
    y += 18

    # Non-recyclable bar
    cv2.circle(frame, (x + 6, y - 5), 5, COLOR["no_recycle"], -1)
    cv2.putText(frame, f"Not Recyclable {no_recycle_count:3d}", (x + 16, y),
                font2, 0.40, COLOR["no_recycle"], 1, cv2.LINE_AA)
    y += 18

    # Total
    cv2.putText(frame, f"Total objects:  {total_objects:3d}", (x, y),
                font2, 0.40, COLOR["white"], 1, cv2.LINE_AA)

    # PAUSED badge
    if paused:
        ph, pw = 36, 110
        px = w // 2 - pw // 2
        py = 12
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (px, py), (px + pw, py + ph), (0, 80, 200), -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "  PAUSED", (px + 8, py + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR["white"], 1, cv2.LINE_AA)


# 
#  HELPER: draw tip bar (bottom of frame)

def draw_tip_bar(frame, tips_queue):
    if not tips_queue:
        return
    h, w = frame.shape[:2]
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 18, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), (40, 40, 60), 1)

    tip = tips_queue[0]
    cv2.putText(frame,
                f"💡  TIP: {tip}",
                (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                COLOR["warning"], 1, cv2.LINE_AA)



#  HELPER: draw controls bar

def draw_controls(frame):
    h, w = frame.shape[:2]
    controls = "[Q] Quit   [SPACE] Pause   [S] Screenshot   [+/-] Confidence"
    cv2.putText(frame, controls,
                (w // 2 - 240, h - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (80, 80, 100), 1, cv2.LINE_AA)



#  MAIN DETECTOR CLASS

class RecyclingDetector:

    def __init__(self, model_size="n", conf_threshold=0.45, camera_index=0):
        self.conf_threshold = conf_threshold
        self.camera_index   = camera_index
        self.paused         = False
        self.screenshot_n   = 0
        self.tips_queue     = deque(maxlen=3)
        self.tip_timer      = 0
        self.tip_interval   = 4.0   # seconds per tip

        print("\n" + "="*60)
        print("  ♻️   RECYCLING DETECTOR  —  Deep Learning + YOLOv8")
        print("="*60)

        # ── Device selection (M2 MPS > CUDA > CPU) ──────────────────────────
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                self.device      = "mps"
                self.device_name = "Apple M2 (MPS)"
            else:
                self.device      = "cpu"
                self.device_name = "CPU (MPS unavailable)"
        elif torch.cuda.is_available():
            self.device      = "cuda"
            self.device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            self.device      = "cpu"
            self.device_name = "CPU"

        print(f"  Device   : {self.device_name}")
        print(f"  Platform : {platform.system()} {platform.machine()}")

        # ── Load YOLOv8 model ────────────────────────────────────────────────
        model_name = f"yolov8{model_size}.pt"
        print(f"  Model    : YOLOv8-{model_size.upper()} ({model_name})")
        print("  Loading  : Downloading if first run (~6MB for nano)...")

        self.model      = YOLO(model_name)
        self.model_name = f"YOLOv8-{model_size.upper()}"

        # Warm up the model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=self.device)

        print(f"  Status   : ✅ Model loaded and warmed up")
        print(f"  Classes  : {len(self.model.names)} COCO classes")
        print(f"  Confidence threshold: {self.conf_threshold:.0%}")
        print("\n  Controls: Q=Quit  SPACE=Pause  S=Screenshot  +/-=Confidence")
        print("="*60 + "\n")

    def lookup(self, class_name):
        name = class_name.lower()
        return RECYCLING_DATABASE.get(name, DEFAULT_ENTRY)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"❌  Cannot open camera index {self.camera_index}.")
            print("    → Try: python detector.py --camera 1")
            sys.exit(1)

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷  Camera opened: {actual_w}×{actual_h}")

        # FPS tracking
        fps_buffer    = deque(maxlen=30)
        prev_time     = time.time()
        last_tip_time = time.time()
        paused_frame  = None

        while True:
            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n👋  Exiting — goodbye!")
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'⏸  Paused' if self.paused else '▶  Resumed'}")
            elif key == ord('s'):
                if paused_frame is not None:
                    fname = f"recycle_detection_{self.screenshot_n:04d}.jpg"
                    cv2.imwrite(fname, paused_frame)
                    self.screenshot_n += 1
                    print(f"📸  Screenshot saved: {fname}")
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                print(f"🎯  Confidence threshold: {self.conf_threshold:.0%}")
            elif key == ord('-'):
                self.conf_threshold = max(0.10, self.conf_threshold - 0.05)
                print(f"🎯  Confidence threshold: {self.conf_threshold:.0%}")

            # ── Paused mode ───────────────────────────────────────────────────
            if self.paused and paused_frame is not None:
                cv2.imshow("♻️  Recycling Detector", paused_frame)
                continue

            # ── Capture frame ─────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                print("⚠️   Failed to grab frame — retrying...")
                time.sleep(0.1)
                continue

            # ── FPS ───────────────────────────────────────────────────────────
            now         = time.time()
            elapsed     = now - prev_time
            prev_time   = now
            fps_buffer.append(1.0 / max(elapsed, 1e-5))
            fps         = float(np.mean(fps_buffer))

            # ── YOLO inference ────────────────────────────────────────────────
            results = self.model.predict(
                frame,
                conf       = self.conf_threshold,
                device     = self.device,
                verbose    = False,
                imgsz      = 640,
                half       = False,   # half precision off for stability on MPS
            )

            # ── Process detections ────────────────────────────────────────────
            recycle_count    = 0
            no_recycle_count = 0
            total_objects    = 0
            current_tips     = []

            if results and results[0].boxes is not None:
                boxes  = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf            = float(box.conf[0])
                    cls_id          = int(box.cls[0])
                    label           = self.model.names[cls_id]

                    recyclable, material, tip = self.lookup(label)

                    if recyclable:
                        recycle_count += 1
                    else:
                        no_recycle_count += 1
                    total_objects += 1

                    # Rotate tips
                    current_tips.append(f"{label}: {tip}")

                    draw_detection_box(
                        frame, x1, y1, x2, y2,
                        label, conf, recyclable, material, tip
                    )

            # ── Tip rotation ──────────────────────────────────────────────────
            if current_tips and (now - last_tip_time) > self.tip_interval:
                self.tips_queue.append(current_tips[int(now) % len(current_tips)])
                last_tip_time = now

            # ── HUD overlays ──────────────────────────────────────────────────
            draw_hud(frame, fps, self.device_name,
                     recycle_count, no_recycle_count,
                     total_objects, self.model_name,
                     self.paused)

            draw_tip_bar(frame, self.tips_queue)
            draw_controls(frame)

            paused_frame = frame.copy()
            cv2.imshow("♻️  Recycling Detector", frame)

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="♻️  Live Webcam Recycling Detector (YOLOv8 + Mac M2)"
    )
    parser.add_argument(
        "--model", default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n=nano(fastest), s=small, m=medium, l=large, x=xlarge"
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Confidence threshold 0.0-1.0 (default: 0.45)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0 = built-in webcam)"
    )
    args = parser.parse_args()

    detector = RecyclingDetector(
        model_size     = args.model,
        conf_threshold = args.conf,
        camera_index   = args.camera,
    )
    detector.run()

