import os
import multiprocessing
from ultralytics import YOLO

def main():
    # ─── CONFIG ────────────────────────────────────────────────────────────────
    BEST_MODEL   = 'D:/underwater_OD/YOLO/DUO_underwater_project/yolov8s_duo_finetuned4/weights/best.pt'
    DATA_YAML    = 'D:/underwater_OD/DUO/splitted/data.yaml'
    TEST_IMAGES  = 'D:/underwater_OD/DUO/splitted/test/images'
    OUTPUT_DIR   = 'D:/underwater_OD/DUO/splitted/test_results'

    # Ensure output folders exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── LOAD MODEL ─────────────────────────────────────────────────────────────
    model = YOLO(BEST_MODEL)

    # ─── 1) EVALUATE ON TEST SPLIT ───────────────────────────────────────────────
    # This writes results.csv and all evaluation plots under OUTPUT_DIR/evaluation
    model.val(
        data=DATA_YAML,
        split='test',
        imgsz=640,
        batch=16,
        workers=2,         # fewer workers to avoid over-spawning
        plots=True,
        project=OUTPUT_DIR,
        name='evaluation',
        exist_ok=True
    )

    # ─── 2) SAVE ANNOTATED PREDICTIONS ──────────────────────────────────────────
    # This writes images + .txt under OUTPUT_DIR/predictions
    model.predict(
        source=TEST_IMAGES,
        save=True,
        save_txt=True,
        workers=2,
        project=OUTPUT_DIR,
        name='predictions',
        exist_ok=True
    )

    print(f"\n✅ Finished. Check out:\n"
          f"  • {OUTPUT_DIR}/evaluation  (metrics & curves)\n"
          f"  • {OUTPUT_DIR}/predictions (annotated images + .txt files)")

if __name__ == '__main__':
    # On Windows, for multiprocessing inside ultralytics:
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
