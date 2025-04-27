import os
import multiprocessing
from ultralytics import YOLO

def main():
    project = 'DUO_underwater_project'
    name    = 'yolov8s_duo_finetuned4'

    # Ensure the weights folder exists
    os.makedirs(os.path.join(project, name, 'weights'), exist_ok=True)

    model = YOLO('yolov8s.pt')

    best_val_map   = 0.0
    patience       = 3
    epochs_no_improve = 0
    total_epochs   = 100

    for epoch in range(1, total_epochs + 1):
        print(f"\n— Epoch {epoch}/{total_epochs} —")

        # Train one epoch without built-in validation
        model.train(
            data='D:/underwater_OD/DUO/splitted/data.yaml',
            epochs=1,
            imgsz=640,
            batch=16,
            optimizer='SGD',
            lr0=0.002,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            device=0,
            workers=2,
            pretrained=True,
            project=project,
            name=name,
            exist_ok=True,
            val=False,
            plots=True
        )

        # Manual validation pass
        metrics = model.val()

        # mean_results() returns (mp, mr, map50, map)  
        mp, mr, map50, map95 = metrics.mean_results()
        print(f" → Validation mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map95:.4f}")

        # Early stopping on the 0.5:0.95 mAP
        if map95 > best_val_map:
            best_val_map = map95
            epochs_no_improve = 0
            print("   ✅ New best mAP — checkpoint saved")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️ No improvement {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"⛔ Early stopping at epoch {epoch}")
            break

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
