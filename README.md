# 🖼️ YOLO Image Augmentation Pipeline

A Python-based image augmentation pipeline built with [Albumentations](https://albumentations.ai/) for **YOLO object detection** datasets. Automatically applies diverse augmentation transforms to images while correctly handling bounding box conversions, saving augmented outputs in YOLO format with visual validation.

---

## ✨ Features

- **50+ Pre-built Augmentation Pipelines** — Weather effects (rain, snow, fog, shadow), blur, brightness/contrast, geometric transforms, and more
- **YOLO Format Support** — Reads and writes bounding boxes in YOLO format seamlessly
- **Bounding Box Preservation** — Transforms bounding boxes alongside images using Albumentations' `BboxParams`
- **Visual Validation** — Generates sample images with drawn bounding boxes to verify augmentation quality
- **Multi-Object Support** — Handles both single and multi-object label files
- **Configurable via YAML** — All paths and class names are configurable through `contants.yaml`
- **Modular Architecture** — Clean separation of concerns across controller modules

---

## 📁 Project Structure

```
augment/
├── run.py                              # Entry point — runs the full pipeline
├── contants.yaml                       # Configuration file (paths, classes)
├── requirements.txt                    # Python dependencies
├── controller/                         # Core pipeline modules
│   ├── workflow.py                     # Main pipeline orchestrator
│   ├── apply_album_aug.py             # Augmentation pipelines & application logic
│   ├── get_album_bb.py                # YOLO → Albumentations bbox parser
│   ├── album_to_yolo_bb.py            # Albumentations → YOLO bbox converter
│   ├── save_augs.py                   # Save augmented images & labels
│   └── validate_results.py            # Draw bounding boxes for visual validation
├── inputs/                             # Input directory (images + YOLO labels)
│   └── <dataset_name>/
│       ├── images/                    # Original images
│       └── labels/                    # YOLO format label files (.txt)
└── output/                             # Output directory (auto-generated)
    ├── images/                        # Augmented images
    ├── labels/                        # Augmented YOLO label files
    └── samplesBboxes/                 # Visual validation samples with drawn bboxes
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/riazraja1122/yolo-image-augmentation.git
   cd yolo-image-augmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** You may also need to install `opencv-python` and `pybboxes`:
   > ```bash
   > pip install opencv-python pybboxes
   > ```

### Configuration

Edit `contants.yaml` to set your paths and class names:

```yaml
inp_img_pth : /path/to/your/input/images
inp_lab_pth : /path/to/your/input/labels
out_img_pth : /path/to/your/output/images
out_lab_pth : /path/to/your/output/labels
out_sample_pth : /path/to/your/output/samplesBboxes
transformed_file_name : aug
CLASSES : [class1, class2, class3]
```

| Parameter | Description |
|---|---|
| `inp_img_pth` | Path to input images directory |
| `inp_lab_pth` | Path to input YOLO label files directory |
| `out_img_pth` | Path to save augmented images |
| `out_lab_pth` | Path to save augmented label files |
| `out_sample_pth` | Path to save visual validation samples |
| `transformed_file_name` | Prefix for augmented file names |
| `CLASSES` | List of class names matching your YOLO dataset |

### Run the Pipeline

```bash
python run.py
```

The pipeline will:
1. Read all images from the input directory
2. Parse corresponding YOLO label files
3. Apply selected augmentation pipelines to each image
4. Save augmented images and labels in YOLO format
5. Generate visual validation samples with bounding boxes drawn

---

## 🎨 Augmentation Pipelines

The project includes **50+ augmentation pipelines** in `controller/apply_album_aug.py`. Active pipelines include:

| Pipeline | Transforms | Effect |
|---|---|---|
| **Blur + Brightness** | `Blur` + `RandomBrightnessContrast` | Simulates camera blur |
| **Snow** | `RandomSnow` + `RandomBrightnessContrast` | Winter weather conditions |
| **Fog** | `RandomFog` + `RandomBrightnessContrast` | Foggy weather simulation |
| **Rain** | `RandomRain` + `RandomBrightnessContrast` | Rainy weather conditions |
| **Shadow** | `RandomShadow` + `RandomBrightnessContrast` | Random shadow overlays |

Additional commented pipelines cover: geometric transforms (rotation, flip, scale, shift), distortions (elastic, grid, optical), color adjustments (RGB shift, HSV, equalize, CLAHE), noise (Gaussian, pixel dropout), motion blur, downscaling, posterization, and more.

### Selecting Pipelines

In `controller/apply_album_aug.py`, modify the `selected_indices` list to choose which pipelines to apply:

```python
selected_indices = [0, 1, 3]  # Apply pipelines at these indices
```

---

## 📦 Pipeline Modules

| Module | Purpose |
|---|---|
| `workflow.py` | Orchestrates the pipeline — iterates over images, reads labels, and calls augmentation |
| `apply_album_aug.py` | Defines augmentation transforms and applies them to images with bounding boxes |
| `get_album_bb.py` | Parses YOLO label files and converts bounding boxes to Albumentations format |
| `album_to_yolo_bb.py` | Converts Albumentations bounding boxes back to YOLO format |
| `save_augs.py` | Saves augmented images (`.jpg`) and label files (`.txt`) |
| `validate_results.py` | Draws YOLO bounding boxes on images for visual quality checks |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **[Albumentations](https://albumentations.ai/)** — Image augmentation library
- **[OpenCV](https://opencv.org/)** — Image reading, writing, and drawing
- **[pybboxes](https://github.com/devrimcavusoglu/pybboxes)** — Bounding box format conversion
- **[PyYAML](https://pyyaml.org/)** — YAML configuration parsing

---

## 📄 Input Format

### Images
- Standard image formats supported by OpenCV (`.jpg`, `.png`, `.bmp`, etc.)

### Labels (YOLO Format)
Each `.txt` label file should have one object per line:
```
<class_index> <x_center> <y_center> <width> <height>
```
All values are normalized (0–1). Example:
```
0 0.5 0.5 0.3 0.4
2 0.2 0.8 0.1 0.15
```

---

## 📤 Output

- **`output/images/`** — Augmented images with unique filenames
- **`output/labels/`** — Corresponding YOLO label files
- **`output/samplesBboxes/`** — Visual validation images with bounding boxes drawn for quick quality inspection

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-augmentation`)
3. Commit your changes (`git commit -m 'Add new augmentation pipeline'`)
4. Push to the branch (`git push origin feature/new-augmentation`)
5. Open a Pull Request

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 📬 Contact

If you have questions or suggestions, feel free to open an issue or reach out.

---

> **Built with ❤️ for the Computer Vision community**
