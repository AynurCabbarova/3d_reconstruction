# 3D Reconstruction

A modular AI-based 3D reconstruction framework that integrates multiple state-of-the-art models into a single application for generating textured 3D scenes and meshes from images.

## Features

- VGGT-based geometry estimation
- Apple Depth Pro monocular depth estimation
- Tencent Hunyuan3D-2 mesh generation
- Open3D visualization
- Globe/map visualization panel
- Modular architecture for integrating additional 3D reconstruction models

## Project Structure

```
3D Mapping/
├── main.py
├── globe_panel.py
├── depth_pro_engine.py
├── hunyuan_engine.py
├── trellis_engine.py
├── vggt_engine.py
├── Hunyuan3D-2/
├── ml-depth-pro/
└── vggt/
```

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Open3D
- CUDA (recommended)

Install dependencies according to each integrated project.

## Pretrained Models

This repository **does not include pretrained model weights** because GitHub has a file size limit.

For example, the following file is intentionally excluded:

```
ml-depth-pro/checkpoints/depth_pro.pt
```

Download the pretrained weights using the original project's instructions.

For Apple Depth Pro:

```bash
cd ml-depth-pro
bash get_pretrained_models.sh
```

or manually place

```
depth_pro.pt
```

inside

```
ml-depth-pro/checkpoints/
```

before running the project.

The same principle applies to any additional pretrained weights required by VGGT, Hunyuan3D-2, or other integrated models.

## Running

```bash
python main.py
```

## Notes

- Large model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`) are intentionally excluded from version control.
- Install pretrained models separately before running inference.
- Original model repositories are included for integration purposes.

## License

Please respect the licenses of the integrated third-party projects:

- Apple Depth Pro
- VGGT
- Hunyuan3D-2
