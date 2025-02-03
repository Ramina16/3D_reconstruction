# 3D_reconstruction
## This project provides code for 3D reconstruction of the scene from depth maps and with COLMAP (https://colmap.github.io/)

## Data storage
### Data for 3D reconstruction from depth maps should consist of following folders:
- color (with image files)
- (optional) depth (with GT depth files in 16-bit format)
- pose (with .txt files with frame_to_world poses for each image)
- intrinsics.txt
You can find example of data in data/scene0005_00

### Data for 3D reconstruction with COLMAP should consist of folder:
- images (with images files)

## **Example Usage:**  
### 3D reconstruction from depth maps:
```bash
python3 -m main -m metric3d_vit_small -step 5 -path_to_scene data/scene0005_00 -save_pc
```
### 3D reconstruction with COLMAP:
```bash
python3 -m colmap_extract_poses --scenedir data/photos
```

## Results examples:
### Mesh and pointcloud for ScanNet scene0005_00 using predicted depth maps by one of the Metric3D v2 model (https://github.com/YvanYin/Metric3D):
<p align="center">
<img width="350" img height="350" alt="image" src="https://github.com/user-attachments/assets/e5d70847-c4b0-4c9e-94e5-fea1c1c318bd" />
<img width="350" img height="350" alt="image" src="https://github.com/user-attachments/assets/0b249326-aec6-47ad-8b93-9fba90840384" />
</p>

### 3D pointcloud for own photos with COLMAP usage:
<p align="center">
<img width="350" img height="350" "alt="image" src="https://github.com/user-attachments/assets/609baa37-8555-43d9-822b-60db336c2f42" />
<img width="350" img height="350" "alt="image" src="https://github.com/user-attachments/assets/dc8d45ee-1391-4c7e-a39a-10cc6943d22f" />
</p>

