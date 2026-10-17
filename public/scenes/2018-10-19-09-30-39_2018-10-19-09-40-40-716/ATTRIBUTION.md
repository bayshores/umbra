# DrivingStereo attribution

This scene is made from the [DrivingStereo dataset](https://drivingstereo-dataset.github.io/), frame `2018-10-19-09-30-39_2018-10-19-09-40-40-716`. The dataset is released under the MIT License, Copyright (c) 2019 drivingstereo-dataset. The full license text is in [LICENSE](LICENSE).

The dataset authors ask users to cite:

```bibtex
@inproceedings{yang2019drivingstereo,
    title={DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios},
    author={Yang, Guorun and Song, Xiao and Huang, Chaoqin and Deng, Zhidong and Shi, Jianping and Zhou, Bolei},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```

Changes made by umbra:

- The left and right images (`recorded.jpg`, `right.jpg`) and the depth map were resized and cropped.
- The depth map was turned into 3-D points and triangles (`positions.bin`, `colors.bin`, `triangles.bin`, `depth.bin`, `support.png`).
- `estimatedpositions.bin`, `estimatedcolors.bin` and `estimatedtriangles.bin` are a depth prediction made by NVIDIA C-FoundationStereo-S (revision `d776ba85`) from the image pair. The app uses them for display only.
- `detections.json` holds RF-DETR vehicle detections and `suggestions.json` holds umbra's placement suggestions.
