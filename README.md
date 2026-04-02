<div align="center">
<div style="text-align: center;">
    <img src="./assets/matanyone2_logo.png" alt="MatAnyone Logo" style="height: 52px;">
    <h2>MatAnyone 2.1: Optimized Video Matting for Apple Silicon</h2>
    <h4>An optimized fork with critical memory improvements for Mac Studio and Apple Silicon devices</h4>
</div>

<div>
    <a href='https://pq-yang.github.io/' target='_blank'>Peiqing Yang</a><sup>1</sup>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou</a><sup>1†</sup>&emsp;
    <a href="https://www.linkedin.com/in/kai-hao-794321382/" target='_blank'>Kai Hao</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com.sg/citations?user=fMXnSGMAAAAJ&hl=en/" target='_blank'>Qingyi Tao</a><sup>2</sup>&emsp;
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; 
    <sup>2</sup>SenseTime Research, Singapore&emsp; 
    <br>
    <sup>†</sup>Project lead
</div>


<div>
    <h4 align="center">
        <a href="https://pq-yang.github.io/projects/MatAnyone2/" target='_blank'>
        <img src="https://img.shields.io/badge/😈-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2512.11782" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2501.14677-b31b1b.svg">
        </a>
        <a href="https://www.youtube.com/watch?v=tyi8CNyjOhc&lc=Ugw1OS7z5QbW29RZCFZ4AaABAg" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://huggingface.co/spaces/PeiqingYang/MatAnyone" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/MatAnyone2&ltext=Visitors&color=3977dd">
    </h4>
</div>

<strong>MatAnyone 2.1</strong> is an optimized fork of the original MatAnyone 2 framework. It preserves fine details by avoiding segmentation-like boundaries, while adding critical memory optimizations that enable processing videos of <strong>any length</strong> with constant RAM usage on Apple Silicon devices.

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.jpg">
</div>

:movie_camera: For more visual results, go checkout our <a href="https://pq-yang.github.io/projects/MatAnyone2/" target="_blank">project page</a>

---
</div>


## 📮 Update
- **[2025.04]** Major memory optimizations for Apple Silicon Macs (M1/M2/M3/M4). Videos of any length can now be processed with constant RAM usage using disk-based streaming.
- [2025.03] Add uv, CLI, and huggingface support for easy installation and usage.
- [2025.03] Release inference codes, evaluation codes, and gradio demo.
- [2025.12] This repo is created.


## 🏄🏻‍♀️ TODO
- [x] Release inference codes and gradio demo. 
- [x] Release evaluation codes.
- [ ] Release training codes for video matting model.
- [ ] Release checkpoint and training codes for quality evaluator model.
- [ ] Release real-world video matting dataset **VMReal**.


## 🔎 Overview
![overall_structure](assets/matanyone1vs2.jpg)

## 🔧 Installation

### Conda
1. Clone this optimized Repo
    ```bash
    git clone https://github.com/emanuelbarriga/MatAnyone2.1
    cd MatAnyone2.1
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n matanyone2 python=3.10 -y
    conda activate matanyone2

    # install python dependencies
    pip install -e .
    # [optional] install python dependencies for gradio demo
    pip3 install -r hugging_face/requirements.txt
    ```

### uv
You may also install via [uv](https://docs.astral.sh/uv/):
```bash
# create a new project and add matanyone2
uv init my-matting-project && cd my-matting-project
uv add matanyone2@git+https://github.com/pq-yang/MatAnyone2.git
```

## 🍎 Mac Studio / Apple Silicon Optimization

MatAnyone2 now includes **ComfyUI-inspired memory optimizations** specifically designed for Apple Silicon Macs with unified memory (M1/M2/M3/M4):

| Feature | Before | After |
|---------|--------|-------|
| Video Loading | All frames in RAM (1.8GB+ for 1080p/10s) | **Disk streaming** (~30MB constant) |
| Processing Results | Accumulated in RAM lists | **Saved directly to SSD** |
| VRAM Management | None | **`gc.collect()` + `torch.mps.empty_cache()`** every frame |
| Tensor Cleanup | Automatic GC | **Explicit `del` + cache clearing** |

### How It Works

1. **Disk-based Frame Caching**: Videos are saved as PNG frames on disk; only 5 frames stay in RAM at any time
2. **Streaming Inference**: The model processes frames one-by-one, immediately saving results to disk
3. **Aggressive Memory Cleanup**: After each frame, intermediate tensors are deleted and MPS cache is cleared
4. **Zero Memory Leaks**: No accumulation of "ghost VRAM" during long video processing

### Result

Your **32GB Mac Studio** can now process videos of **any length** without running out of memory. Whether it's 5 seconds or 5 hours, RAM usage stays constant.

---

## 🔥 Inference

### Download Model
Download our pretrained model from [MatAnyone 2](https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth) to the `pretrained_models` folder (pretrained model can also be automatically downloaded during the first inference).

The directory structure will be arranged as:
```
pretrained_models
   |- matanyone2.pth
```

### Quick Test
We provide some examples in the [`inputs`](./inputs) folder. **For each run, we take a video and its first-frame segmenatation mask as input.** <u>The segmenation mask could be obtained from interactive segmentation models such as [SAM2 demo](https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor)</u>. For example, the directory structure can be arranged as:
```
inputs
   |- video
      |- test-sample1          # folder containing all frames
      |- test-sample2.mp4      # .mp4, .mov, .avi
   |- mask
      |- test-sample1.png      # mask for targer person(s)
      |- test-sample2.png    
```
Run the following command to try it out:

```shell
# intput format: video folder
python inference_matanyone2.py -i inputs/video/test-sample1 -m inputs/mask/test-sample1.png

# intput format: mp4
python inference_matanyone2.py -i inputs/video/test-sample2.mp4 -m inputs/mask/test-sample2.png
```
- The results will be saved in the `results` folder, including the foreground output video and the alpha output video.
- If you want to save the results as per-frame images, you can set `--save-image`.
- If you want to set a limit for the maximum input resolution, you can set `--max-size`, and the video will be downsampled if min(w, h) exceeds. By default, we don't set the limit.

Or you may directly run via CLI command:
```shell
matanyone2 -i inputs/video/test-sample1 -m inputs/mask/test-sample1.png
```
- Run `matanyone2 --help` for a full list of options.

### Python API 🤗
You can load the model directly from Hugging Face using `from_pretrained` and run inference programmatically:

```python
from matanyone2 import MatAnyone2, InferenceCore

model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2")
processor = InferenceCore(model, device="cuda:0")
processor.process_video(
    input_path="inputs/video/test-sample2.mp4",
    mask_path="inputs/mask/test-sample2.png",
    output_path="results",
)
``` 

## 🎪 Interactive Demo
To get rid of the preparation for first-frame segmentation mask, we prepare a gradio demo on [hugging face](https://huggingface.co/spaces/PeiqingYang/MatAnyone2) and could also **launch locally**. Just drop your video/image, assign the target masks with a few clicks, and get the the matting results!

*We integrate MatAnyone Series in the demo. [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) is the default model. You can also choose [MatAnyone](https://github.com/pq-yang/MatAnyone) as your processing model in "Model Selection".*

```shell
cd hugging_face

# install GUI dependencies
pip3 install -r requirements.txt # FFmpeg required

# launch the demo
python app.py
```

By launching, an interactive interface will appear as follow.

![overall_teaser](assets/teaser_demo.gif)

## 📊 Evaluation
Please refer to the [evaluation documentation](docs/EVAL.md) for details.

## 🛠️ Data Pipeline
![data_pipeline](assets/data_pipeline.jpg)


## 📑 Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
  @InProceedings{yang2026matanyone2,
      title     = {{MatAnyone 2}: Scaling Video Matting via a Learned Quality Evaluator},
      author    = {Yang, Peiqing and Zhou, Shangchen and Hao, Kai and Tao, Qingyi},
      booktitle = {CVPR},
      year      = {2026}
      }

  @inProceedings{yang2025matanyone,
      title     = {{MatAnyone}: Stable Video Matting with Consistent Memory Propagation},
      author    = {Yang, Peiqing and Zhou, Shangchen and Zhao, Jixin and Tao, Qingyi and Loy, Chen Change},
      booktitle = {CVPR},
      year      = {2025}
      }
   ```

## 📝 License

This project is licensed under <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## 👏 Acknowledgement

### Original Project
This project is a fork of **[MatAnyone 2](https://github.com/pq-yang/MatAnyone2)** by:
- [Peiqing Yang](https://pq-yang.github.io/) (S-Lab, NTU)
- [Shangchen Zhou](https://shangchenzhou.com/) (S-Lab, NTU)
- [Kai Hao](https://www.linkedin.com/in/kai-hao-794321382/) (S-Lab, NTU)
- [Qingyi Tao](https://scholar.google.com.sg/citations?user=fMXnSGMAAAAJ&hl=en/) (SenseTime Research)

Paper: [MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](https://arxiv.org/abs/2512.11782)

### Optimizations in Version 2.1 (This Fork)
**Memory & Performance Enhancements:**
- Disk-based streaming with lazy frame loading (only 5 frames in RAM)
- Aggressive VRAM cleanup with `gc.collect()` and `torch.mps.empty_cache()`
- Explicit tensor cleanup to prevent "ghost VRAM"
- Optimized inference pipeline for Apple Silicon unified memory

### Dependencies
This project is also built upon [MatAnyone](https://github.com/pq-yang/MatAnyone) and [Cutie](https://github.com/hkchengrex/Cutie), with matting dataset files adapted from [RVM](https://github.com/PeterL1n/RobustVideoMatting). The interactive demo is adapted from [ProPainter](https://github.com/sczhou/ProPainter), leveraging segmentation capabilities from [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and [Segment Anything Model 2](https://github.com/facebookresearch/sam2). Thanks for their awesome works!

## 📧 Contact

For questions about this optimized version, please open an issue in this repository.

For questions about the original MatAnyone 2 project, please contact `peiqingyang99@outlook.com`.
