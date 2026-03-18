# PoseFM
This repository contains implementation of the PoseFM visual odometry method based on Flow Matching.

## Requirements
PoseFM was built with Python 3.11, PyTorch 2.10.0 and CUDA 12. Additionally we rely on [voloader](https://github.com/DominQu/voloader) for VO datasets implementations. For dependency management we use [uv](https://docs.astral.sh/uv/).

## Installation
1. Install [uv](https://docs.astral.sh/uv/) according to its docs.
2. Clone this repository:
    ```bash
    git clone https://github.com/helsinki-sda-group/posefm.git
    ```
3. Navigate to the project directory:
    ```bash
    cd posefm
    ```
You are good to go. You can run python scripts with `uv run`.

## Usage

### Download test data
To test our model please download test data from [TartanAir](https://github.com/castacks/tartanair_tools?tab=readme-ov-file#download-the-testing-data-for-the-cvpr-visual-slam-challenge), [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [TUM-RGBD](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Add the path to the dataset to a corresonding config. For details on dataset structure check [voloader](https://github.com/DominQu/voloader).

### Download model
Our model weights can be downloaded from: ...

### Prepare config
To run PoseFM you need a config file. We have provided example configs for testing the model on different datasets and for training. Please specify dataset path and model weights path.

### Run PoseFM
Inference of PoseFM can be performed with:
```bash
uv run posefm.py --config <path-to-config> --out <output-dir>
```
As a result of this command, PoseFM will create a trajectory for the provided test dataset. If the test dataset contained ground truth poses, then the code will additionally save the GT trajectory and calculate metrics.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please open an issue.