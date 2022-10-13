## Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds

- [Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds](#frame-mining-a-free-lunch-for-learning-robotic-manipulation-from-3d-point-clouds)
  - [Running this Codebase](#running-this-codebase)
    - [Installation](#installation)
    - [Example Training Scripts](#example-training-scripts)
    - [More Explanations of Implementation Details](#more-explanations-of-implementation-details)

### Running this Codebase

If you would like to perform experiments using this code base, follow the instructions below.

#### Installation

For this repo, we require CUDA=11.3. If you haven't had CUDA=11.3 locally yet, download the runfile from NVIDIA at [this link](https://developer.nvidia.com/cuda-11.3.0-download-archive) and install it.

To install, first create an Anaconda environment with python=3.8:

```
conda create -n py38 python=3.8
```

Then install pytorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pytorch3d
```

Install SAPIEN:

```
pip install sapien.whl
```

Install our code base:
```
cd {this_directory}/mani_skill
pip install -e .
cd {this_directory}/pyrl
pip install ninja
pip install -e .
pip install protobuf==3.19.0

sudo apt install libsparsehash-dev # prerequisite for torchsparse
cd {this_directory}/torchsparse
pip install -e .
```

Download `partnet-mobility-dataset.zip` from [this url](https://drive.google.com/drive/folders/1shJIf8IV4nLRguedr4biSJF8nRq1gDCR) and put in under `{this_directory}/pyrl`. Then, `unzip` this file in the same directory. 

Note that the objects meshes under this `partnet-mobility-dataset` are different from the ones used in the original [ManiSkill](https://github.com/haosulab/ManiSkill), since [CoACD](https://github.com/SarahWeiii/CoACD) (instead of VHACD) is used to decompose 3D objects into convex components. Using CoACD significantly improves the fidelity of decomposed shapes (especially in delicate parts like cabinet handles), thereby enhancing realistic contact-rich simulation. CoACD is also used to preprocess object meshes in [ManiSkill2](https://github.com/haosulab/ManiSkill2).


#### Example Training Scripts

First, `cd {this_directory}/pyrl`

You can find example scripts for training single-frame policies and FrameMiner-MixAction(FM-MA) at `./script`. To evaluate an existing model, simply add `--evaluation` to the script arguments. The corresponding config files are in `configs/mfrl/ppo/maniskill`. In addition, you can use `--cfg-options` in the command line to override arguments in the config files (like the example scripts do). 

#### More Explanations of Implementation Details

The file paths shown below are the relative paths of `{this_directory}/pyrl`.

Training / evaluation are initialized through `tools/run_rl.py`. Training loop is in `pyrl/apis/train_rl.py`. Evaluation is in `pyrl/env/evaluation.py`. PPO implementation is in `pyrl/methods/mfrl/ppo.py`.

Environment is built through the `make_gym_env` function in `pyrl/env/env_utils.py`. Environment wrapper is in `pyrl/env/wrappers.py`.

`pyrl/networks/backbones/visuomotor.py` contains two classes: `Visuomotor` for single-frame visuomotor policies, and `VisuomotorTransformerFrame` for FrameMiners (FM-MA, FM-FC, FM-TG). Visual feature extractors for each individual coordinate frame are implemented with PointNet / SparseConv ( `pyrl/networks/backbones/pointnet.py / sp_resnet.py`). If you use `FM-TG`, the `TransformerFrame` class in `pyrl/networks/backbones/transformer.py` will also be used.

More details can be inferred through the configuration files in `configs/mfrl/ppo/maniskill`. The APIs are similar to [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn).





