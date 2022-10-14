## Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds

- [Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds](#frame-mining-a-free-lunch-for-learning-robotic-manipulation-from-3d-point-clouds)
  - [FrameMiner-MixAction in ManiSkill2](#frameminer-mixaction-in-maniskill2)
  - [Running this Codebase](#running-this-codebase)
    - [Installation](#installation)
    - [Example Training Scripts](#example-training-scripts)
    - [More Explanations of Implementation Details](#more-explanations-of-implementation-details)
  - [Caveats for Point Cloud-based Manipulation Learning (especially if you want to adapt our method/code for other tasks)](#caveats-for-point-cloud-based-manipulation-learning-especially-if-you-want-to-adapt-our-methodcode-for-other-tasks)

### FrameMiner-MixAction in ManiSkill2

We have integrated FrameMiner-MixAction in [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn). ManiSkill2-Learn naturally supports manipulation learning in [ManiSkill2](https://github.com/haosulab/ManiSkill2), which is the next generation of the SAPIEN ManiSkill benchmark with significant improvements. 


### Running this Codebase

If you would like to perform experiments using this code base, follow the instructions below. Experiment settings largely follow [ManiSkill1](https://github.com/haosulab/ManiSkill). 

#### Installation

For this repo, we require CUDA=11.3. If you haven't had CUDA=11.3 locally yet, download the runfile from NVIDIA at [this link](https://developer.nvidia.com/cuda-11.3.0-download-archive) and install it.

To install, first create an Anaconda environment with python=3.8:

```
conda create -n frame_mining python=3.8
```

Then install pytorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pytorch3d
```

Install SAPIEN:

```
pip install sapien-2.0.0.dev20220317-cp38-cp38-manylinux2014_x86_64.whl
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

You can find example scripts for training single-frame policies and FrameMiner-MixAction(FM-MA) at `./script`. To evaluate an existing model, simply add `--evaluation` and `--resume-from {path_to_ckpt}` to the script arguments. The corresponding config files are in `configs/mfrl/ppo/maniskill`. In addition, you can use `--cfg-options` in the command line to override arguments in the config files (like the example scripts do). 

*If you train end-effector frame-based policies (single-frame policies or FrameMiners), then for OpenCabinetDoor and OpenCabinetDrawer, pass in `env_cfg.nhand_pose=1` to the script since they are single-arm environments. For PushChair and MoveBucket, pass in `env_cfg.nhand_pose=2` since they are dual arm environments.*

#### More Explanations of Implementation Details

The file paths shown below are the relative paths of `{this_directory}/pyrl`.

Training / evaluation are initialized through `tools/run_rl.py`. Training loop is in `pyrl/apis/train_rl.py`. Evaluation is in `pyrl/env/evaluation.py`. PPO implementation is in `pyrl/methods/mfrl/ppo.py`.

Environment is built through the `make_gym_env` function in `pyrl/env/env_utils.py`. Environment wrapper is in `pyrl/env/wrappers.py`.

`pyrl/networks/backbones/visuomotor.py` contains two classes: `Visuomotor` for single-frame visuomotor policies, and `VisuomotorTransformerFrame` for FrameMiners (FM-MA, FM-FC, FM-TG). Visual feature extractors for each individual coordinate frame are implemented with PointNet / SparseConv ( `pyrl/networks/backbones/pointnet.py / sp_resnet.py`). If you use `FM-TG`, the `TransformerFrame` class in `pyrl/networks/backbones/transformer.py` will also be used.

More details can be inferred through the configuration files in `configs/mfrl/ppo/maniskill`. The APIs are similar to [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn).

### Caveats for Point Cloud-based Manipulation Learning (especially if you want to adapt our method/code for other tasks)

If you look into the configurations files (`configs/mfrl/ppo/maniskill` in this repo, or the ones in `ManiSkill2-Learn`), you might notice that there is an argument `zero_init_output=True`. This initializes the last layer of MLP before the policy / value outputs to zero at the beginning of training. We have found that this is of great help for stabilizing initial-stage training, especially in FrameMiners where there are multiple visual feature extractors.

If you look into our PointNet implementations (`pyrl/networks/backbones/pointnet.py`), you may notice that we have removed the spatial transformation layer from the original PointNet, and we added Layer Normalization to the network. Without Layer Normalization, point cloud-based agent training will easily fail. 



