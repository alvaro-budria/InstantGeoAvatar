# InstantGeoAvatar

Code release for the ACCV 2024 paper [_InstantGeoAvatar: Effective Geometry and Appearance Modeling of Animatable Avatars from Monocular Video_]().

## Installation
To install the necessary dependencies, we recommend taking the following steps (as PyTorch3D and tinycudann can be problematic):

```
conda create -n instantgeoavatar python=3.9
conda activate instantgeoavatar
```

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install --upgrade setuptools wheel
conda install pytorch3d -c pytorch3d -c conda-forge
```


Then install `tinycudann` as described in the official [repository](https://github.com/NVlabs/tiny-cuda-nn.git).
To install from source, make sure to clone recursively and consider first installing `nvcc` compiler with conda and then aliasing to it:

```
conda install nvidia/label/cuda-11.7.0::cuda-nvcc
conda install nvidia/label/cuda-11.7.0::cuda-toolkit
alias nvcc='<path-to-conda-installation>/envs/instantgeoavatar/bin/nvcc'
export CUDA_HOME=<path-to-conda-installation>/envs/instantgeoavatar/
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Finally,
```
pip install pip==23.3.2
pip install -r requirements.txt
```



## Data

### SMPL body model

Download SMPL parametric model from: https://smpl.is.tue.mpg.de/ and place it under `./data/SMPLX/smpl/`:
```
# └── data/SMPLX/smpl/
#         ├── SMPL_FEMALE.pkl
#         ├── SMPL_MALE.pkl
#         └── SMPL_NEUTRAL.pkl
```


### X-Humans dataset

Go to `https://skype-line.github.io/projects/X-Avatar/` and follow the link to the dataset download page. You will need to fill in a form.

Unzip the downloaded files in `data/XHumans`.

Prepare the dataset with

```
cd data
python setup_dataset.py
```


### PeopleSnapshot dataset

Download data from the official dataset [webpage](https://graphics.tu-bs.de/people-snapshot).

Then run
```
path_pplsnap=<PATH_TO_PEOPLESNAPSHOT>
preproc_script=scripts/peoplesnapshot/preprocess_PeopleSnapshot.py

python preproc_script --root path_pplsnap --subject female-3-casual
python preproc_script --root path_pplsnap --subject male-3-casual
python preproc_script --root path_pplsnap --subject female-4-casual
python preproc_script --root path_pplsnap --subject male-4-casual
```



## Training

For the first run, you will have to execute
```
export LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
python launch.py --config configs/demo_instantgeoavatar_xhumans.yaml --pretrain_SMPL_SDF
```
to obtain the initialization checkpoint for the geometry representing the SMPL body geometry prior.


To run training
```
export LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
python launch.py --config configs/demo_instantgeoavatar_xhumans.yaml --from_pretrained
```



## Inference

To perform inference on a new set of body poses, you can run the following command, making sure the dataset and subject in the configuration `.yaml` file match that of the checkpoint path.

```
export LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
cd src
# An example ckpt_path:
ckpt_path = ../exp/InstantGeoAvatar_<dataset>/<subject>/ckpt/<checkpoint>.ckpt
python launch.py --config configs/demo_instantgeoavatar_xhumans.yaml --test --resume ckpt_path
```

Find the generated results at `../exp/InstantGeoAvatar_<dataset>/<subject>/save/test/`.



## Experimenting on custom video

We refer to the instructions [here](https://github.com/tijiang13/InstantAvatar/tree/master?tab=readme-ov-file#play-with-your-own-video) on how to obtain segmentation masks, SMPL parameters and camera parameters needed to train InstantGeoAvatar.



## Acknowledgments

We want to especially acknowledge the following works:

[instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

[X-Humans](https://skype-line.github.io/projects/X-Avatar/)

[InstantAvatar](https://github.com/tijiang13/InstantAvatar/)

[Vid2Avatar](https://github.com/MoyGcc/vid2avatar)

[SMPLX](https://github.com/vchoutas/smplx)



## Citation (BibTex)

```
@article{budria2024instantgeoavatar,
  author    = {Budria, Alvaro and Lopez-Rodriguez, Adrian and Lorente, Oscar and Moreno-Noguer, Francesc},
  title     = {InstantGeoAvatar: Effective Geometry and Appearance Modeling of Animatable Avatars from Monocular Video},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  month     = {December},
  year      = {2024},
}
```