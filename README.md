# Fast Injective Mesh Parameterization via Beltrami Coefficient Prolongation
--------------------------------------------------------------------------------------------------------------------------------------------------------
This code includes the implementation of the Eurographics 2026 paper "Fast Injective Mesh Parameterization via Beltrami Coefficient
Prolongation" authored by Guy Fargion and Ofir Weber. 

The use of this application is limited to academic use only!

The code is provided as-is and without any guarantees.

This GIF submodule is partially based on https://github.com/GuyFa/GIF

The CM submodule is partially based on https://github.com/Roipo/CompMajor

For questions or comments about the code please contact:
Guy Fargion (guy.fargion@gmail.com)

----------------------------------------------------------------------------
The code should be platform independent though we never tested it on other than Windows OS.

## Portable CMake/Pixi build

Dependency management is consolidated in **`pyproject.toml`**, using Pixi for
both Conda and PyPI packages. CMake/CPM supplies the native build, with separate
MKL PARDISO (default), Panua PARDISO, and explicit Eigen fallback backends.

```sh
pixi run test
pixi run parameterize path/to/disk.obj outputs/disk_cm.obj
# Full pipeline with a separately installed/licensed MATLAB:
pixi run -e matlab test
# Optional Linux-only Octave alternative:
pixi run -e octave test
# Original-style GPU path (MATLAB + CUDA required):
pixi run -e matlab parameterize path/to/disk.obj outputs/disk_gpu.obj --device cuda:0 --gif-device gpu
# Optional Torch evaluator under Octave (working CUDA required):
pixi run -e octave parameterize path/to/disk.obj outputs/disk_gpu.obj --device cuda:0 --gif-device gpu
```

See **[docs/build.md](docs/build.md)** for Linux/Windows targets, runtime
requirements, validation status and performance caveats. The Linux MKL-backed
native tests and both complete Octave pipeline variants pass on CPU and CUDA
(using the Torch evaluator). CM-only execution is not a full BCP run; Windows,
MATLAB and performance parity remain unverified.

## Original Windows installation (reference)

### 1) GIF
A Visual Studio 2019 project is provided for easy compilation on Windows machines.

The following prerequisites are necessary for building and running the code:

1) Matlab R2022b

2) Boost 1.83.0 - We downloaded boost from here https://www.boost.org/. Our code only requires the headers of the Boost libraries. Hence, there is no need to build boost.

3) CGAL 5.6 - We installed CGAL 5.6 with the gmp and mpfr auxiliary libraries. There is no need to build CGAL as well.

4) GMM C++ template library version 4.2 (http://getfem.org/download.html).

5) PARDISO 8.2

6) Eigen 3.4.0

7) NLopt 2.10


Other versions of the above listed tools might be compatible but weren't tested!

Now:

1) Install and build the above mentioned prerequisites.

2) Add the following environment variables to your system (see some possible paths):

GMM_INCLUDE_DIR		  (%your GMM folder path%)\gmm-4.2\include

MATLAB_64_DIR		    C:\Program Files\MATLAB\R2022b

CGAL_64_DIR		      C:\Program Files\CGAL-5.6

BOOST_64_DIR		    C:\Program Files\boost\boost_1_83_0

EIGEN_DIR  		      (%your Eigen folder path%)

PARDISO_BIN         (%path to the folder where PARDISO dll and lib are%)

PARDISO_LIC_PATH    (%path to the folder with PARDISO licence%)

OMP_NUM_THREADS  	  number of cores in your CPU (for PARDISO)

NLOPT_DIR           (%path to the nlopt directory)

3) Add the the folder "MatlabScripts" to your Matlab path.

4) Make sure all the required dlls can be loacted by including the relevant paths into the system PATH variable.
For example:

PATH=
%MATLAB_64_DIR%\bin\win64;
%MATLAB_64_DIR%\extern\include;
%MATLAB_64_DIR%\extern\lib\win64\microsoft;
%BOOST_64_DIR%\libs;
%CGAL_64_DIR%\lib;
%CGAL_64_DIR%\auxiliary\gmp\lib;
%CGAL_64_DIR%\include\CGAL;
%PARDISO_BIN%;
%MATLAB_64_DIR%
%NLOPT_DIR%

### 2) CM

After installing GIF, CM should run as well with no further installation, as the environment variables needed for PARDISO already exist. Make sure you clone all the recursive submodules as well.

### 3) PT

This submodule performs computations related to the parallel transport operations. After install GIF, PT should run as well with no further installation.

### 3) BCP
We used python 3.10.

All requirements appear in ```BCP/requirements.txt```.


## Running
```BCP/GIF_and_CM/test_models.py``` contains a script for running the entire algorithm.

The dataset folder is assumed to contain a list of 3D disk-like meshes:

```
dataset
|0_source_textured_by_uv_ours.obj
|1_source_textured_by_uv_ours.obj
| ...
```

and an additional file named ```data.file``` with the list of models:

```
{
"models": [
        "0_source_textured_by_uv_ours.obj",
        "1_source_textured_by_uv_ours.obj",
        ...
    ]
}
```

Now just set the following constants in `test_models.py`:

`DATA_FILE` - the path to `data.json`

`OBJ_RESULTS_PATH` - the path in which the code will save `.obj` result.

`RESULTS_SUMMARY_PATH` - the path to the statistics `.csv` file.

Optional flags:

The use can manually set the target number of faces in the simplified mesh via `TARGET_SIMPLIFICATION_FACES_COUNT` in `MeshProcessor.py`. 

## Object files
The dataset used in the paper is available at https://livebiuac-my.sharepoint.com/:f:/g/personal/guy_fargion_live_biu_ac_il/IgB9ZUKwt23DQbSREXHH049eATo5VFyRF5QgPY1wq1itk2s?e=UY3wN4

The models from the figures in the paper are available at https://livebiuac-my.sharepoint.com/:f:/g/personal/guy_fargion_live_biu_ac_il/IgB8d1R6EdenQaSm7pNEa2FSAac6x-8mLYc2XeSC6yiZzyc?e=GSOp7A
