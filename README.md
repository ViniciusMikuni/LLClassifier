# Classifier Optimality using Diffusion Models

## Using shifter on Perlmutter
All the libraries required to run the code in the repo can be acessed through the docker image ```vmikuni/tensorflow:ngc-23.12-tf2-v1```. You can test it locally by doing:
```bash
shifter --image=vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
```

Alternatively, you can use the tensorflow module provided by NERSC with

```bash
module load tensorflow
```
Remember to pick the most recent Tensorflow version available in the modules (currently 2.15.0).


## Training the model

You can run the training code using a single GPU after loading the module/docker container using the command

```bash
cd scripts
python train.py --dataset [top/gluon]
```

Pre-trained checkpoints are available in the ```checkpoints``` folder.

You can also run the code using multiple GPUs by first starting an interactive job from a ssh session with the commands:

```bash
salloc -C gpu -q interactive  -t 30 -n 4 --ntasks-per-node=4 --gpus-per-task=1  -A m3246 --gpu-bind=none  --image vmikuni/tensorflow:ngc-23.12-tf2-v1 --module=gpu,nccl-2.18
```

in case you want to use the docker container or

```bash
salloc -C gpu -q interactive  -t 30 -n 4 --ntasks-per-node=4 --gpus-per-task=1  -A m3246 --gpu-bind=none
```
using the module.

There you can run the code with the multiple GPUs using the command:

```bash
srun --mpi=pmi2 shifter python train.py --dataset [top/gluon]
```
with the docker container or simply
```bash
module load tensorflow
srun python train.py --dataset [top/gluon]
```
with the module.

## Evaluation: Generation

For this exercise we want to first generate events and later evaluate the likelihood of the events we generated. To generate events you can run:
```bash
python evaluate.py --ngen 1000 --sample --dataset [top/gluon]
```
Where the flag ```ngen``` determines the number of events to be generated. This command will save the generated events for jets and particles, as well as the initial Gaussian values used to transform gaussian-> data.
Alternatively, you can generate events using multiple GPUs using the command:

```bash
srun [--mpi=pmi2 shifter] evaluate.py --ngen 1000 --sample --dataset [top/gluon]
```
In this case, each GPU will generate 1000 events that are going to be combined in the same output file.

## Evaluation: Likelihood
The likelihood determination is almost the same as the generation. Since the likelihood is calculated based on generated samples, only run after the generation. The command to run is:
```bash
python evaluate.py --likelihood --dataset [top/gluon]
```
This command will save to the output file the values of the likelihoods as well as the output Gaussian distribution obtained from the data->Gaussian transformation.
Similarly, you can calculate the likelihood in parallel using the command:
```bash
srun [--mpi=pmi2 shifter] python evaluate.py --likelihood --dataset [top/gluon]
```

##Plotting
The last step is to make a few plots of the generated distributions and the comparison between the input and output Gaussian distributions. Simply run the evaluation script as:
```bash
python evaluate.py --dataset [top/gluon]
```
In this case, a single GPU is more than enough!