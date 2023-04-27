
# Full Waveform Inversion
An PyTorch Implementation For Full-Waveform Modeling With Deep Vision-Models on **OpenFWI** dataset.

### Model includes:
- **Dual-UNets**: One UNet for velocity to amplitude (forward) and one UNet for amplitude to velocity (backward). 
- **Dual-GAN**:   One Pix2Pix for velocity to amplitude (forward) and one Pix2Pix for amplitude to velocity (backward). [(Arxiv: Yi et al., 2017)](https://arxiv.org/abs/1704.02510)
- **Cycle-GAN**:  Dual-GAN with cycle consistency loss. [(Arxiv: Zhu et al., 2017)](https://arxiv.org/abs/1703.10593)
- **iUNet**: Invertible UNet for learning invertible mapping that solves forward and backward at the same time. [(Arxiv: Etmann et al., 2020)](https://arxiv.org/abs/2005.05220)

## Installation

For interactive session on `SLURM Cluster` for 2 GPUs, use the following:
```
interact -A [account] --gres=gpu:2 -p dgx_normal_q --ntasks-per-node=2 -t 60:00:00
```
where `ntasks-per-node` will be the number of GPUs you want to use for each training.

Create a Python environmemt:
``` python3 -m venv venv ```

Then activate the environment:

```source venv/bin/activate```

First update `pip`:

```python3 -m pip install --upgrade pip```

Run the following:

```python3 -m pip install -r requirements.txt```

## Training Instruction

Use `train.py` as follows if you want to train on device 0 ~ 6:
```
python3 train.py --config <config_path> --gpu 0 1 2 3 4 5 6
```

Note: FNO `fno_2d` models do not work for multiple gpus now (imcompatible with `nccl` backend).

## Citate This Work
```
@misc{bu2023fullwaveform,
  author = {Bu, Jie},
  title = {Full Waveform Inversion Baselines For OpenFWI},
  year = {2023},
  note = {GitHub repository},
  url = {https://github.com/jayroxis/full_waveform_inversion}
}
```

### Avaible Strategies For PyTorch Lightning

- bagua
- colossalai
- ddp
- ddp_find_unused_parameters_false
- ddp_fork
- ddp_fork_find_unused_parameters_false
- ddp_fully_sharded
- ddp_notebook
- ddp_notebook_find_unused_parameters_false
- ddp_sharded
- ddp_sharded_find_unused_parameters_false
- ddp_sharded_spawn
- ddp_sharded_spawn_find_unused_parameters_false
- ddp_spawn
- ddp_spawn_find_unused_parameters_false
- deepspeed
- deepspeed_stage_1
- deepspeed_stage_2
- deepspeed_stage_2_offload
- deepspeed_stage_3
- deepspeed_stage_3_offload
- deepspeed_stage_3_offload_nvme
- dp
- fsdp
- fsdp_native
- fsdp_native_full_shard_offload
- horovod
- hpu_parallel
- hpu_single
- ipu_strategy
- single_device
- single_tpu
- tpu_spawn
- tpu_spawn_debug

### Available Backend
Check here in PyTorch Lightning
https://pytorch.org/docs/stable/distributed.html
