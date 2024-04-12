# Train ResNet on CIFAR-10 from scratch
It is an example of training ResNet on CIFAR10 dataset by using colossala. from zifeng ren A0268229U

### Used model and dataset

The model is resnet18 and dataset is CIFAR10.

### Install requirements

```bash
pip install -r requirements.txt
```

### Train
```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp16 -p torch_ddp_fp16

# train with low level zero
colossalai run --nproc_per_node 1 train.py -c ./ckpt-low_level_zero -p low_level_zero

# train with gemini
colossalai run --nproc_per_node 1 train.py -c ./ckpt-gemini -p gemini
```

### Eval

```bash
# evaluate fp32 training
python eval.py -c ./ckpt-fp32 -e 80

# evaluate fp16 mixed precision training
python eval.py -c ./ckpt-fp16 -e 80

# evaluate low level zero training
python eval.py -c ./ckpt-low_level_zero -e 80

# evaluate gemini training
python eval.py -c ./ckpt-gemini -e 80
```
### Experiment Result
you can find more detail in train.log
| Model | Training Loss | Test Accuracy (%) |
|-------|---------------|-------------------|
| ResNet-18     | 0.12         | 83.65             |


---
