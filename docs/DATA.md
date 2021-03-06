# Setting Up Data Paths

Expected datasets structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected datasets structure for CIFAR-10:

```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

Create a directory containing symlinks:

```
mkdir -p /path/sscls/sscls/datasets/data
```

Symlink ImageNet:

```
ln -s /path/imagenet /path/sscls/sscls/datasets/data/imagenet
```

Symlink CIFAR-10:

```
ln -s /path/cifar10 /path/sscls/sscls/datasets/data/cifar10
```

If you want to use Nori or DPFlow, just check the dataset path in 

```
/path/to/sscls/datasets/paths.py.
```
