# Description

This is a TensorFlow2 implementation of Squeezenet for image classification on ImageNet.

# Links

* Original paper: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360).


# Requirements

* Python3.6
* TensorFlow 2.0.x
* CPU/GPU


# Setup

* `pip install -r requirements.txt` 


# Demo Inference

For a quick application of this network, run the following command that runs the trained model on existing images:
```bash
python demo.py --cfg cfg/imagenet.toml
```

This applies demonstration on the two existing images of **Wolf** and **Snow Leopard** in `resources` directory. The configuration file is required to apply pre-processing on input images.  

Expected output:
```bash
    Squeezenet$ python demo.py --cfg cfg/imagenet.toml 
    I Loading model from directory: /home/govind/work/projects/squeezenet_dir/Squeezenet/models
    I Setup complete.
    I Image: ILSVRC2012_val_00000027.JPEG. True wnid: n02114548 Top-5 predictions: 
    I 	Confidence: 0.984 {'wnid': 'n02114548', 'ilsvrc2012_id': 102, 'word': 'white wolf, Arctic wolf, Canis lupus tundrarum'}
    I 	Confidence: 0.013 {'wnid': 'n02120079', 'ilsvrc2012_id': 159, 'word': 'Arctic fox, white fox, Alopex lagopus'}
    I 	Confidence: 0.002 {'wnid': 'n02114367', 'ilsvrc2012_id': 205, 'word': 'timber wolf, grey wolf, gray wolf, Canis lupus'}
    I 	Confidence: 0.000 {'wnid': 'n02090622', 'ilsvrc2012_id': 105, 'word': 'borzoi, Russian wolfhound'}
    I 	Confidence: 0.000 {'wnid': 'n02415577', 'ilsvrc2012_id': 52, 'word': 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis'}
    I Image: ILSVRC2012_val_00000186.JPEG. True wnid: n02128757 Top-5 predictions: 
    I 	Confidence: 0.668 {'wnid': 'n02128757', 'ilsvrc2012_id': 153, 'word': 'snow leopard, ounce, Panthera uncia'}
    I 	Confidence: 0.087 {'wnid': 'n02123159', 'ilsvrc2012_id': 55, 'word': 'tiger cat'}
    I 	Confidence: 0.050 {'wnid': 'n02128925', 'ilsvrc2012_id': 30, 'word': 'jaguar, panther, Panthera onca, Felis onca'}
    I 	Confidence: 0.049 {'wnid': 'n02123045', 'ilsvrc2012_id': 174, 'word': 'tabby, tabby cat'}
    I 	Confidence: 0.039 {'wnid': 'n02091467', 'ilsvrc2012_id': 63, 'word': 'Norwegian elkhound, elkhound'}
```

# Configuration File

A configuration files must be supplied when using this repository. Some features of configuration file include:

 * Change device between CPU or GPU.
 * Switch the mode between training and evaluation.
 * Switch between TF allocating all GPU memory or on-need basis.
 * Force the code to run on Eager mode. This is useful during debugging.
 * Change directories for dumping TensorBoard logs, checkpoints or output model.
 * If training:
     * Enable/disable TensorBoard logging.
     * Enable/disable dumping checkpoints.
     * Option to specify which checkpoint to restart the training from.
     * Enable/disable sleep for x seconds after epoch. This (hopefully) allows your machine to cool down during hours/days long training sessions.
     * Enable/disable evaluation on validation set after each epoch.
 * If evaluating:
    * Choose to evaluate in train/val split. 
    * Choose to load model from a saved model or existing checkpoint.

Please see the configuration file at `cfg/imagenet.toml`. The comments explain the purpose of each parameter.

We currently have only one configuration file, but I think we could have a dedicated configuration file for each new dataset or model.

# Training and Evaluation

You must have a copy of imagenet dataset to run training and evaluation modes. You don't need any annotation file, those are checked-in into this repository (in `resources/imagenet_metadata`) already. You just need to download the image files (for train and validation set) and modify the following two files manually so that all paths point to images in your local filesystem:

 * `resources/imagenet_metadata/input_list_train.txt`
 * `resources/imagenet_metadata/input_list_val.txt`

## Training

To train the model on Imagenet dataset, set the `misc->mode` parameter in `cfg/imagenet.toml` file to `train` and execute:

```bash
python main.py --cfg cfg/imagenet.toml
```

## Evaluation

To train the model on Imagenet dataset, set the `misc->mode` parameter in `cfg/imagenet.toml` file to `eval` and execute:

```bash
python main.py --cfg cfg/imagenet.toml
```

The default configuration parameters in `cfg/imagenet.toml` run the network in evaluation on validation set of Imagenet.


# Logging

All logs of training/evaluation/demo are automatically saved in the `logs` directory. You could change the logging configuration by supplying a different `log_option` parameter to `setup_logger()` function in `main.py`. Please see the docstring of `setup_logger()` to know more.


# Performance

Current accuracy results on ImageNet:


|                                   | Top-1 Accuracy | Top-5 Accuracy |
|-----------------------------------|----------------|----------------|
| Train Set                         |             -  |              - |
| Validation Set                    |        55.11%  |         78.11% |
| Test Set                          |             -  |              - |
| Test Set (Paper's implementation) |        57.5%   |         80.3%  |
| Test Set (AlexNet)                |        57.2%   |         80.3%  |
| Test Set (MobileNetv1)            |        70.6%   |              - |
| Test Set (MobileNetv2)            |        72.0%   |              - |
| Test Set (MobileNetv3)            |        76.6%   |              - |
| Test Set (Vgg16)                  |        71.5%   |              - |

The last column lists the performance that was reported in the SqueezeNet paper. I can't get this work evaluated on ImageNet test set as the evaluation server does not approve my account creation request.
