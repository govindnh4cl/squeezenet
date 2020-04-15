import os
import tensorflow as tf

from my_logger import setup_logger, get_logger
from squeezenet.arg_parsing import parse_args
from squeezenet.config import get_config
from squeezenet.utils import load_saved_model
from squeezenet.input_imagenet import InputImagenet

setup_logger(log_option=1)
logger = get_logger()


def _do_onetime_setup():
    args = parse_args()
    cfg = get_config(args)

    net = load_saved_model(cfg.directories.dir_model)
    pipeline = InputImagenet(cfg, purpose='deploy')

    logger.info('Setup complete.')
    return net, pipeline


def main():
    this_path = os.path.abspath(os.path.dirname(__file__))

    # ------ configurable section starts -------

    in_img_paths = [os.path.join(this_path, 'resources', x)
                    for x in ('ILSVRC2012_val_00000027.JPEG', 'ILSVRC2012_val_00000186.JPEG')]

    correct_wnids = ['n02114548', 'n02128757']

    # ------ configurable section ends -------

    net, pipeline = _do_onetime_setup()

    imgs = [pipeline.preprocess_x_non_graph(in_img_paths[i]) for i in range(len(in_img_paths))]  # List of tensors
    imgs = tf.convert_to_tensor(imgs)  # 4-D tensor of shape (None, 224, 224, 3)
    imgs = pipeline.preprocess_x_with_graph(imgs)

    y_preds = net.call(imgs, False)  # feed forward
    # Second parameter must be False in deploy mode. Needed to indicate eval mode to batch-normalization and dropout

    # Get the index that would sort the prediction scores from hightest to lowest
    sorted_idx = tf.argsort(y_preds, axis=1, direction='DESCENDING')

    # Prints results
    # Loop over all input images
    for img_idx in range(len(in_img_paths)):
        logger.info('Image: {:s}. True wnid: {:s} Top-5 predictions: '
                    .format(os.path.basename(in_img_paths[img_idx]), correct_wnids[img_idx]))
        # Loop over top-5 predictions for this image
        for i in range(5):
            internal_id = sorted_idx[img_idx][i]  # Internal index is from 0 to 999
            confidence = y_preds[img_idx][internal_id]  # Get confidence for this category
            cat = pipeline.get_category_details(internal_id.numpy())
            logger.info('\tConfidence: {:.3f} {:}'.format(confidence, cat))

    return


if __name__ == '__main__':
    main()

