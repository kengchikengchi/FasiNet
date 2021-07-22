
"""
FasiNet-H.
Zekun Zheng, Xiaodong Wang, Xinye Lin, and Shaohe Lv. 2021.
Get the Best of the Three Worlds: Real-Time Neural Image Compression in a Non-GPU Environment.
In Proceedings of the 29th ACM International Conference on Multimedia (MM '21),
October 20-24, 2021, Virtual Event, China.
https://doi.org/10.1145/3474085.3475667.
"""

"""
Deployment tips:

1 Use tf.keras.layers.Conv2DTranspose instead of tfc.SignalConv2D 
in the SynthesisTransform helps getting faster decoding speed and 
better rate-distortion performance

2 Use tfc.SignalConv2D(corr=True) in the AnalysisTransform, otherwise there will be 
obvious artifacts at the top of the restored picture if you use tf.keras.layers.Conv2D here.

3 Adding variance and mean as input to the IQR module will not only increase the calculation,
 but also reduce the rate-distortion performance.
"""

import os
import argparse
import glob
import time
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#Set to "0" to use GPU during training, and "-1" to use CPU during testing


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
  """Loads a PNG image file."""
  # print("load png file", filename)
  try:
      string = tf.read_file(filename)
      image = tf.image.decode_image(string, channels=3)
      image = tf.cast(image, tf.float32)
      image /= 255
  except:
      print("read error for ", filename)
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)



class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, *args, **kwargs):
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            96, (9, 9), name="analysis/layer0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name='gdn0')),
        tfc.SignalConv2D(
            96, (9, 1), name="analysis/layer1", corr=True, strides_down=(4, 1),
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name='gdn1')),
        tfc.SignalConv2D(
            192, (1, 9), name="analysis/layer2", corr=True, strides_down=(1, 4),
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name='gdn2')),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
        tensor = layer(tensor)
    return tensor

class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, *args, **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tf.keras.layers.Conv2DTranspose(
            96, (7, 7), strides=4, padding="same", use_bias=True,
            activation=tfc.GDN(inverse=True, name='igdn0'), name="synthesis/layer0"),
        tf.keras.layers.Conv2DTranspose(
            3, (7, 7), strides=4, padding="same", use_bias=True,
            activation=None, name="synthesis/layer1"),
    ]
    super(SynthesisTransform, self).build(input_shape)


  def call(self, tensor):
    for layer in self._layers:
        tensor = layer(tensor)
    return tensor


class IQR(tf.keras.layers.Layer):

  def __init__(self, *args, **kwargs):
    super(IQR, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
       tf.keras.layers.Conv2DTranspose(
            48, (7, 1), strides=(1, 1), padding="same", use_bias=True,
            activation=tf.nn.relu, name="IQR/layer0"),
       tf.keras.layers.Conv2DTranspose(
            96, (1, 7), strides=(1, 1),padding="same", use_bias=True,
            activation=tf.nn.relu, name="IQR/layer1"),
       tf.keras.layers.Conv2DTranspose(
            192, (1, 1), strides=(1, 1),padding="same", use_bias=True,
            activation=None, name="IQR/layer2"),
    ]
    super(IQR, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
        tensor = layer(tensor)
    return tensor


def compress_image(input_dir, output_dir, checkpoint_dir):

    x = tf.placeholder(tf.float32,  shape=[None, None, None, 3])

    # Instantiate model.
    analysis_transform = AnalysisTransform()
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Load the latest model checkpoint, get the compressed string and the tensor shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        print('compress: load weight', latest)

        tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]

        src_imgs = glob.glob("{}/*".format(input_dir))

        time_series = {"total": 0, "read": 0, "write": 0, "process": 0}
        tic_0 = time.time()
        for input_file in src_imgs:
            # Load input image and add batch dimension.
            tic = time.time()
            local_x = read_png(input_file)
            local_x = tf.expand_dims(local_x, 0)
            local_x.set_shape([1, None, None, 3])

            output_file = os.path.join(output_dir, os.path.splitext(os.path.split(input_file)[1])[0] + '.tfci')

            img_x = sess.run(local_x)
            #record time used
            time_series["read"] = time_series["read"] + time.time()-tic
            tic = time.time()

            g_tensors = sess.run(tensors, feed_dict={x: img_x})
            # record time used
            time_series["process"] = time_series["process"] + time.time() - tic
            tic = time.time()

            # Write a binary file with the shape information and the compressed string.
            packed = tfc.PackedTensors()
            packed.pack(tensors, g_tensors)
            with open(output_file, "wb") as f:
                f.write(packed.string)
            # record time used
            time_series["write"] = time_series["write"] + time.time() - tic

            print('compress {} with time {}'.format(input_file, time_series))
        # record time used
        time_series["total"] = time_series["total"] + time.time() - tic_0
        # print("compression, using time {}".format(time.time()-tic))
        print("compress {} imgs using time {}".format(len(src_imgs), time_series))


def decompress_image(input_dir, output_dir, checkpoint_dir):
    """Decompresses an image."""

    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])

    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])

    # Decompress and transform the image back.
    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
    synthesis_transform = SynthesisTransform()
    iqr_transform=IQR()
    # Decompress and transform the image back.
    y_shape1 = tf.concat([y_shape, [192]], axis=0)
    y_hat = entropy_bottleneck.decompress(
        string, y_shape1, channels=192)
    iqr = iqr_transform(y_hat)
    iqr = 0.5 * tf.nn.tanh(iqr)
    y_hat -= iqr
    x_hat = synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat_crop = x_hat[0, :x_shape[0], :x_shape[1], :]

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        print('decompress: load weight', latest)

        time_series = {"total": 0, "read": 0, "write": 0, "process": 0}
        tic_0 = time.time()
        files = glob.glob("{}/*".format(input_dir))
        for input_file in files:
            tic = time.time()
            #output name of the decompressed image
            output_file = os.path.join(output_dir, os.path.splitext(os.path.split(input_file)[1])[0] + '.png')
            #open the compressed file
            with open(input_file, "rb") as f:
                packed = tfc.PackedTensors(f.read())
            tensors = [string,  x_shape, y_shape]
            arrays = packed.unpack(tensors)
            # test_dict = dict(zip(tensors, arrays))
            # record time used
            time_series["read"] = time_series["read"] + time.time() - tic
            tic = time.time()

            #decompress and save
            g_x_hat_crop = sess.run([x_hat_crop], feed_dict=dict(zip(tensors, arrays)))
            # record time used
            time_series["process"] = time_series["process"] + time.time() - tic
            tic = time.time()

            g_x_hat_crop = np.squeeze(g_x_hat_crop)
            sess.run(write_png(output_file, g_x_hat_crop))
            # record time used
            time_series["write"] = time_series["write"] + time.time() - tic

            print('decompress {} with time {}'.format(input_file, time_series))
        # print("using time {}".format(time.time()-tic))
        # record time used
        time_series["total"] = time_series["total"] + time.time() - tic_0
        print("decompress {} imgs using time {}".format(len(files), time_series))

########################################################
def parse_args():
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--verbose", "-V", action="store_true", default=True,
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train_{}".format(os.path.splitext(os.path.basename(__file__))[0].split('_')[0]),
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--batchsize", type=int, default=32,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=5, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=5000000,
      help="Train up to this number of steps.")
  parser.add_argument(
      "--preprocess_threads", type=int, default=8,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # Parse arguments.
  args = parser.parse_args()
  return args


######################################################
def train2(args, train_dir):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(os.path.join(train_dir, '*.png'))
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    else:
        print("train using {} images".format(len(train_files)))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(4 * args.batchsize)
    # print("total {} images for training".format(len(list(train_dataset))))

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  analysis_transform = AnalysisTransform()
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform()
  iqr_transform = IQR()

  # Build autoencoder.
  y = analysis_transform(x)
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  iqr = iqr_transform(y_tilde)
  iqr = 0.5 * tf.nn.tanh(iqr)
  y_tilde -= iqr
  x_tilde = synthesis_transform(y_tilde)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  train_psnr = tf.reduce_mean(tf.squeeze(tf.image.psnr(x_tilde, x, 1)))
  train_msssim = tf.reduce_mean(tf.squeeze(tf.image.ssim_multiscale(x_tilde, x, 1)))
  # train_msssim = MultiScaleSSIM(x, x_tilde)


  lambda_dict = {"mse": 0, "bpp": 5, "psnr": 0, "msssim": args.lmbda}
  main_lr = 1e-4
  aux_lr = 1e-3

  train_loss = lambda_dict["mse"] * train_mse + lambda_dict["bpp"] * train_bpp \
               + lambda_dict["psnr"] * (50-train_psnr) \
               + lambda_dict["msssim"] * (1-train_msssim)

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=main_lr)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  # train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))
  merge_summary = tf.summary.merge_all()

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]

  with tf.train.MonitoredTrainingSession(
          hooks=hooks, checkpoint_dir=args.checkpoint_dir,
          save_checkpoint_secs=300, save_summaries_secs=60) as sess:
      # train_writer = tf.summary.FileWriter(args.checkpoint_dir, sess.graph)

      while not sess.should_stop():
          tic = time.time()
          _, g_step, g_train_loss, g_train_bpp, g_train_mse, g_train_psnr, g_train_msssim = \
              sess.run([train_op, step, train_loss, train_bpp, train_mse, train_psnr, train_msssim])
          print("step-{}: loss-{}, bpp-{},mse-{}, psnr-{}, msssim-{},time_using-{}".format
                (g_step, g_train_loss, g_train_bpp, g_train_mse, g_train_psnr, g_train_msssim,time.time()-tic))


########################################################


if __name__ == "__main__":
    args = parse_args()
    root_dir = '../../images/'

    #  Set your own training set path
    train_dir = '/home/ubuntu/user_space/train_all'
    if not os.path.exists(train_dir):
        print('train dir not exist', os.path.abspath(train_dir))

    # Set your own testing set path
    test_dir = os.path.join(root_dir, 'koda')

    #  Compressed image path
    output_dir = os.path.join(root_dir, 'koda_result')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #  Decompressed image path
    back_dir = os.path.join(root_dir, 'koda_result_back')
    if not os.path.exists(back_dir):
        os.makedirs(back_dir)

    step = 0
    if step == 0:
        train2(args, train_dir)
    elif step == 1:
        compress_image(test_dir, output_dir, args.checkpoint_dir)
    elif step == 2:
        decompress_image(output_dir, back_dir, args.checkpoint_dir)

