from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import model
import data_utils

from gpu_utils import assign_to_gpu, average_grads_and_vars

from dynamic_eval_opt import DynamicEvalOptimizer as DynamicEvalOpt
import numpy as np

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")

flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for do_test evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")


# Optimization config
flags.DEFINE_float("learning_rate", default=0.00006,
      help="Maximum learning rate.")
flags.DEFINE_float("decay_rate", default=0.001,
      help="Maximum learning rate.")
flags.DEFINE_float("epsilon", default=0.0001,
      help="Maximum learning rate.")


# Training config
flags.DEFINE_integer("train_batch_size", default=1,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=1,
      help="Size of valid batch.")

flags.DEFINE_bool("rms", default=False,
      help="rms style dynamic evaluation.")

flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("ratio", default=1,
      help="divide eval set len by this")
flags.DEFINE_integer("tgt_len", default=70,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS

def get_model_fn(n_token, cutoffs):
  def model_fn(inp, tgt, mems, is_training):
    inp = tf.transpose(inp, [1, 0])
    tgt = tf.transpose(tgt, [1, 0])

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    loss, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=False,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    # number of parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # format_str = '{{:<{0}s}}\t{{}}'.format(
    #     max([len(v.name) for v in tf.trainable_variables()]))
    # for v in tf.trainable_variables():
    #   tf.logging.info(format_str.format(v.name, v.get_shape()))

    if is_training:
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      grads_and_vars = list(zip(grads, all_vars))

      return loss, new_mems, grads_and_vars
    else:

      return loss, new_mems

  return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
  model_fn = get_model_fn(
      n_token=n_token,
      cutoffs=cutoffs)

  model_ret = model_fn(
      inp=inp,
      tgt=tgt,
      mems=mems,
      is_training=is_training)

  return model_ret


def dynamic_eval(n_token, cutoffs, ps_device):
  ##### Get input function and model function
  if FLAGS.rms:
      ##using training data to collect gradient statistics
      train_input_fn, train_record_info = data_utils.get_input_fn(
           record_info_dir=FLAGS.record_info_dir,
           split="train",
           per_host_bsz=FLAGS.train_batch_size,
           tgt_len=FLAGS.tgt_len,
           num_core_per_host=FLAGS.num_core_per_host,
           num_hosts=1,
           use_tpu=False)


      num_batch = train_record_info["num_batch"]

      tf.logging.info("num of batches {}".format(num_batch))

      ##### Create computational graph
      train_set = train_input_fn({
          "batch_size": FLAGS.train_batch_size,
          "data_dir": FLAGS.data_dir})

      input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

      inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
      labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

      per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host


      tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

      for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
            tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

          mems_i = [tf.placeholder(tf.float32,
                                   [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                    for _ in range(FLAGS.n_layer)]

          loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
              n_token=n_token,
              cutoffs=cutoffs,
              is_training=True,
              inp=inputs[i],
              tgt=labels[i],
              mems=mems_i)

          tower_mems.append(mems_i)
          tower_losses.append(loss_i)
          tower_new_mems.append(new_mems_i)
          tower_grads_and_vars.append(grads_and_vars_i)



      ## sum losses across towers
      if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
      else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]

      global_step = tf.train.get_or_create_global_step()


      optimizer = DynamicEvalOpt(learning_rate=FLAGS.learning_rate,decay_rate = FLAGS.decay_rate,eps = FLAGS.epsilon)
      optimizer.gradstat = True
      train_op = optimizer.apply_gradients(grads_and_vars, global_step)


      tower_mems_np = [
          [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
              for layer in range(FLAGS.n_layer)]
          for core in range(FLAGS.num_core_per_host)
      ]

      saver = tf.train.Saver()

      with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())


        if FLAGS.eval_ckpt_path is None:
          eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
          eval_ckpt_path = FLAGS.eval_ckpt_path

        tf.logging.info("Evaluate {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems,tf.size(label_feed), train_op]


        total_loss, prev_step = 0., -1


        total_loss, total_cnt = 0, 0


        format_str = "  >> processing batch for gradient statistics {{:{0}d}}/{{:{0}d}} ..".format(
            len(str(num_batch//5000)))

        ## only small subset of training set used for gradient stats to save time
        for step in range(num_batch//5000):
          if step % (num_batch // 50000) == 0:
            tf.logging.info(format_str.format(step, num_batch//5000))

          feed_dict = {}
          for i in range(FLAGS.num_core_per_host):
            for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
              feed_dict[m] = m_np

          fetched = sess.run(fetches, feed_dict=feed_dict)

          loss_np, tower_mems_np, cnt_np = fetched[:3]
          total_loss += loss_np * cnt_np
          total_cnt += cnt_np

        avg_loss = total_loss / total_cnt
    ##    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
    ##        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))

#####Done gradstat






###starting dynamic eval

  eval_input_fn, eval_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split=FLAGS.eval_split,
      per_host_bsz=FLAGS.eval_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host,
      num_hosts=1,
      use_tpu=False)

  num_batch = eval_record_info["num_batch"]

  tf.logging.info("num of batches {}".format(num_batch))

  ##### Create computational graph
  eval_set = eval_input_fn({
      "batch_size": FLAGS.eval_batch_size,
      "data_dir": FLAGS.data_dir})

  input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

  inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
  labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host


  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

      mems_i = [tf.placeholder(tf.float32,
                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          n_token=n_token,
          cutoffs=cutoffs,
          is_training=True,
          inp=inputs[i],
          tgt=labels[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)



  ## sum losses across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    grads_and_vars = tower_grads_and_vars[0]





  ## configure the optimizer
  global_step = tf.train.get_or_create_global_step()
  if not FLAGS.rms:

    optimizer =  tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)# DynamicEvalPS(learning_rate=FLAGS.learning_rate )
  else:
      optimizer.gradstat = False
  train_op = optimizer.apply_gradients(grads_and_vars, global_step)

  ##### Evaluation loop
  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]
      for core in range(FLAGS.num_core_per_host)
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())


    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path

    tf.logging.info("Evaluate {}".format(eval_ckpt_path))
    saver.restore(sess, eval_ckpt_path)

    fetches = [loss, tower_new_mems,tf.size(label_feed), train_op]


    total_loss, prev_step = 0., -1


    total_loss, total_cnt = 0, 0
    format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
        len(str(num_batch)))
    for step in range(num_batch // FLAGS.ratio):
      if step % (num_batch // (10*FLAGS.ratio)) == 0:
        tf.logging.info(format_str.format(step, num_batch))

      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
          feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, cnt_np = fetched[:3]
      total_loss += loss_np * cnt_np
      total_cnt += cnt_np

    avg_loss = total_loss / total_cnt
    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))





def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]
  tf.logging.info("n_token {}".format(n_token))


  dynamic_eval(n_token, cutoffs, "/gpu:0")



if __name__ == "__main__":
  tf.app.run()
