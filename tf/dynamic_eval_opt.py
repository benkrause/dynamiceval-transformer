# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""GradientDescent for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf
@tf_export(v1=["train.GradientDescentOptimizer"])
class DynamicEvalOptimizer(optimizer.Optimizer):
  """Optimizer for dynamic evaluation. Equivelent to original paper's optimzer under a hyperparameter transformation
  """

  def __init__(self, learning_rate,decay_rate= 0.01, eps = 0.001,use_locking=False, name="Adam"):

    """Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: This is a hack, but set to Adam to be compatible with preloaded model. changing this causes a bug".


    """
    super(DynamicEvalOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._learning_rate_tensor = None
    self._decay_rate = decay_rate
    self._decay_rate_tensor = None
    self._eps= eps
    self._eps_tensor = None
    self.gradstat = True
  def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "dw", self._name)
            self._zeros_slot(v, "ss", self._name)



  def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        dr_t = math_ops.cast(self._decay_rate_tensor, var.dtype.base_dtype)
        eps_t = math_ops.cast(self._eps_tensor, var.dtype.base_dtype)

        m = self.get_slot(var, "dw")
        ss = self.get_slot(var, "ss")


        if self.gradstat:
            #if in this mode, collects statistics on training data without updatimg
            ss_t = ss.assign(ss+grad*grad)

            update = grad*0
        else:
            ss_t = ss.assign(ss+0)
    #        cnt_t = cnt.assign(cnt+0)

            wlr_t = lr_t/((tf.sqrt(ss_t))+eps_t)

            update = wlr_t*grad+dr_t*(tf.sqrt(ss_t))*m

        m_t = m.assign(m-update)
        var_update = state_ops.assign_sub(var,update) #Update 'ref' by subtracting 'value


        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update,m_t,ss_t])

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
    decay_rate = self._call_if_callable(self._decay_rate)
    self._decay_rate_tensor = ops.convert_to_tensor(
        decay_rate, name="decay_rate")
    eps = self._call_if_callable(self._eps)
    self._eps_tensor = ops.convert_to_tensor(
        eps, name="eps")
