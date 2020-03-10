# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf


class SyncExitHook(tf.train.SessionRunHook):
  def __init__(self, num_workers, task_index, is_chief):
    self._task_index = task_index
    self._is_chief = is_chief
    self._num_workers = num_workers
    self._counter_vars = []
    self._counter_add_ops = []
    for i in range(self._num_workers):
      counter_var = tf.Variable(0, name="num_finished_workers-{}".format(i), collections=[tf.GraphKeys.LOCAL_VARIABLES])
      self._counter_vars.append(counter_var)
      counter_var_ops = tf.assign(counter_var, 1, use_locking=True)
      self._counter_add_ops.append(counter_var_ops)

  def end(self, session):
    session.run(self._counter_add_ops[self._task_index])
    num_finished_workers = 0
    while True:
      for i in range(self._num_workers):
        state = session.run(self._counter_vars[i])
        if i == self._task_index and state == 0:
          session.run(self._counter_add_ops[i])
          state = session.run(self._counter_vars[i])

        num_finished_workers = num_finished_workers + state

      tf.logging.info("%d workers have finished ...", num_finished_workers)
      if num_finished_workers >= self._num_workers:
        break

      num_finished_workers = 0
      time.sleep(1)