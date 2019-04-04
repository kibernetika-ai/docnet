from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import tensorflow as tf
import numpy as np
import sys


def convert(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, np.int32): return int(o)
    if isinstance(o, np.int): return int(o)
    if isinstance(o, np.float32): return float(o)
    if isinstance(o, np.float64): return float(o)
    if isinstance(o, np.float): return float(o)
    if isinstance(o, tf.int64): return int(o)
    if isinstance(o, tf.int32): return int(o)
    if isinstance(o, tf.int): return int(o)
    if isinstance(o, tf.float32): return float(o)
    if isinstance(o, tf.float64): return float(o)
    if isinstance(o, tf.float): return float(o)
    return None


class MlBoardReporter(session_run_hook.SessionRunHook):
    def __init__(self, tensors={}, every_steps=None, every_n_secs=60):
        if every_steps is not None:
            every_n_secs = None
        self._timer = tf.train.SecondOrStepTimer(every_steps=every_steps, every_secs=every_n_secs)
        try:
            from mlboardclient.api import client
        except ImportError:
            tf.logging.warning("Can't find mlboardclient.api")
            client = None
        mlboard = None
        if client:
            mlboard = client.Client()
            try:
                mlboard.apps.get()
            except Exception:
                tf.logging.warning("Can't init mlboard env")
                mlboard = None

        self._mlboard = mlboard
        self._tensors = tensors

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        requests = {"global_step": self._global_step_tensor}
        for n, t in self._tensors.items():
            requests[n] = t
        self._generate = (
                self._next_step is None or
                self._timer.should_trigger_for_step(self._next_step))

        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        _ = run_context
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._next_step is None or self._generate:
            global_step = run_context.session.run(self._global_step_tensor)

        if self._mlboard is not None:
            if self._generate and (self._next_step is not None):
                self._timer.update_last_triggered_step(global_step)
                rpt = {}
                for k, v in run_values.results.items():
                    if k == "global_step":
                        continue
                    v = convert(v)
                    if v is not None:
                        rpt[k] = v
                if len(rpt) > 0:
                    try:
                        self._mlboard.update_task_info(rpt)
                    except:
                        print('Unexpected error during submit state: {}'.format(sys.exc_info()[0]))
        self._next_step = global_step + 1
