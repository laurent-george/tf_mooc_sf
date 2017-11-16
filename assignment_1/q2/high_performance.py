"""
This module aims at providing function to create a high performance model

We aim at having 100 % gpu occupation, thus using stagging gpu etc
"""

from tensorflow.python.ops.data_flow_ops import StagingArea

def get_stage_op(next_batch):
    """
    shortcut to create a staging area and associated stage operation

    It is usefull for high performance model where we want to stage the next batch in the gpu memory
    in parallel of processing the current batch

    This is just a shortcut to avoid code duplication (it's not a real generic implementation right now)
    :param next_batch: a batch as a dictionary
    :return:  (stage put op, next_batch_from_stage_area tensor)
    """
    names = list(next_batch.keys())
    shapes = [i.shape for i in next_batch.values()]
    dtypes = [i.dtype for i in next_batch.values()]
    m = StagingArea(dtypes=dtypes, names=names, shapes=shapes, capacity=4)
    stage_put_op = m.put(next_batch)
    next_batch_from_stage_area = m.get()
    return stage_put_op, next_batch_from_stage_area,

