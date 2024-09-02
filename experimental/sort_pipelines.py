import numpy as np


def sort_all_pipelines(all_pipelines):
    for pipeline in all_pipelines:
        device_ids = [stage[0] for stage in pipeline[0]]
        sort_idxs = np.argsort(device_ids)

        for i in range(1, 3):
            pipeline[i] = np.take(pipeline[i], sort_idxs, axis=0).tolist()