import copy
import multiprocessing
import os
from multiprocessing import Queue

from multiprocessing import freeze_support
from multiprocessing import current_process

from classify import configuration
from classify.cv_experimenter import Experimenter
from classify.feature_extractor import Feature_Extractor
from dialogue.data.data_loader import data_loader, DataLoader, Utterance

def range_to_params(ranges_items, params, cache):
    '''
    Search all the combinations
    :param ranges:
    :return:
    '''
    if len(ranges_items) == 0:
        params.append(cache)
        return
    k, vlist  = ranges_items[0]
    next_range = ranges_items[1:]

    for v in vlist:
        c = copy.deepcopy(cache)
        c.append((k,v))
        range_to_params(next_range, params, c)



def init_task_queue():
    queue            = Queue()
    parameter_ranges = {'selected_context_id': [0], 'selected_feature_set_id': [2,6], 'similarity_feature': [False]}
    params           = []
    range_to_params(list(parameter_ranges.items()), params, [])
    # parameter_ranges = {'selected_context_id': list(range(1, 2)), 'selected_feature_set_id': list(range(0, 13)), 'similarity_feature': [True]}
    # print(params)

    [queue.put(dict(p)) for p in params]

    print('No. of param settings = %d' % len(params))

    return queue

def preload_X_Y():
    # initialize a very basic config, don't create folder
    config = configuration.load_basic_config()
    extractor = Feature_Extractor(config)

    data_dict = {}

    # iterate each dataset, load X and Y, put them into dict
    for data_name in config['data_names']:
        config.param['data_name'] = data_name
        loader = data_loader(data_name, {'config': config})
        config['data_loader'] = loader
        loader()
        # load annotated data
        session_ids, annotated_sessions = loader.load_annotated_data()
        # train and test
        X_raw, Y, label_encoder = extractor.split_to_instances(annotated_sessions)
        X, feature_names        = extractor.extract()
        data_dict[data_name]    = (X_raw, feature_names, label_encoder, X, Y)

    return data_dict


def worker(q, data_dict):
    # get a param from queue and
    for param in iter(q.get, None):
        config  = configuration.load_batch_config(param)
        exp = Experimenter(config)

        results = []

        for data_name in config['data_names']:
            config.param['data_name']  = data_name
            X_raw, feature_names, label_encoder, X, Y = data_dict[data_name]
            config['feature_names']    = feature_names
            config['label_encoder']    = label_encoder
            config['X_raw']            = X_raw
            config['Y']                = Y
            # result = exp.run_single_pass_context_feature(X, Y)
            result = exp.run_cross_validation(X, Y)
            results.extend(result)

        exp.export_averaged_summary(results, os.path.join(config.param['experiment_path'], 'summary.csv'))

if __name__ == '__main__':
    freeze_support()
    n_workers   = 2
    workers     = []
    q           = init_task_queue()

    data_dict   = preload_X_Y()

    # worker(q, data_dict)

    for i in range(n_workers):
        p = multiprocessing.Process(target = worker, args = (q, data_dict))
        workers.append(p)
        p.start()

    # stop workers
    for i in range(n_workers):
        q.put(None)