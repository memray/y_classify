import copy
import multiprocessing
import os
import argparse
# from multiprocessing import Queue
from queue import Queue
import time
import numpy as np

from multiprocessing import freeze_support
from multiprocessing import current_process

from dialogue.classify import exp_deepmodel
from dialogue.classify import exp_shallowmodel
from dialogue.classify import configuration
from dialogue.classify.feature_extractor import Feature_Extractor
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

def init_task_queue(selected_context_id, selected_feature_set_id, is_deep_model, add_similarity_feature):
    # queue            = Queue()
    queue = []
    # parameter_ranges = {'selected_context_id': [0], 'deep_model': [True], 'deep_model_name': ['cnn']}
    parameter_ranges = {'deep_model': [is_deep_model], 'selected_context_id': selected_context_id, 'selected_feature_set_id': selected_feature_set_id
, 'similarity_feature': [add_similarity_feature]}
    params           = []
    range_to_params(list(parameter_ranges.items()), params, [])

    # [queue.put(dict(p)) for p in params]
    [queue.append(dict(p)) for p in params]

    print('No. of param settings = %d' % len(params))
    for param in params:
        print(param)

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
        X_raw_feature           = extractor.extract_raw_feature()

        data_dict[data_name]    = (X_raw, X_raw_feature, feature_names, label_encoder, X, Y)

    return data_dict

def filter_X_by_contexts_features(X, config):
    X = np.nan_to_num(X.todense())

    if config.param['context_set'] == 'current':
        excluded_context_keywords = ['next', 'last']
    if config.param['context_set'] == 'next':
        excluded_context_keywords = ['last']
    if config.param['context_set'] == 'last':
        excluded_context_keywords = ['next']
    if config.param['context_set'] == 'all':
        excluded_context_keywords = []

    retained_feature_indices = []
    retained_feature_names = []

    # get the features
    for f_id,f_name in enumerate(config['feature_names']):
        f_start_number = f_name[0:f_name.find('-')]
        if f_start_number.find('.') > 0:
            f_start_number = f_name[0:f_start_number.find('.')]

        if f_start_number not in config.param['feature_set_number']:
            continue

        if not config.param['similarity_feature'] and (f_name.find('similarity') > 0 or f_name.find('overlap') > 0 or f_name.find('distance') > 0):
            continue

        within_context = True
        for context_keyword in excluded_context_keywords:
            if f_name.find(context_keyword) > 0:
                within_context = False
        if not within_context:
            continue

        retained_feature_indices.append(f_id)
        retained_feature_names.append(f_name)

    X_new = copy.deepcopy(X)[:, retained_feature_indices]
    config.logger.info('%' * 50)
    # config.logger.info('retained features: [%s]' % (','.join(retained_feature_names)))
    config.logger.info('context=%s, feature=%s' % (
        config.param['context_set'], config.param['feature_set']))
    config.logger.info(
        'retained feature numbers=[%s]' % ', '.join(list(set([f[0:f.find('-')] for f in retained_feature_names]))))
    config.logger.info('#(data)=%d' % X_new.shape[0])
    config.logger.info('#(feature)=%d' % X_new.shape[1])
    config.logger.info('%' * 50)

    return X_new

def worker(q, data_dict):
    # get a param from queue and
    for param in q:#iter(q.get, None):
        config  = configuration.load_batch_config(param)

        if config['deep_model']:
           exp = exp_deepmodel.DeepExperimenter(config)
        else:
           exp = exp_shallowmodel.ShallowExperimenter(config)

        results = []

        for data_name in config['data_names']:
            config.param['data_name']  = data_name
            X_raw, X_raw_feature, feature_names, label_encoder, X, Y = data_dict[data_name]
            config['feature_names']    = feature_names
            config['label_encoder']    = label_encoder
            config['X_raw']            = X_raw
            config['X_raw_feature']    = X_raw_feature
            config['Y']                = Y

            X_new = filter_X_by_contexts_features(X, config)

            result = exp.run_cross_validation(X_new, Y)
            results.extend(result)

        exp.export_averaged_summary(results, os.path.join(config.param['experiment_path'], 'summary.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='task_runner.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-selected_context_id', required=True, nargs='+', type=int,
                        help="")
    parser.add_argument('-selected_feature_set_id', required=True, nargs='+', type=int,
                        help="")
    parser.add_argument('-is_deep_model', action='store_true',
                        help="")
    parser.add_argument('-add_similarity_feature', action='store_true',
                        help="")
    parser.add_argument('-num_worker', default=1, type=int,
                        help="")
    opt = parser.parse_args()

    # freeze_support()
    n_workers   = opt.num_worker
    workers     = []
    q           = init_task_queue(opt.selected_context_id, opt.selected_feature_set_id, opt.is_deep_model, opt.add_similarity_feature)
    data_dict   = preload_X_Y()

    worker(q, data_dict)

    print('Done: %s' % (time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))))

    '''
    for i in range(n_workers):
        time.sleep(5)
        p = multiprocessing.Process(target = worker, args = (q, data_dict))
        workers.append(p)
        p.start()

    # stop workers
    for i in range(n_workers):
        q.put(None)
        workers[i].join()
    '''
