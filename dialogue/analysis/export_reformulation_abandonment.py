import os
from sklearn import metrics

import numpy as np

import pickle

from classify import configuration
from classify.feature_extractor import Feature_Extractor
from classify.task_runner import preload_X_Y
from dialogue.data.data_loader import data_loader

if __name__ == '__main__':
    experiment_path = '/Users/memray/Project/yahoo/y_classify/output/feature_comparison/context=next.similarity=true/'

    config = configuration.load_basic_config()
    extractor = Feature_Extractor(config)

    data_dict = {}

    # iterate each dataset, load X and Y, put them into dict
    for data_name in config['data_names']:
        config.param['data_name'] = data_name
        loader = data_loader(data_name, {'config': config})
        # load annotated data
        session_ids, annotated_sessions = loader.load_annotated_data()
        # train and test
        X_raw, Y, label_encoder = extractor.split_to_instances(annotated_sessions)
        X, feature_names        = extractor.extract()
        X_raw_feature           = extractor.extract_raw_feature()

        data_dict[data_name]    = (X_raw, X_raw_feature, feature_names, label_encoder, X, Y)

    for folder in os.listdir(experiment_path):
        if folder == '.DS_Store':
            continue

        all_prfs_list = []
        for data_id, data_name in enumerate(config['data_names']):
            prfs_dict  = {}
            print('%s - %s' % (folder, data_name))

            X_raw, X_raw_feature, feature_names, label_encoder, X_all, Y = data_dict[data_name]

            with open(os.path.join(experiment_path, folder, data_name + '.test.pkl'), 'rb') as pkl_file:
                results = pickle.load(pkl_file)

            labels = np.asarray(label_encoder.classes_)

            with open(os.path.join(experiment_path, folder, data_name + '.test.detail.csv'), 'w') as csv_file:
                target_names = ['%s' % l for l in labels]
                # write header
                head = 'dataset'
                for i, label in enumerate(labels):
                    for v in (['precision', 'recall', 'f1-score', 'support']):
                        head += ','+label + '-' + v
                csv_file.write(head+'\n')

                for r_id, result in enumerate(results):
                    print('%s - %s : %d' % (folder, data_name, r_id))
                    result = result[0]
                    y_pred = result['y_pred']
                    y_true = result['y_test']
                    # report = metrics.classification_report(y_true, y_pred, target_names=labels)
                    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred)
                    # print(report)

                    # write contents
                    line = data_name
                    for i, label in enumerate(labels):
                        line += ",%f,%f,%f,%d" % (p[i], r[i], f1[i], s[i])
                        for metric_name, metric_value in (zip(['precision', 'recall', 'f1-score', 'support'], [p[i], r[i], f1[i], s[i]])):
                            prfs_list = prfs_dict.get('%s-%s' % (label, metric_name), [])
                            prfs_list.append(metric_value)
                            prfs_dict['%s-%s' % (label, metric_name)] = prfs_list

                    csv_file.write(line+'\n')

            if data_id == 0:
                with open(os.path.join(experiment_path, folder, 'all.test.detail.csv'), 'w') as csv_file:
                    head = 'dataset'
                    for i, label in enumerate(labels):
                        for v in (['precision', 'recall', 'f1-score', 'support']):
                            head += ',' + label + '-' + v
                    csv_file.write(head + '\n')

            with open(os.path.join(experiment_path, folder, 'all.test.detail.csv'), 'a') as csv_file:
                line = data_name

                for i, label in enumerate(labels):
                    for metric_name in ['precision', 'recall', 'f1-score', 'support']:
                        v = np.average(prfs_dict.get('%s-%s' % (label, metric_name)))
                        line += ",%f" % v

                csv_file.write(line+'\n')

            all_prfs_list.append(prfs_dict)

        with open(os.path.join(experiment_path, folder, 'all.test.detail.csv'), 'a') as csv_file:
            line = 'all'

            for i, label in enumerate(labels):
                for metric_name in ['precision', 'recall', 'f1-score', 'support']:
                    v = np.average([prfs_dict.get('%s-%s' % (label, metric_name)) for prfs_dict in all_prfs_list])
                    line += ",%f" % v

            csv_file.write(line + '\n')

