import sys
import torch
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from classify import configuration
from classify.cv_experimenter import Experimenter
from classify.feature_extractor import Feature_Extractor, ItemSelector
from dialogue.data.data_loader import data_loader, DataLoader, Utterance
from skipthought.skipthoughts import BiSkip, BiSkipClassifier

import numpy as np

def str_to_one_hot(str, word2idx):
    '''
    Given a str, return its ID
    :param doc_str:
    :return:
    '''
    str = str.strip().lower()
    word_list = str.split()
    one_hot = [word2idx[w] for w in word_list if w in word2idx]

    if len(str) == 0 or len(one_hot) == 0:
        word_list = ['UNK']
        one_hot = [word2idx[w] for w in word_list if w in word2idx]

    # print(word_list)
    # print(one_hot)

    return one_hot

def init_Skip_Thought(config, hidden_size, output_size):
    word2idx = {}

    '''
    Load or train Doc2Vec
    '''
    dir_st = config['skipthought_model_path']

    vocab = set()

    all_sessions = config['data_loader']()

    document_dict = {}
    for session in all_sessions:
        for utt in session:
            if utt.msg_text not in document_dict:
                words = utt.msg_text.strip().lower().split()
                [vocab.add(w) for w in words]
                document_dict[utt.msg_text] = (utt.msg_text, words)

    vocab       = ['<eos>', 'UNK'] + list(vocab)
    for id, word in enumerate(vocab):
        word2idx[word] = id

    return BiSkipClassifier(dir_st, vocab, hidden_size=hidden_size, output_size=output_size), word2idx

if __name__ == '__main__':
    # initialize
    config = configuration.load_config()
    extractor = Feature_Extractor(config)
    exp = Experimenter(config)

    best_results = {}
    # iterate each dataset
    for data_name in config['data_names']:
        config.param['data_name'] = data_name

        config.logger.info('*' * 50)
        config.logger.info('-' * 20 + data_name + '-' * 20)
        config.logger.info('*' * 50)
        # initialize data_loader
        loader = data_loader(data_name, {'config': config})
        config['data_loader'] = loader
        loader()
        # load annotated data
        session_ids, annotated_sessions = loader.load_annotated_data()
        loader.stats()

        # train and test
        X_raw, Y                = extractor.split_to_instances(annotated_sessions)
        X_raw_feature           = extractor.extract_raw_feature()

        if config.param['context_set'] == 'current':
            context_range = ['current_user_utterance']
        if config.param['context_set'] == 'next':
            context_range = ['current_user_utterance', 'next_system_utterance', 'next_user_utterance']
        if config.param['context_set'] == 'last':
            context_range = ['last_user_utterance', 'last_system_utterance', 'current_user_utterance']
        if config.param['context_set'] == 'all':
            context_range = ['last_user_utterance', 'last_system_utterance', 'current_user_utterance', 'next_system_utterance', 'next_user_utterance']


        X_texts     = ItemSelector(keys=context_range).transform(X_raw_feature)
        X_texts     = list(zip(*X_texts))

        hidden_size = 2400 * len(context_range)
        output_size = 4

        model, word2idx = init_Skip_Thought(config, hidden_size, output_size)
        optimizer = Adam(params=model.parameters(), lr=1e-4)
        criterion = nn.NLLLoss()

        X_onehot = [[str_to_one_hot(sent, word2idx) for sent in s] for s in X_texts]

        train_id, valid_id, test_id = exp.load_single_run_index(X_texts, Y)
        X_train_text   = [X_texts[i] for i in train_id]
        X_valid_text   = [X_texts[i] for i in valid_id]
        X_test_text    = [X_texts[i] for i in test_id]

        X_train_onehot = [X_onehot[i] for i in train_id]
        Y_train        = [Y[i] for i in train_id]
        X_valid_onehot = [X_onehot[i] for i in valid_id]
        Y_valid        = [Y[i] for i in valid_id]
        X_test_onehot  = [X_onehot[i] for i in test_id]
        Y_test         = [Y[i] for i in test_id]

        epoch_num  = 20
        best_loss = sys.float_info.max

        def pad(x):
            max_length = len(sorted(x, key=len, reverse=True)[0])
            x_new = np.array([xi + [0] * (max_length - len(xi)) for xi in x])
            x_new = Variable(torch.from_numpy(x_new))
            return x_new
                # torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        for epoch in range(epoch_num):

            training_losses = []
            for i, (x,y) in enumerate(zip(X_train_onehot, Y_train)):
                x = pad(x)
                # y = Variable(torch.LongTensor(np.asarray([1  if i==y else 0 for i in range(output_size)]))).view(1,-1)
                y = Variable(torch.LongTensor([y.tolist()]))

                output = model.forward(x).view(1,-1)
                loss = criterion.forward(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(loss.data[0])

                print('Training %d/%d, loss=%.5f' % (i, len(Y_train), loss.data[0]))
                if i > 10:
                    break

            training_loss_mean = torch.mean(torch.FloatTensor(training_losses))

            valid_losses = []
            valid_pred   = []
            for i, (x_t, x,y) in enumerate(zip(X_valid_text, X_valid_onehot, Y_valid)):
                x = pad(x)

                # if i < 164:
                #     continue
                # print('*' * 20 + str(i) + '*' * 20)
                # print('x = ')
                # print(x_t)
                # # print(x)
                # print('y = [%d]' % y)
                # print(x.size())

                output = model.forward(x).view(1,-1)
                y = Variable(torch.LongTensor([y.tolist()]))

                # print('output_size      = %s' % str(output.size()))
                # print('y_tensor         = %s' % str(y.data))
                # print('y_tensor_size    = %s' % str(y.size()))
                loss = criterion.forward(output, y)
                valid_losses.append(loss.data[0])
                prob_i, pred_i = output.data.topk(1)
                valid_pred.append(pred_i.numpy())

                print('Validating %d/%d, loss=%.5f' % (i, len(Y_valid), loss.data[0]))
                # if i > 20:
                #     break


            valid_loss_mean = torch.mean(torch.FloatTensor(valid_losses))
            is_best_loss = valid_loss_mean < best_loss
            best_loss = min(valid_loss_mean, best_loss)

            print('*' * 50)
            print('Epoch=%d' % epoch)
            print('Training loss=%.5f' % training_loss_mean)
            print('Valid loss=%.5f' % valid_loss_mean)
            print('Best loss=%.5f' % best_loss)


            print("Classification report:")
            valid_pred = [v[0][0] for v in valid_pred]
            report = metrics.classification_report(Y_valid, valid_pred,
                                                   target_names=np.asarray(config['label_encoder'].classes_))
            print(report)
            print('*' * 50)

            if is_best_loss:
                stop_increasing = 0
            else:
                stop_increasing += 1

            if stop_increasing >= 0:
                print('Have not increased for %d epoches, stop training' % stop_increasing)


        test_pred   = []
        test_losses = []
        for i, (x,y) in enumerate(zip(X_test_onehot, Y_test)):
            x = pad(x)
            y = Variable(torch.LongTensor([y.tolist()]))

            # print(x_t)
            # print(x)
            # print(x.size())

            optimizer.zero_grad()

            output = model.forward(x).view(1,-1)
            prob_i, pred_i = output.data.topk(1)
            test_pred.append(pred_i.numpy())
            test_losses.append(loss.data[0])

            print('Testing %d/%d, loss=%.5f' % (i, len(Y_test), loss.data[0]))

            # if i > 20:
            #     break

        valid_loss_mean = torch.mean(torch.FloatTensor(test_losses))
        test_pred = [v[0][0] for v in test_pred]
        print('*' * 50)
        print('Testing loss=%.5f' % valid_loss_mean)
        print("Classification report:")
        report = metrics.classification_report(Y_test, test_pred,
                                               target_names=np.asarray(config['label_encoder'].classes_))
        print(report)
        print('*' * 50)

