import sys
import torch
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from dialogue.classify import configuration
from dialogue.classify.cv_experimenter import Experimenter
from dialogue.classify.feature_extractor import Feature_Extractor, ItemSelector
from dialogue.data.data_loader import data_loader, DataLoader, Utterance
from dialogue.skipthought.skipthoughts import BiSkip, BiSkipClassifier

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def init_Skip_Thought_dict(config):
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

    word2idx = {}
    for id, word in enumerate(vocab):
        word2idx[word] = id

    return vocab, word2idx

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

        fixed_emb = True
        hidden_size = 2400 * len(context_range)
        output_size = 4

        vocab, word2idx = init_Skip_Thought_dict(config)
        model = BiSkipClassifier(config['skipthought_model_path'], vocab, hidden_size=hidden_size, output_size=output_size, fixed_emb=fixed_emb, dropout=0.5)
        # optimizer = Adam(params=model.parameters(), lr=1e-4)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.i2o.parameters():
            p.requires_grad = True
        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        # optimizer = Adam(params=[model.i2o], lr=1e-4)
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

        if torch.cuda.is_available():
            model.cuda()
            print('Running on GPU!')

        all_training_losses = []
        all_valid_losses = []

        '''
        Training
        '''
        for epoch in range(epoch_num):

            training_losses = []
            for i, (x,y) in enumerate(zip(X_train_onehot, Y_train)):
                x = pad(x)
                y = Variable(torch.LongTensor([y.tolist()]))

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                output = model.forward(x).view(1,-1)
                loss = criterion.forward(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(loss.data[0])

                if i % 500 == 0:
                    print('Training %d/%d, loss=%.5f' % (i, len(Y_train), np.average(training_losses[-100:])))

            training_loss_mean = torch.mean(torch.FloatTensor(training_losses))
            all_training_losses.append(training_loss_mean)

            '''
            Validating
            '''
            valid_losses = []
            valid_pred   = []
            for i, (x_t, x,y) in enumerate(zip(X_valid_text, X_valid_onehot, Y_valid)):
                x = pad(x)
                y = Variable(torch.LongTensor([y.tolist()]))
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                output = model.forward(x).view(1,-1)

                # print('output_size      = %s' % str(output.size()))
                # print('y_tensor         = %s' % str(y.data))
                # print('y_tensor_size    = %s' % str(y.size()))
                loss = criterion.forward(output, y)
                valid_losses.append(loss.data[0])
                prob_i, pred_i = output.data.topk(1)

                if torch.cuda.is_available():
                    valid_pred.append(pred_i.cpu().numpy())
                else:
                    valid_pred.append(pred_i.numpy())

                if i % 500 == 0:
                    print('Validating %d/%d, loss=%.5f' % (i, len(Y_valid), np.average(valid_losses[-100:])))
                # if i > 20:
                #     break

            valid_loss_mean = torch.mean(torch.FloatTensor(valid_losses))
            all_valid_losses.append(valid_loss_mean)
            is_best_loss = valid_loss_mean < best_loss

            print('*' * 50)
            print('Epoch=%d' % epoch)
            print('Training loss=%.5f' % training_loss_mean)
            print('Valid loss=%.5f' % valid_loss_mean)
            if is_best_loss:
                print('Update best loss=%.5f, last bess loss=%.5f' % (valid_loss_mean, best_loss))
            else:
                print('Best loss is not updated, =%.5f' % best_loss)

            best_loss = min(valid_loss_mean, best_loss)


            print("Classification report:")
            valid_pred = [v[0][0] for v in valid_pred]
            report = metrics.classification_report(Y_valid, valid_pred,
                                                   target_names=np.asarray(config['label_encoder'].classes_))
            print(report)

            print("confusion matrix:")
            confusion_mat = str(metrics.confusion_matrix(Y_valid, valid_pred))
            print('\n' + confusion_mat)

            acc_score = metrics.accuracy_score(Y_valid, valid_pred)
            f1_score = metrics.f1_score(Y_valid, valid_pred, average='macro')

            print("accuracy:   %0.3f" % acc_score)
            print("f1_score:   %0.3f" % f1_score)

            print('*' * 50)

            plt.plot(all_training_losses)
            plt.figure()

            if is_best_loss:
                stop_increasing = 0
            else:
                stop_increasing += 1

            if stop_increasing >= 2:
                print('Have not increased for %d epoches, stop training' % stop_increasing)
                break

        '''
        Testing
        '''
        test_pred   = []
        test_losses = []
        for i, (x,y) in enumerate(zip(X_test_onehot, Y_test)):
            x = pad(x)
            y = Variable(torch.LongTensor([y.tolist()]))

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            output = model.forward(x).view(1,-1)
            prob_i, pred_i = output.data.topk(1)

            if torch.cuda.is_available():
                test_pred.append(pred_i.cpu().numpy())
            else:
                test_pred.append(pred_i.numpy())

            test_losses.append(loss.data[0])

            if i % 500 == 0:
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

        print("confusion matrix:")
        confusion_mat = str(metrics.confusion_matrix(Y_test, test_pred))
        print('\n' + confusion_mat)

        acc_score = metrics.accuracy_score(Y_test, test_pred)
        f1_score = metrics.f1_score(Y_test, test_pred, average='macro')

        print("accuracy:   %0.3f" % acc_score)
        print("f1_score:   %0.3f" % f1_score)

        print('*' * 50)

