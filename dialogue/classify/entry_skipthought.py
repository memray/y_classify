import sys
import torch
from sklearn import metrics
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from dialogue.classify import configuration
from dialogue.classify.exp_shallowmodel import ShallowExperimenter
from dialogue.classify.feature_extractor import Feature_Extractor, ItemSelector
from dialogue.data.data_loader import data_loader, DataLoader, Utterance
from dialogue.deep.skipthought.skipthoughts import BiSkip, BiSkipClassifier

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_learning_curve(train_scores, test_scores, title, curve1_name='curve1_name', curve2_name='curve2_name', ylim=None, save_path=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    train_sizes=np.linspace(.1, 1.0, len(train_scores))
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label=curve1_name)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label=curve2_name)

    plt.legend(loc="best")
    # plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return plt

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
    exp = ShallowExperimenter(config)

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
        X_raw, Y, label_encoder = extractor.split_to_instances(annotated_sessions)
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
        sentence_num = len(context_range)
        hidden_size  = 2400 * len(context_range)
        output_size  = 4

        vocab, word2idx = init_Skip_Thought_dict(config)
        model = BiSkipClassifier(config['skipthought_model_path'], vocab, hidden_size=hidden_size, output_size=output_size, sentence_num=sentence_num, fixed_emb=fixed_emb, dropout=0.5)

        if torch.cuda.is_available():
            model.cuda()
            print('Running on GPU!')

        '''
        clip most of the parameters except for the softmax
        '''
        # optimizer = Adam(params=model.parameters(), lr=1e-4)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True


        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        # optimizer = Adam(params=[model.i2o], lr=1e-4)
        # criterion = nn.NLLLoss()
        criterion = nn.CrossEntropyLoss()

        def pad(x):
            x_ = x.flatten()
            max_length = len(sorted(x_, key=len, reverse=True)[0])
            x_new = np.array([[v + [0] * (max_length - len(v)) for v in xi] for xi in x])
            x_new = torch.from_numpy(x_new)
            return x_new
                # torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        X_onehot = pad(np.asarray([[str_to_one_hot(sent, word2idx) for sent in s] for s in X_texts]))

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
        batch_size = 128

        best_loss = sys.float_info.max
        stop_increasing = 0
        all_training_losses = []
        all_valid_losses = []
        all_accuracy = []
        all_f1_score = []

        train_batch_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train_onehot, Y_train)), batch_size=batch_size, shuffle=True, drop_last=False)
        valid_batch_loader = torch.utils.data.DataLoader(dataset=list(zip(X_valid_onehot, Y_valid)), batch_size=batch_size, shuffle=False, drop_last=False)
        test_batch_loader  = torch.utils.data.DataLoader(dataset=list(zip(X_test_onehot, Y_test)), batch_size=batch_size, shuffle=False, drop_last=False)
        if torch.cuda.is_available():
            train_batch_loader.pin_memory = True
            valid_batch_loader.pin_memory = True
            test_batch_loader.pin_memory = True

        for epoch in range(epoch_num):
            '''
            Training
            '''
            model.train()
            training_losses = []
            for i, (x,y) in enumerate(train_batch_loader):
                x = Variable(x)
                y = Variable(y)

                output = model.forward(x)
                loss = criterion.forward(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(loss.data[0])

                print('Training %d/%d, loss=%.5f, norm=%.5f' % (i, len(Y_train)/batch_size, np.average(loss.data[0]), float(model.fc.weight.norm().data.numpy())))
            all_training_losses.append(training_losses)
            training_loss_mean = np.average(training_losses)

            '''
            Validating
            '''
            model.eval()
            valid_losses = []
            valid_pred   = []
            for i, (x, y) in enumerate(valid_batch_loader):
                x = Variable(x)
                y = Variable(y)

                output = model.forward(x)
                loss = criterion.forward(output, y)
                valid_losses.append(loss.data[0])
                prob_i, pred_i = output.data.topk(1)

                if torch.cuda.is_available():
                    valid_pred.extend(pred_i.cpu().numpy().flatten().tolist())
                else:
                    valid_pred.extend(pred_i.numpy().flatten().tolist())

                print('Validating %d/%d, loss=%.5f' % (i, len(Y_valid)/batch_size, np.average(loss.data[0])))

            valid_loss_mean = np.average(valid_losses)
            all_valid_losses.append(valid_losses)

            print('*' * 50)
            print('Epoch=%d' % epoch)
            print('Training loss=%.5f' % training_loss_mean)
            print('Valid loss=%.5f' % valid_loss_mean)

            print("Classification report:")
            report = metrics.classification_report(Y_valid, valid_pred,
                                                   target_names=np.asarray(config['label_encoder'].classes_))
            print(report)

            print("confusion matrix:")
            confusion_mat = str(metrics.confusion_matrix(Y_valid, valid_pred))
            print('\n' + confusion_mat)

            acc_score = metrics.accuracy_score(Y_valid, valid_pred)
            f1_score = metrics.f1_score(Y_valid, valid_pred, average='macro')
            all_accuracy.append([acc_score])
            all_f1_score.append([f1_score])

            print("accuracy:   %0.3f" % acc_score)
            print("f1_score:   %0.3f" % f1_score)

            is_best_loss = f1_score < best_loss
            rate_of_change = float(f1_score - best_loss)/float(best_loss)

            if is_best_loss:
                print('Update best f1 (%.4f --> %.4f), rate of change (ROC)=%.2f' % (best_loss, f1_score, rate_of_change * 100))
            else:
                print('Best f1 is not updated (%.4f --> %.4f), rate of change (ROC)=%.2f' % (best_loss, f1_score, rate_of_change * 100))

            best_loss = min(f1_score, best_loss)

            print('*' * 50)

            if rate_of_change > -0.01:
                stop_increasing += 1
            else:
                stop_increasing = 0

            if stop_increasing >= config['early_stop_tolerance']:
                print('Have not increased for %d epoches, stop training' % stop_increasing)
                break

        plot_learning_curve(all_training_losses, all_valid_losses, 'Training and Validation', curve1_name='Training Error', curve2_name='Validation Error', save_path=config['experiment_path']+'/%s-train_valid_curve.png' % data_name)
        plot_learning_curve(all_accuracy, all_f1_score, 'Accuracy and F1-score', curve1_name='Accuracy', curve2_name='F1-score', save_path=config['experiment_path']+'/%s-train_f1_curve.png' % data_name)

        '''
        Testing
        '''
        model.eval()
        test_pred   = []
        test_losses = []
        for i, (x, y) in enumerate(test_batch_loader):
            x = Variable(x)
            y = Variable(y)

            output = model.forward(x)
            loss = criterion.forward(output, y)
            test_losses.append(loss.data[0])
            prob_i, pred_i = output.data.topk(1)

            if torch.cuda.is_available():
                test_pred.extend(pred_i.cpu().numpy().flatten().tolist())
            else:
                test_pred.extend(pred_i.numpy().flatten().tolist())

            test_losses.append(loss.data[0])

            print('Testing %d/%d, loss=%.5f' % (i, len(Y_test)/batch_size, loss.data[0]))

        test_loss_mean = np.average(test_losses)
        print('*' * 50)
        print('Testing loss=%.5f' % test_loss_mean)
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

