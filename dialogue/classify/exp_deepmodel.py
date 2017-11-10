import gensim
import numpy as np
import sys
import torch
from sklearn import metrics, preprocessing
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import Adagrad
from torch.utils.data.dataloader import DataLoader
import operator
import matplotlib.pyplot as plt


from dialogue.classify.exp_shallowmodel import ShallowExperimenter
from dialogue.classify.feature_extractor import ItemSelector
from dialogue.deep.cnn.model import CNN
from dialogue.deep.rnn.model import LSTM
from dialogue.deep.skipthought.skipthoughts import BiSkipClassifier

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)

class DeepExperimenter(ShallowExperimenter):
    def init_model(self, vocab_dict, model_type):
        if model_type == 'cnn':
            params = self.config['cnn_setting']
            if params["MODEL"] != "rand":
                # load word2vec
                print("loading word2vec...")
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format(self.config['w2v_path'], binary=True)

                wv_matrix = []
                for i in range(len(vocab_dict["vocab"])):
                    word = vocab_dict["idx_to_word"][i]
                    if word in word_vectors.vocab:
                        wv_matrix.append(word_vectors.word_vec(word))
                    else:
                        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

                # one for UNK and one for zero padding
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
                wv_matrix.append(np.zeros(300).astype("float32"))
                wv_matrix = np.array(wv_matrix)
                params["WV_MATRIX"] = wv_matrix

            params['VOCAB_SIZE'] = len(vocab_dict['vocab'])
            params['MAX_SENT_LEN'] = vocab_dict['MAX_SENT_LEN']
            model = CNN(**params)
            model.parameters_to_normalize = [model.fc]

            if torch.cuda.is_available():
                model.cuda()

        elif model_type == 'skip-thought':
            params = self.config['skipthought_setting']
            model = BiSkipClassifier(params['skipthought_model_path'], vocab_dict["vocab"], hidden_size=params['hidden_size'],
                                     output_size=params['CLASS_SIZE'], sentence_num=params['sentence_num'],
                                     fixed_emb=params['fixed_emb'], dropout=params['DROPOUT_PROB'])
            '''
            make most of the parameters static except for the fc layer
            '''
            # optimizer = Adam(params=model.parameters(), lr=1e-4)
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True

            model.parameters_to_normalize = [model.fc]

        elif model_type == 'lstm':
            params = self.config['lstm_setting']
            params['max_length'] = vocab_dict['MAX_SENT_LEN']
            model = LSTM(len(vocab_dict["vocab"]), params['embedding_size'], params['hidden_size'], output_size=params['CLASS_SIZE'] , batch_size=self.config['batch_size'], dropout=params['DROPOUT_PROB'])
            model.parameters_to_normalize = [model.fc]
        else:
            self.logger.error('No such model: %s' % model_type)

        return model, params

    def get_batch_loader(self, X_raw_feature, Y, train_id, valid_id, test_id, batch_size = 32):
        def convert_and_pad(X_texts, concat_sents = True):
            if concat_sents:
                # pad a zero between each pair of sentences
                x = np.asarray([np.concatenate([str_to_one_hot(sent, word_to_idx) + [0] for sent in s]) for s in X_texts])
                x_ = x.flatten()
                max_length = len(sorted(x_, key=len, reverse=True)[0])
                x_new = np.asarray([xi.tolist() + [0] * (max_length - len(xi)) for xi in x])
            else:
                x = np.asarray([[str_to_one_hot(sent, word_to_idx) for sent in s] for s in X_texts])
                x_ = x.flatten()
                max_length = len(sorted(x_, key=len, reverse=True)[0])
                x_new = np.array([[v + [0] * (max_length - len(v)) for v in xi] for xi in x])
            x_new = torch.from_numpy(x_new)
            return x_new, max_length


        def str_to_one_hot(str, word2idx):
            str = str.strip().lower()
            word_list = str.split()
            one_hot = [word2idx[w] for w in word_list if w in word2idx]

            if len(str) == 0 or len(one_hot) == 0:
                word_list = ['UNK']
                one_hot = [word2idx[w] for w in word_list if w in word2idx]

            return one_hot

        vocab_dict = {}
        data_dict = {}

        if self.config.param['context_set'] == 'current':
            context_range = ['current_user_utterance']
        if self.config.param['context_set'] == 'next':
            context_range = ['current_user_utterance', 'next_system_utterance', 'next_user_utterance']
        if self.config.param['context_set'] == 'last':
            context_range = ['last_user_utterance', 'last_system_utterance', 'current_user_utterance']
        if self.config.param['context_set'] == 'all':
            context_range = ['last_user_utterance', 'last_system_utterance', 'current_user_utterance', 'next_system_utterance', 'next_user_utterance']

        X_texts     = ItemSelector(keys=context_range).transform(X_raw_feature)
        X_texts     = list(zip(*X_texts))

        vocab_set = set()
        self.MAX_SENT_LEN = 0
        [setattr(self, 'MAX_SENT_LEN', max(self.MAX_SENT_LEN, len([vocab_set.add(w) for w in sent.lower().split()]))) for sent in np.asarray(X_texts).flatten()]
        vocab_dict['vocab'] = ['<eos>', 'UNK'] + list(vocab_set)

        word_to_idx = {}
        idx_to_word = {}
        for id, word in enumerate(vocab_dict["vocab"]):
            word_to_idx[word] = id
            idx_to_word[id]   = word

        vocab_dict["word_to_idx"] = word_to_idx
        vocab_dict["idx_to_word"] = idx_to_word

        X_onehot, vocab_dict["MAX_SENT_LEN"] = convert_and_pad(X_texts, concat_sents = self.config['concat_sents'])

        X_train_text = [X_texts[i] for i in train_id]
        X_valid_text = [X_texts[i] for i in valid_id]
        X_test_text = [X_texts[i] for i in test_id]

        X_train_onehot = [X_onehot[i] for i in train_id]
        Y_train = [Y[i] for i in train_id]
        X_valid_onehot = [X_onehot[i] for i in valid_id]
        Y_valid = [Y[i] for i in valid_id]
        X_test_onehot = [X_onehot[i] for i in test_id]
        Y_test = [Y[i] for i in test_id]

        # for id, x, y in zip(train_id, X_train_text, Y_train):
        #     print('id = %d' % id)
        #     print('X: %s' % '\n'.join(x))
        #     print('Y: %s (%d)' % (self.config['label_encoder'].classes_[y], y))
        #     print('*' * 50)

        train_batch_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train_onehot, Y_train)), batch_size=batch_size,
                                                         shuffle=True, drop_last=False)
        valid_batch_loader = torch.utils.data.DataLoader(dataset=list(zip(X_valid_onehot, Y_valid)), batch_size=batch_size,
                                                         shuffle=False, drop_last=False)
        test_batch_loader = torch.utils.data.DataLoader(dataset=list(zip(X_test_onehot, Y_test)), batch_size=batch_size,
                                                        shuffle=False, drop_last=False)

        if torch.cuda.is_available():
            train_batch_loader.pin_memory = True
            valid_batch_loader.pin_memory = True
            test_batch_loader.pin_memory = True

        data_dict['train_batch_loader'] = train_batch_loader
        data_dict['valid_batch_loader'] = valid_batch_loader
        data_dict['test_batch_loader']  = test_batch_loader
        # data_dict['X_train_text']  = X_train_text
        # data_dict['X_valid_text']  = X_valid_text
        # data_dict['X_test_text']  = X_test_text
        # data_dict['X_train_onehot']  = X_train_onehot
        data_dict['Y_train']  = Y_train
        # data_dict['X_valid_onehot']  = X_valid_onehot
        data_dict['Y_valid']  = Y_valid
        # data_dict['X_test_onehot']  = X_test_onehot
        data_dict['Y_test']  = Y_test

        return vocab_dict, data_dict

    def run_cross_validation(self, X, Y):
        train_ids, valid_ids, test_ids = self.load_cv_index_train8_valid1_test1(Y)
        cv_results = []

        for r_id, (train_id, valid_id, test_id) in enumerate(zip(train_ids, valid_ids, test_ids)):
            # if r_id >= 10:
            #     break

            self.logger.info('*' * 20 + ' %s - Round %d ' % (self.config['data_name'], r_id))
            self.config['test_round'] = r_id

            X_raw_feature = self.config['X_raw_feature']
            Y = self.config['Y']

            vocab_dict, data_dict = self.get_batch_loader(X_raw_feature, Y, train_id, valid_id, test_id, batch_size=self.config['batch_size'])

            print('size of vocab=%d', len(vocab_dict['vocab']))
            print('max length=%d', vocab_dict['MAX_SENT_LEN'])

            model, params = self.init_model(vocab_dict, self.config['deep_model_name'])
            tally_parameters(model)

            cv_results.extend(self.run_experiment(model, params, data_dict, exp_name='[%s]%s-fold_%d' % (self.config['data_name'], self.config['deep_model'], r_id)))

        self.export_cv_results(cv_results, test_ids, Y)

        return cv_results

    def run_experiment(self, model, model_param, data_dict, exp_name = ''):
        optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_param['LEARNING_RATE'])
        # optimizer = Adagrad(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_param['LEARNING_RATE'])
        criterion = torch.nn.CrossEntropyLoss()

        train_batch_loader = data_dict['train_batch_loader']
        valid_batch_loader = data_dict['valid_batch_loader']
        test_batch_loader  = data_dict['test_batch_loader']
        Y_train            = data_dict['Y_train']
        Y_valid            = data_dict['Y_valid']
        Y_test             = data_dict['Y_test']

        best_loss = 0 #sys.float_info.max
        stop_increasing = 0
        all_training_losses = []
        all_valid_losses = []
        valid_accuracy = []
        valid_f1_score = []

        for epoch in range(self.config['max_epoch']):
            print('*' * 25 + 'Epoch=%d' % epoch + '*' * 25)
            '''
            Training
            '''
            model.train()
            training_losses = []
            train_pred   = []
            train_y_shuffled   = []
            for i, (x,y) in enumerate(train_batch_loader):
                x = Variable(x)
                y = Variable(y)

                optimizer.zero_grad()

                output = model.forward(x)
                loss = criterion.forward(output, y)
                # prob_i1, pred_i1 = output.data.topk(1)
                prob_i, pred_i = torch.max(output.data, 1)

                # pred_i1 = pred_i1.numpy().flatten().tolist()
                # pred_i = pred_i.numpy().flatten().tolist()

                if torch.cuda.is_available():
                    train_pred.extend(pred_i.cpu().numpy().flatten().tolist())
                else:
                    train_pred.extend(pred_i.numpy().flatten().tolist())

                loss.backward()

                if 'clip_grad_norm' in model_param:
                    grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), model_param['clip_grad_norm'])
                else:
                    grad_norm = 0.0

                optimizer.step()
                training_losses.append(loss.data[0])
                train_y_shuffled.extend(y.data.numpy().tolist())

                # constrain l2-norms of the weight vectors
                if 'NORM_LIMIT' in model_param:
                    weight_norm = sum([float(p.weight.norm().data.numpy()) for p in model.parameters_to_normalize])
                    if weight_norm > model_param["NORM_LIMIT"]:
                        for p in model.parameters_to_normalize:
                            p.weight.data = p.weight.data * model_param["NORM_LIMIT"] / weight_norm

                print('Training %d/%d, loss=%.5f, weight_norm=%.5f, grad_norm=%s' % (i, len(train_batch_loader), np.average(loss.data[0]), weight_norm, str(grad_norm) if 'NORM_LIMIT' in model_param else "N/A"))

            all_training_losses.append(training_losses)
            training_loss_mean = np.average(training_losses)

            print('-' * 20 + 'Training Summary' + '-' * 20)
            print('Training loss=%.5f' % training_loss_mean)

            print("Training classification report:")
            report = metrics.classification_report(train_y_shuffled, train_pred,
                                                   target_names=np.asarray(self.config['label_encoder'].classes_))
            print(report)

            print("Training confusion matrix:")
            confusion_mat = str(metrics.confusion_matrix(train_y_shuffled, train_pred))
            print('\n' + confusion_mat)

            acc_score = metrics.accuracy_score(train_y_shuffled, train_pred)
            f1_score = metrics.f1_score(train_y_shuffled, train_pred, average='macro')
            # train_accuracy.append([acc_score])
            # train_f1_score.append([f1_score])

            print("Training accuracy:   %0.3f" % acc_score)
            print("Training f1_score:   %0.3f" % f1_score)
            print('*' * 100)
            print('*' * 100)

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

                print('Validating %d/%d, loss=%.5f' % (i, len(valid_batch_loader), np.average(loss.data[0])))

            valid_loss_mean = np.average(valid_losses)
            all_valid_losses.append(valid_losses)

            print('-' * 20 + 'Validation Summary' + '-' * 20)
            print('Valid loss=%.5f' % valid_loss_mean)

            print("Validation Classification Report:")
            report = metrics.classification_report(Y_valid, valid_pred,
                                                   target_names=np.asarray(self.config['label_encoder'].classes_))
            print(report)

            print("Validation Confusion Matrix:")
            confusion_mat = str(metrics.confusion_matrix(Y_valid, valid_pred))
            print('\n' + confusion_mat)

            acc_score = metrics.accuracy_score(Y_valid, valid_pred)
            f1_score = metrics.f1_score(Y_valid, valid_pred, average='macro')
            valid_accuracy.append([acc_score])
            valid_f1_score.append([f1_score])

            print("Validation accuracy:   %0.3f" % acc_score)
            print("Validation f1_score:   %0.3f" % f1_score)
            print('*' * 100)
            print('*' * 100)

            is_best_loss = f1_score > best_loss
            rate_of_change = float(f1_score - best_loss)/float(best_loss) if best_loss > 0 else 0

            if is_best_loss:
                print('Update best f1 (%.4f --> %.4f), rate of change (ROC)=%.2f' % (best_loss, f1_score, rate_of_change * 100))
            else:
                print('Best f1 is not updated (%.4f --> %.4f), rate of change (ROC)=%.2f' % (best_loss, f1_score, rate_of_change * 100))

            best_loss = max(f1_score, best_loss)

            print('*' * 50)

            if rate_of_change < 0.01 and epoch > 0:
                stop_increasing += 1
            else:
                stop_increasing = 0

            if stop_increasing >= self.config['early_stop_tolerance']:
                print('Have not increased for %d epoches, stop training' % stop_increasing)
                break

            plot_learning_curve(all_training_losses, all_valid_losses, 'Error Trend: Training and Validation', curve1_name='Training Error', curve2_name='Validation Error', save_path=self.config['experiment_path']+'/%s-train_valid_curve.png' % exp_name)
            plot_learning_curve(valid_accuracy, valid_f1_score, 'Accuracy and F1-score on Validation', curve1_name='Accuracy', curve2_name='F1-score', save_path=self.config['experiment_path']+'/%s-train_f1_curve.png' % exp_name)

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

            print('Testing %d/%d, loss=%.5f' % (i, len(test_batch_loader), loss.data[0]))

        print('-' * 20 + 'Test Summary' + '-' * 20)
        test_loss_mean = np.average(test_losses)
        print('*' * 50)
        print('Testing loss=%.5f' % test_loss_mean)
        print("Testing Classification Report:")
        report = metrics.classification_report(Y_test, test_pred,
                                               target_names=np.asarray(self.config['label_encoder'].classes_))
        print(report)

        print("Testing Confusion Matrix:")
        confusion_mat = str(metrics.confusion_matrix(Y_test, test_pred))
        print('\n' + confusion_mat)

        acc_score = metrics.accuracy_score(Y_test, test_pred)
        f1_score = metrics.f1_score(Y_test, test_pred, average='macro')

        print("Testing accuracy:   %0.3f" % acc_score)
        print("Testing f1_score:   %0.3f" % f1_score)

        print('*' * 100)
        print('*' * 100)
        print('*' * 100)

        result = self.classification_report(Y_test, test_pred, self.config['deep_model_name'], 'test')
        results = [[result]]
        return results

def plot_learning_curve(train_scores, test_scores, title, curve1_name='curve1_name', curve2_name='curve2_name', ylim=None, save_path=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    """
    train_sizes=np.linspace(1, len(train_scores), len(train_scores))
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
