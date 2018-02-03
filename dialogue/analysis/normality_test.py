from scipy import stats
import os

root_folder_path = '../../output/feature_comparison/' # LR exps
# root_folder_path = '../../output/cnn_results/non-concat/' # CNN exps

total_count = 0
valid_count = 0

for sub_folder_name in os.listdir(root_folder_path):
    for result_folder_name in os.listdir(root_folder_path + sub_folder_name):
        for dataname in ['dstc2', 'dstc3', 'family', 'ghome']:
            result_path = os.path.join(root_folder_path, sub_folder_name, result_folder_name, '%s.test.csv' % dataname)
            # print(os.path.abspath(result_path))
            if result_path.find('.DS_Store') > 0:
                continue

            if not os.path.exists(result_path):
                print(result_path)
                continue

            with open(result_path, 'r') as result_csv:
                results = result_csv.readlines()[1:]
                results = [r.split(',') for r in results]
                f_score = [float(r[-1]) for r in results]

                # D'Agostino's K-squared test, not too reliable
                # k2, p = stats.normaltest(f_score)
                # alpha = 1e-3
                k2, p = stats.shapiro(f_score)
                alpha = 0.05

                total_count += 1

                if p < alpha:  # null hypothesis: x comes from a normal distribution
                    print('%s : %s' % (result_folder_name, dataname))
                    print("\tp = {:g}".format(p))
                    print("\t!!!The null hypothesis can be rejected!!!")
                else:
                    pass
                    valid_count += 1
                    # print("\tThe null hypothesis cannot be rejected")

print('valid/all = %d / %d' % (valid_count, total_count))