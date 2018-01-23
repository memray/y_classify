# -*- coding: utf-8 -*-

import click
import os
import pandas as pd
import scipy.stats


def file_split(file):
    s = file.split('.')
    name = '.'.join(s[:-1])  # get directory name
    return name


def getsheets(inputfile):
    name = file_split(inputfile)
    try:
        os.makedirs(name)
    except:
        pass

    df1 = pd.ExcelFile(inputfile)
    for x in df1.sheet_names:
        print(x + '.xlsx', 'Done!')
        df2 = pd.read_excel(inputfile, sheetname=x)
        filename = os.path.join(name, x + '.xlsx')
        df2.to_excel(filename, index=False)
    print('\nAll Done!')


def get_sheet_names(inputfile):
    df = pd.ExcelFile(inputfile)
    for i, flavor in enumerate(df.sheet_names):
        print('{0:>3}: {1}'.format(i + 1, flavor))


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-n', '--sheet-names', is_flag=True)
@click.argument('inputfile')
def cli(sheet_names, inputfile):
    '''Convert a Excel file with multiple sheets to several file with one sheet.
    Examples:
    \b
        getsheets filename
    \b
        getsheets -n filename
    '''
    if sheet_names:
        get_sheet_names(inputfile)
    else:
        getsheets(inputfile)

if __name__ == '__main__':
    xlsx_folder_path        = '../../dataset/result/feature_comparison/'
    base_xlsx_name        = 'context=next.similarity=false.xlsx'
    base_xlsx_path   = os.path.join(xlsx_folder_path, base_xlsx_name)
    base_xlsx        = pd.ExcelFile(base_xlsx_path)

    for xlsx_name in os.listdir(xlsx_folder_path):
        if xlsx_name.strip() == base_xlsx_name.strip():
            continue

        if not xlsx_name.endswith('.xlsx'):
            continue

        # print(os.path.join(xlsx_folder_path, xlsx_name))
        print('%s vs %s' % (base_xlsx_name, xlsx_name))

        xlsx_path   = os.path.join(xlsx_folder_path, xlsx_name)
        result_xlsx = pd.ExcelFile(xlsx_path)

        for sheet_id, sheet_name in enumerate(base_xlsx.sheet_names):
            if sheet_name == 'Summary':
                continue

            sheet1  = base_xlsx.parse(sheetname=sheet_id)
            sheet2  = result_xlsx.parse(sheetname=sheet_id)

            # 8th column is f1_score
            fscore1 = sheet1.iloc[:,8].as_matrix()
            fscore2 = sheet2.iloc[:,8].as_matrix()

            # t, p_value = scipy.stats.wilcoxon(fscore1, fscore2)
            t, p_value = scipy.stats.ttest_ind(fscore1, fscore2)

            print('\tsheet %s, \t t=%f, \t p_value=%f' % (sheet_name, t, p_value))
            pass
