"""
Input is multiple text files.  Each text file represents one document.
Output is stdout, a stream of 2-column TSV
  DocID  \t  JsonAnnotations
where the DocID is based on the filename.

USAGE
proc_text_files.py MODE  [files...]

e.g.
python proc_text_files_to_stdout.py pos *.txt > allpos.anno
"""

import sys, re, os, json
mode = 'nerparse'
folder_path = '/Users/rmeng/Project/presto_data_process/dialogue/stanford_corenlp/examples'

from stanford_corenlp_pywrapper import CoreNLP
ss = CoreNLP(mode)  # need to override corenlp_jars

for filename in os.listdir(folder_path):
    print(os.path.join(folder_path, filename))
    docid = os.path.basename(filename)
    docid = re.sub(r'\.txt$', "", docid)

    text = open(os.path.join(folder_path, filename)).read()
    jdoc = ss.parse_doc(text)
    print("%s\t%s" % (docid, jdoc))


jdoc = ss.parse_doc("")
print("%s\t%s" % (docid, jdoc))