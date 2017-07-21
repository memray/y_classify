from flask import request
import json
from nltk.tokenize import word_tokenize 
from nltk.tree import Tree
import re
import requests
from six.moves.urllib.parse import urlencode
import sys

class ModelPart:
    def __init__(self, port = 4080, host = 'localhost', proxy = '', editorData = None, externalModel = None, moreArgs = {}): 
        self.port = port
        self.host = host
        self.socks_proxy = proxy
        self.editorData = editorData
        self.externalModel = externalModel
        self.moreArgs = moreArgs
        self.editorDict = {}
        self.externalDict = {}
        if self.editorData:
            self.setupEditorData()

        if self.externalModel:
            self.setupExternalModel()

    def getSignature(self, sentence):
        pattern = re.compile('[\s\$\'`,\?!\.\-]')
        return str.lower(re.sub(pattern, '', sentence)) 

    def setupEditorData(self):
        raise NotImplementedError()

    def setupExternalModel(self):
        raise NotImplementedError()

    def createRequest(self):
        raise NotImplementedError()

    def getResults(self): 
        raise NotImplementedError()


class Chunker(ModelPart):
    def setupEditorData(self):
        with open(self.editorData, 'r') as f:
            for line in f:
                tks = line.split('\t')
                sentence = tks[1]
                editor_chunks = tks[2]
                these_chunks = editor_chunks.split('|||') 

                sig = self.getSignature(sentence)
                self.editorDict[sig] = these_chunks

    def getResults(self, input_sentence):
        if (self.editorData is None):
            chunker_request = self.createRequest(input_sentence)
            chunker_response = requests.get(chunker_request, proxies = self.socks_proxy)
            chunks = chunker_response.json()['chunks']
        else:
            chunks = []
            sig = self.getSignature(input_sentence)
            curr = self.editorDict[sig] 
            for chunk in curr:
                words = word_tokenize(chunk)
                chunks.append(u' '.join(words))
        return chunks

    def createRequest(self, sentence):
         chunker_args = dict(self.moreArgs)
         if 'stanford' not in chunker_args:
             chunker_args['stanford'] = True
         url = 'http://{}:{}/chunk?{}'.format(self.host, self.port, urlencode(dict(chunker_args, sentence=sentence), doseq=True))
         return url


class DialogAct(ModelPart):
    def setupEditorData(self): 
        with open(self.editorData, 'r') as f:
            for line in f:
                (chunk, da) = line.strip().split('\t')
                sig = self.getSignature(chunk)
                self.editorDict[sig] = da

    def setupExternalModel(self):
        with open(self.externalModel, 'r') as f:
            for line in f:
                (chunk, da) = line.strip().split('\t')
                sig = self.getSignature(chunk)
                self.externalDict[sig] = da

    def getResults(self, chunks):
        if self.editorData is not None:
            dialog_results = self.readEditorData(chunks)
        elif self.externalModel is not None:
            dialog_results = self.readExternalModel(chunks)
        else:
            dialog_results = self.callNetworkModel(chunks)
        return dialog_results

    def readEditorData(self, chunks):
        dialogResult = []
        for chunk in chunks:
            sig = self.getSignature(chunk)
            da = self.editorDict[sig]
            dialogResult.append({'dialogact' : da, 'text' : chunk})
        return dialogResult 

    def readExternalModel(self, chunks):
        dialogResult = []
        for chunk in chunks:
            sig = self.getSignature(chunk)
            da = self.externalDict[sig]
            dialogResult.append({'dialogact' : da, 'text' : chunk})
        return dialogResult 

    def callNetworkModel(self, chunks):
        dialog_request = self.createRequest(chunks)
        dialog_response = requests.get(dialog_request, proxies = self.socks_proxy)
        return dialog_response.json()

    def createRequest(self, chunks):
        url = 'http://{}:{}/dialog?{}'.format(self.host, self.port, urlencode(dict(self.moreArgs, sentence=chunks), doseq=True))
        return url


class Parser(ModelPart):
    def setupEditorData(self):
        with open(self.editorData, 'r' ) as f:
            for line in f:
                json_obj = json.loads(line)
                sentence = json_obj["2"]
                json_tree = json_obj['1']
                ed_sig = self.getSignature(sentence)
                self.editorDict[ed_sig] = json_tree 

    def getResults(self, chunks):
        parser_results = []  
        if (self.editorData is None):

            parser_request = self.createRequest(chunks)
            parser_response = requests.get(parser_request, proxies = self.socks_proxy)
            
            # The tagger expects a list of {'sentence': sentence, 'tree':tree} dicts
            # extract from the parser response
            for item in parser_response.json():
               parser_results.append({'sentence':item['sentence'], 'tree':item['penn']})
               tree = Tree.fromstring(item['penn'])
        else:
            for chunk in chunks:
                penn_str = self.getPennTree(chunk) 
                parser_results.append({'sentence':chunk, 'tree':penn_str}) 
                tree = Tree.fromstring(penn_str)
        return parser_results

    def createRequest(self, chunks):
        url = 'http://{}:{}/parser/service?{}'.format(self.host, self.port, urlencode(dict(self.moreArgs, sentence=chunks), doseq=True))
        return url

    def getPennTree(self, chunk):
        sig = self.getSignature(chunk) 
        if sig not in self.editorDict:
            sys.exit("COULD NOT FIND "+sig+" in editor data!")

        obj = self.editorDict[sig][0]
        pennList = []
        self.json2penn(obj, pennList)
        penn_str  = ''.join(pennList)
        return penn_str

    def json2penn(self, obj, ans):
        ans.append("(X ")
        if len(obj['children']) == 0:
            ans.append("(X "+str.lower(obj['span'])+")")
        for child in obj['children']:
            self.json2penn(child,ans)
        ans.append(")")


class Tagger(ModelPart):
    def setupEditorData(self):
        pass

    def getResults(self, sentence):
        tagger_request = self.createRequest(sentence)
        tagger_response = requests.get(tagger_request, proxies = self.socks_proxy)
        # tagger_response = requests.post(tagger_request, proxies = self.socks_proxy, json = sentence)
        tagger_results = tagger_response.json()
        return tagger_results

    def createRequest(self, sentence):
        url = 'http://{}:{}/tagger?{}'.format(self.host, self.port, urlencode(dict(self.moreArgs, sentence=sentence), doseq=True))
        return url

if __name__ == '__main__':
    # socksProxy = {'http': 'socks5://socks.yahoo.com:1080'}
    socksProxy = None
    host = 'perf-serving03.broadway.gq1.yahoo.com'
    chunkerPort = 80
    dialogPort = 80
    parserPort = 80
    taggerPort = 80

    moreArgs = {'debug' : 'true', 'binary' : 'false'}

    # Initialize model parts
    chunker = Chunker(chunkerPort, host, socksProxy, None, None, moreArgs)
    da = DialogAct(dialogPort, host, socksProxy, None, None, moreArgs)
    parser = Parser(parserPort, host, socksProxy, None, None, moreArgs)
    tagger = Tagger(taggerPort, host, socksProxy, None, None, moreArgs)

    inputSentence = 'im looking for a restaurant that serves'
    # chunk
    chunks = chunker.getResults(inputSentence)
    print(chunks)

    # DA
    daResult = da.getResults(inputSentence)
    print(daResult)

    # parser
    parseTree = parser.getResults(inputSentence)
    print(parseTree)

    #tagger
    tags = tagger.getResults(inputSentence)
    print(tags)
