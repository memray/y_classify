from classify.feature_extractor import extract_noun_phrases
from stanford_corenlp.pycorenlp.corenlp import StanfordCoreNLP
import numpy as np

if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = (
        'Pusheen and Smitha walked along the beach. Pusheen Kim wanted to surf,'
        'but fell off the surfboard.')
    parse_record = nlp.annotate(text, properties={
        'annotators': 'tokenize, ssplit, pos, lemma, ner, entitymentions, parse',
        'outputFormat': 'json'
    })
    print(parse_record['sentences'][0]['parse'])
    # parse_record = nlp.tokensregex(text, pattern='/Pusheen|Smitha/', filter=False)
    # print(parse_record)
    # parse_record = nlp.semgrex(text, pattern='{tag: VBD}', filter=False)
    # print(parse_record)

    tokens = np.concatenate([[t['originalText'].lower() for t in s['tokens']] for s in parse_record['sentences']])
    pos = np.concatenate([[t['pos'] for t in s['tokens']] for s in parse_record['sentences']])
    # pos = np.concatenate([s['pos'] for s in parse_record['sentences']])
    noun_phrases = extract_noun_phrases(tokens, pos)

    print(noun_phrases)

    entity_list = []
    entity_dict = {}
    for ner in np.concatenate([r['entitymentions'] for r in parse_record['sentences']]):
        entity_dict['entity__%s__%s_%s' % ('current_user_utterance', ner['text'].lower(), ner['ner'])] = True

    print(entity_dict)