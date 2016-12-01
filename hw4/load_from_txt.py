import string
from nltk.corpus import stopwords

commas = ['.', ',', '?', '!', '(', ')', '<', '>', '\'', '\"', '@', '#','$', '%', '&', '-', '+', '*', '/', '\n']

stop_words = stopwords.words('english')

def load_data(path):
    id2word = []
    word2id = {}
    lines = []
    with open(path) as f:
        docs = []
        for l in f:
            doc = []
            l = l.lower()
            for comma in string.punctuation:
                l = l.replace(comma, ' ')
            l.replace('\n', '')
            # lines.append(l)
            line = ''
            for w in l.split():
                if w in stop_words:
                    continue

                line += w + ' '

                if w not in word2id:
                    word2id[w] = len(word2id)
                    temp_object = {}
                    temp_object['num'] = 1
                    temp_object['word'] = w
                    id2word.append(temp_object)
                else:
                    id2word[word2id[w]]['num'] += 1
                doc.append(word2id[w])
            docs.append(doc)

            lines.append(line)

    return word2id, id2word, docs, lines

def load_doc(path):
    docs = []
    with open(path) as f:
        for l in f:
            l = l.lower()
            for comma in string.punctuation:
                l = l.replace(comma, ' ')
            l.replace('\n', '')
            # lines.append(l)
            line = ''
            for w in l.split():
                if w in stop_words:
                    continue

                line += w + ' '
            docs.append(line)

    return docs


def load_check(path):
    checks = []
    with open(path) as f:
        docs = []
        for index, l in enumerate(f):
            if index == 0:
                continue
            array = l.replace('\n', '').split(',')
            array = array[1:]
            array[0] = int(array[0])
            array[1] = int(array[1])
            checks.append(array)

    return checks

def calc_output(clusters, checks, path):
    f = open(path, 'w')
    f.write('ID,Ans')
    for index, check in enumerate(checks):
        same = '0'
        if clusters[check[0]] == clusters[check[1]]:
            same = '1'
        f.write('\n' + str(index) + ',' + same)

