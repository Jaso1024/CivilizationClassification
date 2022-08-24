from nltk.stem import WordNetLemmatizer
import nltk
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from nltk.corpus import stopwords

class Formatter:
    def __init__(self,min_sentence_length=5, level_sample_sizes=True):
        self.data = {}
        self.empires = []
        self.vectorizer = CountVectorizer()
        self.min_sent_len = min_sentence_length
        self.level_sample_sizes = level_sample_sizes

        self.set_empires()
        with open("Resources/Data/EmpireText.txt", "r", encoding="utf8") as file: 
            self.make_dataset(file)
        self.fit_vectorizer()

    def clean_whitespace(self, line):
        line = line.strip()
        line = line.replace("\n", "")
        return line
    
    def fix_era(self, line):
        line = line.replace("C.", "C")
        line = line.replace("E.", "E")
        line = line.replace("D.", "D")
        return line

    def remove_citings(self, line):
        return re.sub("\[.{0,4}]", "", line)
    

    def make_dataset(self, file):
        current_empire = None
        for line in file.readlines()[1:]:
            line = self.clean_whitespace(line)
            line = self.remove_citings(line)
            line = self.fix_era(line)

            if line in self.empires:
                current_empire = line
                continue
            elif len(line) < 7 or "source" in "".join(line).lower():
                continue
            elif line[-1] != ".":
                line += "."

            line = nltk.sent_tokenize(line)

            for sentence in line:
                sentence = self.format_sentence(sentence, current_empire)
                if sentence is not None:
                    self.data[current_empire].extend([sentence])

    def format_sentence(self, sentence, current_empire):
        if len(sentence.split(" ")) > self.min_sent_len:
            sentence = contractions.fix(sentence)
            contains_other_empire = False
            for empire in self.empires:
                if empire == current_empire:
                    continue
                elif re.search(empire.lower(), sentence.lower()) is not None:
                    contains_other_empire = True

            if not contains_other_empire:
                return sentence.strip()
    
    def fix_semicolons(self, sentence, current_empire):
        if sentence[0] == ";":
            self.training[current_empire][-1] += sentence.strip()
            return False
        return True
        
    def set_empires(self):
        with open("Resources/Data/EmpireText.txt", "r", encoding="utf8") as file:
            empires = file.readline().replace("\n", "").split(" ")

            for empire in empires:
                self.data[empire] = []
                self.empires.append(empire)
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN
    
    def limit_length(self):
        import random
        max_len = float('inf')
        for empire in self.empires:
            max_len = len(self.data[empire]) if len(self.data[empire]) < max_len else max_len
        
        for empire in self.empires:
            new_data = []
            for num in range(max_len):
                index = np.random.randint(0,len(self.data[empire]))
                new_data.append(self.data[empire][index])
            self.data[empire] = new_data


    def preprocess_data(self):
        #stop_words = set(stopwords.words('english'))
        text = []
        labels = []
        #if self.level_sample_sizes:
            #self.limit_length()
        for empire in self.empires:
            for sentence in self.data[empire]:
                wnl = WordNetLemmatizer()
                sentence = nltk.word_tokenize(sentence)
                #sentence = [w for w in sentence if not w.lower() in stop_words]
                sentence = nltk.pos_tag(sentence)
                sentence = [wnl.lemmatize(word, self.get_wordnet_pos(pos)) for word, pos in sentence]
                text.append(sentence)
                label = np.zeros(len(self.empires))
                label[self.empires.index(empire)] = 1
                labels.append(label)

        return text, labels

    def fit_vectorizer(self):
        text = list()
        for empire in self.empires:
            text.extend(self.data[empire])
        self.vectorizer.fit(text)

    def get_data(self):
        return self.preprocess_data()
    
    def get_unique_words(self, data):
        unique_words = set()
        print("len", len(data))
        for sentence in data:
                for word in sentence:
                    unique_words.add(word)
        return len(unique_words)

    def vectorize(self, data):
        dataset = []
        for sentence in data:
            dataset.append(" ".join(sentence))
        data = self.vectorizer.transform(dataset)
        return data
                
    def linear_encode(self, labels):
        decoded_labels = []
        for label in labels:
            decoded_labels.append(list(np.array(label).flatten()).index(1))
        return decoded_labels
    
    def tensorflow_format_text(self, data, num_uniques):
        max_len = 0
        for sentence in data:
            max_len = len(sentence) if len(sentence) > max_len else max_len
        encoded_sentences = [one_hot(" ".join(sentence), num_uniques) for sentence in data]
        padded_sequences = pad_sequences(encoded_sentences, maxlen=max_len, padding='post')        
        return padded_sequences, max_len

    

if __name__ == "__main__":
    formatter = Formatter()
    print(formatter.remove_citings("[e] Independence had been delayed before it had an emperor.[8][9][10][11] T"))
                

