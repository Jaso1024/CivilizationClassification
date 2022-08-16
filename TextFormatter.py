from nltk.stem import WordNetLemmatizer
import nltk
import re
import contractions

class Formatter:
    def __init__(self):
        self.training = {}
        self.empires = []
        self.testing = {}

        self.set_empires()
        self.get_training()

    def clean_whitespace(self, line):
        line = line.strip()
        line = line.replace("\n", "")
        return line
    
    def fix_era(self, line):
        line = line.replace("C.", "C")
        line = line.replace("E.", "E")
        line = line.replace("D.", "D")
        return line

    def get_training(self):
        with open("Resources/Data/Training.txt", "r", encoding="utf8") as file:
            current_empire = None
            for line in file.readlines()[1:]:
                line = self.clean_whitespace(line)
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
                        self.training[current_empire].extend([sentence])

    def format_sentence(self, sentence, current_empire):
        if len(sentence.replace(" ","")) > 15:
            sentence = contractions.fix(sentence)
            contains_other_empire = False
            for empire in self.empires:
                if empire == current_empire:
                    continue
                elif re.search(empire.lower(), sentence.lower()) is not None:
                    contains_other_empire = True

            if not contains_other_empire and self.fix_semicolons(sentence, current_empire):
                return sentence.strip()
    
    def fix_semicolons(self, sentence, current_empire):
        if sentence[0] == ";":
            self.training[current_empire][-1] += sentence.strip()
            return False
        return True
        
    def set_empires(self):
        with open("Resources/Data/Training.txt", "r", encoding="utf8") as file:
            empires = file.readline().split(" ")
            for empire in empires:
                self.training[empire] = []
                self.testing[empire] = []
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

    def preprocess_training(self):
        data = {}
        for empire in self.empires:
            text = []
            for sentence in self.training[empire]:
                wnl = WordNetLemmatizer()
                print(sentence)
                sentence = nltk.word_tokenize(sentence)
                sentence = nltk.pos_tag(sentence)
                sentence = [wnl.lemmatize(word, self.get_wordnet_pos(pos)) for word, pos in sentence]
                text.append(sentence)
            data[empire] = text
        return data
                


if __name__ == "__main__":
    formatter = Formatter()
    formatter.preprocess_training()
                

