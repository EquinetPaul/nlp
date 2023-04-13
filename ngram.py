class Ngram():
    def __init__(self, n=1):
        self.n = n
        self.chain_frequency = {}
        self.chain = {}
        
    def train(self, text, separator = " "):
        words = text.split(separator)
        for i in range(len(words)-self.n):
            if self.n>1:
                sequence = tuple([words[i+y] for y in range(self.n)])
            else:
                sequence = words[i]
            get = self.chain_frequency.get(sequence,0)
            if get == 0:
                self.chain_frequency[sequence] = {}
                self.chain_frequency[sequence][words[i+self.n]] = 1
            elif get.get(words[i+self.n],0) == 0:
                self.chain_frequency[sequence][words[i+self.n]] = 1
            else:
                self.chain_frequency[sequence][words[i+self.n]] += 1
                
    def normalize(self):
        self.chain = copy.deepcopy(self.chain_frequency)
        for key, value in self.chain.items():
            sum_frequency = sum(value.values())
            for k, v in value.items():
                self.chain[key][k] = v / sum_frequency    
                
    def generate_sentence(self):
        sentence = "Mesdames et Messieurs"
        while sentence[-1] not in [".", "!", "?"]:
            if self.n>1:
                current_sequence = tuple(sentence.split()[-self.n:])
            else:
                current_sequence = sentence.split()[-1]
            sentence += " " + random.choices(list(self.chain[current_sequence].keys()), weights=list(self.chain[current_sequence].values()))[0]
        sentence = " ".join(sentence.split())
        return sentence
    
    def save(self, filepath):
        with open(filepath, 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)
            
    def load(self, filepath):
        with open(filepath, 'rb') as load_file:
            obj = pickle.load(load_file)
            self.n = obj["n"]
            self.chain_frequency = obj["chain_frequency"]
            self.chain = obj["chain"]

            
# usage
# chain = Ngram(n=2)
# # for sentence in tqdm(df["speech"][df.title.str.startswith("Déclaration")]):
# for sentence in tqdm(df["speech"]):
#     try:
#         sentence = "[start] " + sentence.replace(".", " [end] . [start]").replace("?", " [end] ? [start]").replace("!", " [end] ! [start]").replace(",", ", ")
#         cleaned = " ".join(sentence.replace("\n", "").strip().split())
#         chain.train(cleaned)
#     except Exception as e:
#         pass
# chain.normalize()
