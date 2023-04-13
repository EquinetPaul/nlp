class Ngram():
    def __init__(self, n=1):
        self.n = n
        self.chain_frequency = {}
        self.chain = {}
        
    def train(self, text, separator = " "):
        words = text.split(separator)
        for i in range(len(words)-self.n):
            get = self.chain_frequency.get(tuple([words[i+y] for y in range(self.n)]),0)
            if get == 0:
                self.chain_frequency[tuple([words[i+y] for y in range(self.n)])] = {}
                self.chain_frequency[tuple([words[i+y] for y in range(self.n)])][words[i+self.n]] = 1
            elif get.get(words[i+self.n],0) == 0:
                self.chain_frequency[tuple([words[i+y] for y in range(self.n)])][words[i+self.n]] = 1
            else:
                self.chain_frequency[tuple([words[i+y] for y in range(self.n)])][words[i+self.n]] += 1
                
    def normalize(self):
        self.chain = self.chain_frequency.copy()
        for key, value in self.chain_frequency.items():
            sum_frequency = sum(value.values())
            for k, v in value.items():
                self.chain[key][k] = v / sum_frequency
