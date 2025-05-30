from collections import defaultdict


# 统计每个单词出现的频率 以字典形式返回
def build_stats(sentences):
    word_freq = defaultdict(int)
    for sentence in sentences:
        symbols = sentence.split()
        for symbol in symbols:
            word_freq[symbol] += 1
    return word_freq


# 将单词差分成更小的字符yellow -> y e l l o w
# 生成字母表
def generate_vocab(word_freq):
    alphabet = []
    for word in word_freq.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    return alphabet


# 计算成对出现最多的字符对
def compute_pair_freqs(splits, word_freq):
    pair_freqs = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(char_a, char_b, splits, word_freq):
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == char_a and split[i + 1] == char_b:
                split = split[:i] + [char_a + char_b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


if __name__ == "__main__":
    # Yellow 歌词
    sentences = ["Look at the stars",
                 "Look how they shine for you",
                 "And everything you do",
                 "Yeah, they were all Yellow",
                 "I came along",
                 "I wrote a song for you",
                 "And all the things you do",
                 "And it was called Yellow",
                 "So then I took my turn",
                 "Oh what a thing to have done",
                 "And it was all Yellow",
                 "Your skin"
                 "Oh yeah, your skin and bones",
                 "Turn into something beautiful",
                 "Do you know"
                 "You know I love you so",
                 "You know I love you so",
                 "I swam across",
                 "I jumped across for you",
                 "Oh what a thing to do",
                 "Cos you were all Yellow",
                 "I drew a line",
                 "I drew a line for you",
                 "Oh what a thing to do",
                 "And it was all Yellow",
                 "Your skin",
                 "Oh yeah your skin and bones",
                 "Turn into something beautiful",
                 "And you know",
                 "For you I'd bleed myself dry",
                 "For you I'd bleed myself dry",
                 "It's true",
                 "Look how they shine for you",
                 "Look how they shine for you",
                 "Look how they shine for",
                 "Look how they shine for you",
                 "Look how they shine for you",
                 "Look how they shine",
                 " Look at the stars",
                 "Look how they shine for you",
                 "And all the things that you do"]
    word_freq = build_stats(sentences)  # 构建单词频率
    print("word_freq:", word_freq)

    # 初始词表
    vocab = generate_vocab(word_freq)
    print("vocab:", vocab)
    # 每个词当作key 该词的字符组成一个list当作value
    splits = {word: [c for c in word] for word in word_freq.keys()}
    print("splits:", splits)

    merges = {}
    vocab_size = 100 # 词表大小
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freq)  # 计算成对出现最多的字符对
        best_pair = ""  # 当前成对出现最多的字符对
        max_freq = None  # 当前成对出现最多的字符对的次数
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        print("best_pair: ", best_pair)
        print("max_freq : ", max_freq)
        splits = merge_pair(*best_pair, splits, word_freq)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])  # 加入词表

    print("merges:", merges)
    print("vocab:", vocab)
