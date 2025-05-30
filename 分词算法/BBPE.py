from collections import defaultdict

# 构建频率统计
def build_stats(sentences):
    stats = defaultdict(int)
    for sentence in sentences:
        symbols = sentence.split()
        for symbol in symbols:
            stats[symbol] += 1
    return stats


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


def merge_pair(pair, splits, word_freq):

    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == pair[0] and split[i+1] == pair[1]:  # 检查分割中是否有这对字节
                split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
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
    # 构建初始词汇表，包含一个字节的256个表示
    initial_vocab = [bytes([byte]) for byte in range(256)]
    vocab = initial_vocab.copy()
    print("initial_vocab:", initial_vocab)
    print(len(initial_vocab))
    word_freq = build_stats(sentences)
    print("word_freq:", word_freq)

    splits = {word: [byte.encode("utf-8") for byte in word] for word in word_freq.keys()}
    print("splits: ", splits)

    vocab_size = 100 + 256  # 256是一个字节的256个表示 100 为提取的字节对
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freq)
        best_pair = ()
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        print("best pair:", best_pair)
        splits = merge_pair(best_pair, splits, word_freq)
        vocab.append(best_pair[0]+best_pair[1])
    print("vocab:", vocab)
