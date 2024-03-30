import torch
from torch import nn
from d2l import torch as d2l
from text_preprocessing import read_time_machine, load_corpus_time_machine
import random
# 语言模型：字符级循环神经网络
# 语言模型是自然语言处理的重要技术，它可以用来预测文本序列
# 语言模型的输入是一个文本序列，输出也是一个文本序列
# 使用统计方法时，语言模型通常基于 n 元语法（n-gram）
# tokens = d2l.tokenize(read_time_machine())
# corpus = [token for line in tokens for token in line]
# vocab = d2l.Vocab(corpus)
# print(vocab.token_freqs[:10])

# freqs = [freq for token, freq in vocab.token_freqs]
# d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log',
#          yscale='log')
# d2l.plt.show()

# 下面几行运行会报错
# bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
# bigram_vocab = d2l.Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])

# trigram_tokens = [triple for triple in zip(
#     corpus[:-2], corpus[1:-1], corpus[2:])]
# trigram_vocab = d2l.Vocab(trigram_tokens)
# print(trigram_vocab.token_freqs[:10])


def seq_data_iter_random(corpus, batch_size, num_steps):
    # 生成一个随机偏移
    corpus = corpus[random.randint(0, num_steps-1):]
    # 减去1，因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs*num_steps, num_steps))
    # 在随机抽样中，迭代过程中两个相邻的随机小批量在原始序列上的位置不一定相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos开始的长度为num_steps的序列
        return corpus[pos:pos+num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size*num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 从随机偏移开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset+num_tokens])
    Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches*num_steps, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i:i+num_steps]
        yield X, Y


# for X, Y in seq_data_iter_random(list(range(10)), 2, 3):
#     print('X:', X, '\nY:', Y)


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps,
                              use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
# 总结
# 1. 语言模型估计文本序列的联合概率
# 2. 使用统计方法时，语言模型通常基于 n 元语法（n-gram）
