from nltk import ProbabilisticTree


def read_allowed_words(allowed_words_file):
    return set(line.strip() for line in open(allowed_words_file))


def flatten_tree(tree):
    assert isinstance(tree, ProbabilisticTree)
    return tree.leaves()
