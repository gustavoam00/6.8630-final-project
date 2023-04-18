"""
A wrapper on nltk.PCFG that has some extra functionality.


    verbose: if verbose=True  some extra information will be printed while generating sentences
    random_seed: if random_seed is not None, then it will be used as the random seed
    grammar: grammar is a nltk.PCFG
Usage:
    grammar = PcfgGrammar.read_grammar(grammar=grammar_files, allowed_words_file=allowed_words_file)
    # grammar_files is a list of paths to grammar files. The function reads them all
    # allowed_words_file is an optional parameter.
    # If its noe None, the function checks if all terminals words are in allowed_words

    grammar.get_most_probable_parse(tokens)
    # uses CKYParser to return the most probable parse
    # all parsing functions (except for can_parse function) return MissingVocabException if they encounter unknown vocab
Author: shayan.p
"""
import nltk
import numpy as np

from nltk import PCFG
from nltk.grammar import ProbabilisticProduction, Nonterminal
from collections import defaultdict

from .cky import CKYParser
from .utils import flatten_tree, read_allowed_words


class MissingVocabException(ValueError):
    def __init__(self, missed_words):
        super(MissingVocabException, self).__init__(f"words {missed_words} are not in allowed words")
        self.missed_words = set(missed_words)


def _complain_if_str_passed_as_tokens(tokens):
    if isinstance(tokens, str):
        raise ValueError("passed a string. you should pass a list of tokens instead.")


class PcfgGrammar(PCFG):
    def __init__(self, productions, start=Nonterminal("START")):
        super(PcfgGrammar, self).__init__(start=start, productions=productions)
        self._terminals = []
        for p in productions:
            self._terminals.extend([w for w in p.rhs() if (not isinstance(w, Nonterminal))])
        self._terminals = set(self._terminals)

    def check_missing_vocab(self, tokens):
        _complain_if_str_passed_as_tokens(tokens)
        missed_words = [token for token in tokens if (token not in self._terminals)]
        if missed_words:
            raise MissingVocabException(missed_words)

    def parse(self, tokens):
        _complain_if_str_passed_as_tokens(tokens)
        yield from CKYParser(grammar=self).parse(tokens)

    def get_all_parses(self, tokens, number_limit=1000):
        _complain_if_str_passed_as_tokens(tokens)
        self.check_missing_vocab(tokens)
        result = []
        for i, tree in zip(range(number_limit), self.parse(tokens)):
            result.append(tree)
        return result

    def get_most_probable_parse(self, tokens):
        _complain_if_str_passed_as_tokens(tokens)
        self.check_missing_vocab(tokens)
        return next(self.parse(tokens=tokens), None)

    def get_cross_entropy_over_sentences(self, sentences, verbose=False):
        total_tokens = 0
        total_log_prob = 0
        for line in sentences:
            tokens = line.strip().split()
            sent_log2prob = CKYParser(grammar=self).log2prob_sentence(tokens)
            if sent_log2prob is -np.inf:
                if verbose:
                    print("found an unparsed sentence, thus cross-entropy is infinity")
                return np.inf
            total_log_prob += sent_log2prob
            total_tokens += len(tokens)
        if total_tokens:
            cross_entropy = -total_log_prob / total_tokens
            if verbose:
                print(f"cross-entropy = {cross_entropy:.3f} "
                      f"bits = - ({total_log_prob:.3f} logprob / {total_tokens} words)")
            return cross_entropy
        else:
            if verbose:
                print("no parse tree is given, thus cross-entropy is 0.0")
            return 0.0

    def can_parse(self, tokens):
        _complain_if_str_passed_as_tokens(tokens)
        try:
            self.check_missing_vocab(tokens)
            return self.get_most_probable_parse(tokens) is not None
        except ValueError:
            return False

    # if allowed_words_file is not passed, the grammar will not check if the words are all allowed
    @staticmethod
    def read_grammar(grammar_files, startsym='START', allowed_words_file=None, verbose=False):
        rules = defaultdict(lambda: defaultdict(lambda: 0))
        for filename in grammar_files:
            if verbose:
                print("#reading grammar file: {}".format(filename))
            for linenum, _line in enumerate(open(filename, 'r')):
                if _line.find('#') != -1:
                    _line = _line[:_line.find('#')]  # strip comments
                _line = _line.strip()
                if _line == "":
                    continue
                f = _line.split()
                if len(f) < 2:
                    raise ValueError("Error: unexpected line at line %d: %s"
                                     % (linenum, ' '.join(f)))
                try:
                    count = float(f[0])
                except ValueError:
                    raise ValueError(f"Rule must be COUNT LHS RHS. Found {f}")
                if count <= 0:
                    raise ValueError(f"Probabilities should be positive. Found {f}")

                if len(f) == 2:
                    # empty rule
                    raise ValueError(f"Error: Rule goes to null at line {linenum} {f}")

                lhs = f[1]
                rhs = f[2:]

                if count <= 0:
                    if verbose:
                        print(f"#Ignored rule {lhs} -> {rhs} because count={count} <= 0")
                    continue

                if len(rhs) == 1 and lhs == rhs[0]:
                    if verbose:
                        print(f"#Ignored cycle {lhs} -> {rhs}")
                    continue
                rules[lhs][tuple(rhs)] += count

        rules = {lhs: [(count, rhs) for rhs, count in rules[lhs].items()]
                 for lhs in rules}

        # normalize probabilities
        sum_rules = defaultdict(lambda: 0)
        for lhs, rhs_list in rules.items():
            sum_rules[lhs] += sum(count for count, rhs in rhs_list)
        rules = {lhs: [(count / sum_rules[lhs], rhs) for count, rhs in rhs_list]
                 for lhs, rhs_list in rules.items()}

        to_non_terminal = {lhs: Nonterminal(lhs) for lhs in rules.keys()}

        if allowed_words_file is not None:
            allowed_words = read_allowed_words(allowed_words_file)
        else:
            allowed_words = None

        productions = []

        if startsym not in to_non_terminal:
            to_non_terminal[startsym] = Nonterminal(startsym)

        for lhs, rule_list in rules.items():
            if allowed_words_file is not None:
                for prob, rhs in rule_list:
                    for word in rhs:
                        if (word not in allowed_words) and (word not in to_non_terminal):
                            raise ValueError(f"the word {word} in rule {lhs} -> {rhs} is not allowed")
            for prob, rhs in rule_list:
                _lhs = to_non_terminal[lhs]
                _rhs = [to_non_terminal.get(x, x) for x in rhs]
                prod = ProbabilisticProduction(_lhs, _rhs, prob=prob)
                productions.append(prod)
        return PcfgGrammar(start=to_non_terminal[startsym], productions=productions)


class DynamicGrammar:
    def __init__(self, rules=[]):
        self.lhs_dict = {}  # Dict[lhs, Dict[rhs, weight]]
        for r in rules:  # either tuple or ProbabilisticProduction
            if isinstance(r, nltk.ProbabilisticProduction):
                r = r.lhs(), r.rhs(), r.prob()
            lhs, rhs, w = r
            self.update_weight(lhs, rhs, w)

    def update_weight(self, lhs, rhs, weight):
        rhs = tuple(rhs)
        if lhs not in self.lhs_dict:
            self.lhs_dict[lhs] = {}
        self.lhs_dict[lhs][rhs] = weight

    def add_weight(self, lhs, rhs, delta_weight):
        rhs = tuple(rhs)
        if lhs not in self.lhs_dict:
            self.lhs_dict[lhs] = {}
        self.lhs_dict[lhs][rhs] = self.lhs_dict[lhs].get(rhs, 0) + delta_weight

    def remove_lhs(self, lhs):
        if lhs in self.lhs_dict:
            self.lhs_dict.pop(lhs)

    def remove_rule(self, lhs, rhs):
        rhs = tuple(rhs)
        if lhs in self.lhs_dict:
            if rhs in self.lhs_dict[lhs]:
                self.lhs_dict[lhs].pop(rhs)

    def normalize_weights(self):
        for lhs, rhs_list in list(self.lhs_dict.items()):
            sm = sum(rhs_list.values())
            self.lhs_dict[lhs] = {rhs: w/sm for rhs, w in rhs_list.items()}

    def get_pcfg_grammar(self):
        productions = []
        for lhs, rhs_list in self.lhs_dict.items():
            sm = sum(rhs_list.values())
            productions.extend([ProbabilisticProduction(lhs=lhs, rhs=rhs, prob=w/sm) for rhs, w in rhs_list.items()])
        return PcfgGrammar(productions)

    def export_to_file(self, vocab_filepath, rules_filepath):
        rules = []
        for lhs, rhs_list in self.lhs_dict.items():
            rules.extend([(lhs, rhs, w) for rhs, w in rhs_list.items()])
        vocab_rules, other_rules = DynamicGrammar.split_vocab_rules_productions(rules)
        DynamicGrammar.export_productions_to_file(vocab_rules, vocab_filepath)
        DynamicGrammar.export_productions_to_file(other_rules, rules_filepath)

    @staticmethod
    def split_vocab_rules_productions(rules):
        vocab_productions = []
        other_productions = []
        for lhs, rhs, w in rules:
            if all(isinstance(r, str) for r in rhs):
                vocab_productions.append((lhs, rhs, w))
            else:
                other_productions.append((lhs, rhs, w))
        return vocab_productions, other_productions

    @staticmethod
    def export_productions_to_file(rules, filepath):
        with open(filepath, 'w') as f:
            for lhs, rhs, w in rules:
                f.write(f'{w}\t{lhs}\t{" ".join(map(str, rhs))}\n')
