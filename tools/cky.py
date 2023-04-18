"""
A tool useful for parsing sentences with a grammar.

Usage:
    parser = CKYParser(grammar=grammar)

    parser.parse(tokens)
    # returns a generator that produces all tree parses one by one in decreasing order of probability

    # for example one can iterate over the first 10 parses by:
    for i, tree in zip(range(10), parser.parse(tokens)):
        tree.draw()

    # one can return the most probable parse by:
    parser.parse_one(tokens)

    This parser inherits from nltk.ParserI so it supports all nltk parser methods such as `parser.parse_one`

Author: shayan.p
"""


import copy
import heapq

import numpy as np

from collections import defaultdict
from nltk.parse import ParserI
from nltk.grammar import PCFG
from heapq import heappush, heappop, heapify

from nltk import ProbabilisticTree, ProbabilisticProduction, Nonterminal


class CKYParser(ParserI):
    log_eps = 1e-6
    binarize_padding = "@$@"

    def __init__(self, grammar: "PCFG"):
        self.unary_productions, self.binary_productions = CKYParser.get_binarize_grammar(grammar)
        self.start_terminal = grammar.start()
        
        self.unary_token_forward = {}
        self.unary_token_backward = {}
        for prod in self.unary_productions:
            A, B = prod.lhs(), prod.rhs()[0]
            self.unary_token_forward[A] = self.unary_token_forward.get(A, [])
            self.unary_token_backward[B] = self.unary_token_backward.get(B, [])
            self.unary_token_forward[A].append((B, prod))
            self.unary_token_backward[B].append((A, prod))
            

    @staticmethod
    def get_binarize_grammar(grammar: "PCFG"):
        unary_productions = []
        binary_productions = []
        """
            rule count is to avoid collisions of rules. e.g. if we have:
            A -> B C X [p1]
            A -> B D X [p2]

            we would get
            A -> B 1|A|B  [p1]
            1|A|B -> C X [1]

            A -> B 2|A|B  [p2]
            2|A|B -> D X [1]    
        """
        rule_count = 0
        for rule in grammar.productions():
            if np.log(rule.prob()) == -np.inf:
                continue  # skip the rules with probability 0

            rule_count += 1
            if len(rule.rhs()) > 2:
                # this rule needs to be broken down
                left_side = rule.lhs()
                for k in range(0, len(rule.rhs()) - 2):
                    tsym = rule.rhs()[k]
                    new_sym = Nonterminal(((f'{rule_count}' + CKYParser.binarize_padding) if k == 0 else "")
                                          + left_side.symbol() + CKYParser.binarize_padding + str(tsym))
                    new_production = ProbabilisticProduction(lhs=left_side, rhs=(tsym, new_sym),
                                                             prob=rule.prob() if k == 0 else 1)
                    left_side = new_sym
                    binary_productions.append(new_production)
                last_prd = ProbabilisticProduction(left_side, rule.rhs()[-2:], prob=1)
                binary_productions.append(last_prd)
            elif len(rule.rhs()) == 2:
                binary_productions.append(rule)
            elif len(rule.rhs()) == 1:
                unary_productions.append(rule)
            else:
                assert False
        return unary_productions, binary_productions

    @staticmethod
    def de_binarize_tree(tree: "ProbabilisticTree"):
        copy_tree = copy.deepcopy(tree)
        copy_tree.clear()
        for child in tree:
            if not isinstance(child, ProbabilisticTree):
                copy_tree.append(child)
                continue
            res = CKYParser.de_binarize_tree(child)
            if CKYParser.binarize_padding in child.label():
                copy_tree.extend(res)
            else:
                copy_tree.append(res)
        return copy_tree

    def initializeMatrix(self, tokens):  # insert lexical items w=w1 w2 ... wn
        positions = len(tokens) + 1
        matrix = [[defaultdict(lambda: TreeProbabilityParse(prob=0)) for i in range(positions)] for j in range(positions)]
        for (i, wi) in enumerate(tokens):  # enumerate numbers the elements of w from 0
            matrix[i][i + 1][wi] |= TreeProbabilityParse(prob=1)  # adding the string
            self.extend_unary_rule(matrix[i][i + 1])
        return matrix

    def extend_binary_rule(self, root_dict, left_dict, right_dict):
        for rule in self.binary_productions:
            y, z = rule.rhs()
            if (y in left_dict) and (z in right_dict):
                root_dict[rule.lhs()] |= left_dict[y] & right_dict[z] & rule

    def extend_unary_rule(self, root_dict):
        initial_root_dict = root_dict.copy()
        
        queue = list(root_dict.keys())
        inqueue = set(root_dict.keys())
        pt = 0  # pointer to the start of the queue
        
        while pt < len(queue):
            B = queue[pt]
            inqueue.remove(B)
            pt += 1
            for A, _prod in self.unary_token_backward.get(B, []):
                new_prob_parse = TreeProbabilityParse(prob=0)
                for B0, prod0 in self.unary_token_forward.get(A, []):
                    if B0 in root_dict:
                        new_prob_parse |= root_dict[B0] & prod0
                        
                if A in initial_root_dict:
                    new_prob_parse |= initial_root_dict[A]
                changes = False
                if A not in root_dict:
                    changes = True
                else:
                    prev_prob_parse = root_dict[A]
                    mx_diff = abs(prev_prob_parse.logprob_max() - new_prob_parse.logprob_max())
                    sm_diff = abs(prev_prob_parse.logprob_sum() - new_prob_parse.logprob_sum())
                    changes |= max(mx_diff, sm_diff) > CKYParser.log_eps
                if changes:
                    root_dict[A] = new_prob_parse
                    if A not in inqueue:
                        inqueue.add(A)
                        queue.append(A)

    def closeMatrix(self, matrix):
        positions = len(matrix)
        for width in range(2, positions):
            for start in range(positions - width):
                end = start + width
                for mid in range(start + 1, end):  # so then range stops with end-1
                    self.extend_binary_rule(matrix[start][end], matrix[start][mid], matrix[mid][end])
                self.extend_unary_rule(matrix[start][end])

    def calculateMatrix(self, tokens):
        matrix = self.initializeMatrix(tokens)
        self.closeMatrix(matrix)
        return matrix

    @staticmethod
    def merge_sorted_generators(prob_gen):
        next_logprobs = [prob for prob, gen in prob_gen]
        generators = [gen for prob, gen in prob_gen]
        ret_next = [None for prob, gen in prob_gen]
        queue = [(-prob, i) for i, (prob, gen) in enumerate(prob_gen)]
        heapq.heapify(queue)
        # negate to choose the greatest

        while len(queue) > 0:
            _, idx = heapq.heappop(queue)
            if ret_next[idx] is None:
                ret_next[idx] = next(generators[idx])
            yield ret_next[idx]
            nxt_tree = next(generators[idx], None)
            if nxt_tree is not None:
                next_logprobs[idx] = nxt_tree.logprob()
                ret_next[idx] = nxt_tree
                heapq.heappush(queue, (-next_logprobs[idx], idx))

    def merge_yield_binary_rule(self, rule, i, mid, j, tokens, matrix):
        a_state, b_state = rule.rhs()[0], rule.rhs()[1]  # ->A B
        a_generator = self.explore(a_state, i, mid, tokens, matrix)
        b_generator = self.explore(b_state, mid, j, tokens, matrix)
        queue = []
        a0 = next(a_generator, None)
        b0 = next(b_generator, None)
        if (a0 is None) or (b0 is None):
            return
        alist = [a0]
        blist = [b0]

        # negate to choose the greatest in heap
        neg_get_logprob = lambda a, b: -(a.logprob() if isinstance(a, ProbabilisticTree) else 0) - \
                                   (b.logprob() if isinstance(b, ProbabilisticTree) else 0) - rule.logprob()
        heapq.heappush(queue, (neg_get_logprob(a0, b0), 0, 0))

        while len(queue):
            my_neg_logprob, ai, bi = heapq.heappop(queue)

            yield ProbabilisticTree(str(rule.lhs()), children=[alist[ai], blist[bi]], logprob=-my_neg_logprob)

            if ai+1 == len(alist):
                nxt = next(a_generator, None)
                if nxt is not None:
                    alist.append(nxt)
                    for i, b in enumerate(blist):
                        heapq.heappush(queue, (neg_get_logprob(nxt, b), len(alist)-1, i))
            if bi+1 == len(blist):
                nxt = next(b_generator, None)
                if nxt is not None:
                    blist.append(nxt)
                    for i, a in enumerate(alist):
                        heapq.heappush(queue, (neg_get_logprob(a, nxt), i, len(blist)-1))

    def merge_yield_unary_rule(self, rule, i, j, tokens, matrix):
        for child in self.explore(current_state=rule.rhs()[0], i=i, j=j, tokens=tokens, matrix=matrix):
            my_logprob = (child.logprob() if isinstance(child, ProbabilisticTree) else 0) + rule.logprob()
            yield ProbabilisticTree(str(rule.lhs()), children=[child], logprob=my_logprob)

    def explore(self, current_state, i, j, tokens, matrix):
        if not isinstance(current_state, Nonterminal):
            assert j - i == 1
            yield tokens[i]
        prob_generators = []
        for rule in self.binary_productions:
            if current_state != rule.lhs():
                continue
            b_state, c_state = rule.rhs()  # A->BC
            for mid in range(i + 1, j):
                if (b_state in matrix[i][mid]) and (c_state in matrix[mid][j]):
                    my_prob = matrix[i][mid][b_state] & matrix[mid][j][c_state] & rule
                    prob_generators.append(
                        (my_prob.logprob_max(), self.merge_yield_binary_rule(rule, i, mid, j, tokens, matrix)))
        for rule in self.unary_productions:
            if current_state != rule.lhs():
                continue
            b_state, = rule.rhs()  # A->B
            if b_state in matrix[i][j]:
                my_prob = matrix[i][j][b_state] & rule
                prob_generators.append((my_prob.logprob_max(), self.merge_yield_unary_rule(rule, i, j, tokens, matrix)))
        yield from self.merge_sorted_generators(prob_generators)

    def parse(self, tokens, *args, **kwargs):
        matrix = self.calculateMatrix(tokens)
        yield from map(CKYParser.de_binarize_tree,
                       self.explore(self.start_terminal, 0, len(tokens), tokens, matrix))

    def log2prob_sentence(self, tokens):
        matrix = self.calculateMatrix(tokens)
        if self.start_terminal not in matrix[0][len(tokens)]:
            return -np.inf
        return matrix[0][len(tokens)][self.start_terminal].logprob_sum()

    def logprob_sentence(self, tokens):
        return self.log2prob_sentence(tokens) * np.log(2)
    
    def prob_sentence(self, tokens):  # use this with care! prob=0 doesn't mean there is no parse...
        return np.exp2(self.log2prob_sentence(tokens))


class TreeProbabilityParse:
    def __init__(self, **kwargs):
        self.__max_logprob = None
        self.__sum_logprob = None
        if "prob" in kwargs:
            if "logprob" in kwargs:
                raise TypeError("Must specify either prob or logprob " "(not both)")
            else:
                self._init_set_sum_prob(kwargs["prob"])
                self._init_set_max_prob(kwargs["prob"])
        elif "logprob" in kwargs:
            self._init_set_sum_logprob(kwargs["logprob"])
            self._init_set_max_logprob(kwargs["logprob"])
        else:
            assert False

    def _init_set_max_prob(self, prob):
        if prob == 0:
            self.__max_logprob = -np.inf
        else:
            self.__max_logprob = np.log2(prob)

    def _init_set_max_logprob(self, logprob):
        self.__max_logprob = logprob

    def _init_set_sum_prob(self, prob):
        if prob == 0:
            self.__sum_logprob = -np.inf
        else:
            self.__sum_logprob = np.log2(prob)

    def _init_set_sum_logprob(self, logprob):
        self.__sum_logprob = logprob

    def prob_max(self):
        if self.__max_logprob == -np.inf:
            return 0.0
        else:
            return np.exp2(self.__max_logprob)

    def prob_sum(self):
        if self.__sum_logprob == -np.inf:
            return 0.0
        else:
            return np.exp2(self.__sum_logprob)

    def logprob_max(self):
        return self.__max_logprob

    def logprob_sum(self):
        return self.__sum_logprob

    def __and__(self, other):
        if not isinstance(other, TreeProbabilityParse):
            other = TreeProbabilityParse(logprob=other.logprob())
        mul_max = other.logprob_max() + self.logprob_max()
        mul_sum = other.logprob_sum() + self.logprob_sum()
        p = TreeProbabilityParse(logprob=mul_max)
        p._init_set_sum_logprob(mul_sum)
        return p

    def __or__(self, other):
        if not isinstance(other, TreeProbabilityParse):
            other = TreeProbabilityParse(logprob=other.logprob())
        or_max = max(self.logprob_max(), other.logprob_max())
        or_sum = np.logaddexp2(self.logprob_sum(), other.logprob_sum())
        p = TreeProbabilityParse(logprob=or_max)
        p._init_set_sum_logprob(or_sum)
        return p

    def __repr__(self):
        return f"prob_max={self.prob_max()} prob_sum={self.prob_sum()}"
