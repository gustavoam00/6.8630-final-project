"""
A tool for generating random sentences given a grammar


    verbose: if verbose=True  some extra information will be printed while generating sentences
    random_seed: if random_seed is not None, then it will be used as the random seed
    grammar: grammar is a nltk.PCFG
Usage:
    gen = PcfgGenerator(grammar=grammar)
    tokens = gen.generate()


Author: shayan.p
"""


import random
import sys

from nltk import ProbabilisticTree
from nltk.grammar import Nonterminal


class GeneratorResetException(Exception):
    pass


class PcfgGenerator:
    def __init__(self, grammar, verbose=False, restart_count_limit=10000, random_seed=None):
        self.gram = grammar
        self.verbose = verbose
        self.restart_limit = restart_count_limit
        self.number_rule_expansions = 0
        if random_seed is not None:
            random.seed(random_seed)

    def generate(self):
        self.number_rule_expansions = 0  # restart the counter for rule expansions
        rule = self.gen_pick_one(self.gram.start())
        if self.verbose:
            print("#getrule: {}".format(rule), file=sys.stderr)
        try:
            gen_tree = self.gen_from_rule(rule)
        except GeneratorResetException as e:
            if self.verbose:
                print(e)
            return self.generate()  # reset and try again
        return gen_tree

    def gen_pick_one(self, lhs):
        r = random.random()
        if self.verbose:
            print("#random number: {}".format(r), file=sys.stderr)
        accumulator = 0.0
        rule_picked = None
        for rule in self.gram.productions(lhs=lhs):
            if self.verbose:
                print("#getrule: {}".format(r), file=sys.stderr)
            prob = rule.prob()
            if r < (prob + accumulator):
                rule_picked = rule
                break
            else:
                accumulator += prob
        if rule_picked is None:
            raise ValueError("no rule found for %s" % lhs)
        if self.verbose:
            print("#picked rule %s" % (rule_picked), file=sys.stderr)
        return rule_picked

    def get_yield(self, sym):
        if isinstance(sym, Nonterminal):
            return self.gen_from_rule(self.gen_pick_one(sym))
        else:
            return sym

    def gen_from_rule(self, rule):
        if self.verbose:
            print(rule, file=sys.stderr)
        self.number_rule_expansions += 1
        if self.number_rule_expansions > self.restart_limit:
            raise GeneratorResetException(f"The restart_limit={self.restart_limit} "
                                          f"for rule expansion is reached. Consider increasing it!")
        children = [self.get_yield(t) for t in rule.rhs()]
        logprob = rule.logprob() + sum(child.logprob() for child in children if isinstance(child, ProbabilisticTree))
        return ProbabilisticTree(node=rule.lhs().symbol(), children=children, logprob=logprob)
