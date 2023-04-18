import argparse
import sys

from tools import PcfgGrammar, PcfgGenerator, flatten_tree


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-v', '--verbose', action='count', default=0,
                           help="verbose output")
    argparser.add_argument("-r", "--seed", dest="random_seed", type=int, default=None,
                           help="the random seed for generating random sentences")
    argparser.add_argument("-s", "--startsymbol", dest="startsym", type=str, default="START",
                           help="start symbol")
    argparser.add_argument("-n", "--numsentences", dest="num_sentences", type=int, default=20,
                           help="number of sentences to generate; in --generate mode")
    argparser.add_argument("-a", "--allowedwords", dest="allowed_words_file", type=str,
                           default=None,
                           help="only use this list of words when parsing and generating")
    argparser.add_argument("-g", "--grammars", nargs=argparse.ONE_OR_MORE, dest="grammar_files",
                           type=str, default=["S1.gr", "S2.gr", "Vocab.gr"],
                           help="list of grammar files; typically: S1.gr S2.gr Vocab.gr")

    args = argparser.parse_args()

    if not args.grammar_files:
        print("ERROR: grammar files required", file=sys.stderr)
        argparser.print_help(sys.stderr)
        sys.exit(2)

    # if not args.allowed_words_file:
    #     print("ERROR: allowed words filename required", file=sys.stderr)
    #     argparser.print_help(sys.stderr)
    #     sys.exit(2)

    if args.verbose:
        print("#verbose level: {}".format(args.verbose), file=sys.stderr)
        print("#mode: {}".format("parse" if args.parse_mode else "generate"), file=sys.stderr)
        print("#grammar: {}".format(" ".join(args.grammar_files)), file=sys.stderr)

    grammar = PcfgGrammar.read_grammar(args.grammar_files, args.startsym, args.allowed_words_file, args.verbose)

    gen = PcfgGenerator(grammar, verbose=args.verbose, random_seed=args.random_seed)
    for _ in range(args.num_sentences):
        print(" ".join(flatten_tree(gen.generate())))
