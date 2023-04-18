import argparse
import sys

from tools import PcfgGenerator, PcfgGrammar, flatten_tree, CKYParser


def show_parse(line, tree, gui):
    print(f'P(tree) = {tree.prob()}')
    print(f'log2(P(tree)) = {tree.logprob()}')
    if gui:
        tree.draw()
    else:
        print(tree)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-v', '--verbose', action='count', default=0,
                           help="verbose output")
    argparser.add_argument("-s", "--startsymbol", dest="startsym", type=str, default="START",
                           help="start symbol")
    argparser.add_argument("-a", "--allowedwords", dest="allowed_words_file", type=str,
                           default=None,
                           help="only use this list of words when parsing and generating")
    argparser.add_argument('--gui', dest='gui', action='count', default=False, help='instead of printing the tree,'
                                                                                    'draws the parse tree.')
    argparser.add_argument('--all', dest='all', action='count', default=False, help='instead of only the most probable '
                                                                                    'parse, returns all of them')
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
    sentences = []
    for line in sys.stdin:
        print('------------------------------')
        line = line.strip()
        if line == 'exit':  # exit the repl
            break
        sentences.append(line)
        if line.startswith('#'):  # comments
            if args.verbose:
                print("skip comment line: ", line)
            continue
        tokens = line.split()
        try:
            parsed = grammar.get_most_probable_parse(tokens)
        except ValueError as e:
            print(e)
            continue
        if parsed:
            if args.all:
                for tree in grammar.get_all_parses(tokens):
                    show_parse(line, tree, args.gui)
            else:
                show_parse(line, parsed, args.gui)
            sent_log2prob = CKYParser(grammar=grammar).log2prob_sentence(tokens)
            print(f'log2(P(sentence)) = {sent_log2prob}')
        else:
            print("couldn't parse: ", line)

    grammar.get_cross_entropy_over_sentences(sentences, verbose=True)
