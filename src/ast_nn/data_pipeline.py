import os, sys, warnings, logging
import pandas as pd

from tqdm.auto import tqdm

tqdm.pandas()
logger = logging.getLogger("data_processing")


class Pipeline:
    def __init__(self, ratio, root, language: str):
        self.language = language.lower()
        assert self.language in ("c", "java")
        self.ratio = ratio
        self.root = os.path.join(os.getcwd(), "src", "ast_nn", root)
        self.sources = None
        self.blocks = None
        self.pairs = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    def get_parsed_source(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Parse code using pycparser

        If the user doesn't provide `output_file`, the method reads the
        a DataFrame containing the columns id, code (C/Java code parsed
        by pycparser) and label. Otherwise it reads a Dataframe from
        `input_file` containing the columns id, code (input C/Java code)
        and label, applies the c_parser/javalang to the code column and
        stores the resulting dataframe into `output_file`

        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file

        Returns:
            pd.DataFrame: DataFrame with the columns id, code (C/Java code
                parsed by pycparser/javalang) and label.
        """
        input_path = os.path.join(self.root, self.language, input_file)

        if output_file is None:
            source = pd.read_pickle(input_path)
            logger.info("Inputs have already been processed. Loadgin processed file.")
        else:
            output_path = os.path.join(self.root, self.language, output_file)
            logger.info("No processed inputs found. Parsing inputs.")

            if self.language == "c":
                from pycparser import c_parser

                parser = c_parser.CParser()
                source = pd.read_pickle(input_path)
                source.columns = ["id", "code", "label"]
                source["code"] = source["code"].progress_apply(parser.parse)
                source.to_pickle(output_path) if output_file else None

                logger.info("Finished parsing C code")
            else:
                import javalang

                def parse_program(func):
                    try:
                        tokens = javalang.tokenizer.tokenize(func)
                        parser = javalang.parser.Parser(tokens)
                        tree = parser.parse_member_declaration()
                        return tree
                    except Exception as e:
                        logger.warning(
                            "There was a problem in processing Java code: %s. Skipped", e
                        )

                source = pd.read_csv(input_path, delimiter="\t")
                source.columns = ["id", "code"]
                source["code"] = source["code"].progress_apply(parse_program)
                source.to_pickle(output_path) if output_file else None

                logger.info("Finished parsing Java code")

        self.sources = source

        return source

    def read_pairs(self, filename: str):
        """Create clone pairs

        Args:
            filename (str): [description]
        """
        pairs = pd.read_pickle(os.path.join(self.root, self.language, filename))
        self.pairs = pairs
        logger.info("Finished reading clone pairs")

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root + "/" + self.language + "/"
        data = self.pairs
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(":")]
        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = data_path + "train/"
        check_or_create(train_path)
        self.train_file_path = train_path + "train_.pkl"
        train.to_pickle(self.train_file_path)

        dev_path = data_path + "dev/"
        check_or_create(dev_path)
        self.dev_file_path = dev_path + "dev_.pkl"
        dev.to_pickle(self.dev_file_path)

        test_path = data_path + "test/"
        check_or_create(test_path)
        self.test_file_path = test_path + "test_.pkl"
        test.to_pickle(self.test_file_path)

        logger.info("Finished splitting data into train, dev and test sets")

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root + "/" + self.language + "/"

        if not input_file:
            input_file = self.train_file_path

        pairs = pd.read_pickle(input_file)
        train_ids = pairs["id1"].append(pairs["id2"]).unique()

        trees = self.sources.set_index("id").reindex(train_ids).dropna()
        # trees = self.sources.set_index("id", drop=True).loc[train_ids]

        if not os.path.exists(data_path + "embeddings"):
            os.mkdir(data_path + "embeddings")

        if self.language == "c":
            sys.path.append("../")
            from prepare_data import get_sequences as func
        else:
            from utils import get_sequence as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence

        corpus = trees["code"].apply(trans_to_sequences)
        str_corpus = [" ".join(c) for c in corpus]
        trees["code"] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec

        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path + "embeddings/node_w2v_" + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language == "c":
            from prepare_data import get_blocks as func
        else:
            from utils import get_blocks_v1 as func

        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(
            self.root + "/" + self.language + "/embeddings/node_w2v_" + str(self.size)
        ).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.DataFrame(self.sources, copy=True)

        trees["code"] = trees["code"].apply(trans2seq)
        if "label" in trees.columns:
            trees.drop("label", axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self, data_path, part):
        pairs = pd.read_pickle(data_path)
        pairs["id1"] = pairs["id1"].astype(int)
        pairs["id2"] = pairs["id2"].astype(int)
        df = pd.merge(pairs, self.blocks, how="left", left_on="id1", right_on="id")
        df = pd.merge(df, self.blocks, how="left", left_on="id2", right_on="id")
        df.drop(["id_x", "id_y"], axis=1, inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root + "/" + self.language + "/" + part + "/blocks.pkl")

    # run for processing data to train
    def run(self):
        input_file = "programs.pkl" if self.language == "c" else "programs.tsv"
        if os.path.exists(os.path.join(self.root, self.language, "programs_ast.pkl")):
            self.get_parsed_source(input_file="programs_ast.pkl")
        else:
            self.get_parsed_source(input_file=input_file, output_file="programs_ast.pkl")

        self.read_pairs("clone_ids.pkl")

        # print("split data...")
        self.split_data()
        # print("train word embedding...")
        self.dictionary_and_embedding(None, 128)
        # print("generate block sequences...")
        self.generate_block_seqs()
        # print("merge pairs and blocks...")
        self.merge(self.train_file_path, "train")
        self.merge(self.dev_file_path, "dev")
        self.merge(self.test_file_path, "test")


def process_input(input: str, lang: str, word2vec_path: str):
    from gensim.models.word2vec import Word2Vec

    # get AST of input
    if lang == "c":
        from pycparser import c_parser
        from prepare_data import get_blocks as func

        code_ast = c_parser.CParser().parse(input)
    elif lang == "java":
        import javalang
        from utils import get_blocks_v1 as func

        tokens = javalang.tokenizer.tokenize(input)
        parser = javalang.parser.Parser(tokens)
        code_ast = parser.parse_member_declaration()
    # load appropriate word2vec model
    word2vec = Word2Vec.load(word2vec_path).wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    # convert AST to index representation
    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
        children = node.children
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        func(r, blocks)
        tree = []
        for b in blocks:
            btree = tree_to_index(b)
            tree.append(btree)
        return tree

    code_tree = trans2seq(code_ast)
    return code_tree


# @click.command()
# @click.option(
#     "--lang", required=True, type=str, help="Language for the code input ('c' or 'java')"
# )
def main(lang):
    ppl = Pipeline("3:1:1", "dataset", "java")
    ppl.run()


if __name__ == "__main__":
    main("java")
