import ast, math, random, torch
from torch.autograd import Variable

try:
    import lib
except ImportError:
    import code_sum_drl.src.lib as lib

u_index = {}
u_index_counter = 0


class Dataset(object):
    def __init__(self, data, batchSize, cuda, eval=False):
        self.src = data["src"]
        self.tgt = data["tgt"]
        self.trees = data["trees"]
        self.leafs = data["leafs"]
        self.original_codes = data["original_codes"]
        self.original_comments = data["original_comments"]

        assert len(self.src) == len(self.tgt)
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = int(math.ceil(len(self.src) / batchSize) - 2)
        self.eval = eval

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(lib.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index <= self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, src_lengths = self._batchify(
            self.src[index * self.batchSize : (index + 1) * self.batchSize],
            include_lengths=True,
        )

        leafBatch, leaf_lengths = self._batchify(
            self.leafs[index * self.batchSize : (index + 1) * self.batchSize],
            include_lengths=True,
        )
        srcTrees = self.trees[index * self.batchSize : (index + 1) * self.batchSize]
        tgtBatch = self._batchify(
            self.tgt[index * self.batchSize : (index + 1) * self.batchSize]
        )

        srcCodes = self.original_codes[
            index * self.batchSize : (index + 1) * self.batchSize
        ]
        srcComments = self.original_comments[
            index * self.batchSize : (index + 1) * self.batchSize
        ]
        srcCodeIndexes = [code_to_index(code) for code in srcCodes]

        indices = range(len(srcBatch))
        src_batch = zip(
            indices,
            srcBatch,
            leafBatch,
            leaf_lengths,
            srcTrees,
            tgtBatch,
            srcCodes,
            srcComments,
            srcCodeIndexes,
        )

        src_batch, src_lengths = zip(
            *sorted(zip(src_batch, src_lengths), key=lambda x: -x[1])
        )

        (
            indices,
            srcBatch,
            leafBatch,
            leaf_lengths,
            srcTrees,
            tgtBatch,
            srcCodes,
            srcComments,
            srcCodeIndexes,
        ) = zip(*src_batch)

        tree_lengths = []
        for tree in srcTrees:
            l_c = tree.leaf_count()
            tree_lengths.append(l_c)

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            if self.eval:
                with torch.no_grad():
                    b = Variable(b)
            else:
                b = Variable(b)
            return b

        return (
            (wrap(srcBatch), src_lengths),
            (srcTrees, tree_lengths, (wrap(leafBatch), leaf_lengths)),
            wrap(tgtBatch),
            indices,
            srcCodes,
            srcComments,
            srcCodeIndexes,
        )

    def __len__(self):
        return self.numBatches - 1


def extract_node_info(node, parent=None, position=0):
    """
    Recursively extract node categories from the AST.
    """
    node_name = node.__class__.__name__
    node_position = position

    # Gather information about the node
    node_info = {"name": node_name, "position": node_position}

    # Add parent information if available
    if parent is not None:
        node_info["parent"] = parent.__class__.__name__

    # Recursively process child nodes
    for child_name, child_node in ast.iter_fields(node):
        if isinstance(child_node, list):
            for idx, child in enumerate(child_node):
                if isinstance(child, ast.AST):
                    child_info = extract_node_info(child, node, idx)
                    if "children" not in node_info:
                        node_info["children"] = []
                    node_info["children"].append(child_info)
        elif isinstance(child_node, ast.AST):
            child_info = extract_node_info(child_node, node, 0)
            if "children" not in node_info:
                node_info["children"] = []
            node_info["children"].append(child_info)

    return node_info


def extract_lists(ast_info):
    """
    Extract lists D, C, and U from AST information.
    """
    D = []
    C = []
    U = []

    # Helper function to recursively process AST information
    def process_node(node_info):
        D.append(node_info["position"])
        U.append(node_info["name"])

        if "children" in node_info:
            children = node_info["children"]
            C.append(len(children))
            for child in children:
                process_node(child)
        else:
            C.append(0)

    # Process the AST information
    process_node(ast_info)

    return D, C, U


def code_to_ast(code: str):
    code = code.replace(" DCNL DCSP ", "\n\t")
    code = code.replace(" DCNL  DCSP ", "\n\t")
    code = code.replace(" DCNL ", "\n")
    code = code.replace(" DCSP ", "\t")

    # create the ast
    try:
        tree = ast.parse(code)
        node_info = extract_node_info(tree)
        D, C, U = extract_lists(node_info)

        return D, C, U
    except Exception as e:
        return [], [], []


def ast_to_index(
    D: list,
    C: list,
    U: list,
):
    global u_index, u_index_counter
    for label in U:
        if label not in u_index.keys():
            u_index[label] = u_index_counter
            u_index_counter += 1

    final_d = D
    final_c = C
    final_u = [u_index[label] for label in U]

    return final_d, final_c, final_u


def code_to_index(code: str):
    D, C, U = code_to_ast(code)

    if (D, C, U) == ([], [], []):
        return {"d": [], "c": [], "u": []}
    else:
        D, C, U = ast_to_index(D, C, U)
        # fill the rest with 0 if the length is less than 100
        return {"d": D, "c": C, "u": U}


if __name__ == "__main__":
    path_to_code = (
        "/Users/ahura/Nexus/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code"
    )
    codes = open(path_to_code).readlines()
    errors = 0
    no_errors = 0
    for code in codes:
        # ds, cs, us = code_to_ast(code)
        # ds, cs, us = ast_to_index(ds, cs, us)
        test = code_to_index(code)
        if (test["d"], test["c"], test["u"]) == ([], [], []):
            errors += 1
        else:
            no_errors += 1

    print("errors: ", errors)
    print("no_errors: ", no_errors)
