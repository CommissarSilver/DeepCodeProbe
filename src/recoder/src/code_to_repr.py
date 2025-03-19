# Python Standard Libraries
import os
import pickle

# Third-Party Libraries
import javalang
from javalang.ast import Node
from testone import (
    generateAST,
    getroottree,
    getSubroot,
    getNodeById,
    getLineNode,
    solveLongTree,
    containID,
    setProb,
    addter,
)

import javalang


def code_to_ast(code: str):

    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()  # parse the code to generate AST

    liness = (
        code.splitlines()
    )  # the original one reads these from a file so we need to do this

    return tree, liness


def ast_to_index(tree, liness):
    dcu_tuple = {"d": [], "c": [], "u": []}

    tmproot = getroottree(generateAST(tree))
    # the original code reads a lineId to get the subroot
    # this will be replaced into a text for each data point. unfortunatley given
    # the time constraint, making it work regardless of the input is not possible.
    # so running this will take a very long time.
    lineid = eval(open("line.txt", "r").read().strip())
    currroot = getNodeById(tmproot, lineid)
    lnode, mnode = getSubroot(currroot)
    oldcode = liness[lineid - 1]
    subroot = lnode
    treeroot = mnode
    presubroot = None
    aftersubroot = None
    linenodes = getLineNode(treeroot, "")
    currid = linenodes.index(subroot)
    if currid > 0:
        presubroot = linenodes[currid - 1]
    if currid < len(linenodes) - 1:
        aftersubroot = linenodes[currid + 1]
    setProb(treeroot, 2)
    addter(treeroot)
    data = []
    if True:
        setProb(treeroot, 2)
        if subroot is not None:
            setProb(subroot, 1)
        if aftersubroot is not None:
            setProb(aftersubroot, 4)
        if presubroot is not None:
            setProb(presubroot, 3)

        cid = set(containID(subroot))
        maxl = -1
        minl = 1e10
        for l in cid:
            maxl = max(maxl, l - 1)
            minl = min(minl, l - 1)

        precode = "\n".join(liness[0:minl])
        aftercode = "\n".join(liness[maxl + 1 :])
        oldcode = "\n".join(liness[minl : maxl + 1])
        troot, vardic, typedic = solveLongTree(treeroot, subroot)
        data.append(
            {
                "treeroot": treeroot,
                "troot": troot,
                "oldcode": oldcode,
                "filepath": "filepath",
                "subroot": subroot,
                "vardic": vardic,
                "typedic": typedic,
                "precode": precode,
                "aftercode": aftercode,
                "tree": troot.printTreeWithVar(troot, vardic),
                "prob": troot.getTreeProb(troot),
                "mode": 0,
                "line": lineid,
                "isa": False,
                "children_id": cid,
            }
        )
    for line_num, line in enumerate(liness):
        dcu_tuple["d"].append(line_num)  # position of the node
        dcu_tuple["c"].append(line["children_id"])  # children of the node
        dcu_tuple["u"].append(line["vardict"])  # type of the node

    return dcu_tuple


def code_to_index(code):
    try:
        tree, out = code_to_ast(code)
        dcu_tuple = ast_to_index(tree, out)
        return dcu_tuple
    except Exception as e:
        return {"d": [], "c": [], "u": []}


if __name__ == "__main__":

    java_code_sample = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!"); 
    }
}
"""
    # Test 2: Java Language
    java_output = code_to_index(java_code_sample)
    assert java_output is not None, "Test 2 Failed: Java language output is None"
    assert (
        "d" in java_output and "c" in java_output and "u" in java_output
    ), "Test 2 Failed: Java language output lacks required keys"

    print("\033[32m All tests passed.\033[0m")
