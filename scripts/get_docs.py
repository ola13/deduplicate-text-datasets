import argparse
import pickle
import pprint
import uuid
import subprocess
import string

from bisect import bisect_right
from datasets import load_from_disk
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)


def get_doc_id(pos, pos2id, pos2id_list):
    """
    Gets id of the datapoint at position.
    """
    pos = bisect_right(pos2id_list, pos)
    doc_id = pos2id[pos2id_list[pos - 1]]
    return doc_id


parser = argparse.ArgumentParser()
parser.add_argument("--query", "-q", type=str)
args = parser.parse_args()

pos2id = pickle.load(
    open(
        "/home/piktus_huggingface_co/lumi/dedup/oscar_025/oscar.train.pos2id.pkl", "rb"
    )
)
pos2id_list = sorted(pos2id.keys())
oscar = load_from_disk("/home/piktus_huggingface_co/lumi/preprocessed_data/oscar-dedup")


def _process_result(doc_id, query):
    def find_whitespace(text):
        for i, c in enumerate(text):
            if c in string.whitespace:
                yield i

    doc = oscar[doc_id]
    text = doc["text"]
    print(doc_id, text)
    pos = text.find(query)
    print("Found pos", pos)
    whitespace_idx = [-1] + list(find_whitespace(text)) + [len(text)]
    print(whitespace_idx)
    idx = bisect_right(whitespace_idx, pos)
    print("idx", idx)
    start = whitespace_idx[max(0, idx - 50)] + 1
    end = whitespace_idx[min(len(whitespace_idx) - 1, idx + 50)]
    print("start: {}, end: {}".format(start, end))
    return text[start:end]


query = args.query.encode("utf-8")
tmp_file = "/tmp/fin_{}".format(uuid.uuid4())
open(tmp_file, "wb").write(query)


cmd = (
    "./target/debug/dedup_dataset count-occurrences "
    "--data-file  /home/piktus_huggingface_co/lumi/dedup/oscar_025/oscar.train "
    "--query-file {}".format(tmp_file)
)

print(cmd)
cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
lines = cmd_result.stdout.decode("utf-8").split("\n")

prefix = "Found at: "
results = []
for line in tqdm(lines):
    if line.startswith(prefix):
        pos = int(line.strip()[len(prefix) :])
        doc_id = get_doc_id(pos, pos2id, pos2id_list)
        results.append(_process_result(doc_id, args.query))

for result in results:
    print(result)
    print()
    print()
