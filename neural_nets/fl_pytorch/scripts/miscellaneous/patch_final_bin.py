#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils import utils

if __name__ == "__main__":
    files = sys.argv[1:]
    python = "python"
    
    for fname in files:
        serialized = None
        with open(fname, "rb") as f:
            serialized = pickle.load(f)

        # serialized experimental information is a list of tuples (job_id, H)
        for (k, v) in serialized:
            group_new = v["comment"] + v["group-name"]
            comment_new = ""

            v["comment"] = comment_new
            v["group-name"] = group_new
            v["args"].comment = comment_new
            v["args"].group_name = group_new

            l = len(v["raw_cmdline"])
            for i in range(l):
                if v["raw_cmdline"][i] == "--comment":
                    v["raw_cmdline"][i + 1] = comment_new
                if v["raw_cmdline"][i] == "--group-name":
                    v["raw_cmdline"][i + 1] = group_new
    
        with open(fname + ".update", "wb") as f:
            pickle.dump(serialized, f)
            print("Results has been saved into: ", fname + ".update")
