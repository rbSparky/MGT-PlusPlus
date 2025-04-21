import os
import sys
import types

os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

dummy_graphbolt = types.ModuleType("dgl.graphbolt")
def load_graphbolt():
    print("Dummy load_graphbolt called. Skipping GraphBolt loading.")
dummy_graphbolt.load_graphbolt = load_graphbolt

sys.modules["dgl.graphbolt"] = dummy_graphbolt

import dgl

from dgl import *
