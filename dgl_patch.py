import os
import sys
import types

# Set the flag early
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

# Create a dummy module for dgl.graphbolt
dummy_graphbolt = types.ModuleType("dgl.graphbolt")
# Define a no-op load_graphbolt function (or one that simply prints a message)
def load_graphbolt():
    print("Dummy load_graphbolt called. Skipping GraphBolt loading.")
dummy_graphbolt.load_graphbolt = load_graphbolt
# You may also add any attributes expected by DGL; for now this is sufficient.

# Insert our dummy module into sys.modules under the key 'dgl.graphbolt'
sys.modules["dgl.graphbolt"] = dummy_graphbolt

# Now import dgl; it should see our dummy module
import dgl

# (Optional) Expose all dgl symbols so you can do "import dgl_patch as dgl"
from dgl import *
