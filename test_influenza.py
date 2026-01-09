# %%
import json
from pathlib import Path

INPUT_CX = "A network leading to influenza onset obtained through our network structure estimation..cx"
with open(INPUT_CX, "r", encoding="utf-8") as f:
    cx_data = json.load(f)

# %%
for element in cx_data:
    print(element)
# %%
nodes = cx_data[7]
# %%
nodes_dict = {}
for node in nodes['nodes']:
    nodes_dict[node['@id']] = node['n']

# %%
edges = cx_data[4]
