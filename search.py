import os

os.system("conda activate evo-robots")

for _ in range(5):
    os.system("python simulate.py --generate")