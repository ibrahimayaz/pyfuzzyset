import os
os.chdir("..")

with open("VERSION.txt") as f:
    __version__ = f.read()
