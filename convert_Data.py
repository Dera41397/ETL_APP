from zipfile import ZipFile
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

DATA_PATH = "./"
RECORD_SIZE = 8199
etl = []
info = []

with ZipFile(DATA_PATH + "ETL8G.zip") as etl1:
    names = [n for n in etl1.namelist() if "_" in n]
    for x in names:
        with etl1.open(x) as f:
            while True:
                s = f.read(RECORD_SIZE)
                if (s is None) or (len(s) < RECORD_SIZE):
                    break
                r = struct.unpack(">HH8sIBBBBHHHHBB30x8128s11x", s)
                img = Image.frombytes("F", (128, 127), r[14], "bit", (4, 0))
                img = np.array(img.convert("L"))  # 0..15
                lbl = r[1]
                if lbl in range(int(0x2420), int(0x2474)):
                    etl.append((img, lbl))
                    info.append(r[:-1])


ar = []
with open(DATA_PATH + "JIS0208.TXT") as f:
    for t_line in f:
        if t_line[0] != "#":
            sjis, jis, utf16 = os.path.basename(t_line).split("\t")[0:3]
            ar.append([jis, utf16])
ar = dict(ar)

def decoder(x):
    x = str(hex(x))[2:]
    return chr(int(ar["0x"+x.upper()], 16))

plt.imshow(etl[0][0], cmap="gray_r")
plt.show()

print(decoder(etl[0][1]))