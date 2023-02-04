import struct
from PIL import Image, ImageEnhance
import glob, os

RECORD_SIZE = 8199 

outdir = "./images"
if not os.path.exists(outdir): os.mkdir(outdir)

files = glob.glob("./ETL8G\*")
fc = 0

for fname in files:
  fc = fc + 1
  print(fname) 

  f = open(fname, 'rb')
  f.seek(0)
  i = 0
  while True:
    i = i + 1
    s = f.read(RECORD_SIZE)
    if not s: break
    r = struct.unpack(">HH8sIBBBBHHHHBB30x8128s11x", s)
    iF = Image.frombytes("F", (128, 127), r[14], "bit", (4, 0))
    iP = iF.convert('L')
    code_jis = r[1]
    if code_jis in range(int(0x2420), int(0x2474)):
        dir = outdir + "/" + str(code_jis)
        if not os.path.exists(dir): os.mkdir(dir)
        fn = "{0:02x}_{1:02x}.png".format(code_jis, r[0], r[2])
        fullpath = dir + "/" + fn
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(fullpath, 'PNG')
print("ok")