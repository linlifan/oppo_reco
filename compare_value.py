import numpy as np
import pickle
import argparse
import sys


parser = argparse.ArgumentParser()

handle = open("feed_dict_f32.pkl", "rb")

f32_feed = pickle.load(handle)

handle = open("feed_dict_bf16.pkl", "rb")

bf16_feed = pickle.load(handle)

for k,v in f32_feed.items():
    print(np.sum(bf16_feed[k] - v))

#exit()


if len(sys.argv) > 2:
    file1 = sys.argv[1]
    file2 = sys.argv[2]



#f32 = np.load("float32_genu.npy")
#bf16 = np.load("float32_bf32.npy")
#bf16 = np.load("bfloat16.npy")
f32 = np.load(file1)
bf16 = np.load(file2)

exclude = np.where(f32 == 0)

mask = np.ones(f32.shape, bool)
mask[exclude] = False

print(mask)
#exit()

diff_ratio = np.mean( np.abs( (bf16[mask] - f32[mask]) / f32[mask] ) )

print("f32 mean {0:f}, bf16 mean {1:f}".format(np.mean(f32), np.mean(bf16)))

print("diff ratio {0:f}".format(diff_ratio))
