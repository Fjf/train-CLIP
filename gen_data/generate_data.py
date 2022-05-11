import os
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import math

def generate_data(N, txt):
    # noise images
    image_data = np.random.randint(0,255, (N, 224, 224, 3))
    txts = np.random.choice(np.array(txt), size=N)
    return image_data, txts


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--txt", type=str, default="I don't want to be human. I want see xrays.")
    parser.add_argument("--save_destination", type=str, default="./CLIP/")

    args = parser.parse_args()

    with open(args.txt, "r") as f:
        txts_list = [line.strip() for line in f.readlines() if line.strip() != ""]

    imgs, txts = generate_data(args.N, txts_list)

    os.makedirs(args.save_destination, exist_ok=True)
    os.makedirs(f"{args.save_destination}/imgs", exist_ok=True)
    os.makedirs(f"{args.save_destination}/txts", exist_ok=True)

    n_zeros = int(math.log10(args.N))
    print(n_zeros)
    for i in range(imgs.shape[0]):
        img = Image.fromarray(imgs[i].astype(np.uint8))
        img.save(os.path.join(args.save_destination, "imgs", str(i).zfill(n_zeros) + ".png"))
        with open(os.path.join(args.save_destination, "txts", str(i).zfill(n_zeros) + ".txt"), "w") as f:
            f.write(txts[i])
