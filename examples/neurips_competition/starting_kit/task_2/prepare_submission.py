import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_files", "-lf", type=str, nargs="+", help="the file path list for the cate estimation of each dataset")
    parser.add_argument("--output_dir", "-o", type=str, help="the output directory for CATE submission file")

    args= parser.parse_args()

    CATE_list = []
    for save_dir in args.save_dirs:
        cur_cate = np.load(save_dir)
        CATE_list.append(cur_cate)

    CATE_list = np.array(CATE_list)

    # Create the output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, "cate_estimate.npy"), CATE_list)
