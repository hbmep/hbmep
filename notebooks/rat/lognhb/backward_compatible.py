import os
import pickle

import hbmep as mep
from hbmep.util import site


def main(model_dir):
    build_dir = model_dir.replace("lognhb", "lognhb_backward")
    os.makedirs(build_dir, exist_ok=True)
    src = os.path.join(model_dir, "inference.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior, model_dict = pickle.load(f)

    for u, v in posterior.items():
        print(u, v.shape)
        
    mapping = {site.a: site.a, site.b: site.b, site.g: "L", site.h: "H", site.v: "ℓ", site.c1: site.c1, site.c2: site.c2}
    samples = {mapping[u]: v for u, v in posterior.items() if u in mapping.keys()}
    for u, v in samples.items():
        print(u, v.shape)

    output_path = os.path.join(build_dir, "inference.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, samples,), f)
    print(f"Saved to {output_path}")
    print("\n\n")
    return


if __name__ == "__main__":
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/lognhb/nhb__4000w_1000s_4c_1t_20d_95a_tm/lcirc/rectified_logistic",
        "/home/vishu/reports/hbmep/notebooks/rat/lognhb/nhb__4000w_1000s_4c_1t_20d_95a_tm/lcirc/logistic4",
        "/home/vishu/reports/hbmep/notebooks/rat/lognhb/nhb__4000w_1000s_4c_1t_20d_95a_tm/lshie/rectified_logistic",
        "/home/vishu/reports/hbmep/notebooks/rat/lognhb/nhb__4000w_1000s_4c_1t_20d_95a_tm/lshie/logistic4",
    ]
    [main(model_dir) for model_dir in model_dirs]
