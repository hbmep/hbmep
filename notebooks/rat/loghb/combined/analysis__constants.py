from hbmep import functional as F
from hbmep.util import site

SEPARATOR = "___"
MODEL_DIR = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked_hmaxPooled/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML"
NAMED_PARAMS = [site.a, site.b, site.g, site.h, site.v, "h_max"]
FUNC = F.rectified_logistic

BUILD_DIR = "/home/vishu/reports/hbmep/notebooks/rat/loghb/out/2d/"
NUM_POINTS = 200
# NUM_POINTS = 500
