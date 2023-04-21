from hb_mep.config import HBMepConfig as config


DATA_DIR = "data"
REPORTS_DIR = "reports"

PARTICIPANT = config.PARTICIPANT
INTENSITY = config.INTENSITY
FEATURES = config.FEATURES
RESPONSE = config.RESPONSE

AUC_MAP = \
        {
            "LTrapezius":'auc01', "LDeltoid":'auc02', "LBiceps":'auc03', "LTriceps":'auc04',
            "LECR":'auc05', "LFCR":'auc06', "LAPB":'auc07', "LADM":'auc08', "LTA":'auc09',
            "LEDB":'auc10', "LAH":'auc11', "RTrapezius":'auc12', "RDeltoid":'auc13', "RBiceps":'auc14',
            "RTriceps":'auc15', "RECR":'auc16', "RFCR":'auc17', "RAPB":'auc18', "RADM":'auc19',
            "RTA":'auc20', "REDB":'auc21', "RAH":'auc22'
        }
