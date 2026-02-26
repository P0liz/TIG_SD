from timer import Timer
from datetime import datetime
from os.path import exists, join
from os import makedirs
from config import BUCKET_CONFIG


class Folder:
    run_id = None
    DST = None
    DST_ARC = None
    DST_IND = None
    method = None

    @classmethod
    def initialize(cls, custom_run_id=None, method=None):
        Timer.start = datetime.now()
        cls.run_id = custom_run_id or str(Timer.start.strftime("%s"))
        cls.method = method
        if method == "bucket":
            cls.method = f"bucket_{BUCKET_CONFIG}"
        elif method == None:
            cls.method = "none"

        # Create structure: runs/{method}/run_{id}/
        cls.DST = join("runs", f"archive_{cls.method}", f"run_{cls.run_id}")
        if not exists(cls.DST):
            makedirs(cls.DST)

        cls.DST_ARC = join(cls.DST, "archive")
        cls.DST_IND = join(cls.DST, "inds")
