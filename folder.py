from timer import Timer
from datetime import datetime
from os.path import exists, join
from os import makedirs


class Folder:
    run_id = None
    DST = None
    DST_ARC = None
    DST_IND = None

    @classmethod    # non un metodo d'istanza
    def initialize(cls, custom_run_id=None):
        Timer.start = datetime.now()
        cls.run_id = custom_run_id or str(Timer.start.strftime('%s'))
        cls.DST = "runs/run_" + cls.run_id
        if not exists(cls.DST):
            makedirs(cls.DST)
        cls.DST_ARC = join(cls.DST, "archive")
        cls.DST_IND = join(cls.DST, "inds")
