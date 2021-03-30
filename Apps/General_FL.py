
import SV_algs
from SV_algs import MR,OR, ExactSV,TMC, GroupTest,Fed_SV,GTG,TMR,GTG_OTi,GTG_Ti,GTG_Tib


class FL_SV():
    def __init__(self):
        pass

    def init_SV_server(self):
        if self.args.SV_alg == 'MR':
            return SV_algs.MR.MR()
        elif self.args.SV_alg == 'TMR':
            return SV_algs.TMR.TMR()
        elif self.args.SV_alg == 'GTG_Ti':
            return SV_algs.GTG_Ti.GTG_Ti()
        elif self.args.SV_alg == 'Fed_SV':
            return SV_algs.Fed_SV.Fed_SV()
        elif self.args.SV_alg == 'GTG':
            return SV_algs.GTG.GTG()
        elif self.args.SV_alg == 'GTG_Tib':
            return SV_algs.GTG_Tib.GTG_Tib()
        elif self.args.SV_alg == 'OR':
            return SV_algs.OR.OR()
        elif self.args.SV_alg == 'GTG_OTi':
            return SV_algs.GTG_OTi.GTG_OTi()
        elif self.args.SV_alg == 'ExactSV':
            return SV_algs.ExactSV.ExactSV()
        elif self.args.SV_alg == 'GroupTest':
            return SV_algs.GroupTest.GroupTest()
        elif self.args.SV_alg == 'TMC':
            return SV_algs.TMC.TMC()
        elif self.args.SV_alg == '':
            return SV_algs
        pass