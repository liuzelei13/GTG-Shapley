
import SV_algs
from SV_algs import MR,OR, ExactSV,TMC, GroupTest,Fed_SV,GTG,MR_TMC,TMR,OR_TMC,MR_TMCR


class FL_SV():
    def __init__(self):
        pass

    def init_SV_server(self):
        if self.args.SV_alg == 'MR':
            return SV_algs.MR.MR()
        elif self.args.SV_alg == 'TMR':
            return SV_algs.TMR.TMR()
        elif self.args.SV_alg == 'MR_TMC':
            return SV_algs.MR_TMC.MR_TMC()
        elif self.args.SV_alg == 'Fed_SV':
            return SV_algs.Fed_SV.Fed_SV()
        elif self.args.SV_alg == 'GTG':
            return SV_algs.GTG.GTG()
        elif self.args.SV_alg == 'MR_TMCR':
            return SV_algs.MR_TMCR.MR_TMCR()
        elif self.args.SV_alg == 'OR':
            return SV_algs.OR.OR()
        elif self.args.SV_alg == 'OR_TMC':
            return SV_algs.OR_TMC.OR_TMC()
        elif self.args.SV_alg == 'OR_TMCR':
            return outdated.OR_TMCR.OR_TMCR()
        elif self.args.SV_alg == 'ExactSV':
            return SV_algs.ExactSV.ExactSV()
        elif self.args.SV_alg == 'GroupTest':
            return SV_algs.GroupTest.GroupTest()
        elif self.args.SV_alg == 'TMC':
            return SV_algs.TMC.TMC()
        elif self.args.SV_alg == '':
            return SV_algs
        pass