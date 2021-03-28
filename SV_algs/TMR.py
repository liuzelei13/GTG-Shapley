from typing import Callable,Any
import SV_algs.shapley_utils
from SV_algs.shapley_utils import powersettool
import copy
from scipy.special import comb
import pickle


class ShapleyValue:
    def __init__(self):
        self.FL_name='Null'
        self.SV={} #dict: {id:SV,...}


class TMR(ShapleyValue):
    def __init__(self):
        super().__init__()
        self.SV_t={} #round t: {id:SV,...}
        self.Ut={}

        #TMR paras
        self.round_trunc_threshold=0.01

    def compute_shapley_value(self,t,idxs,**kwargs):
        V_S_t=kwargs['V_func']

        util={}
        powerset=list(powersettool(idxs))

        # TMR round truncation below
        S_0=()
        util[S_0]=V_S_t(t=t,S=S_0)

        S_all=powerset[-1]
        util[S_all]=V_S_t(t=t,S=S_all)

        if abs(util[S_all]-util[S_0])<=self.round_trunc_threshold:
            sv_dict={id:0 for id in idxs}
            return sv_dict
        # TMR round truncation above


        for S in powerset[1:-1]:
            util[S]=V_S_t(t=t,S=S)

        self.SV_t[t]=self.shapley_value(util,idxs)

        self.Ut[t]=copy.deepcopy(util)

        return self.SV_t[t]

    def get_final_result(self):
        for t,shapley_t in self.SV_t.items():
            for id in shapley_t:
                if self.SV.get(id):
                    self.SV[id].append(shapley_t[id])
                else:
                    self.SV[id]=[shapley_t[id]]
        return self.SV


    def shapley_value(self,utility,idxs):
        N=len(idxs)
        sv_dict={id:0 for id in idxs}
        for S in utility.keys():
            if S !=():
                for id in S:
                    marginal_contribution=utility[S]-utility[tuple(i for i in S if i!=id)]
                    sv_dict[id] += marginal_contribution /((comb(N-1,len(S)-1))*N)
        return sv_dict

    def write_results(self,duration,args):
        result_file=open('results/{}_{}_{}_{}_{}.txt'.format(args.SV_alg,args.case,args.model,
                                                             args.num_users, args.traindivision), 'a')
        for id in self.SV:
            lines=['Participant id: '+str(id),'\n',
                   'Shapley Value: '+ str(self.SV[id]),'\n','\n']
            result_file.writelines(lines)
        lines= ['Total Run Time: {0:0.4f}'.format(duration),'\n']
        result_file.writelines(lines)
        pass

    def write_duration_details(self,time_train,n_train,time_assembel,n_assemble,time_eval,n_eval,args):
        result_file=open('results/{}_{}_{}_{}_{}.txt'.format(args.SV_alg,args.case,args.model,
                                                             args.num_users, args.traindivision), 'a')
        lines=['Duration train = %.4f'%(time_train),'\n',
               'Total number of per clients train = %d'%(n_train),'\n',
               'Duration Assemble = %.4f'%(time_assembel),'\n',
               'Total number of per assemble = %d'%(n_assemble),'\n',
               'Duration evaluation = %.4f'%(time_eval),'\n',
               'Total number of per clients evaluation = %d'%(n_eval),'\n']
        result_file.writelines(lines)
        pass
























