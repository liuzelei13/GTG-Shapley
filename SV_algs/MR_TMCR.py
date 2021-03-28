from typing import Callable,Any
import SV_algs.shapley_utils
from SV_algs.shapley_utils import powersettool
import copy
from scipy.special import comb
import numpy as np


class ShapleyValue:
    def __init__(self):
        self.FL_name='Null'
        self.SV={} #dict: {id:SV,...}



class MR_TMCR(ShapleyValue):
    def __init__(self):
        super().__init__()
        self.Ut={} #t: {}
        self.SV_t={} # t: {SV}

        #TMC paras
        self.Contribution_records =[]


        #trunc paras
        self.eps=0.001
        self.round_trunc_threshold=0.01


        #converge paras
        self.CONVERGE_MIN_K = 3*10
        self.last_k=10
        self.CONVERGE_CRITERIA = 0.05

    def compute_shapley_value(self,t,idxs,**kwargs):
        V_S_t=kwargs['V_func']
        N=len(idxs)
        self.Contribution_records=[]

        powerset=list(powersettool(idxs))

        util={}
        S_0=()
        util[S_0]=V_S_t(t=t,S=S_0)

        S_all=powerset[-1]
        util[S_all]=V_S_t(t=t,S=S_all)

        if abs(util[S_all]-util[S_0])<=self.round_trunc_threshold:
            sv_dict={id:0 for id in idxs}
            return sv_dict

        k=0
        while self.isnotconverge(k):
            k+=1
            v=[0 for i in range(N+1)]
            v[0]=util[S_0]
            marginal_contribution_k=[0 for i in range(N)]


            idxs_k = np.random.permutation(idxs)

            for j in range(1,N+1):
                # key = C subset
                C=idxs_k[:j]
                C=tuple(np.sort(C,kind='mergesort'))

                #truncation
                if abs(util[S_all] - v[j-1])>=self.eps:
                    if util.get(C)!=None:
                        v[j]=util[C]
                    else:
                        v[j]=V_S_t(t=t,S=C)
                else:
                    v[j]=v[j-1]

                # record calculated V(C)
                util[C] = v[j]

                # update SV
                marginal_contribution_k[idxs_k[j-1]-1] = v[j] - v[j-1]

            self.Contribution_records.append(marginal_contribution_k)

        # shapley value calculation
        shapley_value = (np.cumsum(self.Contribution_records, 0)/
                         np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-1:].tolist()[0]



        # store round t results
        self.SV_t[t]={key+1: sv for key,sv in enumerate(shapley_value)}

        self.Ut[t]=copy.deepcopy(util)

        return self.SV_t[t]



    def shapley_value(self,utility,idxs):
        N=len(idxs)
        sv_dict={id:0 for id in idxs}
        for S in utility.keys():
            if S !=():
                for id in S:
                    marginal_contribution=utility[S]-utility[tuple(i for i in S if i!=id)]
                    sv_dict[id] += marginal_contribution /((comb(N-1,len(S)-1))*N)
        return sv_dict

    def isnotconverge(self,k):
        if k <= self.CONVERGE_MIN_K:
            return True
        all_vals=(np.cumsum(self.Contribution_records, 0)/
                  np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-self.last_k:]
        #errors = np.mean(np.abs(all_vals[-last_K:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        errors = np.mean(np.abs(all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > self.CONVERGE_CRITERIA:
            return True
        return False

    def get_final_result(self):
        for t,shapley_t in self.SV_t.items():
            for id in shapley_t:
                if self.SV.get(id):
                    self.SV[id].append(shapley_t[id])
                else:
                    self.SV[id]=[shapley_t[id]]
        return self.SV

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
























