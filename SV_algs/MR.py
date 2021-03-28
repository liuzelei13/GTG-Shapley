from typing import Callable,Any
import SV_algs.shapley_utils
from SV_algs.shapley_utils import powersettool
import copy,time
from scipy.special import comb
import pickle


class ShapleyValue:
    def __init__(self):
        self.FL_name='Null'
        self.SV={} #dict: {id:SV,...}


class MR(ShapleyValue):
    def __init__(self):
        super().__init__()
        self.SV_t={} #round t: {id:SV,...}
        self.Ut={}

        #for print only
        self.full_set=()

        #for timer only
        self.st_t=0

    def compute_shapley_value(self,t,idxs,**kwargs):
        #timer only
        self.st_t=time.time()

        V_S_t=kwargs['V_func']

        util={}
        powerset=list(powersettool(idxs))
        for S in powerset:
            util[S]=V_S_t(t=t,S=S)

        #for print only
        self.full_set=powerset[-1]

        self.SV_t[t]=self.shapley_value(util,idxs)

        self.Ut[t]=copy.deepcopy(util)

        self.print_results(t)

        return self.SV_t[t]

    def print_results(self,t):
        print('|---- Results after global round %d :'%(t))
        print("Test Accuracy: %.2f %%"%(100*self.Ut[t][self.full_set]))
        print('total time = %.2f '%( (time.time()-self.st_t) ))
        print('Current round shapley = ')
        print(self.SV_t[t])
        pass


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

    def store_temp_for_simulation(self,args):
        with open('Simulation_temp/MR_Ut_t_case_{}.pkl'.format(args.traindivision),'wb') as fp:
            pickle.dump(self.Ut, fp)
        pass


    def shapley_value(self,utility,idxs):
        N=len(idxs)
        sv_dict={id:0 for id in idxs}
        for S in utility.keys():
            if S !=():
                for id in S:
                    marginal_contribution=utility[S]-utility[tuple(i for i in S if i!=id)]
                    sv_dict[id] += marginal_contribution /((comb(N-1,len(S)-1))*N)
        return sv_dict

























