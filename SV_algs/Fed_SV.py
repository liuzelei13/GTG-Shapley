from typing import Callable,Any
import SV_algs.shapley_utils
from SV_algs.shapley_utils import powersettool
import copy,time
from scipy.special import comb
import numpy as np
from wolframclient.evaluation import WolframLanguageSession,SecuredAuthenticationKey, WolframCloudSession
from wolframclient.language import wlexpr


class ShapleyValue:
    def __init__(self):
        self.FL_name='Null'
        self.SV={} #dict: {id:SV,...}



class Fed_SV(ShapleyValue):
    def __init__(self):
        super().__init__()
        self.Ut={}
        self.SV_t={}

        #TMC paras
        self.Contribution_records =[]

        #converge paras
        self.CONVERGE_MIN_K = 200
        self.last_k=10
        self.CONVERGE_CRITERIA = 0.05

    def compute_shapley_value(self,t,idxs,**kwargs):
        V_S_t=kwargs['V_func']
        N=len(idxs)
        powerset=list(powersettool(idxs))


        util={}
        S_0=()
        util[S_0]=V_S_t(t=t,S=S_0)

        S_all=powerset[-1]
        util[S_all]=V_S_t(t=t,S=S_all)


        # group test relate
        last_uds=[]
        Z=0
        for n in range(1,N):
            Z+=1/n
        Z*=2
        UD=np.zeros([N,N],dtype=np.float32)
        p=np.array([N/(i*(N-i)*Z) for i in range(1,N)])

        k=0
        while self.isnotconverge_Group(last_uds,UD) or k<self.CONVERGE_MIN_K:
            k+=1
            len_k=0
            # 1. draw len_K ~ q(len_k)
            len_k=np.random.choice(np.arange(1,N),p=p)

            # 2. sample S with len_k
            S=np.random.choice(idxs,size=len_k,replace=False)

            # 3. M(S) + V(S)
            S=tuple(np.sort(S,kind='mergesort'))
            if util.get(S)!=None:
                u_S=util[S]
            else:
                u_S=V_S_t(t=t,S=S)

            # 4. Group Testing update UD
            UD=(k-1)/k*UD

            for i in range(0,N):
                for j in range(0,N):
                    delta_beta=S.count(i+1)-S.count(j+1)
                    if delta_beta!=0:
                        value=delta_beta*u_S*Z/k
                        UD[i,j]+=value

            last_uds.append(UD)

        u_N=util[S_all]

        #timer
        st=time.time()
        #timer

        shapley_value=self.solveFeasible(N,u_N,UD)

        #timer
        dura=time.time()-st
        print('Solve Feasible using %.3f seconds'%dura)
        #timer

        self.Ut[t]=copy.deepcopy(util)
        self.SV_t[t]={key+1:sv for key,sv in enumerate(shapley_value)}


        return self.SV_t[t]


    def isnotconverge_Group(self,last_uds,UD):
        if len(last_uds)<=self.CONVERGE_MIN_K:
            return True
        for i in range(-self.last_k,0):
            ele=last_uds[i]
            delta=np.sum(np.abs(UD-ele),axis=(0,1))/len(UD[0])
            if delta > self.CONVERGE_CRITERIA:
                return True
        return False

    def solveFeasible(self,agentNum,u_N,UD):
        session=WolframLanguageSession()
        eps = 1/np.sqrt(agentNum)/agentNum/2.0
        # N[FindInstance[x^2 - 3 y^2 == 1 && 10 < x < 100, {x, y}, Integers]]
        ans = []
        result = []
        while len(result) == 0:
            expr = ""  # expr to evaluate
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "> 0.05 &&"
            expr = expr + "x" + str(agentNum-1) + "> 0.05 &&"
            for i in range(agentNum):
                for j in range(i+1, agentNum):
                    # abs(x_i - x_j) <= U_{i,j}
                    expr = expr + "Abs[x" + str(i) + "-x" + str(j) + "-(" + str(UD[i,j]) + ")]<=" + str(eps) + "&&"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "+"
            expr = expr + "x" + str(agentNum-1) + "==" + str(u_N) + "&&"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + "+"
            expr = expr + "x" + str(agentNum-1) + "<=" + str(u_N)

            expr = expr + ", {"
            for i in range(agentNum-1):
                expr = expr + "x" + str(i) + ","
            expr = expr + "x" + str(agentNum-1) + "}, Reals"

            expr = "N[FindInstance[" + expr + "]]"
            # print(expr)

            result = session.evaluate(wlexpr(expr))
            session.terminate()
            #  print(result)
            if len(result) > 0:
                ans = [result[0][i][1] for i in range(agentNum)]
            eps = eps * 1.1
            print(eps)
        # for i in range(agentNum):
        #     if ans[i] < 0.0000001:
        #         ans[i] = ans[i] + 0.0000001
        print(ans)
        return ans

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


























