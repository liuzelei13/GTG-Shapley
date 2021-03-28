
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle,random,copy
import numpy as np
import time
from Apps.OurMNIST_funcs.models import MLP
import SV_algs
from Apps.General_FL import FL_SV
from Apps.OurMNIST_Client import Client
from Apps.OurMNIST_funcs.utils import get_dataset,calculate_gradients,\
    update_weights_from_gradients,calculate_accumulate_gradients, exp_details



class OurMNIST(FL_SV):
    def __init__(self,args):
        self.args=args

        exp_details(self.args)

        self.data_dir='../data/OurMNIST/'
        self.test_dataset_dir='../data/mnist/'
        self.N=self.args.num_users

        # init FL server
        if args.gpu:
            torch.cuda.set_device(args.gpu)
        self.device = 'cuda' if args.gpu else 'cpu'

        # load clients/dataset
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)

        # init model
        self.global_model=self.init_model()
        self.model_0 = copy.deepcopy(self.global_model) #for OR-like algs
        self.null_M_acc={} #for V(M(S=() ))

        # init Shapley Value server - choose a baseline
        self.SV_server = self.init_SV_server()

        # simulation options
        self.N_all = [i for i in range(1,self.args.num_users+1)]

        # init clients
        self.clients={}
        for id in self.N_all:
            self.clients[id]=Client(args=self.args,dataset=self.train_dataset[id],
                                    idxs=self.user_groups[id],client_id=id)
        self.total_data=sum(len(self.clients[i].idxs) for i in range(1,self.N+1))
        self.fraction = {i:len(self.clients[i].idxs)/self.total_data for i in range(1,self.N+1)}

        # for shapley calc: gradient add up
        self.client_mid_weights={} #dict: [epoch][id]
        self.client_sum_updates={} #dict: [id]

        #timer for simulation, real record
        self.duration_train=0
        self.duration_assemble=0
        self.duration_eval=0
        self.n_train=0
        self.n_assemble=0
        self.n_eval=0

        #for simulation train and V()
        self.simu_U={}
        self.simu_eval_dur_one=1.023
        self.simu_asse_dur_one=1.8e-4
        self.simu_train_dur_one=0.136

        #for OG-SV train timer
        self.counter_og_sv=0
        self.n_train_last=0
        self.global_st=0


        pass


    def init_model(self):
        if self.args.model == 'mlp':
            img_size = self.train_dataset[1][0][0].shape

            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=self.args.num_classes)

        # Set the model to train and send it to device.
        global_model.to(self.device)
        global_model.train()
        return global_model

    def train_FL(self):
        # different SV alg requirments

        # 1. gradient + MR type
        if self.args.gradient and not self.args.aggregate:
            #timer
            st_mr=time.time()
            #timer
            for epoch in range(self.args.epochs):
                #timer
                st_train_time=time.time()
                #timer

                self.store_null_M_perf(epoch)
                N_list=self.train_one_round_with_gradient_saved(epoch=epoch)

                #timer
                self.duration_train+=(time.time()-st_train_time)
                self.n_train+=len(N_list)
                #timer

                shapley_value_t = self.SV_server.compute_shapley_value(t=epoch,idxs=N_list,V_func=self.V_S_t)
            shapley_value = self.SV_server.get_final_result()

            #timer
            duration=time.time()-st_mr
            #timer
            self.SV_server.write_results(duration,self.args)

            # simu
            if self.args.SV_alg=='MR':
                self.SV_server.store_temp_for_simulation(self.args)
                #timer
                self.SV_server.write_duration_details(self.duration_train,self.n_train,
                                                      self.duration_assemble,self.n_assemble,
                                                      self.duration_eval,self.n_eval,
                                                      self.args)
                #timer

        # 2. gradient + OR type
        elif self.args.gradient and self.args.aggregate:
            #timer
            st_or=time.time()
            #timer

            self.store_null_M_perf(0)
            for epoch in range(self.args.epochs):
                #timer
                st_train_time=time.time()
                #timer

                N_list=self.train_one_round_with_update_saved(epoch=epoch)

                #timer
                self.duration_train+=(time.time()-st_train_time)
                self.n_train+=len(N_list)
                #timer

            shapley_value = self.SV_server.compute_shapley_value(idxs=N_list, V_func=self.V_S_0)

            #timer
            duration=time.time()-st_or
            #timer
            self.SV_server.write_results(duration,self.args)

            # simu
            if self.args.SV_alg=='OR':
                self.SV_server.store_temp_for_simulation(self.args)
                #timer
                self.SV_server.write_duration_details(self.duration_train,self.n_train,
                                                      self.duration_assemble,self.n_assemble,
                                                      self.duration_eval,self.n_eval,
                                                      self.args)
                #timer

        # 3. non-gradient algs, OG-Shapley
        elif not self.args.gradient:
            # V(M(D_S))
            shapley_value = self.SV_server.compute_shapley_value(idxs=self.N_all,V_func=self.V_S_D)
            #simu
            if self.args.SV_alg=='ExactSV':
                self.SV_server.store_temp_for_simulation(self.args)
            pass
        return shapley_value


    def store_null_M_perf(self,t):
        test_acc=self.V_M_S(self.global_model)
        self.null_M_acc[t]=test_acc
        pass

    def train_one_round_with_gradient_saved(self,epoch):
        S=[]
        local_weights, local_losses ={},{}
        self.global_model.train()
        for id,client in self.clients.items():
            w,loss = client.update_weights(model=copy.deepcopy(self.global_model),global_round=epoch)
            local_weights[id]=copy.deepcopy(w)
            local_losses[id]=copy.deepcopy(loss)
            #for SV
            S.append(id)
        global_weights=self.fedavg(local_weights,{i:self.fraction[i] for i in S})
        self.global_model.load_state_dict(global_weights)

        #for SV
        self.client_mid_weights[epoch]=local_weights
        return S

    def train_one_round_with_update_saved(self,epoch):
        S=[]
        local_weights, local_losses ={},{}
        self.global_model.train()
        for id,client in self.clients.items():
            w,loss = client.update_weights(model=copy.deepcopy(self.global_model),global_round=epoch)
            local_weights[id]=copy.deepcopy(w)
            local_losses[id]=copy.deepcopy(loss)
            #for SV
            S.append(id)

        #for SV
        if epoch==0:
            M0 = copy.deepcopy(self.global_model.state_dict())
            self.client_sum_updates={id:calculate_gradients(M0,M0) for id in S}
        for id in S:
            self.client_sum_updates[id]=calculate_accumulate_gradients \
                (self.client_sum_updates[id],local_weights[id],self.global_model.state_dict())

        global_weights=self.fedavg(local_weights,{i:self.fraction[i] for i in S})
        self.global_model.load_state_dict(global_weights)

        return S

    def fedavg(self, w: dict, fraction: dict):
        counter=0
        for id in w.keys():
            counter+=1
            if counter==1:
                w_avg=copy.deepcopy(w[id])
                for key in w_avg.keys():
                    w_avg[key] *= (fraction[id]/sum(fraction.values()))
            else:
                for key in w_avg.keys():
                    w_avg[key] += w[id][key]*(fraction[id]/sum(fraction.values()))
        return w_avg

    def V_S_t(self,t,**kwargs):
        #timer
        st_ass=time.time()
        #timer

        S=kwargs['S']
        model_S=copy.deepcopy(self.global_model)
        model_S.train()

        if S==():
            test_acc=self.null_M_acc[t]
            return test_acc

        local_weights={id:self.client_mid_weights[t][id] for id in S}
        global_weights=self.fedavg(local_weights,{i:self.fraction[i] for i in S})
        model_S.load_state_dict(global_weights)

        #timer
        self.duration_assemble+=(time.time()-st_ass)
        self.n_assemble+=len(S)
        #timer

        #timer
        st_eval=time.time()
        #timer

        test_acc=self.V_M_S(model_S)

        #timer
        self.duration_eval+=(time.time()-st_eval)
        self.n_eval+=1
        #timer

        return test_acc

    def V_S_0(self,**kwargs):
        #timer
        st_ass=time.time()
        #timer

        S=kwargs['S']
        model_S=copy.deepcopy(self.model_0)
        model_S.train()
        if S==():
            test_acc=self.null_M_acc[0]
            return test_acc

        local_weights={id:update_weights_from_gradients(self.client_sum_updates[id],copy.deepcopy(model_S.state_dict()))
                       for id in S}
        global_weights=self.fedavg(local_weights,{i:self.fraction[i] for i in S})
        model_S.load_state_dict(global_weights)

        #timer
        self.duration_assemble+=(time.time()-st_ass)
        self.n_assemble+=len(S)
        #timer

        #timer
        st_eval=time.time()
        #timer

        test_acc=self.V_M_S(model_S)

        #timer
        self.duration_eval+=(time.time()-st_eval)
        self.n_eval+=1
        #timer

        return test_acc

    def V_S_D(self,**kwargs):
        S=kwargs['S']

        #timer
        if self.counter_og_sv%10==0:
            self.global_st=time.time()
        self.counter_og_sv+=1
        self.n_train+=len(S)*5
        #timer

        model_S=self.M_S_D(S)

        #timer
        if self.counter_og_sv%10==0:
            dura=time.time()-self.global_st
            n_tr=self.n_train-self.n_train_last
            self.n_train_last=self.n_train
            print("10 subset train time = %.4f ; Average time for one = %.4f"%(dura, dura/n_tr))

        test_acc = self.V_M_S(model_S)
        return test_acc

    def M_S_D(self,S):
        #1. init M_S
        model_S=copy.deepcopy(self.model_0)

        if S==():
            return model_S

        for epoch in range(self.args.epochs):
            local_weights, local_losses ={},{}
            model_S.train()
            for idx in S:
                w,loss = self.clients[idx].update_weights(model=copy.deepcopy(model_S),global_round=epoch)
                local_weights[idx]=copy.deepcopy(w)
                local_losses[idx]=copy.deepcopy(loss)

            global_weights=self.fedavg(local_weights,{i:self.fraction[i] for i in S})
            model_S.load_state_dict(global_weights)
        return model_S

    def V_M_S(self,model):
        """ Returns the test accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if self.args.gpu else 'cpu'
        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy

    def train_FL_simu(self):
        # different SV alg requirments

        # 1. gradient + MR type
        if self.args.gradient and not self.args.aggregate:
            #load Ut from pickle
            f_Ut_dict=open('Simulation_temp/MR_Ut_t_case_{}.pkl'.format(self.args.traindivision),'rb')
            self.simu_U=pickle.load(f_Ut_dict)

            for epoch in range(self.args.epochs):
                N_list=copy.deepcopy(self.N_all)

                #timer
                self.n_train+=len(N_list)
                self.duration_train+= (len(N_list)*self.simu_train_dur_one)
                #timer

                shapley_value_t = self.SV_server.compute_shapley_value(t=epoch,idxs=N_list,V_func=self.V_S_t_simu)
            shapley_value = self.SV_server.get_final_result()

            #simu
            duration = self.duration_train +self.duration_assemble+self.duration_eval
            #simu

            self.SV_server.write_results(duration,self.args)

            # simu
            if self.args.SV_alg=='MR':
                self.SV_server.store_temp_for_simulation(self.args)
                #timer
            self.SV_server.write_duration_details(self.duration_train,self.n_train,
                                                  self.duration_assemble,self.n_assemble,
                                                  self.duration_eval,self.n_eval,
                                                  self.args)
                #timer

        # 2. gradient + OR type
        elif self.args.gradient and self.args.aggregate:
            #load Ut from pickle
            f_Ut_dict=open('Simulation_temp/OR_Ut_case_{}.pkl'.format(self.args.traindivision),'rb')
            self.simu_U=pickle.load(f_Ut_dict)

            for epoch in range(self.args.epochs):
                N_list=copy.deepcopy(self.N_all)

                #timer
                self.n_train+=len(N_list)
                self.duration_train+= (len(N_list)*self.simu_train_dur_one)
                #timer

            shapley_value = self.SV_server.compute_shapley_value(idxs=N_list, V_func=self.V_S_0_simu)

            #simu
            duration = self.duration_train +self.duration_assemble+self.duration_eval
            #simu

            self.SV_server.write_results(duration,self.args)

            # simu
            if self.args.SV_alg=='OR':
                self.SV_server.store_temp_for_simulation(self.args)
            #timer
            self.SV_server.write_duration_details(self.duration_train,self.n_train,
                                                  self.duration_assemble,self.n_assemble,
                                                  self.duration_eval,self.n_eval,
                                                  self.args)
            #timer

        # 3. non-gradient algs, OG-Shapley
        elif not self.args.gradient:
            #load Ut from pickle
            f_Ut_dict=open('Simulation_temp/og_ut_dict.pkl','rb')
            simu_U_all=pickle.load(f_Ut_dict)
            self.simu_U=simu_U_all[self.args.traindivision]

            # V(M(D_S))
            shapley_value = self.SV_server.compute_shapley_value(idxs=self.N_all,V_func=self.V_S_D_simu)

            #simu already found past one
            # if self.args.SV_alg=='ExactSV':
            #     self.SV_server.store_temp_for_simulation(self.args)
            # pass

            #simu
            duration = self.duration_train +self.duration_assemble+self.duration_eval
            #simu

            self.SV_server.write_results(duration,self.args)

            #timer
            self.SV_server.write_duration_details(self.duration_train,self.n_train,
                                                  self.duration_assemble,self.n_assemble,
                                                  self.duration_eval,self.n_eval,
                                                  self.args)
            #timer
        return shapley_value

    def V_S_t_simu(self,t,**kwargs):
        S=kwargs['S']

        #simu counter
        self.n_assemble+=len(S)
        self.duration_assemble+=(len(S)*self.simu_asse_dur_one)
        #simu

        #test_acc=self.V_M_S(model_S)
        test_acc=self.simu_U[t][S]


        #timer
        self.duration_eval+=self.simu_eval_dur_one
        self.n_eval+=1
        #timer

        return test_acc

    def V_S_0_simu(self,**kwargs):
        S=kwargs['S']

        #simu counter
        self.n_assemble+=len(S)
        self.duration_assemble+=(len(S)*self.simu_asse_dur_one)
        #simu

        #test_acc=self.V_M_S(model_S)
        test_acc=self.simu_U[S]


        #timer
        self.duration_eval+=self.simu_eval_dur_one
        self.n_eval+=1
        #timer

        return test_acc

    def V_S_D_simu(self,**kwargs):
        S=kwargs['S']

        #simu counter
        self.n_train+=len(S)*5
        self.duration_train+=(len(S)*5*self.simu_train_dur_one)
        #simu

        test_acc=self.simu_U[S]

        #timer
        self.duration_eval+=self.simu_eval_dur_one
        self.n_eval+=1
        #timer


        return test_acc


