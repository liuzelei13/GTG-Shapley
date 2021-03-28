#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import pickle
import numpy as np
import random # used to shuffle lists of data from dictionary, since our dictionary is too organised (1 then 2 then 3...)
from itertools import chain, combinations
from scipy.special import comb
# from update import test_inference # used in TMC to get the score of the model

random.seed(0) # used a random seed for reproducibility


def powersettool(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(1,N+1):
        shapley_dict[i] = 0
    for key in utility:
        if key != ():
            for contributor in key:
                # print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i!=contributor)]
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution /((comb(N-1,len(key)-1))*N)
    return shapley_dict

def TMC_oneiteration(test_acc, idxs_users, submodel_dict, accuracy_dict, fraction, args, test_dataset, tolerance=0.005, num_tolerance=2,\
                    random_score=0):
    """
    Runs one iteration of OR-TMC-Shapley algorithm
    :param tolerance: percentage difference from the test_acc of the global model
    :param num_tolerance: number of time the performance tolerance has to be met to truncate the loop
    :param test_acc: test accuracy of the global model on the official MNIST testset
    :param submodel_dict: dictionary containing the submodels (only those based on single participant's data)
    :param idxs_users: the array containing the indexes of participants
    :param fraction: list of fraction of data from each user (all users)
    :param args: used in test_inference function 
    :param test_dataset: the test dataset used to evaluate the submodels
    :param accuracy_dict: the dictionary to store the evaluation of the submodels
    :param random_score: the accuracy of the randomly initialized model 

    :returns marginal_contribs: a 1D array storing the marginal contributions of participants
    :returns
    """
    # form a random permutation from the array of indexes of participants
    idxs = np.random.permutation(idxs_users)
    # print("idxs is", idxs) # print check 
    # initialize the marginal contributions of all participants to be 0
    marginal_contribs = np.zeros(len(idxs_users))

    # the truncation counter is incremented when the performance tolerance is met
    # when truncation counter is 2, truncate the algorithm
    truncation_counter = 0  

    # initialize the score to be the random score before the data comes in
    new_score = random_score

    # list to keep track of the data fraction. the first element is the fraction so far, the second element is
    # the fraction of the current user with idx
    fraction_tmc = [0, 0]

    # performance tolerance
    # print("performance tolerance is", tolerance * test_acc) # print check

    for n, idx in enumerate(idxs):
        # print("idx is", idx) # print check
        # add the fraction of data from user with index idx as the second element of fraction_tmc
        fraction_tmc[1] = fraction[idx-1]
        # print("fraction_tmc is", fraction_tmc) # print check 
        old_score = new_score

        # if n == 0:
        #     # initialize the model to the the one from by the first participant in the permutation
        #     model = copy.deepcopy(submodel_dict[(idx,)])
        # else:
        #     # calculate the average of the subset of weights from list of all the weights
        #     subset = idxs[:n+1]
        #     subset = tuple(np.sort(subset, kind='mergesort')) # sort the subset and change it to a tuple
        #     print("subset is", subset) # print check 
            
        #     subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
        #     # subset_weights = average_weights([model.state_dict(), submodel_dict[(idx,)].state_dict()], fraction_tmc) 
        #     # form the model up till that point
        #     model.load_state_dict(subset_weights)
        #     # store it in the originaly submodel_dict for easy reference
        #     submodel_dict[subset].load_state_dict(subset_weights)
        
    
        # calculate the average of the subset of weights from list of all the weights
        subset = idxs[:n+1]
        subset = tuple(np.sort(subset, kind='mergesort')) # sort the subset and change it to a tuple
        # print("subset is", subset) # print check 
        model = copy.deepcopy(submodel_dict[()])
        if accuracy_dict.get(subset) == None:
            if submodel_dict.get(subset) == None:
                subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
                # subset_weights = average_weights([model.state_dict(), submodel_dict[(idx,)].state_dict()], fraction_tmc) 
                # form the model up till that point
                model.load_state_dict(subset_weights)
                # store it in the originaly submodel_dict for easy reference
                submodel_dict[subset] = copy.deepcopy(model)
            else: 
                model = copy.deepcopy(submodel_dict[subset])
            
            # get the new score
            new_score, _ = test_inference(args, model, test_dataset)
            accuracy_dict[subset] = new_score
        else:
            new_score = accuracy_dict[subset]

        # print("new_score is", new_score) # print check 
        marginal_contribs[idx-1] = new_score - old_score
        # print("marginals:", marginal_contribs) # print check 
        # find the distance to full score (the test_acc of the global model on all the 5 datapoints)
        distance_to_full_score = np.abs(new_score - test_acc) 
        # print("distance is", distance_to_full_score) # print check
        if distance_to_full_score <= tolerance * test_acc:
            truncation_counter += 1
            # print("truncation_counter", truncation_counter)
            #truncate when the distance_to_full_score becomes very close for 2 times
            if truncation_counter >= num_tolerance:
                break
        else:
            truncation_counter = 0

        fraction_tmc[0] += fraction_tmc[1] # sum the fractions onto the first element in the list

    # print("number of times in the loop for this TMC iteration is (n+1) = ", n+1) # print check
    return marginal_contribs
def AdjustedTMC_oneiteration(test_acc, idxs_users, submodel_dict, accuracy_dict, fraction, args, test_dataset, tolerance=0.005, num_tolerance=2,\
                    random_score=0):
    """
    Runs one iteration of Adjusted OR-TMC-Shapley algorithm
    :param tolerance: percentage difference from the test_acc of the global model
    :param num_tolerance: number of time the performance tolerance has to be met to truncate the loop
    :param test_acc: test accuracy of the global model on the official MNIST testset
    :param submodel_dict: dictionary containing the submodels (only those based on single participant's data)
    :param idxs_users: the array containing the indexes of participants
    :param fraction: list of fraction of data from each user (all users)
    :param args: used in test_inference function 
    :param test_dataset: the test dataset used to evaluate the submodels
    :param accuracy_dict: the dictionary to store the evaluation of the submodels
    :param random_score: the accuracy of the randomly initialized model 

    :returns marginal_contribs: a 1D array storing the marginal contributions of participants
    :returns
    """
    average_marginal_contribs = np.zeros(len(idxs_users))
    for i in idxs_users:   
        # form a random permutation from the array of indexes of participants
        idxs = np.concatenate((np.array([i]), np.random.permutation([x for x in idxs_users if x != i])))
        # print("idxs is", idxs) # print check 

        # initialize the marginal contributions of all participants to be 0
        marginal_contribs = np.zeros(len(idxs_users))

        # the truncation counter is incremented when the performance tolerance is met
        # when truncation counter is 2, truncate the algorithm
        truncation_counter = 0  

        # initialize the score to be the random score before the data comes in
        new_score = random_score

        # list to keep track of the data fraction. the first element is the fraction so far, the second element is
        # the fraction of the current user with idx
        fraction_tmc = [0, 0]

        # performance tolerance
        # print("performance tolerance is", tolerance * test_acc) # print check

        for n, idx in enumerate(idxs):
            # print("idx is", idx) # print check
            # add the fraction of data from user with index idx as the second element of fraction_tmc
            fraction_tmc[1] = fraction[idx-1]
            # print("fraction_tmc is", fraction_tmc) # print check 
            old_score = new_score

            # if n == 0:
            #     # initialize the model to the the one from by the first participant in the permutation
            #     model = copy.deepcopy(submodel_dict[(idx,)])
            # else:
            #     # calculate the average of the subset of weights from list of all the weights
            #     subset = idxs[:n+1]
            #     subset = tuple(np.sort(subset, kind='mergesort')) # sort the subset and change it to a tuple
            #     print("subset is", subset) # print check 
                
            #     subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
            #     # subset_weights = average_weights([model.state_dict(), submodel_dict[(idx,)].state_dict()], fraction_tmc) 
            #     # form the model up till that point
            #     model.load_state_dict(subset_weights)
            #     # store it in the originaly submodel_dict for easy reference
            #     submodel_dict[subset].load_state_dict(subset_weights)
            
        
            # calculate the average of the subset of weights from list of all the weights
            subset = idxs[:n+1] #deepcopy
            subset = tuple(np.sort(subset, kind='mergesort')) # sort the subset and change it to a tuple
            # print("subset is", subset) # print check 
            model = copy.deepcopy(submodel_dict[()])
            if accuracy_dict.get(subset) == None:
                if submodel_dict.get(subset) == None:
                    subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
                    # subset_weights = average_weights([model.state_dict(), submodel_dict[(idx,)].state_dict()], fraction_tmc) 
                    # form the model up till that point
                    model.load_state_dict(subset_weights)
                    # store it in the originaly submodel_dict for easy reference
                    submodel_dict[subset] = copy.deepcopy(model)
                else: 
                    model = copy.deepcopy(submodel_dict[subset])
                
                # get the new score
                new_score, _ = test_inference(args, model, test_dataset)
                accuracy_dict[subset] = new_score
            else:
                new_score = accuracy_dict[subset]

            # print("new_score is", new_score) # print check 
            marginal_contribs[idx-1] = new_score - old_score
            # print("marginals:", marginal_contribs) # print check 
            # find the distance to full score (the test_acc of the global model on all the 5 datapoints)
            distance_to_full_score = np.abs(new_score - test_acc) 
            # print("distance is", distance_to_full_score) # print check
            if distance_to_full_score <= tolerance * test_acc:
                truncation_counter += 1
                # print("truncation_counter", truncation_counter)
                #truncate when the distance_to_full_score becomes very close for 2 times
                if truncation_counter >= num_tolerance:
                    break
            else:
                truncation_counter = 0

            fraction_tmc[0] += fraction_tmc[1] # sum the fractions onto the first element in the list
        average_marginal_contribs += marginal_contribs

    average_marginal_contribs /= args.num_users # find the average of the num_users number of permutations
    # print("number of times in the loop for this TMC iteration is (n+1) = ", n+1) # print check
    return average_marginal_contribs

def MC_oneiteration(test_acc, idxs_users, submodel_dict, accuracy_dict, fraction, args, test_dataset, tolerance=0.005, num_tolerance=2, \
                     random_score=0):
    """
    Runs one iteration of Random-Monte-Carlo-Shapley algorithm
    :param tolerance: percentage difference from the test_acc of the global model
    :param num_tolerance: number of time the performance tolerance has to be met to truncate the loop
    :param test_acc: test accuracy of the global model on the official MNIST testset
    :param submodel_dict: dictionary containing the submodels (only those based on single participant's data)
    :param idxs_users: the array containing the indexes of participants
    :param fraction: list of fraction of data from each user (all users)
    :param args: used in test_inference function
    :param test_dataset: the test dataset used to evaluate the submodels
    :param accuracy_dict: the dictionary to store the evaluation of the submodels
    :param random_score: the accuracy of the randomly initialized model

    :returns marginal_contribs: a 1D array storing the marginal contributions of participants
    :returns
    """
    # form a random permutation from the array of indexes of participants
    idxs = np.random.permutation(idxs_users)
    # print("idxs is", idxs) # print check
    # initialize the marginal contributions of all participants to be 0
    marginal_contribs = np.zeros(len(idxs_users))

    # the truncation counter is incremented when the performance tolerance is met
    # when truncation counter is 2, truncate the algorithm
    truncation_counter = 0

    # initialize the score to be the random score before the data comes in
    new_score = random_score

    # list to keep track of the data fraction. the first element is the fraction so far, the second element is
    # the fraction of the current user with idx
    fraction_tmc = [0, 0]

    # performance tolerance
    # print("performance tolerance is", tolerance * test_acc) # print check

    for n, idx in enumerate(idxs):
        # print("idx is", idx) # print check
        # add the fraction of data from user with index idx as the second element of fraction_tmc
        fraction_tmc[1] = fraction[idx-1]
        # print("fraction_tmc is", fraction_tmc) # print check
        old_score = new_score


        subset = idxs[:n+1]
        subset = tuple(np.sort(subset, kind='mergesort')) # sort the subset and change it to a tuple
        # print("subset is", subset) # print check
        model = copy.deepcopy(submodel_dict[()])
        if accuracy_dict.get(subset) == None:
            if submodel_dict.get(subset) == None:
                subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset])
                # subset_weights = average_weights([model.state_dict(), submodel_dict[(idx,)].state_dict()], fraction_tmc)
                # form the model up till that point
                model.load_state_dict(subset_weights)
                # store it in the originaly submodel_dict for easy reference
                submodel_dict[subset] = copy.deepcopy(model)
            else:
                model = copy.deepcopy(submodel_dict[subset])

            # get the new score
            new_score, _ = test_inference(args, model, test_dataset)
            accuracy_dict[subset] = new_score
        else:
            new_score = accuracy_dict[subset]

        # print("new_score is", new_score) # print check
        marginal_contribs[idx-1] = new_score - old_score
        # print("marginals:", marginal_contribs) # print check
        # find the distance to full score (the test_acc of the global model on all the 5 datapoints)
        distance_to_full_score = np.abs(new_score - test_acc)
        # print("distance is", distance_to_full_score) # print check
        # if distance_to_full_score <= tolerance * test_acc:
        #     truncation_counter += 1
        #     # print("truncation_counter", truncation_counter)
        #     #truncate when the distance_to_full_score becomes very close for 2 times
        #     if truncation_counter >= num_tolerance:
        #         break
        # else:
        #     truncation_counter = 0

        fraction_tmc[0] += fraction_tmc[1] # sum the fractions onto the first element in the list

    # print("number of times in the loop for this TMC iteration is (n+1) = ", n+1) # print check
    return marginal_contribs

def error(mem, n=2):
    """
    :param mem: 2D array where each row is a set of marginal TMC contributions, and the columns correspond 
    to the participants
    :returns the maximum error (value between 0 and 1). because TMC score is an average, the mem is 1)
    applied np.cumsum and so on so that each row of values become an average of the rows up to that row. 2)
    the error is then calculated as the percentage difference from each row in the last n rows compared 
    with the last row. 3) the errors in each row is then averaged, so we have n averages from n rows. the 
    maximum average will be returned
    """
    if len(mem) < n:
        return 1.0
    all_vals = (np.cumsum(mem, 0)/np.reshape(np.arange(1, len(mem)+1), (-1,1)))[-n:]
    errors = np.mean(np.abs(all_vals[-n:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Dictionary Number   : {args.traindivision}\n')
    print(f'    Global Rounds   : {args.epochs}\n')
    
    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    # print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
