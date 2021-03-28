import os

SV_alg=['MR','OR']



one_test_flag=False

if one_test_flag:
    alg='MR'
    os.system('python Testbed_Controller_v2.py --model=mlp --case=OurMNIST --num_users=5 --traindivision=2 --epochs=5 --local_ep=5 --SV_alg=%s'%(alg))
else:
    for alg in SV_alg:
        for i in range(1,6):
            os.system('python Testbed_Controller_v2.py --model=mlp --case=OurMNIST --num_users=10 --traindivision=%d --epochs=5 --local_ep=10 --SV_alg=%s'%(i,alg))
        pass
    pass

