import os
import matplotlib as plt
import time
from mios.bricks import *
import gear as mios
from abc import ABC, abstractmethod
import bbo.DistributionGaussian as Distribution
import bbo.updaters as Updater

# pseudo-code
# s1: load initial distribution & generate updater 
# s2: roll-out and get reward
#     Loop n_episode times:
#         1. generate n_samples samples based on the distribution ([n_samples, len(param)] = samples.shape)
#         2. for s in samples:
#             mios_execute(encoder(s)) & save the info while executation to info_file
#             costs = np.append(costs, calc(info_file))
#         3. updater.update_distribution(distribution, samples, costs) & save distribution


class Exp(ABC):
    def __init__(self, ip="localhost", path="EXP_1"):
        
        self.robot = ip
        self.distribution = None
        self.updater = None
        self.path = path
        self.object = None
        os.system("mkdir -p " + path)

        # init_distribution
        # init_updater
    
    def init_distribution(self, m_init:list, covar_init:np.ndarray, upper_limit:list, lower_limit):
        """
        initialize distribution
        
        Args:
            m_init (list): _description_
            covar_init (np.ndarray): _description_
        """
                
        # params related to the intial distribution and updater
        # m_init can be a list an array (N);  covar_init (N*N) matrix
        
        # shape check
        if len(m_init) == covar_init.shape[0] and covar_init.shape[0] == covar_init.shape[1]:
            print("shape check pass!")
        else:
            print("\033[91mError: Shape unmatched!\033[0m")
        

        self.distribution = Distribution.DistributionGaussian(m_init, covar_init)
        self.distribution.set_range(upper_limit, lower_limit)
        
    def init_updater(self, learning_rate=0.9, n_updates=20, n_samples=12, n_recalls=2):
        """
        initialize distribution updater

        Args:
            learning_rate (float, optional): learning_rate. Defaults to 0.8.
            n_updates (int, optional): n of distibution updating time. Defaults to 10.
            n_samples (int, optional): n of samples in each iteration. Defaults to 12.
            n_recalls (int, optional): n of recall samples from the previous batch. Defaults to 2.
        """
        
        self.updater = Updater.UpdaterCovarAdaptation(
            eliteness=10,
            weighting_method="PI-BB",
            max_level=None,
            min_level=0.000001,
            diag_only=False,
            learning_rate=learning_rate,
        )
    
        # params related to the learning optimation
        self.n_updates = n_updates # 10    
        self.n_samples = n_samples # 12
        self.n_recalls = n_recalls # 2 
    
    @abstractmethod
    def recover(self):
        """
        reset the robot and environment
        """
        print("------------ robot & env recover ------------")
        
        
    @abstractmethod
    def rollout(self, p, name):
        """
        convert the sampled parameter list into robot commands
        return the the cost and executation result
        
        Args:
            p (_type_): sampled parameter list

        Returns:
            float, str, float: cost, string (decrible the executation result), execution time
        """
        print("------------ trial with real robot ------------")
        return 0, "empty"
    
    def add_error(self):
        print("add random error if necessary")
        
    def calc_approach_pose(self):
        gen_approach_pose(insert_object=self.object, ip = "localhost") 
    
    def start_exp(self):
        
        if self.distribution == None:
            print("\033[91mError: distribution!\033[0m")
            return 0
        if self.updater == None:
            print("\033[91mError: updater!\033[0m")
            return 0
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})

        
        cv = None # v_c[i] = [cost, sample[i]]
        self.calc_approach_pose()
        self.recover()
        
                        
        for ep in range(self.n_updates):        
            
            recall = False  # if recall in this round
            print("################### iteration" + str(ep) + "start ###################") 
        
            if ep == 0:
                samples = self.distribution.generate_samples(self.n_samples) # size: n_samples X dim(mean)
            else:
                if self.n_samples > self.n_recalls:
                    recall = True
                    samples = self.distribution.generate_samples(self.n_samples - self.n_recalls) # size: {n_samples-n_recalls} X dim(mean)
                else:
                    samples = self.distribution.generate_samples(self.n_samples)   
            
            if recall: # choose the n_recalls best samples form the last iteration
                idx = np.argsort(cv[:,0])
                cv_recall = []
                for it in idx[: self.n_recalls]:
                    cv_recall.append(cv[it].tolist()) 
                cv_recall = np.array(cv_recall)
        
                
            
            costs = np.empty(0)
            times = np.empty(0)
            
            # roll-out the new samples
            for item in range(samples.shape[0]):
                c, result, t_exe = self.rollout(p = samples[item], name = "iteration_" + str(ep) + "_trial_" + str(item), mode = 1)
                print("\033[91m", result, "\033[0m")
                self.recover()
                print("iteration_" + str(ep) + " / trial_" + str(item) + " ----  cost: " + str(c))
                costs = np.append(costs, c) 
                times = np.append(times, t_exe)
                time.sleep(0.1)

            cv = np.concatenate([np.reshape(costs, (-1, 1)), samples], axis=1) 
            
            # concatenate the last best n samples, if recall
            if recall:
                cv = np.concatenate([cv, cv_recall], axis=0)
            
            print("################### iteration" + str(ep) + "finshed ###################")
            foldername = self.path + "/iteration_" + str(ep) + "_summary/"
            os.system("mkdir -p " + foldername)
            np.savetxt( foldername + "itr_" + str(ep) + "_resultCV", cv, delimiter=" ")   
            np.savetxt(foldername + "itr_" + str(ep) + "_time", times, delimiter=" ")   
            distribution_new, _ = self.updater.update_distribution(self.distribution, cv[:,1:], cv[:, 0]) 
            self.distribution = distribution_new
            
            distribution_info = np.concatenate([np.array([self.distribution.mean]), self.distribution.covar], axis=0)   
            np.savetxt( foldername + "itr_" + str(ep) + "distribution", distribution_info, delimiter=" ")            
        
        self.recover()
        

    def repeat_batch(self, file:str):
        data = np.loadtxt(file)
        samples = data[:, 1:]
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})
        ep = 0
        
        cv = None # v_c[i] = [cost, sample[i]]
        self.calc_approach_pose()
        self.recover()
        
        costs = np.empty(0)
        # roll-out the new samples
        for item in range(samples.shape[0]):
            c, result = self.rollout(p = samples[item], name = "iteration_" + str(ep) + "_trial_" + str(item))
            print(result)
            self.recover()
            print("iteration_" + str(ep) + " / trial_" + str(item) + " ----  cost: " + str(c))
            costs = np.append(costs, c) 
            time.sleep(0.1)
            
        cv = np.concatenate([np.reshape(costs, (-1, 1)), samples], axis=1) 
                        
        print("################### iteration" + str(ep) + "finshed ###################")
        foldername = self.path + "/iteration_" + str(ep) + "_summary/"
        os.system("mkdir -p " + foldername)
        np.savetxt( foldername + "itr_" + str(ep) + "_resultCV", cv, delimiter=" ")      
        

            
        
        self.recover()
        
            
    def same_start(self, file:str, mode, add_path):
        if self.distribution == None:
            print("\033[91mError: distribution!\033[0m")
            return 0
        if self.updater == None:
            print("\033[91mError: updater!\033[0m")
            return 0
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})

        
        cv = None # v_c[i] = [cost, sample[i]]
        # self.calc_approach_pose()
        self.recover()
        
                        
        for ep in range(self.n_updates):        
            
            recall = False  # if recall in this round
            print("################### iteration" + str(ep) + "start ###################") 
        
            if ep == 0:
                # samples = self.distribution.generate_samples(self.n_samples) # size: n_samples X dim(mean)
                data = np.loadtxt(file)
                samples = data                
                
            else:
                if self.n_samples > self.n_recalls:
                    recall = True
                    samples = self.distribution.generate_samples(self.n_samples - self.n_recalls) # size: {n_samples-n_recalls} X dim(mean)
                else:
                    samples = self.distribution.generate_samples(self.n_samples)   
            
            if recall: # choose the n_recalls best samples form the last iteration
                idx = np.argsort(cv[:,0])
                cv_recall = []
                for it in idx[: self.n_recalls]:
                    cv_recall.append(cv[it].tolist()) 
                cv_recall = np.array(cv_recall)
        
                
            
            costs = np.empty(0)
            times = np.empty(0)
            
            # roll-out the new samples
            for item in range(samples.shape[0]):
                c, result, t_exe = self.rollout(p = samples[item], name = "iteration_" + str(ep) + "_trial_" + str(item), mode=mode)
                print("\033[91m", result, "\033[0m")
                self.recover()
                print("iteration_" + str(ep) + " / trial_" + str(item) + " ----  cost: " + str(c))
                costs = np.append(costs, c)
                times = np.append(times, t_exe) 
                time.sleep(0.1)

            cv = np.concatenate([np.reshape(costs, (-1, 1)), samples], axis=1) 
            
            # concatenate the last best n samples, if recall
            if recall:
                cv = np.concatenate([cv, cv_recall], axis=0)
            
            print("################### iteration" + str(ep) + "finshed ###################")
            foldername = self.path + add_path+ "/iteration_" + str(ep) + "_summary/"
            os.system("mkdir -p " + foldername)
            np.savetxt(foldername + "itr_" + str(ep) + "_resultCV", cv, delimiter=" ")
            np.savetxt(foldername + "itr_" + str(ep) + "_time", times, delimiter=" ")      
            distribution_new, _ = self.updater.update_distribution(self.distribution, cv[:,1:], cv[:, 0]) 
            self.distribution = distribution_new
            
            distribution_info = np.concatenate([np.array([self.distribution.mean]), self.distribution.covar], axis=0)   
            np.savetxt( foldername + "itr_" + str(ep) + "distribution", distribution_info, delimiter=" ")
            
        
        self.recover()
        
        
    def repeat_compare(self, file:str):
        # if self.distribution == None:
        #     print("\033[91mError: distribution!\033[0m")
        #     return 0
        # if self.updater == None:
        #     print("\033[91mError: updater!\033[0m")
        #     return 0
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})

        # cv = None # v_c[i] = [cost, sample[i]]
        # self.calc_approach_pose()
        self.recover()
        
                        
        data = np.loadtxt(file)
        samples = data             
                

            
        # roll-out the new samples
        for item in range(0, data.shape[0]):
        # for item in [27,29,32]:
            print("################### trail" + str(item) + "start ###################")
            costs = np.empty(0)
            costs1 = np.empty(0)

            for i in range(1):
                c, result, t_exe = self.rollout(p = samples[item], name = "mode0trial_" + str(item) + "_rep_" + str(i), mode=0)
                print("\033[91m", result, "\033[0m")
                print("mode0trial_" + str(item) + "_rep_" + str(i) + " ----  cost: " + str(c))
                self.recover()
                
                costs = np.append(costs, c) 
                costs = np.append(costs, t_exe) 
                time.sleep(0.1)
                
                c, result, t_exe = self.rollout(p = samples[item], name = "mode1trial_" + str(item) + "_rep_" + str(i), mode=1)
                print("\033[91m", result, "\033[0m")
                print("mode1trial_" + str(item) + "_rep_" + str(i) + " ----  cost: " + str(c))
                self.recover()
                
                costs1 = np.append(costs1, c) 
                costs1 = np.append(costs1, t_exe) 
                time.sleep(0.1)
            
        

            final_cost = np.append(costs, costs1)     
            filename = self.path + "/trial" + str(item) + "_costsummary"
            os.system("mkdir -p " + self.path)
            np.savetxt( filename, final_cost, delimiter=" ") 
            
        
        self.recover()
    
    
    def transfer_test(self, file:str, mode:int):
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})
        self.recover()
                   
        samples = np.loadtxt(file)
        costs = np.empty(0)
        time_log = np.empty(0)

        # test the candidates
        for item in range(0, samples.shape[0]):
            print("################### trail" + str(item) + "start ###################")
            
        
            c, result, t_exe = self.rollout(p = samples[item], name = "tranferMode" + str(mode) + "item_" + str(item), mode=mode)
            print("\033[91m", result, "\033[0m")
            print("transfer mode" + str(mode) + "candidate" + str(item) + " ----  cost: " + str(c))
            
            # print("please help")
            # xxx = input()
            self.recover()
            
            costs = np.append(costs, c) 
            time_log = np.append(time_log, t_exe) 
            time.sleep(0.1)
            

        os.system("mkdir -p " + self.path)
        np.savetxt(self.path + "/mode" + str(mode) + "_costs", costs, delimiter=" ") 
        np.savetxt(self.path + "/mode" + str(mode) + "_times", time_log, delimiter=" ")
        
        
        self.recover()
        
    def same_start_fine(self, file:str, mode, add_path):
        if self.distribution == None:
            print("\033[91mError: distribution!\033[0m")
            return 0
        if self.updater == None:
            print("\033[91mError: updater!\033[0m")
            return 0
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})

        
        cv = None # v_c[i] = [cost, sample[i]]
        # self.calc_approach_pose()
        self.recover()
        
                        
        for ep in range(self.n_updates):        
            
            recall = False  # if recall in this round
            print("################### iteration" + str(ep) + "start ###################") 
        
            if ep == 0:
                # samples = self.distribution.generate_samples(self.n_samples) # size: n_samples X dim(mean)
                data = np.loadtxt(file)
                samples = data[[1,2,7,7,7], :]                
                
            else:
                if self.n_samples > self.n_recalls:
                    recall = True
                    samples = self.distribution.generate_samples(self.n_samples - self.n_recalls) # size: {n_samples-n_recalls} X dim(mean)
                else:
                    samples = self.distribution.generate_samples(self.n_samples)   
            
            if recall: # choose the n_recalls best samples form the last iteration
                idx = np.argsort(cv[:,0])
                cv_recall = []
                for it in idx[: self.n_recalls]:
                    cv_recall.append(cv[it].tolist()) 
                cv_recall = np.array(cv_recall)
        
                
            
            costs = np.empty(0)
            times = np.empty(0)
            
            # roll-out the new samples
            for item in range(samples.shape[0]):
                c, result, t_exe = self.rollout(p = samples[item], name = "iteration_" + str(ep) + "_trial_" + str(item), mode=mode)
                print("\033[91m", result, "\033[0m")
                self.recover()
                # time.sleep(0.5)
                # self.recover()
                print("iteration_" + str(ep) + " / trial_" + str(item) + " ----  cost: " + str(c))
                costs = np.append(costs, c)
                times = np.append(times, t_exe) 
                time.sleep(0.1)

            cv = np.concatenate([np.reshape(costs, (-1, 1)), samples], axis=1) 
            
            # concatenate the last best n samples, if recall
            if recall:
                cv = np.concatenate([cv, cv_recall], axis=0)
            
            print("################### iteration" + str(ep) + "finshed ###################")
            foldername = self.path + add_path+ "/iteration_" + str(ep) + "_summary/"
            os.system("mkdir -p " + foldername)
            np.savetxt(foldername + "itr_" + str(ep) + "_resultCV", cv, delimiter=" ")
            np.savetxt(foldername + "itr_" + str(ep) + "_time", times, delimiter=" ")      
            distribution_new, _ = self.updater.update_distribution(self.distribution, cv[:,1:], cv[:, 0]) 
            self.distribution = distribution_new
            
            distribution_info = np.concatenate([np.array([self.distribution.mean]), self.distribution.covar], axis=0)   
            np.savetxt( foldername + "itr_" + str(ep) + "distribution", distribution_info, delimiter=" ")
            
        
        self.recover()
    
    
    def gwxd(self, file:str, mode, add_path):
        if self.distribution == None:
            print("\033[91mError: distribution!\033[0m")
            return 0
        if self.updater == None:
            print("\033[91mError: updater!\033[0m")
            return 0
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})

        
        self.recover()
        
        cv = np.loadtxt(file)         
        distribution_new, _ = self.updater.update_distribution(self.distribution, cv[:,1:], cv[:, 0]) 
        self.distribution = distribution_new
        
        steps = 7
                        
        for ep in range(steps, steps + self.n_updates):        
            
            recall = False  # if recall in this round
            print("################### iteration" + str(ep) + "start ###################") 
        
            if ep == 0:
                # samples = self.distribution.generate_samples(self.n_samples) # size: n_samples X dim(mean)
                data = np.loadtxt(file)
                samples = data[:5, :]                
                
            else:
                if self.n_samples > self.n_recalls:
                    recall = True
                    samples = self.distribution.generate_samples(self.n_samples - self.n_recalls) # size: {n_samples-n_recalls} X dim(mean)
                else:
                    samples = self.distribution.generate_samples(self.n_samples)   
            
            if recall: # choose the n_recalls best samples form the last iteration
                idx = np.argsort(cv[:,0])
                cv_recall = []
                for it in idx[: self.n_recalls]:
                    cv_recall.append(cv[it].tolist()) 
                cv_recall = np.array(cv_recall)
        
                
            
            costs = np.empty(0)
            times = np.empty(0)
            
            # roll-out the new samples
            for item in range(samples.shape[0]):
                c, result, t_exe = self.rollout(p = samples[item], name = "iteration_" + str(ep) + "_trial_" + str(item), mode=mode)
                print("\033[91m", result, "\033[0m")
                self.recover()
                print("iteration_" + str(ep) + " / trial_" + str(item) + " ----  cost: " + str(c))
                costs = np.append(costs, c)
                times = np.append(times, t_exe) 
                time.sleep(0.1)

            cv = np.concatenate([np.reshape(costs, (-1, 1)), samples], axis=1) 
            
            # concatenate the last best n samples, if recall
            if recall:
                cv = np.concatenate([cv, cv_recall], axis=0)
            
            print("################### iteration" + str(ep) + "finshed ###################")
            foldername = self.path + add_path+ "/iteration_" + str(ep) + "_summary/"
            os.system("mkdir -p " + foldername)
            np.savetxt(foldername + "itr_" + str(ep) + "_resultCV", cv, delimiter=" ")
            np.savetxt(foldername + "itr_" + str(ep) + "_time", times, delimiter=" ")      
            distribution_new, _ = self.updater.update_distribution(self.distribution, cv[:,1:], cv[:, 0]) 
            self.distribution = distribution_new
            
            distribution_info = np.concatenate([np.array([self.distribution.mean]), self.distribution.covar], axis=0)   
            np.savetxt( foldername + "itr_" + str(ep) + "distribution", distribution_info, delimiter=" ")
            
        
        self.recover()
        
    def video_replay(self, file:str, mode):
        
        call_method("localhost", 12000, "set_grasped_object", {"object": self.object})
        self.recover()
        
                        
        data = np.loadtxt(file)
        samples = data[:, 1:]          
        print("replay mode", mode)               

            
        # roll-out the new samples
        for item in range(0, data.shape[0]):
        # for item in [27,29,32]:
            print("################### trail" + str(item) + "start ###################")
            costs = np.empty(0)
            costs1 = np.empty(0)

            for i in range(1):
                if mode == 0:
                    c, result, t_exe = self.rollout(p = samples[item], name = "mode0trial_" + str(item) + "_rep_" + str(i), mode=0)
                    print("\033[91m", result, "\033[0m")
                    print("mode0trial_" + str(item) + "_rep_" + str(i) + " ----  cost: " + str(c))
                    self.recover()
                
                    costs = np.append(costs, c) 
                    costs = np.append(costs, t_exe) 
                    time.sleep(0.1)
                
                if mode == 1:
                    c, result, t_exe = self.rollout(p = samples[item], name = "mode1trial_" + str(item) + "_rep_" + str(i), mode=1)
                    print("\033[91m", result, "\033[0m")
                    print("mode1trial_" + str(item) + "_rep_" + str(i) + " ----  cost: " + str(c))
                    self.recover()
                    
                    costs1 = np.append(costs1, c) 
                    costs1 = np.append(costs1, t_exe) 
                    time.sleep(0.1)
            
