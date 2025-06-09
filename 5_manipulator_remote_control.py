from mios.exp import Exp
import mios.bricks as Brick
import numpy as np
from mios.moving import *
from spatialmath.base import rpy2tr, transl
import math
from mios.gear import *

class FFInsertionEXP(Exp):
    
    def set_object(self, obj:str):
        self.object = obj
    
        
    def recover(self):
        Brick.recovery(obj=self.object)
    
    # def rollout(self, p, name):
    #     return Brick.rollout(p, ip="localhost", obj=self.object, name=name)
    def rollout(self, p, name, mode=1):
        if mode == 0:
            return Brick.rollout0(p, ip="localhost", obj=self.object, name=name)
        if mode == 1:
            return Brick.rollout1(p, ip="localhost", obj=self.object, name=name) 
        
    
    def add_error(self):
        Brick.error_simulate(insert_object=self.object, ip = "localhost")
           
def il_insertion_test(peg_name, recorded_file_name):
    robot = "localhost"
    print(call_method(robot, 12000, "get_state"))
    call_method(robot, 12000, "set_grasped_object", {"object": peg_name})
    print(f'peg name: {peg_name}')
    content = {
        "skill": {
            "objects": {
                "Container": f"{peg_name}_hole", 
                "Approach": f"{peg_name}_app",
                "Insertable": peg_name
            },
            "time_max": 10, 
            "p0": {
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1],
                "DeltaX": [0, 0, 0, 0, 0, 0],
                "K_x": [1500, 1500, 1500, 100, 100, 100]
            },
            "p1": {
                "dX_d": [0.03, 0.1],
                "ddX_d": [0.5, 0.1],
                "K_x": [500, 500, 800, 100, 100, 100]
            },
            "p2": {
                "mode": 0, # 0: no detection, 1: state detection
                "name": recorded_file_name,
                "search_a": [5, 5, 10, 0, 0, 0],
                "search_f": [1.2, 1.2, 0, 1.2, 1.2, 0],
                "search_phi": [0, math.pi/2, 3.14159265358979323846/2, 3.14159265358979323846/2, 0, 0],
                "K_x": [500, 500, 800, 100, 100, 100],
                "f_push": [0, 0, 0, 0, 0, 0],
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1]
            },
            "il" : {
                "port_src": 2333,
                "f_push": 8, 
                "if_ds": True, # True: apply Filter;  # False: no Filter
                "model_ip": "10.157.175.109" # ip of the GPU PC
            }
            
        },
        "control": {
            "control_mode": 0
        },
        "user": {
            "env_X": [0.01, 0.01, 0.002, 0.05, 0.05, 0.05],
            "env_dX": [0.001, 0.001, 0.001, 0.005, 0.005, 0.005],
            "F_ext_contact": [3.0, 2.0]
        }
    }
    t = Task(robot)
    t.add_skill("insertion", "ILInsertion", content)
    t.start()
    time.sleep(0.5)
    result = t.wait()
    print("Insertion Result: " + str(result))
    return result

def insert_repeat_IL(peg_name = "peg_IL_test", insert_repeat_num = 3): 
    print(f'insert_repeat_num: {insert_repeat_num}')
    
    with open(f'{peg_name}_result.txt', 'w') as file:
        for i in range(insert_repeat_num):
            iteration_name = i // 10
            trial_name = i % 10
            recorded_file_name = f"{peg_name}_iteration_{iteration_name}_trial_{trial_name}"
            print(f'recorded_file_name: {recorded_file_name}')
            
            result = il_insertion_test(peg_name, recorded_file_name)
            file.write(f'{recorded_file_name}: {result} \n')
            
            print('!!! insert finished')

            # time.sleep(0.5)        
            Brick.recovery(peg_name)
            print('!!! recovery finished')
        
    print(f'!!! insert_repeat for {peg_name} finished !!!')

def insert_auto_IL():
    print('***** Using Diffusion Model *****')
    # peg_name for cuboid insertion experiments
    insert_repeat_num = 3

    peg_name_all = ["peg_IL_test"]  
    
    print(f'peg_name_all: {peg_name_all}')
    print(f'insert_repeat_num: {insert_repeat_num}')
    
    for peg_name in peg_name_all:
        # for initialization
        Brick.recovery(peg_name)
        print(f'!!! {peg_name} recovery finished !!!')
        # time.sleep(0.5)        
        
        # begin insertion experiments
        insert_repeat_IL(peg_name, insert_repeat_num)
        print(f'!!! insert_auto of {peg_name} finished !!!')
        
    print('all exps are successful implemented!')


def peg_pose_record():
    
    peg_name_all = "peg_IL_test"

    print(f'peg_name_all: {peg_name_all}')
    
    with open(f'pose_record_hole.txt', 'w') as file:
        
        for peg_name in peg_name_all:
        
            r = call_method("localhost", 12000, "get_object", {"object": f"{peg_name}_hole"})  
            # print(f'{peg_name}_hole: {r}')
            
            O_T_OB = np.reshape(r["result"]["O_T_OB"], (4,4)).T
            # print(f'{peg_name}_hole: {O_T_OB}')
            
            file.write(f'{peg_name}_hole: {O_T_OB} \n')
            
    with open(f'pose_record_app.txt', 'w') as file:  
        for peg_name in peg_name_all:
              
            r = call_method("localhost", 12000, "get_object", {"object": f"{peg_name}_app"})  
            # print(f'{peg_name}_app: {r}')
            O_T_OB = np.reshape(r["result"]["O_T_OB"], (4,4)).T
            # print(f'{peg_name}_app: {O_T_OB}')
            
            file.write(f'{peg_name}_app: {O_T_OB} \n')
        
    print('all pegs are successful recorded!')


# if __name__ == "__main__":
    # main()
    # error("cylinder_m", "e2")

