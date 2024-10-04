from task import *
from moving import *
import time
import math


robot = "localhost"
   
def grasp(width):
    payload = {
        "width": width,
        "speed": 0.05,
        "force": 0.1,
        "epsilon_inner": 1,
        "epsilon_outer": 1, 

    }
    return call_method(robot, 12000, "grasp", payload)

def move_gripper(width):
    payload = {
        "width": width,
        "speed": 0.05,
    }
    return call_method(robot, 12000, "move_gripper", payload)

def modify_taught_pose(x, y, z, name:str):
    payload = {
        "object": name,
        "data": {
            "x": x,
            "y": y,
            "z": z,
            #"R": [0, 1, 0, 1, 0, 0, 0, 0, -1],
            "R": [1, 0, 0, 0, -1, 0, 0, 0, -1],
        },
    }
    return call_method(robot, 12000, "set_partial_object_data", payload)
    
def moveJ(q_g):        
    """
    call mios for movign the lefthand to desired joint position

    Paramter
    --------
    q_g: list, len(7)
    """
    parameters = {
        "parameters": {
        "pose": "NoneObject",
        "q_g": q_g,
        "speed": 0.5,
        "acc": 0.7,        
        }       
    }
    return start_task_and_wait(robot, "MoveToJointPose", parameters, False)


def insertion():
    print(call_method(robot, 12000, "get_state"))
    call_method(robot, 12000, "set_grasped_object", {"object": "hex1"})
    content = {
        "skill": {
            "objects": {
                "Container": "hole",
                "Approach": "app1",
                "Insertable": "hex1"
            },
            "time_max": 30,
            "p0": {
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1],
                "DeltaX": [0, 0, 0, 0, 0, 0],
                "K_x": [1500, 1500, 1500, 100, 100, 100]
            },
            "p1": {
                "dX_d": [0.03, 0.1],
                "ddX_d": [0.5, 0.1],
                "K_x": [500, 500, 500, 100, 100, 100]
            },
            "p2": {
                "search_a": [5, 5, 10, 0, 0, 0],
                "search_f": [1.2, 1.2, 0, 1.2, 1.2, 0],
                "search_phi": [0, math.pi/2, 3.14159265358979323846/2, 3.14159265358979323846/2, 0, 0],
                "K_x": [500, 500, 500, 100, 100, 100],
                "f_push": [0, 0, 0, 0, 0, 0],
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1]
            },
            "p3": {
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1],
                "f_push": 10,
                "K_x": [500, 500, 0, 800, 800, 800]
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
    t.add_skill("insertion", "FFInsertion", content)
    t.start()
    time.sleep(0.5)
    result = t.wait()
    print("Result: " + str(result))
    return result
    
        
def extract_skill():
    call_method(robot, 12000, "set_grasped_object", {"object": "hex1"})
    extraction_context = {
        "skill": {
            "objects": {
                "Container": "hole",
                "ExtractTo": "app1",
                "Extractable": "hex1"
            },
            "time_max": 10,
            "p0": {
                "search_a": [0, 0, 0, 0, 0, 0],
                "search_f": [0, 0, 0, 0, 0, 0],
                "K_x": [1500, 1500, 1500, 150, 150, 150],
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1]
            },
            "p1": {
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1],
                "K_x": [1000, 1000, 1500, 100, 100, 100]
            }
        },
        "control": {
            "control_mode": 0
        },
        "user": {
            "env_X": [0.005, 0.01, 0.01, 0.05, 0.05, 0.05],
            "env_dX": [0.001, 0.001, 0.001, 0.005, 0.005, 0.005]
        }
    }
    
    t = Task(robot)
    t.add_skill("extraction", "TaxExtraction", extraction_context)
    t.start()
    time.sleep(0.1)
    result = t.wait()
    print("Result: " + str(result))
    
