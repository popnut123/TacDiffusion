#!/usr/bin/python3 -u
from task import *
from ws_client import *


def teach_location(robot: str, location: str):
    call_method(robot, 12000, "teach_object", {"object": location})


def move_to_location(robot: str, location: str):
    context = {
        "skill": {
            "p0":{
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1],
                "K_x": [1500, 1500, 1500, 150, 150, 150],

            },
            "objects": {
                    "GoalPose": location
                }
        },
        "control": {
            "control_mode": 2
        }
    }
    t = Task(robot)
    t.add_skill("move", "TaxMove", context)
    t.start()
    result = t.wait()
    print("Result: " + str(result["result"]["error"]))

def moveJ(robot: str, q_g):        
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