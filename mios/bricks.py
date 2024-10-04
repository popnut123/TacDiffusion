import numpy as np
import math
from moving import *
from spatialmath.base import rpy2tr, transl

def encoder(p, obj, name):
    """
    Convert the given parameters into the payload of mios & execute it with mios
    Args:
        p (list, len(p)=18): input parameter list
        part-1: search_a[0, 1, 2, 3, 4, 5]  p[0-5]
        pert-2: search_f[0, 1, 3, 4, 5]     p[6-10]
        part-3: search_phi[0, 1, 3, 4, 5]   p[11-15]
        part-4: K_x[0, 1, 3, 4, 5]          p[16-17]
    """
    
    if isinstance(p, np.ndarray):
        p = p.tolist()
    
    print("Receiving smaple ", p,  "for roll-out", p)
        
    search_a = p[0:6]
    # search_a[2] = search_a[2]*2
    # /////////// transfer /////////////
    # search_a[3] = search_a[3]*2
    # search_a[4] = search_a[4]*2
    # search_a[5] = search_a[5]/2
    # /////////// transfer /////////////
    
    search_f = p[6:11]
    search_f.insert(2,0)
    search_phi = p[11:16]
    search_phi.insert(2,0)
    K_x = [p[16],p[16],0,p[17],p[17],p[17]]
    
    # obj = "hex1"

    # clip and scale each element in p back to the original ranges
    content = {
        "skill": {
            "objects": {
                "Container": obj + "_hole",
                "Approach": obj + "_app",
                "Insertable": obj
            },
            "time_max":15,
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
                # "mode": 0,       # 0" none feedback; 1" feedback x,y; 2" xy_new_feedback; 3" fix rpy
                "mode": 0, # 0: no detection, 1: state detection
                "name": name,
                "search_a": search_a,
                "search_f":search_f,
                "search_phi": search_phi,
                "K_x": K_x,
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
    print(content)
    return content

def get_limit_mean():
    """ get the upper_limit, lower_limit, mean of the parameters to be learnt
    Returns:
        a,b,c (list, length=18): upper_limit, lower_limit, mean (for initial distribution)
    """
    
    up_a = [4, 4, 10, 2, 2, 4]
    low_a = [0, 0, 0, 0, 0, 0]
    up_f = [3, 3, 1, 1, 1]
    low_f = [0, 0, 0, 0, 0]
    up_phi = [math.pi, math.pi, math.pi, math.pi, math.pi]
    low_phi = [-math.pi, -math.pi, -math.pi, -math.pi, -math.pi]
    up_K = [1000, 75]
    low_K = [0, 0]
    upper = up_a + up_f + up_phi + up_K
    lower = low_a + low_f + low_phi + low_K
    mean = [(i+j)/2 for (i,j) in zip(upper, lower)]

    return upper, lower, mean

def error_simulate(insert_object="hex1", ip = "localhost"):
    """Gernerate random error for each batch;
    modify the "cointainer" pose and adjust the pose of "approach", correspondingly 

    Args:
        container_object (str): the name of the container object saved in the mios & MangoDB
        ip (str): ip address of the PC, which controls the robot
        
    """
    container_object = insert_object + "_hole"
    app_object = insert_object + "_app"
    
    tolerance = np.array([2e-3, 2e-3, 0, 2, 2, 2]) # unit: [mm, degree]     
    err_sample = np.random.random_sample((6,))
    # print("err_sample:", err_sample)
    err = 2 * tolerance * err_sample - tolerance
    print("err:", err)
    OB_T_OBerr = rpy2tr(err[-3:], unit='deg') # T of the estimated container_object frame wrt original container_object frame
    OB_T_OBerr[:3,3] = np.reshape(err[:3], (3, ))
    
    # calc err OB frame
    r = call_method(ip, 12000, "get_object", {"object":container_object})  
    O_T_OB = np.reshape(r["result"]["O_T_OB"], (4,4)).T
    O_T_OBerr = O_T_OB @ OB_T_OBerr
    
    # set err OB frame
    name_OB_e = container_object + "_e"
    call_method(ip, 12000, "teach_object", {"object": name_OB_e})
    payload = {
        "object": name_OB_e,
        "data": {
            "x": O_T_OBerr[0,3],
            "y": O_T_OBerr[1,3],
            "z": O_T_OBerr[2,3],
            "R": np.reshape(O_T_OBerr[:3,:3].T, (-1,)).tolist(),
        },
    }
    call_method(ip, 12000, "set_partial_object_data", payload)
    
    
    # calculate approach pose based container pose
    hole_T_app = transl(0, 0, -0.02)
    # print(O_T_OBerr)
    O_T_APP = O_T_OBerr @ hole_T_app
    # print(O_T_APP)
    
    name_APP_e = app_object + "_e"
    call_method(ip, 12000, "teach_object", {"object": name_APP_e})
    payload = {
        "object": name_APP_e,
        "data": {
            "x": O_T_APP[0,3],
            "y": O_T_APP[1,3],
            "z": O_T_APP[2,3],
            "R": np.reshape(O_T_APP[:3,:3].T, (-1,)).tolist(),
        },
    }
    # print(payload["data"]["R"])
    call_method(ip, 12000, "set_partial_object_data", payload)
    print("add random error for container pose; please use the following object names for the following insertion experiment: ")

    print(insert_object)
    print(name_APP_e)
    print(name_OB_e)
    # move_to_location(ip, insert_object + "_hole")
    # move_to_location(ip, name_OB_e)
    # move_to_location(ip, name_APP_e)
    
    # hex1
    # hex1_hole
    # hex1_app
    
def gen_approach_pose(insert_object, dist=0.085, ip = "localhost"):
    """generate the "Approach": obj + "_app" based on t

    Args:
        insert_object (str): name of the "Insertable"
        ip (str, optional): _description_. Defaults to "localhost".
    
        # "Container": obj + "_hole",
        # "Approach": obj + "_app",
        # "Insertable": obj
    """
    container_object = insert_object + "_hole"
    app_object = insert_object + "_app"
    
    r = call_method(ip, 12000, "get_object", {"object":container_object})  
    O_T_hole = np.reshape(r["result"]["O_T_OB"], (4,4)).T
    
    # calculate approach pose based container pose
    hole_T_app = transl(0, 0, -dist)
    O_T_APP = O_T_hole @ hole_T_app
    
    call_method(ip, 12000, "teach_object", {"object": app_object})
    payload = {
        "object": app_object,
        "data": {
            "x": O_T_APP[0,3],
            "y": O_T_APP[1,3],
            "z": O_T_APP[2,3],
            "R": np.reshape(O_T_APP[:3,:3].T, (-1,)).tolist(),
        },
    }
    # print(payload["data"]["R"])
    call_method(ip, 12000, "set_partial_object_data", payload)
    print("generate the appraoch pose based on the container pose, which is 6cm behind the container pose in z-axis")
    
    
def rollout0(p, ip, obj, name):
    """
    Convert the given parameters into the payload of mios & execute it with mios
    Args:
        p (list, len(p)=18): input parameter list
        robot: ip address of the PC, which controls the robot
    """
    
    content = encoder(p, obj, name)
    content["skill"]["p2"]["mode"] = 0
    t = Task(ip)
    t.add_skill("insertion", "FFInsertion", content)
    t.start()
    time.sleep(0.5)
    result = t.wait()
    print(">>>>>>>>>> Result <<<<<<<<<<" + str(result))
    succ = result["result"]["task_result"]["success"]
    costs = result["result"]["task_result"]["skill_results"]["insertion"]["cost"]  
    
    if succ == True:
        c =  costs["time"]/15 # exe_time / time_max
        res = "success"    
        
    else:
        c = 1 + math.exp(costs["desired_pose"])
        res = "failed"
    
    return c, res, costs["time"]
        
        
    # call mios to execute & return execution result
    # cost calculation
    
def rollout1(p, ip, obj, name):
    """
    Convert the given parameters into the payload of mios & execute it with mios
    Args:
        p (list, len(p)=18): input parameter list
        robot: ip address of the PC, which controls the robot
    """
    
    content = encoder(p, obj, name)
    content["skill"]["p2"]["mode"] = 1
    t = Task(ip)
    t.add_skill("insertion", "FFInsertion", content)
    t.start()
    time.sleep(0.5)
    result = t.wait()
    print(">>>>>>>>>> Result <<<<<<<<<<" + str(result))
    succ = result["result"]["task_result"]["success"]
    costs = result["result"]["task_result"]["skill_results"]["insertion"]["cost"]  
    
    if succ == True:
        c =  costs["time"]/15 # exe_time / time_max
        res = "success"    
        
    else:
        c = 1 + math.exp(costs["desired_pose"])
        res = "failed"
    
    return c, res, costs["time"]
    
def recovery(obj):
    extraction_context = {
        "skill": {
            "objects": {
                "Container": obj + "_hole",
                "ExtractTo": obj + "_app",
                "Extractable": obj
            },
            "time_max": 15,
            "p0": {
                "search_a": [0, 0, 0, 0, 0, 0],
                "search_f": [0, 0, 0, 0, 0, 0],
                "K_x": [1500, 1500, 1500, 150, 150, 150],
                "dX_d": [0.1, 0.5],
                "ddX_d": [0.5, 1]
            },
            "p1": {
                "dX_d": [0.05, 0.25],
                "ddX_d": [0.5, 1],
                "K_x": [1000, 1000, 1500, 100, 100, 100]
            }
        },
        "control": {
            "control_mode": 0  # 0" non-feedback;  1" feedback-xy; 2" ...
        },
        "user": {
            "env_X": [0.005, 0.01, 0.01, 0.05, 0.05, 0.05],
            "env_dX": [0.001, 0.001, 0.001, 0.005, 0.005, 0.005]
        }
    }
    
    t = Task("localhost")
    t.add_skill("extraction", "TaxExtraction", extraction_context)
    t.start()
    time.sleep(0.1)
    result = t.wait()
    # move_to_location("localhost","app1")
    # moveJ("localhost", [0.7551658471712469, 0.3145440098276278, -0.0020069767591500715, -1.9978433348505118, 0.0199849793828196, 2.3259511258602057, 0.775138029407372]) #twist
    # moveJ("localhost", [0.4460773134267041,   0.20413696931713188,   -0.058014110954107086,   -2.247725224147367,   0.0361153903851916,   2.4246578111243973,   0.8941369903599501]) # key
    moveJ("localhost", [0.0803120750194111,   0.17031096563632026,   0.510383292659322,   -2.2621084585695885,   -0.09551886005147923,   2.397220746119817,   0.8516882177690664]) # lingyun chair
    # moveJ("localhost", [-0.11708726646084533, 0.008788426567465465, 0.1662015935951989, -2.4383742417452625, -0.008180353962712817, 2.41141472292263, 0.8943674993087991]) # peg-c2

    # moveJ("localhost", [-0.141590935587883, -0.036755463176652, 0.058244734347922815, -2.426610706696234, 0.01416963450776206, 2.3437419615586594, 0.7186153313559546]) # pegs
    # moveJ("localhost", [-0.12073499628631658, -0.22682361728684944, 0.031210873097704164, -2.3957437287921324, 0.01387878938978681, 2.165390047470728, 0.7358379275773186])
    print("Extraction Result: " + str(result))

def cost():
    pass   


# TODO: read mios cost defination
# TODO: try extraction
# TODO: choose learning parameters
# TODO: decide if save wrench_inner, _outer, _ext while learning
    

def error(insert_object, tail):
    container_object = insert_object + "_hole"
    app_object = insert_object + "_app"
    
    teach_location("localhost", insert_object+tail)
    

    err = [0.0, 0, 0, 0, 0, -3] # training -2
    # err = [0.0, 0.0, 0, 0, 0, -5] # 100 compare
    # err = [0, 0, 0, 1, 1, 1] # c-e
    # err = [0, 0, 1, -1, 1, 0] # c-e1
    # err  = [0, 0.001, 0, 2, 2, 0]
    print("err:", err)
    OB_T_OBerr = rpy2tr(err[-3:], unit='deg') # T of the estimated container_object frame wrt original container_object frame
    OB_T_OBerr[:3,3] = np.reshape(err[:3], (3, ))
    
    # calc err OB frame
    r = call_method("localhost", 12000, "get_object", {"object":container_object})  
    print(r)
    O_T_OB = np.reshape(r["result"]["O_T_OB"], (4,4)).T
    O_T_OBerr = O_T_OB @ OB_T_OBerr
    
    # set err OB frame
    name_OB_e = insert_object+tail+"_hole"
    call_method("localhost", 12000, "teach_object", {"object": name_OB_e})
    payload = {
        "object": name_OB_e,
        "data": {
            "x": O_T_OBerr[0,3],
            "y": O_T_OBerr[1,3],
            "z": O_T_OBerr[2,3],
            "R": np.reshape(O_T_OBerr[:3,:3].T, (-1,)).tolist(),
        },
    }
    call_method("localhost", 12000, "set_partial_object_data", payload)
    
    
    # calculate approach pose based container pose
    hole_T_app = transl(0, 0, -0.085)
    # print(O_T_OBerr)
    O_T_APP = O_T_OBerr @ hole_T_app
    # print(O_T_APP)
    
    name_APP_e = insert_object+tail + "_app"
    call_method("localhost", 12000, "teach_object", {"object": name_APP_e})
    payload = {
        "object": name_APP_e,
        "data": {
            "x": O_T_APP[0,3],
            "y": O_T_APP[1,3],
            "z": O_T_APP[2,3],
            "R": np.reshape(O_T_APP[:3,:3].T, (-1,)).tolist(),
        },
    }
    # print(payload["data"]["R"])
    call_method("localhost", 12000, "set_partial_object_data", payload)
    print("add random error for container pose; please use the following object names for the following insertion experiment: ")
    move_to_location("localhost",app_object) 
    time.sleep(1)
    move_to_location("localhost", name_APP_e)
    call_method("localhost", 12000, "teach_object", {"object": insert_object+tail})       


def stage_peg(name):
    r = call_method("localhost", 12000, "get_object", {"object":name})  
    print(r)
    O_T_OB = np.reshape(r["result"]["O_T_OB"], (4,4)).T
    OB_T_app = transl(0, 0, -0.015)
    print(O_T_OB)
    O_T_app = O_T_OB @ OB_T_app
    print(O_T_app)

    call_method("localhost", 12000, "teach_object", {"object": name+"_app"})
    payload = {
        "object": name+"_app",
        "data": {
            "x": O_T_app[0,3],
            "y": O_T_app[1,3],
            "z": O_T_app[2,3],
            "R": np.reshape(O_T_app[:3,:3].T, (-1,)).tolist(),
        },
    }
    call_method("localhost", 12000, "set_partial_object_data", payload)
    
    
    OB_T_hole = transl(0,0, 0.065)
    O_T_hole = O_T_OB @ OB_T_hole
    call_method("localhost", 12000, "teach_object", {"object": name+"_hole"})
    payload = {
        "object": name+"_hole",
        "data": {
            "x": O_T_hole[0,3],
            "y": O_T_hole[1,3],
            "z": O_T_hole[2,3],
            "R": np.reshape(O_T_hole[:3,:3].T, (-1,)).tolist(),
        },
    }
    call_method("localhost", 12000, "set_partial_object_data", payload)
    