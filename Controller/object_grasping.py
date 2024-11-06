import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum, splines
import motive_client, robot2sr_controller as rsr_ctrl
import threading

agent: rsr.Robot = None
object: manipulandum.Shape = None
obstacles = []

mocap = motive_client.MocapReader()

simulation = True

def updateConfig():
    global agent, object

    agents_config, objects_config, _, msg = mocap.getConfig()

    if agents_config and objects_config:
        agent_config = agents_config[0]
        if agent:
            agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
            agent.k1 = agent_config['k1']
            agent.k2= agent_config['k2']
            agent.head.pose = agent_config['head']
            agent.tail.pose = agent_config['tail']
        else:
            agent = rsr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])

        object_config = objects_config[0]
        object_pose = [object_config['x'], object_config['y'], object_config['theta']]
        if object:
            if not simulation:
                object.pose = object_pose
            else:
                pass
        else:
            object = manipulandum.Shape(object_config['id'], object_pose)
    else:
        print(msg)

def updateConfigLoop():

    while True:
        updateConfig()


if __name__ == "__main__":

    # ------------------------ Start tracking -----------------------
    '''
    Detect locations and geometries of:
        - robot
        - object to grasp
        - obstacles
    '''

    print('Start Motive streaming....')
    mocap.startDataListener() 

    update_thread = threading.Thread(target=updateConfigLoop)
    update_thread.daemon = True  
    update_thread.start()

    while not agent or not object:
        pass
    # ---------------------------------------------------------------

    # ------------------ Create the environment map -----------------
    '''
    Steps:
        - transform obstacles by expanding them based on robot's 
        geometry (Minkowski sum)
    '''
    
    
    # ---------------------------------------------------------------
    
    # ---------------- Determine grasping parameters ----------------
    '''
    Options:
        - grasp from the side opposite to the heading direction
        - optimization approach based on minimum forces and shape

    !! Estimate the grasp feasibility with given obstacles
    '''
    # ---------------------------------------------------------------

    # ------------------------ Path planning ------------------------
    '''
    ** Start first with path planning that does not require reshaping 
    of the robot 
    '''
    # ---------------------------------------------------------------

