import pybullet as p
from activations import fn_to_string

from pyrosim.nndf import NNDF

from pyrosim.linksdf  import LINK_SDF

from pyrosim.linkurdf import LINK_URDF

from pyrosim.staticsdf       import STATIC_SDF

from pyrosim.model import MODEL

from pyrosim.sdf   import SDF

from pyrosim.urdf  import URDF

from pyrosim.joint import JOINT

SDF_FILETYPE  = 0

URDF_FILETYPE = 1

NNDF_FILETYPE   = 2

# global availableLinkIndex

# global linkNamesToIndices

def End():

    if filetype == SDF_FILETYPE:

        sdf.Save_End_Tag(f)

    elif filetype == NNDF_FILETYPE:

        nndf.Save_End_Tag(f)
    else:
        urdf.Save_End_Tag(f)

    f.close()

def End_Model():

    model.Save_End_Tag(f)

def Get_Touch_Sensor_Value_For_Link(linkName):

    touchValue = -1.0

    desiredLinkIndex = linkNamesToIndices[linkName]

    pts = p.getContactPoints()

    for pt in pts:

        linkIndex = pt[4]

        if ( linkIndex == desiredLinkIndex ):

            touchValue = 1.0

    return touchValue

def Get_Rotational_Sensor_Value_For_Joint(jointName, bodyID):

    torque = 0

    desiredJointIndex = jointNamesToIndices[jointName]
    
    torque = p.getJointState(bodyID, desiredJointIndex)[3]
    
    rotation_of_joint = p.getJointState(bodyID, desiredJointIndex)[0]
    
    return rotation_of_joint

def Get_Velocity_Sensor_Value_For_Link(linkName, bodyID):

    vel = 0
    linkNameIndex = linkNamesToIndices[linkName]
    
    vel = p.getLinkState(bodyID, linkNameIndex)[6] # linear velocity
    
    return (abs(vel[0]) + abs(vel[1]) + abs(vel[2])) / 3.0

def Get_Base_Velocity_Sensor_Value(bodyID):

    vel = p.getBaseVelocity(bodyID)
    return (abs(vel[0][0]) + abs(vel[0][1]) + abs(vel[0][2])) / 3.0

def Prepare_Link_Dictionary(bodyID):

    global linkNamesToIndices

    linkNamesToIndices = {}

    for jointIndex in range( 0 , p.getNumJoints(bodyID) ):

        jointInfo = p.getJointInfo( bodyID , jointIndex )

        jointName = jointInfo[1]

        jointName = jointName.decode("utf-8")

        jointName = jointName.split("_")

        linkName = jointName[1]

        linkNamesToIndices[linkName] = jointIndex

        if jointIndex==0:

           rootLinkName = jointName[0]

           linkNamesToIndices[rootLinkName] = -1 

def Prepare_Joint_Dictionary(bodyID):

    global jointNamesToIndices

    jointNamesToIndices = {}

    for jointIndex in range( 0 , p.getNumJoints(bodyID) ):

        jointInfo = p.getJointInfo( bodyID , jointIndex )

        jointName = jointInfo[1].decode("utf-8")

        jointNamesToIndices[jointName] = jointIndex

def Prepare_To_Simulate(bodyID):

    Prepare_Link_Dictionary(bodyID)

    Prepare_Joint_Dictionary(bodyID)
    

def Send_Cube(name="default",pos=[0,0,0],size=[1,1,1], static=False, color_name="Cyan", color_rgba=[0,1,1,1], mass=1.0):

    global availableLinkIndex

    global links

    if filetype == SDF_FILETYPE:

        Start_Model(name,pos,static)
        
        link = LINK_SDF(name,pos,size, static, color_name, color_rgba)

        links.append(link)
    else:
        link = LINK_URDF(name,pos,size, static, mass, color_name, color_rgba)

        links.append(link)

    link.Save(f)

    if filetype == SDF_FILETYPE:

        End_Model()

    linkNamesToIndices[name] = availableLinkIndex

    availableLinkIndex = availableLinkIndex + 1

def Send_Joint(name,parent,child,type,position,jointAxis = "0 1 0"):

    joint = JOINT(name,parent,child,type,position)

    joint.Save(f, jointAxis)

def Send_Motor_Neuron(name,jointName, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "motor"  jointName = "' + jointName + '" activation="'+str(activation)+ '" />\n')

def Send_Touch_Sensor_Neuron(name,linkName, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "touch_sensor" linkName = "' + linkName + '" activation="'+str(activation) +'" />\n')
    
def Send_Rotation_Sensor_Neuron(name, jointName, bodyID, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "rotation_sensor" jointName = "' + jointName + '" bodyID="' + str(bodyID) + '" activation="'+str(activation) +'"  />\n')
    
def Send_Link_Velocity_Sensor_Neuron(name, linkName, bodyID, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "link_velocity_sensor" linkName = "' + linkName + '" bodyID="' + str(bodyID) + '" activation="'+str(activation) +'"  />\n')
    
    
def Send_Base_Velocity_Sensor_Neuron(name, bodyID, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "base_velocity_sensor" linkName = "' + "BASE" + '" bodyID="' + str(bodyID) + '" activation="'+str(activation) +'"  />\n')
    
def Send_CPG(name, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "cpg"' + ' activation="'+str(activation) + '"  />\n')
    
def Send_Hidden_Neuron(name, activation):
    activation = fn_to_string(activation)
    f.write('    <neuron name = "' + str(name) + '" type = "hidden"'+' activation="'+str(activation)+ '"/>\n')

def Send_Synapse( sourceNeuronName , targetNeuronName , weight ):

    f.write('    <synapse sourceNeuronName = "' + str(sourceNeuronName) + '" targetNeuronName = "' + str(targetNeuronName) + '" weight = "' + str(weight) + '" />\n')

 
def Set_Motor_For_Joint(bodyIndex,jointName,controlMode,targetPosition,maxForce):

    p.setJointMotorControl2(

        bodyIndex      = bodyIndex,

        jointIndex     = jointNamesToIndices[jointName],

        controlMode    = controlMode,

        targetPosition = targetPosition,

        force          = maxForce)

def Start_NeuralNetwork(filename):

    global filetype

    filetype = NNDF_FILETYPE

    global f

    f = open(filename,"w")

    global nndf

    nndf = NNDF()

    nndf.Save_Start_Tag(f)

def Start_SDF(filename):

    global availableLinkIndex

    availableLinkIndex = -1

    global linkNamesToIndices

    linkNamesToIndices = {}

    global filetype

    filetype = SDF_FILETYPE

    global f
 
    f = open(filename,"w")

    global sdf

    sdf = SDF()

    sdf.Save_Start_Tag(f)

    global links

    links = []

def Start_URDF(filename):

    global availableLinkIndex

    availableLinkIndex = -1

    global linkNamesToIndices

    linkNamesToIndices = {}

    global filetype

    filetype = URDF_FILETYPE

    global f

    f = open(filename,"w")

    global urdf 

    urdf = URDF()

    urdf.Save_Start_Tag(f)

    global links

    links = []

def Start_Model(modelName,pos,static):

    global model 

    model = MODEL(modelName,pos)

    model.Save_Start_Tag(f)

    if static:
        s = STATIC_SDF()
        s.Save(f)