import pyrosim.pyrosim as pyrosim


x, y, z = 0, 0, 0.5
length, width, height = 1, 1, 1

def Create_World():
    pyrosim.Start_SDF("world.sdf")
    pyrosim.Send_Cube(name="Box", pos=[x-2., y+2., z], size=[length, width, height])
    pyrosim.End()
    
def Create_Robot():
    pyrosim.Start_URDF("body.urdf")
    pyrosim.Send_Cube(name="Torso", pos=[x, y, z], size=[length, width, height])
    pyrosim.End()


Create_World()
Create_Robot()
