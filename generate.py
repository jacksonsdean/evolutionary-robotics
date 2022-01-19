import pyrosim.pyrosim as pyrosim
pyrosim.Start_SDF("boxes.sdf")
x, y, z = 0, 0, 0.5
length, width, height = 1, 1, 1

for outer_idx_x in range(6):
    for outer_idx_y in range(6):
        # towers
        length, width, height = 1, 1, 1
        for box_idx in range(10):
            pyrosim.Send_Cube(name="Box", pos=[x, y, z], size=[
                              length, width, height])
            z += 1
            length = .90 * length
            width = .90 * width
            height = .90 * height
        z = 0.5
        y += 1
    x += 1
    y = 0

pyrosim.End()
