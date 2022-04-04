from pyrosim.material import MATERIAL

from pyrosim.commonFunctions import Save_Whitespace

class VISUAL_URDF: 

    def __init__(self,origin,geometry, color_name="Cyan", color_rgba=[0, 1, 1, 1]):

        self.origin = origin

        self.geometry = geometry 

        self.material = MATERIAL(color_name, color_rgba)

        self.depth = 2

    def Save(self,f):

        self.Save_Start_Tag(f)

        self.Save_Elements(f)

        self.Save_End_Tag(f)

# ------------------ Private methods ------------------

    def Save_Start_Tag(self,f):

        Save_Whitespace(self.depth,f)

        f.write('<visual>\n')

    def Save_Elements(self,f):

        self.origin.Save(f)

        self.geometry.Save(f)

        self.material.Save(f)

    def Save_End_Tag(self,f):

        Save_Whitespace(self.depth,f)

        f.write('</visual>\n')
