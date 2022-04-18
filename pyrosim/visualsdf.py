from pyrosim.commonFunctions import Save_Whitespace
from pyrosim.material import MATERIAL

class VISUAL_SDF: 

    def __init__(self,geometry, color_name, color_rgba):

        self.geometry = geometry 

        self.material = MATERIAL(color_name, color_rgba)
        
        self.depth = 3
        
        

    def Save(self,f):

        self.Save_Start_Tag(f)

        self.Save_Elements(f)

        self.Save_End_Tag(f)

# ------------------ Private methods ------------------

    def Save_Start_Tag(self,f):

        Save_Whitespace(self.depth,f)

        f.write('<visual>\n')

    def Save_Elements(self,f):

        self.geometry.Save(f)
        
        self.material.Save(f)
        

    def Save_End_Tag(self,f):

        Save_Whitespace(self.depth,f)

        f.write('</visual>\n')
