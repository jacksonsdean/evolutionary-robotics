from pyrosim.commonFunctions import Save_Whitespace

class MATERIAL: 

    def __init__(self, name="Cyan", color_rgba=[0, 1, 1, 1]):
        self.depth  = 3

        self.string1 = f'<material name="{name}">'

        self.string2 = f'    <color rgba="{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}"/>'

        self.string3 = f'</material>'

    def Save(self,f):

        Save_Whitespace(self.depth,f)

        f.write( self.string1 + '\n' )

        Save_Whitespace(self.depth,f)

        f.write( self.string2 + '\n' )

        Save_Whitespace(self.depth,f)

        f.write( self.string3 + '\n' )
