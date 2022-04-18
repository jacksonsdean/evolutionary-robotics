from pyrosim.commonFunctions import Save_Whitespace

class STATIC_SDF: 

    def __init__(self):
        self.depth = 3
        
        self.static = "<static>true</static>"
        

    def Save(self,f):

        self.Save_Start_Tag(f)

        self.Save_Elements(f)

        self.Save_End_Tag(f)

# ------------------ Private methods ------------------

    def Save_Start_Tag(self,f):

        Save_Whitespace(self.depth,f)


    def Save_Elements(self,f):

        f.write(self.static + '\n')

    def Save_End_Tag(self,f):

        Save_Whitespace(self.depth,f)
