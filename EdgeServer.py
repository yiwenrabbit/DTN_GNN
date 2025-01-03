edge_max_cpu = 3e9

class Edge():
   def __init__(self, location, edge_id):
        self.location = location
        self.edge_id = edge_id          #set of digital twins
        self.CPU = edge_max_cpu
        self.current_CPU = edge_max_cpu
