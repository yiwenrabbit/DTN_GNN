edge_max_cpu = 3e9

class Edge():
    def __init__(self, location, edge_id):
        self.location = location
        self.edge_id = edge_id          #set of digital twins
        self.CPU = edge_max_cpu
        self.current_CPU = edge_max_cpu

    def computing_power_cost(self, amount=None):
        if amount > self.current_CPU:
            print('Erro: Exceed the available CPU')
        else:
            self.current_CPU -= amount


