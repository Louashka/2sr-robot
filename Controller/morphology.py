import pyomo.environ as pe
import pyomo.opt as po
from Model import manipulandum, global_var as gv

class ShapeFit:
    def __init__(self, delta_x1: float, delta_y1: float, delta_x2:float, delta_y2:float, theta1:float, theta2:float) -> None:
        self.delta_x1 = delta_x1
        self.delta_y1 = delta_y1

        self.delta_x2 = delta_x2
        self.delta_y2 = delta_y2

        self.theta1 = theta1
        self.theta2 = theta2

        self.__buildModel()

    def __buildModel(self):

        self.model = pe.ConcreteModel()
        self.solver = po.SolverFactory('ipopt')

        self.model.delta_x1 = pe.Param(initialize=self.delta_x1)
        self.model.delta_y1 = pe.Param(initialize=self.delta_y1)

        self.model.delta_x2 = pe.Param(initialize=self.delta_x2)
        self.model.delta_y2 = pe.Param(initialize=self.delta_y2)

        self.model.theta1 = pe.Param(initialize=self.theta1)
        self.model.theta2 = pe.Param(initialize=self.theta2)

        self.model.k1 = pe.Var(domain=pe.Reals, bounds = (-pe.pi/(2*gv.L_VSS), pe.pi/(2*gv.L_VSS)))
        self.model.k2 = pe.Var(domain=pe.Reals, bounds = (-pe.pi/(2*gv.L_VSS), pe.pi/(2*gv.L_VSS)))
        self.model.theta0 = pe.Var(domain=pe.Reals, bounds = (-pe.pi, pe.pi))

         # Define the objective
        @self.model.Objective()
        def __objRule(m):
            return 
        
        @self.model.Constraint()
        def __k1_x(m):
            return m.k1 == (pe.sin(m.theta0) - pe.sin(m.theta1)) / m.delta_x1
        
        @self.model.Constraint()
        def __k1_y(m):
            return m.k1 == (-pe.cos(m.theta0) + pe.cos(m.theta1)) / m.delta_y1
        
        @self.model.Constraint()
        def __k2_x(m):
            return m.k2 == (pe.sin(m.theta2) - pe.sin(m.theta0)) / m.delta_x2
        
        @self.model.Constraint()
        def __k2_y(m):
            return m.k2 == (-pe.cos(m.theta2) + pe.cos(m.theta0)) / m.delta_y2