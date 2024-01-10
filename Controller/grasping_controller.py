from Model import manipulandum
import pyomo.environ as pe
import pyomo.opt as po
import numpy as np

class Grasp:
    def __init__(self, obj: manipulandum.Shape, target_pose: list) -> None:
        self.obj = obj
        self.target_pose = target_pose

        self.cpn = 3 # Number of contact points between a robot and manipulandum

        self.__buildModel()

    def __buildModel(self):

        self.model = pe.ConcreteModel()
        self.solver = po.SolverFactory('ipopt')

        # Define sets
        self.model.dof = pe.RangeSet(1, 3) 
        self.model.cpn = pe.RangeSet(1, self.cpn)
        self.model.fn = pe.RangeSet(1, 2 * self.cpn) # Number of force values
        self.model.links = pe.RangeSet(1, self.cpn-1)
        self.model.m = pe.RangeSet(1, self.obj.m)
        self.model.coef_n = pe.RangeSet(1, 2) 

        # Define parameters
        def __getNeighbours(m, i, j):
            if i == j:
                return 1
            elif j == i + 1 or i == self.cpn and j == 1:
                return -1
            else:
                return 0
            
        self.model.weights = pe.Param(self.model.coef_n, initialize={1: 0.5,  2: 0.5})
        self.model.coeffs = pe.Param(self.model.coef_n, initialize={1: 0.196, 2: 0.981})
        self.model.D = pe.Param(self.model.cpn, self.model.cpn, initialize=__getNeighbours)      
        self.model.q_d = pe.Param(self.model.dof, initialize=dict(zip(self.model.dof, self.target_pose)))
        self.model.q = pe.Param(self.model.dof, initialize=dict(zip(self.model.dof, self.obj.pose)))
        self.model.dt = pe.Param(initialize=0.01)

        # Define variables

        # Coordinates of contact points
        self.model.cp_x = pe.Var(self.model.cpn, domain=pe.Reals)
        self.model.cp_y = pe.Var(self.model.cpn, domain=pe.Reals)
        self.model.cp_theta = pe.Var(self.model.cpn, domain=pe.Reals)

        # Contact points, s[i] is in [0, 1)
        self.model.s = pe.Var(self.model.cpn, domain=pe.NonNegativeReals, bounds=(0, 1))
        # Optimal force to be applied at the contact points
        self.model.force = pe.Var(self.model.fn, domain=pe.NonNegativeReals)
        # Expected manipulandum's configuration after grasping and motion
        self.model.q_new = pe.Var(self.model.dof, domain=pe.Reals)

        # Define the objective
        @self.model.Objective()
        def __objRule(m):

            tracking_error = sum((m.q_d[i] - m.q_new[i])**2 for i in m.dof)
            # contact_forces = sum(m.force[i]**2 for i in m.fn)

            # return m.weights[1] * tracking_error + m.weights[2] * contact_forces
            return tracking_error

        # Define constraints

        @self.model.Constraint()
        def __constraintPoseX(m):
            return m.q_new[1] == m.q[1] + sum(m.coeffs[1] * (m.force[i*2-1] - m.force[i*2]) * pe.cos(m.q[3] + m.cp_theta[i]) - 
                    m.coeffs[2] * (m.force[i*2-1] + m.force[i*2]) * pe.sin(m.q[3] + m.cp_theta[i]) for i in m.cpn) * m.dt

        @self.model.Constraint()
        def __constraintPoseY(m):
            return m.q_new[2] == m.q[2] + sum(m.coeffs[1] * (m.force[i*2] - m.force[i*2-1]) * pe.sin(m.q[3] + m.cp_theta[i]) + 
                    m.coeffs[2] * (m.force[i*2-1] + m.force[i*2]) * pe.cos(m.q[3] + m.cp_theta[i]) for i in m.cpn) * m.dt

        @self.model.Constraint()
        def __constraintPoseTheta(m):
            return m.q_new[3] == m.q[3] + sum(m.coeffs[1] * (m.force[i*2-1] - m.force[i*2]) * (m.cp_x[i] * pe.sin(m.cp_theta[i]) - 
                            m.cp_y[i] * pe.cos(m.cp_theta[i])) + m.coeffs[2] * (m.force[i*2-1] + m.force[i*2]) * (m.cp_x[i] * 
                            pe.cos(m.cp_theta[i]) + m.cp_y[i] * pe.sin(m.cp_theta[i])) for i in m.cpn) * m.dt

        @self.model.Constraint(self.model.cpn)
        def __constraintContactPointX(m, i):
            return m.cp_x[i] == self.__getPoint(m.s[i])[0]

        @self.model.Constraint(self.model.cpn)
        def __constraintContactPointY(m, i):
            return m.cp_y[i] == self.__getPoint(m.s[i])[1]

        @self.model.Constraint(self.model.cpn)
        def __constraintContactPointTheta(m, i):
            return m.cp_theta[i] == self.__getDirec(m.s[i])

        @self.model.Constraint(self.model.links)
        def __constraintLinks(m, i):
            lhs = sum(m.D[i, j] * m.s[j] for j in m.cpn)
            rhs = 0.077 # To be calculated!
            return lhs == rhs

    def __getPoint(self, s):
        x = 0
        y = 0

        for h in self.model.m:
            arg = 2 * h * s * np.pi
            exp = [pe.cos(arg), pe.sin(arg)]

            coef = self.obj.coeffs[h-1,:]
            x += coef[0] * exp[0] + coef[1] * exp[1]
            y += coef[2] * exp[0] + coef[3] * exp[1]

        return [x, y]

    def __getTangent(self, s):
        dx = 0
        dy = 0

        for h in self.model.m:
            c = 2 * h * np.pi
            arg = c * s
            exp = [-c * pe.sin(arg),  c * pe.cos(arg)]

            coef = self.obj.coeffs[h-1,:]
            dx += coef[0] * exp[0] + coef[1] * exp[1]
            dy += coef[2] * exp[0] + coef[3] * exp[1]

        theta = pe.atan(dy/dx)

        return theta

    def __getDirec(self, s):
        theta = self.__getTangent(s)

        p1 = self.__getPoint(s)
        p2 = [p1[0] + pe.cos(theta), p1[1] + pe.sin(theta)]
        p3 = [p1[0] + pe.cos(theta + np.pi/2), p1[1] + pe.sin(theta + np.pi/2)]

        cross_prod1 = (p1[0] - p2[0]) * (- p2[1]) - (p1[1] - p2[1]) * (- p2[0])
        cross_prod2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])

        condition = (abs(cross_prod1 * cross_prod2) - cross_prod1 * cross_prod2) / (-2 * cross_prod1 * cross_prod2)
        theta += condition * np.pi

        return theta
    
    def solve(self):
        try:
            self.results = self.solver.solve(self.model)
            return True
        except:
            return False

    def parseResults(self):
        s = [pe.value(self.model.s[key]) for key in self.model.cpn]
        force = [pe.value(self.model.force[key]) for key in self.model.fn]
        q_new = [pe.value(self.model.q_new[key]) for key in self.model.dof]

        return self.results, s, force, q_new




