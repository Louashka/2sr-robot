import pyomo.environ as pe
import pyomo.opt as po
import numpy as np

class Model():
	def __init__(self, l, l0, th0, p1, p0) -> None:
		self.l = l
		self.l0 = l0
		self.th0 = th0
		self.p1 = p1
		self.p0 = p0

		self.__model = pe.ConcreteModel()
        self.__solver = po.SolverFactory('ipopt')

		self.__build_model()

	def __build_model(self):
		# Define sets

		# Define parameters

		# Define variables

		# Define the objective