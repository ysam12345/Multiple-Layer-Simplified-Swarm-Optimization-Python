import numpy as np
from tqdm import tqdm
from random import random
from typing import Callable, List, Union
from copy import copy
import logging

from sso import SSO


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class MSSO(object):
    """
    A class to perform multi-layer SSO algorithm.
    """

    def __init__(self, layers:int, leader_id:int, epochs:int,
                fit_functions:List[Callable], edge_function: Callable,
                variable_range: List[List[int]],
                sol_num:int, var_num:int, generations:int,
                cg:float=0.4, cp:float=0.7, cw:float=0.9,
                default_solution_value:Union[float, int]=0):
        """
        Parameters
        ----------
        layers : int
            Indicate how many layers and fit functions are used.
        leader_id : int
            Indicate the global best should select by which fit funciton.
        epochs: int
            Indicate the training epoch of MSSO.
        fit_functions : list[fuction]
            List of functions of fit_function.
        edge_function : function
            Function to accept or reject a list of variable.
        variable_range : list[list[int]]
            The range of each variable should be in.
            variable_range[0] = [1, 3] indicate that x should be in range(1, 3)/
        sol_num : int
            Inidcate number of solution agent.
        var_num : int
            Inidcate number of variables in each solution agent.
        generations : int
            Indicate how many generations should run.
        cg : float, optional
            The SSO update parameter. (default is 0.4)
        cp : float, optional
            The SSO update parameter. (default is 0.7)
        cw : float, optional
            The SSO update parameter. (default is 0.9)
        default_solution_value: int or float, optional
            The defult value of solutions. (default is 0)
        """
        self.layers = layers
        self.leader_id = leader_id
        self.epochs = epochs
        self.fit_fucntions = fit_functions
        self.edge_function = edge_function
        self.variable_range = variable_range
        self.sol_num = sol_num
        self.var_num = var_num
        self.generations = generations
        self.cg = cg
        self.cp = cp
        self.cw = cw
        self.default_solution_value = default_solution_value
        
        self.best_variables = self.get_rand_variables()
        sso_2 = SSO(fit_function=self.fit_fucntions[1], 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=1000,
                #fixed_variables=[None, variables[1]],
                fixed_variables=[None, None],
                cg=0.3, cp=0.5, cw=0.6,
                default_solution_value=-1*np.inf,
                show_progreess=False)
        self.best_variables = sso_2.train()

        self.best_fitness = self.fit_fucntions[0](self.best_variables)

    def get_a_rand_variable(self, var_idx:int) -> float:
        """Generate a random variable by variable range and return.

        Args:
            var_idx(int): The index of variable.

        Returns:
            Generated a random variable.

        """
        return (self.variable_range[var_idx][1] - self.variable_range[var_idx][0]) * random() + self.variable_range[var_idx][0]
    
    def get_rand_variables(self) -> np.array:
        """Generate random variables by variable range and check if it's accept by the edge function.
        If it's reject by the edge function, re-generate it until it's acceptable.

        Returns:
            Generated random variables.

        """
        accepted = False
        while not accepted:
            variables = []
            for i in range(self.var_num):
                variables.append(self.get_a_rand_variable(i))
            accepted = self.edge_function(variables)
        return np.array(variables)

    def train(self):
        logging.info(f'Start training MSSO')
        progress_bar = tqdm(range(1, self.epochs+1))
        for epoch in progress_bar:
            variables = copy(self.best_variables)
            sso_1 = SSO(fit_function=self.fit_fucntions[0], 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=1000,
                fixed_variables=[None, variables[1]],
                cg=0.3, cp=0.5, cw=0.6,
                default_solution_value=-1*np.inf,
                show_progreess=False)
            ans_1 = sso_1.train()
            sso_2 = SSO(fit_function=self.fit_fucntions[1], 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=1000,
                fixed_variables=[ans_1[0], None],
                cg=0.3, cp=0.5, cw=0.6,
                default_solution_value=-1*np.inf,
                show_progreess=False)
            ans_2 = sso_2.train()
            fitness = self.fit_fucntions[0](ans_2)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_variables = ans_2
            progress_bar.set_description(f"Epoch: {epoch}, Best Variable: {self.best_variables}, Best Fitness: {self.best_fitness}")

if __name__  == '__main__':
    def fit_function_1(variables:List[float]) -> float:
        # x + 3y
        return variables[0] + 3 * variables[1]

    def fit_function_2(variables:List[float]) -> float:
        # -y
        return -1 * variables[1]

    def edge_function(variables:List[float]) -> bool:
        """Function to determine if variables are in acceptable numerical range of the problem.
        -x + y <= 3
        x + 2y <= 12
        4x - y <= 12
        x >= 0, y >= 0

        Args:
            variables (list[float]): The variables, variables[0] indicate x and variables[1] indicate y.

        Returns:
            bool: The return value. True for accept, False for reject.
        """
        return -1 * variables[0] + variables[1] <= 3 \
            and variables[0] + 2 * variables[1] <= 12 \
            and 4 * variables[0] - variables[1] <= 12 \
            and variables[0] >= 0 \
            and variables[1] >= 0 
    
    # x should be in range(0, 100)
    # y should be in range(0, 100)
    variable_range = [
        [0, 10],
        [0, 10]
    ]

    
    msso = MSSO(layers=1, leader_id=1, epochs=30000,
                fit_functions=[fit_function_1, fit_function_2],
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=200,
                cg=0.3, cp=0.5, cw=0.6,
                default_solution_value=-1*np.inf)
    msso.train()