import numpy as np
from tqdm import tqdm
from random import random
from typing import Callable, List, Union
from copy import copy
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class MSSO(object):
    """
    A class to perform multi-layer SSO algorithm.
    """

    def __init__(self, layers:int, leader_id:int,
                fit_functions:List[Callable], edge_function: Callable,
                variable_range: List[List[int]],
                sol_num:int, var_num:int, generations:int,
                cg:float=0.4, cp:float=0.7, cw:float=0.9,
                defult_solution_value:Union[float, int]=0):
        """
        Parameters
        ----------
        layers : int
            Indicate how many layers and fit functions are used.
        leader_id : int
            Indicate the global best should select by which fit funciton.
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
        defult_solution_value: int or float, optional
            The defult value of solutions. (default is 0)
        """
        self.layers = layers
        self.leader_id = leader_id
        self.fit_fucntions = fit_functions
        self.edge_function = edge_function
        self.variable_range = variable_range
        self.sol_num = sol_num
        self.var_num = var_num
        self.generations = generations
        self.cg = cg
        self.cp = cp
        self.cw = cw
        self.global_best_sol_indexs = 0

        self.particles, self.particles_best = self.get_init_particles()
        self.solutions, self.solutions_best = self.get_init_solutions(default_value=defult_solution_value)

    def get_init_particles(self) -> (np.array, np.array):
        """Generate initial particles and return.

        Returns:
            The initialized particales and particles_best.

        """
        particles = np.zeros([self.sol_num, self.var_num], dtype = float)
        particles_best = np.zeros([self.sol_num, self.var_num], dtype = float) 
        for sol_idx in range(self.sol_num):
            rand_variables = self.get_rand_variables()
            particles[sol_idx] = rand_variables
            particles_best[sol_idx] = rand_variables
        return particles, particles_best

    def get_a_rand_variable(self, var_idx:int) -> float:
        """Generate a random variable by variable range and return.

        Args:
            var_idx(int): The index of variable.

        Returns:
            Generated a random variable.

        """
        return (self.variable_range[var_idx][1] - self.variable_range[var_idx][0]) * random() + self.variable_range[var_idx][0]
    
    def get_rand_variables(self, fix_variables:list=[]) -> np.array:
        """Generate random variables by variable range and check if it's accept by the edge function.
        If it's reject by the edge function, re-generate it until it's acceptable.

        Args:
            fix_variables(list): The variable values that should be fixed when generate random variables.

        Returns:
            Generated random variables.

        """
        # let fix_variables full of None if it's a empty list.
        if not fix_variables:
            fix_variables = self.var_num * [None]

        accepted = False
        while not accepted:
            variables = copy(fix_variables)
            for i in range(self.var_num):
                if variables[i] is None:
                    variables[i] = self.get_a_rand_variable(i)
            accepted = self.edge_function(variables)
        return np.array(variables)

    def get_init_solutions(self, default_value = 0) -> (np.array, np.array):
        """Generate initial solutions and return.

        Returns:
            The initialized solutions and solutions_best.

        """
        solutions = np.full((self.sol_num), default_value)
        solutions_best = np.full((self.sol_num), default_value)
        return solutions, solutions_best

    def train(self):
        logging.info(f'Start training')
        progress_bar = tqdm(range(1, self.generations+1))
        for generation in progress_bar:
            # train each layer with different leader
            for layer_idx in range(self.layers):
                fix_var_idxs = [i for i in range(self.layers) if i != layer_idx]
                for sol_idx in range(self.sol_num):
                    # update and evaluate particles best for each solution agent
                    self.update_variables(sol_idx, fix_var_idxs=fix_var_idxs)
                    
            # calucation global best after update all variable and particles best
            for sol_idx in range(self.sol_num):
                self.evaluate_particles_best(sol_idx)
                self.evaluate_global_best(sol_idx, leader_id=self.leader_id)
            progress_bar_message = f"Generation: {generation}, Best Var: {[f'{var:.2f}' for var in self.particles_best[self.global_best_sol_indexs]]}, Best Sol: {self.solutions_best[self.global_best_sol_indexs]:.2f}"
            progress_bar.set_description(progress_bar_message)
            #progress_bar.set_description(f"Generation: {generation}, Best Variable: {self.particles_best[self.global_best_sol_index]}, Best Solution: {self.solutions_best[self.global_best_sol_index]}")
            #logging.info(f"Generation: {generation}, Best Variable: {self.particles_best[self.global_best_sol_index]}, Best Solution: {self.solutions_best[self.global_best_sol_index]}")
    
    def update_variables(self, sol_idx:int, fix_var_idxs=[]):
        """Update variables of a solution agent.

        Args:
            sol_idx(int): The index of solution agent.
            fix_var_idxs(list): The index of variables that shouldn't be updated. Default is empty list.

        Returns:
            The initialized solutions and solutions_best.

        """
        rand = random()
        # generate fix variables by fix_var_idxs
        fix_variables = []
        for var_idx in range(self.var_num):
            if var_idx in fix_var_idxs:
                fix_variables.append(self.particles[sol_idx][var_idx])
            else:
                fix_variables.append(None)
        if rand < self.cg: 
            self.particles[sol_idx] = np.copy(self.particles_best[self.global_best_sol_indexs])
        elif rand < self.cp: 
            self.particles[sol_idx] = np.copy(self.particles_best[sol_idx])
        elif rand > self.cw:
            self.particles[sol_idx]  = self.get_rand_variables(fix_variables=fix_variables)

    
    def evaluate_particles_best(self, sol_idx:int):
        for function_id in range(len(self.fit_fucntions)):
            self.solutions[sol_idx] = self.fit_fucntions[function_id](self.particles[sol_idx])
            # find max
            if self.solutions[sol_idx] > self.solutions_best[sol_idx]:
                self.solutions_best[sol_idx] = self.solutions[sol_idx]
                self.particles_best[sol_idx] = np.copy(self.particles[sol_idx])
                

    def evaluate_global_best(self, sol_idx:int, leader_id:int):
        self.solutions[sol_idx] = self.fit_fucntions[leader_id](self.particles[sol_idx])
        if self.solutions[sol_idx] > self.solutions_best[self.global_best_sol_indexs]:  
            self.global_best_sol_indexs = sol_idx
    
if __name__  == '__main__':
    def fit_function_1(variables:List[float]) -> float:
        return -2 * variables[0] + 11 * variables[1]

    def fit_function_2(variables:List[float]) -> float:
        return -1 * variables[0] - 3 * variables[1]

    def edge_function(variables:List[float]) -> bool:
        """Function to determine if variables are in acceptable numerical range of the problem.

        Args:
            variables (list[float]): The variables, variables[0] indicate x and variables[1] indicate y.

        Returns:
            bool: The return value. True for accept, False for reject.
        """
        return variables[0] - 2 * variables[1] <= 4 \
            and 2 * variables[0] - variables[1] <= 24 \
            and 3 * variables[0] + 4 * variables[1] <= 96 \
            and variables[0] + 7 * variables[1] <= 126 \
            and -4 * variables[0] + 5 * variables[1] <= 65 \
            and variables[0] + 4 * variables[1] >= 8 \
            and variables[0] >= 0 \
            and variables[1] >= 0 

    # x should be in range(0, 100)
    # y should be in range(0, 100)
    variable_range = [
        [0, 100],
        [0, 100]
    ]

    
    msso = MSSO(layers=2, leader_id=0,
                fit_functions=[fit_function_1, fit_function_2], 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=100, var_num=2, generations=100000,
                #cg=0.4, cp=0.415, cw=0.420,
                defult_solution_value=-1*np.inf)
    msso.train()