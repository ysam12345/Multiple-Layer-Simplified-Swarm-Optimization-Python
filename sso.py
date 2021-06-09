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


class SSO(object):
    """
    A class to perform SSO algorithm.
    """

    def __init__(self, fit_function:Callable, edge_function: Callable,
                variable_range: List[List[int]],
                sol_num:int, var_num:int, generations:int,
                fixed_variables: List[float]=[],
                cg:float=0.4, cp:float=0.7, cw:float=0.9,
                default_solution_value:Union[float, int]=0,
                show_progreess=True):
        """
        Parameters
        ----------
        fit_function : list[fuction]
            Fitness function.
        edge_function : function
            Function to accept or reject a list of variable.
        variable_range : list[list[int]]
            The range of each variable should be in.
            variable_range[0] = [1, 3] indicate that x should be in range(1, 3)
        fixed_variables : list[float]
            The variable values that should be fixed when generate random variables.
            fixed_variables[0] = None indicate that x is not fixed.
            fixed_variables[0] = 0.5 indicate that x is fixed as value 0.5.
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
        show_progreess: bool, optional
            Wether to show tqdm progress bar. (default is True)
        """
        self.fit_fucntion = fit_function
        self.edge_function = edge_function
        self.variable_range = variable_range
        self.sol_num = sol_num
        self.var_num = var_num
        self.generations = generations
        self.fixed_variables = fixed_variables if fixed_variables else self.var_num * [None]
        self.cg = cg
        self.cp = cp
        self.cw = cw
        
        self.global_best_sol_idx = 0
        self.particles, self.particles_best = self.get_init_particles()
        self.solutions, self.solutions_best = self.get_init_solutions(default_value=default_solution_value)
        self.show_progreess = show_progreess

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
    
    def get_rand_variables(self) -> np.array:
        """Generate random variables by variable range and check if it's accept by the edge function.
        If it's reject by the edge function, re-generate it until it's acceptable.

        Returns:
            Generated random variables.

        """
        accepted = False
        while not accepted:
            variables = copy(self.fixed_variables)
            for i in range(self.var_num):
                if variables[i] is None:
                    variables[i] = self.get_a_rand_variable(i)
            accepted = self.edge_function(variables)
        return np.array(variables)

    def get_init_solutions(self, default_value:int = 0) -> (np.array, np.array):
        """Generate initial solutions and return.

        Returns:
            The initialized solutions and solutions_best.

        """
        solutions = np.full((self.sol_num), default_value)
        solutions_best = np.full((self.sol_num), default_value)
        return solutions, solutions_best

    def train(self):
        if self.show_progreess:
            logging.info(f'Start training SSO')
            progress_bar = tqdm(range(1, self.generations+1))
            r = progress_bar
        else:
            r = range(1, self.generations+1)
        for generation in r:
            for sol_idx in range(self.sol_num):
                # update variables of each solution agent
                self.update_variables(sol_idx)
                #calucation particles best and global best
                self.evaluate_particles_best(sol_idx)
                self.evaluate_global_best(sol_idx)
            if self.show_progreess:
                progress_bar.set_description(f"Generation: {generation}, Best Variable: {self.particles_best[self.global_best_sol_idx]}, Best Solution: {self.solutions_best[self.global_best_sol_idx]}")
        return self.particles_best[self.global_best_sol_idx]

    def update_variables(self, sol_idx:int):
        """Update variables of a solution agent.

        Args:
            sol_idx(int): The index of solution agent.

        Returns:
            The initialized solutions and solutions_best.

        """
        rand = random()

        if rand < self.cg: 
            self.particles[sol_idx] = np.copy(self.particles_best[self.global_best_sol_idx])
        elif rand < self.cp: 
            self.particles[sol_idx] = np.copy(self.particles_best[sol_idx])
        elif rand > self.cw:
            self.particles[sol_idx]  = self.get_rand_variables()

    
    def evaluate_particles_best(self, sol_idx:int):
        self.solutions[sol_idx] = self.fit_fucntion(self.particles[sol_idx])
        # find max
        if self.solutions[sol_idx] > self.solutions_best[sol_idx]:
            self.solutions_best[sol_idx] = np.copy(self.solutions[sol_idx])
            self.particles_best[sol_idx] = np.copy(self.particles[sol_idx])
                

    def evaluate_global_best(self, sol_idx:int):
        if self.solutions_best[sol_idx] > self.solutions_best[self.global_best_sol_idx]: 
            self.global_best_sol_idx = sol_idx
    
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

    
    sso = SSO( fit_function=fit_function_1, 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=1000,
                cg=0.3, cp=0.5, cw=0.6,
                default_solution_value=-1*np.inf)
    sso.train()