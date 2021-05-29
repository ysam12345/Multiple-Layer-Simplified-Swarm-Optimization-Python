import numpy as np
import logging
from typing import List
from sso import SSO


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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
        [0, 100],
        [0, 100]
    ]

    
    sso = SSO( fit_function=fit_function_2, 
                edge_function = edge_function,
                variable_range = variable_range,
                sol_num=20, var_num=2, generations=1000,
                cg=0.3, cp=0.5, cw=0.6,
                defult_solution_value=-1*np.inf)
    sso.train()