import numpy as np
from scipy.optimize import leastsq

class Sample:
    def __init__(self, file, beta, initialize_values=None, dynamic_parameters_name=None):
        """
        Initialize the Sample class instance.

        Args:
            file (str): Path to the text file containing data.
            beta (list): List of beta values.
            initialize_values (list, optional): List of initial parameter values. Defaults to None.
            dynamic_parameters_name (list, optional): List of dynamic parameter names. Defaults to None.
        """
        data = np.loadtxt(file)
        self.time_minutes = data[:, 0] / 60.
        self.polarization = data[:, 1]

        self.beta = beta
        self.n = 100

        self.initialize_values = initialize_values or []
        self.dynamic_parameters_name = dynamic_parameters_name or []

        self.parameters = {
            "Pe": self.initialize_values[0] if len(self.initialize_values) > 0 else 0.9,
            "TD": self.initialize_values[1] if len(self.initialize_values) > 1 else self.beta[0],
            "TL": self.initialize_values[2] if len(self.initialize_values) > 2 else self.beta[1],
            "alpha": self.initialize_values[3] if len(self.initialize_values) > 3 else self.beta[2],
            "Pcal_2": self.initialize_values[4] if len(self.initialize_values) > 4 else self.polarization[0],
        }

        self.dynamic_parameters = [self.parameters[name] for name in self.dynamic_parameters_name]
        self.r = self.objective_function(self.dynamic_parameters)  # Calculate r value
        self.optimize_parameters()

    def theoretical_value(self, beta):
        """
        Calculate the theoretical polarization values based on the initialized parameters.

        Args:
            beta (list): List of parameter values.

        Returns:
            list: List of calculated polarization values.
        """
        # Get parameter values from beta
        Pe = beta[0]
        TD = beta[1] / 60.
        TL = beta[2] / 60.
        alpha = beta[3]
        beta_value = beta[4]
        Pcal_2 = beta[5]

        # Placeholder values for some variables
        S = 0
        I = 0
        t_s = 0

        n = self.n
        t = self.time_minutes
        Pcal = [0] * int(len(t))  # Initialize an array to store calculated polarization values

        # Calculate theoretical polarization values for each time step
        for i in range(int(len(t))):
            t_cal = (t[0] + i * 20)

            for j in range(n):
                t_cal_2 = (t_cal + j * 20 / n) / 60.
                t_cal_2 = (t_cal_2 * 60.)

                # Calculate the difference between the theoretical and calculated polarization
                dif = (Pe - Pcal_2) / TD - (1 / TL + S + alpha * I * t_s + beta_value * I) * Pcal_2

                # Update the calculated polarization value
                Pcal_2 = Pcal_2 + dif * j * 20 / n / 60.

            if i == 0:
                Pcal[i] = self.polarization[i]
                continue

            Pcal[i] = Pcal_2

        # Check for negative parameter values and reset Pcal if necessary
        if TD < 0 or TL < 0 or alpha < 0 or beta_value < 0:
            Pcal = [0] * int(len(t))

        return Pcal  # Return the calculated polarization values

    def objective_function(self, beta):
        """
        Calculate the objective function for parameter optimization.

        Args:
            beta (list): List of parameter values for optimization.

        Returns:
            float: Value of the objective function.
        """
        r = (self.polarization - self.theoretical_value(beta)) ** 2
        r[9] = 0.
        r[10] = 0.
        r[81] = 0.
        r[82] = 0.
        r[125] = 0.
        r[126] = 0.
        r_sum = sum(r)
        beta_list = list(beta)
        beta_list[1] = beta_list[1] / 60.
        beta_list[2] = beta_list[2] / 60.
        print('TD={:.3f} h, TL={:.3f} h, alpha={:e}, beta={}, r_sum={:.3f}'.format(beta_list[1], beta_list[2], beta_list[3], beta_list[4], r_sum))
        return r_sum
    
    def optimize_parameters(self):
        """
        Optimize the parameters using leastsq.
        """
        betaL, _ = leastsq(self.objective_function, self.dynamic_parameters, maxfev=1000)
        print("Optimized beta:", betaL)
        self.beta = betaL  # Update self.beta with optimized values


def main():
    # パラメータ設定
    file_path = '1540-4080.txt'
    beta_values = [TD_value, TL_value, alpha_value, beta_value]  # これを適切な値で置き換える
    initialize_values = [Pe_initial, TD_initial, TL_initial, alpha_initial, Pcal_2_initial]  # 初期化するパラメータの値
    dynamic_parameters_name = ['TD', 'TL', 'alpha', 'beta']  # ダイナミックパラメータの名前

    # Sampleクラスの初期化
    sample = Sample(file_path, beta_values, initialize_values, dynamic_parameters_name)

    # パラメータ最適化
    sample.optimize_parameters()

    # 最適化後のパラメータを表示
    print("Optimized beta:", sample.beta)

if __name__ == "__main__":
    main()