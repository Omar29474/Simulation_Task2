import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Function to Allow User-Defined Functions via Input
def user_function(*args):
    """
    User-defined function for integration. Replace this with any mathematical function.
    For example, we use a Gaussian function here.
    """
    return np.exp(-np.sum(np.square(args)))  # Example: Gaussian function

# Step 2: Monte Carlo Integration for Multi-Dimensional Functions
def monte_carlo_integration(func, bounds, N, d):
    """
    Monte Carlo integration to estimate the integral of a function in d dimensions.
    
    Parameters:
    - func: Function to integrate, user-defined
    - bounds: List of tuples specifying bounds for each dimension [(a1, b1), (a2, b2), ..., (ad, bd)]
    - N: Number of random samples
    - d: Number of dimensions
    
    Returns:
    - integral_estimate: Estimated value of the integral
    - variance_estimate: Estimated variance of the integral
    """
    # Generate random samples within the bounds
    samples = np.random.rand(N, d)
    for i in range(d):
        samples[:, i] = samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    
    # Evaluate the function at the random samples
    function_values = np.apply_along_axis(func, 1, samples)
    
    # Estimate the integral (mean of the function values * volume of the domain)
    volume = np.prod([b - a for a, b in bounds])  # Calculate the volume of the d-dimensional domain
    integral_estimate = volume * np.mean(function_values)
    
    # Estimate the variance
    variance_estimate = np.var(function_values) / N
    
    return integral_estimate, variance_estimate, function_values

# Step 3: Compute the 95% Confidence Interval
def compute_confidence_interval(integral_estimate, variance_estimate, N, confidence_level=0.95):
    """
    Compute the 95% confidence interval for the estimated integral.
    """
    # Compute standard error
    standard_error = np.sqrt(variance_estimate) / np.sqrt(N)
    
    # Get the z-score for the desired confidence level (for 95%, it's approximately 1.96)
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Compute the confidence interval
    margin_of_error = z_score * standard_error
    lower_bound = integral_estimate - margin_of_error
    upper_bound = integral_estimate + margin_of_error
    
    return lower_bound, upper_bound

# Step 4: Determine the number of samples required for a given precision K
def determine_samples_for_precision(func, bounds, desired_precision, d, confidence_level=0.95):
    """
    Determine the number of samples required to achieve a given precision.
    """
    # Initial estimate with a small number of samples
    N = 1000
    integral_estimate, variance_estimate, _ = monte_carlo_integration(func, bounds, N, d)
    
    # Compute standard error and required N for desired precision
    standard_error = np.sqrt(variance_estimate) / np.sqrt(N)
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    
    required_samples = (z_score * standard_error / desired_precision) ** 2
    return int(np.ceil(required_samples))

# Step 5: Visualize Convergence
def visualize_convergence(func, bounds, N_values, d):
    """
    Visualize the convergence of the integral approximation as N increases.
    """
    estimates = []
    for N in N_values:
        integral_estimate, _, _ = monte_carlo_integration(func, bounds, N, d)
        estimates.append(integral_estimate)
    
    # Plot the convergence
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, estimates, marker='o', label="Estimated Integral")
    plt.xscale("log")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Integral Estimate")
    plt.title("Convergence of Monte Carlo Integration")
    plt.grid(True)
    plt.legend()
    plt.show()

# Step 6: Main Function to Execute the Task
def run_monte_carlo_integration_task():
    """
    Run the Monte Carlo Integration task with user inputs.
    """
    # User input for dimensions and bounds
    d = int(input("Enter the number of dimensions (d): "))
    bounds = []
    for i in range(d):
        a, b = map(float, input(f"Enter the bounds for dimension {i + 1} as [a, b]: ").split(","))
        bounds.append((a, b))
    
    # Ask user for the precision and number of samples
    desired_precision = float(input("Enter the desired precision (K): "))
    N = int(input("Enter the number of random samples (N): "))
    
    # Determine the required number of samples for the desired precision
    required_samples = determine_samples_for_precision(user_function, bounds, desired_precision, d)
    print(f"Required number of samples to achieve precision {desired_precision}: {required_samples}")
    
    # Run the Monte Carlo integration
    integral_estimate, variance_estimate, _ = monte_carlo_integration(user_function, bounds, N, d)
    lower_bound, upper_bound = compute_confidence_interval(integral_estimate, variance_estimate, N)
    
    print(f"Estimated Integral: {integral_estimate}")
    print(f"Variance Estimate: {variance_estimate}")
    print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")
    
    # Visualize the convergence of the estimated integral as N increases
    N_values = np.logspace(1, 6, num=10, dtype=int)
    visualize_convergence(user_function, bounds, N_values, d)

if __name__ == "__main__":
    run_monte_carlo_integration_task()
