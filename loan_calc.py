python
import math

def calculate_loan_payment(interest: float, term: int, present_value: float) -> float:
    """
    Calculates the monthly payment for a loan.

    Args:
        interest (float): The annual interest rate (e.g., 0.05 for 5%).
        term (int): The loan term in years.
        present_value (float): The principal loan amount.

    Returns:
        float: The calculated monthly loan payment.
    """
    if interest < 0 or term <= 0 or present_value <= 0:
        raise ValueError("Interest, term, and present value must be positive.")

    monthly_interest_rate = interest / 12
    number_of_payments = term * 12

    if monthly_interest_rate == 0:
        # Simple principal division if interest is 0
        payment = present_value / number_of_payments
    else:
        # Standard loan payment formula
        numerator = monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments
        denominator = (1 + monthly_interest_rate)**number_of_payments - 1
        payment = present_value * (numerator / denominator)

    return payment