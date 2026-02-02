python
import math

def calculate_loan_payment(interest: float, term: int, present_value: float) -> float:
    """
    Calculates the monthly loan payment.

    Args:
        interest (float): The annual interest rate (e.g., 5 for 5%).
        term (int): The loan term in years.
        present_value (float): The principal loan amount (present value).

    Returns:
        float: The calculated monthly loan payment.
    """
    # Convert annual interest rate to monthly decimal rate
    monthly_interest_rate = (interest / 100) / 12

    # Calculate total number of payments (in months)
    num_payments = term * 12

    if monthly_interest_rate == 0:
        # If interest rate is 0, payment is simply present_value / num_payments
        if num_payments == 0:
            return 0.0  # Avoid division by zero if term is 0
        return present_value / num_payments
    else:
        # Apply the standard loan payment formula
        # P = L [ i(1 + i)^n ] / [ (1 + i)^n – 1]
        numerator = monthly_interest_rate * math.pow(1 + monthly_interest_rate, num_payments)
        denominator = math.pow(1 + monthly_interest_rate, num_payments) - 1
        
        if denominator == 0:
            # This case should ideally not happen for non-zero interest
            # unless num_payments is 0, which is handled above.
            # However, for robustness, if for some reason it's zero,
            # it implies no payments are due or an invalid state.
            return float('inf') 
            
        payment = present_value * (numerator / denominator)
        return payment