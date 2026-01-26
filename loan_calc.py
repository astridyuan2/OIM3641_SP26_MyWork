python
def calculate_loan_payment(interest_rate_annual, term_years, present_value):
    payments_per_year = 12

    if interest_rate_annual == 0:
        total_payments = term_years * payments_per_year
        if total_payments == 0:
            return float('nan')
        return present_value / total_payments

    periodic_interest_rate = interest_rate_annual / payments_per_year
    total_payments = term_years * payments_per_year

    pow_factor = (1 + periodic_interest_rate)**total_payments

    numerator = periodic_interest_rate * pow_factor
    denominator = pow_factor - 1

    if denominator == 0:
        return float('nan')

    payment = present_value * (numerator / denominator)

    return payment