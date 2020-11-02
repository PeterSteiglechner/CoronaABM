import numpy as np
import scipy.stats as stats

RISKGROUPS = [0, 1, 2]      # Group: 0 = children (risk-free), 1 = adults, (small risk), 2 = high-risk group.
FRACTIONS_RISKGROUPS = [0.2, 0.5, 0.3]      # Fractions of risk groups

"""
Parmaeters for Corona
"""
# AGENT_INFECTIOUSNESS = Distribution from which we draw the infectiousness of an symptomatic individual
#                           within the infectiousness period.
# Educated Guess
# this needs to be adjusted such that we get a mean R-value in the range of realistic values.
AGENT_INFECTIOUSNESS = stats.beta(1,4)
#AGENT_INFECTIOUSNESS = stats.beta(1,3)
FACTOR_INFECTIOUSNESS = 1
#
# P_SYMPTOMATIC = Probability of each group that, if infected with the virus how likely symptoms occur.
# Educated Guess
P_SYMPTOMATIC = [0.1, 0.5, 0.8]

# INCUBATION_TIME_DIST = Distribution of Incubation Periods
# estimated by
# Lauer, et. al (2020) "The Incubation Period of Coronavirus Disease 2019 (COVID-19)
#			From Publicly Reported Confirmed Cases: Estimation and Application"
#			(Table in Appendix 2)
INCUBATION_TIME_DIST = stats.gamma(5.807, 0.948)
# Γ(α,λ)=Γ(2.810,0.419) # https://www.medrxiv.org/content/10.1101/2020.10.20.20216143v1


# CASE_FATALITY_RATE = Probability that given a person has symptoms, the Covid infection leads to her death
#                       for each risk group
# Educated Guess
CASE_FATALITY_RATE = [0, 0.005, 0.05]

# TIME_ONSET_TO_DEATH = Distribution of time between onset of symptoms and time of death.
# Educated Guess
TIME_ONSET_TO_DEATH = stats.gamma(10,2)

# RECOVERY_TIME_SYMPTOMATIC = Distribution of time between onset of symptoms and recovery of non-fatal cases.
# Educated Guess, see
RECOVERY_TIME_SYMPTOMATIC = stats.norm(7,2)

# RECOVERY_TIME_ASYMPTOMATIC = Distribution of time between onset of symptoms and recovery of non-fatal cases.
# Educated Guess, see
RECOVERY_TIME_ASYMPTOMATIC = RECOVERY_TIME_SYMPTOMATIC

##### INFECTIOUSNESS PROFILE:
# for asymptomatic compared to a Symotmatic Case.
INFECTIOUSNESS_ASYMPTOMATIC = 0.2
# INFECTIOUSNESS AT TIME OF RECOVERY
#INFECTIOUSNESS_START = 1/2
#INFECTIOUSNESS_RECOVERY = 1/100
#START_INFECTION_BEFORE_SYMPTOM = 2
#I_MAX = 1

#R0_CORRECTION = 2.5/12

# He 2020 after correction (data taken from correction)
G = stats.gamma(97.18750, scale=1 / 3.71875, loc=-25.62500)