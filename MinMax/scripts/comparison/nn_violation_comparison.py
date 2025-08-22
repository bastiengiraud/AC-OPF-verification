"""
Compare 4 types of NN models:
1. Power NN without w/c penalties
2. Power NN with w/c penalties
3. Voltage NN without w/c penalties
4. Voltage NN with w/c penalties

Compare on a test set of 1000 runs, the average and max violations of:
- Generator limits
- Current/line limits
- Bus voltage limits
- Power balance

To DO:
- train the 4 neural networks, and fix the storing logic
- implement the validation logic to compare the models
- implement the feasibility check function, comparing feasibility of the models

"""