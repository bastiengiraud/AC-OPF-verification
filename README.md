# AC Verification Project

This repository contains all the code for the AC Verification project, which includes two main components:

- **MinMax:** The core logic for training the neural networks.
- **Verification:** The core logic for the verification.

## Getting Started

Under MinMax/Models/best_models you can find the weights of the trained neural networks. The files that end with .._final are the trained neural networks from the 118 bus system from the HPC.

Under MinMax/scripts/comparison you can see some tables with some comparisons.

Under verification, you can see a table with the full verification. If you run lirpa_verification.py, you do the verification with CROWN. 

Under verification/alpha-beta-CROWN/complete-verifier you can find an initial set up for ab-crown verification.

Under verification/alpha-beta-CROWN/complete-verifier/acopf you can find the vnnlib files for the ab-crown verification.

Under verification/support you can find the scripts to generate the vnnlib files.