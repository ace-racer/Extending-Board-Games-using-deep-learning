## Set up keras/TF environment in school machines
## Login as administrator
User name: .\issuser Password: Pass1-iss

### Install anaconda, if not there
### Install git, if not there

### Open conda prompt as an admin user
1. List all environments:: conda env list

If the required environment does not already exist, then create it using the below steps:
1. Create the environment: conda create --name chess [refer](https://conda.io/docs/user-guide/tasks/manage-environments.html)
2. activate chess
3. conda install scikit-learn
4. conda install -c anaconda keras-gpu 
5. conda install -c anaconda numpy 
6. conda install -c conda-forge opencv 
7. conda install -c conda-forge matplotlib 
8. Clone the repo and download the data to folder in D drive
9. Check how to enable keras in the jupyter note book runtime (Stackoverflow link) - TODO
10. Run jupyter using the runtime to also run keras
