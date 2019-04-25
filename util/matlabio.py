def getmat(addr, vars_name):
    from scipy.io import loadmat
    content = loadmat(addr, mat_dtype=True)
    vars = []
    for var in vars_name:
        vars.append(content[var])
    return vars
