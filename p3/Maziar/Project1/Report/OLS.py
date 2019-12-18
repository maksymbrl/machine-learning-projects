def OLS(X, expected_value):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(expected_value)
    return beta