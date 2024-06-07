def calculateMccF1(x, y):
    import sys
    import os

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # import R's "base" package
    mccf1 = importr('mccf1')

    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")
    p = robjects.FloatVector(x)
    t = robjects.FloatVector(y)
    calculateMccf1 = robjects.r['mccf1']
    summary = robjects.r['summary']
    out = summary(calculateMccf1(t, p), 50)[0][0]
    sys.stdout = old_stdout # reset old stdout
    return out


def getOptimalF1Score(y_true, y_score):

    from numpy import arange
    from numpy import argmax
    from sklearn.metrics import f1_score

    thresholds = arange(0, 1, 0.001)
    scores = [f1_score(y_true, (y_score >= t).astype('int')) for t in thresholds]
    ix = argmax(scores)
    return thresholds[ix], scores[ix]