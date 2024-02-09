def calculateSapsiiScore(age, heartrate, meanbp, temperature, gcs, vent, p2f, bun, urine, sodium, labresult_potassium, labresult_bicarbonate, bilirubin, wbc, metastaticcancer, lymphoma, leukemia, aids):
    score = 0

    if (age < 40):
        score += 0
    elif (age < 60):
        score += 7
    elif (age < 70):
        score += 12
    elif (age < 75):
        score += 15
    elif (age < 80):
        score += 16
    else:
        score += 18

    if (heartrate < 40):
        score += 11
    elif (heartrate < 70):
        score += 2
    elif (heartrate < 120):
        score += 0
    elif (heartrate < 160):
        score += 4
    else:
        score += 7

    if (meanbp < 70):
        score += 13
    elif (meanbp < 100):
        score += 5
    elif (meanbp < 200):
        score += 0
    else:
        score += 2

    if (temperature > 39):
        score += 3
    else:
        score += 0

    if (gcs > 13):
        score += 0
    elif (gcs > 10):
        score += 5
    elif (gcs > 8):
        score += 7
    elif (gcs > 5):
        score += 13
    else:
        score += 26

    if((vent == 0)):
        score += 0
    elif ((vent == 1) & (p2f < 100)):
        score += 11
    elif ((vent == 1) & (p2f < 200)):
        score += 9
    else:
        score += 6

    if (bun < 28):
        score += 0
    elif (bun < 84):
        score += 6
    else:
        score += 10

    if (urine == -1):
        score += 0
    elif (urine < 500):
        score += 11
    elif (urine < 1000):
        score += 4
    else:
        score += 0

    if (sodium < 125):
        score += 5
    elif (sodium < 145):
        score += 0
    else:
        score += 1

    if (labresult_potassium < 3):
        score += 3
    elif (labresult_potassium < 5):
        score += 0
    else:
        score += 3

    if (labresult_bicarbonate < 15):
        score += 6
    elif (labresult_bicarbonate < 20):
        score += 3
    else:
        score += 0

    if (bilirubin < 4):
        score += 0
    elif (bilirubin < 6):
        score += 4
    else:
        score += 9

    if (wbc < 1):
        score += 12
    elif (wbc < 20):
        score += 0
    else:
        score += 3

    if (metastaticcancer == 1):
        score += 9

    if (lymphoma == 1):
        score += 10

    if (leukemia == 1):
        score += 10

    if (aids == 1):
        score += 17

    return score


def calculateSapsiiMortalityRisk(saps_ii):
    import numpy as np
    logit = (-7.7631 + (0.0737 * saps_ii)+ (0.9971 * np.log(saps_ii + 1)))
    saps_ii_risk = np.exp(logit)/(1 + np.exp(logit))
    return saps_ii_risk
