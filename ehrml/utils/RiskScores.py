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


def calculateApacheiiScore(age, severeOrganFailure, typeOfAdmission, rectalTemperatureCelsius, meanArterialPressure, heartRate, respiratoryRate, PaO2, FiO2, PaCO2, arterialPh, serumSodium, serumPotassium, serumCreatinine, renalFailure, hematocrit, wbc, gcs):
    score = 0

    if (age <= 44):
        score += 0
    elif (age < 54):
        score += 2
    elif (age < 64):
        score += 3
    elif (age < 74):
        score += 5
    else:
        score += 6

    if severeOrganFailure:
        if ((typeOfAdmission == 'nonoperative') or (typeOfAdmission == 'emergency postoperative')):
            score += 5
        elif (typeOfAdmission == 'elective postoperative'):
            score += 2

    if (rectalTemperatureCelsius >= 41):
        score += 4
    elif (rectalTemperatureCelsius > 39):
        score += 3
    elif (rectalTemperatureCelsius > 38.5):
        score += 1
    elif (rectalTemperatureCelsius > 36):
        score += 0
    elif (rectalTemperatureCelsius > 34):
        score += 1
    elif (rectalTemperatureCelsius > 32):
        score += 2
    elif (rectalTemperatureCelsius > 30):
        score += 3
    else:
        score += 4

    if (meanArterialPressure > 159):
        score += 4
    elif (meanArterialPressure > 129):
        score += 3
    elif (meanArterialPressure > 109):
        score += 2
    elif (meanArterialPressure > 69):
        score += 0
    elif (meanArterialPressure > 49):
        score += 2
    else:
        score += 4

    if (heartRate > 180):
        score += 4
    elif (heartRate > 140):
        score += 3
    elif (heartRate > 110):
        score += 2
    elif (heartRate > 70):
        score += 0
    elif (heartRate > 55):
        score += 2
    elif (heartRate > 40):
        score += 3
    else:
        score += 4

    if (respiratoryRate > 50):
        score += 4
    elif (respiratoryRate > 35):
        score += 3
    elif (respiratoryRate > 25):
        score += 1
    elif (respiratoryRate > 12):
        score += 0
    elif (respiratoryRate > 10):
        score += 1
    elif (respiratoryRate > 6):
        score += 2
    else:
        score += 4

    if (FiO2 < 50):
        atmosphericPressure = 760
        h2OPressure = 47
        aaGradient = ((FiO2) * (atmosphericPressure - h2OPressure) - (PaCO2/0.8)) - PaO2
        if (aaGradient > 499):
            score += 4
        elif (aaGradient > 350):
            score += 3
        elif (aaGradient > 200):
            score += 2
        else:
            score += 0
    else:
        if PaO2 > 70:
            score += 0
        elif PaO2 > 61:
            score += 1
        elif PaO2 > 55:
            score += 3
        else:
            score += 4

    if (arterialPh > 7.7):
        score += 4
    elif (arterialPh > 7.6):
        score += 3
    elif (arterialPh > 7.5):
        score += 1
    elif (arterialPh > 7.33):
        score += 0
    elif (arterialPh > 7.25):
        score += 2
    elif (arterialPh > 7.15):
        score += 3
    else:
        score += 4

    if (serumSodium > 180):
        score += 4
    elif (serumSodium > 160):
        score += 3
    elif (serumSodium > 155):
        score += 2
    elif (serumSodium > 150):
        score += 1
    elif (serumSodium > 130):
        score += 0
    elif (serumSodium > 120):
        score += 2
    elif (serumSodium > 111):
        score += 3
    else:
        score += 4

    if (serumPotassium > 7):
        score += 4
    elif (serumPotassium > 6):
        score += 3
    elif (serumPotassium > 5.5):
        score += 2
    elif (serumPotassium > 3.5):
        score += 0
    elif (serumPotassium > 3):
        score += 1
    elif (serumPotassium > 2.5):
        score += 2
    else:
        score += 4

    if (serumCreatinine > 3.5):
        if (renalFailure == 'Acute renal failure'):
            score += 8
        elif (renalFailure == 'CHRONIC renal failure'):
            score += 4
    elif (serumCreatinine > 2.0):
        if (renalFailure == 'Acute renal failure'):
            score += 6
        elif (renalFailure == 'CHRONIC renal failure'):
            score += 3
    elif (serumCreatinine > 1.5):
        if (renalFailure == 'Acute renal failure'):
            score += 4
        elif (renalFailure == 'CHRONIC renal failure'):
            score += 2
    elif (serumCreatinine > 0.6):
        score += 0
    else:
        score += 2

    if (hematocrit > 60):
        score += 4
    elif (hematocrit > 50):
        score += 2
    elif (hematocrit > 46):
        score += 1
    elif (hematocrit > 30):
        score += 0
    elif (hematocrit > 20):
        score += 2
    else:
        score += 4

    if (wbc > 40):
        score += 4
    elif (wbc > 20):
        score += 2
    elif (wbc > 15):
        score += 1
    elif (wbc > 3):
        score += 0
    elif (wbc > 1):
        score += 2
    else:
        score += 4

    score += (15 - gcs)

    return score


def calculateApacheiiMortalityRisk(apache_ii, admissionType):
    if (admissionType == 'nonoperative'):
        if (apache_ii > 34):
            return 85
        elif (apache_ii > 30):
            return 73
        elif (apache_ii > 25):
            return 55
        elif (apache_ii > 20):
            return 40
        elif (apache_ii > 15):
            return 25
        elif (apache_ii > 10):
            return 15
        elif (apache_ii > 5):
            return 8
        else:
            return 4
    elif (admissionType == 'postoperative'):
        if (apache_ii > 34):
            return 88
        elif (apache_ii > 30):
            return 73
        elif (apache_ii > 25):
            return 35
        elif (apache_ii > 20):
            return 30
        elif (apache_ii > 15):
            return 12
        elif (apache_ii > 10):
            return 7
        elif (apache_ii > 5):
            return 3
        else:
            return 1
    return None


def calculatePBS(temperature, systolicBp, intravenousVasopressor, mechanicalVentillation, cardiacArrest, mentalStatus):

    score = 0

    if (temperature <= 35):
        score += 2
    elif (temperature <= 36):
        score += 1
    elif (temperature < 39):
        score += 0
    elif (temperature < 40):
        score += 1
    else:
        score += 2

    if (systolicBp < 90):
        score += 2
    elif (intravenousVasopressor):
        score += 2

    if (mechanicalVentillation):
        score += 2

    if (cardiacArrest):
        score += 2

    if (mentalStatus == 'Alert'):
        score += 0
    elif (mentalStatus == 'Disoriented'):
        score += 1
    elif (mentalStatus == 'Stuporous'):
        score += 2
    elif (mentalStatus == 'Comatose'):
        score += 4

    return score
