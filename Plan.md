For each preprocesseor:

    1. Look up for the pkl file specified in the YAML

    2. If pkl file found then load it and
        check isFit()
        if not is fit -> yell at user
        else -> continue

    3. If not found then yell at user


To fit each preprocessor:
    1. Call the file name of the dataset to target
    2. Fit the preprocessor on it and save the pkl file with a certain name

Each preprocessor is fit manually ahead of time that way it saves time and complexity when actually loading the functions.