
import pandas as pd
import numpy as np


def reverse_binary_map(feature):
    return feature.map({1: True, 0: False})


df = pd.read_pickle('defaultdataset.pickle')
default_values = pd.read_pickle('defaultvalues.pickle')

df.drop(columns=['price', 'registration', 'city(L/100Km)',
        'country(L/100Km)'], axis=1, inplace=True)

default_values[default_values.iloc[:, 29:].columns] = default_values[default_values.iloc[:,
                                                                                         29:].columns].apply(reverse_binary_map)


def make(most=False):
    make_list = df.make.unique()
    if most:
        most_brand = df.make.value_counts().index.to_list()[0]
        sorted_makes = np.sort(make_list)
        # returns most value index number
        return sorted_makes.tolist().index(most_brand)
    else:
        return np.sort(make_list)


def model(make, most=False):
    model_list = df.model[df.make == make].unique()
    if most:
        most_model = df.model[df.make == make].value_counts().index.to_list()[
            0]
        sorted_makes = np.sort(model_list)
        # returns most value index number
        return sorted_makes.tolist().index(most_model)
    else:
        return np.sort(model_list)


def body_type(make, model, most=False):
    body_type_list = df.body_type[(df.make == make) & (
        df.model == model)].unique()
    if most:
        most_body_type = df.body_type[(df.make == make) & (
            df.model == model)].value_counts().index.to_list()[0]
        sorted_body_types = np.sort(body_type_list)
        return sorted_body_types.tolist().index(most_body_type)
    else:
        return np.sort(body_type_list).tolist()


def colour():
    return list(df.colour.unique())


def province():
    return list(df.province.unique())


def fuel_type():
    return list(df.fuel_type.unique())


def co2_emission_calculater(fuel_type, combination):
    if fuel_type == 'Gasoline':
        co2_emission = 2392  # Standart
        return (co2_emission * combination) / 100
    elif fuel_type == 'Diesel':
        co2_emission = 2640  # Standart
        return (co2_emission * combination) / 100
    elif fuel_type == 'LPG':
        co2_emission = 2640  # Standart
        return (co2_emission * combination) / 100
    elif fuel_type == 'Electric':
        co2_emission = 0  # Standart
        return (co2_emission * combination) / 100
    else:
        return


def drivetrain(make, model, most=False):

    drivetrain = df.drivetrain.unique()
    if most:
        drivetrain_most = df.drivetrain[(df.make == make) & (
            df.model == model)].value_counts().index.to_list()[0]
        sorted_drivetrain_most = np.sort(drivetrain)
        return sorted_drivetrain_most.tolist().index(drivetrain_most)
    else:
        return np.sort(drivetrain).tolist()


def data():
    return df

