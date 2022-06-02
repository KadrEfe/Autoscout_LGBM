import dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


def binary_map(feature):
    return feature.map({True: 1, False: 0})


def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """

    if (option == "Online"):
        binary_list = ['Tinted windows', 'general_inspection', 'Panorama roof', 'Fog lights', 'Sport seats', 'Roof rack', 'Electrically adjustable seats', 'Lumbar support', 'Keyless central door lock', 'Seat heating', 'Parking assist system camera',
                       'Trailer hitch', 'Adaptive Cruise Cntrl', 'On-board computer', 'Power steering', 'Electric tailgate', 'Immobilizer']
        df[binary_list] = df[binary_list].apply(binary_map)
        df.drop(columns=['make', 'model'], axis=1, inplace=True)

        for column in df.select_dtypes(exclude=[np.number]).columns:

            df[f'{column}'] = labelencoder.fit_transform(df[f'{column}'])

        return df.iloc[0]

    elif(option == "Batch"):
        df['make_model'] = df.make + '_' + df.model
        df.drop(columns=['make', 'model'], axis=1, inplace=True)
        column = ['province', 'make_model', 'body_type', 'vehicle_age', 'mileage', 'Gearbox', 'gears', 'seller', 'colour', 'power(kW)', 'cylinders',
                  'drivetrain', 'type', 'full_service_history', 'fuel_type', 'combination(L/100Km)', 'non_smoker_vehicle', 'empty_weight', 'co2_emissions', 'emission_class(Euro)', 'seats', 'warranty_months', 'engine_size', 'upholstery_colour', 'doors', 'upholstery', 'general_inspection', 'Android Auto', 'Apple CarPlay', 'Bluetooth', 'CD player', 'Digital cockpit', 'Digital radio', 'Hands-free equipment', 'Induction charging for smartphones', 'Integrated music streaming', 'MP3', 'On-board computer', 'Radio', 'Sound system', 'Television', 'USB', 'WLAN ', 'WiFi hotspot', 'All season tyres', 'Alloy wheels', 'Ambient lighting', 'Automatically dimming interior mirror', 'Cargo barrier', 'Catalytic Converter', 'E10-enabled', 'Electronic parking brake', 'Emergency tyre', 'Emergency tyre repair kit', 'Handicapped enabled', 'Headlight washer system', 'Range extender', 'Right hand drive', 'Roof rack', 'Shift paddles', 'Ski bag', 'Sliding door', 'Smokers package', 'Spare tyre', 'Spoiler', 'Sport package', 'Sport seats', 'Sport suspension', 'Steel wheels', 'Summer tyres', 'Touch screen', 'Trailer hitch', 'Tuned car', 'Winter package', 'Winter tyres', 'Voice control', '2 zones', '3 zones', '4 zones',
                  'Air conditioning', 'Air suspension', 'Armrest', 'Auxiliary heating', 'Electric backseat adjustment', 'Electric tailgate', 'Electrical side mirrors', 'Electrically adjustable seats', 'Electrically heated windshield', 'Fold flat passenger seat', 'Heads-up display', 'Heated steering wheel', 'Hill Holder', 'Keyless central door lock', 'Leather seats', 'Light sensor', 'Lumbar support', 'Massage seats', 'Navigation system', 'Panorama roof', 'Park Distance Control', 'Parking assist system camera', 'Parking assist system self-steering', 'Parking assist system sensors front', 'Parking assist system sensors rear', 'Power windows', 'Rain sensor', 'Seat heating', 'Seat ventilation', 'Sliding door left', 'Sliding door right', 'Split rear seats', 'Start-stop system', 'Sunroof', 'Tinted windows', 'Wind deflector', 'camera', 'Automatic Climate Control', 'Cruise Control', 'Leather Steering Wheel', 'Multi-function Steering Wheel', 'ABS', 'Adaptive headlights', 'Alarm system', 'Bi-Xenon headlights', 'Blind spot monitor', 'Central door lock', 'Daytime running lights', 'Distance warning system', 'Driver drowsiness detection', 'Driver-side airbag', 'Emergency brake assistant', 'Emergency system', 'Fog lights', 'Full-LED headlights', 'Glare-free high beam headlights', 'Head airbag', 'High beam assist', 'Immobilizer', 'Isofix', 'LED Daytime Running Lights', 'LED Headlights', 'Lane departure warning system', 'Laser headlights', 'Night view assist', 'Passenger-side airbag', 'Power steering', 'Rear airbag', 'Side airbag', 'Speed limit control system', 'Tire pressure monitoring system', 'Traffic sign recognition', 'Xenon headlights', 'Adaptive Cruise Cntrl', 'Central door lock with remote contrl', 'Electronic stability contrl', 'Traction contrl']

        dataset.default_values.reset_index(
            drop=True, inplace=True)
        # variables = dataset.default_values[]
        # print(variables)
        df = pd.get_dummies(df).reindex(columns=column, fill_value=0)
        print(df)
        for column in df.select_dtypes(exclude=[np.number]).columns:

            df[f'{column}'] = labelencoder.fit_transform(df[f'{column}'])

        return df
