
# Import libraries
import joblib
import streamlit as st
import pandas as pd
from PIL import Image
from preprocessing import preprocess
import dataset

# load the model from disk

Lgbm = joblib.load('LGBMregressor.sav')


def main():
    # Setting Application title
    st.title('🚗 DataList AutoScout24 Price Prediction 🚗')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict Price.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('autoscout24.jpg')
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Price')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")

        # Based on our optimal features selection

        st.subheader("Basic Data")

        make = st.selectbox('What is the car brand ?:',
                            dataset.make(), index=dataset.make(most=True))
        model = st.selectbox(f'What is the model of {make}', dataset.model(
            make), index=dataset.model(make, most=True))

        # take default variable from dataset
        variables = dataset.default_values[dataset.default_values.make_model ==
                                           make+'_'+model].to_dict(orient='records')[0]

        # dataset.province().index(variables['province'])
        with st.form("my_form"):

            province = st.selectbox(
                'Where do you sell from ? ', dataset.province())

            type = st.selectbox('Catagory', ('Used', 'Demonstration', 'Pre-registered', 'New',
                                             'Antique / Classic'), index=('Used', 'Demonstration', 'Pre-registered', 'New',
                                                                          'Antique / Classic').index(variables['type']))
            body_type = st.selectbox(f'What is the body type of this {model}', dataset.body_type(
                make, model), index=dataset.body_type(make, model).index(variables['body_type']))
            seats = st.slider('Seats', min_value=1, max_value=9,
                              value=int(variables['seats']))
            doors = st.slider('Door', min_value=1, max_value=6,
                              value=int(variables['doors']))

            st.subheader('Vehicle History')
            if type == 'New' or type == 'Pre-registered':
                vehicle_age = st.slider(
                    'Vehicle Age', min_value=0, max_value=50, value=0)
                mileage = st.number_input('Mileage', min_value=0, value=0)
            else:
                vehicle_age = st.slider(
                    'Vehicle Age', min_value=0, max_value=50, value=int(variables['vehicle_age']))
                mileage = st.number_input(
                    'Mileage', min_value=0, value=int(variables['mileage']))
            general_inspection = st.checkbox(
                'General inspection', value=variables['general_inspection'])
            full_service_history = st.checkbox(
                'Full Service History', value=variables['full_service_history'])
            warranty_months = st.number_input(
                'Warranty Months', min_value=0, max_value=60, value=int(variables['warranty_months']))

            st.subheader("Technical data")
            power = st.slider('Power(kW)', min_value=2,
                              max_value=500, value=int(variables['power(kW)']))
            engine_size = st.slider(
                'Engine Size', min_value=500, max_value=5000, value=int(variables['engine_size']))
            cylinders = st.slider(
                "Cylinders", min_value=1, max_value=12, value=int(variables['cylinders']))
            gears = st.slider('Gears', min_value=1, max_value=10,
                              value=int(variables['gears']))
            gear_box = st.selectbox('Gearbox', (
                'Automatic', 'Manual', 'Semi-automatic'), index=(
                    'Automatic', 'Manual', 'Semi-automatic').index(variables['Gearbox']))
            st.write('Empty Weight')
            st.write("Please Choose between this parameters ;  ")
            st.markdown("0-1000 ==> very light")
            st.markdown("1000-1400 ==> light")
            st.markdown("1400-1800 ==> normal")
            st.markdown("1800-2200 ==> heavy")
            st.markdown("2200-5000 ==> very heavy")
            empty_weight = st.select_slider(
                'Empty Weight', options=('very_light', 'light', 'normal', 'heavy', 'very_heavy'), value=variables['empty_weight'])
            drivetrain = st.selectbox('Drive train', dataset.drivetrain(
                make, model), index=dataset.drivetrain(
                make, model).index(variables['drivetrain']))

            st.subheader('Energy consumption ')
            fuel_type = st.selectbox('Fuel Type', dataset.fuel_type(
            ), index=dataset.fuel_type().index(variables['fuel_type']))
            combination = st.slider(
                'Combination L/100km', min_value=0.0, max_value=30.0, value=float(variables['combination(L/100Km)']), step=0.5)
            co2_emissions = st.number_input(
                'Co2 Emission', min_value=0, max_value=500, value=int(variables['co2_emissions']))

            st.subheader("Color and Upholstery")
            colour = st.selectbox('Colour', dataset.colour(
            ), index=dataset.colour().index(variables['colour']))

            upholstery = st.selectbox(
                'Upholstery', ('Cloth', 'Part\\Full leather', 'Other'), index=('Cloth', 'Part\\Full leather', 'Other').index(variables['upholstery']))
            upholstery_colour = st.selectbox(
                'Upholstery colour', ('Black', 'Grey', 'Other', 'Beige'), index=('Black', 'Grey', 'Other', 'Beige').index(variables['upholstery_colour']))

            st.subheader('Options')

            col1, col2 = st.columns(2)

            with col1:
                tinted_windows = st.checkbox(
                    'Tinted windows', value=variables['Tinted windows'])
                panoroma_roof = st.checkbox(
                    'Panoroma Roof', value=variables['Panorama roof'])
                sport_seats = st.checkbox(
                    'Sport Seats', value=variables['Sport seats'])
                roof_rack = st.checkbox(
                    'Roof rack', value=variables['Roof rack'])
                electrically_adjustable_seats = st.checkbox(
                    'Electrically adjustable seats', value=variables['Electrically adjustable seats'])
                adaptive_cruise_cntl = st.checkbox(
                    'Adaptive cruise control', value=variables['Adaptive Cruise Cntrl'])
                onboard_computer = st.checkbox(
                    'On-board computer', value=variables['On-board computer'])
                power_steering = st.checkbox(
                    'Power steering', value=variables['Power steering'])
            with col2:

                lumbar_support = st.checkbox(
                    'Lumbar support', value=variables['Lumbar support'])
                seat_heating = st.checkbox(
                    'Seat heating', value=variables['Seat heating'])
                parking_assist_system_camera = st.checkbox(
                    'Parking assist system camera', value=variables['Parking assist system camera'])
                trailer_hitch = st.checkbox(
                    'Trailer Hitch', value=variables['Trailer hitch'])
                fog_lights = st.checkbox(
                    'Fog Lights', value=variables['Fog lights'])
                keyless_central_door_lock = st.checkbox(
                    'Keyless central door lock', value=variables['Keyless central door lock'])

                electric_tailgate = st.checkbox(
                    'Electric tailgat', value=variables['Electric tailgate'])
                immobilezer = st.checkbox(
                    'immobilezer', value=variables['Immobilizer'])

            predict = st.form_submit_button('Predict')

            if predict:

                data = {
                    'make': make,
                    'mileage': mileage,
                    'make_model': make+'_'+model,
                    'co2_emissions': co2_emissions,
                    'power(kW)': power,
                    'vehicle_age': vehicle_age,
                    'province': province,
                    'engine_size': engine_size,
                    'colour': colour,
                    'combination(L/100Km)': combination,
                    'body_type': body_type,
                    'gears': gears,
                    'Gearbox': gear_box,
                    'empty_weight': empty_weight,
                    'seats': seats,
                    'warranty_months': warranty_months,
                    'cylinders': cylinders,
                    'fuel_type': fuel_type,
                    'upholstery_colour': upholstery_colour,
                    'drivetrain': drivetrain,
                    'doors': doors,
                    'full_service_history': full_service_history,
                    'type': type,
                    'Tinted windows': tinted_windows,
                    'general_inspection': general_inspection,
                    'upholstery': upholstery,
                    'Panorama roof': panoroma_roof,
                    'Fog lights': fog_lights,
                    'Sport seats': sport_seats,
                    'Roof rack': roof_rack,
                    'Electrically adjustable seats': electrically_adjustable_seats,
                    'Lumbar support': lumbar_support,
                    'Keyless central door lock': keyless_central_door_lock,
                    'Seat heating': seat_heating,
                    'Parking assist system camera': parking_assist_system_camera,
                    'Trailer hitch': trailer_hitch,
                    'Adaptive Cruise Cntrl': adaptive_cruise_cntl,
                    'On-board computer': onboard_computer,
                    'Power steering': power_steering,
                    'Electric tailgate': electric_tailgate,
                    'Immobilizer': immobilezer}

                variables.update(data)

                features_df = pd.DataFrame.from_dict([variables])
                features_data = pd.DataFrame.from_dict([data])
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.write('Overview of input is shown below')
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.dataframe(features_data)
                preprocess_df = preprocess(features_df, 'Online')
                prediction = Lgbm.predict([preprocess_df])
                price = prediction[0]
                price = round(price, None)
                price = '💰💰''$$'+str(price)+'$$''💰💰'

                st.success(price)
                st.stop()
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
            # get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            # preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            data = pd.DataFrame(data)
            if st.button('Predict'):
                # get Batch prediction
                prediction = Lgbm.predict(preprocess_df)
                price = prediction
                prediction_df = pd.DataFrame(
                    price, columns=["Predictions"])
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)


if __name__ == '__main__':
    main()
