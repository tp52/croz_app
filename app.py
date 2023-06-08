from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained ML model
model = joblib.load('model.pkl')

app.run(host='0.0.0.0')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        cylinders = int(request.form['cylinders'])
        odometer = int(request.form['odometer'])
        manufacturer = request.form['manufacturer'].lower()
        condition = request.form['condition']
        fuel = request.form['fuel']
        car_type = request.form['type']
        drive_type = request.form['drive_type']
        transmission=request.form['transmission']

        car_info = {
            'Year of Production': year,
            'Number of Cylinders': cylinders,
            'Odometer': odometer,
            'Manufacturer': manufacturer,
            'Condition': condition,
            'Fuel': fuel,
            'Car Type': car_type,
            'Drive Type': drive_type,
            'transmission':transmission
        }

        # Create a DataFrame from the user inputs
        data = pd.DataFrame({
            'year':[year],
            'cylinders': [cylinders],
            'manufacturer': [manufacturer],
            'condition': [condition],
            'fuel': [fuel],
            'type': [car_type],
            'drive': [drive_type],
            'transmission':[transmission]
        })


        data_copy = data.copy()

        type_encoded = pd.get_dummies(data['type'], prefix='type', dtype='int64')
        missing_columns = ['type_SUV', 'type_bus', 'type_convertible', 'type_coupe', 'type_hatchback', 'type_mini-van',
                           'type_offroad', 'type_other', 'type_pickup', 'type_sedan', 'type_truck', 'type_van',
                           'type_wagon']
        for column in missing_columns:
            if column not in type_encoded.columns:
                type_encoded[column] = 0
        # Concatenate the encoded columns with the original DataFrame
        data_copy = pd.concat([data, type_encoded], axis=1)
        # Drop the original 'type' column if desired
        data_copy.drop('type', axis=1, inplace=True)

        if manufacturer not in ['gmc', 'chevrolet', 'toyota', 'ford', 'jeep', 'nissan', 'ram', 'others', 'honda', 'dodge', 'volkswagen', 'hyundai', 'mercedes-benz', 'bmw', 'subaru']:
            data['manufacturer']="others"
        manufacturer_encoded = pd.get_dummies(data['manufacturer'], prefix='manufacturer', dtype='int64')
        # Concatenate the encoded columns with the original DataFrame
        missing_columns = ['manufacturer_gmc', 'manufacturer_chevrolet', 'manufacturer_toyota', 'manufacturer_ford',
                           'manufacturer_jeep', 'manufacturer_nissan', 'manufacturer_ram', 'manufacturer_honda',
                           'manufacturer_dodge', 'manufacturer_volkswagen', 'manufacturer_hyundai',
                           'manufacturer_mercedes-benz', 'manufacturer_bmw', 'manufacturer_subaru',
                           'manufacturer_others']
        for column in missing_columns:
            if column not in manufacturer_encoded.columns:
                manufacturer_encoded[column] = 0
        data_copy = pd.concat([data_copy, manufacturer_encoded], axis=1)
        # Drop the original 'type' column if desired
        data_copy.drop('manufacturer', axis=1, inplace=True)

        #DRIVE
        drive_encoded = pd.get_dummies(data_copy['drive'], prefix='drive', dtype='int64')
        # Concatenate the encoded columns with the original DataFrame
        missing_columns = ['drive_4wd', 'drive_fwd', 'drive_rwd']
        for column in missing_columns:
            if column not in drive_encoded.columns:
                drive_encoded[column] = 0
        data_copy = pd.concat([data_copy, drive_encoded], axis=1)
        # Drop the original 'type' column if desired
        data_copy.drop('drive', axis=1, inplace=True)

        fuel_encoded = pd.get_dummies(data['fuel'], prefix='fuel', dtype='int64')
        # Concatenate the encoded columns with the original DataFrame
        missing_columns = ['fuel_diesel', 'fuel_electric', 'fuel_hybrid', 'fuel_other','fuel_gas']
        for column in missing_columns:
            if column not in fuel_encoded.columns:
                fuel_encoded[column] = 0
        data_copy = pd.concat([data_copy, fuel_encoded], axis=1)
        # Drop the original 'type' column if desired
        data_copy.drop('fuel', axis=1, inplace=True)

        transmission_encoded=pd.get_dummies(data['transmission'], prefix='transmission', dtype='int64')
        missing_columns=['transmission_automatic','transmission_manual','transmission_other']
        for column in missing_columns:
            if column not in transmission_encoded.columns:
               transmission_encoded[column] = 0

        data_copy = pd.concat([data_copy, transmission_encoded], axis=1)
        # Drop the original 'type' column if desired
        data_copy.drop('transmission', axis=1, inplace=True)

        data_copy['age'] = 2022.0 - year
        data_copy['odometer*age']=data_copy['age']*odometer
        data=data_copy
        #print(fuel_encoded)

        encoded_dict = {'salvage': 0, 'fair': 1, 'good': 2, 'excellent': 3, 'like new': 4, 'new': 5}
        data['condition']=encoded_dict[condition]

        data=data.sort_index(axis=1)

        prediction = model.predict(data)
        #print(prediction)
        #print(data)
        return render_template('index.html', car_info=car_info, prediction=round(prediction[0]))

    return render_template('index.html', car_info=None, prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
