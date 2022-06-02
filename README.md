# Autoscout_LGBM

**Please Note : The dataset provided for learning purpose.And this is a startup project at junior level.**


- Programing, Python
- Colab, platform to collaborate
- Deployment, Streamlit

  Its important to know price of a car roughly before selling or buying it.For car price prediction,we used data which obtained from "autoscout.nl" and searched some information about how the prices vary with those features (variables,columns).
We aim to create an interactive online interface ,where the users can enter the features about a particular car to have an idea about its market price.
This interface will use our machine learning model for prediction in the background.




## Data summary:
  This data obtained from "autoscout.nl".Dataframe (df) has 71104 records(rows), 49 features(columns),size 110 MB.
Almost all columns explained briefly below.
At the end of EDA processes ,69510 rows,160 columns,size 85MB.



province  :*Shows where car sell from*<br />
make_model :*Brand and model*<br />
vehicle_age :*Vehicle age*<br />
drivetrain :*Shows where is the engine power*<br />
non_smoker_vehicle :*Smoke in car or no*<br />
empty_weight :*The vehicle's weight without passenger*<br />
mileage :*Total traveled distance*<br />
co2_emissions :*Carbondioxide emission amount (g/km)*<br />
doors :*Number of doors of the vehicle*<br />
gears :*Number of gears of the vehicle*<br />
colour :*Colour of the vehicle*<br />
upholstery :*Upholstery type of vehicle -cloth,leather*<br />
combination (L/100Km) :*Avarage fuel consumption*<br />
Safety&Security :*Features for safety&security for exp;Central door lock,Driver-side airbag*<br />
price :*Price of vehicle*<br />
seller :*Who is selling,private or galery*<br />
location :*Location of vehicle*<br />
power :*Shows engine power in cc or kwh*<br />
seats :*Number of total seats of vehicle*<br />
warranty :*Warranty by galery or seller -month*<br />
general_inspection :*Shows whether vehicle need inspection*<br />
previous_owner :*Who was previous owner*<br />
engine_size :*Hows size of vehicle engine in cc or kwh*<br />
Fuel_type :*What fuel(or whether electric) type vehicle needs -Gasoline,Diesel,electric*<br />
upholstery_colour :*Colour of upholstery -Black,brown*<br />
Comfort&Convenience :*Systems ,features like  -'Air conditioning','Navigation system','Power windows'*<br />
make :*Brand of car -BMW,renault ...*<br />
gearbox :*Type of gear -automatic, manual*<br />
emission_class :*The Euro emissions standards, from Euro 1 to Euro 6*<br />
short_description :*Cars description by seller*<br />
type :*New, used*<br />
country_version :*Cars country version(Esp,NL)*<br />
first_registration :*When the car registered "month/year"*<br />
full_service_history:*Used throughout the pre-owned car market, and shows that a vehicle has been well maintained -yes,no*<br />
cylinders :*How many cylinders powering the car -4,6,8*<br />



### Python libraries : 

joblib==1.1.0<br />
lightgbm==2.2.3<br />
numpy==1.21.4<br />
pandas==1.3.4<br />
scikit-learn==1.0.2<br />
streamlit==1.8.1<br />


## Read data from google drive,afterwards take a copy of data to clean.

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/autoscout_data_2000.csv')
df = data.copy()  

Here initial dataframe
![1](https://user-images.githubusercontent.com/70334899/171736085-e612e7c2-6bff-4965-85b7-6f4535896fc8.png)



#### You can view EDA_1*,EDA_2* ,EDA_3* for all steps at the bottom.




## Data Cleaning and Preparation

  Dataframe was in "csv" format.This caused some issues during saving and reloading processes(caused broken records).
We converted "csv" to "pickle" which works flawlessly.


Columns had messy information,used regular expression in python.

*df['engine_size']=df['engine_size'].str.replace(r'[^0-9]+', ‘’)*




![cleaned](https://user-images.githubusercontent.com/70334899/171736132-a20eb472-93ea-4c66-a32a-4e0bec3ef2c3.png)





### Detect Broken Datas/Records
  We realized that 10 unpair records was broken and some ,deleted those records. 
Some records are meaningless in DF .  Firtsly ,we searched for domain knowledge to distinguish whether they are meaningful.
They are not real values ,might be entered by users mistakenly.

If filling them with functions is accurate, convert those values to NaN values then fill them again.
If it doesnt worth to deal,just drop.
Here some examples:
-empty_weight<500 kg
-price>10000000 €
-engine_power<50 kW

### Creating useful features
- Added province feature, according to location feature,used another csv data which obtained from
another website to derive province feature.
- Unified make(brand) and model columns,makes more sense for filling nan values , same models have similar features.
- Derived new features for more readable features,just optional, vehicle_age.

### Filling NaN values 
NaN values filled with the proper functions such as "fill_most".We have several pre-prepared functions ,such as fill() ,fill_prop(), fill_most(), which can be seen in EDA_2 file.

### Converting features
- Converted data dtype(all data types were string) to the most proper data type for that particular feature.For instance: engine size 2000cc(string) to 2000(integer).
- Use "get_dummies" function to extract variables from "Safety & Security,Comfort & Convenience".The get_dummies function is used to convert categorical variables into a value of 0 or 1.This is more proper for ML model.
**Note**:There was some dublicated names -like control and contro appear like different features due to one letter difference-.
renamed the incorrect names to merge the correct names(in this instance "contro" merged in "control" ,to avoid duplication).

![dummy](https://user-images.githubusercontent.com/70334899/171738560-32a1f364-176e-4e4f-9af2-c8b2c7241aa5.PNG)




## Visualize outliers -EDA_3 


### Detect Anomalies and Replace Outliers
You can check *EDA_3* to see which functions we used for replace_outliers(),  outliers(),  capping_outliers().

- Used some visualization tool and deal with outliers by using functions which indicated before.
Additionally during visualization,colab needed upgrade matplotlib,for better visualization.

![capping](https://user-images.githubusercontent.com/70334899/171738809-1ca55704-70f9-48fa-8014-77d4d72a843f.PNG)


Applied label encoding to convert non-numeric values to numerics for ML model.

```
 from sklearn.preprocessing import LabelEncoder
 labelencoder = LabelEncoder()
 for column in df.select_dtypes(exclude=[np.number]).columns:
         df[f'{column}'] = labelencoder.fit_transform(df[f'{column}']) 
```

![labeling](https://user-images.githubusercontent.com/70334899/171739579-ce41619b-c356-484b-935d-51c7ec0d92be.PNG)
![labeling2](https://user-images.githubusercontent.com/70334899/171739584-691e3925-b922-4c65-8e44-04fba1b3a667.PNG)




## ML Model and Deployment -ML-Colab


- Tried to find out which is the best ML model for our dataset,compare R2 and RMSE values.
- Tested some models,such as Lasso(),  KNeighborsRegressor(), LightGBM(), and decided on LGBM ,one of the newest popular model.
 
![modelscores](https://user-images.githubusercontent.com/70334899/171739662-8148e8ef-5e05-4260-b508-7887cd8e1224.PNG)
![lgbmscore](https://user-images.githubusercontent.com/70334899/171739675-caad4648-6936-4a25-aa24-2e0da3fd2072.PNG)



- Focused on what we can do for a better result.
- Variable importance refers to how much a given model uses that variable to make accurate predictions.

![image](https://user-images.githubusercontent.com/70334899/171741272-4691fc1f-d9ad-4c38-9837-c9db8e2af771.png)

- Multicollinearity occurs when there is a high correlation between the independent variables in the regression analysis which impacts the overall interpretation of the results. It reduces the power of coefficients.Aware of that you need leave at least one of those similar features,if you delete all,ML model cant use that feature to predict similar features.It doesnt affect predictions result.We show "vif values" here and drop to avoid multicollinearity.

![image](https://user-images.githubusercontent.com/70334899/171741344-b95301dd-f271-41d4-8dd7-d692be134b55.png)

- For interactive interface,the users must fill those features manually to predict price of a particular car.
But here it automatically fills all features according to the most frequent features of this brand/model("make_model" column).Then the users can change features themselves.So it takes less time to fill default values.

Try to [DEMO](https://share.streamlit.io/kadrefe/autoscout_lgbm/main/web_deploy.py)

*Streamlit screenshots * : 
![image](https://user-images.githubusercontent.com/70334899/171741586-93d1dae0-2b1a-4a22-9564-ab7a9e9ffb8f.png)
![image](https://user-images.githubusercontent.com/70334899/171741669-438547e2-656a-4cb7-82a4-f772f0de7d66.png)
![image](https://user-images.githubusercontent.com/70334899/171741733-ebe46fdd-0d98-4673-92b1-e51f4cbb1101.png)






Link for [EDA_1](https://colab.research.google.com/drive/1XBUgyPCuSIzbWqiNBVWO8fIB1SaLHuWU?usp=sharing)


Link for [EDA_2](https://colab.research.google.com/drive/17QkwZaafpTEgmkU2yQujx9CqMpcijkvB?usp=sharing)


Link for [EDA_3](https://colab.research.google.com/drive/150BfB1v7rjLZVKIEtA5AxJZ1c4j7Yplx?usp=sharing)

Link for [ML-Colab](https://colab.research.google.com/drive/1CffULUbOuMyiXhiXBcsjX0myCv4ixTE8?usp=sharing)







