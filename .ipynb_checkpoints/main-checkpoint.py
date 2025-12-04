import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
#import matplotlib
#import matplotlib.pyplot as plt


df = pd.read_csv('vehicleprice.csv')
n_df = df.dropna()
n_df = n_df.drop('Location', axis=1)
n_df = n_df.drop('Title', axis=1)



encoder = LabelEncoder()
#n_df['BrandModel'] = n_df['Brand'] + '-' + n_df['Model']
n_df['Brand_encoded'] = encoder.fit_transform(n_df['Brand'])
n_df['Model_encoded'] = encoder.fit_transform(n_df['Model'])
#n_df['Year_encoded'] = pd.to_numeric(df['Year'], errors='coerce')
n_df['Car/Suv_encoded'] = encoder.fit_transform(n_df['Car/Suv'])
n_df['UsedOrNew_encoded'] = encoder.fit_transform(n_df['UsedOrNew'])
n_df['Transmission_encoded'] = encoder.fit_transform(n_df['Transmission'])
n_df['Engine_encoded'] = encoder.fit_transform(n_df['Engine'])
n_df['DriveType_encoded'] = encoder.fit_transform(n_df['DriveType'])
n_df['FuelType_encoded'] = encoder.fit_transform(n_df['FuelType'])
n_df['FuelConsumption_encoded'] = encoder.fit_transform(n_df['FuelConsumption'])
n_df['Kilometres_encoded'] = encoder.fit_transform(n_df['Kilometres'])
n_df['ColourExtInt_encoded'] = encoder.fit_transform(n_df['ColourExtInt'])
n_df['CylindersinEngine_encoded'] = encoder.fit_transform(n_df['CylindersinEngine'])
n_df['BodyType_encoded'] = encoder.fit_transform(n_df['BodyType'])
n_df['Doors_encoded'] = encoder.fit_transform(n_df['Doors'])
n_df['Seats_encoded'] = encoder.fit_transform(n_df['Seats'])
n_df['Price_encoded'] = pd.to_numeric(df['Price'], errors='coerce')


#X = n_df[['Brand_encoded', 'Model_encoded', 'Car/Suv_encoded','UsedOrNew_encoded','Transmission_encoded','Engine_encoded','DriveType_encoded',
 #         'FuelType_encoded','FuelConsumption_encoded','Kilometres_encoded','ColourExtInt_encoded','CylindersinEngine_encoded','BodyType_encoded'
  #        ,'Doors_encoded','Seats_encoded']]
#y = n_df['Price_encoded']

X = n_df.drop('Price_encoded',axis=1)
y = n_df['Price_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print ("X_train Shape:", X_train.shape)
#print("X_test Shape:", X_test.shape)
#print("Y_train Shape:", y_train.shape)
#print("Y_test Shape:", y_test.shape)

select_k_best = SelectKBest(score_func=chi2, k=10)

X_train_k_best = select_k_best.fit_transform(X_train,y_train)
X_test_k_best = select_k_best.transform(X_test)



print("Selected features:", X_train.columns[select_k_best.get_support()])



print(df.dtypes)
#df.plot()
#plt.show()
#for x in df.index:
 #   df['Year'] = pd.to_numeric(df['Year'])
#x = df["Brand"].mode()[0]
#x = df["Year"].median()
#print(x)

#print(new_df.info())
#print(df.info())
#print(df.head())
#print(n_df.info())
#print(df.duplicated())
#print(df.to_string()) it prints entire dataset

#print(matplotlib.__version__)