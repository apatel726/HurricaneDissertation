def hurricane_ai(input):
  '''
  input = {
    -120 : timestep,
    -96 : timestep,
    -72 : timestep,
    -48 : timestep,
    -24 : timestep,
    0 : timestep
  }
  output = {
    24 : prediction,
    48 : prediction,
    72 : prediction,
    96 : prediction,
    120 : prediction
  }
  timestep = {
      'lat' : float,
      'long' : float,
      'max-wind' : float,
      'min_pressure' : float,
      'entry-time' : datetime
  }
  prediction = {
    'lat' : float,
    'long' : float,
    'max-winds' : float
  }
  '''
  # Take entries and transform them into our data model
  extract = []
  temp = None
  for index, value in enumerate([-120, -96, -72, -48, -24, 0]):
    if not index :
      temp = input[value]
      continue
    else:
      extract.append(list(feature_extraction(input[value], temp).values()))
      temp = input[value]
  
  state = np.expand_dims(scaler.transform(extract), axis = 0)
  print('extract: {}, state: {}'.format(extract, state))
  # Finally, use our hurricane ai to predict storm state
  lat = [output[0] for output in scaler.inverse_transform(
      [[lat[0],0,0,0,0,0,0,0,0,0,0] for lat in model_lat.predict(state)[0]])]
  long = [output[1] for output in scaler.inverse_transform(
      [[0,long[0],0,0,0,0,0,0,0,0,0] for long in model_long.predict(state)[0]])]
  wind = [output[2] for output in scaler.inverse_transform(
      [[0,0,wind[0],0,0,0,0,0,0,0,0] for wind in model_wind.predict(state)[0]])]
   
  output = dict()
  for index, value in enumerate([24, 48, 72, 96, 120]) :
    output[value] = {
        'lat' : lat[index],
        'long' : long[index],
        'max_wind' : wind[index]
    }
  
  return output
  
  from dateutil.parser import parse
input = {
  0 : {
      'entry_time' : parse('Fri Aug 30 2019 1100 PM'),
      'lat' : 25.5,
      'long' : 71.4,
      'max_wind' : 140 / 1.51 , # mph to knots
      'min_pressure' : 948.0
    }
  -24 : {
      'entry_time' : parse('Thu Aug 29 2019 1100 PM'),
      'lat' : 23.3,
      'long' : 68.4,
      'max_wind' : 105 / 1.51 , # mph to knots
      'min_pressure' : 977.0
    },
  -48 : {
      'entry_time' : parse('Wed Aug 28 2019 1100 PM'),
      'lat' : 19.7,
      'long' : 66.0,
      'max_wind' : 85 / 1.51 , # mph to knots
      'min_pressure' : 986.0
    },
  -72 : {
      'entry_time' : parse('Tue Aug 27 2019 1100 PM'),
      'lat' : 16.0,
      'long' : 63.0,
      'max_wind' : 50 / 1.51 , # mph to knots
      'min_pressure' : 1006.0
    },
  -96 : {
      'entry_time' : parse('Mon Aug 26 2019 1100 PM'),
      'lat' : 13.2,
      'long' : 59.7,
      'max_wind' : 50 / 1.51 , # mph to knots
      'min_pressure' : 1003.0
    },
  -120 : {
      'entry_time' : parse('Sun Aug 25 2019 1100 PM'),
      'lat' : 11.7,
      'long' : 55.3,
      'max_wind' : 50 / 1.51 , # mph to knots
      'min_pressure' : 1003.0
    }
}
