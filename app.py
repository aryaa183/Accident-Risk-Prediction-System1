# app.py
# Smart Accident Risk Predictor - Hackathon-ready backend
# Author: ChatGPT
# Run: python app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, render_template_string
from math import ceil
import datetime, random
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# ---------------------------
# Configuration / Tunables
# ---------------------------
BASE_RISK = 20
ROAD_TYPE_MULT = {'Highway': 1.25, 'City': 1.0, 'Rural': 1.1}
WEATHER_MULT = {'Clear': 0.9, 'Rainy': 1.4, 'Foggy': 1.6, 'Storm': 2.0, 'Night': 1.3, 'Heatwave': 1.1}
TIME_OF_DAY_MULT = {'Morning': 1.0, 'Afternoon': 1.0, 'Evening': 1.2, 'Night': 1.4}
LIGHTING_MULT = {'Daylight': 0.9, 'Dusk/Dawn': 1.2, 'Dark': 1.4}

WEATHER_TIPS = {
    'Clear': ['Maintain posted speed limit and stay alert.',
              'Keep a safe following distance of 2–3 seconds.',
              'Ensure sunglasses/visor ready for glare.'],
    'Rainy': ['Reduce speed by 20–40% depending on intensity.',
              'Avoid sudden braking or sharp steering.',
              'Watch for standing water and hydroplaning.'],
    'Foggy': ['Reduce speed to 30–50 km/h.',
              'Use low-beam lights and fog lamps only.',
              'Avoid overtaking and keep >6 seconds distance.'],
    'Night': ['Use high beams only on empty roads; switch to low for traffic.',
              'Watch for pedestrians and animals; reduce speed.',
              'Avoid driving if feeling drowsy; plan a rest stop.'],
    'Storm': ['If winds are high or visibility is near zero, postpone travel.',
              'Find a safe place to park—avoid bridges and open fields.',
              'Secure cargo and reduce speed drastically.'],
    'Heatwave': ['Check tyre pressure and engine coolant.',
                 'Carry extra water and avoid long highway stints without a break.',
                 'Allow AC cycles to prevent overheating.']
}

EMERGENCY_SERVICES = {
    'pune': [{'name': 'Ruby Hall Clinic', 'type': 'Hospital', 'eta_min': 15},
             {'name': 'Pune Expressway Patrol', 'type': 'Highway Patrol', 'eta_min': 8}],
    'mumbai': [{'name': 'Jaslok Hospital', 'type': 'Hospital', 'eta_min': 20},
               {'name': 'Mumbai Highway Patrol', 'type': 'Highway Patrol', 'eta_min': 12}],
}

# ---------------------------
# Utility Functions
# ---------------------------
def clamp01(x):
    return max(0, min(100, x))

def compute_risk(inputs):
    risk = BASE_RISK
    rt = inputs.get('road_type', 'Highway')
    risk *= ROAD_TYPE_MULT.get(rt, 1.0)
    w = inputs.get('weather', 'Clear')
    risk *= WEATHER_MULT.get(w, 1.0)
    tod = inputs.get('time_of_day', 'Morning')
    risk *= TIME_OF_DAY_MULT.get(tod, 1.0)
    lighting = inputs.get('lighting', 'Daylight')
    risk *= LIGHTING_MULT.get(lighting, 1.0)

    try: speed_limit = float(inputs.get('speed_limit') or 100)
    except: speed_limit = 100

    recommended_speed = speed_limit
    if w=='Rainy': recommended_speed = speed_limit*0.7
    elif w=='Foggy': recommended_speed = min(speed_limit,50)
    elif w=='Storm': recommended_speed = speed_limit*0.5
    elif tod=='Night' or lighting in ['Dusk/Dawn','Dark']: recommended_speed = speed_limit*0.85

    if speed_limit > recommended_speed+5: risk += (speed_limit-recommended_speed)*0.6

    try: age = int(inputs.get('driver_age') or 30)
    except: age=30
    if age<22: risk*=1.15
    elif age>70: risk*=1.2

    try: distance_km = float(inputs.get('distance_km') or 200)
    except: distance_km=200.0
    if distance_km>180: risk*=1.12

    overall = clamp01(round(risk,1))

    breakdown = {
        'weather_risk': int(clamp01(WEATHER_MULT.get(w,1.0)*25)),
        'road_condition_risk': int(clamp01(ROAD_TYPE_MULT.get(rt,1.0)*30)),
        'time_of_day_risk': int(clamp01(TIME_OF_DAY_MULT.get(tod,1.0)*20)),
        'driver_risk': int(20 if age<22 or age>70 else 8),
        'distance_fatigue_risk': int(12 if distance_km>180 else 5)
    }
    return overall, breakdown, int(ceil(recommended_speed))

def get_frequent_patterns():
    """
    Mock dataset: each row represents a past accident/high-risk observation.
    Columns are one-hot encoded features for Apriori.
    """
    data = [
        {'Weather_Rainy':1,'Time_Night':1,'Road_Highway':1,'HighRisk':1},
        {'Weather_Foggy':1,'Time_Dusk':1,'Road_Ghat':1,'HighRisk':1},
        {'Weather_Rainy':1,'Time_Morning':1,'Road_Highway':1,'HighRisk':1},
        {'Weather_Clear':1,'Time_Night':1,'Road_City':1,'HighRisk':0},
        {'Weather_Foggy':1,'Time_Night':1,'Road_Ghat':1,'HighRisk':1},
        # you can add more rows here
    ]
    df = pd.DataFrame(data)

    # Remove HighRisk column for Apriori input
    df_ap = df.drop('HighRisk', axis=1)

    # Fill any missing values with 0
    df_ap = df_ap.fillna(0)

    # Ensure all columns are int type (1/0)
    df_ap = df_ap.astype(int)

    # Frequent itemsets with min support 0.3
    frequent_itemsets = apriori(df_ap, min_support=0.3, use_colnames=True)

    # Generate rules with confidence >= 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    return rules


def detect_hazards(inputs):
    hazards=[]
    w = inputs.get('weather','Clear')
    rt = inputs.get('road_type','Highway')
    start = (inputs.get('start_point') or '').lower()
    dest = (inputs.get('destination') or '').lower()
    route_notes = (inputs.get('route_notes') or '').lower()

    if w=='Foggy': hazards.append('Low visibility expected — fog sections reported.')
    if w in ['Rainy','Storm'] and rt=='Highway': hazards.append('Highway surface may be slippery; watch for standing water.')
    if inputs.get('speed_limit'):
        try: speed_limit=float(inputs.get('speed_limit'))
        except: speed_limit=100
        if speed_limit>=100 and w!='Clear': hazards.append('High speed + adverse weather — consider slowing down or choosing alternate.')
    if 'ghat' in start or 'ghat' in dest or 'ghat' in route_notes:
        hazards.append('Ghat/steep section detected — check brakes and avoid overtaking.')
    if start and dest and (start in ['pune','mumbai'] or dest in ['pune','mumbai']):
        hazards.append('Known accident hotspots near expressway tolls and ghat sections.')
    return hazards

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index(): return redirect(url_for('form'))

SIMPLE_FORM_HTML = """
<!doctype html>
<title>Accident Risk Predictor</title>
<h2>Accident Risk Predictor — Test Form</h2>
<form method="post" action="/predict">
  Start Point: <input name="start_point" value="Pune"><br>
  Destination: <input name="destination" value="Mumbai"><br>
  Time of Day:
    <select name="time_of_day">
      <option>Morning</option><option>Afternoon</option>
      <option>Evening</option><option>Night</option>
    </select><br>
  Weather:
    <select name="weather">
      <option>Clear</option><option>Rainy</option><option>Foggy</option>
      <option>Storm</option><option>Heatwave</option><option>Night</option>
    </select><br>
  Road Type:
    <select name="road_type">
      <option>Highway</option><option>City</option><option>Rural</option>
    </select><br>
  Lighting:
    <select name="lighting">
      <option>Daylight</option><option>Dusk/Dawn</option><option>Dark</option>
    </select><br>
  Speed Limit (km/h): <input name="speed_limit" value="100"><br>
  Driver Age: <input name="driver_age" value="28"><br>
  Distance (km): <input name="distance_km" value="150"><br>
  Route notes: <input name="route_notes" value=""><br>
  <button type="submit">Get Report</button>
</form>
"""

@app.route('/form')
def form(): return render_template_string(SIMPLE_FORM_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    inputs = {k: request.form.get(k,'').strip() for k in [
        'start_point','destination','time_of_day','weather','road_type','lighting','speed_limit','driver_age','distance_km','route_notes'
    ]}

    overall_score, breakdown, recommended_speed = compute_risk(inputs)
    hazards = detect_hazards(inputs)
    level = 'HIGH' if overall_score>=75 else 'MEDIUM' if overall_score>=45 else 'LOW'
    weather_tips = WEATHER_TIPS.get(inputs.get('weather','Clear'), WEATHER_TIPS['Clear'])

    vehicle_tips=[]
    if inputs.get('road_type')=='Highway':
        vehicle_tips.append('Check tyre pressure: recommended 30–35 psi for passenger cars.')
        vehicle_tips.append('Ensure fuel is sufficient for at least 25% extra travel time.')
    else:
        vehicle_tips.append('Carry a basic tool kit and first aid kit.')
    try: dist=float(inputs.get('distance_km') or 0)
    except: dist=0
    if dist>180: vehicle_tips.append('Plan a 15-min rest stop every 2–3 hours. Driver rotation if possible.')

    dest_city = inputs.get('destination','').lower()
    services = EMERGENCY_SERVICES.get(dest_city, [
        {'name': f'Nearest Hospital ({dest_city.title()})','type':'Hospital','eta_min':15},
        {'name': f'{dest_city.title()} Highway Patrol','type':'Highway Patrol','eta_min':10}
    ])

    alternates = [
        {'name':f'Alternate Route {i+1}',
         'extra_time_min':random.randint(15,40),
         'risk_score':random.randint(35,60),
         'reason':'Avoid congested areas / road works'}
        for i in range(2)
    ]

    # ---------------- Apriori Integration ----------------
    rules = get_frequent_patterns()
    patterns_detected = []
    weather_input = f"Weather_{inputs.get('weather')}"
    time_input = f"Time_{inputs.get('time_of_day')}"
    road_input = f"Road_{inputs.get('road_type')}"
    for _, row in rules.iterrows():
        antecedents = set(row['antecedents'])
        if weather_input in antecedents or time_input in antecedents or road_input in antecedents:
            patterns_detected.append(', '.join(antecedents))

    timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = {
        'inputs':inputs,
        'overall_score':overall_score,
        'level':level,
        'breakdown':breakdown,
        'recommended_speed':recommended_speed,
        'weather_tips':weather_tips,
        'hazards':hazards,
        'vehicle_tips':vehicle_tips,
        'services':services,
        'alternates':alternates,
        'patterns_detected':patterns_detected,   # <-- Added Apriori results
        'timestamp':timestamp
    }

    return render_template('result.html', result=result)

# ---------------------------
# Quick-action endpoints
# ---------------------------
@app.route('/api/send_eta', methods=['POST'])
def api_send_eta():
    data = request.get_json() or {}
    print("ETA requested:", data)
    return jsonify({'status':'ok','detail':'ETA sent (mock)','payload':data})

@app.route('/api/sos', methods=['POST'])
def api_sos():
    data = request.get_json() or {}
    print("SOS triggered:", data)
    return jsonify({'status':'ok','detail':'SOS triggered (mock)','payload':data})

@app.route('/api/report_hazard', methods=['POST'])
def api_report_hazard():
    data = request.get_json() or {}
    return jsonify({'status':'ok','saved':True,'report':data})

@app.route('/debug/health')
def health(): return 'OK',200

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
