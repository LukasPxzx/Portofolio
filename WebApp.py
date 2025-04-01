import logging
import time
import hashlib
from flask import Flask, request, redirect, url_for, flash, render_template, session, jsonify
from flask_socketio import SocketIO
from database import (
    create_tables,
    get_schema,
    get_property_types,
    get_locations,
    get_prediction_data,
    insert_prediction_data,
    insert_static_data,
    reset_database,
    connect_db,
    get_users,
    get_properties,
    classify_humidity,
    classify_temperature,
    classify_luminosity,
    get_db_connection
)
from Sensors import gen_7700, gen_sht4x, gen_sfa30
from threading import Thread
import sqlite3

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'keydrop'  # Needed for flashing messages

recording = False  # Declarar Importante

#Util Functions
def hash_password(password):
    """ Hash a password using SHA-256. """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def background_sensor_reading():
    bus_number = 1  
    index = 0  
    while True:
        try:
            #Read from SHT4x (Temperature and Humidity)
            temperature, rh = None, None
            for attempt in range(3):
                try:
                    temperature, rh = gen_sht4x(0x44, bus_number)
                    break
                except Exception as e:
                    logging.error(f"SHT4x Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)

            if temperature is None or rh is None:
                logging.error("Failed to read from SHT4x after 3 attempts.")
                continue
            
            #Read from 7700 (Luminosity)
            lux = None
            for attempt in range(3):
                try:
                    lux = gen_7700(0x10, bus_number)[0]
                    break
                except Exception as e:
                    logging.error(f"7700 Attempt {attempt + 1} failed: {e}")
                    time.sleep(1)

            if lux is None:
                logging.error("Failed to read from 7700 after 3 attempts.")
                continue

            #Read HCHO from SFA30
            hcho_data = gen_sfa30(0x5D, bus_number)  
            if hcho_data is None or len(hcho_data) < 1:  #1 value -> [hcho]
                logging.error("Failed to read from SFA30 or received unexpected data.")
                continue

            hcho = hcho_data[0]  #Extract HCHO
            
            #Insert prediction data into the database if recording is active
            if recording:
                insert_prediction_data(temperature=temperature, rh=rh, lux=lux, hcho=hcho)

            #Emit new data to clients with an index
            socketio.emit('new_data', {
                'prediction_data': [
                    [index, temperature, rh, lux, hcho]  #index must be included!!!
                ]
            })

            logging.info(f"Temperature: {temperature:.2f} °C, Humidity: {rh:.2f} %, Luminosity: {lux:.2f} lux, HCHO: {hcho:.2f} µg/m³")
            index += 1  #Go through the index in every step
        except Exception as e:
            logging.error(f"Error reading sensor data: {e}")

        time.sleep(1)  

# GET Routes
@app.route('/')
def index():
    return render_template("index_login.html")  

@app.route('/test')
def test():
    return render_template("test.html")

@app.route('/home', methods=['GET'])
def home():
    """ Redirect to login if not authenticated """
    if 'username' not in session:
        return redirect(url_for('index'))  # Redirect to login if != logged in
    return render_template("index_home.html")

@app.route('/data')
def data():
    """ Display data if user is authenticated """
    if 'username' not in session:
        return redirect(url_for('index'))
    
    users = get_users()
    locations = get_locations()
    property_types = get_property_types()
    prediction_data = get_prediction_data()
    
    #Get properties of database
    properties = get_properties()  #Call after updates

    logging.debug(f"Fetched properties: {properties}")  #Log properties
    return render_template('data.html', 
                           property_types=property_types, 
                           locations=locations, 
                           prediction_data=prediction_data, 
                           users=users, 
                           properties=properties)  #pass properties

@app.route('/pose')
def pose():
    return render_template("index_pose.html")

@app.route('/prediction')
def prediction():
    """ Display prediction page with current data """
    if 'username' not in session:
        return redirect(url_for('index'))
    prediction_data = get_prediction_data()  #Call the Data
    return render_template('prediction.html', recording=recording, prediction_data=prediction_data)

@app.route('/schema')
def schema():
    """ Display the database schema """
    schema = get_schema()
    return render_template('schema.html', schema=schema)

@app.route('/users')
def users():
    connection = sqlite3.connect('your_database.db')
    cursor = connection.cursor()
    cursor.execute("SELECT id, username, password FROM users")
    users = cursor.fetchall()
    logging.debug(f"Retrieved users: {users}")  #Log Retrieevd users
    connection.close()
    return render_template('users.html', users=users)

@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    """Fetch the latest sensor data including temperature, humidity, luminosity, and HCHO."""
    logging.info("Received request for /get_sensor_data")
    conn = connect_db()  #Connect to database
    cursor = conn.cursor()

    cursor.execute('''
        SELECT temperature, humidity, luminosity, hcho 
        FROM Prediction_Data 
        ORDER BY id DESC 
        LIMIT 1
    ''')
    row = cursor.fetchone()
    conn.close()

    if row:
        temperature, humidity, luminosity, hcho = row
        logging.info(f"Retrieved from DB - Temperature: {temperature}, Humidity: {humidity}, Luminosity: {luminosity}, HCHO: {hcho}")
        return jsonify({"Ts": temperature, "RH2M": humidity, "luminosity": luminosity, "hcho": hcho}), 200
    else:
        logging.warning("No sensor data available.")
    return jsonify({"error": "No sensor data available."}), 404

@app.route('/get_last_prediction/<int:property_id>', methods=['GET'])
def get_last_prediction(property_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM RecordedPredictionData WHERE property_id = ? ORDER BY timestamp DESC LIMIT 1', (property_id,))
    last_prediction = cursor.fetchone()

    conn.close()
    
    if last_prediction:
        return jsonify(dict(last_prediction))  #Return the last prediction as JSON
    return jsonify(None)
@app.route('/get_properties', methods=['GET'])
def get_properties_route():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, property_type_id, price, location_id, mold_growth, 
                   luminosity, humidity, temperature, luminosity_class, 
                   temperature_class, humidity_class, hcho, image 
            FROM Properties
        ''')
        properties = cursor.fetchall()

        #Convert the obtained rows into a list of dictionaries
        properties_list = []
        for row in properties:
            properties_list.append({
                "id": row[0],
                "property_type_id": row[1],
                "price": row[2],
                "location_id": row[3],
                "mold_growth": row[4],
                "luminosity": row[5],
                "humidity": row[6],
                "temperature": row[7],
                "luminosity_class": row[8],
                "temperature_class": row[9],
                "humidity_class": row[10],
                "hcho": row[11],  
                "image": row[12]
            })

        conn.close()
        return jsonify(properties_list)
    except Exception as e:
        logging.error(f"Error fetching properties: {e}")
        return jsonify({"error": str(e)}), 500  #Response 500 status
@app.route('/get_property_types', methods=['GET'])
def get_property_types():
    """Retrieve all property types from the database."""
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PropertyTypes')
        property_types = cursor.fetchall()
        
        #Create another lsit of dictionaries for the rows
        property_types_list = [
            {"id": row[0], "type_name": row[1]} for row in property_types
        ]
        
        return jsonify(property_types_list)
    
@app.route('/get_users', methods=['GET'])
def get_users_route():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT id, username FROM Users')
        users = cursor.fetchall()

        users_list = []
        for row in users:
            users_list.append({
                "id": row[0],
                "username": row[1]
            })

        conn.close()
        return jsonify(users_list)
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_locations', methods=['GET'])
def get_locations_route():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT id, location_name FROM Locations')
        locations = cursor.fetchall()

        locations_list = []
        for row in locations:
            locations_list.append({
                "id": row[0],
                "location_name": row[1]
            })

        conn.close()
        return jsonify(locations_list)
    except Exception as e:
        logging.error(f"Error fetching locations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_prediction_data', methods=['GET'])
def get_prediction_data_route():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT id, temperature, humidity, luminosity, hcho FROM Prediction_Data')
        prediction_data = cursor.fetchall()

        prediction_data_list = []
        for row in prediction_data:
            prediction_data_list.append({
                "id": row[0],
                "temperature": row[1],
                "humidity": row[2],
                "luminosity": row[3],  
                "hcho": row[4]
            })

        return jsonify(prediction_data_list)
    except Exception as e:
        logging.error(f"Error fetching prediction data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_last_luminosity_hcho/<int:id>', methods=['GET'])
def get_last_luminosity_hcho(id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT luminosity, hcho FROM Prediction_Data WHERE id = ? ORDER BY id DESC LIMIT 1', (id,))
        row = cursor.fetchone()

        if row:
            return jsonify({
                "luminosity": row[0], 
                "hcho": row[1]         
            })
        else:
            return jsonify({"error": "No data found for the given ID."}), 404

    except Exception as e:
        logging.error(f"Error fetching luminosity and HCHO data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

#POST Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """ This route handles the login logic """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        #Connect to the database
        conn = connect_db()
        cursor = conn.cursor()
        try:
            #Obtain the hashed password for the given username
            cursor.execute('SELECT password FROM Users WHERE username = ?', (username,))
            user = cursor.fetchone()
        finally:
            conn.close()

        if user:
            #Compare the provided password with the stored hashed password
            if hash_password(password) == user[0]:  #Chekc password matches
                logging.info(f"User '{username}' logged in successfully.")
                session['username'] = username  #Store user name in Session
                return jsonify({"username": username}), 200  #Return JSON response, all good :)
            else:
                logging.warning(f"Failed login attempt for user '{username}': Incorrect password.")
                return jsonify({"error": "Invalid username or password"}), 401
        else:
            logging.warning(f"Failed login attempt: Username '{username}' not found.")
            return jsonify({"error": "Invalid username or password"}), 401 #Not Found , not good T_T

    return render_template("index_login.html")

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    """ Toggle recording state and reset database if starting recording """
    global recording
    if not recording:  #Only reset if we are starting to record
        reset_database()  #Call your reset function to clear the database
    recording = not recording  #Toggle the recording state
    logging.info(f"Recording state changed to: {recording}")  
    return redirect(url_for('prediction'))  #Redirect back to the prediction page

@app.route('/add_property', methods=['POST'])
def add_property():
    property_type_id = request.form['property_type_id']
    price = request.form['price']
    location_id = request.form['location_id']
    mold_growth = request.form['mold_growth']
    
    try:
        luminosity = float(request.form['luminosity'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
    except ValueError:
        logging.error("Invalid input for luminosity, humidity, or temperature.")
        return redirect(url_for('data'))

    image = request.form['image']

    #Classify the values
    luminosity_class = classify_luminosity(luminosity)
    temperature_class = classify_temperature(temperature)
    humidity_class = classify_humidity(humidity)

    #Insert into the database
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Properties 
            (property_type_id, price, location_id, mold_growth, luminosity, humidity, temperature, luminosity_class, temperature_class, humidity_class, image) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (property_type_id, price, location_id, mold_growth, luminosity, humidity, temperature, luminosity_class, temperature_class, humidity_class, image))

        conn.commit()
        logging.info("Property added successfully with classes.")
    except Exception as e:
        logging.error(f"Error adding property: {e}")
    finally:
        conn.close()

    return redirect(url_for('data'))

@app.route('/store_prediction_data', methods=['POST'])
def store_prediction_data():
    try:
        data = request.get_json()  #Get the JSON data sent from the frontend
        
        #Extract necessary fields from the incoming data
        date = data.get('date')
        temperature = data.get('TS')
        humidity = data.get('RH2M')
        mold_growth = data.get('mold_growth')  #This can be None if not sent
        luminosity = data.get('luminosity')  
        hcho = data.get('hcho')             

        logging.info("Received data:", {
            "date": date,
            "temperature": temperature,
            "humidity": humidity,
            "mold_growth": mold_growth,
            "luminosity": luminosity,
            "hcho": hcho
        })


        return jsonify({"status": "success", "message": "Data stored successfully."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/update_property', methods=['POST'])
def update_property():
    try:
        data = request.json

        #Debug
        logging.info(f"Received JSON data: {data}")

        #Double Check
        property_id = data['id']
        mold_growth = data['mold_growth']
        humidity = data['humidity']
        temperature = data['temperature']
        luminosity = data['luminosity']
        hcho = data['hcho']

        #Validate
        if any(v is None for v in [mold_growth, humidity, temperature, luminosity, hcho]):
            logging.error("One or more fields are missing.")
            return jsonify({"error": "One or more fields are missing."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        #Update the property in the Properties table
        cursor.execute('''
            UPDATE Properties
            SET mold_growth = ?, humidity = ?, temperature = ?, luminosity = ?, hcho = ?
            WHERE id = ?
        ''', (mold_growth, humidity, temperature, luminosity, hcho, property_id))

        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error updating property: {str(e)}")
        logging.error(f"Failed data: {data}")  # Log the data that caused the error
        return jsonify({"error": str(e)}), 500
    
@app.route('/search_properties', methods=['GET'])
def search_properties():
    #Extract search params from request.args
    property_type_id = request.args.get('property_type_id')
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    location_id = request.args.get('location_id')
    
    #Logic to filter properties based on search parameters
    with connect_db() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM Properties WHERE 1=1"
        params = []
        
        if property_type_id:
            query += " AND property_type_id = ?"
            params.append(property_type_id)
        if min_price:
            query += " AND price >= ?"
            params.append(min_price)
        if max_price:
            query += " AND price <= ?"
            params.append(max_price)
        if location_id:
            query += " AND location_id = ?"
            params.append(location_id)
        
        cursor.execute(query, params)
        properties = cursor.fetchall()

    return render_template('search_results.html', properties=properties)

@app.route('/logout')
def logout():
    """ Logout the user and redirect to login page """
    session.pop('username', None)  #Remove the username from the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))  #Redirect to login page

@app.route('/reset')
def reset():
    """ Reset the database and redirect to the prediction page """
    reset_database()  
    insert_static_data()  #Make sure that static data is inserted again after the reset
    return redirect(url_for('prediction'))

    
if __name__ == '__main__':
    create_tables()
    insert_static_data()

    #Start the background thread for sensor reading
    Thread(target=background_sensor_reading, daemon=True).start()
    
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)