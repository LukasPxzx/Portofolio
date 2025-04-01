import sqlite3
import hashlib
import logging

#Set up logging
logging.basicConfig(level=logging.DEBUG)

def connect_db():
    """Connect to the SQLite database."""
    return sqlite3.connect('real_estate.db')

def create_tables():
    """Create the necessary tables in the database."""
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PropertyTypes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_name TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_type_id INTEGER,
            price REAL NOT NULL,
            location_id INTEGER,
            mold_growth REAL,
            luminosity REAL,
            humidity REAL,
            temperature REAL,
            luminosity_class TEXT,
            temperature_class TEXT,
            humidity_class TEXT,
            hcho REAL,
            image TEXT NOT NULL,
            FOREIGN KEY (property_type_id) REFERENCES PropertyTypes(id),
            FOREIGN KEY (location_id) REFERENCES Locations(id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Prediction_Data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            luminosity REAL,
            humidity REAL,
            temperature REAL,
            hcho REAL  
        )
        ''')

        conn.commit()

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def insert_prediction_data(temperature, rh, lux, hcho):
    with connect_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO Prediction_Data (temperature, humidity, luminosity, hcho)
                VALUES (?, ?, ?, ?)
            ''', (temperature, rh, lux, hcho))
            conn.commit()
        except sqlite3.Error as e:
            logging.error("An error occurred during prediction data insertion: %s", e)

def insert_static_data():
    """Insert static data into the database, including properties but excluding users."""
    with connect_db() as conn:
        cursor = conn.cursor()
        try:
            #Sample property types
            property_types = [
                ('Penthouse',),
                ('House / Chalet',),
                ('Duplex',),
                ('Studio',),
                ('Commercial Space',),
                ('Office',),
                ('Flat',),
                ('Parking Spot',),
                ('Land',)
            ]
            cursor.executemany('INSERT OR IGNORE INTO PropertyTypes (type_name) VALUES (?)', property_types)

            #Sample locations
            locations = [
                ('Central and Western',),
                ('Eastern',),
                ('Islands',),
                ('Kowloon City',),
                ('Kwai Tsing',),
                ('North',),
                ('Sai Kung',),
                ('Sha Tin',),
                ('Sham Shui Po',),
                ('Southern',),
                ('Tai Po',),
                ('Tsuen Wan',),
                ('Tuen Mun',),
                ('Wanchai',),
                ('Yau Tsim Mong',),
                ('Yuen Long',)
            ]
            cursor.executemany('INSERT OR IGNORE INTO Locations (location_name) VALUES (?)', locations)

            # Sample users (hashed passwords)
            userlist = [
                ('admin', 'admin'), 
                ('Lucas', 'Lucas'), 
                ('Dickson', 'Dick'),
                ('Tony', 'Tony')
            ]
            userlist_hashed = [(username, hash_password(password)) for username, password in userlist]
            cursor.executemany('INSERT OR IGNORE INTO Users (username, password) VALUES (?, ?)', userlist_hashed)

            #Sample properties
            properties = [
                (1, 25000000, 1, 80.0, 65.0, 22.0, 'https://assets.savills.com/properties/HK300320236636S/20240829_081653620_iOS_m_lis.jpg'),  # Penthouse in Central
                (2, 18000000, 2, 70.0, 70.0, 21.0, 'https://assets.savills.com/properties/HK300320236636S/20240829_082207489_iOS_m_lis.jpg'),  # House in Kowloon City
                (3, 22000000, 3, 85.0, 60.0, 23.5, 'https://assets.savills.com/properties/HK300320236636S/20240829_080813210_iOS_m_lis.jpg'),  # Duplex in Sai Kung
                (4, 9000000, 4, 40.0, 80.0, 19.0, 'https://assets.savills.com/properties/HK300320236636S/20240829_080536777_iOS_m_lis.jpg'),    # Studio in Sham Shui Po
                (5, 15000000, 5, 75.0, 75.0, 20.0, 'https://assets.savills.com/properties/HK300320236636S/20240829_081453273_iOS_m_lis.jpg'),  # Commercial Space in Wanchai
                (6, 12000000, 6, 65.0, 68.0, 22.5, 'https://assets.savills.com/properties/HK300320236636S/20240829_080858724_iOS_m_lis.jpg'),  # Office in Tsim Sha Tsui
                (7, 9500000, 7, 50.0, 72.0, 20.5, 'https://assets.savills.com/properties/HK300320236636S/20240829_081347238_iOS_m_lis.jpg'),   # Flat in Yau Tsim Mong
                (8, 3000000, 8, 55.0, 64.0, 23.0, 'https://assets.savills.com/properties/HK300320236636S/20240829_080512856_iOS_m_lis.jpg'),   # Parking Spot in Tsuen Wan
                (9, 20000000, 9, 60.0, 66.0, 21.5, 'https://assets.savills.com/properties/HK300320236636S/7d8ae635-357c-41f9-a1fa-25bdd92a2b731_l_gal.jpg'),  # Land in Tuen Mun
                (10, 35000000, 10, 90.0, 62.0, 24.0, 'https://assets.savills.com/properties/HK300320236636S/WhatsApp%20Image%202024-07-12%20at%2014.25.01_dbfa77ea_l_gal.jpg') # Luxury Flat in Central
            ]
            cursor.executemany('INSERT INTO Properties (property_type_id, price, location_id, luminosity, humidity, temperature, image) VALUES (?, ?, ?, ?, ?, ?, ?)', properties)

            conn.commit()
            logging.info("Static data inserted successfully.")
        except sqlite3.Error as e:
            logging.error("An error occurred during static data insertion: %s", e)

def classify_luminosity(luminosity):
    logging.debug(f"Classifying luminosity: {luminosity}")
    if luminosity is None:
        return 'N/A'
    if luminosity <= 40:
        return 'Low'
    elif 41 <= luminosity <= 75:
        return 'Average'
    else:
        return 'High'

def classify_temperature(temperature):
    if temperature is None:
        return 'N/A'
    if temperature < 15:
        return 'Low'
    elif 15 <= temperature <= 25:
        return 'Average'
    else:
        return 'High'

def classify_humidity(humidity):
    
    if humidity is None:
        return 'N/A'
    if humidity <= 30:
        return 'Low'
    elif 31 <= humidity <= 60:
        return 'Average'
    else:
        return 'High'
    
    
def get_db_connection():
    """Get a database connection with row factory for named access."""
    conn = sqlite3.connect('real_estate.db')
    conn.row_factory = sqlite3.Row  #Access by column name
    return conn

def get_schema():
    """Retrieve the database schema."""
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logging.debug("Tables found in the database: %s", tables)

        schema = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema[table_name] = cursor.fetchall()

        return schema

def get_property_types():
    """Retrieve all property types from the database."""
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PropertyTypes')
        property_types = cursor.fetchall()
        return [{"id": row[0], "type_name": row[1]} for row in property_types] 
    
def get_properties():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Properties')
        return [{"id": row[0], "price": row[1]} for row in cursor.fetchall()]  

def get_locations():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, location_name FROM Locations')
        return [{"id": row[0], "location_name": row[1]} for row in cursor.fetchall()]

def get_users():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, username FROM Users') 
        return [{"id": row[0], "username": row[1]} for row in cursor.fetchall()]
    
def get_prediction_data():
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Prediction_Data')
        return [{"id": row[0], "temperature": row[1], "humidity": row[2]} for row in cursor.fetchall()]

def reset_database():
    """Reset the database to its initial state."""
    with connect_db() as conn:
        cursor = conn.cursor()
        try:
            #Drop all tables in order
            cursor.execute('DROP TABLE IF EXISTS Prediction_Data')
            cursor.execute('DROP TABLE IF EXISTS Properties')
            cursor.execute('DROP TABLE IF EXISTS Locations')
            cursor.execute('DROP TABLE IF EXISTS PropertyTypes')
            cursor.execute('DROP TABLE IF EXISTS Users')
            logging.info("Database reset: All tables dropped.")

            #Recreate tables
            create_tables()
            logging.info("Database reset: All tables recreated.")

            #Insert static data after recreating tables
            insert_static_data()
            logging.info("Static data inserted after database reset.")

        except sqlite3.Error as e:
            logging.error("An error occurred while resetting the database: %s", e)
