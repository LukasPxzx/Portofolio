import time
from smbus2 import SMBus, i2c_msg 

def gen_sht4x(DEV_ADDR_SHT4X=0x44, bus_number=1):
    with SMBus(bus_number) as bus:  
        #Read COMMAND
        bus.write_byte(DEV_ADDR_SHT4X, 0xFD)
        time.sleep(0.01) 

        #I2C for reading 6 bytes
        rx_bytes = i2c_msg.read(DEV_ADDR_SHT4X, 6)
        bus.i2c_rdwr(rx_bytes)  

        data = list(rx_bytes) 
        print("Raw SHT4x data:", data)  

        #Process temperature 
        t_ticks = int(data[0] * 256 + data[1])
        #Process humidity 
        rh_ticks = int(data[3] * 256 + data[4])

        #Convert ticks to Celsius
        temperature = -45 + 175 * t_ticks / 65535

        #Convert ticks to RH%
        rh = -6 + 125 * rh_ticks / 65535
        rh = max(0, min(rh, 100))  # Clamp to [0, 100]
        
        return [temperature, rh]

def gen_7700(DEV_ADDR_7700=0x10, bus_number=1):
    with SMBus(bus_number) as bus:  
        #Parameters
        ALS_GAIN = 0b01       #Gain x2
        ALS_IT = 0b1100       #Integration time 25 ms
        ALS_PERS = 0b11       #Persistence set to 8 readings
        ALS_INT_EN = 0b0      #Interrupt disabled
        ALS_SD = 0b0          #Power on

        #Create the configuration register value
        config_value = (ALS_GAIN << 11) | (ALS_IT << 6) | (ALS_PERS << 4) | (ALS_INT_EN << 1) | ALS_SD

        #Sensor COMMAND
        bus.write_word_data(DEV_ADDR_7700, 0x00, config_value)
        time.sleep(0.01)  

        #Now read the raw ALS data
        raw_data = bus.read_word_data(DEV_ADDR_7700, 0x04)  
        print("Raw 7700 data:", raw_data) 

        
        gain_factor = 2  #0b01 means gain x2

        #Calculate lux
        lux = raw_data / gain_factor * 0.01  #Adjust 0.01 for responsivity
        return [lux]

def gen_sfa30(DEV_ADDR_SFA30=0x5D, bus_number=1, retries=3):
    with SMBus(bus_number) as bus:
        for attempt in range(retries):
            try:
                #Start measurement
                bus.write_byte(DEV_ADDR_SFA30, 0x00) 
                time.sleep(0.1)  

                #I2C 6 bytes again
                rx_bytes = i2c_msg.read(DEV_ADDR_SFA30, 6)
                bus.i2c_rdwr(rx_bytes)

                
                data = list(rx_bytes)  #Convert rx_bytes to a list
                print("Raw SFA30 data:", data)

                
                hcho_ticks = int(data[0] * 256 + data[1])
                #We will not use humidity and temperature as before
                #humidity_ticks = int(data[2] * 256 + data[3])
                #temperature_ticks = int(data[4] * 256 + data[5])

                #Convert HCHO ticks to actual value
                hcho = hcho_ticks * 0.1  

                return [hcho]  #Return only HCHO value
            except IOError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)  
        print("Failed to read SFA30 sensor after multiple attempts.")
        return None