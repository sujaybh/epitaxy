import pyvisa
import time
import csv

# Configuration
GPIB_ADDRESS = 'GPIB0::17::INSTR' # Factory default is 17 [cite: 3]
LOG_FILENAME = 'measurement_log.csv'

def initialize_instrument():
    rm = pyvisa.ResourceManager()
    try:
        # Open connection to the HP 4280A
        instr = rm.open_resource(GPIB_ADDRESS)
        instr.timeout = 10000 # 10 seconds
        
        # CLEAR Hp4280a (Line 70)
        instr.clear()
        
        # TR3: Hold Mode, BC: Buffer Clear, IB2: Internal Bias [cite: 33, 38, 27]
        instr.write("TR3, BC, IB2")
        
        # BL1: Block Data Transfer Mode (Line 100) 
        instr.write("BL1")
        
        # Set Sweep Parameters (Line 110) [cite: 59, 73, 84, 88]
        # PS: Start -5V, PP: Stop 5V, PE: Step 0.5V, PL/PD: 3ms times
        instr.write("PS-5, PP5, PE.5, PL3E-3, PD3E-3")
        
        # MD1: Data Ready SRQ Mask Off (Enable Interrupt) 
        instr.write("MD1")
        
        return instr
    except Exception as e:
        print(f"Error connecting to instrument: {e}")
        return None

def run_sweep(instr):
    print("Starting Sweep...")
    # SW1: Start Sweep (Line 130) [cite: 33]
    instr.write("SW1")
    
    # Poll Status Byte (Serial Poll) until Data Ready (Bit 0) is set 
    # In the BASIC code, this was handled by an interrupt (Line 60/180)
    data_ready = False
    while not data_ready:
        status_byte = instr.stb # Read Status Byte (SPOLL)
        if status_byte & 1: # Checking Bit 0 (Data Ready)
            data_ready = True
        else:
            time.sleep(0.1)
    
    print("Data Ready. Retrieving results...")
    
    # BS?: Query Block Size/Status (Line 210) 
    # This usually returns metadata about the block transfer
    meta_data = instr.query("BS?")
    
    # In Block Mode (BL1), read the raw data
    # Note: Depending on your specific VISA setup, you may use read_raw() or read()
    raw_data = instr.read()
    return meta_data, raw_data

def log_to_csv(meta, data):
    with open(LOG_FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.ctime(), "Metadata:", meta])
        writer.writerow(["Raw Data:", data])
    print(f"Data logged to {LOG_FILENAME}")

# Main Execution
if __name__ == "__main__":
    device = initialize_instrument()
    if device:
        metadata, results = run_sweep(device)
        log_to_csv(metadata, results)
        device.close()
