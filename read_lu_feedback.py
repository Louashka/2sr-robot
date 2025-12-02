import serial
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class BlockFeedback:
    """Data structure for feedback from one block"""
    motor_currents: Tuple[float, float]
    encoder_positions: Tuple[float, float]
    temperature: float
    block_label: str
    timestamp: float

class FeedbackParser:
    """
    Parses feedback from Arduino blocks
    Format: "F<current1>,<current2>,<pos1>,<pos2>,<temp_with_label>\n"
    Where temp_with_label is like "A56.7" or "B58.2"
    """
    
    def __init__(self, port='COM1', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(2)  # Wait for Arduino to initialize
        
        # Store latest feedback from each block
        self.block_data = {
            'A': None,
            'B': None
        }
        
        # Store raw temperature readings (before mapping)
        self.raw_temperatures = {
            'A': 25.0,
            'B': 25.0
        }
        
    def parse_feedback_line(self, line: str) -> Optional[BlockFeedback]:
        """
        Parse a single feedback line
        Returns BlockFeedback object or None if parsing fails
        """
        if not line.startswith('F'):
            return None
            
        try:
            # Remove the 'F' prefix
            data_str = line[1:]
            
            # Split by comma
            parts = data_str.split(',')
            
            if len(parts) != 5:
                print(f"Warning: Expected 5 parts, got {len(parts)}: {parts}")
                return None
            
            # Parse first 4 numeric values
            current1 = float(parts[0])
            current2 = float(parts[1])
            pos1 = float(parts[2])
            pos2 = float(parts[3])
            
            # Parse the temperature with block label
            # Format is like "A56.7" or "B58.2"
            temp_str = parts[4].strip()
            
            # Extract block label and temperature
            match = re.match(r'([AB])([-\d.]+)', temp_str)
            if match:
                block_label = match.group(1)
                temperature = float(match.group(2))
            else:
                print(f"Warning: Could not parse temperature string: {temp_str}")
                return None
            
            # Create feedback object
            feedback = BlockFeedback(
                motor_currents=(current1, current2),
                encoder_positions=(pos1, pos2),
                temperature=temperature,
                block_label=block_label,
                timestamp=time.time()
            )
            
            return feedback
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing feedback line: {line}")
            print(f"Error details: {e}")
            return None
    
    def read_feedback(self) -> Dict[str, BlockFeedback]:
        """
        Read all available feedback from serial port
        Returns dictionary with latest feedback for each block
        """
        try:
            line = self.ser.readline().decode().strip()
            
            if line.startswith('F'):
                # Parse feedback data
                feedback = self.parse_feedback_line(line)
                if feedback:
                    # Store in appropriate block slot
                    self.block_data[feedback.block_label] = feedback
                    
            elif line.startswith('A') or line.startswith('B'):
                # Legacy temperature-only format (from your original code)
                # Format: "A56.7" or "B58.2"
                match = re.match(r'([AB])([-\d.]+)', line)
                if match:
                    block = match.group(1)
                    temp = float(match.group(2))
                    self.raw_temperatures[block] = temp
                    
        except (UnicodeDecodeError, serial.SerialException) as e:
            print(f"Serial read error: {e}")
        
        return self.block_data
    
    def get_latest_feedback(self, block: str) -> Optional[BlockFeedback]:
        """Get the most recent feedback for a specific block"""
        return self.block_data.get(block)
    
    def get_combined_feedback(self) -> Dict:
        """
        Get combined feedback from both blocks
        """
        self.read_feedback()  # Update with latest data
        
        result = {
            'timestamp': time.time(),
            'blocks': {}
        }
        
        for block_label in ['A', 'B']:
            if self.block_data[block_label]:
                fb = self.block_data[block_label]
                result['blocks'][block_label] = {
                    'currents': fb.motor_currents,
                    'positions': fb.encoder_positions,
                    'temperature': fb.temperature,
                    'age': time.time() - fb.timestamp  # How old is this data
                }
        
        # Calculate combined metrics
        if all(self.block_data[b] is not None for b in ['A', 'B']):
            # Total current (useful for contact detection)
            result['total_current'] = sum(
                sum(self.block_data[b].motor_currents) for b in ['A', 'B']
            )
            
            # Average temperature
            result['avg_temperature'] = sum(
                self.block_data[b].temperature for b in ['A', 'B']
            ) / 2
            
        return result
    
    def wait_for_feedback(self, block: str, timeout: float = 1.0) -> Optional[BlockFeedback]:
        """
        Wait for fresh feedback from a specific block
        """
        start_time = time.time()
        initial_data = self.block_data.get(block)
        
        while time.time() - start_time < timeout:
            self.read_feedback()
            current_data = self.block_data.get(block)
            
            # Check if we got new data (timestamp changed)
            if current_data and (initial_data is None or 
                                 current_data.timestamp > initial_data.timestamp):
                return current_data
            
            time.sleep(0.01)
        
        return None
    
    def close(self):
        """Close serial connection"""
        if self.ser.is_open:
            self.ser.close()

# Integration with your robot control class
class RobotWithFeedback:
    """
    Example integration with your robot control system
    """
    
    def __init__(self, port='/dev/ttyUSB0'):
        self.feedback_parser = FeedbackParser(port)
        
        # Baseline currents for contact detection
        self.baseline_currents = {'A': (0, 0), 'B': (0, 0)}
        self.contact_threshold_multiplier = 1.3
        
    def calibrate_baseline_currents(self):
        """
        Calibrate baseline currents for contact detection
        """
        print("Calibrating baseline currents...")
        
        current_samples = {'A': [], 'B': []}
        
        # Collect samples over 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            feedback = self.feedback_parser.read_feedback()
            
            for block in ['A', 'B']:
                if feedback.get(block):
                    current_samples[block].append(feedback[block].motor_currents)
            
            time.sleep(0.05)
        
        # Calculate baselines
        for block in ['A', 'B']:
            if current_samples[block]:
                avg_current1 = sum(c[0] for c in current_samples[block]) / len(current_samples[block])
                avg_current2 = sum(c[1] for c in current_samples[block]) / len(current_samples[block])
                self.baseline_currents[block] = (avg_current1, avg_current2)
                print(f"Block {block} baseline: {avg_current1:.3f}, {avg_current2:.3f} A")
        
        return self.baseline_currents
    
    def check_contact(self, block: Optional[str] = None) -> bool:
        """
        Check if contact is detected
        If block is None, checks both blocks
        """
        feedback = self.feedback_parser.get_combined_feedback()
        
        if block:
            # Check specific block
            if block in feedback['blocks']:
                currents = feedback['blocks'][block]['currents']
                baseline = self.baseline_currents[block]
                
                # Check if either motor exceeds threshold
                for i in range(2):
                    if currents[i] > baseline[i] * self.contact_threshold_multiplier:
                        return True
        else:
            # Check both blocks
            for b in ['A', 'B']:
                if b in feedback['blocks']:
                    currents = feedback['blocks'][b]['currents']
                    baseline = self.baseline_currents[b]
                    
                    for i in range(2):
                        if currents[i] > baseline[i] * self.contact_threshold_multiplier:
                            return True
        
        return False
    
    def get_temperatures(self) -> Dict[str, float]:
        """Get current temperatures of both blocks"""
        feedback = self.feedback_parser.get_combined_feedback()
        temps = {}
        
        for block in ['A', 'B']:
            if block in feedback['blocks']:
                temps[block] = feedback['blocks'][block]['temperature']
        
        return temps
    
    def monitor_feedback(self, duration: float = 10.0):
        """
        Monitor and display feedback for testing
        """
        print("Monitoring feedback...")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                feedback = self.feedback_parser.get_combined_feedback()
                
                # Clear screen (Unix/Linux/Mac)
                print("\033[2J\033[H")
                
                print(f"Time: {time.time() - start_time:.1f}s\n")
                
                for block in ['A', 'B']:
                    if block in feedback['blocks']:
                        data = feedback['blocks'][block]
                        print(f"Block {block}:")
                        print(f"  Currents: {data['currents'][0]:.3f}, {data['currents'][1]:.3f} A")
                        print(f"  Positions: {data['positions'][0]:.3f}, {data['positions'][1]:.3f} rad")
                        print(f"  Temperature: {data['temperature']:.1f}°C")
                        print(f"  Data age: {data['age']:.3f}s")
                        
                        # Contact detection
                        if self.baseline_currents[block] != (0, 0):
                            contact = self.check_contact(block)
                            print(f"  Contact: {'YES' if contact else 'No'}")
                        print()
                
                if 'total_current' in feedback:
                    print(f"Total current: {feedback['total_current']:.3f} A")
                if 'avg_temperature' in feedback:
                    print(f"Average temperature: {feedback['avg_temperature']:.1f}°C")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

# Example usage and testing
def test_feedback_parser():
    """Test the feedback parser"""
    
    # Initialize parser
    robot = RobotWithFeedback(port='COM1')  
    
    print("=== Feedback Parser Test ===\n")
    
    # Test 1: Read raw feedback
    print("Test 1: Reading raw feedback for 3 seconds...")
    start = time.time()
    while True:
        feedback = robot.feedback_parser.read_feedback()
        for block, data in feedback.items():
            if data:
                print(f"Block {block}: currents={data.motor_currents}, "
                      f"temp={data.temperature:.1f}°C")
        time.sleep(0.5)
    
    # # Test 2: Calibrate baseline currents
    # print("\nTest 2: Calibrating baseline currents...")
    # robot.calibrate_baseline_currents()
    
    # # Test 3: Monitor with contact detection
    # print("\nTest 3: Monitoring with contact detection...")
    # robot.monitor_feedback(duration=10)
    
    # # Clean up
    # robot.feedback_parser.close()
    # print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_feedback_parser()