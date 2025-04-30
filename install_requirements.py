import subprocess
import sys

with open('requirements.txt', 'r') as file:
    for line in file:
        package = line.strip()
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
