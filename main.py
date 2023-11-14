from MLLauncher import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No arguments")

    json_name = sys.argv[1]

    if not os.path.exists(json_name):
        print(f"{json_name} file doesn't exist")

    ml_launcher = MLLauncher(json_name)
    ml_launcher.run()
    
    print('Done')
