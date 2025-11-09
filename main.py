import subprocess
import sys
import os

# List of scripts to run in sequence (from skeleton directory)
scripts = [
    #'L04_digit_CNN.py',
    #'improved_digit_cnn.py',
    #'predict_my_digits.py',
    #'text_extraction.py',
    #'animal_classifier.py',
    #'object_detection.py'
]

# List of test modules to run
test_scripts = [
    #'tests.test_predict_my_digits',
    #'test_improved_digit_cnn',
    'test_animal_classifier',
    #'test_object_detection',
    #'test_text_extraction'
]

# Run each script from the skeleton directory
for script in scripts:
    print(f"Running {script}...")
    try:
        result = subprocess.run([sys.executable, script], cwd='skeleton')
        if result.returncode != 0:
            print(f"Errors in {script}: return code {result.returncode}")
        else:
            # After training improved_digit_cnn.py, copy model to tests directory for testing
            if script == 'improved_digit_cnn.py' and os.path.exists('skeleton/improved_digit_cnn.pth'):
                import shutil
                shutil.copy('skeleton/improved_digit_cnn.pth', 'tests/improved_digit_cnn.pth')
                print("Copied model to tests directory for testing.\n")

            # After training animal_classifier.py, copy model to tests directory for testing
            if script == 'animal_classifier.py' and os.path.exists('skeleton/animal_classifier.pth'):
                import shutil
                shutil.copy('skeleton/animal_classifier.pth', 'tests/animal_classifier.pth')
                print("Copied model to tests directory for testing.\n")
        print(f"Finished {script}\n")
    except Exception as e:
        print(f"Failed to run {script}: {e}\n")

# Run tests
print("Running tests...")
for test_script in test_scripts:
    print(f"Running {test_script}...")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = '../skeleton;..'
        result = subprocess.run([sys.executable, '-m', 'unittest', test_script], cwd='tests', env=env)
        if result.returncode != 0:
            print(f"Errors in {test_script}: return code {result.returncode}")
        print(f"Finished {test_script}\n")
    except Exception as e:
        print(f"Failed to run {test_script}: {e}\n")