import sys
from importlib.metadata import version, PackageNotFoundError

def check_versions():
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
        
    mismatches = []
    for line in lines:
        line = line.split('#')[0].strip()
        if not line:
            continue
            
        if "==" not in line:
            continue
            
        pkg, expected_ver = line.split("==")
        
        # sb3-contrib is installed as sb3_contrib
        if pkg == "sb3_contrib":
            pkg = "sb3-contrib"
        
        try:
            installed_ver = version(pkg)
            if installed_ver != expected_ver:
                mismatches.append(f"{pkg}: expected {expected_ver}, found {installed_ver}")
        except PackageNotFoundError:
            mismatches.append(f"{pkg}: NOT INSTALLED")

    if mismatches:
        print("Version Validation Failed:")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)
    
    print("All required packages are installed with correct versions.")

if __name__ == "__main__":
    check_versions()
