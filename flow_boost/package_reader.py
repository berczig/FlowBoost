import os, re

project_path = "/Users/au596283/MLProjects/SpherePacking"

# Map to PyPI package names if import name differs
mapping = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "mpl_toolkits": "matplotlib",
    "yaml": "PyYAML",
}

imports = set()
pattern = re.compile(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)')

for root, dirs, files in os.walk(project_path):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pattern.match(line)
                    if m:
                        pkg = m.group(1).split(".")[0]
                        imports.add(mapping.get(pkg, pkg))

# Output to a clean requirements.txt
with open("minimal_requirements.txt", "w") as out:
    for pkg in sorted(imports):
        out.write(pkg + "\n")

print("Minimal requirements saved to minimal_requirements.txt")
