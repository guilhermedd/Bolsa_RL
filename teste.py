import os

cwd = os.getcwd()
print("cwd = ", cwd)
# main.py
with open(f"{cwd}/output.txt", "w") as f:
    f.write("Hello, Docker!")
    print(f"Arquivo criado em {cwd}/output.txt")
