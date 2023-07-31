import numpy as np

def menu():
    print("************WAV File Steganalysis**************")
    print()

    choice = input("""
                        1. List Available Files for Steganalysis
                        2. Analyze a File
                   
                        Please enter your choice: """)
    
    if choice == "1":
        print("List")
    elif choice == "2":
        print("Analyze")
    else:
        print("Please only select 1 or 2")
        print("Please try again")
        menu()

def main():
    menu()

main()