import os

# Create the entire project structure at once
def setup_complete_project():
    project_structure = [
        'urban-microclimate-business-intelligence-platform',
        'urban-microclimate-business-intelligence-platform/data',
        'urban-microclimate-business-intelligence-platform/data/raw',
        'urban-microclimate-business-intelligence-platform/data/processed',
        'urban-microclimate-business-intelligence-platform/src',
        'urban-microclimate-business-intelligence-platform/notebooks',
        'urban-microclimate-business-intelligence-platform/results',
        'urban-microclimate-business-intelligence-platform/tests',
        'urban-microclimate-business-intelligence-platform/config',
        'urban-microclimate-business-intelligence-platform/docs'
    ]
    
    for folder in project_structure:
        os.makedirs(folder, exist_ok=True)
    
    print("✅ Complete project structure created!")
    return project_structure

# Execute this first
setup_complete_project()
