class AegisError(Exception):
    pass

class ModelLoadingError(AegisError):
    pass

class DataLoadingError(AegisError):
    pass

# Add more custom error classes as needed
