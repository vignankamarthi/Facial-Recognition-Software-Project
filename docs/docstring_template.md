# Docstring and Method Signature Standards

## NumPy-Style Docstring Format

All functions and methods in this project use the NumPy docstring format. This format provides a clear structure for documenting parameters, return values, exceptions, examples, and more.

### Basic Function Template

```python
def function_name(param1, param2, optional_param=None):
    """
    Brief one-line description of the function.
    
    More detailed description that explains what the function does, its purpose,
    and any important information that users should know.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    optional_param : type, optional
        Description of optional_param (default: None).
    
    Returns
    -------
    type
        Description of return value.
    
    Raises
    ------
    ExceptionType
        Description of when this exception is raised.
    
    Examples
    --------
    >>> # Example showing how to use the function
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    
    Notes
    -----
    Any additional notes or caveats about using the function.
    
    See Also
    --------
    related_function : Brief description of related function.
    another_function : Brief description of another related function.
    """
    # Function implementation...
```

### Class Template

```python
class ClassName:
    """
    Brief one-line description of the class.
    
    More detailed description of the class and its behavior.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    
    Attributes
    ----------
    attr1 : type
        Description of attr1.
    attr2 : type
        Description of attr2.
    
    Examples
    --------
    >>> # Example showing how to use the class
    >>> obj = ClassName(1, 2)
    >>> obj.some_method()
    
    Notes
    -----
    Any additional notes about using the class.
    """
    
    def __init__(self, param1, param2):
        """Initialize the class."""
        self.attr1 = param1
        self.attr2 = param2
    
    def method_name(self, param1, param2):
        """
        Brief one-line description of the method.
        
        Parameters
        ----------
        param1 : type
            Description of param1.
        param2 : type
            Description of param2.
        
        Returns
        -------
        type
            Description of return value.
        """
        # Method implementation...
```

## Method Signature Standards

### Parameter Naming Conventions

1. Use descriptive parameter names that clearly indicate the purpose of the parameter.
2. For paths, use `file_path` for files and `directory_path` for directories.
3. Use consistent pluralization for parameters:
   - Use singular names for single items: `image`, `face_location`, `file_path`
   - Use plural names for collections: `images`, `face_locations`, `file_paths`
4. Use consistent casing:
   - Use snake_case for all parameter names
   - Avoid camelCase or PascalCase in parameter names

### Parameter Ordering

1. Required parameters should come first.
2. Optional parameters (with defaults) should come after required parameters.
3. `*args` and `**kwargs` should come last (if used).
4. Boolean flags should typically come after other parameters.

### Return Values

1. For image processing methods, return tuples in the form: `(result_image, metadata_dict)`
   - `result_image` can be None on failure
   - `metadata_dict` should contain information about the processing
2. For error cases, return: `(None, error_dict)`
   - `error_dict` should contain at minimum:
     - `'error'`: Error message
     - `'type'`: Error type (e.g., 'OpenCV', 'Camera', etc.)
3. For collection methods, always return empty collections (not None) when no results are found:
   - Return `[]` for empty lists
   - Return `{}` for empty dictionaries
4. For boolean operations (success/failure), return:
   - `(True, result_dict)` for success
   - `(False, error_dict)` for failure

### Method Naming Conventions

1. Use verb-noun pattern for actions: `detect_faces()`, `process_image()`, `save_result()`
2. Use descriptive method names that clearly indicate the purpose of the method
3. Use `get_` prefix for retrieval operations: `get_image_files()`, `get_known_faces_dir()`
4. Use `is_` or `has_` prefix for boolean queries: `is_image_file()`, `has_faces()`
5. Use the following prefixes consistently:
   - `create_`: Methods that create new objects
   - `load_`: Methods that load data from files
   - `save_`: Methods that save data to files
   - `process_`: Methods that transform data
   - `analyze_`: Methods that examine data without modifying it

## Examples

### Before Standardization

```python
def processImg(img, detectFaces=False, matchFaces=False, save=False, savePath=""):
    """
    Process an image with selected operations
    
    Args:
        img: Input image
        detectFaces: Whether to detect faces
        matchFaces: Whether to match faces against known faces
        save: Whether to save the result
        savePath: Where to save the result
        
    Returns:
        The processed image
    """
    # ...
```

### After Standardization

```python
def process_image(image, detect_faces=False, match_faces=False, save_result=False, output_directory=None):
    """
    Process an image with selected operations.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image to process
    detect_faces : bool, optional
        Whether to detect faces in the image (default: False)
    match_faces : bool, optional
        Whether to match faces against known faces (default: False)
    save_result : bool, optional
        Whether to save the processed image (default: False)
    output_directory : str, optional
        Directory to save the result if save_result is True
        
    Returns
    -------
    tuple
        (processed_image, metadata_dict) where:
        - processed_image : numpy.ndarray or None
          The processed image, or None if processing failed
        - metadata_dict : dict
          Dictionary containing metadata about the processing:
          - "face_count" : int
            Number of faces detected
          - "face_locations" : list
            List of face location tuples
          - "identified_faces" : list
            List of identified face names
          - "output_path" : str
            Path where the result was saved (if save_result=True)
          - "error" : str
            Error message (if processing failed)
    
    Raises
    ------
    ValueError
        If the input image is None
        
    Examples
    --------
    >>> # Process an image with face detection
    >>> image = cv2.imread("test.jpg")
    >>> processed_image, metadata = process_image(image, detect_faces=True)
    >>> print(f"Detected {metadata['face_count']} faces")
    """
    # ...
```
