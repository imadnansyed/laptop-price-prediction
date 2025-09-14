import sys

def error_details(err, err_details:sys):
    _,_, exc_tb=error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error = str(err)
    
    error_message = f"Error occurred in script: {file_name} at line number: {line_no} with error message: {error}"
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_details(err=error_message, err_details=error_details)
        
    def __str__(self):
        return self.error_message