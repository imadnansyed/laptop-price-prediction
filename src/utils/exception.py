import sys

def error_details(err, err_details: sys):
    _, _, exc_tb = sys.exc_info()  # <-- fixed here
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error = str(err)
    error_message = (
        f"Error occurred in script: {file_name} at line number: {line_no} "
        f"with error message: {error}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, err_details: sys):
        super().__init__(error_message)
        # call the function, not the module
        self.error_message = error_details(err=error_message, err_details=err_details)

    def __str__(self):
        return self.error_message
