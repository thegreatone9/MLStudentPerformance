import logging
import logger
import sys

def error_message_detail(error, error_detail: sys):
    _, _, exception_info = error_detail.exc_info()
    
    file_name = exception_info.tb_frame.f_code.co_filename
    
    error_message="Error occured in Python script name [{0}], line number [{1}], error message = [{2}]".format(
        file_name,
        exception_info.tb_lineno,
        str(error)
    )
    
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        custom_error_message = error_message_detail(error, error_detail)
        super().__init__(custom_error_message)
        self.error_message = custom_error_message

    def __str__(self):
        return self.error_message